"""
omegaflux.evaluate_convergence - the trained surrogate against original and converged labels (WO-B, B7).

For the 150 convergence-check rooms (all in the held-out E1 partition), compares the surrogate's
predictions with
  ab5   the original training labels (-ab 5 -ad 5000)
  ab8   the converged reference      (-ab 8 -ad 20000)
  ab12  the ladder check             (-ab 12, ten L-shapes)
and puts the label shift next to the surrogate error, per plan class.

Inputs : the model folder (features.json + omegaflux_xgb_{low,up}.json),
         data/processed/omegaflux_sensors_pooled.csv (descriptors),
         data/convergence_check/convergence_sensors.csv.gz (labels per setting).
Sensors are aligned by order within each room; the script checks that the pooled table's targets equal
the ab5 labels at every sensor before using a room.

Usage:  python -m omegaflux.evaluate_convergence --model OmegaFlux_Models/core
Output: <model>/convergence_eval/{surrogate_vs_labels.csv, room_level.csv, summary.json, summary.md}
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit

from omegaflux import train as T

POOLED_CSV = os.path.join("data", "processed", "omegaflux_sensors_pooled.csv")
CONV_SENSORS = os.path.join("data", "convergence_check", "convergence_sensors.csv.gz")
CLASSES = {"rect": "Rectangular", "rect_rotated": "Rotated", "randomized": "L-shaped"}
ALIGN_TOL = 0.02   # pp; pooled targets vs ab5 labels (labels are stored to 2 decimals)


def load_model(folder):
    with open(os.path.join(folder, "features.json")) as fh:
        features = json.load(fh)["features"]
    models = []
    for t in ("low", "up"):
        m = xgb.XGBRegressor()
        m.load_model(os.path.join(folder, f"omegaflux_xgb_{t}.json"))
        models.append(m)
    return features, models


def held_out_groups(path):
    g = pd.read_csv(path, usecols=["Group_ID"], dtype={"Group_ID": str})["Group_ID"]
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=T.TEST_FRACTION,
                                   random_state=T.SEED).split(g, groups=g))
    return set(g.iloc[te])


def read_pooled_rooms(path, groups, features):
    cols = ["Group_ID", "Source_Tag", "Target_UDI_low", "Target_UDI_up"] + features
    parts = []
    for chunk in pd.read_csv(path, usecols=cols, dtype={"Group_ID": str}, chunksize=500_000):
        parts.append(chunk[chunk["Group_ID"].isin(groups)])
    return pd.concat(parts, ignore_index=True)


def metrics(a, p):
    d = p - a
    ss = ((a - a.mean()) ** 2).sum()
    return {"RMSE": float(np.sqrt((d ** 2).mean())), "MAE": float(np.abs(d).mean()),
            "bias": float(d.mean()), "R2": float(1 - (d ** 2).sum() / ss) if ss > 0 else float("nan"),
            "N": int(len(a))}


def main():
    ap = argparse.ArgumentParser(description="Surrogate against original and converged Radiance labels")
    ap.add_argument("--model", default=os.path.join("OmegaFlux_Models", "core"))
    ap.add_argument("--data", default=POOLED_CSV)
    ap.add_argument("--conv", default=CONV_SENSORS)
    args = ap.parse_args()
    out = os.path.join(args.model, "convergence_eval")
    os.makedirs(out, exist_ok=True)

    features, (m_low, m_up) = load_model(args.model)
    conv = pd.read_csv(args.conv)
    conv["Group_ID"] = conv["class"] + ":" + conv["room_id"]
    rooms = list(dict.fromkeys(conv["Group_ID"]))
    T.log(f"{len(rooms)} convergence rooms, model {args.model} ({len(features)} descriptors)")

    test = held_out_groups(args.data)
    not_held = [g for g in rooms if g not in test]
    T.log(f"Held-out check: {len(rooms) - len(not_held)}/{len(rooms)} rooms in the E1 test partition"
          + (f"; NOT held out: {not_held[:5]}" if not_held else ""))

    pooled = read_pooled_rooms(args.data, set(rooms), features)
    rows, skipped = [], []
    for gid in rooms:
        c = conv[conv["Group_ID"] == gid].sort_values("sensor_idx")
        p = pooled[pooled["Group_ID"] == gid]
        if len(p) != len(c):
            skipped.append((gid, f"{len(p)} pooled vs {len(c)} convergence sensors"))
            continue
        dev = max(np.abs(p["Target_UDI_low"].to_numpy() - c["udi_low_ab5"].to_numpy()).max(),
                  np.abs(p["Target_UDI_up"].to_numpy() - c["udi_up_ab5"].to_numpy()).max())
        if dev > ALIGN_TOL:
            skipped.append((gid, f"pooled targets differ from ab5 labels by up to {dev:.2f} pp"))
            continue
        X = p[features].astype("float32").to_numpy()
        lo, up = np.clip(m_low.predict(X), 0, 100), np.clip(m_up.predict(X), 0, 100)
        frame = c[["Group_ID", "room_id", "class", "sensor_idx"]].copy()
        frame["pred_useful"] = np.clip(100 - lo - up, 0, 100)
        for s in ("ab5", "ab8", "ab12"):
            frame[f"useful_{s}"] = c[f"udi_useful_{s}"].to_numpy()
        rows.append(frame)
    for gid, why in skipped:
        T.log(f"[SKIP] {gid}: {why}")
    if not rows:
        raise SystemExit("no rooms could be aligned")
    s = pd.concat(rows, ignore_index=True)
    s.to_csv(os.path.join(out, "surrogate_vs_labels.csv.gz"), index=False, compression="gzip")

    # ---- per class: surrogate error against each label set, and the label shift itself
    table = []
    for klass, g in list(s.groupby("class")) + [("all", s)]:
        row = {"class": CLASSES.get(klass, klass), "rooms": int(g["room_id"].nunique())}
        for ref in ("ab5", "ab8"):
            for k, v in metrics(g[f"useful_{ref}"].to_numpy(), g["pred_useful"].to_numpy()).items():
                if k != "N":
                    row[f"surrogate_vs_{ref}_{k}"] = v
        shift = metrics(g["useful_ab5"].to_numpy(), g["useful_ab8"].to_numpy())
        row.update({"label_shift_MAE": shift["MAE"], "label_shift_bias": shift["bias"]})
        rm = g.groupby("room_id")[["pred_useful", "useful_ab5", "useful_ab8"]].mean()
        if len(rm) > 2:
            row["rho_rooms_vs_ab5"] = float(spearmanr(rm["pred_useful"], rm["useful_ab5"]).statistic)
            row["rho_rooms_vs_ab8"] = float(spearmanr(rm["pred_useful"], rm["useful_ab8"]).statistic)
        table.append(row)
    tab = pd.DataFrame(table)
    tab.to_csv(os.path.join(out, "summary_by_class.csv"), index=False)

    # ---- ladder rooms
    lad = s.dropna(subset=["useful_ab12"])
    ladder = None
    if len(lad):
        ladder = {ref: metrics(lad[f"useful_{ref}"].to_numpy(), lad["pred_useful"].to_numpy())
                  for ref in ("ab5", "ab8", "ab12")}

    # ---- room level
    rl = s.groupby(["class", "room_id"])[["pred_useful", "useful_ab5", "useful_ab8"]].mean().reset_index()
    rl["err_vs_ab5"] = rl["pred_useful"] - rl["useful_ab5"]
    rl["err_vs_ab8"] = rl["pred_useful"] - rl["useful_ab8"]
    rl.to_csv(os.path.join(out, "room_level.csv"), index=False)

    summary = {"model": args.model, "n_features": len(features), "rooms_used": int(s["room_id"].nunique()),
               "rooms_skipped": skipped, "rooms_not_held_out": not_held,
               "by_class": table, "ladder_rooms": ladder}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    show = tab[["class", "rooms", "surrogate_vs_ab5_RMSE", "surrogate_vs_ab8_RMSE",
                "surrogate_vs_ab5_MAE", "surrogate_vs_ab8_MAE", "surrogate_vs_ab5_bias",
                "surrogate_vs_ab8_bias", "label_shift_MAE", "label_shift_bias"]
               + [c for c in ("rho_rooms_vs_ab5", "rho_rooms_vs_ab8") if c in tab]].round(3)
    head = "| " + " | ".join(show.columns) + " |\n|" + "---|" * len(show.columns) + "\n"
    body = "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in show.itertuples(index=False))
    md = "# Surrogate against original (ab5) and converged (ab8) labels\n\n" + head + body
    if ladder:
        md += f"\n\nLadder rooms ({lad['room_id'].nunique()} L-shapes): " + ", ".join(
            f"vs {k}: RMSE {v['RMSE']:.2f}, bias {v['bias']:+.2f}" for k, v in ladder.items())
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(md + "\n")
    T.log("\n" + md)
    T.log(f"\nOutputs -> {out}/")


if __name__ == "__main__":
    main()
