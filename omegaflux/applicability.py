"""
omegaflux.applicability - when should a prediction not be trusted? (WO-B, B8)

A grid point is flagged as outside the model's applicability domain when its descriptors lie far from
the training data: the mean distance to its k nearest training points, in standardised descriptor space,
exceeds a threshold. The threshold is calibrated on held-out rooms of the training family (the 95th
percentile by default, so about 5% of in-family points are flagged by construction). The check is then
applied to a transfer test, where the question is whether the flag finds the large errors.

Contexts (same splits as train.py):
  E2  train rectangular + rotated rooms -> test L-shapes (reserved tuning L-shapes excluded)
  E3  train rectangular + L-shaped      -> test rotated
  E4  train rooms with <= 2 windows     -> test rooms with >= 3 windows
Calibration rooms are the E1 held-out rooms inside the training family. They are removed from the
neighbour reference set (otherwise each would find itself at distance zero) but stay in model training,
so the model is the one train.py evaluates.

It also breaks the transfer-test error down by what each grid point sees (mechanism check): points
with no window in view, points whose view of a window is partly blocked by the room's own walls, and
points with a clear view, and by decile of total window solid angle - separately for the under- and
over-illuminated fractions, so the source of a bias is visible.

Usage:  python -m omegaflux.applicability --model OmegaFlux_Models/core --context E2 [--save-points]
Output: <model>/applicability_<context>/{summary.md, summary.json, by_distance_decile.csv,
        room_level.csv, error_vs_distance.png, by_view_class.csv, by_solid_angle_decile.csv,
        error_by_view.png, [test_points.csv.gz]}
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors

from omegaflux import train as T

POOLED_CSV = os.path.join("data", "processed", "omegaflux_sensors_pooled.csv")


def masks(df, context):
    tag = df["Source_Tag"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=T.TEST_FRACTION, random_state=T.SEED)
    tr_idx, te_idx = next(gss.split(df, groups=df["Group_ID"]))
    e1_test = np.zeros(len(df), bool)
    e1_test[te_idx] = True
    if context == "E2":
        ls = np.array(sorted(df.iloc[tr_idx].loc[lambda d: d["Source_Tag"] == "randomized", "Group_ID"].unique()))
        k = max(1, int(round(len(ls) * T.TUNE_HOLDOUT_LSHAPE_FRACTION)))
        reserve = set(np.random.default_rng(T.SEED).choice(ls, size=k, replace=False)) if len(ls) else set()
        train = np.isin(tag, ["rect", "rect_rotated"])
        test = (tag == "randomized") & ~df["Group_ID"].isin(reserve).to_numpy()
    elif context == "E3":
        train = np.isin(tag, ["rect", "randomized"])
        test = tag == "rect_rotated"
    elif context == "E4":
        nw = df["Meta_N_Windows"].to_numpy()
        train, test = nw <= 2, nw >= 3
    else:
        raise SystemExit(f"unknown context {context}")
    calib = train & e1_test
    reference = train & ~e1_test
    return train, test, calib, reference


def knn_distance(nn, Z, k, batch=200_000):
    out = np.empty(len(Z), dtype=np.float32)
    for i in range(0, len(Z), batch):
        d, _ = nn.kneighbors(Z[i:i + batch], n_neighbors=k)
        out[i:i + batch] = d.mean(axis=1)
    return out


def err_stats(e):
    return {"N": int(len(e)), "RMSE": float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan"),
            "MAE": float(np.mean(np.abs(e))) if len(e) else float("nan"),
            "bias": float(np.mean(e)) if len(e) else float("nan"),
            "P95_abs": float(np.percentile(np.abs(e), 95)) if len(e) else float("nan")}


def main():
    ap = argparse.ArgumentParser(description="Applicability-domain flag for OmegaFlux predictions")
    ap.add_argument("--model", default=os.path.join("OmegaFlux_Models", "core"),
                    help="model folder: features.json and hyperparameters.json are read from it")
    ap.add_argument("--context", default="E2", choices=["E2", "E3", "E4"])
    ap.add_argument("--k", type=int, default=10, help="neighbours averaged for the distance")
    ap.add_argument("--quantile", type=float, default=0.95, help="calibration quantile for the threshold")
    ap.add_argument("--ref-size", type=int, default=200_000, help="random reference points for the neighbour search")
    ap.add_argument("--room-flag-share", type=float, default=0.5,
                    help="a room is flagged when at least this share of its grid points is flagged")
    ap.add_argument("--data", default=POOLED_CSV)
    ap.add_argument("--save-points", action="store_true", help="also write per-point test predictions")
    args = ap.parse_args()
    out = os.path.join(args.model, f"applicability_{args.context}")
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(args.model, "features.json")) as fh:
        features = json.load(fh)["features"]
    with open(os.path.join(args.model, "hyperparameters.json")) as fh:
        params = json.load(fh)["selected"]

    T.banner(f"APPLICABILITY - context {args.context}, {len(features)} descriptors")
    view_cols = ["Total_Omega", "Visible_Window_Count", "Occlusion_Multiplier", "Inner_Corner_Dmin"]
    cols = ["Group_ID", "Source_Tag", "Target_UDI_low", "Target_UDI_up", "Meta_N_Windows"] + features + view_cols
    df = pd.read_csv(args.data, usecols=list(dict.fromkeys(cols)), dtype={"Group_ID": str, "Source_Tag": str})
    for c in features:
        df[c] = df[c].astype("float32")
    train, test, calib, reference = masks(df, args.context)
    T.log(f"train {train.sum():,} | test {test.sum():,} | calibration {calib.sum():,} | reference pool {reference.sum():,}")

    # ---- the model this context evaluates (as train.py's topology step)
    res, m_low, m_up = T.train_and_eval(f"{args.context}_model", df[train], df[test], features, params)
    T.log(f"{args.context} model: R2 {res['useful_R2']:.4f} | RMSE {res['useful_RMSE']:.2f}")

    def predict(mask):
        X = df.loc[mask, features].to_numpy()
        p_low, p_up, p_use = T.predict_useful(m_low, m_up, X)
        a_low = df.loc[mask, "Target_UDI_low"].to_numpy()
        a_up = df.loc[mask, "Target_UDI_up"].to_numpy()
        a_use = np.clip(100 - a_low - a_up, 0, 100)
        return p_low, p_up, p_use, a_low, a_up, a_use

    def useful_error(mask):
        _, _, p, _, _, a = predict(mask)
        return p - a

    # ---- distance to the training data in standardised descriptor space
    ref_rows = np.flatnonzero(reference)
    rng = np.random.default_rng(T.SEED)
    if len(ref_rows) > args.ref_size:
        ref_rows = rng.choice(ref_rows, args.ref_size, replace=False)
    R = df.iloc[ref_rows][features].to_numpy()
    mu, sd = R.mean(axis=0), R.std(axis=0)
    sd[sd < 1e-9] = 1.0                      # constant descriptors (e.g. occlusion terms in convex rooms)
    nn = NearestNeighbors(n_neighbors=args.k, n_jobs=-1).fit((R - mu) / sd)
    t0 = time.perf_counter()
    d_cal = knn_distance(nn, (df.loc[calib, features].to_numpy() - mu) / sd, args.k)
    d_test = knn_distance(nn, (df.loc[test, features].to_numpy() - mu) / sd, args.k)
    T.log(f"Neighbour distances: {len(d_cal) + len(d_test):,} points in {time.perf_counter() - t0:.0f} s")
    thr = float(np.quantile(d_cal, args.quantile))

    e_cal = useful_error(calib)
    tp_low, tp_up, tp_use, ta_low, ta_up, ta_use = predict(test)
    e_test = tp_use - ta_use
    flag = d_test > thr
    T.log(f"Threshold {thr:.3f} | flagged: calibration {np.mean(d_cal > thr):.1%}, test {flag.mean():.1%}")

    summary = {
        "context": args.context, "features": features, "k": args.k, "quantile": args.quantile,
        "threshold": thr, "ref_size": int(len(ref_rows)),
        "model": {k: res[k] for k in ("useful_R2", "useful_RMSE", "useful_MAE") if k in res},
        "share_flagged_calibration": float(np.mean(d_cal > thr)),
        "share_flagged_test": float(flag.mean()),
        "calibration_error": err_stats(e_cal),
        "test_unflagged": err_stats(e_test[~flag]),
        "test_flagged": err_stats(e_test[flag]),
        "test_all": err_stats(e_test),
        "spearman_distance_vs_abs_error": float(spearmanr(d_test, np.abs(e_test)).statistic),
    }

    # ---- error by distance decile (test points)
    q = pd.qcut(d_test, 10, labels=False, duplicates="drop")
    dec = (pd.DataFrame({"decile": q, "dist": d_test, "abs_err": np.abs(e_test), "err": e_test, "flag": flag})
           .groupby("decile").agg(dist_min=("dist", "min"), dist_max=("dist", "max"),
                                  MAE=("abs_err", "mean"), RMSE=("err", lambda x: float(np.sqrt(np.mean(x ** 2)))),
                                  share_flagged=("flag", "mean"), N=("err", "size")).reset_index())
    dec.to_csv(os.path.join(out, "by_distance_decile.csv"), index=False)

    # ---- room level: does the share of flagged points identify the badly predicted rooms?
    g = df.loc[test, "Group_ID"].to_numpy()
    room = (pd.DataFrame({"Group_ID": g, "flag": flag, "err": e_test, "abs_err": np.abs(e_test), "dist": d_test})
            .groupby("Group_ID").agg(share_flagged=("flag", "mean"), room_bias=("err", "mean"),
                                     room_MAE=("abs_err", "mean"), mean_dist=("dist", "mean"), N=("err", "size"))
            .reset_index())
    room["room_flagged"] = room["share_flagged"] >= args.room_flag_share
    room.to_csv(os.path.join(out, "room_level.csv"), index=False)
    fl, un = room[room["room_flagged"]], room[~room["room_flagged"]]
    summary["rooms"] = {
        "n": int(len(room)), "flagged": int(len(fl)), "room_flag_share": args.room_flag_share,
        "MAE_flagged_rooms": float(fl["room_MAE"].mean()) if len(fl) else float("nan"),
        "MAE_unflagged_rooms": float(un["room_MAE"].mean()) if len(un) else float("nan"),
        "spearman_share_flagged_vs_room_MAE": float(spearmanr(room["share_flagged"], room["room_MAE"]).statistic),
        "worst_decile_rooms_caught": float(room.loc[room["room_MAE"] >= room["room_MAE"].quantile(0.9),
                                                    "room_flagged"].mean()),
    }
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- mechanism: error by what the grid point sees
    tv = df.loc[test, view_cols].reset_index(drop=True)
    view = np.where(tv["Visible_Window_Count"].to_numpy() <= 0, "no window in view",
                    np.where(tv["Occlusion_Multiplier"].to_numpy() < 0.99, "view partly blocked by own walls",
                             "clear view of a window"))
    pts = pd.DataFrame({"Group_ID": g,
                        "view": view, "total_omega": tv["Total_Omega"].to_numpy(),
                        "inner_corner_dmin": tv["Inner_Corner_Dmin"].to_numpy(),
                        "dist": d_test, "flag": flag,
                        "pred_low": tp_low, "act_low": ta_low, "pred_up": tp_up, "act_up": ta_up,
                        "pred_useful": tp_use, "act_useful": ta_use})
    pts["err_useful"] = pts["pred_useful"] - pts["act_useful"]
    pts["err_low"] = pts["pred_low"] - pts["act_low"]
    pts["err_up"] = pts["pred_up"] - pts["act_up"]

    def breakdown(by):
        gb = pts.groupby(by, observed=True)
        t = gb.agg(N=("err_useful", "size"), act_useful=("act_useful", "mean"), act_low=("act_low", "mean"),
                   bias_useful=("err_useful", "mean"), MAE_useful=("err_useful", lambda x: float(np.abs(x).mean())),
                   bias_low=("err_low", "mean"), bias_up=("err_up", "mean"),
                   share_flagged=("flag", "mean"), mean_dist=("dist", "mean")).reset_index()
        t["share_of_points"] = t["N"] / t["N"].sum()
        t["share_of_abs_error"] = gb["err_useful"].apply(lambda x: float(np.abs(x).sum())).values / np.abs(pts["err_useful"]).sum()
        return t

    order = ["clear view of a window", "view partly blocked by own walls", "no window in view"]
    pts["view"] = pd.Categorical(pts["view"], categories=order, ordered=True)
    by_view = breakdown("view")
    by_view["view"] = by_view["view"].astype(str)
    by_view.to_csv(os.path.join(out, "by_view_class.csv"), index=False)
    pts["omega_decile"] = pd.qcut(pts["total_omega"].rank(method="first"), 10, labels=False)
    by_om = breakdown("omega_decile")
    om_rng = pts.groupby("omega_decile")["total_omega"].agg(["min", "max"]).reset_index(drop=True)
    by_om.insert(1, "omega_min", om_rng["min"].values)
    by_om.insert(2, "omega_max", om_rng["max"].values)
    by_om.to_csv(os.path.join(out, "by_solid_angle_decile.csv"), index=False)
    summary["by_view_class"] = by_view.to_dict(orient="records")
    if args.save_points:
        pts.to_csv(os.path.join(out, "test_points.csv.gz"), index=False, compression="gzip")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    x = np.arange(len(by_view))
    ax.bar(x - 0.2, by_view["bias_useful"], 0.4, color="#C44E52", label="bias, UDI$_{useful}$")
    ax.bar(x + 0.2, by_view["MAE_useful"], 0.4, color="#8C8C8C", label="mean absolute error")
    ax.set_xticks(x)
    short = {"clear view of a window": "clear view", "view partly blocked by own walls": "partly blocked\nby own walls",
             "no window in view": "no window\nin view"}
    ax.set_xticklabels([f"{short.get(v, v)}\n({s:.0%} of points)" for v, s in zip(by_view["view"], by_view["share_of_points"])], fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("pp")
    ax.set_title(f"Error by view class ({args.context})")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    xm = np.maximum(0.5 * (by_om["omega_min"] + by_om["omega_max"]), 1e-4)
    ax.plot(xm, by_om["bias_useful"], "o-", color="#C44E52", label="UDI$_{useful}$")
    ax.plot(xm, by_om["bias_low"], "s--", color="#4C72B0", label="UDI$_{low}$")
    ax.plot(xm, by_om["bias_up"], "^:", color="#DD8452", label="UDI$_{up}$")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Total window solid angle (decile midpoint, fraction of hemisphere)")
    ax.set_ylabel("Bias, predicted minus simulated (pp)")
    ax.set_title("Bias by solid-angle decile")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "error_by_view.png"), dpi=300)
    plt.close(fig)

    # ---- figure
    fig, ax = plt.subplots(figsize=(7, 4.2))
    mid = 0.5 * (dec["dist_min"] + dec["dist_max"])
    ax.plot(mid, dec["MAE"], "o-", color="#C44E52", label="test (transfer)")
    ax.axhline(summary["calibration_error"]["MAE"], color="#4C72B0", ls="--", lw=1.2,
               label="held-out rooms of the training family")
    ax.axvline(thr, color="k", ls=":", lw=1.2, label=f"threshold ({args.quantile:.0%} of in-family points below)")
    ax.set_xlabel(f"Mean distance to the {args.k} nearest training points (standardised descriptors)")
    ax.set_ylabel("Mean absolute error, UDI$_{useful}$ (pp)")
    ax.set_title(f"Error against distance from the training data ({args.context})")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "error_vs_distance.png"), dpi=300)
    plt.close(fig)

    s, r = summary, summary["rooms"]
    md = (f"# Applicability flag - {args.context}\n\n"
          f"Model on {len(features)} descriptors; threshold = {args.quantile:.0%} quantile of in-family held-out distances "
          f"({thr:.3f}); k = {args.k}; reference {s['ref_size']:,} points.\n\n"
          f"| set | points | share flagged | MAE | RMSE | P95 abs |\n|---|---|---|---|---|---|\n"
          f"| held-out, training family | {s['calibration_error']['N']:,} | {s['share_flagged_calibration']:.1%} | "
          f"{s['calibration_error']['MAE']:.2f} | {s['calibration_error']['RMSE']:.2f} | {s['calibration_error']['P95_abs']:.2f} |\n"
          f"| transfer test, unflagged | {s['test_unflagged']['N']:,} | 0% | {s['test_unflagged']['MAE']:.2f} | "
          f"{s['test_unflagged']['RMSE']:.2f} | {s['test_unflagged']['P95_abs']:.2f} |\n"
          f"| transfer test, flagged | {s['test_flagged']['N']:,} | 100% | {s['test_flagged']['MAE']:.2f} | "
          f"{s['test_flagged']['RMSE']:.2f} | {s['test_flagged']['P95_abs']:.2f} |\n\n"
          f"Share of transfer-test points flagged: {s['share_flagged_test']:.1%}. "
          f"Spearman(distance, |error|) = {s['spearman_distance_vs_abs_error']:.3f}.\n\n"
          f"Rooms: {r['flagged']} of {r['n']} flagged (>= {r['room_flag_share']:.0%} of points); "
          f"mean room MAE flagged {r['MAE_flagged_rooms']:.2f} vs unflagged {r['MAE_unflagged_rooms']:.2f}; "
          f"worst 10% of rooms caught: {r['worst_decile_rooms_caught']:.0%}; "
          f"Spearman(share flagged, room MAE) = {r['spearman_share_flagged_vs_room_MAE']:.3f}.\n")
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    md += ("\n## Error by what the point sees (transfer test)\n\n"
           "| view | share of points | share of abs. error | simulated UDI useful | bias useful | MAE useful | bias low | bias up | share flagged |\n"
           "|---|---|---|---|---|---|---|---|---|\n"
           + "".join(f"| {r.view} | {r.share_of_points:.1%} | {r.share_of_abs_error:.1%} | {r.act_useful:.1f} | "
                     f"{r.bias_useful:+.2f} | {r.MAE_useful:.2f} | {r.bias_low:+.2f} | {r.bias_up:+.2f} | {r.share_flagged:.1%} |\n"
                     for r in by_view.itertuples()))
    md += ("\nBias by decile of total window solid angle (useful / low / up): "
           + "; ".join(f"[{r.omega_min:.3f}-{r.omega_max:.3f}] {r.bias_useful:+.1f}/{r.bias_low:+.1f}/{r.bias_up:+.1f}"
                       for r in by_om.itertuples()) + "\n")
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(md)
    T.log("\n" + md)
    T.log(f"Outputs -> {out}/")


if __name__ == "__main__":
    main()
