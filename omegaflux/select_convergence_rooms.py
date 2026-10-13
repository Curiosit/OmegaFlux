"""
select_convergence_rooms.py

WO-A, step A1: reproduce the E1 held-out split exactly as train.py makes it,
then select rooms for the simulation convergence check.

Run from the repository root:
    python -m omegaflux.select_convergence_rooms --setup-dir <folder with the three setup files>

The setup files are the Grasshopper input tables (one row per room) used to generate the dataset.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# --- must match train.py --------------------------------------------------
SEED = 42
TEST_FRACTION = 0.10
POOLED_CANDIDATES = [
    os.path.join("data", "processed", "omegaflux_sensors_pooled.csv"),  # train.py
    os.path.join("data_processed", "omegaflux_sensors_pooled.csv"),     # explain.py / make_figures.py
]
PRED_CACHE = os.path.join("OmegaFlux_Models", "test_predictions.npz")  # optional cross-check

# --- setup tables (Grasshopper inputs) ------------------------------------
SETUP = {
    "rect":         ("simulation_setup_absolute_metrics.csv", ","),
    "rect_rotated": ("simulation_setup_rotated_metrics.csv", ","),
    "randomized":   ("grasshopper_l_shapes_1125_LHS_Batched_Final.csv", "$"),
}

# --- sample design ----------------------------------------------------------
N_PER_SOURCE = 50
N_V2_PER_SOURCE = 10
N_PILOT = 5
OUT_DIR = os.path.join("data", "convergence_check")

# Manuscript Table 3/4: the split is only trusted if these reproduce exactly
EXPECTED_TEST_ROOMS = 675
EXPECTED_TEST_POINTS = {"rect": 171_648, "rect_rotated": 44_805, "randomized": 102_912}


# ===========================================================================
# 1. Held-out rooms
# ===========================================================================
def load_pooled():
    for p in POOLED_CANDIDATES:
        if os.path.exists(p):
            print(f"Reading {p} (3 columns only) ...")
            return pd.read_csv(p, usecols=["Group_ID", "Source_Tag", "Iteration_Name"], dtype=str)
    raise SystemExit(f"Pooled CSV not found in any of: {POOLED_CANDIDATES}")


def held_out_rooms(skip_verify=False):
    df = load_pooled()
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED)
    _, te = next(gss.split(df, groups=df["Group_ID"]))
    te_df = df.iloc[te]

    rooms = (te_df.groupby("Group_ID")
             .agg(Source_Tag=("Source_Tag", "first"),
                  Iteration_Name=("Iteration_Name", "first"),
                  n_sensors=("Group_ID", "size"))
             .reset_index())

    pts = te_df["Source_Tag"].value_counts().to_dict()
    print(f"\nHeld-out rooms : {len(rooms)} (expected {EXPECTED_TEST_ROOMS})")
    for tag, exp in EXPECTED_TEST_POINTS.items():
        got = pts.get(tag, 0)
        print(f"  {tag:13s} {got:>8,} grid points (expected {exp:,}) {'OK' if got == exp else 'MISMATCH'}")

    ok = len(rooms) == EXPECTED_TEST_ROOMS and all(pts.get(k, 0) == v for k, v in EXPECTED_TEST_POINTS.items())

    if os.path.exists(PRED_CACHE):
        cached = set(np.load(PRED_CACHE, allow_pickle=False)["group_id"].astype(str))
        same = cached == set(rooms["Group_ID"])
        print(f"  cross-check vs {PRED_CACHE}: {'identical' if same else 'DIFFERENT'}")
        ok = ok and same

    if not ok and not skip_verify:
        raise SystemExit("\nSplit does NOT reproduce the manuscript. Stop and report the numbers above.")
    print("Split reproduced exactly.\n" if ok else "[debug] verification skipped.\n")
    return rooms


# ===========================================================================
# 2. Join held-out rooms to their setup rows
# ===========================================================================
def norm(s):
    s = str(s).strip()
    return s[:-2] if s.endswith(".0") else s


SETUP_DIR = "."


def load_setup(tag):
    path, sep = SETUP[tag]
    path = os.path.join(SETUP_DIR, path)
    raw = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)  # kept verbatim for writing back
    return raw, path, sep


def match_keys(held, raw, tag):
    """Find which setup column the pooled Iteration_Name corresponds to."""
    cands = {}
    for col in ("Iteration_ID", "Iteration_Name"):
        if col in raw.columns:
            cands[col] = raw[col].map(norm)
    if "Row_Nr" in raw.columns:
        cands["Row_Nr"] = raw["Row_Nr"].map(norm)
        cands["Row_Nr-1"] = (raw["Row_Nr"].astype(int) - 1).astype(str)
    cands["row_index+1"] = pd.Series(np.arange(1, len(raw) + 1).astype(str), index=raw.index)
    cands["row_index"] = pd.Series(np.arange(len(raw)).astype(str), index=raw.index)

    names = held["Iteration_Name"].map(norm)
    best, best_rate = None, -1.0
    for k, keys in cands.items():
        rate = names.isin(set(keys)).mean()
        if rate > best_rate:
            best, best_rate = k, rate
    print(f"  {tag:13s} Iteration_Name matches setup '{best}' for {best_rate:.1%} of held-out rooms")
    if best_rate < 1.0:
        raise SystemExit(f"Cannot map all held-out {tag} rooms to the setup table. "
                         f"Examples of pooled Iteration_Name: {names.head(5).tolist()}")
    key = cands[best]
    idx = {v: i for i, v in key.items()}
    return names.map(idx)  # setup row index per held-out room


def shoelace(pts):
    return 0.5 * abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                         for i in range(len(pts))))


def describe(tag, raw, rows):
    """Stratum and sensitivity descriptors for each selected setup row."""
    r = raw.loc[rows]
    if tag in ("rect", "rect_rotated"):
        f = lambda c: r[c].astype(float)
        glz = sum(f(f"{w}_Width_m") * f(f"{w}_Height_m")
                  for w in ("Front_Win1", "Front_Win2", "Side_Win1", "Side_Win2"))
        area = f("Room_Width_m") * f("Room_Depth_m")
        stratum = r["Bin_ID"] if tag == "rect" else r["Bin_ID"] + "|" + r["Orientation_deg"].map(norm)
        size = f("Room_Depth_m")                                   # deeper = more interreflection-dependent
    else:
        corners = r["Room_corners"].map(lambda s: [tuple(map(float, m.split(",")))
                                                   for m in re.findall(r"\{([^}]*)\}", s)])
        area = corners.map(shoelace)
        widths = r["Window_widths"].map(lambda s: [float(x) for x in s.split(";")])
        heights = r["Window_heights"].map(lambda s: [float(x) for x in s.split(";")])
        glz = pd.Series([sum(a * b for a, b in zip(w, h)) for w, h in zip(widths, heights)], index=r.index)
        stratum = widths.map(lambda w: f"{len(w)}win")
        size = area                                                  # larger plan = longer bounce paths
    return pd.DataFrame({"stratum": stratum.values, "glazing_ratio": (glz / area).values,
                         "size_metric": size.values}, index=r.index)


# ===========================================================================
# 3. Stratified selection
# ===========================================================================
def allocate(sizes, n, rng):
    """Equal allocation across strata (round-robin), capped by availability."""
    quota = {s: 0 for s in sizes}
    left = n
    while left > 0:
        progressed = False
        for s in rng.permutation(list(sizes)):
            if left == 0:
                break
            if quota[s] < sizes[s]:
                quota[s] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    return quota


def select_source(cand, n, rng):
    forced = {cand["glazing_ratio"].idxmin(): "forced_min_glazing",
              cand["size_metric"].idxmax(): "forced_max_size"}
    quota = allocate(cand.groupby("stratum").size().to_dict(), n, rng)
    parts = []
    for s, q in quota.items():
        pool = cand[cand["stratum"] == s]
        f = pool[pool.index.isin(forced)].copy()
        f["reason"] = [forced[i] for i in f.index]
        rest = pool[~pool.index.isin(forced)]
        k = min(max(q - len(f), 0), len(rest))
        pick = rest.sample(n=k, random_state=int(rng.integers(1_000_000_000))).copy()
        pick["reason"] = "random"
        parts.append(pd.concat([f, pick]))
    return pd.concat(parts)


def pick_v2(sel, n, rng):
    """All forced rooms, then a systematic sample spread across strata."""
    forced = sel[sel["reason"] != "random"]
    rest = (sel[sel["reason"] == "random"]
            .sample(frac=1.0, random_state=int(rng.integers(1_000_000_000)))
            .sort_values("stratum", kind="stable"))
    k = max(n - len(forced), 0)
    idx = np.unique(np.linspace(0, len(rest) - 1, k).round().astype(int)) if k else []
    return set(forced.index) | set(rest.index[idx])


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-verify", action="store_true", help="debug only: continue if the split check fails")
    ap.add_argument("--setup-dir", default=".", help="folder containing the three setup tables")
    args = ap.parse_args()
    global SETUP_DIR
    SETUP_DIR = args.setup_dir

    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    held = held_out_rooms(args.skip_verify)
    held.to_csv(os.path.join(OUT_DIR, "held_out_rooms.csv"), index=False)

    selected_all = []
    for tag in SETUP:
        raw, path, sep = load_setup(tag)
        h = held[held["Source_Tag"] == tag].reset_index(drop=True)
        rows = match_keys(h, raw, tag)
        cand = describe(tag, raw, rows.values)
        cand["Group_ID"] = h["Group_ID"].values
        cand["Iteration_Name"] = h["Iteration_Name"].values
        cand["n_sensors"] = h["n_sensors"].values

        sel = select_source(cand, N_PER_SOURCE, rng)
        v2 = pick_v2(sel, N_V2_PER_SOURCE, rng)
        sel["v2_flag"] = sel.index.isin(v2)
        sel["Source_Tag"] = tag
        sel["setup_file"] = path
        sel["setup_row"] = sel.index
        selected_all.append(sel)

        # Grasshopper inputs in the ORIGINAL format (same columns, separator, line endings)
        base = os.path.splitext(os.path.basename(path))[0]
        order = sorted(sel.index)
        raw.loc[order].to_csv(os.path.join(OUT_DIR, f"{base}_V1.csv"), sep=sep, index=False, lineterminator="\r\n")
        raw.loc[sorted(v2)].to_csv(os.path.join(OUT_DIR, f"{base}_V2.csv"), sep=sep, index=False, lineterminator="\r\n")

    sel = pd.concat(selected_all, ignore_index=True)

    # timing pilot: rooms nearest the 0/25/50/75/100th percentile of sensor count
    pilot = set()
    for q in np.linspace(0, 1, N_PILOT):
        target = sel["n_sensors"].quantile(q)
        order = (sel["n_sensors"] - target).abs().sort_values().index
        pilot.add(next(i for i in order if i not in pilot))
    sel["pilot_flag"] = sel.index.isin(pilot)

    cols = ["Group_ID", "Source_Tag", "Iteration_Name", "setup_file", "setup_row", "stratum",
            "n_sensors", "glazing_ratio", "size_metric", "reason", "v2_flag", "pilot_flag"]
    sel[cols].to_csv(os.path.join(OUT_DIR, "selected_rooms.csv"), index=False)

    print("\nSelection summary")
    print(sel.groupby(["Source_Tag", "stratum"]).agg(rooms=("Group_ID", "size"), v2=("v2_flag", "sum"))
          .to_string())
    print(f"\nTotal {len(sel)} rooms | V2 {int(sel['v2_flag'].sum())} | pilot {int(sel['pilot_flag'].sum())}")
    print(f"Pilot rooms:\n{sel.loc[sel['pilot_flag'], ['Source_Tag', 'Iteration_Name', 'n_sensors']].to_string(index=False)}")
    print(f"\nOutputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
