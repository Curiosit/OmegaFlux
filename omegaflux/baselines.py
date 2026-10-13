"""
omegaflux.baselines - same-data baselines for the per-sensor descriptor model (WO-B, B5).

Two kinds of comparison, both on exactly the splits train.py uses (E1 held-out rooms; E2 rectangles ->
L-shapes, excluding the reserved tuning L-shapes; E3 axis-aligned -> rotated; E4 <=2 -> >=3 windows):

  coord_xgb   Window-relative coordinates, the encoding of earlier per-sensor surrogates (sensor position
              relative to each window, window size and facing, room bounding box), with XGBoost and the
              same hyperparameters as the descriptor model. Windows fill K_MAX slots, nearest first;
              empty slots are NaN, which XGBoost treats as missing. E4 therefore runs, but slots 3-4 are
              never filled in training: the model cannot learn what an extra window does.
  mlp12/mlp26 A multilayer perceptron on the 12 or 26 descriptors, to separate the effect of the
              representation from the choice of learner.

Usage (from the repository root, after extract_features and train):
  python -m omegaflux.baselines --models coord_xgb --tests E1,E2,E3,E4 --params-from OmegaFlux_Models/core
  python -m omegaflux.baselines --models mlp12,mlp26 --tests E1,E2
Results are appended to <out>/results_table.csv; rerun with other --models/--tests to fill in the table.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from omegaflux import train as T
from omegaflux.extract_features import DATASETS, build_apertures, load_table, parse_room_geometry
from omegaflux.features import FEATURE_SETS

POOLED_CSV = os.path.join("data", "processed", "omegaflux_sensors_pooled.csv")
OUT_DIR = os.path.join("OmegaFlux_Models", "baselines")
K_MAX = 4                       # window slots (training data has at most 4 windows per room)
SLOT_FIELDS = ["along", "depth", "dz", "dist", "width", "height", "nx", "ny"]
COORD_ROOM = ["Meta_BBox_W", "Meta_BBox_D"]
META_NEEDED = ["Meta_L_char", "Meta_Sensor_X_norm", "Meta_Sensor_Y_norm", "Meta_Sensor_Z",
               "Meta_N_Windows", "Meta_BBox_W", "Meta_BBox_D"]
TESTS = ["E1", "E2", "E3", "E4"]


# =============================================================================
# Window-relative coordinate encoding
# =============================================================================
def room_geometry():
    """Group_ID -> (min_x, min_y, window array [n, 8]) parsed from the raw exports, as extract_features does."""
    geo = {}
    for enabled, path, suffix, label in DATASETS:
        if not enabled or not os.path.exists(path):
            T.log(f"[WARN] raw export missing: {path} ({label} rooms get no coordinate encoding)")
            continue
        df = load_table(path).dropna(subset=["Iteration_Name"])
        for _, row in df.iterrows():
            try:
                corners, specs = parse_room_geometry(row)
            except Exception:
                continue
            if len(corners) < 3:
                continue
            walls = [(corners[i], corners[(i + 1) % len(corners)]) for i in range(len(corners))]
            aps = build_apertures(corners, specs, walls)
            # per window: centre x, y, z, width, height, outward normal x, y
            win = np.array([[a["center"][0], a["center"][1], a["center"][2],
                             a["width"], a["height"], a["normal"][0], a["normal"][1]]
                            for a in aps], dtype=float).reshape(-1, 7)
            geo[f"{suffix}:{row['Iteration_Name']}"] = (min(p[0] for p in corners),
                                                       min(p[1] for p in corners), win)
    return geo


def coord_block(sx, sy, sz, win):
    """Per-sensor slot features for one room: sensors [m] x windows [n] -> [m, K_MAX * len(SLOT_FIELDS)]."""
    m = len(sx)
    out = np.full((m, K_MAX, len(SLOT_FIELDS)), np.nan, dtype=np.float32)
    if len(win) == 0:
        return out.reshape(m, -1)
    cx, cy, cz, w, h, nx, ny = (win[:, i] for i in range(7))
    rx, ry = -ny, nx                                    # along-wall direction
    dx = sx[:, None] - cx[None, :]
    dy = sy[:, None] - cy[None, :]
    along = dx * rx + dy * ry
    depth = -(dx * nx + dy * ny)                        # distance into the room from the window plane
    dz = cz[None, :] - sz[:, None]                      # window centre above the sensor
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    order = np.argsort(dist, axis=1)[:, :K_MAX]         # nearest window first
    rows = np.arange(m)[:, None]
    k = order.shape[1]
    stacks = [along, depth, dz, dist,
              np.broadcast_to(w, dist.shape), np.broadcast_to(h, dist.shape),
              np.broadcast_to(nx, dist.shape), np.broadcast_to(ny, dist.shape)]
    for j, arr in enumerate(stacks):
        out[:, :k, j] = arr[rows, order]
    return out.reshape(m, -1)


def add_coordinates(df):
    cols = [f"Coord_w{s + 1}_{f}" for s in range(K_MAX) for f in SLOT_FIELDS]
    T.log("Parsing raw exports for window geometry...")
    geo = room_geometry()
    block = np.full((len(df), len(cols)), np.nan, dtype=np.float32)
    t0, n_sensors, missing = time.perf_counter(), 0, 0
    for gid, idx in df.groupby("Group_ID", sort=False).indices.items():
        if gid not in geo:
            missing += 1
            continue
        min_x, min_y, win = geo[gid]
        g = df.iloc[idx]
        lc = g["Meta_L_char"].to_numpy()
        sx = g["Meta_Sensor_X_norm"].to_numpy() * lc + min_x
        sy = g["Meta_Sensor_Y_norm"].to_numpy() * lc + min_y
        sz = g["Meta_Sensor_Z"].to_numpy()
        block[idx] = coord_block(sx, sy, sz, win)
        n_sensors += len(idx)
    per_pt = (time.perf_counter() - t0) / max(n_sensors, 1) * 1e3
    if missing:
        T.log(f"[WARN] {missing} rooms in the pooled table have no parsed geometry; their coordinates are NaN")
    T.log(f"Coordinate encoding: {len(cols)} columns, {per_pt:.4f} ms per grid point (excluding file parsing)")
    for j, c in enumerate(cols):
        df[c] = block[:, j]
    return cols + COORD_ROOM, per_pt


# =============================================================================
# MLP learner with the predict() interface train.evaluate expects
# =============================================================================
def fit_mlp_pair(df_tr, features, max_rows, seed):
    if len(df_tr) > max_rows:
        df_tr = df_tr.sample(max_rows, random_state=seed)
    X = df_tr[features].astype("float32").to_numpy()
    models = []
    for tgt in ("Target_UDI_low", "Target_UDI_up"):
        mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
                           batch_size=2048, learning_rate_init=1e-3, max_iter=60,
                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=6,
                           random_state=seed)
        m = make_pipeline(StandardScaler(), mlp)
        T.log(f"  MLP -> {tgt} on {len(X):,} rows ...")
        m.fit(X, df_tr[tgt].to_numpy())
        models.append(m)
    return models


def run_mlp(name, df_tr, df_te, features, max_rows, seed):
    t0 = time.perf_counter()
    m_low, m_up = fit_mlp_pair(df_tr, features, max_rows, seed)
    train_s = time.perf_counter() - t0
    res = T.evaluate(name, m_low, m_up, df_te[features].astype("float32").to_numpy(),
                     df_te["Target_UDI_low"].values, df_te["Target_UDI_up"].values, df_te["Group_ID"].values)
    res.update({"n_features": len(features), "n_train_rooms": int(df_tr["Group_ID"].nunique()),
                "n_train_rows": int(min(len(df_tr), max_rows)), "train_seconds": round(train_s, 1)})
    return res


# =============================================================================
# Splits (identical to train.py)
# =============================================================================
def splits(df, tests):
    gss = GroupShuffleSplit(n_splits=1, test_size=T.TEST_FRACTION, random_state=T.SEED)
    tr_idx, te_idx = next(gss.split(df, groups=df["Group_ID"]))
    df_train = df.iloc[tr_idx]
    ls = np.array(sorted(df_train.loc[df_train["Source_Tag"] == "randomized", "Group_ID"].unique()))
    reserve = set()
    if len(ls):
        k = max(1, int(round(len(ls) * T.TUNE_HOLDOUT_LSHAPE_FRACTION)))
        reserve = set(np.random.default_rng(T.SEED).choice(ls, size=k, replace=False))
    out = {}
    if "E1" in tests:
        out["E1_main"] = (tr_idx, te_idx)
    tag = df["Source_Tag"].to_numpy()
    if "E2" in tests:
        te = np.where((tag == "randomized") & ~df["Group_ID"].isin(reserve).to_numpy())[0]
        out["E2_holdout_Lshapes"] = (np.where(np.isin(tag, ["rect", "rect_rotated"]))[0], te)
    if "E3" in tests:
        out["E3_holdout_rotated"] = (np.where(np.isin(tag, ["rect", "randomized"]))[0],
                                     np.where(tag == "rect_rotated")[0])
    if "E4" in tests:
        nw = df["Meta_N_Windows"].to_numpy()
        out["E4_unseen_aperture_count"] = (np.where(nw <= 2)[0], np.where(nw >= 3)[0])
    return out


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="OmegaFlux same-data baselines")
    ap.add_argument("--models", default="coord_xgb,mlp12,mlp26", help="coord_xgb, mlp12, mlp26")
    ap.add_argument("--tests", default="E1,E2,E3,E4")
    ap.add_argument("--params-from", default=os.path.join("OmegaFlux_Models", "core"),
                    help="hyperparameters.json for coord_xgb (same as the descriptor model)")
    ap.add_argument("--mlp-max-rows", type=int, default=1_000_000,
                    help="random training rows for the MLP (sklearn MLP is CPU-bound)")
    ap.add_argument("--data", default=POOLED_CSV)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    bad = [m for m in models if m not in ("coord_xgb", "mlp12", "mlp26")] + [t for t in tests if t not in TESTS]
    if bad:
        raise SystemExit(f"unknown --models/--tests: {bad}")
    os.makedirs(args.out, exist_ok=True)

    T.banner("BASELINES - loading pooled table")
    usecols = (["Group_ID", "Source_Tag", "Iteration_Name", "Target_UDI_low", "Target_UDI_up"]
               + FEATURE_SETS["all"] + META_NEEDED)
    df = pd.read_csv(args.data, usecols=usecols,
                     dtype={"Group_ID": str, "Source_Tag": str, "Iteration_Name": str})
    for c in FEATURE_SETS["all"] + META_NEEDED:
        df[c] = df[c].astype("float32")
    T.log(f"{len(df):,} grid points / {df['Group_ID'].nunique():,} rooms")

    with open(os.path.join(args.params_from, "hyperparameters.json")) as fh:
        params = json.load(fh)["selected"]

    coord_cols, coord_ms = ([], None)
    if "coord_xgb" in models:
        coord_cols, coord_ms = add_coordinates(df)

    feature_sets = {"coord_xgb": coord_cols, "mlp12": FEATURE_SETS["core"], "mlp26": FEATURE_SETS["all"]}
    results = []
    for test, (tr, te) in splits(df, tests).items():
        d_tr, d_te = df.iloc[tr], df.iloc[te]
        for model in models:
            name = f"{model}__{test}"
            T.banner(name)
            if model == "coord_xgb":
                r, _, _ = T.train_and_eval(name, d_tr, d_te, feature_sets[model], params)
                r["coord_ms_per_point"] = coord_ms
            else:
                r = run_mlp(name, d_tr, d_te, feature_sets[model], args.mlp_max_rows, T.SEED)
            r.update({"baseline": model, "test": test})
            T.log(f"{name}: R2 {r['useful_R2']:.4f} | RMSE {r['useful_RMSE']:.2f} | rho {r.get('room_spearman_rho', float('nan')):.4f}")
            results.append(r)
            # write after every run, so a crash (e.g. memory) keeps what finished
            path = os.path.join(args.out, "results_table.csv")
            old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
            new = pd.DataFrame([r])
            if len(old):
                old = old[old["experiment"] != name]
            pd.concat([old, new], ignore_index=True).to_csv(path, index=False)

    with open(os.path.join(args.out, f"manifest_baselines_{'_'.join(models)}.json"), "w") as fh:
        json.dump({"models": models, "tests": tests, "params": params, "k_max": K_MAX,
                   "slot_fields": SLOT_FIELDS, "coord_room_fields": COORD_ROOM,
                   "mlp_max_rows": args.mlp_max_rows, "seed": T.SEED}, fh, indent=2)
    T.log(f"\nResults -> {os.path.join(args.out, 'results_table.csv')}")


if __name__ == "__main__":
    main()
