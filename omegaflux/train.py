import argparse
import json
import os
import platform
import time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from xgboost import XGBRegressor

# =============================================================================
# CONFIGURATION
# =============================================================================
SEED = 42
DATA_DIR = "data_processed"
POOLED_CSV = os.path.join(DATA_DIR, "omegaflux_sensors_pooled.csv")
OUT_DIR = "OmegaFlux_Models"

TEST_FRACTION = 0.10
VAL_FRACTION = 0.10

TUNE_OBJECTIVE = "topology"
TUNE_HOLDOUT_LSHAPE_FRACTION = 0.25
FIXED_MAX_DEPTH = 6
FIXED_LEARNING_RATE = 0.15

FS_SUBSAMPLE = 200_000
FS_TREES = 200
FS_MAX_K = 12
FS_E2_WEIGHT = 0.6
FS_FULL_CHECKPOINTS = [4, 6, 8, 10, 12]

CV_FOLDS = 5
TUNE_SUBSAMPLE = 250_000
TUNE_ITER = 30
N_ESTIMATORS_MAX = 4000
EARLY_STOPPING_ROUNDS = 50
SHAP_SUBSAMPLE = 20_000
BENCH_SENSORS = 500
BENCH_REPEATS = 50

DEFAULT_PARAMS = dict(max_depth=9, min_child_weight=5, learning_rate=0.07,
                      subsample=0.75, colsample_bytree=0.8,
                      reg_alpha=0.5, reg_lambda=10.0)

# =============================================================================
# FEATURE SCHEMA
# =============================================================================
FEATURES = [
    "Total_Omega", "Global_Cos_Theta",
    "Global_Sill_Clearance", "Global_Header_Clearance", "Global_inv_dmin2",
    "Opaque_Wall_dmin", "Inner_Corner_Dmin", "Outer_Corner_Dmin",
    "South_Weighted_Omega", "East_Weighted_Omega", "West_Weighted_Omega",
    "Annual_Direct_Sun_Fraction", "Smooth_Mean_Depth", "Smooth_Depth_StdDev",
    "Min_Occluder_Distance", "Window_Angular_Spread", "Window_Sky_Weighted_Omega",
    "Occlusion_Multiplier", "Visible_Window_Count", "Mean_Window_Distance",
    "Dominant_Win_Plane_Dist", "Dominant_Win_H_Offset",
    "Dominant_Win_Ray_Forward", "Dominant_Win_Ray_Backward",
    "Dominant_Win_Ray_Left", "Dominant_Win_Ray_Right",
]

FEATURE_BLOCKS = {
    "solid_angle": ["Total_Omega", "Global_Cos_Theta", "Window_Sky_Weighted_Omega",
                    "Occlusion_Multiplier", "Window_Angular_Spread"],
    "distance": ["Global_inv_dmin2", "Opaque_Wall_dmin", "Mean_Window_Distance",
                 "Dominant_Win_Plane_Dist"],
    "plan_morphology": ["Inner_Corner_Dmin", "Outer_Corner_Dmin", "Smooth_Mean_Depth",
                        "Smooth_Depth_StdDev", "Min_Occluder_Distance",
                        "Visible_Window_Count"],
    "directional_solar": ["South_Weighted_Omega", "East_Weighted_Omega",
                          "West_Weighted_Omega", "Annual_Direct_Sun_Fraction"],
    "local_envelope": ["Global_Sill_Clearance", "Global_Header_Clearance",
                       "Dominant_Win_H_Offset", "Dominant_Win_Ray_Forward",
                       "Dominant_Win_Ray_Backward", "Dominant_Win_Ray_Left",
                       "Dominant_Win_Ray_Right"],
}

BASELINE_ROOM_PARAMS = [
    "Meta_BBox_W", "Meta_BBox_D", "Meta_Room_Height", "Meta_N_Windows",
    "Meta_Total_Win_Area", "Meta_WWR_proxy", "Meta_Orientation_deg",
    "Meta_Sensor_X_norm", "Meta_Sensor_Y_norm",
]

# =============================================================================
# HELPERS
# =============================================================================
def log(msg=""):
    print(msg, flush=True)

def banner(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)

def split_params(params):
    if isinstance(params, dict) and "Target_UDI_low" in params:
        return dict(params["Target_UDI_low"]), dict(params["Target_UDI_up"])
    return dict(params), dict(params)

def make_model(params, early_stop=True):
    p = dict(params)
    p.update(n_estimators=N_ESTIMATORS_MAX, random_state=SEED,
             n_jobs=-1, verbosity=0, tree_method="hist")
    if early_stop:
        p["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return XGBRegressor(**p)

def fit_pair(X_tr, y_tr_low, y_tr_up, X_val, y_val_low, y_val_up, params):
    p_low, p_up = split_params(params)
    m_low = make_model(p_low)
    m_up = make_model(p_up)
    m_low.fit(X_tr, y_tr_low, eval_set=[(X_val, y_val_low)], verbose=False)
    m_up.fit(X_tr, y_tr_up, eval_set=[(X_val, y_val_up)], verbose=False)
    return m_low, m_up

def predict_useful(m_low, m_up, X):
    p_low = np.clip(m_low.predict(X), 0.0, 100.0)
    p_up = np.clip(m_up.predict(X), 0.0, 100.0)
    return p_low, p_up, np.clip(100.0 - p_low - p_up, 0.0, 100.0)

def score_block(actual, pred):
    err = np.abs(actual - pred)
    return {
        "R2": float(r2_score(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "MAE": float(mean_absolute_error(actual, pred)),
        "P95": float(np.percentile(err, 95)),
        "N": int(len(actual)),
    }

def evaluate(name, m_low, m_up, X_test, y_low, y_up, groups_test):
    p_low, p_up, p_useful = predict_useful(m_low, m_up, X_test)
    a_useful = np.clip(100.0 - y_low - y_up, 0.0, 100.0)

    res = {"experiment": name}
    for tgt, a, p in (("low", y_low, p_low), ("up", y_up, p_up),
                      ("useful", a_useful, p_useful)):
        for k, v in score_block(np.asarray(a), np.asarray(p)).items():
            res[f"{tgt}_{k}"] = v

    dfr = pd.DataFrame({"g": groups_test, "a": a_useful, "p": p_useful})
    room = dfr.groupby("g", observed=True)[["a", "p"]].mean()
    if len(room) > 2:
        rho, _ = spearmanr(room["a"], room["p"])
        res["room_spearman_rho"] = float(rho)
        res["room_R2"] = float(r2_score(room["a"], room["p"]))
        res["room_MAE"] = float(mean_absolute_error(room["a"], room["p"]))
    res["n_test_rooms"] = int(len(room))
    return res

def inner_split(X, y_low, y_up, groups, frac, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    a, b = next(gss.split(X, y_low, groups=groups))
    return (X.iloc[a], y_low.iloc[a], y_up.iloc[a], groups.iloc[a],
            X.iloc[b], y_low.iloc[b], y_up.iloc[b], groups.iloc[b])

def train_and_eval(name, df_tr, df_te, features, params, seed=SEED):
    Xtr_all = df_tr[features].astype("float32")
    Xte = df_te[features].astype("float32")
    (X_tr, ylo_tr, yup_tr, _,
     X_va, ylo_va, yup_va, _) = inner_split(
        Xtr_all, df_tr["Target_UDI_low"], df_tr["Target_UDI_up"],
        df_tr["Group_ID"], VAL_FRACTION, seed)

    t0 = time.perf_counter()
    m_low, m_up = fit_pair(X_tr, ylo_tr, yup_tr, X_va, ylo_va, yup_va, params)
    train_s = time.perf_counter() - t0

    res = evaluate(name, m_low, m_up, Xte,
                   df_te["Target_UDI_low"].values, df_te["Target_UDI_up"].values,
                   df_te["Group_ID"].values)
    res.update({
        "n_features": len(features),
        "n_train_rooms": int(df_tr["Group_ID"].nunique()),
        "n_train_rows": int(len(df_tr)),
        "best_iter_low": int(getattr(m_low, "best_iteration", -1)),
        "best_iter_up": int(getattr(m_up, "best_iteration", -1)),
        "train_seconds": round(train_s, 1),
    })
    return res, m_low, m_up

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="OmegaFlux Stage 2: Training & Evaluation")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "full"], 
                        help="Run mode: 'fast' (no tuning) or 'full' (with hyperparameter tuning and extensive tests)")
    args = parser.parse_args()

    is_full = (args.mode == "full")

    RUN_TUNING = is_full
    RUN_CROSSVAL = is_full
    RUN_SIZE_SWEEP = is_full
    RUN_FEATURE_SELECTION = is_full
    
    RUN_MAIN = True
    RUN_TOPOLOGY_TESTS = True
    RUN_BASELINES = True
    RUN_SHAP = True
    RUN_FIGURES = True

    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    banner(f"OmegaFlux - Stage 2: training and evaluation (Mode: {args.mode.upper()})")

    if not os.path.exists(POOLED_CSV):
        log(f"Missing {POOLED_CSV}. Run feature extraction first.")
        return

    df = pd.read_csv(POOLED_CSV, dtype={"Iteration_Name": str, "Group_ID": str, "Source_Tag": str})
    log(f"Loaded {len(df):,} sensor records from {POOLED_CSV}")

    required = {"Group_ID", "Source_Tag", "Target_UDI_low", "Target_UDI_up"}
    missing = required - set(df.columns)
    if missing:
        log(f"[ERROR] missing columns: {missing}")
        return

    features = [f for f in FEATURES if f in df.columns]
    log(f"\nUsing {len(features)} independent features")

    banner("DATASET OVERVIEW")
    useful = np.clip(100.0 - df["Target_UDI_low"] - df["Target_UDI_up"], 0, 100)
    log(f"Sensor records        : {len(df):,}")
    log(f"Distinct rooms        : {df['Group_ID'].nunique():,}")
    log(f"Rooms per source      : {df.groupby('Source_Tag')['Group_ID'].nunique().to_dict()}")
    log(f"Mean useful UDI       : {useful.mean():.2f}%")
    log(f"Std dev useful UDI    : {useful.std():.2f}%")
    log(f"Median useful UDI     : {useful.median():.2f}%")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED)
    tr_idx, te_idx = next(gss.split(df, groups=df["Group_ID"]))
    df_train, df_test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()

    assert set(df_train["Group_ID"]).isdisjoint(set(df_test["Group_ID"])), "room leakage between train and test"

    banner("PARTITION (geometry-level, disjoint by construction)")
    log(f"Train: {len(df_train):,} records / {df_train['Group_ID'].nunique():,} rooms")
    log(f"Test : {len(df_test):,} records / {df_test['Group_ID'].nunique():,} rooms")

    tune_lshape_groups = set()
    if TUNE_OBJECTIVE == "topology" and RUN_TUNING:
        ls = np.array(sorted(df_train.loc[df_train["Source_Tag"] == "randomized", "Group_ID"].unique()))
        if len(ls):
            k = max(1, int(round(len(ls) * TUNE_HOLDOUT_LSHAPE_FRACTION)))
            tune_lshape_groups = set(np.random.default_rng(SEED).choice(ls, size=k, replace=False))
            log(f"\nReserved {len(tune_lshape_groups):,} of {len(ls):,} training L-shape rooms for tuning validation.")

    results = []
    params = dict(DEFAULT_PARAMS)

    # --- Hyperparameter Search ---
    if RUN_TUNING:
        banner("HYPERPARAMETER SEARCH (GroupKFold)")
        if TUNE_OBJECTIVE == "topology":
            conv = df_train[df_train["Source_Tag"].isin(["rect", "rect_rotated"])]
            noncv = df_train[df_train["Group_ID"].isin(tune_lshape_groups)]
            n_a = min(int(TUNE_SUBSAMPLE * 0.75), len(conv))
            n_b = min(TUNE_SUBSAMPLE - n_a, len(noncv))
            sub = pd.concat([conv.sample(n_a, random_state=SEED), noncv.sample(n_b, random_state=SEED)])
            tr_i = np.where(sub["Source_Tag"].isin(["rect", "rect_rotated"]).values)[0]
            va_i = np.where((sub["Source_Tag"] == "randomized").values)[0]
            cv_obj = [(tr_i, va_i)]
        else:
            sub = df_train.sample(n=min(TUNE_SUBSAMPLE, len(df_train)), random_state=SEED)
            cv_obj = GroupKFold(n_splits=3)

        X_t = sub[features].astype("float32")
        g_t = sub["Group_ID"]

        grid = {
            "min_child_weight": [20, 50, 100, 200, 400],
            "subsample": [0.4, 0.5, 0.6, 0.7],
            "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7],
            "reg_alpha": [0.5, 1.0, 5.0, 20.0, 50.0],
            "reg_lambda": [1.0, 5.0, 10.0, 50.0, 100.0, 200.0],
        }
        
        tuned = {}
        for tgt in ("Target_UDI_low", "Target_UDI_up"):
            search = RandomizedSearchCV(
                XGBRegressor(n_estimators=900, max_depth=FIXED_MAX_DEPTH,
                             learning_rate=FIXED_LEARNING_RATE, random_state=SEED, 
                             n_jobs=-1, verbosity=0, tree_method="hist"),
                param_distributions=grid, n_iter=TUNE_ITER,
                scoring="neg_root_mean_squared_error",
                cv=cv_obj, random_state=SEED, n_jobs=-1, verbose=0)
            search.fit(X_t, sub[tgt], groups=g_t)
            tuned[tgt] = dict(search.best_params_)
            tuned[tgt].update(max_depth=FIXED_MAX_DEPTH, learning_rate=FIXED_LEARNING_RATE)
            log(f"\n{tgt}: RMSE {-search.best_score_:.4f}")
            for k, v in sorted(search.best_params_.items()):
                log(f"  {k:20s} {v}")
        params = tuned
        with open(f"{OUT_DIR}/hyperparameters.json", "w") as fh:
            json.dump({"objective": TUNE_OBJECTIVE, "selected": tuned}, fh, indent=2)

    # --- E1: Main Model ---
    m_low = m_up = None
    if RUN_MAIN:
        banner("E1 - MAIN MODEL")
        res, m_low, m_up = train_and_eval("E1_main", df_train, df_test, features, params)
        results.append(res)
        log(f"useful : R2 {res['useful_R2']:.4f} | RMSE {res['useful_RMSE']:.2f}% | MAE {res['useful_MAE']:.2f}%")
        log(f"low    : R2 {res['low_R2']:.4f} | RMSE {res['low_RMSE']:.2f}%")
        log(f"up     : R2 {res['up_R2']:.4f} | RMSE {res['up_RMSE']:.2f}%")

        joblib.dump(m_low, f"{OUT_DIR}/omegaflux_xgb_low.joblib")
        joblib.dump(m_up, f"{OUT_DIR}/omegaflux_xgb_up.joblib")
        m_low.save_model(f"{OUT_DIR}/omegaflux_xgb_low.json")
        m_up.save_model(f"{OUT_DIR}/omegaflux_xgb_up.json")

        mb_low = os.path.getsize(f"{OUT_DIR}/omegaflux_xgb_low.json") / 1024 ** 2
        mb_up = os.path.getsize(f"{OUT_DIR}/omegaflux_xgb_up.json") / 1024 ** 2
        log(f"\nModel size: low {mb_low:.1f} MB, up {mb_up:.1f} MB, combined {mb_low + mb_up:.1f} MB")
        
        res["model_mb_low"] = round(mb_low, 2)
        res["model_mb_up"] = round(mb_up, 2)

        log("\nBreakdown by geometry class (held-out rooms):")
        Xte = df_test[features].astype("float32")
        _, _, p_use = predict_useful(m_low, m_up, Xte)
        a_use = np.clip(100.0 - df_test["Target_UDI_low"] - df_test["Target_UDI_up"], 0, 100).values
        for tag in sorted(df_test["Source_Tag"].unique()):
            m = (df_test["Source_Tag"] == tag).values
            s = score_block(a_use[m], p_use[m])
            log(f"  {tag:16s} R2 {s['R2']:6.4f} | RMSE {s['RMSE']:5.2f}% | N={s['N']:,}")
            results.append({"experiment": f"E1_subset_{tag}", **{f"useful_{k}": v for k, v in s.items()}})

    # --- Repeated Grouped CV ---
    if RUN_CROSSVAL:
        banner(f"REPEATED {CV_FOLDS}-FOLD GroupKFold")
        gkf = GroupKFold(n_splits=CV_FOLDS)
        fold_scores = []
        for k, (a, b) in enumerate(gkf.split(df, groups=df["Group_ID"]), 1):
            d_tr, d_te = df.iloc[a], df.iloc[b]
            r, _, _ = train_and_eval(f"CV_fold{k}", d_tr, d_te, features, params, seed=SEED + k)
            fold_scores.append(r)
            log(f"  fold {k}: R2 {r['useful_R2']:.4f} | RMSE {r['useful_RMSE']:.2f}%")
            
        arr = pd.DataFrame(fold_scores)
        log(f"\n  R2   = {arr['useful_R2'].mean():.4f} +/- {arr['useful_R2'].std():.4f}")
        results.extend(fold_scores)
        results.append({
            "experiment": "CV_summary",
            "useful_R2": float(arr["useful_R2"].mean()),
            "useful_RMSE": float(arr["useful_RMSE"].mean())
        })

    # --- E2/E3/E4: Leave-one-topology-out ---
    if RUN_TOPOLOGY_TESTS:
        banner("E2-E4 - ZERO-SHOT GENERALISATION")
        tags = set(df["Source_Tag"].unique())
        experiments = []
        if {"rect", "rect_rotated", "randomized"} <= tags:
            experiments.append(("E2_holdout_Lshapes", ["rect", "rect_rotated"], ["randomized"], "train rectangles -> predict L-shapes"))
            experiments.append(("E3_holdout_rotated", ["rect", "randomized"], ["rect_rotated"], "train axis-aligned -> predict rotated"))

        for name, tr_tags, te_tags, desc in experiments:
            d_tr = df[df["Source_Tag"].isin(tr_tags)]
            d_te = df[df["Source_Tag"].isin(te_tags)]
            if name.startswith("E2") and tune_lshape_groups:
                d_te = d_te[~d_te["Group_ID"].isin(tune_lshape_groups)]
            if len(d_tr) == 0 or len(d_te) == 0:
                continue
            r, _, _ = train_and_eval(name, d_tr, d_te, features, params)
            results.append(r)
            log(f"{name}  ({desc}) -> R2 {r['useful_R2']:.4f} | RMSE {r['useful_RMSE']:.2f}%")

        if {"Meta_BBox_W", "Meta_Room_Height"} <= set(df.columns):
            H = df["Meta_Room_Height"]
            tmp = df.assign(_wh=(df["Global_Sill_Clearance"] + df["Global_Header_Clearance"]) * H, _sill=0.8 - df["Global_Sill_Clearance"] * H)
            room = tmp.groupby("Group_ID").agg(tag=("Source_Tag", "first"), w=("Meta_BBox_W", "first"), d=("Meta_BBox_D", "first"), wh=("_wh", "median"), sill=("_sill", "median"))
            inside = room.index[(room.tag == "randomized") & room.w.between(4.0, 6.0) & room.d.between(4.0, 6.0) & (room.wh <= 1.85) & (room.sill >= 0.35)]
            inside = set(inside) - tune_lshape_groups
            d_te = df[df["Group_ID"].isin(inside)]
            if d_te["Group_ID"].nunique() >= 15:
                d_tr = df[df["Source_Tag"].isin(["rect", "rect_rotated"])]
                r, _, _ = train_and_eval("E2c_controlled_envelope", d_tr, d_te, features, params)
                results.append(r)
                log(f"E2c_controlled_envelope -> R2 {r['useful_R2']:.4f} | RMSE {r['useful_RMSE']:.2f}%")

        if "Meta_N_Windows" in df.columns:
            lo_n = df[df["Meta_N_Windows"] <= 2]
            hi_n = df[df["Meta_N_Windows"] >= 3]
            if len(lo_n) > 0 and len(hi_n) > 0:
                r, _, _ = train_and_eval("E4_unseen_aperture_count", lo_n, hi_n, features, params)
                results.append(r)
                log(f"E4_unseen_aperture_count -> R2 {r['useful_R2']:.4f} | RMSE {r['useful_RMSE']:.2f}%")

    # --- Baselines & Ablations ---
    if RUN_BASELINES:
        banner("BASELINES AND ABLATIONS")
        a_use = np.clip(100.0 - df_test["Target_UDI_low"] - df_test["Target_UDI_up"], 0, 100).values

        tr_mean = np.clip(100.0 - df_train["Target_UDI_low"] - df_train["Target_UDI_up"], 0, 100).mean()
        s = score_block(a_use, np.full_like(a_use, tr_mean))
        results.append({"experiment": "B0_mean_predictor", **{f"useful_{k}": v for k, v in s.items()}})
        log(f"B0 mean predictor        R2 {s['R2']:7.4f} | RMSE {s['RMSE']:5.2f}%")

        b1 = [c for c in BASELINE_ROOM_PARAMS if c in df.columns]
        if b1:
            r, _, _ = train_and_eval("B1_room_parameters", df_train, df_test, b1, params)
            results.append(r)
            log(f"B1 room parameters       R2 {r['useful_R2']:7.4f} | RMSE {r['useful_RMSE']:5.2f}%")

        r, _, _ = train_and_eval("B2_total_omega_only", df_train, df_test, ["Total_Omega"], params)
        results.append(r)
        log(f"B2 Total_Omega only      R2 {r['useful_R2']:7.4f} | RMSE {r['useful_RMSE']:5.2f}%")

        for block, cols in FEATURE_BLOCKS.items():
            cols = [c for c in cols if c in features]
            if cols:
                r, _, _ = train_and_eval(f"B3_block_{block}", df_train, df_test, cols, params)
                results.append(r)
                log(f"B3 block {block:18s} R2 {r['useful_R2']:7.4f} | RMSE {r['useful_RMSE']:5.2f}%")

        for block, cols in FEATURE_BLOCKS.items():
            rem = [f for f in features if f not in cols]
            if len(rem) >= 2:
                r, _, _ = train_and_eval(f"B3_without_{block}", df_train, df_test, rem, params)
                results.append(r)
                log(f"B3 without {block:16s} R2 {r['useful_R2']:7.4f} | RMSE {r['useful_RMSE']:5.2f}%")

        Xtr_all = df_train[features].astype("float32")
        (X_tr, ylo, _, _, X_va, ylo_va, _, _) = inner_split(
            Xtr_all, np.clip(100 - df_train["Target_UDI_low"] - df_train["Target_UDI_up"], 0, 100),
            df_train["Target_UDI_up"], df_train["Group_ID"], VAL_FRACTION, SEED)
        
        m_direct = make_model(params)
        m_direct.fit(X_tr, ylo, eval_set=[(X_va, ylo_va)], verbose=False)
        p_direct = np.clip(m_direct.predict(df_test[features].astype("float32")), 0, 100)
        s = score_block(a_use, p_direct)
        results.append({"experiment": "B5_direct_useful_regression", **{f"useful_{k}": v for k, v in s.items()}})
        log(f"B5 direct UDI_useful     R2 {s['R2']:7.4f} | RMSE {s['RMSE']:5.2f}%")

    # --- Feature Selection ---
    if RUN_FEATURE_SELECTION:
        banner("FEATURE SELECTION (greedy forward)")
        gss_fs = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        a_i, b_i = next(gss_fs.split(df_train, groups=df_train["Group_ID"]))
        fs_tr = df_train.iloc[a_i]
        fs_va_e1 = df_train.iloc[b_i]
        if len(fs_tr) > FS_SUBSAMPLE: fs_tr = fs_tr.sample(FS_SUBSAMPLE, random_state=SEED)
        if len(fs_va_e1) > 60_000: fs_va_e1 = fs_va_e1.sample(60_000, random_state=SEED)

        conv_fs = df_train[df_train["Source_Tag"].isin(["rect", "rect_rotated"])]
        if len(conv_fs) > FS_SUBSAMPLE: conv_fs = conv_fs.sample(FS_SUBSAMPLE, random_state=SEED)
        ls_fs = df_train[df_train["Group_ID"].isin(tune_lshape_groups)]
        if len(ls_fs) > 60_000: ls_fs = ls_fs.sample(60_000, random_state=SEED)
        have_e2 = len(ls_fs) > 5_000

        p_low_fs, _ = split_params(params)
        p_low_fs = dict(p_low_fs)
        p_low_fs.update(n_estimators=FS_TREES, random_state=SEED, n_jobs=-1, verbosity=0, tree_method="hist")

        def fs_score(cols):
            m = XGBRegressor(**p_low_fs)
            m.fit(fs_tr[cols].astype("float32"), fs_tr["Target_UDI_low"], verbose=False)
            r1 = r2_score(fs_va_e1["Target_UDI_low"], m.predict(fs_va_e1[cols].astype("float32")))
            r2v = np.nan
            if have_e2:
                m2 = XGBRegressor(**p_low_fs)
                m2.fit(conv_fs[cols].astype("float32"), conv_fs["Target_UDI_low"], verbose=False)
                r2v = r2_score(ls_fs["Target_UDI_low"], m2.predict(ls_fs[cols].astype("float32")))
            comb = r1 if not have_e2 else ((1 - FS_E2_WEIGHT) * r1 + FS_E2_WEIGHT * r2v)
            return comb, r1, r2v

        chosen, remaining, fs_rows = [], list(features), []
        t_fs = time.perf_counter()
        
        while len(chosen) < min(FS_MAX_K, len(features)):
            best = None
            for f in remaining:
                comb, r1, r2v = fs_score(chosen + [f])
                if best is None or comb > best[0]:
                    best = (comb, r1, r2v, f)
            comb, r1, r2v, f = best
            chosen.append(f); remaining.remove(f)
            fs_rows.append({"k": len(chosen), "added": f, "objective": comb, "subset": ";".join(chosen)})
            log(f"{len(chosen):2d}  {f:30s} {comb:8.4f}")
            
        pd.DataFrame(fs_rows).to_csv(f"{OUT_DIR}/feature_selection_path.csv", index=False)

        d_tr_e2 = df[df["Source_Tag"].isin(["rect", "rect_rotated"])]
        d_te_e2 = df[(df["Source_Tag"] == "randomized") & (~df["Group_ID"].isin(tune_lshape_groups))]
        tmp = os.path.join(OUT_DIR, "_fs_tmp.json")
        for k in sorted(set(FS_FULL_CHECKPOINTS + [len(features)])):
            cols = chosen[:k] if k <= len(chosen) else features
            if k > len(chosen) and k != len(features): continue
            r, sl, su = train_and_eval(f"FS_k{k}", df_train, df_test, cols, params)
            sl.save_model(tmp); mb = os.path.getsize(tmp) / 1024 ** 2
            su.save_model(tmp); mb += os.path.getsize(tmp) / 1024 ** 2
            r["model_mb_total"] = round(mb, 2)
            if len(d_te_e2):
                rr, _, _ = train_and_eval(f"FS_k{k}_E2", d_tr_e2, d_te_e2, cols, params)
                r["useful_R2_E2"] = rr["useful_R2"]
            results.append(r)
            del sl, su
        if os.path.exists(tmp):
            os.remove(tmp)

    # --- Size Sweep ---
    if RUN_SIZE_SWEEP:
        banner("MODEL SIZE vs ACCURACY")
        sweep = [
            dict(max_depth=4,  learning_rate=0.20),
            dict(max_depth=5,  learning_rate=0.15),
            dict(max_depth=6,  learning_rate=0.15),
            dict(max_depth=8,  learning_rate=0.10),
            dict(max_depth=10, learning_rate=0.07),
        ]
        d_tr_e2 = df[df["Source_Tag"].isin(["rect", "rect_rotated"])]
        d_te_e2 = df[(df["Source_Tag"] == "randomized") & (~df["Group_ID"].isin(tune_lshape_groups))]
        tmp = os.path.join(OUT_DIR, "_sweep_tmp.json")
        for cfg in sweep:
            pl, pu = split_params(params)
            pl.update(cfg); pu.update(cfg)
            p = {"Target_UDI_low": pl, "Target_UDI_up": pu}
            tag = f"SIZE_d{cfg['max_depth']}_lr{cfg['learning_rate']}"
            r, sl, su = train_and_eval(tag, df_train, df_test, features, p)
            sl.save_model(tmp)
            mb = os.path.getsize(tmp) / 1024 ** 2
            su.save_model(tmp)
            mb += os.path.getsize(tmp) / 1024 ** 2
            r["model_mb_total"] = round(mb, 2)
            if len(d_te_e2):
                r2, _, _ = train_and_eval(tag + "_E2", d_tr_e2, d_te_e2, features, p)
                r["useful_R2_E2"] = r2["useful_R2"]
            results.append(r)
            log(f"depth {cfg['max_depth']:2d}, lr {cfg['learning_rate']:.2f} | R2 {r['useful_R2']:8.4f}")
            del sl, su
        if os.path.exists(tmp):
            os.remove(tmp)

    # --- TreeSHAP ---
    shap_summary = {}
    if RUN_SHAP and m_low is not None:
        banner("INTERPRETABILITY - TreeSHAP")
        import xgboost as xgb
        Xs = df_test[features].astype("float32").sample(min(SHAP_SUBSAMPLE, len(df_test)), random_state=SEED)
        for label, model in (("UDI_low", m_low), ("UDI_up", m_up)):
            bst = model.get_booster()
            best = getattr(model, "best_iteration", None)
            rng_it = (0, int(best) + 1) if best is not None else (0, bst.num_boosted_rounds())
            parts = []
            for i in range(0, len(Xs), 2000):
                parts.append(bst.predict(xgb.DMatrix(Xs.iloc[i:i + 2000]), pred_contribs=True, iteration_range=rng_it))
            contrib = np.vstack(parts)
            sv, base = contrib[:, :-1], contrib[:, -1]
            imp = np.abs(sv).mean(axis=0)
            order = np.argsort(imp)[::-1]
            shap_summary[label] = {features[i]: float(imp[i]) for i in order}
            
        with open(f"{OUT_DIR}/shap_importance.json", "w") as fh:
            json.dump(shap_summary, fh, indent=2)

    # --- Latency ---
    bench = {}
    if m_low is not None:
        banner("LATENCY")
        Xb = df_test[features].astype("float32").iloc[:BENCH_SENSORS]
        m_low.predict(Xb); m_up.predict(Xb)
        times = []
        for _ in range(BENCH_REPEATS):
            t0 = time.perf_counter()
            m_low.predict(Xb); m_up.predict(Xb)
            times.append((time.perf_counter() - t0) * 1000)
        times = np.array(times)
        bench = {"sensors": int(len(Xb)), "inference_ms_mean": float(times.mean())}
        log(f"Inference: {times.mean():.2f} +/- {times.std():.2f} ms ({times.mean() / len(Xb) * 1000:.1f} us/sensor)")

    # --- Figures ---
    if RUN_FIGURES and m_low is not None:
        Xte = df_test[features].astype("float32")
        _, _, p_use = predict_useful(m_low, m_up, Xte)
        a_use = np.clip(100.0 - df_test["Target_UDI_low"] - df_test["Target_UDI_up"], 0, 100).values
        err = p_use - a_use

        plt.figure(figsize=(7, 7))
        hb = plt.hexbin(a_use, p_use, gridsize=50, cmap="inferno", mincnt=1, bins="log")
        plt.plot([0, 100], [0, 100], "w--", lw=2, label="1:1")
        plt.colorbar(hb, label="Sensors (log)")
        plt.xlabel("Simulated UDI$_{useful}$ (% of occupied hours)")
        plt.ylabel("Predicted UDI$_{useful}$ (% of occupied hours)")
        plt.title("Predicted against simulated useful fraction")
        plt.legend(loc="upper left"); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/scatter_density.png", dpi=300); plt.close()

        plt.figure(figsize=(7, 5))
        plt.hist(err, bins=100, range=(-15, 15), color="#e53e3e", edgecolor="k", lw=.4)
        plt.axvline(0, color="k", ls="--", lw=2)
        plt.xlabel("Prediction error (percentage points)")
        plt.ylabel("Sensor count"); plt.title("Error distribution")
        plt.grid(axis="y", alpha=.4); plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/error_distribution.png", dpi=300); plt.close()

        plt.figure(figsize=(7, 5))
        hb2 = plt.hexbin(Xte["Mean_Window_Distance"], np.abs(err), gridsize=40, cmap="viridis", mincnt=1, bins="log")
        plt.colorbar(hb2, label="Sensors (log)")
        plt.xlabel("Mean window distance / $L_{char}$ (dimensionless)")
        plt.ylabel("Absolute error (percentage points)")
        plt.title("Error against normalised room depth")
        plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/error_vs_depth.png", dpi=300); plt.close()

        if "Inner_Corner_Dmin" in Xte.columns:
            plt.figure(figsize=(7, 5))
            hb3 = plt.hexbin(Xte["Inner_Corner_Dmin"], np.abs(err), gridsize=40, cmap="magma", mincnt=1, bins="log")
            plt.colorbar(hb3, label="Sensors (log)")
            plt.xlabel("Distance to nearest re-entrant corner / $L_{char}$")
            plt.ylabel("Absolute error (percentage points)")
            plt.title("Error against normalised distance to the nearest re-entrant corner")
            plt.grid(alpha=.3); plt.tight_layout()
            plt.savefig(f"{OUT_DIR}/error_vs_inner_corner.png", dpi=300); plt.close()

    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{OUT_DIR}/results_table.csv", index=False)
    
    with open(f"{OUT_DIR}/manifest_training.json", "w") as fh:
        json.dump({
            "seed": SEED, "n_features": len(features), "features": features,
            "params": params, "benchmark": bench,
            "test_fraction": TEST_FRACTION, "val_fraction": VAL_FRACTION,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "early_stopping_uses_test_set": False,
            "environment": {"python": platform.python_version(),
                            "numpy": np.__version__, "pandas": pd.__version__,
                            "platform": platform.platform(),
                            "cpu_count": os.cpu_count()},
        }, fh, indent=2)

    banner("DONE")
    log(f"Results table -> {OUT_DIR}/results_table.csv")
    log(f"Manifest      -> {OUT_DIR}/manifest_training.json")


if __name__ == "__main__":
    main()