import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
DATA_DIR = "data_processed"
OUT_DIR = "OmegaFlux_Models"
FIG_DIR = os.path.join(OUT_DIR, "figures_final")
POOLED_CSV = os.path.join(DATA_DIR, "omegaflux_sensors_pooled.csv")
PRED_CACHE = os.path.join(OUT_DIR, "test_predictions.npz")
SHAP_NPZ = os.path.join(OUT_DIR, "shap_values.npz")
TEST_FRACTION = 0.10
USE_UNDERSCORES = True
FORCE_REPREDICT = False

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

CLASS_LABEL = {
    "rect": "Rectangular", 
    "rect_rotated": "Rotated",
    "randomized": "L-shaped"
}

def lab(f):
    return f if USE_UNDERSCORES else f.replace("_", " ")

def build_prediction_cache():
    print(f"Loading {POOLED_CSV} ...")
    df = pd.read_csv(POOLED_CSV, dtype={"Iteration_Name": str, "Group_ID": str, "Source_Tag": str})
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED).split(df, groups=df["Group_ID"]))
    d_te = df.iloc[te]
    print(f"Test partition: {len(d_te):,} grid points / {d_te['Group_ID'].nunique():,} rooms")
    X = d_te[FEATURES].astype("float32")
    preds = {}
    for name in ("low", "up"):
        m = xgb.XGBRegressor()
        m.load_model(f"{OUT_DIR}/omegaflux_xgb_{name}.json")
        best = getattr(m, "best_iteration", None)
        n_tot = m.get_booster().num_boosted_rounds()
        print(f"  {name}: predicting with {int(best)+1 if best is not None else n_tot} of {n_tot} trees")
        preds[name] = np.clip(m.predict(X), 0.0, 100.0)
        del m
    a_low = d_te["Target_UDI_low"].to_numpy()
    a_up = d_te["Target_UDI_up"].to_numpy()
    np.savez_compressed(
        PRED_CACHE,
        a_useful=np.clip(100.0 - a_low - a_up, 0, 100),
        p_useful=np.clip(100.0 - preds["low"] - preds["up"], 0, 100),
        a_low=a_low, a_up=a_up, p_low=preds["low"], p_up=preds["up"],
        mean_window_distance=d_te["Mean_Window_Distance"].to_numpy(),
        inner_corner_dmin=d_te["Inner_Corner_Dmin"].to_numpy(),
        source_tag=d_te["Source_Tag"].to_numpy().astype(str),
        group_id=d_te["Group_ID"].to_numpy().astype(str),
    )
    print(f"Cached -> {PRED_CACHE}")

def plot_beeswarm(ax, sv, Xs, title, max_display=16):
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:max_display]
    rs = np.random.default_rng(SEED)
    for row, jj in enumerate(order[::-1]):
        v = sv[:, jj]
        xv = Xs.iloc[:, jj].to_numpy()
        lo, hi = np.percentile(xv, [1, 99])
        c = np.clip((xv - lo) / (hi - lo + 1e-12), 0, 1)
        sc = ax.scatter(v, np.full(len(v), row) + rs.normal(0, 0.10, len(v)), c=c, cmap="coolwarm", s=3, alpha=0.45, linewidths=0, rasterized=True)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([lab(FEATURES[k]) for k in order[::-1]], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Attribution (percentage points)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    return sc

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10})
    if FORCE_REPREDICT or not os.path.exists(PRED_CACHE):
        build_prediction_cache()
    else:
        print(f"Reusing {PRED_CACHE} (set FORCE_REPREDICT = True to rebuild)")
    C = np.load(PRED_CACHE, allow_pickle=False)
    a_use, p_use = C["a_useful"], C["p_useful"]
    err = p_use - a_use
    abs_err = np.abs(err)
    print(f"Loaded {len(a_use):,} test predictions | RMSE {np.sqrt((err**2).mean()):.2f} pp, MAE {abs_err.mean():.2f} pp")
    
    fig, ax = plt.subplots(figsize=(7, 7))
    hb = ax.hexbin(a_use, p_use, gridsize=50, cmap="inferno", mincnt=1, bins="log")
    ax.plot([0, 100], [0, 100], "w--", lw=2, label="1:1")
    fig.colorbar(hb, ax=ax, label="Grid points (log)")
    ax.set_xlabel("Simulated UDI$_{useful}$ (% of occupied hours)")
    ax.set_ylabel("Predicted UDI$_{useful}$ (% of occupied hours)")
    ax.set_title("Predicted against simulated useful fraction, held-out rooms")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/scatter_density.png")
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(err, bins=100, range=(-15, 15), color="#e53e3e", edgecolor="k", lw=0.4)
    ax.axvline(0, color="k", ls="--", lw=2)
    ax.set_xlabel("Prediction error (percentage points)")
    ax.set_ylabel("Grid-point count")
    ax.set_title("Distribution of prediction error, held-out rooms")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/error_distribution.png")
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(C["mean_window_distance"], abs_err, gridsize=40, cmap="viridis", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="Grid points (log)")
    ax.set_xlabel(f"{lab('Mean_Window_Distance')} (dimensionless)")
    ax.set_ylabel("Absolute error (percentage points)")
    ax.set_title("Error against normalised aperture distance")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/error_vs_mean_window_distance.png")
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(C["inner_corner_dmin"], abs_err, gridsize=40, cmap="magma", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="Grid points (log)")
    ax.set_xlabel(f"{lab('Inner_Corner_Dmin')} (dimensionless)")
    ax.set_ylabel("Absolute error (percentage points)")
    ax.set_title("Error against normalised distance to the nearest re-entrant corner")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/error_vs_inner_corner.png")
    plt.close(fig)
    
    print("Prediction figures generated.")
    
    if not os.path.exists(SHAP_NPZ):
        print(f"\n{SHAP_NPZ} not found.")
        raise SystemExit(0)
    S = np.load(SHAP_NPZ, allow_pickle=True)
    shap_low, shap_up = S["shap_low"], S["shap_up"]
    shap_useful = S["shap_useful"]
    Xs = pd.DataFrame(S["X"], columns=[str(f) for f in S["features"]])
    tags = S["tags"].astype(str)
    print(f"\nLoaded cached attribution: {shap_low.shape[0]:,} points, {shap_low.shape[1]} descriptors")
    
    KEY = "Window_Sky_Weighted_Omega"
    j = FEATURES.index(KEY)
    x, y = Xs[KEY].to_numpy(), shap_low[:, j]
    edges = np.linspace(0, np.percentile(x, 99.5), 61)
    mid = 0.5 * (edges[:-1] + edges[1:])
    binned = np.array([y[(x >= a) & (x < b)].mean() if ((x >= a) & (x < b)).sum() >= 20 else np.nan for a, b in zip(edges[:-1], edges[1:])])
    g = np.isfinite(binned)
    mid_g, bin_g = mid[g], binned[g]
    cross = np.nan
    s = np.where(np.diff(np.sign(bin_g)) != 0)[0]
    if len(s):
        i0 = s[0]
        x0, x1, y0, y1 = mid_g[i0], mid_g[i0 + 1], bin_g[i0], bin_g[i0 + 1]
        cross = x0 - y0 * (x1 - x0) / (y1 - y0)
    asym = bin_g[mid_g > np.percentile(mid_g, 70)].mean()
    below = np.where(bin_g <= 0.90 * asym)[0]
    sat = mid_g[below[0]] if len(below) else np.nan
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=2, alpha=0.12, c="#4a5568", linewidths=0, rasterized=True)
    ax.plot(mid_g, bin_g, color="#c53030", lw=2.4, label="binned mean")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    if np.isfinite(cross):
        ax.axvline(cross, color="#2b6cb0", lw=1.2, ls=":")
        ax.annotate(f"zero crossing\n{cross:.3f}", (cross, ax.get_ylim()[1] * 0.55), xytext=(12, 0), textcoords="offset points", color="#2b6cb0", fontsize=9)
    if np.isfinite(sat):
        ax.axvline(sat, color="#2f855a", lw=1.2, ls=":")
        ax.annotate(f"90% of asymptote\n{sat:.3f}", (sat, ax.get_ylim()[1] * 0.30), xytext=(12, 0), textcoords="offset points", color="#2f855a", fontsize=9)
    ax.set_xlabel(f"{lab(KEY)} (fraction of the hemisphere)")
    ax.set_ylabel("Attribution to UDI$_{low}$ (percentage points)")
    ax.set_title("Attributed response of under-illumination to sky-weighted subtense")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/dose_response.png")
    plt.close(fig)
    print(f"  dose-response: crossing {cross:.3f}, asymptote {asym:+.1f} pp, 90% at {sat:.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    sc = plot_beeswarm(axes[0], shap_low, Xs, "Under-illuminated fraction")
    plot_beeswarm(axes[1], shap_up, Xs, "Over-illuminated fraction")
    cb = fig.colorbar(sc, ax=axes, pad=0.02, fraction=0.03)
    cb.set_label("Descriptor value (low → high)")
    fig.savefig(f"{FIG_DIR}/beeswarm.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  beeswarm: attribution spans [{shap_low.min():.0f}, {shap_low.max():.0f}] pp for UDI_low, [{shap_up.min():.0f}, {shap_up.max():.0f}] pp for UDI_up")
    
    per = {}
    for t in np.unique(tags):
        a = np.abs(shap_useful[tags == t]).mean(axis=0)
        per[CLASS_LABEL.get(t, t)] = a / a.sum() * 100.0
    cls = pd.DataFrame(per, index=[lab(f) for f in FEATURES])
    cls["_r"] = cls.mean(axis=1)
    cls = cls.sort_values("_r", ascending=False).drop(columns="_r")
    cls.round(4).to_csv(f"{OUT_DIR}/shap_by_class.csv")
    sub = cls.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    yy = np.arange(len(sub))
    w = 0.8 / len(sub.columns)
    for k, col in enumerate(sub.columns):
        ax.barh(yy + (k - (len(sub.columns) - 1) / 2) * w, sub[col], height=w, label=col)
    ax.set_yticks(yy)
    ax.set_yticklabels(sub.index, fontsize=9)
    ax.set_xlabel("Share of total attribution (%)")
    ax.set_title("Attribution shift across plan classes")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/shap_by_class.png")
    plt.close(fig)
    print("Attribution figures generated.")
    print(f"\nAll figures -> {FIG_DIR}/")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  {f}")

if __name__ == "__main__":
    main()