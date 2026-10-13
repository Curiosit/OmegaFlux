import gc
import argparse
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
DATA_DIR = "data/processed"   # same location train.py uses
OUT_DIR = "OmegaFlux_Models/core"   # set by --out
FIG_DIR = os.path.join(OUT_DIR, "figures_shap")
POOLED_CSV = os.path.join(DATA_DIR, "omegaflux_sensors_pooled.csv")
TEST_FRACTION = 0.10
N_SHAP = 20_000
CHUNK = 2_000

from omegaflux.features import FEATURES_ALL as FEATURES   # replaced from features.json in configure()

LABEL = {f: f for f in FEATURES}
CLASS_LABEL = {
    "rect": "Rectangular", 
    "rect_rotated": "Rotated",
    "randomized": "L-shaped"
}

def tree_shap(model_path, X, label):
    m = xgb.XGBRegressor()
    m.load_model(model_path)
    bst = m.get_booster()
    n_total = bst.num_boosted_rounds()
    best = getattr(m, "best_iteration", None)
    rng_it = (0, int(best) + 1) if best is not None and np.isfinite(best) else (0, n_total)
    print(f"[{label}] booster holds {n_total} trees; attributing over {rng_it[1]} (best_iteration = {best})")
    out, t0 = [], time.perf_counter()
    for i in range(0, len(X), CHUNK):
        out.append(bst.predict(xgb.DMatrix(X.iloc[i:i + CHUNK]), pred_contribs=True, iteration_range=rng_it))
        done = min(i + CHUNK, len(X))
        el = time.perf_counter() - t0
        print(f"\r[{label}] {done:,}/{len(X):,} | {el:6.1f} s | {el / done * 1e3:5.2f} ms/point", end="")
    print()
    contrib = np.vstack(out)
    sv, base = contrib[:, :-1], contrib[:, -1]
    err = float(np.abs(sv.sum(axis=1) + base - m.predict(X)).max())
    print(f"[{label}] additivity: max |Σφ + base − predict| = {err:.2e} pp")
    assert err < 1e-2, "additivity failed"
    del m, bst
    gc.collect()
    return sv, float(base[0])

def check(table, title, imp, sign):
    print(f"\n{title}")
    print(f"{'descriptor':32s}{'within':>9s}{'pooled':>9s}{'exp':>5s}  rooms  verdict")
    ok_n = n = 0
    for f in imp["descriptor"]:
        if f not in table:
            continue
        r_w = sign.loc[f, "rho_within_mean"]
        r_p = sign.loc[f, "rho_pooled"]
        if not np.isfinite(r_w):
            continue
        exp = table[f]
        n += 1
        ok = np.sign(r_w) == exp
        ok_n += ok
        flag = "" if np.sign(r_p) == np.sign(r_w) else "  (pooled inverts)"
        print(f"{LABEL[f]:32s}{r_w:+9.3f}{r_p:+9.3f}{exp:+5d}{int(sign.loc[f,'n_rooms_used']):7d}  {'OK' if ok else '** MISMATCH **'}{flag}")
    return ok_n, n

def beeswarm(sv, X, title, path, max_display=16):
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:max_display]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(order) + 1.8))
    rs = np.random.default_rng(SEED)
    for row, jj in enumerate(order[::-1]):
        v = sv[:, jj]
        xv = X.iloc[:, jj].to_numpy()
        lo, hi = np.percentile(xv, [1, 99])
        c = np.clip((xv - lo) / (hi - lo + 1e-12), 0, 1)
        sc = ax.scatter(v, np.full(len(v), row) + rs.normal(0, 0.10, len(v)), c=c, cmap="coolwarm", s=3, alpha=0.45, linewidths=0, rasterized=True)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([LABEL[FEATURES[k]] for k in order[::-1]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Attribution (percentage points)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Descriptor value (low → high)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def configure(out_dir, data_dir):
    """Point the module at one model folder and read its feature list from features.json."""
    global OUT_DIR, FIG_DIR, POOLED_CSV, FEATURES, LABEL
    OUT_DIR = out_dir
    FIG_DIR = os.path.join(OUT_DIR, "figures_shap")
    POOLED_CSV = os.path.join(data_dir, "omegaflux_sensors_pooled.csv")
    fj = os.path.join(OUT_DIR, "features.json")
    if os.path.exists(fj):
        with open(fj) as fh:
            FEATURES = json.load(fh)["features"]
    else:
        print(f"[WARN] {fj} not found; assuming all 26 descriptors")
    LABEL = {f: f for f in FEATURES}

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10})
    print(f"Loading {POOLED_CSV} ...")
    _df = pd.read_csv(POOLED_CSV, dtype={"Iteration_Name": str, "Group_ID": str, "Source_Tag": str})
    _tr, _te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED).split(_df, groups=_df["Group_ID"]))
    df_test = _df.iloc[_te].copy()
    del _df, _tr, _te
    gc.collect()
    print(f"Test partition: {len(df_test):,} grid points / {df_test['Group_ID'].nunique():,} rooms")
    _parts = []
    for _tag, _grp in df_test.groupby("Source_Tag"):
        _n = min(len(_grp), max(1000, int(round(N_SHAP * len(_grp) / len(df_test)))))
        _parts.append(_grp.sample(_n, random_state=SEED))
    shap_sample = pd.concat(_parts).sample(frac=1.0, random_state=SEED)
    X_shap = shap_sample[FEATURES].astype("float32")
    tags = shap_sample["Source_Tag"].to_numpy()
    rooms = shap_sample["Group_ID"].to_numpy()
    print(f"SHAP sample: {len(X_shap):,} points, {pd.Series(tags).value_counts().to_dict()}, {len(np.unique(rooms)):,} rooms")
    shap_low, base_low = tree_shap(f"{OUT_DIR}/omegaflux_xgb_low.json", X_shap, "UDI_low")
    shap_up, base_up = tree_shap(f"{OUT_DIR}/omegaflux_xgb_up.json", X_shap, "UDI_up")
    shap_useful = -(shap_low + shap_up)
    np.savez_compressed(
        f"{OUT_DIR}/shap_values.npz",
        shap_low=shap_low, shap_up=shap_up, shap_useful=shap_useful,
        base_low=base_low, base_up=base_up,
        X=X_shap.to_numpy(), features=np.array(FEATURES),
        tags=tags, rooms=rooms
    )
    print(f"\nSaved -> {OUT_DIR}/shap_values.npz")
    imp = pd.DataFrame({"descriptor": FEATURES, "label": [LABEL[f] for f in FEATURES]})
    for tgt, sv in (("low", shap_low), ("up", shap_up), ("useful", shap_useful)):
        a = np.abs(sv).mean(axis=0)
        imp[f"mean_abs_{tgt}"] = a
        imp[f"share_{tgt}_pct"] = a / a.sum() * 100.0
    imp = imp.sort_values("share_low_pct", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("Mean |SHAP| in percentage points of annual occupied hours\n")
    print(imp[["label", "mean_abs_low", "share_low_pct", "mean_abs_up", "share_up_pct"]].round(3).to_string(index=False))
    imp.to_csv(f"{OUT_DIR}/shap_shares.csv", index=False)
    print("\nConcentration:")
    for tgt in ("low", "up"):
        s = np.sort(imp[f"share_{tgt}_pct"].to_numpy())[::-1]
        c = np.cumsum(s)
        print(f"  UDI_{tgt}: {int((c < 80).sum() + 1)} descriptors reach 80%, {int((c < 95).sum() + 1)} reach 95%, of {len(FEATURES)}")
    print("\nSky-weighted : undiscounted subtense")
    for tgt in ("low", "up"):
        r = imp.set_index("descriptor")
        ratio = r.loc["Window_Sky_Weighted_Omega", f"mean_abs_{tgt}"] / r.loc["Total_Omega", f"mean_abs_{tgt}"]
        print(f"  UDI_{tgt}: {ratio:.2f}")
    CORE = {
        "Window_Sky_Weighted_Omega": -1, "Total_Omega": -1, "Global_inv_dmin2": -1,
        "South_Weighted_Omega": -1, "Visible_Window_Count": -1, "Occlusion_Multiplier": -1,
        "Mean_Window_Distance": +1, "Dominant_Win_Plane_Dist": +1, "Global_Cos_Theta": +1,
    }
    SECONDARY = {
        "Window_Angular_Spread": -1, "Annual_Direct_Sun_Fraction": -1, "Global_Header_Clearance": -1, 
        "Global_Sill_Clearance": -1, "East_Weighted_Omega": -1, "West_Weighted_Omega": -1,
        "Min_Occluder_Distance": -1, "Dominant_Win_Ray_Forward": -1, "Smooth_Mean_Depth": +1, 
        "Opaque_Wall_dmin": +1,
    }
    MIN_PTS = 40
    rows = []
    uniq = np.unique(rooms)
    for j, f in enumerate(FEATURES):
        x_all = X_shap[f].to_numpy()
        pooled = spearmanr(x_all, shap_low[:, j]).statistic if np.ptp(x_all) > 0 else np.nan
        within = []
        for g in uniq:
            m = rooms == g
            if m.sum() < MIN_PTS: continue
            xv = x_all[m]
            if np.ptp(xv) == 0: continue
            r = spearmanr(xv, shap_low[m, j]).statistic
            if np.isfinite(r): within.append(r)
        rows.append({
            "descriptor": f, "label": LABEL[f], "rho_pooled": pooled,
            "rho_within_mean": np.mean(within) if within else np.nan, "n_rooms_used": len(within)
        })
    sign = pd.DataFrame(rows).set_index("descriptor")
    a_c, n_c = check(CORE, "CORE signs (UDI_low)", imp, sign)
    a_s, n_s = check(SECONDARY, "SECONDARY signs (UDI_low)", imp, sign)
    print(f"\nCORE      {a_c}/{n_c} descriptors attribute in the expected direction")
    print(f"SECONDARY {a_s}/{n_s}")
    sign.round(4).to_csv(f"{OUT_DIR}/shap_sign_check.csv")
    KEY = "Window_Sky_Weighted_Omega"
    j = FEATURES.index(KEY)
    x, y = X_shap[KEY].to_numpy(), shap_low[:, j]
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
    sat = np.nan
    below = np.where(bin_g <= 0.90 * asym)[0]
    if len(below):
        sat = mid_g[below[0]]
    pd.DataFrame({"subtense": mid_g, "mean_shap_low": bin_g}).to_csv(f"{OUT_DIR}/dose_response.csv", index=False)
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
    ax.set_xlabel("Window_Sky_Weighted_Omega (fraction of the hemisphere)")
    ax.set_ylabel("Attribution to UDI$_{low}$ (percentage points)")
    ax.set_title("Attributed response of under-illumination to sky-weighted subtense")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/dose_response.png")
    plt.close(fig)
    print(f"  dose-response: crossing {cross:.3f}, asymptote {asym:+.1f} pp, 90% at {sat:.3f}")
    beeswarm(shap_low, X_shap, "Under-illuminated fraction", f"{FIG_DIR}/beeswarm_low.png")
    beeswarm(shap_up, X_shap, "Over-illuminated fraction", f"{FIG_DIR}/beeswarm_up.png")
    per = {}
    for t in np.unique(tags):
        a = np.abs(shap_useful[tags == t]).mean(axis=0)
        per[CLASS_LABEL.get(t, t)] = a / a.sum() * 100.0
    cls = pd.DataFrame(per, index=[LABEL[f] for f in FEATURES])
    cls["_rank"] = cls.mean(axis=1)
    cls = cls.sort_values("_rank", ascending=False).drop(columns="_rank")
    cls.round(4).to_csv(f"{OUT_DIR}/shap_by_class.csv")
    sub = cls.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    yy = np.arange(len(sub))
    w = 0.8 / len(sub.columns)
    for k, col in enumerate(sub.columns):
        ax.barh(yy + (k - (len(sub.columns) - 1) / 2) * w, sub[col], height=w, label=col)
    ax.set_yticks(yy)
    ax.set_yticklabels(sub.index)
    ax.set_xlabel("Share of total attribution (%)")
    ax.set_title("Attribution shift across plan classes")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/shap_by_class.png")
    plt.close(fig)
    z = (X_shap["Global_Sill_Clearance"].to_numpy() == 0)
    with open(f"{OUT_DIR}/shap_summary.json", "w") as fh:
        json.dump({
            "n_points": int(len(X_shap)),
            "n_rooms": int(len(np.unique(rooms))),
            "shares_low_pct": dict(zip(imp["descriptor"], imp["share_low_pct"].round(3))),
            "shares_up_pct": dict(zip(imp["descriptor"], imp["share_up_pct"].round(3))),
            "dose_response": {"crossing": float(cross), "asymptote_pp": float(asym), "saturation_90pct": float(sat)},
            "sign_check_core": {"agree": int(a_c), "tested": int(n_c)},
            "sign_check_secondary": {"agree": int(a_s), "tested": int(n_s)},
            "sill_clearance_censored_pct": float(z.mean() * 100),
        }, fh, indent=2)
    print(f"\nSaved -> {OUT_DIR}/shap_summary.json")
    print(f"Figures -> {FIG_DIR}/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR, help="model folder (contains features.json)")
    ap.add_argument("--data", default=DATA_DIR, help="folder with omegaflux_sensors_pooled.csv")
    args = ap.parse_args()
    configure(args.out, args.data)
    main()