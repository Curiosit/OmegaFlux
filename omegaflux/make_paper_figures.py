"""
omegaflux.make_paper_figures - all figures for paper 1, in one style, from outputs already on disk.

Nothing is retrained except the single prediction pass for the room example. A figure whose inputs
are missing is skipped with a message, so the script can be rerun as runs complete.

  fig_convergence      label shift by original under-illumination band, and the -ab ladder
  fig_transfer         E2 along both selection paths with seed spread; one-descriptor ablation
  fig_baselines        descriptors (XGBoost, MLP) against window coordinates, E1-E4
  fig_applicability    error against distance from the training data (E4, E2) and the E2 bias by solid angle
  fig_attribution      TreeSHAP, under- and over-illuminated models (two different panels)
  fig_room_examples    simulated, predicted and error maps: rectangular, rotated, two L-shapes
  (fig_room_example    the same for one room, and --candidates for a contact sheet)

Expected run folders (as created by the commands in README / the .bat files):
  OmegaFlux_Models/core, all, all_topo, fsE1_k6|k9|k12, fsTr_k6|k9, seed{1,2}_{core12,all26,fsE1_k12},
  abl_{inv_dmin2,opaque_wall,occlusion}, seed{1,2}_abl_*, seed{1,2}_fsE1_k9, baselines,
  core/applicability_E2, core/applicability_E4

Usage:  python -m omegaflux.make_paper_figures [--models OmegaFlux_Models] [--out figures/paper]
        [--room <Group_ID>] [--only fig_transfer,fig_baselines]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
W1, W2 = 3.54, 7.48            # single / double column width (in), 90 / 190 mm
C = {"core": "#2F6690", "all": "#9A9A9A", "fsE1": "#C44E52", "coord": "#DD8452",
     "mlp": "#55A868", "rect": "#4C72B0", "rect_rotated": "#55A868", "randomized": "#C44E52",
     "low": "#4C72B0", "up": "#DD8452", "useful": "#C44E52"}
CLASS_NAME = {"rect": "Rectangular", "rect_rotated": "Rotated", "randomized": "L-shaped"}
TEST_NAME = {"E1": "E1 held-out rooms", "E2": "E2 unseen plan shape",
             "E3": "E3 unseen orientation", "E4": "E4 more windows"}
TEST_TICK = {"E1": "E1\nheld-out\nrooms", "E2": "E2\nunseen\nplan shape",
             "E3": "E3\nunseen\norientation", "E4": "E4\nmore\nwindows"}
EXP_ROW = {"E1": "E1_main", "E2": "E2_holdout_Lshapes", "E3": "E3_holdout_rotated",
           "E4": "E4_unseen_aperture_count"}
PAPER_NAME = {   # descriptor names used in the paper
    "Total_Omega": "Window solid angle",
    "Window_Sky_Weighted_Omega": "Wing-discounted solid angle",
    "Global_Cos_Theta": "Window elevation cosine",
    "Global_Sill_Clearance": "Sill clearance",
    "Outer_Corner_Dmin": "Distance to nearest corner",
    "South_Weighted_Omega": "South-facing solid angle",
    "East_Weighted_Omega": "East-facing solid angle",
    "West_Weighted_Omega": "West-facing solid angle",
    "Annual_Direct_Sun_Fraction": "Direct-sun fraction",
    "Smooth_Depth_StdDev": "Enclosure anisotropy",
    "Window_Angular_Spread": "Window angular spread",
    "Dominant_Win_Ray_Forward": "Depth behind point",
    "Global_inv_dmin2": "Inverse sq. window distance",
    "Occlusion_Multiplier": "Occlusion multiplier",
    "Opaque_Wall_dmin": "Distance to nearest wall",
}
SOLID_ANGLE_PAIR = ("Total_Omega", "Window_Sky_Weighted_Omega")   # identical in convex plans

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "lines.linewidth": 1.3, "lines.markersize": 4, "savefig.dpi": 300,
})

M = "OmegaFlux_Models"
OUT = os.path.join("figures", "paper")
made, skipped = [], []


def path(*p):
    return os.path.join(M, *p)


def results(folder):
    f = path(folder, "results_table.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


def value(folder, exp, col="useful_R2"):
    d = results(folder)
    if d is None or exp not in set(d["experiment"]):
        return np.nan
    return float(d.loc[d["experiment"] == exp, col].iloc[-1])


def panel_label(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    made.append(name)
    print(f"[OK] {name}")


def need(name, *files):
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        skipped.append((name, missing))
        print(f"[SKIP] {name}: missing {', '.join(missing)}")
        return False
    return True


# ---------------------------------------------------------------------------
# 1. Convergence check
# ---------------------------------------------------------------------------
def fig_convergence():
    sens = os.path.join("data", "convergence_check", "convergence_sensors.csv.gz")
    rooms = os.path.join("data", "convergence_check", "convergence_rooms.csv")
    if not need("fig_convergence", sens, rooms):
        return
    s = pd.read_csv(sens)
    s["d"] = s["udi_useful_ab8"] - s["udi_useful_ab5"]
    bands = [-0.1, 10, 30, 50, 70, 90, 100.1]
    labels = ["0–10", "10–30", "30–50", "50–70", "70–90", "90–100"]
    s["band"] = pd.cut(s["udi_low_ab5"], bands, labels=labels)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    x, w = np.arange(len(labels)), 0.26
    for k, klass in enumerate(["rect", "rect_rotated", "randomized"]):
        g = s[s["class"] == klass]
        m, lo, hi, nroom = [], [], [], []
        for b in labels:
            sel = g["band"] == b
            v = g.loc[sel, "d"]
            ok = len(v) >= 30 and g.loc[sel, "room_id"].nunique() >= 3
            m.append(v.mean() if ok else np.nan)
            lo.append(np.percentile(v, 25) if ok else np.nan)
            hi.append(np.percentile(v, 75) if ok else np.nan)
            nroom.append(g.loc[sel, "room_id"].nunique())
        m = np.array(m)
        ax.bar(x + (k - 1) * w, m, w, color=C[klass], label=CLASS_NAME[klass])

        ax.errorbar(x + (k - 1) * w, m, yerr=[m - np.array(lo), np.array(hi) - m], fmt="none",
                    ecolor="k", lw=0.6, capsize=1.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Under-illuminated hours at the sensor, original labels (%)")
    ax.set_ylabel("Change in UDI$_{useful}$ (pp)\n-ab 8 minus -ab 5")
    ax.legend(loc="upper left")
    ax.text(0.02, 0.62, "bars: mean; whiskers: IQR\nbands with < 30 sensors or < 3 rooms omitted", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5, color="0.35")
    panel_label(ax, "a")

    ax = axes[1]
    r = pd.read_csv(rooms).dropna(subset=["useful_bias_ab8_to_ab12"]).sort_values("useful_bias_ab5_to_ab8")
    y = np.arange(len(r))
    ax.barh(y + 0.2, r["useful_bias_ab5_to_ab8"], 0.4, color="0.55", label="-ab 5 → 8")
    ax.barh(y - 0.2, r["useful_bias_ab8_to_ab12"], 0.4, color=C["randomized"], label="-ab 8 → 12")
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("Expert_LShape_", "L") for n in r["room_id"]], fontsize=6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Mean change in UDI$_{useful}$ (pp)")
    ax.set_title("Ladder, ten most affected L-shapes")
    ax.legend(loc="lower right")
    panel_label(ax, "b")
    fig.tight_layout()
    save(fig, "fig_convergence")


# ---------------------------------------------------------------------------
# 2. Transfer by descriptor set
# ---------------------------------------------------------------------------
def seeds(folders, exp="E2_holdout_Lshapes"):
    v = np.array([value(f, exp) for f in folders])
    return v[np.isfinite(v)]


def fig_transfer():
    if not need("fig_transfer", path("core", "results_table.csv"), path("fsE1_k12", "results_table.csv")):
        return
    core = seeds(["core", "seed1_core12", "seed2_core12"])
    fse1 = seeds(["fsE1_k12", "seed1_fsE1_k12", "seed2_fsE1_k12"])
    all26 = seeds(["all_topo", "seed1_all26", "seed2_all26"])
    path_tr = {6: seeds(["fsTr_k6"]), 9: seeds(["fsTr_k9"]), 12: core}
    path_e1 = {6: seeds(["fsE1_k6"]), 9: seeds(["fsE1_k9"]), 12: fse1}

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    xpos = {6: 0, 9: 1, 12: 2, 26: 3.3}            # categorical axis, gap before "all 26"
    for pth, col, lab, dx in ((path_tr, C["core"], "selected with transfer objective", -0.06),
                              (path_e1, C["fsE1"], "selected for in-distribution accuracy", 0.06)):
        ks = [k for k in (6, 9, 12) if len(pth[k])]
        ax.plot([xpos[k] + dx for k in ks], [pth[k].mean() for k in ks], "o-", color=col, label=lab)
        for k in ks:
            if len(pth[k]) > 1:
                ax.vlines(xpos[k] + dx, pth[k].min(), pth[k].max(), color=col, lw=2.2, alpha=0.5)
    if len(all26):
        ax.errorbar(xpos[26], all26.mean(), yerr=[[all26.mean() - all26.min()], [all26.max() - all26.mean()]],
                    fmt="s", color=C["all"], capsize=3, label="all 26 descriptors")
    ax.set_xticks([xpos[k] for k in (6, 9, 12, 26)])
    ax.set_xticklabels(["6", "9", "12", "all 26"])
    ax.set_xlim(-0.4, 3.7)
    ax.set_xlabel("Number of descriptors")
    ax.set_ylabel("R$^2$, rectangles → L-shapes (E2)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5)
    ax.text(0.99, 0.97, "vertical bars: range\nover 3 model seeds", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color="0.35")
    panel_label(ax, "a")

    ax = axes[1]
    abl = [("abl_inv_dmin2", "Global_inv_dmin2"), ("abl_opaque_wall", "Opaque_Wall_dmin"),
           ("abl_occlusion", "Occlusion_Multiplier")]
    base = {s: value(f, "E2_holdout_Lshapes") for s, f in (("42", "fsE1_k9"), ("1", "seed1_fsE1_k9"),
                                                            ("2", "seed2_fsE1_k9"))}
    y = np.arange(len(abl))
    for yi, (folder, name) in zip(y, abl):
        runs = {"42": folder, "1": f"seed1_{folder}", "2": f"seed2_{folder}"}
        d = np.array([value(f, "E2_holdout_Lshapes") - base[s] for s, f in runs.items()])
        d = d[np.isfinite(d)]
        col = "0.7" if name == "Occlusion_Multiplier" else C["fsE1"]
        if len(d):
            ax.barh(yi, d.mean(), color=col)
            if len(d) > 1:
                ax.errorbar(d.mean(), yi, xerr=[[d.mean() - d.min()], [d.max() - d.mean()]], fmt="none",
                            ecolor="k", lw=0.8, capsize=2)
            ax.scatter(d, np.full(len(d), yi), s=8, color="k", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([PAPER_NAME[n] for _, n in abl])
    ax.axvline(0, color="k", lw=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Change in E2 R$^2$ when added\nto the shared nine descriptors")
    panel_label(ax, "b")
    fig.tight_layout()
    save(fig, "fig_transfer")


# ---------------------------------------------------------------------------
# 3. Baselines
# ---------------------------------------------------------------------------
def fig_baselines():
    bl = results("baselines")
    if bl is None or results("core") is None:
        skipped.append(("fig_baselines", ["baselines or core results"]))
        print("[SKIP] fig_baselines: missing baselines or core results")
        return
    def bl_val(model, test, col):
        n = f"{model}__{EXP_ROW[test]}"
        return float(bl.loc[bl["experiment"] == n, col].iloc[-1]) if n in set(bl["experiment"]) else np.nan
    def seeded(folders, t, c):
        v = np.array([value(f, EXP_ROW[t], c) for f in folders])
        return v[np.isfinite(v)]
    core_f = {"E2": ["core", "seed1_core12", "seed2_core12"]}
    all_f = {"E1": ["all"], "E2": ["all_topo", "seed1_all26", "seed2_all26"]}
    rows = [
        ("Descriptors (12), XGBoost", C["core"], lambda t, c: seeded(core_f.get(t, ["core"]), t, c)),
        ("Descriptors (26), XGBoost", C["all"], lambda t, c: seeded(all_f.get(t, ["all_topo"]), t, c)),
        ("Descriptors (12), MLP", C["mlp"], lambda t, c: bl_val("mlp12", t, c)),
        ("Descriptors (26), MLP", "#A8D5B5", lambda t, c: bl_val("mlp26", t, c)),
        ("Window coordinates, XGBoost", C["coord"], lambda t, c: bl_val("coord_xgb", t, c)),
    ]
    tests = ["E1", "E2", "E3", "E4"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.1))
    for ax, col, ylab, lab in ((axes[0], "useful_R2", "R$^2$, UDI$_{useful}$", "a"),
                               (axes[1], "useful_RMSE", "RMSE, UDI$_{useful}$ (pp)", "b")):
        x, w = np.arange(len(tests)), 0.16
        for k, (name, colr, f) in enumerate(rows):
            runs = [np.atleast_1d(f(t, col)) for t in tests]
            runs = [r[np.isfinite(r)] for r in runs]
            v = np.array([r.mean() if len(r) else np.nan for r in runs])
            ax.bar(x + (k - 2) * w, np.nan_to_num(v, nan=0), w, color=colr, label=name)
            for xi, r in zip(x + (k - 2) * w, runs):
                if len(r) > 1:
                    ax.errorbar(xi, r.mean(), yerr=[[r.mean() - r.min()], [r.max() - r.mean()]], fmt="none",
                                ecolor="k", lw=0.7, capsize=1.5)
            for xi, vi in zip(x + (k - 2) * w, v):
                if np.isfinite(vi) and col == "useful_R2" and vi < 0:
                    ax.text(xi, vi - 0.01, f"{vi:.2f}", ha="center", va="top", fontsize=5.5)
                if not np.isfinite(vi):
                    ax.text(xi, 0.01 if col == "useful_R2" else 0.2, "n/r", rotation=90, ha="center",
                            va="bottom", fontsize=5.5, color="0.4")
        ax.set_xticks(x)
        ax.set_xticklabels([TEST_TICK[t] for t in tests])
        ax.set_ylabel(ylab)
        ax.axhline(0, color="k", lw=0.6)
        panel_label(ax, lab)
    axes[0].set_ylim(min(-0.12, axes[0].get_ylim()[0]), 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3, fontsize=6.8)
    note = "whiskers: range over 3 model seeds"
    if any(t.get_text() == "n/r" for t in axes[1].texts):
        note += "\nn/r: not run"
    axes[1].text(0.02, 0.97, note, transform=axes[1].transAxes, ha="left", va="top", fontsize=6.5, color="0.35")
    fig.tight_layout()
    save(fig, "fig_baselines")


# ---------------------------------------------------------------------------
# 4. Applicability (distance check), E4 against E2
# ---------------------------------------------------------------------------
def fig_applicability():
    folders = [("E4", path("core", "applicability_E4")), ("E2", path("core", "applicability_E2"))]
    files = [os.path.join(f, n) for _, f in folders for n in ("by_distance_decile.csv", "summary.json")]
    files.append(os.path.join(folders[1][1], "by_solid_angle_decile.csv"))
    if not need("fig_applicability", *files):
        return
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.8), gridspec_kw={"wspace": 0.42})
    for ax, (ctx, f), lab in zip(axes[:2], folders, "ab"):
        dec = pd.read_csv(os.path.join(f, "by_distance_decile.csv"))
        s = json.load(open(os.path.join(f, "summary.json")))
        mid = 0.5 * (dec["dist_min"] + dec["dist_max"])
        ax.plot(mid, dec["MAE"], "o-", color=C["useful"], label="transfer test (distance deciles)")
        ax.axhline(s["calibration_error"]["MAE"], color=C["rect"], ls="--", lw=1,
                   label="held-out rooms of the training family")
        ax.axvline(s["threshold"], color="k", ls=":", lw=1, label="flag threshold (95% in-family)")
        ax.set_xlim(left=0)
        ax.set_xlabel("Distance to 10 nearest training\npoints (standardised descriptors)")
        rho = s["spearman_distance_vs_abs_error"]
        fl, un = s["test_flagged"]["MAE"], s["test_unflagged"]["MAE"]
        ax.set_title(f"{TEST_NAME[ctx]}\n{s['share_flagged_test']:.0%} of points flagged")
        ax.text(0.97, 0.97, f"MAE flagged {fl:.2f}\nunflagged {un:.2f} pp\n\u03c1 = {rho:+.2f}",
                transform=ax.transAxes, va="top", ha="right", fontsize=6.5)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        panel_label(ax, lab)
    axes[0].set_ylabel("Mean absolute error, UDI$_{useful}$ (pp)")

    ax = axes[2]
    om = pd.read_csv(os.path.join(folders[1][1], "by_solid_angle_decile.csv"))
    xm = np.maximum(0.5 * (om["omega_min"] + om["omega_max"]), 1e-4)
    ax.plot(xm, om["bias_useful"], "o-", color=C["useful"], label="UDI$_{useful}$")
    ax.plot(xm, om["bias_low"], "s--", color=C["low"], label="UDI$_{low}$")
    ax.plot(xm, om["bias_up"], "^:", color=C["up"], label="UDI$_{up}$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Total window solid angle at the\npoint (fraction of hemisphere)")
    ax.set_ylabel("Bias, predicted \u2212 simulated (pp)")
    ax.set_title("E2: bias by solid angle\n(deciles of L-shape points)")
    ax.legend(loc="upper right", fontsize=6.5)
    panel_label(ax, "c")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.36, 0.98), ncol=3, fontsize=6.5)
    save(fig, "fig_applicability")


# ---------------------------------------------------------------------------
# 6. Attribution (TreeSHAP), solid-angle pair combined
# ---------------------------------------------------------------------------
def fig_attribution():
    f = path("core", "shap_values.npz")
    if not need("fig_attribution", f):
        return
    S = np.load(f, allow_pickle=True)
    feats = [str(x) for x in S["features"]]
    X = pd.DataFrame(S["X"], columns=feats)
    sv = {"low": pd.DataFrame(S["shap_low"], columns=feats), "up": pd.DataFrame(S["shap_up"], columns=feats)}
    a, b = SOLID_ANGLE_PAIR
    if a in feats and b in feats:            # identical in convex plans: report combined attribution only
        for k in sv:
            sv[k][a] = sv[k][a] + sv[k][b]
            sv[k] = sv[k].drop(columns=b)
        X = X.drop(columns=b)
    names = {**PAPER_NAME, a: "Window solid angle\n(incl. wing-discounted)"}
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.4), gridspec_kw={"wspace": 0.95})
    for ax, key, title, lab in ((axes[0], "low", "Under-illuminated fraction, UDI$_{low}$", "a"),
                                (axes[1], "up", "Over-illuminated fraction, UDI$_{up}$", "b")):
        s = sv[key]
        order = s.abs().mean().sort_values(ascending=True).index
        for row, c in enumerate(order):
            v = s[c].to_numpy()
            xv = X[c].to_numpy()
            lo, hi = np.percentile(xv, [1, 99])
            col = np.clip((xv - lo) / (hi - lo + 1e-12), 0, 1)
            sc = ax.scatter(v, row + rng.normal(0, 0.09, len(v)), c=col, cmap="coolwarm", s=1.5,
                            alpha=0.4, linewidths=0, rasterized=True)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([names.get(c, c) for c in order], fontsize=6.5)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Attribution (pp)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.text(-0.85, 1.04, lab, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Descriptor value (low → high)", fontsize=7)
    cb.set_ticks([])
    save(fig, "fig_attribution")


# ---------------------------------------------------------------------------
# 7. Room example
# ---------------------------------------------------------------------------
def room_outline(group_id):
    """Corners and window segments of one room, parsed from the raw export as extract_features does."""
    try:
        from omegaflux.extract_features import DATASETS, build_apertures, load_table, parse_room_geometry
    except Exception:
        return None
    suffix, name = group_id.split(":", 1)
    for enabled, raw_path, suf, _ in DATASETS:
        if suf != suffix or not os.path.exists(raw_path):
            continue
        df = load_table(raw_path)
        row = df[df["Iteration_Name"].astype(str) == name]
        if row.empty:
            return None
        corners, specs = parse_room_geometry(row.iloc[0])
        walls = [(corners[i], corners[(i + 1) % len(corners)]) for i in range(len(corners))]
        windows = []
        for a in build_apertures(corners, specs, walls):
            (cx, cy, _), w = a["center"], a["width"]
            nx, ny = a["normal"]
            rx, ry = -ny, nx                       # along the wall
            windows.append(((cx - rx * w / 2, cy - ry * w / 2), (cx + rx * w / 2, cy + ry * w / 2)))
        return corners, windows
    return None


def draw_outline(ax, outline, x0, y0):
    corners, windows = outline
    xs = [c[0] - x0 for c in corners] + [corners[0][0] - x0]
    ys = [c[1] - y0 for c in corners] + [corners[0][1] - y0]
    ax.plot(xs, ys, color="k", lw=2.2, solid_capstyle="projecting", solid_joinstyle="miter", zorder=4)
    for (ax_, ay), (bx, by) in windows:
        ax.plot([ax_ - x0, bx - x0], [ay - y0, by - y0], color="white", lw=2.6, solid_capstyle="butt", zorder=5)

def fig_room_candidates(d, cand):
    """Contact sheet: simulated field and error map for each candidate room, to pick one with --room."""
    n = len(cand)
    fig, axes = plt.subplots(2, n, figsize=(max(W2, 1.6 * n), 3.6), squeeze=False)
    for j, (gid, row) in enumerate(cand.iterrows()):
        r = d[d["Group_ID"] == gid]
        lc = r["Meta_L_char"].to_numpy()
        x, y = r["Meta_Sensor_X_norm"].to_numpy() * lc, r["Meta_Sensor_Y_norm"].to_numpy() * lc
        outline = room_outline(gid)
        ms = (0.25 / max(np.ptp(x), np.ptp(y), 1.0) * 100) ** 2
        m = max(5, float(np.abs(r["err"]).quantile(0.98)))
        for i, (val, cmap, lim) in enumerate(((r["act"], "viridis", (0, 100)), (r["err"], "RdBu_r", (-m, m)))):
            ax = axes[i, j]
            ax.scatter(x, y, c=val, cmap=cmap, vmin=lim[0], vmax=lim[1], s=ms, marker="s", linewidths=0)
            if outline is not None:
                draw_outline(ax, outline, min(c[0] for c in outline[0]), min(c[1] for c in outline[0]))
            ax.set_aspect("equal")
            ax.axis("off")
        axes[0, j].set_title(f"{gid.split(':')[-1].replace('Expert_LShape_', 'L')}\nMAE {row['mae']:.1f} | bias {row['bias']:+.1f}\n"
                             f"low {row['low']:.0f}% | up {row['up']:.0f}%", fontsize=6)
    axes[0, 0].text(-0.15, 0.5, "simulated\nUDI$_{useful}$", transform=axes[0, 0].transAxes, ha="right", va="center", fontsize=6.5)
    axes[1, 0].text(-0.15, 0.5, "predicted −\nsimulated", transform=axes[1, 0].transAxes, ha="right", va="center", fontsize=6.5)
    save(fig, "fig_room_candidates")
    print("Candidates (pass one with --room):")
    for gid in cand.index:
        print(f"  --room \"{gid}\"")


def fig_room_example(room=None, klass="randomized", both_arms=True, candidates=0, max_up=None):
    pooled = os.path.join("data", "processed", "omegaflux_sensors_pooled.csv")
    if not need("fig_room_example", pooled, path("core", "features.json")):
        return
    import xgboost as xgb
    from sklearn.model_selection import GroupShuffleSplit
    from omegaflux import train as T
    feats = json.load(open(path("core", "features.json")))["features"]
    g = pd.read_csv(pooled, usecols=["Group_ID"], dtype={"Group_ID": str})["Group_ID"]
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=T.TEST_FRACTION, random_state=T.SEED).split(g, groups=g))
    test = set(g.iloc[te])
    if room:
        test = {room}
    cols = ["Group_ID", "Source_Tag", "Target_UDI_low", "Target_UDI_up", "Meta_L_char",
            "Meta_Sensor_X_norm", "Meta_Sensor_Y_norm", "Meta_N_Windows", "Visible_Window_Count"] + feats
    parts = [c[c["Group_ID"].isin(test) & ((c["Source_Tag"] == klass) | bool(room))]
             for c in pd.read_csv(pooled, usecols=list(dict.fromkeys(cols)), dtype={"Group_ID": str},
                                  chunksize=500_000)]
    d = pd.concat(parts, ignore_index=True)
    if d.empty:
        skipped.append(("fig_room_example", ["no matching held-out room"]))
        print("[SKIP] fig_room_example: no matching held-out room")
        return
    models = []
    for t in ("low", "up"):
        m = xgb.XGBRegressor()
        m.load_model(path("core", f"omegaflux_xgb_{t}.json"))
        models.append(m)
    X = d[feats].astype("float32").to_numpy()
    p = np.clip(100 - np.clip(models[0].predict(X), 0, 100) - np.clip(models[1].predict(X), 0, 100), 0, 100)
    d["pred"] = p
    d["act"] = np.clip(100 - d["Target_UDI_low"] - d["Target_UDI_up"], 0, 100)
    d["err"] = d["pred"] - d["act"]
    if not room:   # high contrast, typical error: highest spread among rooms with median-range MAE
        st = d.groupby("Group_ID").agg(sd=("act", "std"), mae=("err", lambda e: np.abs(e).mean()),
                                       up=("Target_UDI_up", "mean"), low=("Target_UDI_low", "mean"),
                                       bias=("err", "mean"), n_win=("Meta_N_Windows", "first"),
                                       blind=("Visible_Window_Count", lambda v: float((v <= 0).mean())))
        if both_arms and klass == "randomized":
            # windows reaching both arms of the L: several windows and almost no point without a window in view
            arms = st[(st["n_win"] >= 2) & (st["blind"] < 0.02)]
            if len(arms) >= 5:
                st = arms
        if max_up is not None:     # rooms where low useful UDI mostly means too little light
            st = st[st["up"] <= max_up] if (st["up"] <= max_up).sum() >= 5 else st
        lo, hi = st["mae"].quantile([0.4, 0.6])
        cand = st[(st["mae"] >= lo) & (st["mae"] <= hi)]
        cand = cand if len(cand) else st
        if candidates:
            fig_room_candidates(d, cand.sort_values("sd", ascending=False).head(candidates))
            return
        room = cand["sd"].idxmax()
    r = d[d["Group_ID"] == room]
    lc = r["Meta_L_char"].to_numpy()
    x, y = r["Meta_Sensor_X_norm"].to_numpy() * lc, r["Meta_Sensor_Y_norm"].to_numpy() * lc
    outline = room_outline(room)
    if outline is None:
        print(f"[WARN] fig_room_example: no raw geometry for {room}; walls and windows not drawn")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.6))
    vmax = 100
    extent = max(np.ptp(x), np.ptp(y), 1.0)
    msize = (0.25 / extent * 150) ** 2          # one marker per 0.25 m grid cell, no gaps
    for ax, val, title, cmap, lim in ((axes[0], r["act"], "Simulated UDI$_{useful}$ (%)", "viridis", (0, vmax)),
                                      (axes[1], r["pred"], "Predicted UDI$_{useful}$ (%)", "viridis", (0, vmax)),
                                      (axes[2], r["err"], "Predicted − simulated (pp)", "RdBu_r", None)):
        if lim is None:
            m = max(5, float(np.abs(val).quantile(0.98)))
            lim = (-m, m)
        sc = ax.scatter(x, y, c=val, cmap=cmap, vmin=lim[0], vmax=lim[1], s=msize, marker="s", linewidths=0)
        if outline is not None:
            cx_ = [c[0] for c in outline[0]]
            cy_ = [c[1] for c in outline[0]]
            draw_outline(ax, outline, min(cx_), min(cy_))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    mae = np.abs(r["err"]).mean()
    if outline is not None:
        fig.text(0.5, 0.01, "black: walls; white: window openings", ha="center", fontsize=6.5, color="0.35")
    fig.suptitle(f"Held-out room {room.split(':')[-1]} ({CLASS_NAME.get(r['Source_Tag'].iloc[0], '')}, "
                 f"{len(r)} sensors): per-sensor MAE {mae:.2f} pp, room-mean difference {r['err'].mean():+.2f} pp",
                 fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig_room_example")


def held_out_predictions(classes):
    """Held-out (E1 test) grid points of the given classes, with core-model predictions."""
    import xgboost as xgb
    from sklearn.model_selection import GroupShuffleSplit
    from omegaflux import train as T
    pooled = os.path.join("data", "processed", "omegaflux_sensors_pooled.csv")
    feats = json.load(open(path("core", "features.json")))["features"]
    g = pd.read_csv(pooled, usecols=["Group_ID"], dtype={"Group_ID": str})["Group_ID"]
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=T.TEST_FRACTION, random_state=T.SEED).split(g, groups=g))
    test = set(g.iloc[te])
    cols = ["Group_ID", "Source_Tag", "Target_UDI_low", "Target_UDI_up", "Meta_L_char",
            "Meta_Sensor_X_norm", "Meta_Sensor_Y_norm", "Meta_N_Windows", "Visible_Window_Count"] + feats
    d = pd.concat([c[c["Group_ID"].isin(test) & c["Source_Tag"].isin(classes)]
                   for c in pd.read_csv(pooled, usecols=list(dict.fromkeys(cols)), dtype={"Group_ID": str},
                                        chunksize=500_000)], ignore_index=True)
    ms = []
    for t in ("low", "up"):
        m = xgb.XGBRegressor()
        m.load_model(path("core", f"omegaflux_xgb_{t}.json"))
        ms.append(m)
    X = d[feats].astype("float32").to_numpy()
    d["pred"] = np.clip(100 - np.clip(ms[0].predict(X), 0, 100) - np.clip(ms[1].predict(X), 0, 100), 0, 100)
    d["act"] = np.clip(100 - d["Target_UDI_low"] - d["Target_UDI_up"], 0, 100)
    d["err"] = d["pred"] - d["act"]
    return d


def pick_room(d, klass, max_up=None, exclude=()):
    st = d[d["Source_Tag"] == klass].groupby("Group_ID").agg(
        sd=("act", "std"), mae=("err", lambda e: np.abs(e).mean()), up=("Target_UDI_up", "mean"),
        n_win=("Meta_N_Windows", "first"), blind=("Visible_Window_Count", lambda v: float((v <= 0).mean())))
    st = st.drop(index=[r for r in exclude if r in st.index])
    if klass == "randomized":
        arms = st[(st["n_win"] >= 2) & (st["blind"] < 0.02)]
        st = arms if len(arms) >= 5 else st
    if max_up is not None and (st["up"] <= max_up).sum() >= 5:
        st = st[st["up"] <= max_up]
    lo, hi = st["mae"].quantile([0.4, 0.6])
    cand = st[(st["mae"] >= lo) & (st["mae"] <= hi)]
    return (cand if len(cand) else st)["sd"].idxmax()


def fig_room_examples(rooms=None, max_up=None):
    """Simulated, predicted and error maps for several held-out rooms (columns), on one metric scale."""
    pooled = os.path.join("data", "processed", "omegaflux_sensors_pooled.csv")
    if not need("fig_room_examples", pooled, path("core", "features.json")):
        return
    order = ["rect", "rect_rotated", "randomized", "randomized"]
    d = held_out_predictions(set(order))
    given = [r.strip() for r in rooms] if rooms else ["auto"] * len(order)
    given += ["auto"] * (len(order) - len(given))
    rooms = []
    for k, r in zip(order, given):          # "auto" = pick a typical, high-contrast room of that class
        rooms.append(pick_room(d, k, max_up, exclude=rooms + given) if r == "auto" else r)
    missing = [r for r in rooms if r not in set(d["Group_ID"])]
    if missing:
        print(f"[SKIP] fig_room_examples: not held-out rooms of these classes: {missing}")
        skipped.append(("fig_room_examples", missing))
        return
    data = []
    for r in rooms:
        q = d[d["Group_ID"] == r]
        lc = q["Meta_L_char"].to_numpy()
        data.append((r, q, q["Meta_Sensor_X_norm"].to_numpy() * lc, q["Meta_Sensor_Y_norm"].to_numpy() * lc,
                     room_outline(r)))
    span = max(max(np.ptp(x), np.ptp(y)) for _, _, x, y, _ in data) + 0.8      # one metric scale for all panels
    emax = max(5.0, float(np.abs(pd.concat([q["err"] for _, q, _, _, _ in data])).quantile(0.98)))
    n = len(data)
    fig, axes = plt.subplots(3, n, figsize=(W2, 6.2), squeeze=False, gridspec_kw={"hspace": 0.12, "wspace": 0.05})
    panel_w_pt = W2 / (n + 0.6) * 72
    msize = (0.25 / span * panel_w_pt) ** 2
    rows = (("act", "viridis", (0, 100), "Simulated\nUDI$_{useful}$ (%)"),
            ("pred", "viridis", (0, 100), "Predicted\nUDI$_{useful}$ (%)"),
            ("err", "RdBu_r", (-emax, emax), "Predicted −\nsimulated (pp)"))
    for j, (gid, q, x, y, outline) in enumerate(data):
        cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
        for i, (col, cmap, lim, _) in enumerate(rows):
            ax = axes[i, j]
            sc = ax.scatter(x, y, c=q[col], cmap=cmap, vmin=lim[0], vmax=lim[1], s=msize, marker="s", linewidths=0)
            if outline is not None:
                draw_outline(ax, outline, min(c[0] for c in outline[0]), min(c[1] for c in outline[0]))
            ax.set_xlim(cx - span / 2, cx + span / 2)
            ax.set_ylim(cy - span / 2, cy + span / 2)
            ax.set_aspect("equal")
            ax.axis("off")
            if j == n - 1:
                cb = fig.colorbar(sc, ax=axes[i, :].tolist(), fraction=0.02, pad=0.01)
                cb.set_label(rows[i][3], fontsize=6.5)
        mae, bias = float(np.abs(q["err"]).mean()), float(q["err"].mean())
        name = gid.split(":")[-1].replace("Expert_LShape_", "L-shape ").replace("_Rotated_", " rotated ")
        axes[0, j].set_title(f"{CLASS_NAME[q['Source_Tag'].iloc[0]]}\n{name}\nMAE {mae:.1f} pp, bias {bias:+.1f} pp",
                             fontsize=6.8)
    axes[0, 0].plot([0, 0], [0, 0])  # keep layout stable
    sx0 = axes[2, 0].get_xlim()[0] + 0.3
    sy0 = axes[2, 0].get_ylim()[0] + 0.2
    axes[2, 0].plot([sx0, sx0 + 2], [sy0, sy0], color="k", lw=1.5)
    axes[2, 0].text(sx0 + 1, sy0 + 0.25, "2 m", ha="center", fontsize=6)
    fig.text(0.5, 0.06, "black: walls; white: window openings; all plans at the same scale", ha="center",
             fontsize=6.5, color="0.35")
    print("Rooms shown: " + ", ".join(rooms))
    save(fig, "fig_room_examples")


FIGS = {"fig_convergence": fig_convergence, "fig_transfer": fig_transfer, "fig_baselines": fig_baselines,
        "fig_applicability": fig_applicability,
        "fig_attribution": fig_attribution, "fig_room_examples": fig_room_examples}


def main():
    global M, OUT
    ap = argparse.ArgumentParser(description="Paper-1 figures")
    ap.add_argument("--models", default=M)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--room", default=None, help="Group_ID for the single-room figure (--only fig_room_example)")
    ap.add_argument("--rooms", default=None,
                    help="comma-separated Group_IDs for fig_room_examples, in the column order rectangular, "
                         "rotated, L-shape, L-shape; 'auto' picks one (default: all auto)")
    ap.add_argument("--room-class", default="randomized", choices=["rect", "rect_rotated", "randomized"])
    ap.add_argument("--any-arms", action="store_true",
                    help="L-shape example: do not require windows in both arms")
    ap.add_argument("--candidates", type=int, default=0,
                    help="instead of the room example, draw a contact sheet of N candidate rooms to choose from")
    ap.add_argument("--max-up", type=float, default=None,
                    help="candidate rooms: maximum room-mean over-illuminated share (%%), e.g. 10")
    ap.add_argument("--only", default=None, help="comma-separated figure names")
    args = ap.parse_args()
    M, OUT = args.models, args.out
    names = [n.strip() for n in args.only.split(",")] if args.only else list(FIGS)
    for n in names:
        if n == "fig_room_examples":
            fig_room_examples(args.rooms.split(",") if args.rooms else None, args.max_up)
        elif n == "fig_room_example":
            fig_room_example(args.room, args.room_class, both_arms=not args.any_arms,
                             candidates=args.candidates, max_up=args.max_up)
        else:
            FIGS[n]()
    print(f"\n{len(made)} figures written to {OUT}/" + (f"; skipped: {[s[0] for s in skipped]}" if skipped else ""))


if __name__ == "__main__":
    main()
