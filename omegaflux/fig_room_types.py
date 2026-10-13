"""omegaflux.fig_room_types - Figure 2 of the paper: isometric room types (bins A-E rectangular, bin F L-shaped).

Drawn from the sampling ranges in Appendix A (Table A2); no data files needed.
Usage:  python -m omegaflux.fig_room_types [--out figures/paper]
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
_ap.add_argument("--out", default=os.path.join("figures", "paper"))
OUT = _ap.parse_args().out
os.makedirs(OUT, exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch, FancyArrowPatch
from matplotlib.path import Path

C, S = np.cos(np.radians(30)), np.sin(np.radians(30))
H, CUT, T, SLAB = 2.7, 0.22, 0.20, 0.25
FLOOR, WALL, GLASS, GLASS_E = "#efe3cc", "#8c8c8c", "#bcd3e6", "#5f86a8"
INK, GREY = "#111111", "#444444"

K = 1.0
def P(x, y, z=0.0, o=(0, 0)):
    # viewer at south-east: south wall front-left, east wall front-right
    return (o[0] + K * (x + y) * C * 0.5, o[1] + K * ((y - x) * S * 0.5 + z * 0.5))

def proj(pts, o):
    return [P(*p, o=o) if len(p) == 3 else P(p[0], p[1], 0, o) for p in pts]

def offset_poly(poly, d):
    """outward offset of a CCW orthogonal polygon by d"""
    n = len(poly); out = []
    for i in range(n):
        p0, p1, p2 = np.array(poly[i - 1]), np.array(poly[i]), np.array(poly[(i + 1) % n])
        e1, e2 = p1 - p0, p2 - p1
        n1 = np.array([e1[1], -e1[0]]) / np.linalg.norm(e1)
        n2 = np.array([e2[1], -e2[0]]) / np.linalg.norm(e2)
        out.append(tuple(p1 + d * (n1 + n2)))
    return out

def inside(poly, x, y):
    return Path(poly).contains_point((x, y))

def draw_room(ax, plan, o, windows, grid=0.5):
    outer = offset_poly(plan, T)
    # slab sides (south + east faces, visible)
    xs = [p[0] for p in outer]; ys = [p[1] for p in outer]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    for a, b in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1))]:
        q = [(a[0], a[1], CUT), (b[0], b[1], CUT), (b[0], b[1], -SLAB), (a[0], a[1], -SLAB)]
        ax.add_patch(Polygon(proj(q, o), closed=True, fc="white", ec=INK, lw=1.0, zorder=1))
    for a, b in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1))]:
        ax.plot(*zip(P(a[0], a[1], 0, o), P(b[0], b[1], 0, o)), color=INK, lw=0.6, zorder=1.1)
    # floor + grid
    ax.add_patch(Polygon(proj(plan, o), closed=True, fc=FLOOR, ec="none", zorder=2))
    gx = np.arange(min(p[0] for p in plan) + grid / 2, max(p[0] for p in plan), grid)
    gy = np.arange(min(p[1] for p in plan) + grid / 2, max(p[1] for p in plan), grid)
    pts = [P(x, y, 0.0, o) for x in gx for y in gy if inside(plan, x, y)]
    ax.scatter(*zip(*pts), s=1.2, c="#6b5a4a", zorder=3, lw=0)
    # back windows (behind wall ring)
    for w in windows:
        if not w["front"]:
            draw_window(ax, w, o, z=4)
    # wall ring at cut height
    ring = Path.make_compound_path(
        Path([P(x, y, CUT, o) for x, y in outer] + [P(*outer[0], CUT, o)], closed=True),
        Path([P(x, y, CUT, o) for x, y in plan[::-1]] + [P(*plan[-1], CUT, o)], closed=True))
    ax.add_patch(PathPatch(ring, fc=WALL, ec=INK, lw=0.8, zorder=5))
    # dashed room volume
    for x, y in plan:
        ax.plot(*zip(P(x, y, CUT, o), P(x, y, H, o)), color=GREY, lw=0.6, ls=(0, (1.5, 2)), zorder=6)
    top = [P(x, y, H, o) for x, y in plan] + [P(*plan[0], H, o)]
    ax.plot(*zip(*top), color=GREY, lw=0.6, ls=(0, (1.5, 2)), zorder=6)
    for w in windows:
        if w["front"]:
            draw_window(ax, w, o, z=8)

def draw_window(ax, w, o, z):
    (xa, ya), (xb, yb) = w["a"], w["b"]
    q = [(xa, ya, w["sill"]), (xb, yb, w["sill"]), (xb, yb, w["head"]), (xa, ya, w["head"])]
    ax.add_patch(Polygon(proj(q, o), closed=True, fc=GLASS, ec=GLASS_E, lw=1.1,
                         alpha=0.75 if w["front"] else 0.95, zorder=z))

def dim(ax, a, b, o, off):
    pa, pb = np.array(P(*a, 0, o)), np.array(P(*b, 0, o))
    ax.add_patch(FancyArrowPatch(pa + off, pb + off, arrowstyle="<|-|>", mutation_scale=8,
                                 lw=0.9, color=INK, zorder=9))
    return (pa + pb) / 2 + off

def label(ax, xy_text, anchor, title, lines, ha):
    ax.plot([xy_text[0] + (0.15 if ha == "left" else -0.15), anchor[0]],
            [xy_text[1] - 0.05, anchor[1]], color=GREY, lw=0.5, ls=(0, (1, 1.5)), zorder=10)
    ax.plot(*anchor, "o", ms=2.6, color=INK, zorder=11)
    ax.text(*xy_text, title, ha=ha, va="bottom", fontsize=9.5, color=INK, zorder=12)
    ax.text(xy_text[0], xy_text[1] - 0.05, "\n".join(lines), ha=ha, va="top",
            fontsize=7.6, color=GREY, linespacing=1.35, zorder=12)

def north(ax, xy, o_unused=None):
    x, y = xy
    v = np.array(P(0, 1)) - np.array(P(0, 0)); v = v / np.linalg.norm(v) * 0.55
    ax.add_patch(FancyArrowPatch((x, y), (x + v[0], y + v[1]), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.0, color=INK))
    ax.text(x + v[0] * 1.35, y + v[1] * 1.35, "N", ha="center", va="center", fontsize=9, weight="bold")

fig, ax = plt.subplots(figsize=(7.48, 7.75))
ax.set_aspect("equal"); ax.axis("off")

# ---------- (a) rectangular, bin C example: 2 south + 1 east
K = 0.95
oa = (4.1, 7.15)
W, D = 5.0, 5.0
plan_a = [(0, 0), (W, 0), (W, D), (0, D)]
win_a = [dict(a=(0.7, 0), b=(1.7, 0), sill=0.8, head=2.3, front=True),
         dict(a=(3.1, 0), b=(4.1, 0), sill=0.8, head=2.3, front=True),
         dict(a=(W, 1.6), b=(W, 3.4), sill=0.6, head=2.2, front=True)]
draw_room(ax, plan_a, oa, win_a)
ax.text(0.0, 10.1, "(a)  Rectangular rooms — bins A–E", fontsize=11.5, weight="bold", va="top")
w_mid = dim(ax, (0, -0.55), (W, -0.55), oa, np.array([0, -0.12]))
d_mid = dim(ax, (W + 0.55, 0), (W + 0.55, D), oa, np.array([0, -0.12]))
label(ax, (3.3, 9.25), P(0, D, H, oa), "Room height", ["2.70 m"], "right")
label(ax, (3.3, 8.4), P(0.0, D * 0.55, CUT, oa), "Wall", ["0.20 m, opaque, plan cut at 0.22 m"], "right")
label(ax, (3.3, 7.5), P(1.25, 3.25, 0, oa), "Analysis grid", ["0.25 m spacing, 0.80 m above floor,", "whole floor area"], "right")
label(ax, (3.3, 6.35), tuple(w_mid), "Room width W", ["4.0–6.0 m"], "right")
label(ax, (9.0, 9.4), P(W, 2.5, 2.2, oa), "Window", ["height 1.2–1.8 m, head ≤ 2.50 m", "width r·ℓ, r 0.15–0.60; glazing τ 0.64",
                                                   "1–2 per wall, south and east walls"], "left")
label(ax, (9.0, 8.05), P(W * 0.5, D * 0.5, H, oa), "Reflectances", ["ceiling 0.8, walls 0.5,", "floor 0.2, ground 0.1"], "left")
label(ax, (9.0, 7.05), tuple(d_mid), "Room depth D", ["4.0–6.0 m"], "left")
ax.text(9.0, 6.3, "Orientation", fontsize=9.5, va="bottom")
ax.text(9.0, 6.25, "0° (bins A–D);\n90°, 180°, 270° (bin E)", fontsize=7.6, color=GREY, va="top", linespacing=1.35)
north(ax, (0.3, 5.65))

# ---------- (b) L-shaped, bin F: cut-out at north-west corner
K = 0.66
ob = (3.85, 1.75)
W, D, CW, CD = 7.5, 7.5, 3.4, 3.4
plan_b = [(0, 0), (W, 0), (W, D), (CW, D), (CW, D - CD), (0, D - CD)]
win_b = [dict(a=(0, 1.0), b=(0, 3.0), sill=0.5, head=2.5, front=False),     # west
         dict(a=(4.6, D), b=(6.6, D), sill=0.3, head=2.4, front=False),     # north
         dict(a=(W, 2.2), b=(W, 4.4), sill=0.9, head=2.3, front=True)]      # east
draw_room(ax, plan_b, ob, win_b)
ax.text(0.0, 5.0, "(b)  L-shaped rooms — bin F", fontsize=11.5, weight="bold", va="top")
w_mid = dim(ax, (0, -0.55), (W, -0.55), ob, np.array([0, -0.12]))
d_mid = dim(ax, (W + 0.55, 0), (W + 0.55, D), ob, np.array([0, -0.12]))
label(ax, (3.3, 4.1), P(CW * 0.5, D - CD, H, ob), "Cut-out", ["width 0.20–0.60 W, depth 0.20–0.60 D,",
                                                                "capped at W − 2.5 m and D − 2.5 m;", "cycled over the four corners"], "right")
label(ax, (3.3, 2.75), P(CW, D - CD, CUT, ob), "Re-entrant corner", ["the only source of sky", "obstruction in the study"], "right")
label(ax, (3.3, 1.55), tuple(w_mid), "Room width W", ["5.0–12.0 m"], "right")
label(ax, (9.0, 4.1), P(5.6, D, 2.4, ob), "Window", ["height 1.2–2.4 m, sill 0.05–1.45 m,", "head ≤ 2.65 m, width ratio 0.20–0.80",
                                                   "1–3 per room, any wall ≥ 1.5 m"], "left")
label(ax, (9.0, 2.75), P(5.0, 2.0, 0, ob), "Analysis grid", ["as (a), following", "the L-shaped plan"], "left")
label(ax, (9.0, 1.55), tuple(d_mid), "Room depth D", ["5.0–12.0 m"], "left")

# ---------- footer + legend
ax.text(0.0, -0.3, "All rooms stand free on an unobstructed ground plane; sky occlusion comes only from the room's own form.",
        fontsize=7.8, color=GREY, va="top")
lx, ly = 0.0, -0.85
items = [("patch", GLASS, GLASS_E, "window"), ("patch", WALL, INK, "wall (plan cut)"),
         ("dot", "#6b5a4a", None, "analysis grid point"), ("line", GREY, None, "room volume")]
for kind, fc, ec, txt in items:
    if kind == "patch":
        ax.add_patch(Polygon([(lx, ly - 0.07), (lx + 0.45, ly - 0.07), (lx + 0.45, ly + 0.07), (lx, ly + 0.07)],
                             fc=fc, ec=ec, lw=0.6))
    elif kind == "dot":
        ax.plot(lx + 0.22, ly, "o", ms=3, color=fc)
    else:
        ax.plot([lx, lx + 0.45], [ly, ly], color=fc, lw=0.8, ls=(0, (1.5, 2)))
    ax.text(lx + 0.6, ly, txt, fontsize=8, va="center")
    lx += 0.6 + len(txt) * 0.105 + 0.45

ax.set_xlim(-0.1, 12.6); ax.set_ylim(-1.05, 10.25)
fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig2_room_types.{ext}"), dpi=400, bbox_inches="tight", pad_inches=0.05)
print("ok")
