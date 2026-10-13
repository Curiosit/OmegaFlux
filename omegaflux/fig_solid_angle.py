"""omegaflux.fig_solid_angle - Figure 4 of the paper: solid angle of a window seen from a sensor.

(a) the window split into two triangles projected onto the unit sphere (Eq. 2); (b) the same window in an
equal-area projection of the hemisphere, so the patch share of the disc equals Omega / 2 pi.
Usage:  python -m omegaflux.fig_solid_angle [--out figures/paper]
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
from matplotlib.patches import Polygon, Circle, PathPatch, FancyArrowPatch
from matplotlib.path import Path

C, S = np.cos(np.radians(30)), np.sin(np.radians(30))
FLOOR, WALL, GLASS, GLASS_E = "#efe3cc", "#8c8c8c", "#bcd3e6", "#5f86a8"
T1C, T2C = "#9fbfdc", "#d9a77a"          # triangle fills (blue / warm), print-safe
INK, GREY = "#111111", "#444444"
CUT = 0.22

def P(x, y, z=0.0):
    return np.array([(x + y) * C * 0.5, (y - x) * S * 0.5 + z * 0.5])

# ---------------- geometry (m) ----------------
W, D, H = 5.0, 5.0, 2.7
p = np.array([3.3, 2.3, 0.8])                        # sensor on the work plane
x0, x1, zs, zh = 1.1, 3.1, 0.9, 2.4                  # window on the north wall (y = D)
c1, c2, c3, c4 = (np.array([x0, D, zs]), np.array([x1, D, zs]),
                  np.array([x1, D, zh]), np.array([x0, D, zh]))
TRI1, TRI2 = (c1, c2, c3), (c1, c3, c4)

def omega_tri(a, b, c):
    r1, r2, r3 = a - p, b - p, c - p
    l1, l2, l3 = map(np.linalg.norm, (r1, r2, r3))
    num = np.dot(r1, np.cross(r2, r3))
    den = l1 * l2 * l3 + np.dot(r1, r2) * l3 + np.dot(r2, r3) * l1 + np.dot(r3, r1) * l2
    return abs(2 * np.arctan2(num, den))

O1, O2 = omega_tri(*TRI1), omega_tri(*TRI2)
OM = O1 + O2

def edge_samples(tri, n=60):
    pts = []
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append(a + (b - a) * t)
    return np.array(pts)

fig = plt.figure(figsize=(7.48, 3.55))
axA = fig.add_axes([0.0, 0.0, 0.64, 1.0]); axA.set_aspect("equal"); axA.axis("off")
axB = fig.add_axes([0.66, 0.06, 0.33, 0.84]); axB.set_aspect("equal"); axB.axis("off")

# ---------------- (a) room view ----------------
ax = axA
floor = [(0, 0), (W, 0), (W, D), (0, D)]
ax.add_patch(Polygon([P(x, y) for x, y in floor], fc=FLOOR, ec="none", zorder=1))
# back walls (north, west) at full height, light
north = [P(0, D, 0), P(W, D, 0), P(W, D, H), P(0, D, H)]
west = [P(0, 0, 0), P(0, D, 0), P(0, D, H), P(0, 0, H)]
for poly in (west, north):
    ax.add_patch(Polygon(poly, fc="#e6e6e6", ec=GREY, lw=0.6, zorder=2))
# window triangles
ax.add_patch(Polygon([P(*v) for v in TRI1], fc=T1C, ec=GLASS_E, lw=1.0, zorder=3))
ax.add_patch(Polygon([P(*v) for v in TRI2], fc=T2C, ec=GLASS_E, lw=1.0, zorder=3, alpha=0.9))
ax.add_patch(Polygon([P(*v) for v in (c1, c2, c3, c4)], fc="none", ec=INK, lw=1.1, zorder=4))
ax.plot(*zip(P(*c1), P(*c3)), color=INK, lw=0.8, ls=(0, (3, 2)), zorder=4)
for tri, lab in ((TRI1, "$T_1$"), (TRI2, "$T_2$")):
    ax.text(*P(*np.mean(tri, axis=0)), lab, ha="center", va="center", fontsize=9, zorder=5)
# front edges of floor (open cut) and room outline
for a, b in [((0, 0), (W, 0)), ((W, 0), (W, D))]:
    ax.plot(*zip(P(*a), P(*b)), color=INK, lw=1.0, zorder=2)
for a, b in [((W, 0), (W, D)), ((0, 0), (W, 0))]:
    ax.plot(*zip(P(a[0], a[1], H), P(b[0], b[1], H)), color=GREY, lw=0.6, ls=(0, (1.5, 2)), zorder=2)
for x, y in [(W, 0), (W, D)]:
    ax.plot(*zip(P(x, y, 0), P(x, y, H)), color=GREY, lw=0.6, ls=(0, (1.5, 2)), zorder=2)
ax.plot(*zip(P(0, 0, 0), P(0, 0, H)), color=GREY, lw=0.6, zorder=2)

# unit sphere around p (drawn at radius R for legibility)
R = 1.05
rs = 0.5 * R / np.sqrt(2 / 3)
ctr = P(*p)
ax.add_patch(Circle(ctr, rs, fc="white", ec=GREY, lw=0.8, alpha=0.55, zorder=6))
th = np.linspace(0, 2 * np.pi, 200)
eq = np.array([P(*(p + R * np.array([np.cos(t), np.sin(t), 0]))) for t in th])
ax.plot(eq[:, 0], eq[:, 1], color=GREY, lw=0.5, ls=(0, (2, 2)), zorder=6)
# projected triangles on the sphere
for tri, col in ((TRI1, T1C), (TRI2, T2C)):
    e = edge_samples(tri)
    dirs = (e - p) / np.linalg.norm(e - p, axis=1)[:, None]
    pts = np.array([P(*(p + R * d)) for d in dirs])
    ax.add_patch(Polygon(pts, fc=col, ec=GLASS_E, lw=0.7, zorder=8))
# rays from p to the window corners
for v in (c1, c2, c3):
    ax.plot(*zip(ctr, P(*v)), color=INK, lw=0.7, zorder=7)
ax.plot(*zip(ctr, P(*c4)), color=GREY, lw=0.5, ls=(0, (2, 1.5)), zorder=7)
ax.plot(*ctr, "o", ms=4, color=INK, zorder=10)
ax.text(ctr[0] + 0.07, ctr[1] - 0.06, "$p$", fontsize=10, ha="left", va="top", zorder=10)
# r-vectors for T1
for v, lab, off in ((c1, "$\\mathbf{r}_1$", (-0.13, 0.0)), (c2, "$\\mathbf{r}_2$", (0.12, -0.04)),
                    (c3, "$\\mathbf{r}_3$", (0.12, 0.02))):
    m = P(*(p + (v - p) * 0.6))
    ax.text(m[0] + off[0], m[1] + off[1], lab, fontsize=9, ha="center", va="center", zorder=10,
            bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.8))
# labels with leaders
def lab(xy_t, anchor, title, sub, ha):
    ax.plot([xy_t[0], anchor[0]], [xy_t[1], anchor[1]], color=GREY, lw=0.5, ls=(0, (1, 1.5)), zorder=11)
    ax.text(xy_t[0], xy_t[1] + 0.02, title, fontsize=8.6, ha=ha, va="bottom", color=INK)
    if sub:
        ax.text(xy_t[0], xy_t[1] - 0.02, sub, fontsize=7.2, ha=ha, va="top", color=GREY, linespacing=1.3)
ax.text(ctr[0] + rs * 1.15, ctr[1] - rs * 0.75, "unit sphere around $p$:\neach triangle projects\nto a spherical patch",
        fontsize=7.4, color=GREY, ha="left", va="center", linespacing=1.3, zorder=11,
        bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.85))
ax.text(-0.05, 3.05, "(a)", fontsize=11, weight="bold", va="top")
ax.set_xlim(-0.1, 5.0); ax.set_ylim(-1.3, 3.1)

# ---------------- (b) view from the sensor: equal-area hemisphere ----------------
ax = axB
def fish(d):
    d = d / np.linalg.norm(d)
    theta = np.arccos(np.clip(d[2], -1, 1))
    az = np.arctan2(d[0], d[1])
    r = np.sqrt(2) * np.sin(theta / 2)
    return np.array([r * np.sin(az), r * np.cos(az)])
ax.add_patch(Circle((0, 0), 1.0, fc="#f6f6f6", ec=INK, lw=0.9))
for el in (30, 60):
    r = np.sqrt(2) * np.sin(np.radians(90 - el) / 2)
    ax.add_patch(Circle((0, 0), r, fc="none", ec="#bbbbbb", lw=0.5, ls=(0, (2, 2))))
    ax.text(0.03, -r + 0.02, f"{el}°", fontsize=6.5, color=GREY, va="bottom")
ax.plot([-1, 1], [0, 0], color="#cccccc", lw=0.4); ax.plot([0, 0], [-1, 1], color="#cccccc", lw=0.4)
for t, (x, y) in {"N": (0, 1.1), "E": (1.1, 0), "S": (0, -1.1), "W": (-1.1, 0)}.items():
    ax.text(x, y, t, ha="center", va="center", fontsize=8, weight="bold")
for tri, col in ((TRI1, T1C), (TRI2, T2C)):
    pts = np.array([fish(e - p) for e in edge_samples(tri)])
    ax.add_patch(Polygon(pts, fc=col, ec=GLASS_E, lw=0.7))
ax.plot(0, 0, "o", ms=3, color=INK)
ax.text(0, -1.42, f"$\\Omega = \\Omega_{{T_1}} + \\Omega_{{T_2}} = {OM:.2f}$ sr\n"
        f"$\\Omega / 2\\pi = {OM / (2 * np.pi):.3f}$",
        ha="center", va="top", fontsize=8.2, linespacing=1.5)
ax.text(-1.32, 1.6, "(b)", fontsize=11, weight="bold", va="top")
ax.text(-1.05, 1.6, "Hemisphere above $p$,\nequal-area projection", fontsize=8, va="top", color=GREY)
ax.set_xlim(-1.35, 1.3); ax.set_ylim(-1.95, 1.62)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig4_solid_angle.{ext}"), dpi=400, bbox_inches="tight", pad_inches=0.05)
print(f"Omega1={O1:.4f} Omega2={O2:.4f} total={OM:.4f}")
