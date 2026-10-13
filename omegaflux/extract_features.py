from __future__ import annotations

import json
import math
import argparse
import os
import platform
import re
import sys
import time

import numpy as np
import pandas as pd

TWO_PI = 2.0 * math.pi

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 42

SITE_LATITUDE_DEG = 52.17
SITE_NAME = "Warsaw Okecie, PL"

SUN_DECLINATIONS_DEG = [-23.45, -11.5, 0.0, 11.5, 23.45]
SUN_HOUR_ANGLE_RANGE = (-120, 121, 10)
SUN_MIN_ELEVATION_DEG = 2.0

ROOM_HEIGHT_MODE = "fixed"
ROOM_HEIGHT_M = 3.0
ROOM_HEIGHT_INFER_MARGIN_M = 0.6

VISIBILITY_SAMPLES = 1
CORNER_PENUMBRA_RADIUS_M = 0.40

APPLY_LEGACY_TARGET_FILTER = False
TARGET_SUM_TOLERANCE = 0.5

OUTPUT_DIR = os.path.join("data", "processed")   # where train.py reads it (was data_processed/)

DATASETS = [
    (True, "data/raw/study-ml-rect.xlsx",       "rect",         "Rectangular"),
    (True, "data/raw/study-ml-rotated.xlsx",    "rect_rotated", "Rotated"),
    (True, "data/raw/study-ml-shapes.csv",      "randomized",   "Randomized L-Shape"),
]

# =============================================================================
# FEATURE SCHEMA
# =============================================================================

from omegaflux.features import FEATURES_ALL as FEATURES_INDEPENDENT  # single source of truth

assert len(FEATURES_INDEPENDENT) == 26

# =============================================================================
# GEOMETRIC PRIMITIVES
# =============================================================================

def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    elif ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"unsupported extension: {path}")
    df = df.dropna(how="all")
    df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c, inplace=True)

    if "Sensor_location_matrix" not in df.columns and "Sensor_location_matrix X" in df.columns:
        z = "0.8"
        combo = []
        for xs, ys in zip(df["Sensor_location_matrix X"], df["Sensor_location_matrix Y"]):
            xt = [t.strip() for t in str(xs).split("X") if t.strip() not in ("", "nan")]
            yt = [t.strip() for t in str(ys).split("Y") if t.strip() not in ("", "nan")]
            combo.append("".join(f"{{{a}, {b}, {z}}}" for a, b in zip(xt, yt)))
        df["Sensor_location_matrix"] = combo
    return df

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 - edge0 == 0:
        return 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)

def dist_pt_seg_2d(px, py, ax, ay, bx, by):
    ab_x, ab_y = bx - ax, by - ay
    ap_x, ap_y = px - ax, py - ay
    ab_len_sq = ab_x * ab_x + ab_y * ab_y
    if ab_len_sq == 0:
        return math.hypot(px - ax, py - ay), ax, ay
    t = max(0.0, min(1.0, (ap_x * ab_x + ap_y * ab_y) / ab_len_sq))
    proj_x = ax + t * ab_x
    proj_y = ay + t * ab_y
    return math.hypot(px - proj_x, py - proj_y), proj_x, proj_y

def dist_pt_ray_seg(px, py, ax, ay, bx, by):
    ab_x, ab_y = bx - ax, by - ay
    ap_x, ap_y = px - ax, py - ay
    ab_len_sq = ab_x * ab_x + ab_y * ab_y
    if ab_len_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (ap_x * ab_x + ap_y * ab_y) / ab_len_sq))
    return math.hypot(px - (ax + t * ab_x), py - (ay + t * ab_y))

def lines_intersect_2d(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, tol=1e-5):
    if (abs(p1x - p3x) < tol and abs(p1y - p3y) < tol) or \
       (abs(p1x - p4x) < tol and abs(p1y - p4y) < tol) or \
       (abs(p2x - p3x) < tol and abs(p2y - p3y) < tol) or \
       (abs(p2x - p4x) < tol and abs(p2y - p4y) < tol):
        return False

    def ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

    return (ccw(p1x, p1y, p3x, p3y, p4x, p4y) != ccw(p2x, p2y, p3x, p3y, p4x, p4y)
            and ccw(p1x, p1y, p2x, p2y, p3x, p3y) != ccw(p1x, p1y, p2x, p2y, p4x, p4y))

def is_same_wall(w1, w2, tol=1e-3):
    ax, ay = w1[0]
    bx, by = w1[1]
    cx, cy = w2[0]
    dx, dy = w2[1]
    return (math.hypot(ax - cx, ay - cy) < tol and math.hypot(bx - dx, by - dy) < tol) or \
           (math.hypot(ax - dx, ay - dy) < tol and math.hypot(bx - cx, by - cy) < tol)

def ray_intersects_tri_3d(ox, oy, oz, dx, dy, dz, t0, t1, t2, eps=1e-6):
    e1x, e1y, e1z = t1[0] - t0[0], t1[1] - t0[1], t1[2] - t0[2]
    e2x, e2y, e2z = t2[0] - t0[0], t2[1] - t0[1], t2[2] - t0[2]
    hx, hy, hz = dy * e2z - dz * e2y, dz * e2x - dx * e2z, dx * e2y - dy * e2x
    a = e1x * hx + e1y * hy + e1z * hz
    if -eps < a < eps:
        return False, 0.0, 0.0
    f = 1.0 / a
    sx, sy, sz = ox - t0[0], oy - t0[1], oz - t0[2]
    u = f * (sx * hx + sy * hy + sz * hz)
    if u < 0.0 or u > 1.0:
        return False, 0.0, 0.0
    qx, qy, qz = sy * e1z - sz * e1y, sz * e1x - sx * e1z, sx * e1y - sy * e1x
    v = f * (dx * qx + dy * qy + dz * qz)
    if v < 0.0 or u + v > 1.0:
        return False, 0.0, 0.0
    t = f * (e2x * qx + e2y * qy + e2z * qz)
    if t > eps:
        return True, ox + t * dx, oy + t * dy
    return False, 0.0, 0.0

def solid_angle_tri(px, py, pz, a, b, c):
    r1x, r1y, r1z = a[0] - px, a[1] - py, a[2] - pz
    r2x, r2y, r2z = b[0] - px, b[1] - py, b[2] - pz
    r3x, r3y, r3z = c[0] - px, c[1] - py, c[2] - pz
    l1 = math.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
    l2 = math.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
    l3 = math.sqrt(r3x * r3x + r3y * r3y + r3z * r3z)
    if l1 == 0 or l2 == 0 or l3 == 0:
        return 0.0
    cx, cy, cz = (r2y * r3z - r2z * r3y,
                  r2z * r3x - r2x * r3z,
                  r2x * r3y - r2y * r3x)
    scalar_triple = r1x * cx + r1y * cy + r1z * cz
    den = (l1 * l2 * l3
           + (r1x * r2x + r1y * r2y + r1z * r2z) * l3
           + (r2x * r3x + r2y * r3y + r2z * r3z) * l1
           + (r3x * r1x + r3y * r1y + r3z * r1z) * l2)
    if den == 0:
        return 0.0
    return abs(2.0 * math.atan2(scalar_triple, den))

def solid_angle_polygon(px, py, pz, corners):
    if len(corners) < 3:
        return 0.0
    omega = 0.0
    for i in range(1, len(corners) - 1):
        omega += solid_angle_tri(px, py, pz, corners[0], corners[i], corners[i + 1])
    return omega

def ray_cast_2d(ox, oy, dx, dy, segments, l_char):
    min_dist = float("inf")
    for w in segments:
        vx, vy = w[1][0] - w[0][0], w[1][1] - w[0][1]
        qx, qy = ox - w[0][0], oy - w[0][1]
        cross = dx * vy - dy * vx
        if abs(cross) < (l_char * 1e-6):
            continue
        u = (vx * qy - vy * qx) / cross
        t = (dx * qy - dy * qx) / cross
        if u > (l_char * 1e-5) and -1e-4 <= t <= 1.0001:
            if u < min_dist:
                min_dist = u
    return min_dist if min_dist != float("inf") else l_char * 2.0

def generate_annual_sun_vectors(latitude_deg: float):
    rays = []
    lat = math.radians(latitude_deg)
    for dec_deg in SUN_DECLINATIONS_DEG:
        dec = math.radians(dec_deg)
        for hour_angle_deg in range(*SUN_HOUR_ANGLE_RANGE):
            ha = math.radians(hour_angle_deg)
            sin_elev = (math.sin(lat) * math.sin(dec)
                        + math.cos(lat) * math.cos(dec) * math.cos(ha))
            elev = math.asin(max(-1.0, min(1.0, sin_elev)))
            if elev > math.radians(SUN_MIN_ELEVATION_DEG):
                dx = -math.cos(dec) * math.sin(ha)
                dy = (math.sin(dec) * math.cos(lat)
                      - math.cos(dec) * math.sin(lat) * math.cos(ha))
                dz = math.sin(elev)
                n = math.sqrt(dx * dx + dy * dy + dz * dz)
                rays.append((dx / n, dy / n, dz / n))
    return rays

def sort_quad_corners_perimeter(corners):
    cx = sum(c[0] for c in corners) / 4.0
    cy = sum(c[1] for c in corners) / 4.0
    cz = sum(c[2] for c in corners) / 4.0
    v1 = (corners[1][0] - corners[0][0], corners[1][1] - corners[0][1], corners[1][2] - corners[0][2])
    v2 = (corners[2][0] - corners[0][0], corners[2][1] - corners[0][1], corners[2][2] - corners[0][2])
    nx, ny, nz = (v1[1] * v2[2] - v1[2] * v2[1],
                  v1[2] * v2[0] - v1[0] * v2[2],
                  v1[0] * v2[1] - v1[1] * v2[0])
    n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n_len < 1e-6:
        v2 = (corners[3][0] - corners[0][0], corners[3][1] - corners[0][1], corners[3][2] - corners[0][2])
        nx, ny, nz = (v1[1] * v2[2] - v1[2] * v2[1],
                      v1[2] * v2[0] - v1[0] * v2[2],
                      v1[0] * v2[1] - v1[1] * v2[0])
        n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = (nx / n_len, ny / n_len, nz / n_len) if n_len > 1e-6 else (0.0, 0.0, 1.0)
    v1_len = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    ux, uy, uz = (v1[0] / v1_len, v1[1] / v1_len, v1[2] / v1_len) if v1_len > 1e-6 else (1.0, 0.0, 0.0)
    vx, vy, vz = ny * uz - nz * uy, nz * ux - nx * uz, nx * uy - ny * ux
    angles_pts = []
    for c in corners:
        relx, rely, relz = c[0] - cx, c[1] - cy, c[2] - cz
        u_val = relx * ux + rely * uy + relz * uz
        v_val = relx * vx + rely * vy + relz * vz
        angles_pts.append((math.atan2(v_val, u_val), c))
    angles_pts.sort(key=lambda x: x[0])
    return [item[1] for item in angles_pts]

def point_in_polygon(px, py, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        pi, pj = poly[i], poly[j]
        if ((pi[1] > py) != (pj[1] > py)) and \
           (px < (pj[0] - pi[0]) * (py - pi[1]) / (pj[1] - pi[1] + 1e-9) + pi[0]):
            inside = not inside
        j = i
    return inside

# =============================================================================
# GEOMETRY RECONSTRUCTION
# =============================================================================

def parse_room_geometry(row):
    if "Room_corners" in row.index and pd.notna(row.get("Room_corners")):
        room_corners = [
            (float(p.split(",")[0]), float(p.split(",")[1]))
            for p in re.findall(r"\{([^}]+)\}", str(row["Room_corners"]))
        ]
        centers = [
            (float(c.split(",")[0]), float(c.split(",")[1]), float(c.split(",")[2]))
            for c in re.findall(r"\{([^}]+)\}", str(row["Window_centerpoints"]))
        ]
        widths = [float(w.strip()) for w in str(row["Window_widths"]).split(";") if w.strip()]
        heights = [float(h.strip()) for h in str(row["Window_heights"]).split(";") if h.strip()]
        return room_corners, list(zip(centers, widths, heights))

    room_width = float(row["Room-width"])
    room_depth = float(row["Room-depth"])
    orientation_deg = float(row.get("Orientation", row.get("Orientation_deg", 0.0)) or 0.0)
    rad = math.radians(orientation_deg)
    c, s = math.cos(rad), math.sin(rad)

    def rot_pt(px, py):
        return px * c - py * s, px * s + py * c

    room_corners = [rot_pt(x, y) for x, y in
                    [(0.0, 0.0), (room_width, 0.0), (room_width, room_depth), (0.0, room_depth)]]

    specs = []
    for prefix in ("Front_Win1", "Front_Win2"):
        w = float(row.get(f"{prefix}_Width_m", 0.0) or 0.0)
        if w > 0:
            ch = float(row.get(f"{prefix}_Center_H_m", 0.0) or 0.0)
            cv = float(row.get(f"{prefix}_Center_V_m", 0.0) or 0.0)
            h = float(row.get(f"{prefix}_Height_m", 0.0) or 0.0)
            rx, ry = rot_pt(ch, 0.0)
            specs.append(((rx, ry, cv), w, h))
    for prefix in ("Side_Win1", "Side_Win2"):
        w = float(row.get(f"{prefix}_Width_m", 0.0) or 0.0)
        if w > 0:
            ch = float(row.get(f"{prefix}_Center_H_m", 0.0) or 0.0)
            cv = float(row.get(f"{prefix}_Center_V_m", 0.0) or 0.0)
            h = float(row.get(f"{prefix}_Height_m", 0.0) or 0.0)
            rx, ry = rot_pt(room_width, ch)
            specs.append(((rx, ry, cv), w, h))
    return room_corners, specs

def build_apertures(room_corners, window_specs, wall_segments):
    apertures = []
    for center_pt, w, h in window_specs:
        min_dist, best_w = float("inf"), None
        for seg in wall_segments:
            d, _, _ = dist_pt_seg_2d(center_pt[0], center_pt[1],
                                     seg[0][0], seg[0][1], seg[1][0], seg[1][1])
            if d < min_dist:
                min_dist, best_w = d, seg
        if best_w is None:
            continue

        vx, vy = best_w[1][0] - best_w[0][0], best_w[1][1] - best_w[0][1]
        length = math.hypot(vx, vy)
        if length == 0:
            continue
        nx, ny = -vy / length, vx / length
        if point_in_polygon(center_pt[0] + nx * 0.1, center_pt[1] + ny * 0.1, room_corners):
            nx, ny = -nx, -ny
        rx, ry = vx / length, vy / length

        z1, z2 = center_pt[2] - h / 2.0, center_pt[2] + h / 2.0
        corners_3d = [
            (center_pt[0] - rx * (w / 2), center_pt[1] - ry * (w / 2), z1),
            (center_pt[0] + rx * (w / 2), center_pt[1] + ry * (w / 2), z1),
            (center_pt[0] + rx * (w / 2), center_pt[1] + ry * (w / 2), z2),
            (center_pt[0] - rx * (w / 2), center_pt[1] - ry * (w / 2), z2),
        ]
        sorted_c = sort_quad_corners_perimeter(corners_3d)
        triangles = [(sorted_c[0], sorted_c[1], sorted_c[2]),
                     (sorted_c[0], sorted_c[2], sorted_c[3])]
        occluding_walls = [seg for seg in wall_segments if not is_same_wall(seg, best_w)]

        cx = center_pt[0] + nx * 0.05
        cy = center_pt[1] + ny * 0.05
        base_ang = math.atan2(ny, nx)
        sky_rays, strikes = 0, 0
        for deg in range(-90, 91, 15):
            sky_rays += 1
            ex = cx + math.cos(base_ang + math.radians(deg)) * 50.0
            ey = cy + math.sin(base_ang + math.radians(deg)) * 50.0
            for occ in occluding_walls:
                if lines_intersect_2d(cx, cy, ex, ey, occ[0][0], occ[0][1], occ[1][0], occ[1][1]):
                    strikes += 1
                    break
        sky_ratio = max(0.05, 1.0 - (strikes / sky_rays)) if sky_rays else 1.0

        if VISIBILITY_SAMPLES <= 1:
            sample_pts = [(sum(c[0] for c in sorted_c) / 4.0,
                           sum(c[1] for c in sorted_c) / 4.0)]
        else:
            n = VISIBILITY_SAMPLES
            sample_pts = []
            for i in range(n):
                f = (i + 0.5) / n - 0.5
                sample_pts.append((center_pt[0] + rx * w * f, center_pt[1] + ry * w * f))

        apertures.append({
            "corners": sorted_c,
            "normal": (nx, ny),
            "center": center_pt,
            "width": w,
            "height": h,
            "triangles": triangles,
            "occluding_walls": occluding_walls,
            "sky_vis_ratio": sky_ratio,
            "sample_pts": sample_pts,
        })
    return apertures

# =============================================================================
# PER-SENSOR FEATURE EXTRACTION
# =============================================================================

def extract_sensor_features(sx, sy, sz, room_corners, wall_segments, apertures,
                            l_char, room_height, sun_vectors, is_ccw):
    n_sun = len(sun_vectors)
    total_omega = 0.0
    raw_total_omega = 0.0
    south_w = east_w = west_w = sky_w = 0.0
    total_area = area_cos_theta = 0.0
    dmin_sq_list = []
    clearance_data = []
    visible_count = 0
    sum_visible_dist = 0.0
    max_win_omega = 0.0
    dominant_normal = dominant_center = None
    total_angular_spread = 0.0
    min_occluder_distance = float("inf")

    total_len = weighted_dist_sum = 0.0
    variances = []
    for seg in wall_segments:
        d, _, _ = dist_pt_seg_2d(sx, sy, seg[0][0], seg[0][1], seg[1][0], seg[1][1])
        length = math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1])
        weighted_dist_sum += d * length
        total_len += length
        variances.append((d, length))
    smooth_mean_depth = (weighted_dist_sum / total_len) if total_len > 0 else 0.0
    smooth_depth_stddev = (math.sqrt(sum(l * (d - smooth_mean_depth) ** 2 for d, l in variances) / total_len)
                           if total_len > 0 else 0.0)

    solar_hits = 0
    for rdx, rdy, rdz in sun_vectors:
        hit = False
        for ap in apertures:
            for t0, t1, t2 in ap["triangles"]:
                ok, hx, hy = ray_intersects_tri_3d(sx, sy, sz, rdx, rdy, rdz, t0, t1, t2)
                if not ok:
                    continue
                blocked = False
                for occ in ap["occluding_walls"]:
                    if lines_intersect_2d(sx, sy, hx - rdx * 0.05, hy - rdy * 0.05,
                                          occ[0][0], occ[0][1], occ[1][0], occ[1][1]):
                        blocked = True
                        break
                if not blocked:
                    hit = True
                    break
            if hit:
                break
        if hit:
            solar_hits += 1

    for ap in apertures:
        corners = ap["corners"]
        wnx, wny = ap["normal"]
        cx, cy, cz = ap["center"]

        unblocked = 0.0
        for tx, ty in ap["sample_pts"]:
            dist = math.hypot(sx - tx, sy - ty)
            if dist > 0.05:
                sample_x = tx + ((sx - tx) / dist) * 0.05
                sample_y = ty + ((sy - ty) / dist) * 0.05
            else:
                sample_x, sample_y = tx, ty
            blocked = False
            for occ in ap["occluding_walls"]:
                if lines_intersect_2d(sx, sy, sample_x, sample_y,
                                      occ[0][0], occ[0][1], occ[1][0], occ[1][1]):
                    blocked = True
                    occ_d, _, _ = dist_pt_seg_2d(sx, sy, occ[0][0], occ[0][1], occ[1][0], occ[1][1])
                    if occ_d < min_occluder_distance:
                        min_occluder_distance = occ_d
                    break
            score = 0.0 if blocked else 1.0
            if not blocked:
                for rc in room_corners:
                    d_vert = dist_pt_ray_seg(rc[0], rc[1], sx, sy, sample_x, sample_y)
                    if d_vert < CORNER_PENUMBRA_RADIUS_M:
                        score = max(0.0, 1.0 - smoothstep(
                            0.0, CORNER_PENUMBRA_RADIUS_M,
                            CORNER_PENUMBRA_RADIUS_M - d_vert))
            unblocked += score
        unblocked /= len(ap["sample_pts"])

        w_k = max(0.0, smoothstep(0.0, 1.0, unblocked))
        raw_omega = solid_angle_polygon(sx, sy, sz, corners)
        raw_total_omega += raw_omega
        win_omega = raw_omega * w_k
        total_omega += win_omega

        d = max(0.1, math.hypot(cx - sx, cy - sy))

        if w_k > 0.05:
            visible_count += 1
            sum_visible_dist += d
            if win_omega > max_win_omega:
                max_win_omega = win_omega
                dominant_normal = (wnx, wny)
                dominant_center = (cx, cy)
            angles = [math.atan2(c[1] - sy, c[0] - sx) for c in corners]
            spread = max(
                [abs(angles[i] - angles[j]) if abs(angles[i] - angles[j]) <= math.pi
                 else 2 * math.pi - abs(angles[i] - angles[j])
                 for i in range(4) for j in range(i + 1, 4)], default=0.0)
            total_angular_spread += spread * w_k

        if w_k > 0.01:
            south_w += win_omega * max(0.0, -wny)
            east_w += win_omega * max(0.0, wnx)
            west_w += win_omega * max(0.0, -wnx)
            sky_w += raw_omega * w_k * ap["sky_vis_ratio"]

            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            zs = [c[2] for c in corners]
            w_width = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
            h = max(max(zs) - min(zs), 0.1)
            win_area = w_width * h * w_k

            d3d = max(math.sqrt((cx - sx) ** 2 + (cy - sy) ** 2 + (cz - sz) ** 2), 1e-3)
            area_cos_theta += (d / d3d) * win_area
            total_area += win_area
            dmin_sq_list.append((w_width / 2.0) ** 2 + d ** 2
                                + max(0.0, abs(cz - sz) - h / 2.0) ** 2)
            clearance_data.append({
                "sill": sz - min(zs),
                "header": max(zs) - sz,
                "weight": win_area / (d ** 3 + 1e-3),
            })

    if dominant_normal is not None:
        dnx, dny = dominant_normal
        dcx, dcy = dominant_center
        plane_dist = abs((sx - dcx) * dnx + (sy - dcy) * dny)
        h_offset = abs(-(sx - dcx) * dny + (sy - dcy) * dnx)
        ray_fwd = ray_cast_2d(sx, sy, -dnx, -dny, wall_segments, l_char)
        ray_bwd = ray_cast_2d(sx, sy, dnx, dny, wall_segments, l_char)
        ray_lft = ray_cast_2d(sx, sy, -dny, dnx, wall_segments, l_char)
        ray_rgt = ray_cast_2d(sx, sy, dny, -dnx, wall_segments, l_char)
    else:
        plane_dist = h_offset = l_char * 2.0
        ray_fwd = ray_bwd = ray_lft = ray_rgt = l_char * 2.0

    min_opaque = float("inf")
    min_inner_corner = float("inf")
    n_c = len(room_corners)
    for i in range(n_c):
        p_prev = room_corners[(i - 1) % n_c]
        p_curr = room_corners[i]
        p_next = room_corners[(i + 1) % n_c]
        dw, _, _ = dist_pt_seg_2d(sx, sy, p_curr[0], p_curr[1], p_next[0], p_next[1])
        if dw < min_opaque:
            min_opaque = dw
        v1x, v1y = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
        v2x, v2y = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
        cross = v1x * v2y - v1y * v2x
        reflex = (cross < 0) if is_ccw else (cross > 0)
        if reflex:
            dcorner = math.hypot(sx - p_curr[0], sy - p_curr[1])
            if dcorner < min_inner_corner:
                min_inner_corner = dcorner

    clip = lambda v, lo=0.0, hi=1.0: float(np.clip(v, lo, hi))
    cos_theta = (area_cos_theta / total_area) if total_area else 0.0
    w_sum = sum(i["weight"] for i in clearance_data)

    omega_saturated = total_omega > TWO_PI * (1.0 + 1e-9)

    feats = {
        "_omega_saturated":           bool(omega_saturated),
        "Total_Omega":                clip(total_omega / TWO_PI),
        "Global_Cos_Theta":           clip(cos_theta),
        "Global_Sill_Clearance":      clip((sum(i["sill"] * i["weight"] for i in clearance_data) / w_sum / room_height) if w_sum > 0 else 0.0, lo=-1.0),  # signed since v1.1.0
        "Global_Header_Clearance":    clip((sum(i["header"] * i["weight"] for i in clearance_data) / w_sum / room_height) if w_sum > 0 else 0.0),
        "Global_inv_dmin2":           float((1.0 / max(min(dmin_sq_list), 1e-4) if dmin_sq_list else 0.0) * (l_char ** 2)),
        "Opaque_Wall_dmin":           clip(max(0.05, min_opaque) / l_char),
        "Inner_Corner_Dmin":          clip((min_inner_corner if min_inner_corner != float("inf") else l_char) / l_char),
        "Outer_Corner_Dmin":          clip(max(0.01, min(math.hypot(sx - c[0], sy - c[1]) for c in room_corners)) / l_char),
        "South_Weighted_Omega":       clip(south_w / TWO_PI),
        "East_Weighted_Omega":        clip(east_w / TWO_PI),
        "West_Weighted_Omega":        clip(west_w / TWO_PI),
        "Annual_Direct_Sun_Fraction": clip(solar_hits / n_sun if n_sun else 0.0),
        "Smooth_Mean_Depth":          clip(smooth_mean_depth / l_char),
        "Smooth_Depth_StdDev":        clip(smooth_depth_stddev / l_char),
        "Min_Occluder_Distance":      clip((min_occluder_distance if min_occluder_distance != float("inf") else l_char * 2.0) / l_char),
        "Window_Angular_Spread":      clip(total_angular_spread / TWO_PI),
        "Window_Sky_Weighted_Omega":  clip(sky_w / TWO_PI),
        "Occlusion_Multiplier":       clip((total_omega / raw_total_omega) if raw_total_omega else 0.0),
        "Visible_Window_Count":       clip(visible_count / 10.0),
        "Mean_Window_Distance":       clip((sum_visible_dist / visible_count if visible_count else l_char * 2.0) / l_char),
        "Dominant_Win_Plane_Dist":    clip(plane_dist / l_char),
        "Dominant_Win_H_Offset":      clip(h_offset / l_char),
        "Dominant_Win_Ray_Forward":   clip(ray_fwd / l_char),
        "Dominant_Win_Ray_Backward":  clip(ray_bwd / l_char),
        "Dominant_Win_Ray_Left":      clip(ray_lft / l_char),
        "Dominant_Win_Ray_Right":     clip(ray_rgt / l_char),
    }

    return feats

# =============================================================================
# DATASET PROCESSOR
# =============================================================================

def process_dataset(df_raw, suffix, label, sun_vectors, stats):
    df = df_raw.dropna(subset=["Iteration_Name"]).copy()
    n_rooms = len(df)
    records = []
    room_times = []

    print(f"\nExploding {n_rooms} {label} rooms -> per-sensor records...")

    dup = df["Iteration_Name"].duplicated().sum()
    if dup:
        print(f"  [WARN] {dup} duplicate Iteration_Name values within this dataset. ")

    for count, (_, row) in enumerate(df.iterrows(), 1):
        if count % 25 == 0 or count == n_rooms:
            sys.stdout.write(f"\r  room {count}/{n_rooms} ({count / n_rooms * 100:5.1f}%)")
            sys.stdout.flush()

        t_room = time.perf_counter()

        udi_low = [float(x) for x in str(row.get("raw_UDI_low", "")).split(";") if x.strip()]
        udi_up = [float(x) for x in str(row.get("raw_UDI_up", "")).split(";") if x.strip()]
        sensors = re.findall(r"\{([^}]+)\}", str(row.get("Sensor_location_matrix", "")))

        if not sensors or not (len(sensors) == len(udi_low) == len(udi_up)):
            stats["rooms_malformed"] += 1
            continue

        try:
            room_corners, window_specs = parse_room_geometry(row)
        except Exception as exc:
            stats["rooms_malformed"] += 1
            print(f"\n  [WARN] geometry parse failed for {row['Iteration_Name']}: {exc}")
            continue

        if len(room_corners) < 3:
            stats["rooms_malformed"] += 1
            continue

        wall_segments = [(room_corners[i], room_corners[(i + 1) % len(room_corners)])
                         for i in range(len(room_corners))]
        min_x = min(p[0] for p in room_corners)
        min_y = min(p[1] for p in room_corners)
        bbox_w = max(p[0] for p in room_corners) - min_x
        bbox_d = max(p[1] for p in room_corners) - min_y
        l_char = max(1.0, math.hypot(bbox_w, bbox_d))
        perimeter = sum(math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1])
                        for s in wall_segments)
        orientation_deg = float(row.get("Orientation",
                                        row.get("Orientation_deg", 0.0)) or 0.0)

        apertures = build_apertures(room_corners, window_specs, wall_segments)
        total_win_area = sum(ap["width"] * ap["height"] for ap in apertures)

        if ROOM_HEIGHT_MODE == "fixed":
            room_height = ROOM_HEIGHT_M
            tops = [max(c[2] for c in ap["corners"]) for ap in apertures]
            if tops and max(tops) > room_height + 1e-6:
                stats["rooms_window_above_ceiling"] += 1
        else:
            tops = [max(c[2] for c in ap["corners"]) for ap in apertures]
            room_height = max(tops) + ROOM_HEIGHT_INFER_MARGIN_M if tops else ROOM_HEIGHT_M
        room_height = max(room_height, 0.1)

        signed_area = sum(
            room_corners[i][0] * room_corners[(i + 1) % len(room_corners)][1]
            - room_corners[(i + 1) % len(room_corners)][0] * room_corners[i][1]
            for i in range(len(room_corners))) / 2.0
        is_ccw = signed_area > 0

        group_id = f"{suffix}:{row['Iteration_Name']}"

        for i, sensor_str in enumerate(sensors):
            lo = float(np.clip(udi_low[i], 0.0, 100.0))
            up = float(np.clip(udi_up[i], 0.0, 100.0))

            if not (np.isfinite(lo) and np.isfinite(up)):
                stats["sensors_nan"] += 1
                continue
            if lo + up > 100.0 + TARGET_SUM_TOLERANCE:
                stats["sensors_sum_gt_100"] += 1
                continue

            useful = 100.0 - lo - up
            if useful <= 0.0:
                stats["legacy_would_drop_zero_useful"] += 1
            if lo == 0.0 and up == 0.0:
                stats["legacy_would_drop_full_useful"] += 1
            if APPLY_LEGACY_TARGET_FILTER and (useful <= 0.0 or (lo == 0.0 and up == 0.0)):
                stats["sensors_legacy_filtered"] += 1
                continue

            try:
                sx, sy, sz = [float(c) for c in sensor_str.split(",")]
            except ValueError:
                stats["sensors_nan"] += 1
                continue

            feats = extract_sensor_features(
                sx, sy, sz, room_corners, wall_segments, apertures,
                l_char, room_height, sun_vectors, is_ccw)

            if feats.pop("_omega_saturated", False):
                stats["sensors_omega_saturated"] += 1

            feats["Group_ID"] = group_id
            feats["Iteration_Name"] = str(row["Iteration_Name"])
            feats["Source_Tag"] = suffix

            feats["Meta_BBox_W"] = bbox_w
            feats["Meta_BBox_D"] = bbox_d
            feats["Meta_L_char"] = l_char
            feats["Meta_Room_Height"] = room_height
            feats["Meta_N_Windows"] = len(apertures)
            feats["Meta_Total_Win_Area"] = total_win_area
            feats["Meta_WWR_proxy"] = (total_win_area / (perimeter * room_height)
                                       if perimeter * room_height > 0 else 0.0)
            feats["Meta_Orientation_deg"] = orientation_deg
            feats["Meta_Sensor_X_norm"] = (sx - min_x) / l_char
            feats["Meta_Sensor_Y_norm"] = (sy - min_y) / l_char
            feats["Meta_Sensor_Z"] = sz

            feats["Target_UDI_low"] = lo
            feats["Target_UDI_up"] = up
            feats["Target_UDI_useful"] = max(0.0, useful)
            records.append(feats)

        room_times.append(time.perf_counter() - t_room)
        stats["rooms_ok"] += 1

    df_out = pd.DataFrame(records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"omegaflux_sensors_{suffix}.csv")
    df_out.to_csv(out_path, index=False)

    if room_times:
        rt = np.array(room_times)
        n_sens = len(df_out) / max(1, len(room_times))
        print(f"\n  [OK] {len(df_out):,} sensor records -> {out_path}")
        print(f"       extraction: {rt.mean() * 1000:8.1f} ms/room (median "
              f"{np.median(rt) * 1000:.1f}, total {rt.sum():.1f} s), "
              f"~{n_sens:.0f} sensors/room, "
              f"{rt.sum() / max(1, len(df_out)) * 1000:.3f} ms/sensor")
        stats["extraction_ms_per_room"][suffix] = float(rt.mean() * 1000)
        stats["extraction_ms_per_sensor"][suffix] = float(rt.sum() / max(1, len(df_out)) * 1000)

    return df_out

# =============================================================================
# MAIN
# =============================================================================

POOLED_CSV_PATH = os.path.join("data", "processed", "omegaflux_sensors_pooled.csv")

def main():
    parser = argparse.ArgumentParser(description="OmegaFlux Feature Extraction")
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted feature table")
    args = parser.parse_args()

    print("=" * 78)
    print("OmegaFlux - Stage 1: Feature Extraction")
    print("=" * 78)
    

    if os.path.exists(POOLED_CSV_PATH) and not args.force:
        print(f"[INFO] Found existing feature table at '{POOLED_CSV_PATH}'.")
        print("[SKIP] Skipping feature extraction. (Use --force to overwrite)")
        return

    np.random.seed(RANDOM_SEED)

    sun_vectors = generate_annual_sun_vectors(SITE_LATITUDE_DEG)
    print("=" * 78)
    print("OmegaFlux - Stage 1: per-sensor feature extraction")
    print("=" * 78)
    print(f"Site               : {SITE_NAME} (lat {SITE_LATITUDE_DEG} deg)")
    print(f"Sun vectors        : {len(sun_vectors)}")
    print(f"Room height        : {ROOM_HEIGHT_MODE} ({ROOM_HEIGHT_M} m)")
    print(f"Visibility samples : {VISIBILITY_SAMPLES}")
    print(f"Legacy target filter: {APPLY_LEGACY_TARGET_FILTER}")
    print(f"Features emitted   : {len(FEATURES_INDEPENDENT)}")

    stats = {
        "rooms_ok": 0, "rooms_malformed": 0, "rooms_window_above_ceiling": 0,
        "sensors_nan": 0, "sensors_sum_gt_100": 0, "sensors_legacy_filtered": 0,
        "sensors_omega_saturated": 0,
        "legacy_would_drop_zero_useful": 0, "legacy_would_drop_full_useful": 0,
        "extraction_ms_per_room": {}, "extraction_ms_per_sensor": {},
    }

    frames = []
    for enabled, path, suffix, label in DATASETS:
        if not enabled:
            continue
        if not os.path.exists(path):
            print(f"\n[SKIP] {path} not found.")
            continue
        print(f"\n[LOAD] {label}: {path}")
        df_raw = load_table(path)

        if "Iteration_Nr" in df_raw.columns:
            if "Iteration_Name" not in df_raw.columns or df_raw["Iteration_Name"].nunique() <= 1:
                df_raw["Iteration_Name"] = df_raw["Iteration_Nr"].astype(str)

        frames.append(process_dataset(df_raw, suffix, label, sun_vectors, stats))

    if not frames:
        print("\nNothing processed.")
        return

    df_all = pd.concat(frames, ignore_index=True)
    pooled = os.path.join(OUTPUT_DIR, "omegaflux_sensors_pooled.csv")
    df_all.to_csv(pooled, index=False)

    total_legacy = (stats["legacy_would_drop_zero_useful"]
                    + stats["legacy_would_drop_full_useful"])
    n = len(df_all) + stats["sensors_legacy_filtered"]

    print("\n" + "=" * 78)
    print("EXTRACTION SUMMARY")
    print("=" * 78)
    print(f"Rooms processed              : {stats['rooms_ok']:,}")
    print(f"Rooms malformed / skipped    : {stats['rooms_malformed']:,}")
    print(f"Rooms w/ window above ceiling: {stats['rooms_window_above_ceiling']:,}")
    print(f"Distinct groups              : {df_all['Group_ID'].nunique():,}")
    print(f"Sensor records retained      : {len(df_all):,}")
    print(f"Sensors dropped (NaN/parse)  : {stats['sensors_nan']:,}")
    print(f"Sensors dropped (low+up>100) : {stats['sensors_sum_gt_100']:,}")
    sat = stats["sensors_omega_saturated"]
    print(f"Sensors w/ Omega_total > 2pi : {sat:,} "
          f"({sat / max(1, len(df_all)) * 100:.4f}%)")
    if sat / max(1, len(df_all)) > 0.001:
        print("  [WARN] Omega_total > 2pi is geometrically impossible. Check")
        print("         aperture-to-wall assignment and duplicated apertures.")
    print("-" * 78)
    print("Effect of the ORIGINAL censoring filter (now disabled):")
    print(f"  zero-useful sensors        : {stats['legacy_would_drop_zero_useful']:,}")
    print(f"  fully-useful sensors       : {stats['legacy_would_drop_full_useful']:,}")
    print(f"  combined                   : {total_legacy:,} "
          f"({total_legacy / max(1, n) * 100:.3f}% of records)")
    print("-" * 78)
    u = df_all["Target_UDI_useful"]
    print(f"Mean useful UDI              : {u.mean():.2f}%")
    print(f"Std dev useful UDI           : {u.std():.2f}%")
    print(f"Median useful UDI            : {u.median():.2f}%")
    print(f"Pitch-black (UDI_low > 90)   : {(df_all['Target_UDI_low'] > 90).mean() * 100:.2f}%")
    print(f"Overlit (UDI_up > 90)        : {(df_all['Target_UDI_up'] > 90).mean() * 100:.2f}%")
    print("=" * 78)
    print(f"Pooled dataset -> {pooled}")

    manifest = {
        "stage": "spatial_explosion",
        "seed": RANDOM_SEED,
        "site": {"name": SITE_NAME, "latitude_deg": SITE_LATITUDE_DEG},
        "sun_vectors": len(sun_vectors),
        "sun_sampling": {"declinations_deg": SUN_DECLINATIONS_DEG,
                         "hour_angle_range": list(SUN_HOUR_ANGLE_RANGE),
                         "min_elevation_deg": SUN_MIN_ELEVATION_DEG},
        "room_height_mode": ROOM_HEIGHT_MODE,
        "room_height_m": ROOM_HEIGHT_M,
        "visibility_samples": VISIBILITY_SAMPLES,
        "corner_penumbra_radius_m": CORNER_PENUMBRA_RADIUS_M,
        "legacy_target_filter_applied": APPLY_LEGACY_TARGET_FILTER,
        "features_independent": FEATURES_INDEPENDENT,
        "counts": {k: v for k, v in stats.items() if not isinstance(v, dict)},
        "extraction_ms_per_room": stats["extraction_ms_per_room"],
        "extraction_ms_per_sensor": stats["extraction_ms_per_sensor"],
        "records": int(len(df_all)),
        "groups": int(df_all["Group_ID"].nunique()),
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "platform": platform.platform()},
    }
    with open(os.path.join(OUTPUT_DIR, "manifest_extraction.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest       -> {os.path.join(OUTPUT_DIR, 'manifest_extraction.json')}")

if __name__ == "__main__":
    main()