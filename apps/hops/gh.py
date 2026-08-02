import os
import time
import math
import joblib
import numpy as np
from flask import Flask
from sklearn.metrics import mean_squared_error, r2_score
import ghhops_server as hs

app = Flask(__name__)
hops: hs.HopsFlask = hs.Hops(app)

TWO_PI = 2.0 * math.pi

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.getenv("OMEGAFLUX_MODEL_DIR", os.path.join(BASE_DIR, "OmegaFlux_Models"))

model_low_path = os.path.join(MODEL_DIR, "omegaflux_xgb_low.joblib")
model_up_path = os.path.join(MODEL_DIR, "omegaflux_xgb_up.joblib")

if not os.path.exists(model_low_path):
    model_low_path = os.path.join("models", "omegaflux_xgb_low.joblib")
    model_up_path = os.path.join("models", "omegaflux_xgb_up.joblib")

model_low = joblib.load(model_low_path)
model_up = joblib.load(model_up_path)


def smoothstep(edge0, edge1, x):
    if edge1 - edge0 == 0: return 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def distance_point_to_ray_segment(p, a, b):
    ab_x, ab_y = b.X - a.X, b.Y - a.Y
    ap_x, ap_y = p.X - a.X, p.Y - a.Y
    ab_len_sq = ab_x**2 + ab_y**2
    if ab_len_sq == 0: return math.hypot(p.X - a.X, p.Y - a.Y)
    t = max(0.0, min(1.0, (ap_x * ab_x + ap_y * ab_y) / ab_len_sq))
    proj_x = a.X + t * ab_x
    proj_y = a.Y + t * ab_y
    return math.hypot(p.X - proj_x, p.Y - proj_y)


def sort_quad_corners_perimeter(corners):
    c_np = [np.array([c.X, c.Y, c.Z]) for c in corners]
    center = sum(c_np) / 4.0
    v1 = c_np[1] - c_np[0]
    v2 = c_np[2] - c_np[0]
    norm = np.cross(v1, v2)
    norm_len = np.linalg.norm(norm)
    if norm_len < 1e-6:
        v2 = c_np[3] - c_np[0]
        norm = np.cross(v1, v2)
        norm_len = np.linalg.norm(norm)
    norm = norm / (norm_len + 1e-9) if norm_len > 1e-6 else np.array([0.0, 0.0, 1.0])
    u_axis = v1 / (np.linalg.norm(v1) + 1e-9)
    v_axis = np.cross(norm, u_axis)
    angles_pts = []
    for pt in c_np:
        rel = pt - center
        u_val = np.dot(rel, u_axis)
        v_val = np.dot(rel, v_axis)
        angle = math.atan2(v_val, u_val)
        angles_pts.append((angle, pt))
    angles_pts.sort(key=lambda x: x[0])
    return [item[1] for item in angles_pts]


def ray_intersects_triangle_3d(orig, ray_dir, p0, p1, p2, eps=1e-6):
    edge1 = p1 - p0
    edge2 = p2 - p0
    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)
    if -eps < a < eps: return False, None
    f = 1.0 / a
    s = orig - p0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0: return False, None
    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)
    if v < 0.0 or u + v > 1.0: return False, None
    t = f * np.dot(edge2, q)
    if t > eps: return True, orig + t * ray_dir
    return False, None


def solid_angle_triangle(p, a, b, c):
    r1 = np.array([a.X - p.X, a.Y - p.Y, a.Z - p.Z])
    r2 = np.array([b.X - p.X, b.Y - p.Y, b.Z - p.Z])
    r3 = np.array([c.X - p.X, c.Y - p.Y, c.Z - p.Z])
    l1, l2, l3 = np.linalg.norm(r1), np.linalg.norm(r2), np.linalg.norm(r3)
    if l1 == 0 or l2 == 0 or l3 == 0: return 0.0
    scalar_triple = np.dot(r1, np.cross(r2, r3))
    denominator = l1 * l2 * l3 + np.dot(r1, r2) * l3 + np.dot(r2, r3) * l1 + np.dot(r3, r1) * l2
    if denominator == 0: return 0.0
    return abs(2.0 * np.arctan2(scalar_triple, denominator))


def solid_angle_polygon(p, corners):
    if len(corners) < 3: return 0.0
    total_omega = 0.0
    p0 = corners[0]
    for i in range(1, len(corners) - 1):
        total_omega += solid_angle_triangle(p, p0, corners[i], corners[i+1])
    return total_omega


def line_segments_intersect_2d(p1, p2, p3, p4):
    def ccw(A, B, C): return (C.Y - A.Y) * (B.X - A.X) > (B.Y - A.Y) * (C.X - A.X)
    if (abs(p1.X - p3.X) < 1e-4 and abs(p1.Y - p3.Y) < 1e-4) or \
       (abs(p1.X - p4.X) < 1e-4 and abs(p1.Y - p4.Y) < 1e-4) or \
       (abs(p2.X - p3.X) < 1e-4 and abs(p2.Y - p3.Y) < 1e-4) or \
       (abs(p2.X - p4.X) < 1e-4 and abs(p2.Y - p4.Y) < 1e-4): return False
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def is_same_wall_segment(p1, p2, host_p1, host_p2, tol=1e-3):
    d11 = math.hypot(p1.X - host_p1.X, p1.Y - host_p1.Y)
    d22 = math.hypot(p2.X - host_p2.X, p2.Y - host_p2.Y)
    if d11 < tol and d22 < tol: return True
    d12 = math.hypot(p1.X - host_p2.X, p1.Y - host_p2.Y)
    d21 = math.hypot(p2.X - host_p1.X, p2.Y - host_p1.Y)
    return d12 < tol and d21 < tol


def ray_cast_2d(origin, direction, segments):
    min_dist = float('inf')
    dx, dy = direction[0], direction[1]
    for p1, p2 in segments:
        vx, vy = p2.X - p1.X, p2.Y - p1.Y
        qx, qy = origin.X - p1.X, origin.Y - p1.Y
        cross = dx * vy - dy * vx
        if abs(cross) < 1e-6: continue
        u = (vx * qy - vy * qx) / cross
        t = (dx * qy - dy * qx) / cross
        if u > 1e-4 and 0.0 <= t <= 1.0:
            if u < min_dist: min_dist = u
    return min_dist if min_dist != float('inf') else 20.0


def get_window_room_exposure_ratio(win, wall_segments):
    nx, ny = win['normal'][0], win['normal'][1]
    base_angle = math.atan2(ny, nx)
    cx, cy = win['center'].X + nx * 0.05, win['center'].Y + ny * 0.05
    origin = type('Obj', (object,), {'X': cx, 'Y': cy})()
    host_p1, host_p2 = win['host_p1'], win['host_p2']
    total_rays, strike_count = 0, 0
    for deg in range(-90, 91, 15):
        total_rays += 1
        rad = base_angle + math.radians(deg)
        dx, dy = math.cos(rad), math.sin(rad)
        ray_end = type('Obj', (object,), {'X': cx + dx * 50.0, 'Y': cy + dy * 50.0})()
        hit_wall = False
        for p3_w, p4_w in wall_segments:
            if is_same_wall_segment(p3_w, p4_w, host_p1, host_p2): continue
            if line_segments_intersect_2d(origin, ray_end, p3_w, p4_w):
                hit_wall = True
                break
        if hit_wall: strike_count += 1
    if total_rays == 0: return 1.0
    return max(0.05, 1.0 - (float(strike_count) / float(total_rays)))


def get_window_visibility_and_occlusion_dist(sensor_pt, corners, room_corners, host_p1, host_p2):
    if len(corners) < 4: return 1.0, 5.0
    
    wall_vec = np.array([host_p2.X - host_p1.X, host_p2.Y - host_p1.Y, host_p2.Z - host_p1.Z])
    wall_len = np.linalg.norm(wall_vec)
    wall_unit = wall_vec / wall_len if wall_len > 1e-6 else np.array([1.0, 0.0, 0.0])
    
    corner_projections = []
    for c in corners:
        c_vec = np.array([c.X - host_p1.X, c.Y - host_p1.Y, c.Z - host_p1.Z])
        proj = np.dot(c_vec, wall_unit)
        corner_projections.append((proj, c))
    corner_projections.sort(key=lambda x: x[0])
    
    left_corners = [x[1] for x in corner_projections[:2]]
    right_corners = [x[1] for x in corner_projections[2:]]
    
    left_pt = type('Obj', (object,), {'X': sum(c.X for c in left_corners)/len(left_corners), 'Y': sum(c.Y for c in left_corners)/len(left_corners), 'Z': sum(c.Z for c in left_corners)/len(left_corners)})()
    right_pt = type('Obj', (object,), {'X': sum(c.X for c in right_corners)/len(right_corners), 'Y': sum(c.Y for c in right_corners)/len(right_corners), 'Z': sum(c.Z for c in right_corners)/len(right_corners)})()
    center_pt = type('Obj', (object,), {'X': sum(c.X for c in corners)/len(corners), 'Y': sum(c.Y for c in corners)/len(corners), 'Z': sum(c.Z for c in corners)/len(corners)})()
    
    target_pts = [left_pt, center_pt, right_pt]
    all_blockers = [(room_corners[i], room_corners[(i + 1) % len(room_corners)]) for i in range(len(room_corners))]

    penumbra_radius = 0.40
    unblocked_ray_score = 0.0

    for target in target_pts:
        dir_x, dir_y = sensor_pt.X - target.X, sensor_pt.Y - target.Y
        dist = math.hypot(dir_x, dir_y)
        if dist > 0.05: tx, ty = target.X + (dir_x / dist) * 0.05, target.Y + (dir_y / dist) * 0.05
        else: tx, ty = target.X, target.Y
        sample_target = type('Obj', (object,), {'X': tx, 'Y': ty})()
        
        blocked = False
        for p3_wall, p4_wall in all_blockers:
            if is_same_wall_segment(p3_wall, p4_wall, host_p1, host_p2): continue
            if line_segments_intersect_2d(sensor_pt, sample_target, p3_wall, p4_wall):
                blocked = True
                break
        
        ray_score = 0.0 if blocked else 1.0

        if not blocked and room_corners:
            for corner in room_corners:
                d_vert = distance_point_to_ray_segment(corner, sensor_pt, sample_target)
                if d_vert < penumbra_radius:
                    penumbra_factor = smoothstep(0.0, penumbra_radius, penumbra_radius - d_vert)
                    ray_score = max(0.0, 1.0 - penumbra_factor)

        unblocked_ray_score += ray_score

    unblocked_ratio = max(0.0, smoothstep(0.0, 1.0, unblocked_ray_score / 3.0))
    wall_dist = get_distance_to_room_boundary(sensor_pt, room_corners)
    win_dist = min(math.hypot(sensor_pt.X - c.X, sensor_pt.Y - c.Y) for c in corners) if corners else 10.0
    return unblocked_ratio, max(0.01, min(wall_dist, win_dist))


def distance_point_to_segment_2d(p, a, b):
    ab_x, ab_y = b.X - a.X, b.Y - a.Y
    ap_x, ap_y = p.X - a.X, p.Y - a.Y
    ab_len_sq = ab_x**2 + ab_y**2
    if ab_len_sq == 0: return np.sqrt((p.X - a.X)**2 + (p.Y - a.Y)**2)
    t = max(0.0, min(1.0, (ap_x * ab_x + ap_y * ab_y) / ab_len_sq))
    return np.sqrt((p.X - (a.X + t * ab_x))**2 + (p.Y - (a.Y + t * ab_y))**2)


def get_distance_to_room_boundary(pt, room_corners):
    min_dist = float('inf')
    if not room_corners or len(room_corners) < 2: return 0.5
    for i in range(len(room_corners)):
        dist = distance_point_to_segment_2d(pt, room_corners[i], room_corners[(i + 1) % len(room_corners)])
        if dist < min_dist: min_dist = dist
    return max(0.05, min_dist)


def get_corner_distances(pt, room_corners):
    if not room_corners or len(room_corners) < 3: return 10.0, 0.5
    area = sum((room_corners[i].X * room_corners[(i + 1) % len(room_corners)].Y - room_corners[(i + 1) % len(room_corners)].X * room_corners[i].Y) for i in range(len(room_corners)))
    is_ccw = (area / 2.0) > 0
    inner_dmin, outer_dmin, has_inner = float('inf'), float('inf'), False
    for i in range(len(room_corners)):
        p_prev, p_curr, p_next = room_corners[(i - 1) % len(room_corners)], room_corners[i], room_corners[(i + 1) % len(room_corners)]
        v1, v2 = np.array([p_curr.X - p_prev.X, p_curr.Y - p_prev.Y]), np.array([p_next.X - p_curr.X, p_next.Y - p_curr.Y])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        is_inner = (cross < 0) if is_ccw else (cross > 0)
        dist = np.sqrt((pt.X - p_curr.X)**2 + (pt.Y - p_curr.Y)**2)
        if is_inner:
            has_inner = True
            if dist < inner_dmin: inner_dmin = dist
        else:
            if dist < outer_dmin: outer_dmin = dist
    if not has_inner or inner_dmin == float('inf'): inner_dmin = 10.0
    if outer_dmin == float('inf'): outer_dmin = 0.5
    return max(0.01, inner_dmin), max(0.01, outer_dmin)


def point_in_polygon(pt_x, pt_y, polygon):
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        pi, pj = polygon[i], polygon[j]
        if ((pi.Y > pt_y) != (pj.Y > pt_y)) and (pt_x < (pj.X - pi.X) * (pt_y - pi.Y) / (pj.Y - pi.Y + 1e-9) + pi.X):
            inside = not inside
        j = i
    return inside


def generate_annual_sun_vectors(latitude_deg: float):
    rays = []
    lat = math.radians(latitude_deg)
    for dec_deg in [-23.45, -11.5, 0.0, 11.5, 23.45]:
        dec = math.radians(dec_deg)
        for hour_angle_deg in range(-120, 121, 10):
            ha = math.radians(hour_angle_deg)
            sin_elev = (math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha))
            elev = math.asin(max(-1.0, min(1.0, sin_elev)))
            if elev > math.radians(2.0):
                dx = -math.cos(dec) * math.sin(ha)
                dy = (math.sin(dec) * math.cos(lat) - math.cos(dec) * math.sin(lat) * math.cos(ha))
                dz = math.sin(elev)
                n = math.sqrt(dx * dx + dy * dy + dz * dz)
                rays.append(np.array([dx / n, dy / n, dz / n]))
    return rays


south_dir = np.array([0.0, -1.0])
east_dir  = np.array([1.0,  0.0])
west_dir  = np.array([-1.0, 0.0])
south_solar_rays = generate_annual_sun_vectors(52.17)
total_rays_count = len(south_solar_rays)


@hops.component(
    "/omegaflux_predict",
    name="OmegaFlux Predict (26-Feature ML)",
    nickname="OmegaFlux",
    description="Predicts UDI metrics using 26 Normalized ML Features",
    inputs=[
        hs.HopsPoint("Points", "Pts", "Sensor grid points", access=hs.HopsParamAccess.LIST),
        hs.HopsPoint("Room Corners", "RoomPts", "Corner points of room perimeter", access=hs.HopsParamAccess.LIST),
        hs.HopsPoint("Window Corners", "Corners", "Flat list of 4 corner points per window", access=hs.HopsParamAccess.LIST),
        hs.HopsNumber("Simulated Useful UDI", "Simulated", "Optional ground truth UDI", access=hs.HopsParamAccess.LIST, optional=True),
    ],
    outputs=[
        hs.HopsNumber("Useful UDI", "Useful", "Predicted Useful UDI percentage", access=hs.HopsParamAccess.LIST),
        hs.HopsNumber("UDI Low", "Low", "Predicted UDI Low percentage", access=hs.HopsParamAccess.LIST),
        hs.HopsNumber("UDI Up", "Up", "Predicted UDI Up percentage", access=hs.HopsParamAccess.LIST),
        hs.HopsNumber("R2 Score", "R2", "Coefficient of Determination ($R^2$)", access=hs.HopsParamAccess.ITEM),
        hs.HopsNumber("RMSE", "RMSE", "Root Mean Squared Error", access=hs.HopsParamAccess.ITEM),
        hs.HopsNumber("MSE", "MSE", "Mean Squared Error", access=hs.HopsParamAccess.ITEM),
    ]
)
def omegaflux_predict(points, room_corners, window_corners, simulated_udi=None):
    start_time = time.perf_counter()
    
    if room_corners:
        box_min_x, box_max_x = min(pt.X for pt in room_corners), max(pt.X for pt in room_corners)
        box_min_y, box_max_y = min(pt.Y for pt in room_corners), max(pt.Y for pt in room_corners)
        wall_segments = [(room_corners[i], room_corners[(i+1)%len(room_corners)]) for i in range(len(room_corners))]
        l_char = max(1.0, math.hypot(box_max_x - box_min_x, box_max_y - box_min_y))
    else:
        box_min_x, box_max_x, box_min_y, box_max_y = 0.0, 1.0, 0.0, 1.0
        wall_segments = []
        l_char = 1.0

    parsed_windows = []
    
    for i in range(0, len(window_corners), 4):
        corners = window_corners[i:i+4]
        if len(corners) < 4: continue
        
        c_mean = type('Obj', (object,), {'X': sum(c.X for c in corners)/4.0, 'Y': sum(c.Y for c in corners)/4.0, 'Z': sum(c.Z for c in corners)/4.0})()
        
        min_dist = float('inf')
        best_p1, best_p2 = None, None
        for j in range(len(room_corners)):
            p1, p2 = room_corners[j], room_corners[(j+1)%len(room_corners)]
            dist = distance_point_to_segment_2d(c_mean, p1, p2)
            if dist < min_dist:
                min_dist, best_p1, best_p2 = dist, p1, p2
                
        if best_p1 is None or best_p2 is None: continue
        
        vx, vy = best_p2.X - best_p1.X, best_p2.Y - best_p1.Y
        length = math.hypot(vx, vy)
        if length == 0: continue
        nx, ny = -vy / length, vx / length 
        
        if point_in_polygon(c_mean.X + nx * 0.1, c_mean.Y + ny * 0.1, room_corners): nx, ny = -nx, -ny
            
        window_normal = np.array([nx, ny])
        sorted_c_np = sort_quad_corners_perimeter(corners)
            
        triangles = [
            (sorted_c_np[0], sorted_c_np[1], sorted_c_np[2]),
            (sorted_c_np[0], sorted_c_np[2], sorted_c_np[3])
        ]
        parsed_windows.append({
            'corners': corners, 'normal': window_normal, 
            'host_p1': best_p1, 'host_p2': best_p2, 
            'center': c_mean, 'triangles': triangles
        })

    room_height = 3.0

    udi_useful_out, udi_low_out, udi_up_out = [], [], []
    
    for pt in points:
        try:
            total_omega, south_weighted_omega, east_weighted_omega, west_weighted_omega = 0.0, 0.0, 0.0, 0.0
            window_sky_weighted_omega = 0.0
            total_area, area_weighted_cos_theta = 0.0, 0.0
            
            all_dmins, all_occl_distances, window_angles = [], [], []
            visible_windows_data = []
            all_vis_weights = []
            
            best_win_omega = -1.0
            dominant_win = None

            opaque_wall_dmin = get_distance_to_room_boundary(pt, room_corners)
            inner_corner_dmin, outer_corner_dmin = get_corner_distances(pt, room_corners)

            dx_left, dx_right = pt.X - box_min_x, box_max_x - pt.X
            dy_bottom, dy_top = pt.Y - box_min_y, box_max_y - pt.Y
            wall_distances = [dx_left, dx_right, dy_bottom, dy_top]
            smooth_mean_depth = float(np.mean(wall_distances))
            wall_distance_stddev = float(np.std(wall_distances))

            pt_np = np.array([pt.X, pt.Y, pt.Z])
            solar_hits = 0

            for ray_dir in south_solar_rays:
                ray_hit_window = False
                for win in parsed_windows:
                    for (t0, t1, t2) in win['triangles']:
                        hit_valid, hit_pt = ray_intersects_triangle_3d(pt_np, ray_dir, t0, t1, t2)
                        if hit_valid:
                            nudge_target = type('Obj', (object,), {'X': hit_pt[0] - ray_dir[0]*0.05, 'Y': hit_pt[1] - ray_dir[1]*0.05})()
                            blocked = False
                            for p3_w, p4_w in wall_segments:
                                if is_same_wall_segment(p3_w, p4_w, win['host_p1'], win['host_p2']): continue
                                if line_segments_intersect_2d(pt, nudge_target, p3_w, p4_w):
                                    blocked = True
                                    break
                            if not blocked:
                                ray_hit_window = True
                                break
                    if ray_hit_window: break
                if ray_hit_window: solar_hits += 1

            annual_direct_sun_fraction = float(solar_hits) / float(total_rays_count)

            for win in parsed_windows:
                corners, window_normal = win['corners'], win['normal']
                cx, cy, cz = win['center'].X, win['center'].Y, win['center'].Z

                visibility_weight, min_occl_dist = get_window_visibility_and_occlusion_dist(pt, corners, room_corners, win['host_p1'], win['host_p2'])
                all_vis_weights.append(visibility_weight)
                all_occl_distances.append(min_occl_dist)
                
                if visibility_weight > 0.05:
                    for c_pt in corners:
                        window_angles.append(math.atan2(c_pt.Y - pt.Y, c_pt.X - pt.X))
                
                raw_win_omega = solid_angle_polygon(pt, corners)
                win_omega = raw_win_omega * visibility_weight
                total_omega += win_omega
                
                if win_omega > best_win_omega:
                    best_win_omega = win_omega
                    dominant_win = win

                if visibility_weight > 0.01:
                    south_weighted_omega += win_omega * max(0.0, np.dot(window_normal, south_dir))
                    east_weighted_omega += win_omega * max(0.0, np.dot(window_normal, east_dir))
                    west_weighted_omega += win_omega * max(0.0, np.dot(window_normal, west_dir))
                    
                    sky_vis_ratio = get_window_room_exposure_ratio(win, wall_segments)
                    window_sky_weighted_omega += raw_win_omega * visibility_weight * sky_vis_ratio

                    xs, ys, zs = [c.X for c in corners], [c.Y for c in corners], [c.Z for c in corners]
                    w_width = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
                    z1, z2 = min(zs), max(zs)
                    h = max(z2 - z1, 0.1)
                    win_area = (w_width * h) * visibility_weight

                    d = max(0.1, np.sqrt((cx - pt.X)**2 + (cy - pt.Y)**2))
                    d3d_cent = max(np.sqrt((cx - pt.X)**2 + (cy - pt.Y)**2 + (cz - pt.Z)**2), 0.001)

                    area_weighted_cos_theta += (d / d3d_cent) * win_area
                    total_area += win_area
                    all_dmins.append(max(0, w_width / 2.0)**2 + d**2 + max(0, abs(cz - pt.Z) - h / 2.0)**2)

                    if visibility_weight > 0.0:
                        dist_weight = win_area / (d**3 + 1e-3)
                        visible_windows_data.append({
                            'sill': pt.Z - z1, 'header': z2 - pt.Z, 'weight': dist_weight, 'd': d
                        })

            visible_window_count = float(len(visible_windows_data))
            mean_window_distance = float(np.mean([item['d'] for item in visible_windows_data])) if visible_windows_data else l_char
            occlusion_multiplier = float(min(all_vis_weights)) if all_vis_weights else 1.0

            if dominant_win is not None:
                dom_nx, dom_ny = dominant_win['normal'][0], dominant_win['normal'][1]
                dom_ray_forward = ray_cast_2d(pt, (dom_nx, dom_ny), wall_segments)
                dom_ray_backward = ray_cast_2d(pt, (-dom_nx, -dom_ny), wall_segments)
                dom_ray_left = ray_cast_2d(pt, (-dom_ny, dom_nx), wall_segments)
                dom_ray_right = ray_cast_2d(pt, (dom_ny, -dom_nx), wall_segments)
                
                a = dominant_win['host_p1']
                b = dominant_win['host_p2']
                vx, vy = b.X - a.X, b.Y - a.Y
                length = math.hypot(vx, vy)
                if length > 1e-6:
                    dom_plane_dist = abs((vy * (pt.X - a.X) - vx * (pt.Y - a.Y)) / length)
                    cx, cy = dominant_win['center'].X, dominant_win['center'].Y
                    dom_h_offset = abs((pt.X - cx) * (vx/length) + (pt.Y - cy) * (vy/length))
                else:
                    dom_plane_dist, dom_h_offset = l_char, l_char
            else:
                dom_ray_forward = ray_cast_2d(pt, (0, 1), wall_segments)
                dom_ray_backward = ray_cast_2d(pt, (0, -1), wall_segments)
                dom_ray_left = ray_cast_2d(pt, (-1, 0), wall_segments)
                dom_ray_right = ray_cast_2d(pt, (1, 0), wall_segments)
                dom_plane_dist, dom_h_offset = l_char, l_char

            global_cos_theta = area_weighted_cos_theta / total_area if total_area > 0 else 0.0
            
            if len(visible_windows_data) == 0:
                global_sill_clearance, global_header_clearance = 0.0, 0.0
            elif len(visible_windows_data) == 1:
                global_sill_clearance = visible_windows_data[0]['sill']
                global_header_clearance = visible_windows_data[0]['header']
            else:
                sum_weights = sum(item['weight'] for item in visible_windows_data)
                global_sill_clearance = sum(item['sill'] * item['weight'] for item in visible_windows_data) / sum_weights if sum_weights > 0 else 0.0
                global_header_clearance = sum(item['header'] * item['weight'] for item in visible_windows_data) / sum_weights if sum_weights > 0 else 0.0
            
            raw_inv_dmin2 = 1.0 / max(min(all_dmins), 1e-4) if all_dmins else 0.0
            min_occluder_distance = min(all_occl_distances) if all_occl_distances else 20.0

            if not window_angles:
                angular_spread = 0.0
            else:
                window_angles = sorted(window_angles)
                max_gap = max((window_angles[k] - window_angles[k-1] + 2*math.pi) % (2*math.pi) for k in range(len(window_angles)))
                angular_spread = 2 * math.pi - max_gap

            features = [
                float(np.clip(total_omega / TWO_PI, 0.0, 1.0)),
                float(np.clip(global_cos_theta, 0.0, 1.0)),
                float(np.clip(global_sill_clearance / room_height, 0.0, 1.0)),
                float(np.clip(global_header_clearance / room_height, 0.0, 1.0)),
                float(raw_inv_dmin2 * (l_char ** 2)),
                float(np.clip(opaque_wall_dmin / l_char, 0.0, 1.0)),
                float(np.clip(inner_corner_dmin / l_char, 0.0, 1.0)),
                float(np.clip(outer_corner_dmin / l_char, 0.0, 1.0)),
                float(np.clip(south_weighted_omega / TWO_PI, 0.0, 1.0)),
                float(np.clip(east_weighted_omega / TWO_PI, 0.0, 1.0)),
                float(np.clip(west_weighted_omega / TWO_PI, 0.0, 1.0)),
                float(np.clip(annual_direct_sun_fraction, 0.0, 1.0)),
                float(np.clip(smooth_mean_depth / l_char, 0.0, 1.0)),
                float(np.clip(wall_distance_stddev / l_char, 0.0, 1.0)),
                float(np.clip(min_occluder_distance / l_char, 0.0, 1.0)),
                float(np.clip(angular_spread / TWO_PI, 0.0, 1.0)),
                float(np.clip(window_sky_weighted_omega / TWO_PI, 0.0, 1.0)),
                float(np.clip(occlusion_multiplier, 0.0, 1.0)),
                float(np.clip(visible_window_count / 10.0, 0.0, 1.0)),
                float(np.clip(mean_window_distance / l_char, 0.0, 1.0)),
                float(np.clip(dom_plane_dist / l_char, 0.0, 1.0)),
                float(np.clip(dom_h_offset / l_char, 0.0, 1.0)),
                float(np.clip(dom_ray_forward / l_char, 0.0, 1.0)),
                float(np.clip(dom_ray_backward / l_char, 0.0, 1.0)),
                float(np.clip(dom_ray_left / l_char, 0.0, 1.0)),
                float(np.clip(dom_ray_right / l_char, 0.0, 1.0))
            ]
            
            X_input = np.array([features], dtype=np.float32)
            low = max(0.0, min(100.0, float(model_low.predict(X_input)[0])))
            up = max(0.0, min(100.0, float(model_up.predict(X_input)[0])))
            useful = max(0.0, min(100.0, 100.0 - low - up))
            
            udi_useful_out.append(useful)
            udi_low_out.append(low)
            udi_up_out.append(up)

        except Exception:
            udi_useful_out.append(0.0)
            udi_low_out.append(0.0)
            udi_up_out.append(0.0)
            
    r2, rmse, mse = 0.0, 0.0, 0.0
    if simulated_udi and len(simulated_udi) == len(udi_useful_out):
        y_true = np.array(simulated_udi, dtype=np.float32)
        y_pred = np.array(udi_useful_out, dtype=np.float32)
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

    return udi_useful_out, udi_low_out, udi_up_out, r2, rmse, mse


if __name__ == "__main__":
    app.run(debug=True, port=5000)