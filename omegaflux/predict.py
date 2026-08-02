import os
import math
import numpy as np
import xgboost as xgb

from omegaflux.extract_features import (
    generate_annual_sun_vectors,
    build_apertures,
    extract_sensor_features,
    point_in_polygon,
    SITE_LATITUDE_DEG,
    ROOM_HEIGHT_M
)

OUT_DIR = "OmegaFlux_Models"

def generate_sensor_grid(room_corners, spacing):
    """
    Generates a regular grid of sensor points inside the room polygon.
    The grid starts exactly at spacing / 2 (e.g., 12.5cm for a 0.25m grid).
    """
    min_x = min(p[0] for p in room_corners)
    max_x = max(p[0] for p in room_corners)
    min_y = min(p[1] for p in room_corners)
    max_y = max(p[1] for p in room_corners)
    
    z = 0.8 
    
    sensors = []
    x_coords = np.arange(min_x + spacing / 2.0, max_x, spacing)
    y_coords = np.arange(min_y + spacing / 2.0, max_y, spacing)
    
    for x in x_coords:
        for y in y_coords:
            if point_in_polygon(x, y, room_corners):
                sensors.append((x, y, z))
            
    return sensors

def parse_window_specs(room_corners, window_specs):
    """
    Intelligently parses window specifications into the absolute 3D coordinate 
    tuples expected by the extractor: ((cx, cy, cz), width, height).
    
    Accepts three formats:
    1. Parametric dict: {'wall': 0, 'centre': 2.5, 'width': 1.8, 'sill': 0.9, 'height': 1.5}
    2. Absolute dict:   {'centre': (2.5, 0.0, 1.65), 'width': 1.8, 'height': 1.5}
    3. Raw tuple:       ((2.5, 0.0, 1.65), 1.8, 1.5)
    """
    formatted_windows = []
    
    for w in window_specs:
        # Format 3: Raw tuple ((cx, cy, cz), w, h)
        if isinstance(w, (tuple, list)) and len(w) == 3:
            formatted_windows.append(tuple(w))
            continue
            
        if isinstance(w, dict):
            width = w["width"]
            height = w["height"]
            
            # Format 1: Parametric dictionary (wall index, linear offset, and sill)
            if "wall" in w:
                wall_idx = w["wall"]
                centre_offset = w.get("centre", w.get("center"))
                sill = w["sill"]
                
                p1 = room_corners[wall_idx]
                p2 = room_corners[(wall_idx + 1) % len(room_corners)]
                
                vx, vy = p2[0] - p1[0], p2[1] - p1[1]
                wall_length = math.hypot(vx, vy)
                
                if wall_length == 0:
                    continue
                    
                dx, dy = vx / wall_length, vy / wall_length
                
                cx = p1[0] + dx * centre_offset
                cy = p1[1] + dy * centre_offset
                cz = sill + (height / 2.0)
                
                formatted_windows.append(((cx, cy, cz), width, height))
                
            # Format 2: Absolute dictionary (3D center tuple)
            elif "centre" in w or "center" in w:
                centre = w.get("centre", w.get("center"))
                if isinstance(centre, (tuple, list)) and len(centre) == 3:
                    formatted_windows.append((tuple(centre), width, height))
                    
    return formatted_windows

def predict_room(room_corners, window_specs, grid=0.25):
    """
    Predicts UDI_useful for a given room layout.
    
    :param room_corners: List of (x, y) tuples defining the room perimeter (CCW).
    :param window_specs: List of windows. Accepts both parametric dicts and absolute coordinates.
    :param grid: Either a float for sensor spacing in meters, or a list of (x,y,z) tuples.
    :return: Numpy array of [x, y, UDI_useful]
    """
    
    sun_vectors = generate_annual_sun_vectors(SITE_LATITUDE_DEG)
    wall_segments = [(room_corners[i], room_corners[(i + 1) % len(room_corners)]) for i in range(len(room_corners))]
    
    min_x = min(p[0] for p in room_corners)
    min_y = min(p[1] for p in room_corners)
    bbox_w = max(p[0] for p in room_corners) - min_x
    bbox_d = max(p[1] for p in room_corners) - min_y
    l_char = max(1.0, math.hypot(bbox_w, bbox_d))
    
    signed_area = sum(
        room_corners[i][0] * room_corners[(i + 1) % len(room_corners)][1]
        - room_corners[(i + 1) % len(room_corners)][0] * room_corners[i][1]
        for i in range(len(room_corners))
    ) / 2.0
    is_ccw = signed_area > 0

    formatted_windows = parse_window_specs(room_corners, window_specs)
    apertures = build_apertures(room_corners, formatted_windows, wall_segments)
    
    if isinstance(grid, (list, tuple, np.ndarray)):
        sensors = grid
    else:
        sensors = generate_sensor_grid(room_corners, float(grid))
        
    feature_records = []
    
    for (sx, sy, sz) in sensors:
        feats = extract_sensor_features(
            sx, sy, sz, room_corners, wall_segments, apertures,
            l_char, ROOM_HEIGHT_M, sun_vectors, is_ccw
        )
        feats.pop("_omega_saturated", None)
        feature_records.append(list(feats.values()))
        
    if not feature_records:
        return np.array([])
        
    X_pred = np.array(feature_records, dtype="float32")
    
    m_low = xgb.XGBRegressor()
    m_up = xgb.XGBRegressor()
    
    m_low.load_model(os.path.join(OUT_DIR, "omegaflux_xgb_low.json"))
    m_up.load_model(os.path.join(OUT_DIR, "omegaflux_xgb_up.json"))
    
    p_low = np.clip(m_low.predict(X_pred), 0.0, 100.0)
    p_up = np.clip(m_up.predict(X_pred), 0.0, 100.0)
    p_useful = np.clip(100.0 - p_low - p_up, 0.0, 100.0)
    
    results = []
    for i, (sx, sy, sz) in enumerate(sensors):
        results.append([sx, sy, p_useful[i]])
        
    return np.array(results)