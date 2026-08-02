import math
import numpy as np
import pandas as pd
from scipy.stats import qmc

def generate_bins_A_to_D(samples_per_bin=1125):
    all_valid_data = []
    bins_config = [
        {"front_count": 2, "side_count": 0},
        {"front_count": 1, "side_count": 1},
        {"front_count": 2, "side_count": 1},
        {"front_count": 2, "side_count": 2}
    ]
    global_row_count = 1
    
    for b_idx, config in enumerate(bins_config):
        f_count = config["front_count"]
        s_count = config["side_count"]
        bin_letter = chr(65 + b_idx) 
        sampler = qmc.LatinHypercube(d=12, seed=200 + b_idx)
        sample = sampler.random(n=samples_per_bin)
        
        valid_bin_count = 0
        for row in sample:
            room_width = round(4.0 + row[0] * 2.0, 2)
            room_depth = round(4.0 + row[1] * 2.0, 2)
            canyon_hw = 0.0
            orientation = 0
            
            f_h = round(1.2 + row[4] * 0.6, 2)
            s_h = round(1.2 + row[5] * 0.6, 2)
            
            f_cv_min, f_cv_max = 0.4 + f_h/2, 2.5 - f_h/2
            f_cv = round(f_cv_min + row[6] * (f_cv_max - f_cv_min), 2)
            
            s_cv_min, s_cv_max = 0.4 + s_h/2, 2.5 - s_h/2
            s_cv = round(s_cv_min + row[7] * (s_cv_max - s_cv_min), 2)
            
            if f_count == 1:
                f_w_ratio = 0.20 + row[8] * 0.40
                f_w = round(f_w_ratio * room_width, 2)
                f_ch1_ratio_min = 0.05 + f_w_ratio/2
                f_ch1_ratio_max = 0.95 - f_w_ratio/2
                f_ch1 = round((f_ch1_ratio_min + row[9] * (f_ch1_ratio_max - f_ch1_ratio_min)) * room_width, 2)
                f_ch2 = 0.0
            else:
                f_w_ratio = 0.15 + row[8] * 0.20
                f_w = round(f_w_ratio * room_width, 2)
                f_c1_min, f_c1_max = 0.05 + f_w_ratio/2, 0.475 - f_w_ratio/2
                f_ch1 = round((f_c1_min + row[9] * (f_c1_max - f_c1_min)) * room_width, 2)
                f_c2_min, f_c2_max = 0.525 + f_w_ratio/2, 0.95 - f_w_ratio/2
                f_ch2 = round((f_c2_min + row[10] * (f_c2_max - f_c2_min)) * room_width, 2)
                
            if s_count == 0:
                s_w, s_ch1, s_ch2 = 0.0, 0.0, 0.0
            elif s_count == 1:
                s_w_ratio = 0.20 + row[11] * 0.40
                s_w = round(s_w_ratio * room_depth, 2)
                s_ch1_min = 0.05 + s_w_ratio/2
                s_ch1_max = 0.95 - s_w_ratio/2
                s_ch1 = round((s_ch1_min + row[10] * (s_ch1_max - s_ch1_min)) * room_depth, 2)
                s_ch2 = 0.0
            else:
                s_w_ratio = 0.15 + row[11] * 0.20
                s_w = round(s_w_ratio * room_depth, 2)
                s_c1_min, s_c1_max = 0.05 + s_w_ratio/2, 0.475 - s_w_ratio/2
                s_ch1 = round((s_c1_min + row[9] * (s_c1_max - s_c1_min)) * room_depth, 2)
                s_c2_min, s_c2_max = 0.525 + s_w_ratio/2, 0.95 - s_w_ratio/2
                s_ch2 = round((s_c2_min + row[10] * (s_c2_max - s_c2_min)) * room_depth, 2)
                
            iteration_num = valid_bin_count + 1
            iteration_id = f"{bin_letter}_{iteration_num}"
            
            all_valid_data.append({
                "Row_Nr": global_row_count,
                "Iteration_ID": iteration_id,
                "Bin_ID": f"Bin_{bin_letter}",
                "Room_Width_m": room_width,
                "Room_Depth_m": room_depth,
                "Orientation_deg": orientation,
                "Canyon_HW": canyon_hw,
                "Front_Win1_Center_H_m": f_ch1,
                "Front_Win1_Center_V_m": f_cv,
                "Front_Win1_Width_m": f_w,
                "Front_Win1_Height_m": f_h,
                "Front_Win2_Center_H_m": f_ch2,
                "Front_Win2_Center_V_m": f_cv if f_count == 2 else 0.0,
                "Front_Win2_Width_m": f_w if f_count == 2 else 0.0,
                "Front_Win2_Height_m": f_h if f_count == 2 else 0.0,
                "Side_Win1_Center_H_m": s_ch1,
                "Side_Win1_Center_V_m": s_cv if s_count > 0 else 0.0,
                "Side_Win1_Width_m": s_w,
                "Side_Win1_Height_m": s_h if s_count > 0 else 0.0,
                "Side_Win2_Center_H_m": s_ch2,
                "Side_Win2_Center_V_m": s_cv if s_count == 2 else 0.0,
                "Side_Win2_Width_m": s_w if s_count == 2 else 0.0,
                "Side_Win2_Height_m": s_h if s_count == 2 else 0.0,
            })
            
            valid_bin_count += 1
            global_row_count += 1
            
    return pd.DataFrame(all_valid_data)

def generate_bin_E(samples_per_bin=281):
    all_valid_data = []
    bins_config = [
        {"front_count": 2, "side_count": 0},
        {"front_count": 1, "side_count": 1},
        {"front_count": 2, "side_count": 1},
        {"front_count": 2, "side_count": 2}
    ]
    orientations = [90, 180, 270]
    global_row_count = 1
    
    for b_idx, config in enumerate(bins_config):
        f_count = config["front_count"]
        s_count = config["side_count"]
        bin_letter = chr(65 + b_idx) 
        sampler = qmc.LatinHypercube(d=12, seed=300 + b_idx)
        sample = sampler.random(n=samples_per_bin) 
        
        valid_bin_count = 0
        for row in sample:
            room_width = round(4.0 + row[0] * 2.0, 2)
            room_depth = round(4.0 + row[1] * 2.0, 2)
            canyon_hw = 0.0
            orientation = orientations[int(row[3] * len(orientations)) % len(orientations)]
            
            f_h = round(1.2 + row[4] * 0.6, 2)
            s_h = round(1.2 + row[5] * 0.6, 2)
            
            f_cv_min, f_cv_max = 0.4 + f_h/2, 2.5 - f_h/2
            f_cv = round(f_cv_min + row[6] * (f_cv_max - f_cv_min), 2)
            
            s_cv_min, s_cv_max = 0.4 + s_h/2, 2.5 - s_h/2
            s_cv = round(s_cv_min + row[7] * (s_cv_max - s_cv_min), 2)
            
            if f_count == 1:
                f_w_ratio = 0.20 + row[8] * 0.40
                f_w = round(f_w_ratio * room_width, 2)
                f_ch1_ratio_min = 0.05 + f_w_ratio/2
                f_ch1_ratio_max = 0.95 - f_w_ratio/2
                f_ch1 = round((f_ch1_ratio_min + row[9] * (f_ch1_ratio_max - f_ch1_ratio_min)) * room_width, 2)
                f_ch2 = 0.0
            else:
                f_w_ratio = 0.15 + row[8] * 0.20
                f_w = round(f_w_ratio * room_width, 2)
                f_c1_min, f_c1_max = 0.05 + f_w_ratio/2, 0.475 - f_w_ratio/2
                f_ch1 = round((f_c1_min + row[9] * (f_c1_max - f_c1_min)) * room_width, 2)
                f_c2_min, f_c2_max = 0.525 + f_w_ratio/2, 0.95 - f_w_ratio/2
                f_ch2 = round((f_c2_min + row[10] * (f_c2_max - f_c2_min)) * room_width, 2)
                
            if s_count == 0:
                s_w, s_ch1, s_ch2 = 0.0, 0.0, 0.0
            elif s_count == 1:
                s_w_ratio = 0.20 + row[11] * 0.40
                s_w = round(s_w_ratio * room_depth, 2)
                s_ch1_min = 0.05 + s_w_ratio/2
                s_ch1_max = 0.95 - s_w_ratio/2
                s_ch1 = round((s_ch1_min + row[10] * (s_ch1_max - s_ch1_min)) * room_depth, 2)
                s_ch2 = 0.0
            else:
                s_w_ratio = 0.15 + row[11] * 0.20
                s_w = round(s_w_ratio * room_depth, 2)
                s_c1_min, s_c1_max = 0.05 + s_w_ratio/2, 0.475 - s_w_ratio/2
                s_ch1 = round((s_c1_min + row[9] * (s_c1_max - s_c1_min)) * room_depth, 2)
                s_c2_min, s_c2_max = 0.525 + s_w_ratio/2, 0.95 - s_w_ratio/2
                s_ch2 = round((s_c2_min + row[10] * (s_c2_max - s_c2_min)) * room_depth, 2)
                
            iteration_num = valid_bin_count + 1
            iteration_id = f"{bin_letter}_Rotated_{iteration_num}"
            
            all_valid_data.append({
                "Row_Nr": global_row_count,
                "Iteration_ID": iteration_id,
                "Bin_ID": f"Bin_{bin_letter}",
                "Room_Width_m": room_width,
                "Room_Depth_m": room_depth,
                "Orientation_deg": orientation,
                "Canyon_HW": canyon_hw,
                "Front_Win1_Center_H_m": f_ch1,
                "Front_Win1_Center_V_m": f_cv,
                "Front_Win1_Width_m": f_w,
                "Front_Win1_Height_m": f_h,
                "Front_Win2_Center_H_m": f_ch2,
                "Front_Win2_Center_V_m": f_cv if f_count == 2 else 0.0,
                "Front_Win2_Width_m": f_w if f_count == 2 else 0.0,
                "Front_Win2_Height_m": f_h if f_count == 2 else 0.0,
                "Side_Win1_Center_H_m": s_ch1,
                "Side_Win1_Center_V_m": s_cv if s_count > 0 else 0.0,
                "Side_Win1_Width_m": s_w,
                "Side_Win1_Height_m": s_h if s_count > 0 else 0.0,
                "Side_Win2_Center_H_m": s_ch2,
                "Side_Win2_Center_V_m": s_cv if s_count == 2 else 0.0,
                "Side_Win2_Width_m": s_w if s_count == 2 else 0.0,
                "Side_Win2_Height_m": s_h if s_count == 2 else 0.0,
            })
            
            valid_bin_count += 1
            global_row_count += 1
            
    return pd.DataFrame(all_valid_data)

def generate_bin_F(num_rooms=1125, batch_size=125, seed=42):
    dataset = []
    cutout_positions = ['TopRight', 'TopLeft', 'BottomRight', 'BottomLeft']
    num_batches = math.ceil(num_rooms / batch_size)
    room_counter = 0
    
    for b in range(num_batches):
        n_in_batch = min(batch_size, num_rooms - room_counter)
        sampler = qmc.LatinHypercube(d=7, seed=seed + b)
        lhs_samples = sampler.random(n=n_in_batch)
        
        for idx in range(n_in_batch):
            lhs = lhs_samples[idx]
            W = 5.0 + lhs[0] * 7.0
            D = 5.0 + lhs[1] * 7.0
            CW = min((0.2 + lhs[2] * 0.4) * W, W - 2.5)
            CD = min((0.2 + lhs[3] * 0.4) * D, D - 2.5)
            win_w_r = 0.2 + lhs[4] * 0.6 
            win_h = 1.2 + lhs[5] * 1.2 
            max_sill = 2.65 - win_h
            sill_h = 0.05 + lhs[6] * (max_sill - 0.05) 
            cz = sill_h + (win_h / 2.0)
            cutout_pos = cutout_positions[room_counter % 4]
            
            if cutout_pos == 'TopRight':
                corners = [(0,0), (W,0), (W, D-CD), (W-CW, D-CD), (W-CW, D), (0,D)]
            elif cutout_pos == 'TopLeft':
                corners = [(0,0), (W,0), (W,D), (CW, D), (CW, D-CD), (0, D-CD)]
            elif cutout_pos == 'BottomRight':
                corners = [(0,0), (W-CW,0), (W-CW, CD), (W, CD), (W,D), (0,D)]
            else:
                corners = [(CW,0), (W,0), (W,D), (0,D), (0, CD), (CW, CD)]
                
            walls = []
            for j in range(6):
                p1 = corners[j]
                p2 = corners[(j+1) % 6]
                vx, vy = p2[0] - p1[0], p2[1] - p1[1]
                length = math.hypot(vx, vy)
                nx, ny = vy / length, -vx / length
                walls.append({'idx': j, 'p1': p1, 'p2': p2, 'length': length, 'nx': nx, 'ny': ny})
                
            valid_walls = [w for w in walls if w['length'] >= 1.5]
            sun_walls = [w for w in valid_walls if w['ny'] < -0.1 or w['nx'] > 0.1 or w['nx'] < -0.1]
            if not sun_walls:
                sun_walls = valid_walls
                
            num_windows = (room_counter % 3) + 1
            first_wall = sun_walls[room_counter % len(sun_walls)]
            walls_to_use = [first_wall]
            
            remaining_walls = [w for w in valid_walls if w['idx'] != first_wall['idx']]
            if num_windows > 1 and remaining_walls:
                step = room_counter % len(remaining_walls)
                additional = remaining_walls[step:] + remaining_walls[:step]
                walls_to_use.extend(additional[:num_windows - 1])
                
            windows = []
            for w in walls_to_use:
                max_width = w['length'] - 0.4 
                win_width = min(win_w_r * w['length'], max_width)
                win_width = max(0.8, win_width)
                cx = (w['p1'][0] + w['p2'][0]) / 2.0
                cy = (w['p1'][1] + w['p2'][1]) / 2.0
                windows.append({'center': (cx, cy, cz), 'width': win_width, 'height': win_h})
                
            corners_str = ";".join([f"{{{round(c[0],3)},{round(c[1],3)}}}" for c in corners])
            centers_str = ";".join([f"{{{round(w['center'][0],3)},{round(w['center'][1],3)},{round(w['center'][2],3)}}}" for w in windows])
            widths_str = ";".join([str(round(w['width'],3)) for w in windows])
            heights_str = ";".join([str(round(w['height'],3)) for w in windows])
            
            dataset.append({
                "Iteration_Name": f"Expert_LShape_{room_counter+1}",
                "Orientation": 0.0, 
                "Room_corners": corners_str,
                "Window_centerpoints": centers_str,
                "Window_widths": widths_str,
                "Window_heights": heights_str,
                "Canyon_HW": 0.0 
            })
            
            room_counter += 1
            
    return pd.DataFrame(dataset)

def main():
    print("Bins A-D")
    df_aligned = generate_bins_A_to_D()
    df_aligned.to_csv("bins_A-D_rectangular.csv", index=False)
    
    print("Bin E")
    df_rotated = generate_bin_E()
    df_rotated.to_csv("bin_E_rotated.csv", index=False)
    
    print("Bin F")
    df_lshape = generate_bin_F()
    df_lshape.to_csv("bin_F_lshaped.csv", index=False)

if __name__ == "__main__":
    main()