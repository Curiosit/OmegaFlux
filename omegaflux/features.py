"""
Single source of truth for the OmegaFlux descriptor sets.

Every consumer (training, attribution, figures, ONNX export, prediction, the web app and the
Hops service) selects its input columns from FEATURES_ALL by name. Subsets keep extraction
order, so a model input vector is always a subset of the 26-vector in the same order.
"""

# All 26 descriptors, in the order extract_features.py (and engine.ts / gh.py) produce them.
FEATURES_ALL = [
    "Total_Omega", "Global_Cos_Theta", "Global_Sill_Clearance", "Global_Header_Clearance",
    "Global_inv_dmin2", "Opaque_Wall_dmin", "Inner_Corner_Dmin", "Outer_Corner_Dmin",
    "South_Weighted_Omega", "East_Weighted_Omega", "West_Weighted_Omega",
    "Annual_Direct_Sun_Fraction", "Smooth_Mean_Depth", "Smooth_Depth_StdDev",
    "Min_Occluder_Distance", "Window_Angular_Spread", "Window_Sky_Weighted_Omega",
    "Occlusion_Multiplier", "Visible_Window_Count", "Mean_Window_Distance",
    "Dominant_Win_Plane_Dist", "Dominant_Win_H_Offset", "Dominant_Win_Ray_Forward",
    "Dominant_Win_Ray_Backward", "Dominant_Win_Ray_Left", "Dominant_Win_Ray_Right",
]

# The 12 descriptors chosen by forward selection (combined in-distribution / transfer
# objective). Listed here in selection order for reference; FEATURES_CORE below is in
# extraction order.
CORE_SELECTION_ORDER = [
    "Window_Sky_Weighted_Omega", "Smooth_Depth_StdDev", "Global_Cos_Theta",
    "South_Weighted_Omega", "Window_Angular_Spread", "Outer_Corner_Dmin",
    "Annual_Direct_Sun_Fraction", "East_Weighted_Omega", "Total_Omega",
    "Dominant_Win_Ray_Forward", "West_Weighted_Omega", "Global_Sill_Clearance",
]
FEATURES_CORE = [f for f in FEATURES_ALL if f in set(CORE_SELECTION_ORDER)]

FEATURE_SETS = {"all": FEATURES_ALL, "core": FEATURES_CORE}

# Descriptor definitions that changed between releases (recorded in features.json).
FEATURE_CHANGES = {
    "Global_Sill_Clearance": "signed since v1.1.0 (negative when the sill is above the work plane); clipped at 0 in v1.0.0",
}


def indices(names):
    """Positions of `names` within FEATURES_ALL (for consumers that build the full 26-vector)."""
    pos = {f: i for i, f in enumerate(FEATURES_ALL)}
    missing = [n for n in names if n not in pos]
    if missing:
        raise KeyError(f"not in FEATURES_ALL: {missing}")
    return [pos[n] for n in names]
