import argparse
import json
import os
import shutil

import onnxmltools
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType

# =============================================================================
# CONFIGURATION
# =============================================================================
OUT_DIR = "OmegaFlux_Models/core"
WEB_MODELS_DIR = os.path.join("apps", "web", "public", "models")
WEB_NAMES = {"low": "OmegaFluxLow.onnx", "up": "OmegaFluxUp.onnx"}   # names the web app loads


def load_features(out_dir):
    fj = os.path.join(out_dir, "features.json")
    if not os.path.exists(fj):
        raise SystemExit(f"[ERROR] {fj} not found. Train with omegaflux.train first.")
    with open(fj) as fh:
        return json.load(fh)["features"]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def export_model_to_onnx(out_dir, model_prefix, n_features):
    json_path = os.path.join(out_dir, f"omegaflux_xgb_{model_prefix}.json")
    onnx_path = os.path.join(out_dir, f"omegaflux_{model_prefix}.onnx")

    if not os.path.exists(json_path):
        print(f"[ERROR] Trained model not found at {json_path}. Run train.py first.")
        return None

    print(f"Loading XGBoost model: {json_path}")
    model = xgb.XGBRegressor()
    model.load_model(json_path)
    booster = model.get_booster()
    # Early stopping keeps EARLY_STOPPING_ROUNDS trees beyond the best iteration. XGBoost's own
    # predict() ignores them; the ONNX graph would not, so cut the booster at the best iteration.
    best = getattr(model, "best_iteration", None)
    if best is not None and best + 1 < booster.num_boosted_rounds():
        print(f"Trimming booster to best iteration: {best + 1} of {booster.num_boosted_rounds()} rounds")
        booster = booster[: best + 1]
    booster.feature_names = None          # onnxmltools expects positional f0..fN names

    print(f"Converting {model_prefix.upper()} model to ONNX ({n_features} inputs)...")
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_xgboost(booster, initial_types=initial_type)

    onnxmltools.utils.save_model(onnx_model, onnx_path)
    mb_size = os.path.getsize(onnx_path) / (1024 ** 2)
    print(f"[OK] Exported {onnx_path} ({mb_size:.1f} MB)\n")
    return onnx_path


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Export OmegaFlux models to ONNX")
    ap.add_argument("--out", default=OUT_DIR, help="model folder (contains features.json)")
    ap.add_argument("--web", action="store_true",
                    help=f"also copy the models and features.json into {WEB_MODELS_DIR}")
    args = ap.parse_args()

    print("=" * 78)
    print("OmegaFlux - ONNX Model Export")
    print("=" * 78)

    features = load_features(args.out)
    print(f"{len(features)} input features from {args.out}/features.json")

    exported = {p: export_model_to_onnx(args.out, p, len(features)) for p in ("low", "up")}

    if args.web and all(exported.values()):
        os.makedirs(WEB_MODELS_DIR, exist_ok=True)
        for p, path in exported.items():
            shutil.copy(path, os.path.join(WEB_MODELS_DIR, WEB_NAMES[p]))
        shutil.copy(os.path.join(args.out, "features.json"), os.path.join(WEB_MODELS_DIR, "features.json"))
        print(f"[OK] Copied ONNX models and features.json to {WEB_MODELS_DIR}/")

    print("DONE")


if __name__ == "__main__":
    main()
