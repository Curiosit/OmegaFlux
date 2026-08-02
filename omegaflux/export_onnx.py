import os
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# =============================================================================
# CONFIGURATION
# =============================================================================
OUT_DIR = "OmegaFlux_Models"
N_FEATURES = 26

# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def export_model_to_onnx(model_prefix):
    json_path = os.path.join(OUT_DIR, f"omegaflux_xgb_{model_prefix}.json")
    onnx_path = os.path.join(OUT_DIR, f"omegaflux_{model_prefix}.onnx")
    
    if not os.path.exists(json_path):
        print(f"[ERROR] Trained model not found at {json_path}. Run train.py first.")
        return

    print(f"Loading XGBoost model: {json_path}")
    model = xgb.XGBRegressor()
    model.load_model(json_path)

    print(f"Converting {model_prefix.upper()} model to ONNX format...")
    # Define the input shape: [None, 26] allows for variable batch sizes with exactly 26 features
    initial_type = [('float_input', FloatTensorType([None, N_FEATURES]))]
    
    # Convert the model
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

    print(f"Saving ONNX model: {onnx_path}")
    onnxmltools.utils.save_model(onnx_model, onnx_path)
    
    mb_size = os.path.getsize(onnx_path) / (1024 ** 2)
    print(f"[OK] Exported {onnx_path} ({mb_size:.1f} MB)\n")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 78)
    print("OmegaFlux - ONNX Model Export")
    print("=" * 78)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    export_model_to_onnx("low")
    export_model_to_onnx("up")
    
    print("DONE")

if __name__ == "__main__":
    main()