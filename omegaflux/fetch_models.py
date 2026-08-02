import os
import urllib.request
from urllib.error import URLError

# =============================================================================
# CONFIGURATION
# =============================================================================
# Using the GitHub 'latest' release download endpoint based on the repo URL
BASE_URL = "https://github.com/Curiosit/omegaflux/releases/latest/download/"
OUT_DIR = "models"

MODELS = [
    "omegaflux_xgb_low.joblib",
    "omegaflux_xgb_up.joblib",
    "omegaflux_low.onnx",
    "omegaflux_up.onnx"
]

# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def download_model(filename):
    url = BASE_URL + filename
    out_path = os.path.join(OUT_DIR, filename)
    
    print(f"Fetching {filename}...")
    try:
        # Download the file
        urllib.request.urlretrieve(url, out_path)
        
        # Calculate and display file size
        mb_size = os.path.getsize(out_path) / (1024 ** 2)
        print(f"[OK] Saved to {out_path} ({mb_size:.1f} MB)")
        
    except URLError as e:
        print(f"[ERROR] Failed to download {filename}. Reason: {e.reason}")
        print(f"        Please verify the release assets exist at {url}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 78)
    print("OmegaFlux - Fetch Pre-trained Models")
    print("=" * 78)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    for model_file in MODELS:
        download_model(model_file)
        
    print("\nDONE")

if __name__ == "__main__":
    main()