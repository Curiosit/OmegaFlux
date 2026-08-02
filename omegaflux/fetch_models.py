import os
import shutil
import urllib.request

# Directories
MODELS_DIR = "models"
WEB_MODELS_DIR = os.path.join("apps", "web", "public", "models")

# GitHub release URL base (update tag if necessary)
GITHUB_RELEASE_URL = "https://github.com/Curiosit/omegaflux/releases/download/v0.1.0/"

FILES = [
    "omegaflux_xgb_low.joblib",
    "omegaflux_xgb_up.joblib",
    "omegaflux_low.onnx",
    "omegaflux_up.onnx",
]

def main():
    print("=" * 78)
    print("OmegaFlux - Fetching Pre-trained Models")
    print("=" * 78)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(WEB_MODELS_DIR, exist_ok=True)

    for filename in FILES:
        dest_path = os.path.join(MODELS_DIR, filename)
        url = GITHUB_RELEASE_URL + filename
        print(f"Downloading {filename}...")
        
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"[OK] Saved to {dest_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            continue

        # If it's an ONNX file, also copy it to the web app public models folder
        if filename.endswith(".onnx"):
            web_dest = os.path.join(WEB_MODELS_DIR, filename)
            shutil.copy(dest_path, web_dest)
            print(f"[OK] Copied ONNX model to web app: {web_dest}")

    print("\n[DONE] Model fetching complete.")

if __name__ == "__main__":
    main()