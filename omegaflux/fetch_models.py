import argparse
import os
import shutil
import urllib.request

# Pre-trained models are attached to GitHub releases, one folder per feature set.
REPO_RELEASES = "https://github.com/Curiosit/omegaflux/releases/download"
DEFAULT_TAG = "v1.1.0"
WEB_MODELS_DIR = os.path.join("apps", "web", "public", "models")
WEB_NAMES = {"omegaflux_low.onnx": "OmegaFluxLow.onnx", "omegaflux_up.onnx": "OmegaFluxUp.onnx"}

FILES = [
    "omegaflux_xgb_low.joblib",          # Hops service
    "omegaflux_xgb_up.joblib",
    "omegaflux_xgb_low.json",            # omegaflux.predict
    "omegaflux_xgb_up.json",
    "omegaflux_low.onnx",
    "omegaflux_up.onnx",
    "features.json",
]


def main():
    ap = argparse.ArgumentParser(description="Download pre-trained OmegaFlux models")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="release tag (v1.0.0 = original 26-descriptor models)")
    ap.add_argument("--out", default=os.path.join("OmegaFlux_Models", "core"))
    ap.add_argument("--web", action="store_true", help=f"also copy ONNX models and features.json to {WEB_MODELS_DIR}")
    args = ap.parse_args()

    print("=" * 78)
    print(f"OmegaFlux - Fetching pre-trained models ({args.tag})")
    print("=" * 78)
    os.makedirs(args.out, exist_ok=True)

    for filename in FILES:
        dest = os.path.join(args.out, filename)
        url = f"{REPO_RELEASES}/{args.tag}/{filename}"
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[OK] Saved to {dest}")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            continue
        if args.web and (filename in WEB_NAMES or filename == "features.json"):
            os.makedirs(WEB_MODELS_DIR, exist_ok=True)
            target = os.path.join(WEB_MODELS_DIR, WEB_NAMES.get(filename, filename))
            shutil.copy(dest, target)
            print(f"[OK] Copied to {target}")

    print("\n[DONE] Model fetching complete.")


if __name__ == "__main__":
    main()
