import argparse
import os
import zipfile
import glob

DATA_RAW_DIR = os.path.join("data", "raw")
TARGET_FILES = [
    os.path.join("data", "study-ml-rect.xlsx"),
    os.path.join("data", "study-ml-rotated.xlsx"),
    os.path.join("data", "study-ml-shapes.csv"),
]

def main():
    parser = argparse.ArgumentParser(description="OmegaFlux Unzip Data")
    parser.add_argument("--force", action="store_true", help="Force re-extract zip files even if target files exist")
    args = parser.parse_args()

    print("=" * 78)
    print("OmegaFlux - Extracting Raw Data")
    print("=" * 78)
    
    # Skip if all extracted files already exist and --force was NOT provided
    if not args.force and all(os.path.exists(f) for f in TARGET_FILES):
        print("[INFO] All raw simulation files already exist in 'data/'.")
        print("[SKIP] Skipping extraction. (Use --force to re-extract)")
        return

    if not os.path.exists(DATA_RAW_DIR):
        print(f"[ERROR] Directory '{DATA_RAW_DIR}' does not exist.")
        print("Please ensure your raw data .zip files are placed in 'data/raw/'.")
        return

    zip_files = glob.glob(os.path.join(DATA_RAW_DIR, "*.zip"))
    
    if not zip_files:
        print(f"[INFO] No .zip files found in '{DATA_RAW_DIR}'.")
        return

    for zip_path in zip_files:
        filename = os.path.basename(zip_path)
        print(f"Extracting {filename}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("data/raw")
            print(f"[OK] Successfully extracted {filename}")
        except zipfile.BadZipFile:
            print(f"[ERROR] {filename} is corrupt or not a valid zip file.")
            
    print("\nDONE")

if __name__ == "__main__":
    main()