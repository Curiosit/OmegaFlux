import os
import zipfile
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_RAW_DIR = os.path.join("data", "raw")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 78)
    print("OmegaFlux - Extracting Raw Data")
    print("=" * 78)
    
    if not os.path.exists(DATA_RAW_DIR):
        print(f"[ERROR] Directory '{DATA_RAW_DIR}' does not exist.")
        print("Please ensure your raw data .zip files are placed in 'data/raw/'.")
        return

    # Find all .zip files in the directory
    zip_files = glob.glob(os.path.join(DATA_RAW_DIR, "*.zip"))
    
    if not zip_files:
        print(f"[INFO] No .zip files found in '{DATA_RAW_DIR}'.")
        return

    for zip_path in zip_files:
        filename = os.path.basename(zip_path)
        print(f"Extracting {filename}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_RAW_DIR)
            print(f"[OK] Successfully extracted {filename}")
        except zipfile.BadZipFile:
            print(f"[ERROR] {filename} is corrupt or not a valid zip file.")
            
    print("\nDONE")

if __name__ == "__main__":
    main()