import argparse
import subprocess
import sys

def run_pipeline(mode):
    print("=== Starting OmegaFlux Pipeline ===")
    
    steps = [
        ("Unzip Raw Data", ["python", "-m", "omegaflux.unzip_data"]),
        ("Feature Extraction", ["python", "-m", "omegaflux.extract_features"]),
        ("Model Training", ["python", "-m", "omegaflux.train", "--mode", mode]),
        ("TreeSHAP Attribution", ["python", "-m", "omegaflux.explain"]),
        ("Figure Generation", ["python", "-m", "omegaflux.make_figures"]),
        ("ONNX Model Export", ["python", "-m", "omegaflux.export_onnx"])
    ]
    
    for step_name, command in steps:
        print(f"\n--- Running: {step_name} ---")
        try:
            subprocess.run(command, check=True)
            print(f"[OK] {step_name} completed successfully.")
        except subprocess.CalledProcessError:
            print(f"[ERROR] {step_name} failed. Exiting pipeline.")
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmegaFlux Master Pipeline")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "full"], 
                        help="Run mode for training: 'fast' (no tuning) or 'full'")
    args = parser.parse_args()
    
    run_pipeline(args.mode)