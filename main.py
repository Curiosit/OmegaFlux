import argparse
import subprocess
import sys

def run_pipeline(mode, force):
    print("=== Starting OmegaFlux Pipeline ===")
    py = sys.executable
    
    # Build commands, appending --force if requested
    unzip_cmd = [py, "-m", "omegaflux.unzip_data"]
    if force:
        unzip_cmd.append("--force")
        
    extract_cmd = [py, "-m", "omegaflux.extract_features"]
    if force:
        extract_cmd.append("--force")
        
    train_cmd = [py, "-m", "omegaflux.train", "--mode", mode]
    if force:
        train_cmd.append("--force")
    
    steps = [
        ("Unzip Raw Data", unzip_cmd),
        ("Feature Extraction", extract_cmd),
        ("Model Training", train_cmd),
        ("TreeSHAP Attribution", [py, "-m", "omegaflux.explain"]),
        ("Figure Generation", [py, "-m", "omegaflux.make_figures"]),
        ("ONNX Model Export", [py, "-m", "omegaflux.export_onnx"])
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
    parser.add_argument("--force", action="store_true", 
                        help="Force re-running unzip, extraction, and training even if output artifacts already exist")
    args = parser.parse_args()
    
    run_pipeline(args.mode, args.force)