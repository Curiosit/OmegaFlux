import subprocess
import sys

def run_pipeline():
    print("=== Starting OmegaFlux Pipeline ===")
    
    steps = [
        ("Feature Extraction", ["python", "-m", "omegaflux.extract_features"]),
        ("Model Training", ["python", "-m", "omegaflux.train"]),
        ("TreeSHAP Attribution", ["python", "-m", "omegaflux.explain"]),
        ("Figure Generation", ["python", "-m", "omegaflux.make_figures"])
    ]
    
    for step_name, command in steps:
        print(f"\n--- Running: {step_name} ---")
        try:
            subprocess.run(command, check=True)
            print(f"[OK] {step_name} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {step_name} failed. Exiting pipeline.")
            sys.exit(1)

if __name__ == "__main__":
    run_pipeline()