import os
import subprocess
import sys
import venv

ENV_DIR = ".venv"

def setup_environment():
    print("=== Setting up OmegaFlux Environment ===")
    
    if not os.path.exists(ENV_DIR):
        print("Creating virtual environment...")
        venv.create(ENV_DIR, with_pip=True)
    else:
        print("Virtual environment already exists.")

    pip_executable = os.path.join(ENV_DIR, "Scripts", "pip") if os.name == "nt" else os.path.join(ENV_DIR, "bin", "pip")
    
    print("Installing base package and dev dependencies...")
    subprocess.run([pip_executable, "install", "-e", ".[dev]"], check=True)
    
    print("Installing Hops optional dependencies...")
    subprocess.run([pip_executable, "install", "-e", ".[hops]"], check=True)
    
    print("\n[OK] Environment setup complete.")
    print(f"Activate it using: source {ENV_DIR}/bin/activate (Linux/Mac) or {ENV_DIR}\\Scripts\\activate (Windows)")

if __name__ == "__main__":
    setup_environment()