#!/usr/bin/env python3
"""
Setup script to create a virtual environment and install dependencies using uv.
This script:
1. Checks if uv is installed, installs it if not
2. Creates a virtual environment using uv
3. Installs dependencies from pyproject.toml
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, check=True, shell=False):
    """Run a command and return its output."""
    print(f"Running: {command}")
    if isinstance(command, str) and not shell:
        command = command.split()
    
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check,
        shell=shell
    )
    
    return result

def check_uv_installed():
    """Check if uv is installed."""
    try:
        result = run_command("uv --version", check=False)
        return result.returncode == 0
    except Exception:
        return False

def install_uv():
    """Install uv using pip."""
    print("Installing uv...")
    run_command([sys.executable, "-m", "pip", "install", "uv"])

def setup_environment():
    """Set up the virtual environment and install dependencies."""
    # Check if uv is installed
    if not check_uv_installed():
        install_uv()
    
    # Create virtual environment
    print("Creating virtual environment...")
    venv_path = Path(".venv")
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
    else:
        run_command("uv venv")
    
    # Install dependencies
    print("Installing dependencies...")
    
    # Determine the activate script path based on OS
    if sys.platform == "win32":
        activate_script = ".venv\\Scripts\\activate"
        activate_cmd = f"call {activate_script} && uv pip install -e ."
        # Run the activation and installation in a shell
        run_command(activate_cmd, shell=True)
    else:
        # For Unix systems, use bash explicitly
        activate_script = ".venv/bin/activate"
        # Use bash explicitly with the -c option
        run_command(["/bin/bash", "-c", f"source {activate_script} && uv pip install -e ."])
    
    print("\nSetup complete! To activate the virtual environment, run:")
    if sys.platform == "win32":
        print(".venv\\Scripts\\activate")
    else:
        print("source .venv/bin/activate")

if __name__ == "__main__":
    setup_environment()