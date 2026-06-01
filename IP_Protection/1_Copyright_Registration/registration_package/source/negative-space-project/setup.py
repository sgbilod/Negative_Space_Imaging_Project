"""
Main setup script for the Negative Space Imaging Project.
This script will:
1. Set up the Python environment
2. Install required dependencies
3. Verify the installation
4. Create test data directories
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80)

def run_command(command, description=None):
    """Run a shell command and print its output"""
    if description:
        print(f"\n> {description}...")
    
    print(f"$ {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(f"Error: {result.stderr}")
    
    return result.returncode == 0

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    major, minor, _ = sys.version_info
    print(f"Detected Python {major}.{minor}")
    
    if major < 3 or (major == 3 and minor < 6):
        print("Error: This project requires Python 3.6 or higher")
        return False
    
    return True

def setup_virtual_environment():
    """Set up a virtual environment for the project"""
    print_header("Setting Up Virtual Environment")
    
    # Check if venv module is available
    try:
        import venv
        print("venv module is available")
    except ImportError:
        print("Error: venv module not available. Please install it or use virtualenv.")
        return False
    
    # Create virtual environment if it doesn't exist
    venv_dir = Path("venv")
    if venv_dir.exists():
        print("Virtual environment already exists")
    else:
        print("Creating virtual environment...")
        try:
            import venv
            venv.create(venv_dir, with_pip=True)
            print("Virtual environment created successfully")
        except Exception as e:
            print(f"Error creating virtual environment: {str(e)}")
            return False
    
    return True

def install_dependencies():
    """Install project dependencies from requirements.txt"""
    print_header("Installing Dependencies")
    
    # Determine pip command based on OS
    if platform.system() == "Windows":
        pip_cmd = r"venv\Scripts\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing required packages"):
        return False
    
    return True

def verify_installation():
    """Verify that key packages were installed correctly"""
    print_header("Verifying Installation")
    
    # Determine Python command based on OS
    if platform.system() == "Windows":
        python_cmd = r"venv\Scripts\python"
    else:
        python_cmd = "venv/bin/python"
    
    # List of key packages to verify
    key_packages = [
        "numpy",
        "opencv-python",
        "torch",
        "scikit-image",
        "matplotlib"
    ]
    
    all_installed = True
    for package in key_packages:
        cmd = f'{python_cmd} -c "import {package.replace("-", "_")}; print(\'{package} installed successfully\')"'
        success = run_command(cmd)
        if not success:
            print(f"Failed to import {package}")
            all_installed = False
    
    return all_installed

def create_test_directories():
    """Create directories for test data and outputs"""
    print_header("Creating Project Directories")
    
    directories = [
        "data/images",
        "data/depth_maps",
        "data/calibration",
        "data/metadata",
        "output/models",
        "output/visualizations",
        "output/signatures"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Creating directory: {directory}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Directory already exists: {directory}")
    
    return True

def main():
    """Main setup function"""
    print_header("Negative Space Imaging Project Setup")
    
    # Steps to complete setup
    if not check_python_version():
        return False
    
    if not setup_virtual_environment():
        return False
    
    if not install_dependencies():
        return False
    
    if not verify_installation():
        print("\nWarning: Some dependencies could not be verified.")
        print("You may need to install additional system libraries or address compatibility issues.")
    
    if not create_test_directories():
        return False
    
    print_header("Setup Complete")
    print("The Negative Space Imaging Project has been set up successfully!")
    print("\nTo activate the virtual environment:")
    if platform.system() == "Windows":
        print("    venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")
    
    print("\nTo run a test script:")
    if platform.system() == "Windows":
        print("    venv\\Scripts\\python tests\\unit_tests\\test_basic.py")
    else:
        print("    venv/bin/python tests/unit_tests/test_basic.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
