# Environment Verification Script for Negative_Space_Imaging_Project

import sys
import subprocess
import json

REQUIRED_PYTHON = (3, 13, 6)
REQUIRED_PACKAGES = [
    "numpy",
    "psutil",
    "matplotlib",
    "pillow",
    "contourpy",
    "fastapi",
    "cryptography",
    "bandit",
    "pytest"
]

results = {}

# Check Python version
def check_python_version():
    version = sys.version_info
    results["python_version"] = f"{version.major}.{version.minor}.{version.micro}"
    results["python_version_ok"] = version >= REQUIRED_PYTHON

# Check required packages
def check_packages():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    results["missing_packages"] = missing
    results["all_packages_ok"] = len(missing) == 0

# Check pip install reproducibility
def check_pip_install():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--dry-run"])
        results["pip_install_ok"] = True
    except Exception as e:
        results["pip_install_ok"] = False
        results["pip_error"] = str(e)

if __name__ == "__main__":
    check_python_version()
    check_packages()
    check_pip_install()
    with open("environment_verification.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Environment verification complete. See environment_verification.json for details.")
