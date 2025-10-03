#!/usr/bin/env python
"""
Setup script for health monitoring system
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install required packages."""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "psutil",  # For system metrics
            "tk",      # For GUI (usually included with Python)
        ])
        print("Successfully installed requirements")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False


def create_directories():
    """Create required directories."""
    project_root = Path(__file__).parent

    dirs = [
        project_root / "reports",
        project_root / "logs",
    ]

    for dir_path in dirs:
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")


def main():
    """Main setup function."""
    print("Setting up health monitoring system...")

    if install_requirements():
        create_directories()
        print("\nSetup complete!")
        print("\nTo start the monitoring interface:")
        print("1. Navigate to the project root")
        print("2. Run: python scripts/health_monitor.py")
    else:
        print("\nSetup failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
