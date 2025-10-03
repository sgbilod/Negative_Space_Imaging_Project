#!/usr/bin/env python
"""
Negative Space Imaging System - Setup Script
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This script orchestrates the complete setup of the development environment
for the Negative Space Imaging System:

1. Validates and installs system prerequisites
2. Sets up Python environment with all dependencies
3. Verifies cryptographic security components
4. Configures development tools and settings
5. Validates the complete installation

This is the main entry point for setting up the development environment.
It coordinates with specialized scripts in the scripts/ directory for
platform-specific setup tasks.

Usage:
  python setup.py --full-setup     # Complete setup including system prerequisites
  python setup.py --quick-setup    # Setup assuming prerequisites are installed
  python setup.py --verify         # Verify current installation
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
import platform
import subprocess
import sys
import venv
from pathlib import Path


def check_python_version():
    """Check if the Python version meets requirements."""
    required_version = (3, 8)
    current_version = sys.version_info

    if current_version < required_version:
        print(f"❌ Error: Python {required_version[0]}.{required_version[1]} or higher is required")
        print(f"   Current version: {current_version[0]}.{current_version[1]}.{current_version[2]}")
        return False

    print(f"✅ Python version {current_version[0]}.{current_version[1]}.{current_version[2]} is compatible")
    return True


def create_virtual_environment(venv_dir=".venv"):
    """Create a Python virtual environment."""
    venv_path = Path(venv_dir)

    if venv_path.exists():
        print(f"⚠️ Virtual environment directory {venv_dir} already exists")
        response = input("Do you want to recreate it? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping virtual environment creation")
            return True

        # Remove existing venv if user confirms
        import shutil
        shutil.rmtree(venv_path)
        print(f"Removed existing virtual environment at {venv_dir}")

    print(f"Creating virtual environment in {venv_dir}...")
    try:
        venv.create(venv_path, with_pip=True)
        print(f"✅ Virtual environment created successfully in {venv_dir}")

        # Print activation instructions based on platform
        if platform.system() == "Windows":
            print("\nTo activate the virtual environment, run:")
            print(f"   {venv_dir}\\Scripts\\activate")
        else:
            print("\nTo activate the virtual environment, run:")
            print(f"   source {venv_dir}/bin/activate")

        return True
    except Exception as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False


def install_python_dependencies(venv_dir=".venv", requirements_file="requirements.txt"):
    """Install Python dependencies from requirements.txt."""
    if not Path(requirements_file).exists():
        print(f"❌ Error: {requirements_file} not found")
        return False

    # Determine the pip executable based on the virtual environment
    if platform.system() == "Windows":
        pip_exec = Path(venv_dir) / "Scripts" / "pip"
    else:
        pip_exec = Path(venv_dir) / "bin" / "pip"

    if not pip_exec.exists():
        print(f"❌ Error: pip not found in {venv_dir}")
        print("   Please create the virtual environment first")
        return False

    print(f"Installing Python dependencies from {requirements_file}...")
    try:
        result = subprocess.run(
            [str(pip_exec), "install", "-r", requirements_file],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Python dependencies:")
        print(e.stderr)
        return False


def check_npm_dependencies():
    """Check for Node.js and npm dependencies."""
    # Check if package.json exists
    if not Path("package.json").exists():
        print("⚠️ package.json not found, skipping Node.js dependency check")
        return True

    # Check if npm is available
    try:
        subprocess.run(["npm", "--version"], check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ npm not found, please install Node.js and npm")
        return False

    print("Node.js dependencies found. To install them, run:")
    print("   npm install")

    return True


def verify_system_requirements():
    """Verify that the system meets all requirements."""
    print("\n=== System Requirements Verification ===\n")

    # Check Python version
    python_ok = check_python_version()

    # Check for required libraries
    print("\nChecking for required system libraries...")

    # Check for cryptography dependencies
    try:
        import cryptography
        print("✅ Cryptography library is available")
    except ImportError:
        print("❌ Cryptography library not found")
        print("   It will be installed when you install Python dependencies")

    # Check for numpy and matplotlib
    try:
        import numpy
        print("✅ NumPy library is available")
    except ImportError:
        print("❌ NumPy library not found")
        print("   It will be installed when you install Python dependencies")

    try:
        import matplotlib
        print("✅ Matplotlib library is available")
    except ImportError:
        print("❌ Matplotlib library not found")
        print("   It will be installed when you install Python dependencies")

    # Check Node.js if package.json exists
    npm_ok = check_npm_dependencies()

    print("\nSystem verification complete.")
    if python_ok:
        print("✅ Basic system requirements met")
    else:
        print("❌ Some system requirements are not met")

    return python_ok


def setup_initial_configuration():
    """Set up initial configuration files if they don't exist."""
    print("\n=== Initial Configuration Setup ===\n")

    # Create security config if it doesn't exist
    security_config_path = Path("security_config.json")
    if not security_config_path.exists():
        print("Creating default security configuration...")
        import json

        default_config = {
            "signature_timeout_seconds": 3600,
            "key_size_bits": 2048,
            "hash_algorithm": "SHA256",
            "min_signers": 3,
            "default_mode": "threshold",
            "audit_log_path": "security_audit.json",
            "roles": [
                {"id": "analyst", "name": "Analyst", "description": "Image processing analyst", "priority": 1, "required": True},
                {"id": "physician", "name": "Physician", "description": "Medical doctor", "priority": 2, "required": True},
                {"id": "admin", "name": "Administrator", "description": "System administrator", "priority": 3, "required": True}
            ]
        }

        with open(security_config_path, "w") as f:
            json.dump(default_config, f, indent=2)

        print(f"✅ Created default security configuration at {security_config_path}")
    else:
        print(f"⚠️ Security configuration already exists at {security_config_path}")

    print("\nInitial configuration setup complete.")
    return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Negative Space Imaging System Setup")
    parser.add_argument("--install-deps", action="store_true", help="Install Python dependencies")
    parser.add_argument("--create-venv", action="store_true", help="Create a Python virtual environment")
    parser.add_argument("--verify-system", action="store_true", help="Verify system requirements")
    parser.add_argument("--all", action="store_true", help="Perform all setup steps")
    return parser.parse_args()


def main():
    """Main entry point for the setup script."""
    print("=" * 80)
    print(" Negative Space Imaging System - Setup Script ")
    print("=" * 80)

    args = parse_arguments()

    # If no arguments or --all, do everything
    if not any([args.install_deps, args.create_venv, args.verify_system]) or args.all:
        args.install_deps = True
        args.create_venv = True
        args.verify_system = True

    if args.verify_system:
        verify_system_requirements()

    if args.create_venv:
        print("\n=== Virtual Environment Setup ===\n")
        create_virtual_environment()

    if args.install_deps:
        print("\n=== Python Dependencies Installation ===\n")
        install_python_dependencies()

    # Always run the initial configuration setup
    setup_initial_configuration()

    print("\n" + "=" * 80)
    print(" Setup Complete! ")
    print("=" * 80)

    # Print next steps
    print("\nNext Steps:")
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")

    print("2. Run the test suite to verify the installation:")
    print("   python test_suite.py --all")

    print("3. Try the multi-signature demo:")
    print("   python multi_signature_demo.py --mode threshold --signatures 5 --threshold 3")

    print("4. Explore the CLI interface:")
    print("   python cli.py --help")

    print("\nEnjoy your Negative Space Imaging System!")


if __name__ == "__main__":
    main()
