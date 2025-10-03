#!/usr/bin/env python
"""
Python Environment Configuration for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This script helps set up the Python environment for the Negative Space Imaging
Project by:

1. Validating the Python version
2. Creating a virtual environment if one doesn't exist
3. Installing required packages
4. Configuring environment variables
5. Running dependency validation

Usage:
    python setup_environment.py [--force]
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Constants
MIN_PYTHON_VERSION = (3, 8)
VENV_DIR = ".venv"
CONFIG_FILE = "environment_config.json"


def print_header(message: str) -> None:
    """Print a formatted header message."""
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80)


def print_step(message: str) -> None:
    """Print a step message."""
    print(f"\n📋 {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"❌ {message}")


def check_python_version() -> bool:
    """Check if the Python version meets the minimum requirements."""
    print_step("Checking Python version...")

    current_version = sys.version_info
    current_version_str = f"{current_version.major}.{current_version.minor}.{current_version.micro}"
    required_version_str = f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"

    if current_version.major < MIN_PYTHON_VERSION[0] or (
        current_version.major == MIN_PYTHON_VERSION[0] and
        current_version.minor < MIN_PYTHON_VERSION[1]
    ):
        print_error(
            f"Python {required_version_str}+ is required, but you have {current_version_str}"
        )
        return False

    print_success(f"Python version {current_version_str} meets requirements")
    return True


def setup_virtual_environment(force: bool = False) -> bool:
    """Create a virtual environment if it doesn't exist."""
    print_step("Setting up virtual environment...")

    venv_path = Path(VENV_DIR)

    # Check if venv already exists
    if venv_path.exists() and not force:
        print_success(f"Virtual environment already exists at {venv_path}")
        return True

    # Remove existing venv if force flag is set
    if venv_path.exists() and force:
        print_warning(f"Removing existing virtual environment at {venv_path}")
        shutil.rmtree(venv_path)

    # Create new virtual environment
    try:
        print(f"Creating virtual environment at {venv_path}...")
        venv.create(venv_path, with_pip=True)
        print_success(f"Virtual environment created at {venv_path}")
        return True
    except Exception as e:
        print_error(f"Failed to create virtual environment: {str(e)}")
        return False


def get_venv_python() -> str:
    """Get the path to the Python executable in the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def get_venv_pip() -> str:
    """Get the path to the pip executable in the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    return os.path.join(VENV_DIR, "bin", "pip")


def install_requirements() -> bool:
    """Install required packages from requirements.txt."""
    print_step("Installing required packages...")

    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_error(f"Requirements file not found: {requirements_file}")
        return False

    pip_path = get_venv_pip()

    try:
        print(f"Installing packages from {requirements_file}...")
        subprocess.check_call(
            [pip_path, "install", "-r", str(requirements_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print_success("All required packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install packages: {e}")
        return False


def validate_dependencies() -> bool:
    """Run dependency validator to check for any issues."""
    print_step("Validating dependencies...")

    validator_script = Path("dependency_validator.py")
    if not validator_script.exists():
        print_warning(f"Dependency validator not found: {validator_script}")
        return True  # Not critical, so return True

    python_path = get_venv_python()

    try:
        print(f"Running dependency validator...")
        result = subprocess.run(
            [python_path, str(validator_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0:
            print_success("All dependencies validated successfully")
            return True
        else:
            print_warning("Dependency validation found issues:")
            print(result.stdout)
            return False
    except Exception as e:
        print_warning(f"Failed to run dependency validator: {str(e)}")
        return False  # Not critical for environment setup


def setup_environment_variables() -> bool:
    """Setup environment variables for the project."""
    print_step("Setting up environment variables...")

    # Default configuration
    default_config = {
        "LOGGING_LEVEL": "INFO",
        "DATA_DIRECTORY": os.path.join(os.getcwd(), "data"),
        "TEMP_DIRECTORY": os.path.join(os.getcwd(), "temp"),
        "MAX_THREADS": 4,
    }

    config_path = Path(CONFIG_FILE)

    # Load existing config or create new one
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"Loaded existing configuration from {config_path}")
        except Exception as e:
            print_warning(f"Failed to load existing configuration: {str(e)}")
            config = default_config
    else:
        config = default_config
        try:
            with open(config_path, "w") as f:
                json.dump(config, indent=2, sort_keys=True, fp=f)
            print(f"Created new configuration file at {config_path}")
        except Exception as e:
            print_warning(f"Failed to create configuration file: {str(e)}")

    # Create directories if they don't exist
    for key in ["DATA_DIRECTORY", "TEMP_DIRECTORY"]:
        if key in config:
            directory = Path(config[key])
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    print(f"Created directory: {directory}")
                except Exception as e:
                    print_warning(f"Failed to create directory {directory}: {str(e)}")

    print_success("Environment variables configured")
    return True


def show_activation_instructions() -> None:
    """Show instructions for activating the virtual environment."""
    print_header("Virtual Environment Activation")

    if platform.system() == "Windows":
        print("\nTo activate the virtual environment, run:")
        print(f"    {VENV_DIR}\\Scripts\\activate")
    else:
        print("\nTo activate the virtual environment, run:")
        print(f"    source {VENV_DIR}/bin/activate")

    print("\nAfter activation, you can run the project commands directly.")


def main() -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Setup Python environment for Negative Space Imaging Project")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of virtual environment"
    )
    args = parser.parse_args()

    print_header("Negative Space Imaging Project - Environment Setup")

    # Step 1: Check Python version
    if not check_python_version():
        return 1

    # Step 2: Setup virtual environment
    if not setup_virtual_environment(args.force):
        return 1

    # Step 3: Install requirements
    if not install_requirements():
        return 1

    # Step 4: Setup environment variables
    if not setup_environment_variables():
        print_warning("Environment variable setup had issues, but continuing...")

    # Step 5: Validate dependencies
    if not validate_dependencies():
        print_warning("Dependency validation found issues, but continuing...")

    # Show activation instructions
    show_activation_instructions()

    print_header("Setup Complete!")
    print("\nYour Python environment is now ready for the Negative Space Imaging Project.")
    print("Happy coding! 🚀\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
