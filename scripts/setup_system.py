#!/usr/bin/env python
"""
Setup Script for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/setup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Setup")


def get_requirements() -> Dict[str, List[str]]:
    """Get all required packages."""
    return {
        "core": [
            "pyyaml",
            "psutil",
            "cryptography",
            "numpy",
            "scipy",
            "pandas",
            "requests"
        ],
        "security": [
            "cryptography",
            "pycryptodomex",
            "certifi"
        ],
        "imaging": [
            "pillow",
            "scikit-image",
            "opencv-python"
        ],
        "gpu": [
            "torch",
            "torchvision",
            "cuda-python"
        ],
        "monitoring": [
            "psutil",
            "py-cpuinfo",
            "gputil"
        ],
        "testing": [
            "pytest",
            "pytest-cov",
            "hypothesis"
        ]
    }


def install_packages(packages: List[str], upgrade: bool = False) -> bool:
    """Install Python packages."""
    try:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade" if upgrade else "--upgrade-strategy",
            "only-if-needed"
        ]
        cmd.extend(packages)

        logger.info(f"Installing packages: {', '.join(packages)}")
        subprocess.check_call(cmd)
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Package installation failed: {e}")
        return False


def setup_directories():
    """Create required directories."""
    try:
        dirs = [
            "logs",
            "config",
            "data",
            "data/cache",
            "data/temp",
            "reports",
            "models"
        ]

        project_root = Path(__file__).parent
        for dir_path in dirs:
            (project_root / dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

        return True

    except Exception as e:
        logger.error(f"Directory setup failed: {e}")
        return False


def validate_system():
    """Validate system setup."""
    try:
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False

        # Check for required commands
        required_commands = ["git", "python"]
        for cmd in required_commands:
            try:
                subprocess.check_call(
                    [cmd, "--version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error(f"Required command not found: {cmd}")
                return False

        return True

    except Exception as e:
        logger.error(f"System validation failed: {e}")
        return False


def setup_virtual_env():
    """Set up virtual environment."""
    try:
        venv_path = Path(".venv")

        # Create virtual environment if it doesn't exist
        if not venv_path.exists():
            logger.info("Creating virtual environment")
            subprocess.check_call([
                sys.executable,
                "-m",
                "venv",
                str(venv_path)
            ])

        # Get the path to the virtual environment Python
        if sys.platform == "win32":
            venv_python = venv_path / "Scripts" / "python.exe"
        else:
            venv_python = venv_path / "bin" / "python"

        if not venv_python.exists():
            logger.error("Virtual environment Python not found")
            return False

        # Upgrade pip in the virtual environment
        subprocess.check_call([
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip"
        ])

        logger.info("Virtual environment set up successfully")
        return True

    except Exception as e:
        logger.error(f"Virtual environment setup failed: {e}")
        return False


def main():
    """Main setup function."""
    try:
        logger.info("Starting system setup")

        # Validate system
        if not validate_system():
            logger.error("System validation failed")
            return 1

        # Set up virtual environment
        if not setup_virtual_env():
            logger.error("Virtual environment setup failed")
            return 1

        # Create directories
        if not setup_directories():
            logger.error("Directory setup failed")
            return 1

        # Install all requirements
        requirements = get_requirements()
        for category, packages in requirements.items():
            logger.info(f"Installing {category} requirements")
            if not install_packages(packages):
                logger.error(f"Failed to install {category} requirements")
                return 1

        logger.info("System setup completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
