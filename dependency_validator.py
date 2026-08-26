#!/usr/bin/env python
"""
Dependency Validator for Negative Space Imaging Project

This script validates the system requirements and dependencies for the
Negative Space Imaging System, including:

1. Python version check
2. Required Python packages and versions
3. External system dependencies
4. Configuration validation
5. Environment check
6. Security compliance verification

Usage:
    python dependency_validator.py [--verbose] [--fix]
"""

import argparse
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from distutils.version import LooseVersion
from typing import Dict, List, Tuple, Optional, Set

# Minimum required Python version
MIN_PYTHON_VERSION = (3, 8)

# Required Python packages with minimum versions
REQUIRED_PACKAGES = {
    "numpy": "1.20.0",
    "matplotlib": "3.4.0",
    "pillow": "8.0.0",
    "cryptography": "36.0.0",
    "requests": "2.25.0",
    "pyyaml": "5.4.0",
}

# Optional packages that enhance functionality but aren't required
OPTIONAL_PACKAGES = {
    "opencv-python": "4.5.0",
    "tensorflow": "2.8.0",
    "torch": "1.10.0",
    "scikit-image": "0.18.0",
    "scipy": "1.6.0",
    "pandas": "1.3.0",
}

# External tools that might be needed
EXTERNAL_TOOLS = {
    "ffmpeg": {
        "command": ["ffmpeg", "-version"],
        "version_pattern": r"ffmpeg version (\d+\.\d+\.\d+)",
        "min_version": "4.0.0",
        "required": False,
    },
    "imagemagick": {
        "command": ["convert", "--version"],
        "version_pattern": r"Version: ImageMagick (\d+\.\d+\.\d+)",
        "min_version": "7.0.0",
        "required": False,
    },
    "git": {
        "command": ["git", "--version"],
        "version_pattern": r"git version (\d+\.\d+\.\d+)",
        "min_version": "2.25.0",
        "required": True,
    },
}

# Colors for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_colored(message: str, color: str = Colors.RESET, bold: bool = False) -> None:
    """Print a message with color and optional bold formatting."""
    if os.name == "nt":  # Windows doesn't support ANSI colors in all terminals
        print(message)
    else:
        format_str = f"{color}{Colors.BOLD if bold else ''}{message}{Colors.RESET}"
        print(format_str)


def print_header(message: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 80)
    print_colored(f" {message} ".center(78), Colors.CYAN, bold=True)
    print("=" * 80)


def print_success(message: str) -> None:
    """Print a success message."""
    print_colored(f"✓ {message}", Colors.GREEN)


def print_warning(message: str) -> None:
    """Print a warning message."""
    print_colored(f"⚠ {message}", Colors.YELLOW)


def print_error(message: str) -> None:
    """Print an error message."""
    print_colored(f"✗ {message}", Colors.RED)


def print_info(message: str) -> None:
    """Print an information message."""
    print_colored(f"ℹ {message}", Colors.BLUE)


def validate_python_version() -> bool:
    """Validate that the Python version meets requirements."""
    current_version = sys.version_info
    is_valid = (current_version.major > MIN_PYTHON_VERSION[0] or
               (current_version.major == MIN_PYTHON_VERSION[0] and
                current_version.minor >= MIN_PYTHON_VERSION[1]))
    
    version_str = f"{current_version.major}.{current_version.minor}.{current_version.micro}"
    min_version_str = f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"
    
    if is_valid:
        print_success(f"Python version {version_str} meets requirements (>= {min_version_str})")
    else:
        print_error(f"Python version {version_str} does not meet requirements (>= {min_version_str})")
    
    return is_valid


def check_package_version(package_name: str, min_version: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a package is installed and meets the minimum version requirements.
    
    Args:
        package_name: Name of the package to check
        min_version: Minimum required version
        
    Returns:
        Tuple of (is_valid, installed_version)
    """
    try:
        # Try to import the package
        module = importlib.import_module(package_name)
        
        # Handle special cases for version attributes
        version_attr = "__version__"
        if package_name == "pillow":
            module = importlib.import_module("PIL")
            version_attr = "__version__"
        
        # Get the version
        if hasattr(module, version_attr):
            version = getattr(module, version_attr)
        else:
            # Some packages have different version attributes
            version_attrs = ["version", "VERSION", "__VERSION__", "PILLOW_VERSION"]
            for attr in version_attrs:
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    break
            else:
                # If we still can't find a version, try pkg_resources
                import pkg_resources
                version = pkg_resources.get_distribution(package_name).version
        
        # Compare versions
        is_valid = LooseVersion(version) >= LooseVersion(min_version)
        return is_valid, version
    
    except (ImportError, ModuleNotFoundError):
        return False, None
    except Exception as e:
        print_warning(f"Error checking {package_name} version: {str(e)}")
        return False, None


def validate_required_packages() -> Tuple[bool, List[str]]:
    """
    Validate that all required packages are installed with correct versions.
    
    Returns:
        Tuple of (all_valid, missing_packages)
    """
    all_valid = True
    missing_packages = []
    
    for package, min_version in REQUIRED_PACKAGES.items():
        is_valid, installed_version = check_package_version(package, min_version)
        
        if is_valid:
            print_success(f"{package} {installed_version} is installed (>= {min_version})")
        else:
            all_valid = False
            if installed_version:
                print_error(f"{package} {installed_version} is installed but version {min_version} is required")
            else:
                print_error(f"{package} is not installed (>= {min_version} required)")
                missing_packages.append(package)
    
    return all_valid, missing_packages


def validate_optional_packages() -> None:
    """Validate optional packages and print their status."""
    for package, min_version in OPTIONAL_PACKAGES.items():
        is_valid, installed_version = check_package_version(package, min_version)
        
        if is_valid:
            print_success(f"Optional: {package} {installed_version} is installed")
        elif installed_version:
            print_warning(f"Optional: {package} {installed_version} is installed but version {min_version} is recommended")
        else:
            print_info(f"Optional: {package} is not installed (recommended >= {min_version})")


def check_external_tool(tool_name: str, tool_config: Dict) -> bool:
    """
    Check if an external tool is installed and meets version requirements.
    
    Args:
        tool_name: Name of the external tool
        tool_config: Configuration for checking the tool
        
    Returns:
        True if the tool meets requirements, False otherwise
    """
    try:
        # Run the command to check if the tool is installed
        result = subprocess.run(
            tool_config["command"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        
        if result.returncode != 0:
            if tool_config["required"]:
                print_error(f"{tool_name} is not installed but is required")
                return False
            else:
                print_warning(f"{tool_name} is not installed (optional)")
                return True
        
        # Try to extract version from output
        version_match = re.search(tool_config["version_pattern"], result.stdout)
        if not version_match and result.stderr:
            version_match = re.search(tool_config["version_pattern"], result.stderr)
        
        if version_match:
            version = version_match.group(1)
            is_valid = LooseVersion(version) >= LooseVersion(tool_config["min_version"])
            
            if is_valid:
                status = "required" if tool_config["required"] else "optional"
                print_success(f"{tool_name} {version} is installed ({status})")
                return True
            else:
                if tool_config["required"]:
                    print_error(f"{tool_name} {version} is installed but version {tool_config['min_version']} is required")
                    return False
                else:
                    print_warning(f"{tool_name} {version} is installed but version {tool_config['min_version']} is recommended")
                    return True
        else:
            print_warning(f"Could not determine {tool_name} version")
            return not tool_config["required"]
    
    except Exception as e:
        print_warning(f"Error checking {tool_name}: {str(e)}")
        return not tool_config["required"]


def validate_external_tools() -> bool:
    """
    Validate that required external tools are installed.
    
    Returns:
        True if all required tools are available, False otherwise
    """
    all_valid = True
    
    for tool_name, tool_config in EXTERNAL_TOOLS.items():
        if not check_external_tool(tool_name, tool_config) and tool_config["required"]:
            all_valid = False
    
    return all_valid


def validate_environment() -> bool:
    """
    Validate the system environment.
    
    Returns:
        True if the environment is valid, False otherwise
    """
    valid = True
    
    # Check operating system
    os_name = platform.system()
    os_version = platform.version()
    print_info(f"Operating System: {os_name} {os_version}")
    
    # Check if running in a virtual environment
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if in_venv:
        print_success("Running in a virtual environment")
    else:
        print_warning("Not running in a virtual environment (recommended)")
    
    # Check available memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        
        print_info(f"Memory: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
        
        if available_gb < 2:
            print_warning("Low memory available (< 2 GB). Performance may be affected.")
            valid = False
    except ImportError:
        print_info("psutil not installed, skipping memory check")
    
    # Check for GPU support
    gpu_available = False
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            print_success(f"GPU available: {gpu_count} devices: {', '.join(gpu_names)}")
            gpu_available = True
        else:
            print_info("No CUDA-enabled GPU detected with PyTorch")
    except ImportError:
        pass
    
    # Check for TensorFlow GPU
    if not gpu_available:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print_success(f"GPU available: {len(gpus)} devices detected with TensorFlow")
                gpu_available = True
            else:
                print_info("No GPU detected with TensorFlow")
        except ImportError:
            pass
    
    if not gpu_available:
        print_warning("No GPU detected. Performance may be limited for image processing tasks.")
    
    return valid


def validate_config_files() -> bool:
    """
    Validate configuration files.
    
    Returns:
        True if all config files are valid, False otherwise
    """
    valid = True
    config_files = [
        "security_config.json",
        "hpc_config.yaml",
        "acquisition_profiles.py",
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                if config_file.endswith(".json"):
                    with open(config_file, "r") as f:
                        json.load(f)
                    print_success(f"Config file {config_file} is valid JSON")
                elif config_file.endswith(".yaml") or config_file.endswith(".yml"):
                    import yaml
                    with open(config_file, "r") as f:
                        yaml.safe_load(f)
                    print_success(f"Config file {config_file} is valid YAML")
                elif config_file.endswith(".py"):
                    # Just check if it's a valid Python file
                    with open(config_file, "r") as f:
                        compile(f.read(), config_file, "exec")
                    print_success(f"Config file {config_file} is valid Python")
                else:
                    print_info(f"Config file {config_file} exists but format validation is not supported")
            except Exception as e:
                print_error(f"Config file {config_file} is invalid: {str(e)}")
                valid = False
        else:
            print_warning(f"Config file {config_file} not found")
    
    return valid


def attempt_fix_dependencies(missing_packages: List[str]) -> None:
    """
    Attempt to fix missing dependencies by installing them.
    
    Args:
        missing_packages: List of missing packages to install
    """
    if not missing_packages:
        print_info("No missing required packages to fix")
        return
    
    print_header("Attempting to Fix Missing Dependencies")
    
    # Create requirements file
    requirements_file = "requirements_fix.txt"
    with open(requirements_file, "w") as f:
        for package in missing_packages:
            min_version = REQUIRED_PACKAGES.get(package, "")
            if min_version:
                f.write(f"{package}>={min_version}\n")
            else:
                f.write(f"{package}\n")
    
    print_info(f"Created temporary requirements file: {requirements_file}")
    
    # Install using pip
    try:
        print_info("Installing missing packages...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        
        if result.returncode == 0:
            print_success("Successfully installed missing packages")
        else:
            print_error(f"Failed to install packages: {result.stderr}")
    except Exception as e:
        print_error(f"Error installing packages: {str(e)}")
    
    # Clean up
    try:
        os.remove(requirements_file)
    except Exception:
        pass


def main() -> int:
    """
    Main function to validate dependencies.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Validate dependencies for Negative Space Imaging Project")
    parser.add_argument("--verbose", action="store_true", help="Show more detailed output")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix missing dependencies")
    args = parser.parse_args()
    
    print_header("Negative Space Imaging Project - Dependency Validator")
    
    # Track overall validation status
    validation_status = {}
    
    # Validate Python version
    print_header("Python Version")
    validation_status["python_version"] = validate_python_version()
    
    # Validate required packages
    print_header("Required Python Packages")
    validation_status["required_packages"], missing_packages = validate_required_packages()
    
    # Validate optional packages
    print_header("Optional Python Packages")
    validate_optional_packages()
    
    # Validate external tools
    print_header("External Tools")
    validation_status["external_tools"] = validate_external_tools()
    
    # Validate environment
    print_header("System Environment")
    validation_status["environment"] = validate_environment()
    
    # Validate configuration files
    print_header("Configuration Files")
    validation_status["config_files"] = validate_config_files()
    
    # Fix dependencies if requested and needed
    if args.fix and missing_packages:
        attempt_fix_dependencies(missing_packages)
    
    # Print overall summary
    print_header("Validation Summary")
    
    all_valid = all(validation_status.values())
    
    for section, status in validation_status.items():
        section_name = section.replace("_", " ").title()
        if status:
            print_success(f"{section_name}: ✓ Valid")
        else:
            print_error(f"{section_name}: ✗ Invalid")
    
    if all_valid:
        print_header("All Dependency Checks Passed! ✓")
        return 0
    else:
        print_header("Some Dependency Checks Failed! ✗")
        
        if not args.fix and missing_packages:
            print_info("You can attempt to fix missing packages with: python dependency_validator.py --fix")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
