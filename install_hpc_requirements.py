#!/usr/bin/env python
"""
HPC Requirements Installer
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This script checks for and installs required HPC libraries including:
- MPI (Message Passing Interface)
- CUDA (GPU computing)
- Dask (Distributed computing)
- Ray (Distributed computing)

The script validates the installation and provides diagnostic information.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class InstallStatus(Enum):
    """Installation status."""
    INSTALLED = "installed"
    MISSING = "missing"
    OUTDATED = "outdated"
    ERROR = "error"


@dataclass
class Dependency:
    """HPC dependency information."""
    name: str
    package_name: str
    min_version: Optional[str] = None
    optional: bool = False
    description: str = ""
    check_command: Optional[str] = None
    install_command: Optional[str] = None


# Define HPC dependencies
HPC_DEPENDENCIES: List[Dependency] = [
    # Core distributed computing
    Dependency(
        name="Dask",
        package_name="dask[distributed]",
        min_version="2023.1.0",
        optional=False,
        description="Parallel computing with task scheduling",
    ),
    Dependency(
        name="Ray",
        package_name="ray",
        min_version="2.5.0",
        optional=False,
        description="Distributed computing framework",
    ),
    # MPI support
    Dependency(
        name="mpi4py",
        package_name="mpi4py",
        min_version="3.1.0",
        optional=True,
        description="MPI for Python - requires system MPI installation",
        check_command="mpirun --version",
    ),
    # GPU/CUDA support
    Dependency(
        name="CuPy",
        package_name="cupy-cuda12x",
        min_version="12.0.0",
        optional=True,
        description="GPU-accelerated NumPy-compatible library",
    ),
    Dependency(
        name="RAPIDS cuDF",
        package_name="cudf-cu12",
        min_version="23.04",
        optional=True,
        description="GPU DataFrame library",
    ),
    # Job scheduling
    Dependency(
        name="Prefect",
        package_name="prefect",
        min_version="2.10.0",
        optional=True,
        description="Workflow orchestration",
    ),
    # Performance monitoring
    Dependency(
        name="py-spy",
        package_name="py-spy",
        min_version="0.3.0",
        optional=True,
        description="Sampling profiler for Python",
    ),
    # Networking
    Dependency(
        name="aiohttp",
        package_name="aiohttp",
        min_version="3.8.0",
        optional=False,
        description="Async HTTP client/server",
    ),
    # Data serialization
    Dependency(
        name="PyArrow",
        package_name="pyarrow",
        min_version="12.0.0",
        optional=False,
        description="Columnar data format and IPC",
    ),
]


def get_python_version() -> Tuple[int, int, int]:
    """Get Python version as tuple."""
    return sys.version_info[:3]


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = get_python_version()
    if version < (3, 9):
        logger.error(f"Python 3.9+ required, found {'.'.join(map(str, version))}")
        return False
    logger.info(f"Python version: {'.'.join(map(str, version))}")
    return True


def check_package_installed(package_name: str) -> Tuple[InstallStatus, Optional[str]]:
    """
    Check if a Python package is installed.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        Tuple of (status, version)
    """
    try:
        # Handle packages with extras like "dask[distributed]"
        base_package = package_name.split("[")[0]
        
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", base_package],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            # Parse version from output
            for line in result.stdout.split("\n"):
                if line.startswith("Version:"):
                    version = line.split(":")[1].strip()
                    return InstallStatus.INSTALLED, version
            return InstallStatus.INSTALLED, None
        else:
            return InstallStatus.MISSING, None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout checking {package_name}")
        return InstallStatus.ERROR, None
    except Exception as e:
        logger.error(f"Error checking {package_name}: {e}")
        return InstallStatus.ERROR, None


def check_system_dependency(command: str) -> bool:
    """Check if a system command is available."""
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_cuda_available() -> Tuple[bool, Optional[str]]:
    """Check if CUDA is available."""
    # Check for nvidia-smi
    if shutil.which("nvidia-smi") is None:
        return False, None
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            driver_version = result.stdout.strip().split("\n")[0]
            return True, driver_version
    except (subprocess.TimeoutExpired, Exception):
        pass
    
    return False, None


def check_mpi_available() -> Tuple[bool, Optional[str]]:
    """Check if MPI is available."""
    mpi_commands = ["mpirun", "mpiexec", "srun"]
    
    for cmd in mpi_commands:
        if shutil.which(cmd):
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    version = result.stdout.split("\n")[0]
                    return True, version
            except (subprocess.TimeoutExpired, Exception):
                continue
    
    return False, None


def install_package(package_name: str, upgrade: bool = False) -> bool:
    """
    Install a Python package using pip.
    
    Args:
        package_name: Name of the package to install
        upgrade: Whether to upgrade existing installation
        
    Returns:
        True if installation was successful
    """
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package_name)
    
    try:
        logger.info(f"Installing {package_name}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully installed {package_name}")
            return True
        else:
            logger.error(f"Failed to install {package_name}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Installation of {package_name} timed out")
        return False
    except Exception as e:
        logger.error(f"Error installing {package_name}: {e}")
        return False


def check_all_dependencies() -> Dict[str, Dict]:
    """
    Check status of all HPC dependencies.
    
    Returns:
        Dictionary with dependency status information
    """
    results: Dict[str, Dict] = {}
    
    for dep in HPC_DEPENDENCIES:
        status, version = check_package_installed(dep.package_name)
        
        results[dep.name] = {
            "package": dep.package_name,
            "status": status.value,
            "version": version,
            "required_version": dep.min_version,
            "optional": dep.optional,
            "description": dep.description,
        }
        
        # Check version compatibility
        if status == InstallStatus.INSTALLED and version and dep.min_version:
            try:
                from packaging import version as pkg_version
                if pkg_version.parse(version) < pkg_version.parse(dep.min_version):
                    results[dep.name]["status"] = InstallStatus.OUTDATED.value
            except ImportError:
                pass
    
    return results


def install_missing_dependencies(
    skip_optional: bool = False,
    upgrade_outdated: bool = True,
) -> Dict[str, bool]:
    """
    Install missing HPC dependencies.
    
    Args:
        skip_optional: Skip optional dependencies
        upgrade_outdated: Upgrade outdated packages
        
    Returns:
        Dictionary with installation results
    """
    results: Dict[str, bool] = {}
    
    for dep in HPC_DEPENDENCIES:
        if skip_optional and dep.optional:
            logger.info(f"Skipping optional dependency: {dep.name}")
            continue
        
        status, version = check_package_installed(dep.package_name)
        
        if status == InstallStatus.MISSING:
            results[dep.name] = install_package(dep.package_name)
        elif status == InstallStatus.OUTDATED and upgrade_outdated:
            results[dep.name] = install_package(dep.package_name, upgrade=True)
        else:
            results[dep.name] = True
    
    return results


def validate_installation() -> bool:
    """
    Validate the HPC installation.
    
    Returns:
        True if validation passed
    """
    logger.info("Validating HPC installation...")
    
    validation_passed = True
    
    # Check Python version
    if not check_python_version():
        validation_passed = False
    
    # Check CUDA
    cuda_available, cuda_version = check_cuda_available()
    if cuda_available:
        logger.info(f"CUDA available, driver version: {cuda_version}")
    else:
        logger.warning("CUDA not available - GPU acceleration disabled")
    
    # Check MPI
    mpi_available, mpi_version = check_mpi_available()
    if mpi_available:
        logger.info(f"MPI available: {mpi_version}")
    else:
        logger.warning("MPI not available - distributed MPI disabled")
    
    # Check required packages
    required_missing = []
    for dep in HPC_DEPENDENCIES:
        if dep.optional:
            continue
        status, _ = check_package_installed(dep.package_name)
        if status != InstallStatus.INSTALLED:
            required_missing.append(dep.name)
    
    if required_missing:
        logger.error(f"Missing required packages: {', '.join(required_missing)}")
        validation_passed = False
    else:
        logger.info("All required packages installed")
    
    return validation_passed


def print_status_report() -> None:
    """Print a formatted status report."""
    print("\n" + "=" * 60)
    print("HPC Dependencies Status Report")
    print("=" * 60)
    
    # System info
    print(f"\nSystem: {platform.system()} {platform.release()}")
    print(f"Python: {'.'.join(map(str, get_python_version()))}")
    
    # CUDA status
    cuda_available, cuda_version = check_cuda_available()
    print(f"CUDA: {'Available (' + cuda_version + ')' if cuda_available else 'Not available'}")
    
    # MPI status
    mpi_available, mpi_version = check_mpi_available()
    print(f"MPI: {'Available' if mpi_available else 'Not available'}")
    
    # Package status
    print("\nPackage Status:")
    print("-" * 60)
    
    results = check_all_dependencies()
    for name, info in results.items():
        status_icon = {
            "installed": "✓",
            "missing": "✗",
            "outdated": "⚠",
            "error": "!",
        }.get(info["status"], "?")
        
        optional = " (optional)" if info["optional"] else ""
        version = f" v{info['version']}" if info["version"] else ""
        
        print(f"  {status_icon} {name}{version}{optional}")
    
    print("-" * 60)


def main() -> int:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HPC Requirements Installer")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check dependencies without installing",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install missing dependencies",
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional dependencies",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate installation",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade outdated packages",
    )
    
    args = parser.parse_args()
    
    # Default to check if no action specified
    if not any([args.check, args.install, args.validate]):
        args.check = True
    
    if args.check:
        print_status_report()
    
    if args.install:
        print("\nInstalling HPC dependencies...")
        results = install_missing_dependencies(
            skip_optional=args.skip_optional,
            upgrade_outdated=args.upgrade,
        )
        
        successful = sum(1 for v in results.values() if v)
        print(f"\nInstalled {successful}/{len(results)} packages successfully")
    
    if args.validate:
        if validate_installation():
            print("\n✓ HPC installation validated successfully")
            return 0
        else:
            print("\n✗ HPC installation validation failed")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
