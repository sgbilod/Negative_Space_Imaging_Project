#!/usr/bin/env python
"""Environment verification script for Negative Space Imaging Project."""

import sys
import os
import pkg_resources
import importlib
from pathlib import Path

def verify_python_version():
    """Verify Python version meets requirements."""
    required_version = (3, 13)
    current_version = sys.version_info[:2]

    print(f"Checking Python version...")
    print(f"Current: {'.'.join(map(str, current_version))}")
    print(f"Required: {'.'.join(map(str, required_version))}")

    if current_version < required_version:
        raise RuntimeError(f"Python {'.'.join(map(str, required_version))} or higher required")

def verify_dependencies():
    """Verify all required dependencies are installed."""
    print("\nChecking dependencies...")

    required_packages = [
        'numpy',
        'matplotlib',
        'psutil',
        'PIL',
        'contourpy'
    ]

    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package}: {version}")
        except ImportError as e:
            print(f"✗ {package}: {str(e)}")
            raise

def verify_pythonpath():
    """Verify PYTHONPATH includes project root."""
    print("\nChecking PYTHONPATH...")

    project_root = str(Path(__file__).parent.parent.absolute())
    python_path = os.environ.get("PYTHONPATH", "")

    if project_root not in python_path:
        raise RuntimeError(
            f"Project root not in PYTHONPATH\n"
            f"Current: {python_path}\n"
            f"Required: {project_root}"
        )
    print(f"✓ Project root found in PYTHONPATH")

def verify_imports():
    """Verify critical project imports work."""
    print("\nVerifying critical imports...")

    modules = [
        "sovereign.pipeline.implementation",
        "sovereign.quantum_engine",
        "sovereign.quantum_state"
    ]

    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {str(e)}")
            raise

def main():
    """Run all verifications."""
    try:
        verify_python_version()
        verify_dependencies()
        verify_pythonpath()
        verify_imports()
        print("\n✓ Environment verification completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Environment verification failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
