#!/usr/bin/env python
"""
Environment Validation Script for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This script validates and sets up the development environment by:
1. Checking system prerequisites (Visual C++, OpenSSL)
2. Validating Python version and architecture
3. Setting up virtual environment with proper configurations
4. Installing dependencies in the correct order
5. Verifying critical package installations
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import pkg_resources
import json

class EnvironmentValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.requirements = {
            'system': {
                'windows': ['Visual C++ 14.0+', 'OpenSSL 1.1.0+'],
                'linux': ['OpenSSL 1.1.0+', 'gcc', 'python3-dev'],
                'darwin': ['OpenSSL 1.1.0+', 'gcc']
            },
            'python': '>=3.8',
            'critical_packages': [
                'cryptography>=45.0.0',
                'numpy>=1.24.0',
                'pillow>=9.5.0',
                'opencv-python>=4.7.0'
            ]
        }

    def check_system_prerequisites(self):
        """Verify system-level dependencies."""
        system = platform.system().lower()
        missing = []

        if system == 'windows':
            # Check Visual C++
            try:
                subprocess.run(['cl'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError:
                missing.append('Visual C++ 14.0+')

            # Check OpenSSL
            try:
                subprocess.run(['openssl', 'version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError:
                missing.append('OpenSSL 1.1.0+')

        return missing

    def setup_virtual_environment(self):
        """Create and configure virtual environment."""
        venv_path = self.project_root / '.venv'
        if not venv_path.exists():
            subprocess.run([sys.executable, '-m', 'venv', str(venv_path)])

        # Set environment variables for cryptography
        os.environ['CRYPTOGRAPHY_DONT_BUILD_RUST'] = '1'

    def install_dependencies(self):
        """Install project dependencies in the correct order."""
        pip_cmd = [sys.executable, '-m', 'pip']

        # First, upgrade pip and setuptools
        subprocess.run([*pip_cmd, 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])

        # Install critical packages first
        for package in self.requirements['critical_packages']:
            try:
                subprocess.run([*pip_cmd, 'install', '--upgrade', package])
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")
                return False

        # Install remaining requirements
        requirements_file = self.project_root / 'requirements.txt'
        if requirements_file.exists():
            subprocess.run([*pip_cmd, 'install', '-r', str(requirements_file)])

        return True

    def verify_installation(self):
        """Verify that critical packages are properly installed."""
        verification_results = {}

        for package_req in self.requirements['critical_packages']:
            package_name = package_req.split('>=')[0]
            try:
                pkg = pkg_resources.working_set.by_key[package_name]
                verification_results[package_name] = {
                    'installed': True,
                    'version': pkg.version
                }
            except KeyError:
                verification_results[package_name] = {
                    'installed': False,
                    'version': None
                }

        return verification_results

    def run(self):
        """Run the complete validation process."""
        print("Starting environment validation...")

        # Check system prerequisites
        missing_prereqs = self.check_system_prerequisites()
        if missing_prereqs:
            print("Missing system prerequisites:", missing_prereqs)
            print("Please install these prerequisites before continuing.")
            return False

        # Setup virtual environment
        print("Setting up virtual environment...")
        self.setup_virtual_environment()

        # Install dependencies
        print("Installing dependencies...")
        if not self.install_dependencies():
            print("Failed to install all dependencies.")
            return False

        # Verify installation
        print("Verifying installations...")
        results = self.verify_installation()

        # Save verification results
        results_file = self.project_root / 'environment_verification.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        all_installed = all(r['installed'] for r in results.values())
        if all_installed:
            print("Environment setup completed successfully.")
        else:
            print("Some packages failed to install correctly. Check environment_verification.json for details.")

        return all_installed

if __name__ == '__main__':
    validator = EnvironmentValidator()
    success = validator.run()
    sys.exit(0 if success else 1)
