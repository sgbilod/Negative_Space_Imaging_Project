"""
Copyright (c) 2025 Negative Space Imaging Project
All rights reserved.

This source code is protected by international copyright laws and treaties.
Unauthorized reproduction or distribution of this code, or any portion of it,
may result in severe civil and criminal penalties, and will be prosecuted
to the maximum extent possible under law.

File: run_protection.py
Last Modified: 2025-08-06T02:06:31.712826
"""
"""
Main IP Protection Automation Runner
"""
import os
from pathlib import Path
from ip_protection_hooks import IPProtectionHooks
from documentation_generator import IPDocumentationGenerator
from license_checker import LicenseComplianceChecker
from copyright_registrator import CopyrightRegistrationAutomator

def main():
    # Set up paths
    project_root = Path("c:/Users/sgbil/OneDrive/Desktop/Negative_Space_Imaging_Project")
    
    print("Starting IP Protection Automation...")

    # Initialize components
    hooks = IPProtectionHooks(str(project_root))
    doc_gen = IPDocumentationGenerator(str(project_root))
    license_checker = LicenseComplianceChecker(str(project_root))
    copyright_reg = CopyrightRegistrationAutomator(str(project_root))

    # Run automated processes
    print("\n1. Running Git Hooks Setup...")
    # hooks.setup_git_hooks()  # Would set up actual git hooks

    print("\n2. Generating Documentation...")
    doc_gen.generate_documentation()

    print("\n3. Checking License Compliance...")
    license_checker.scan_dependencies()
    license_checker.generate_notice_file()

    print("\n4. Preparing Copyright Registration...")
    copyright_reg.prepare_registration()

    print("\n5. Starting Continuous Monitoring...")
    hooks.monitor_changes()

if __name__ == "__main__":
    main()
