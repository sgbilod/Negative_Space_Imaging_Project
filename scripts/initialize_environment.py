#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Initialize Project Generator Environment
Author: Stephen Bilodeau
Date: August 13, 2025

This script sets up the environment for the project generator.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# Constants
HOME_DIR = os.path.expanduser("~")
CONFIG_FILE = os.path.join(HOME_DIR, ".project_manager_config.json")
PROJECTS_DIR = os.path.join(HOME_DIR, "Projects")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
BAT_DESTINATION = os.path.join(HOME_DIR, "create-project.bat")
PS1_DESTINATION = os.path.join(HOME_DIR, "Documents", "WindowsPowerShell", "Modules", "ProjectManager", "ProjectManager.psm1")

# Default configuration
DEFAULT_CONFIG = {
    "projects_dir": PROJECTS_DIR,
    "templates_dir": TEMPLATES_DIR,
    "author": "Stephen Bilodeau",
    "git_username": "StephenBilodeau",
    "git_email": "stephenbilodeau@example.com",
    "vs_code_enabled": True,
    "git_enabled": True,
    "auto_open_vs_code": True,
    "index_file": os.path.join(PROJECTS_DIR, "project_index.json")
}


def create_projects_dir():
    """Create the projects directory if it doesn't exist."""
    if not os.path.exists(PROJECTS_DIR):
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        print(f"Created projects directory: {PROJECTS_DIR}")
    else:
        print(f"Projects directory already exists: {PROJECTS_DIR}")


def create_config_file():
    """Create the configuration file if it doesn't exist."""
    if not os.path.exists(CONFIG_FILE):
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"Created configuration file: {CONFIG_FILE}")
    else:
        print(f"Configuration file already exists: {CONFIG_FILE}")


def install_batch_shortcut():
    """Install the batch file shortcut to the user's home directory."""
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create-project.bat")
    if os.path.exists(source):
        shutil.copy2(source, BAT_DESTINATION)
        print(f"Installed batch shortcut to: {BAT_DESTINATION}")
    else:
        print(f"Error: Batch file not found: {source}")


def install_powershell_module():
    """Install the PowerShell module to the user's PowerShell Modules directory."""
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Create-Project.ps1")
    if os.path.exists(source):
        module_dir = os.path.dirname(PS1_DESTINATION)
        os.makedirs(module_dir, exist_ok=True)

        # Copy content but change function name to use an approved verb
        with open(source, 'r') as f:
            content = f.read()

        # Replace function name with approved verb
        content = content.replace("Create-Project", "New-Project")
        content = content.replace("Create-Project", "New-Project")
        content = content.replace("args", "scriptArgs")

        # Create manifest file
        manifest = {
            "ModuleVersion": "1.0.0",
            "Author": "Stephen Bilodeau",
            "Description": "Project Manager Module",
            "PowerShellVersion": "5.0",
            "FunctionsToExport": ["New-Project"],
            "CmdletsToExport": [],
            "VariablesToExport": [],
            "AliasesToExport": []
        }

        manifest_content = "@{\n"
        for key, value in manifest.items():
            if isinstance(value, list):
                value_str = "@(" + ", ".join([f"'{v}'" for v in value]) + ")"
                manifest_content += f"    {key} = {value_str}\n"
            else:
                manifest_content += f"    {key} = '{value}'\n"
        manifest_content += "}"

        # Write files
        with open(PS1_DESTINATION, 'w') as f:
            f.write(content)

        with open(os.path.join(module_dir, "ProjectManager.psd1"), 'w') as f:
            f.write(manifest_content)

        print(f"Installed PowerShell module to: {module_dir}")
    else:
        print(f"Error: PowerShell script not found: {source}")


def check_dependencies():
    """Check for required dependencies."""
    try:
        subprocess.run(
            ["python", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Python is installed and available")
    except Exception:
        print("Warning: Python may not be in the PATH")

    try:
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Git is installed and available")
    except Exception:
        print("Warning: Git may not be installed or not in the PATH")

    try:
        subprocess.run(
            ["code", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("VS Code is installed and available")
    except Exception:
        print("Warning: VS Code may not be installed or not in the PATH")


def main():
    """Main function to initialize the environment."""
    print("Initializing project generator environment...")

    # Check dependencies
    check_dependencies()

    # Create projects directory
    create_projects_dir()

    # Create configuration file
    create_config_file()

    # Install shortcuts
    install_batch_shortcut()
    install_powershell_module()

    print("\nInitialization complete!")
    print("\nTo create a new project:")
    print("1. Run 'create-project.bat' from Explorer")
    print("2. Or run 'New-Project' from PowerShell")
    print("3. Or run 'python scripts/project_generator.py gui' from the command line")

    return 0


if __name__ == "__main__":
    sys.exit(main())
