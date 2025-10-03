#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project Creator - Python version
Author: Stephen Bilodeau
Date: August 2025

This script automates project creation with standardized templates
and consistent organization for Stephen Bilodeau's projects.
"""

import os
import sys
import json
import shutil
import argparse
import datetime
import subprocess
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create a new project with standardized structure")
    parser.add_argument(
        "project_name",
        help="Name of the new project"
    )
    parser.add_argument(
        "--template",
        choices=["default", "python", "web", "research", "data"],
        default="default",
        help="Template to use (default: default)"
    )
    parser.add_argument(
        "--path",
        default=os.path.expanduser("~/Projects"),
        help="Path where the project should be created (default: ~/Projects)"
    )
    parser.add_argument(
        "--description",
        default="Project created with automatic project generator",
        help="Brief description of the project"
    )
    parser.add_argument(
        "--open-vscode",
        action="store_true",
        help="Open the project in VS Code after creation"
    )
    return parser.parse_args()

def ensure_template_directory(templates_path):
    """Ensure the templates directory exists with a default template."""
    if not os.path.exists(templates_path):
        print(f"Creating templates directory: {templates_path}")
        os.makedirs(templates_path, exist_ok=True)

    default_template_path = os.path.join(templates_path, "default")
    if not os.path.exists(default_template_path):
        print("Creating default template structure...")
        os.makedirs(default_template_path, exist_ok=True)
        os.makedirs(os.path.join(default_template_path, "src"), exist_ok=True)
        os.makedirs(os.path.join(default_template_path, "docs"), exist_ok=True)
        os.makedirs(os.path.join(default_template_path, "tests"), exist_ok=True)
        os.makedirs(os.path.join(default_template_path, ".github", "workflows"), exist_ok=True)

        # Create README.md
        with open(os.path.join(default_template_path, "README.md"), "w", encoding="utf-8") as f:
            f.write("""# Project Title

A brief description of the project.

## Description

A detailed description of the project.

## Getting Started

Instructions on setting up and running the project.

## Author

Stephen Bilodeau
""")

        # Create .gitignore
        with open(os.path.join(default_template_path, ".gitignore"), "w", encoding="utf-8") as f:
            f.write("""# Common ignored files
__pycache__/
*.py[cod]
*$py.class
*.so
.env
.venv
env/
venv/
ENV/
.vs/
.vscode/
node_modules/
dist/
build/
*.log
""")

        # Create LICENSE file
        current_year = datetime.datetime.now().year
        with open(os.path.join(default_template_path, "LICENSE"), "w", encoding="utf-8") as f:
            f.write(f"""MIT License

Copyright (c) {current_year} Stephen Bilodeau

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")

def create_project(args):
    """Create a new project based on the specified template."""
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(script_dir, "templates")
    git_username = "StephenBilodeau"
    git_email = "stephenbilodeau@example.com"

    # Ensure template directory exists
    ensure_template_directory(templates_path)

    # Check if the specific template exists
    selected_template_path = os.path.join(templates_path, args.template)
    if not os.path.exists(selected_template_path):
        print(f"Template '{args.template}' not found. Using default template instead.")
        args.template = "default"
        selected_template_path = os.path.join(templates_path, args.template)

    # Ensure the destination path exists
    if not os.path.exists(args.path):
        print(f"Creating projects directory: {args.path}")
        os.makedirs(args.path, exist_ok=True)

    # Create the full project path
    project_path = os.path.join(args.path, args.project_name)
    if os.path.exists(project_path):
        print(f"Error: Project '{args.project_name}' already exists at {project_path}")
        sys.exit(1)

    # Create the project directory
    print(f"Creating project: {args.project_name} using {args.template} template...")
    os.makedirs(project_path, exist_ok=True)

    # Copy template files to the new project
    for item in os.listdir(selected_template_path):
        s = os.path.join(selected_template_path, item)
        d = os.path.join(project_path, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    # Update README.md with the project name and description
    readme_path = os.path.join(project_path, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        readme_content = readme_content.replace("# Project Title", f"# {args.project_name}")
        readme_content = readme_content.replace("A detailed description of the project.", args.description)

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

    # Create project-specific VS Code settings
    vscode_path = os.path.join(project_path, ".vscode")
    if not os.path.exists(vscode_path):
        os.makedirs(vscode_path, exist_ok=True)

    # Create VS Code workspace file
    workspace_config = {
        "folders": [
            {
                "path": "."
            }
        ],
        "settings": {
            "editor.formatOnSave": True,
            "files.autoSave": "afterDelay",
            "workbench.colorTheme": "Default Dark+"
        }
    }

    workspace_file = os.path.join(project_path, f"{args.project_name}.code-workspace")
    with open(workspace_file, "w", encoding="utf-8") as f:
        json.dump(workspace_config, f, indent=4)

    # Initialize git repository
    print("Initializing Git repository...")
    current_dir = os.getcwd()
    os.chdir(project_path)

    try:
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "config", "user.name", git_username], check=True)
        subprocess.run(["git", "config", "user.email", git_email], check=True)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit - Project structure created by automation script"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
    finally:
        os.chdir(current_dir)

    # Create project metadata file
    metadata_file = os.path.join(project_path, ".project-info.json")
    metadata = {
        "name": args.project_name,
        "description": args.description,
        "template": args.template,
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "author": "Stephen Bilodeau",
        "path": project_path
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Add project to the master index
    master_index_path = os.path.join(args.path, "project-index.md")
    if not os.path.exists(master_index_path):
        with open(master_index_path, "w", encoding="utf-8") as f:
            f.write("""# Project Index

List of all projects created with the project generator script.

| Project | Description | Template | Created | Path |
|---------|-------------|----------|---------|------|
""")

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    index_line = f"| {args.project_name} | {args.description} | {args.template} | {today} | {project_path} |"

    with open(master_index_path, "a", encoding="utf-8") as f:
        f.write(index_line + "\n")

    # Open in VS Code if requested
    if args.open_vscode:
        print("Opening project in VS Code...")
        try:
            # Try different possible paths to VS Code
            vscode_paths = [
                "code",  # If in PATH
                os.path.join(os.environ.get('LOCALAPPDATA', ''),
                            'Programs', 'Microsoft VS Code', 'bin', 'code.cmd'),
                os.path.join(os.environ.get('PROGRAMFILES', ''),
                            'Microsoft VS Code', 'bin', 'code.cmd'),
                os.path.join(os.environ.get('PROGRAMFILES(X86)', ''),
                            'Microsoft VS Code', 'bin', 'code.cmd')
            ]

            for vscode_path in vscode_paths:
                try:
                    if sys.platform == "win32" and vscode_path != "code":
                        subprocess.run([vscode_path, workspace_file], check=True, shell=True)
                    else:
                        subprocess.run([vscode_path, workspace_file], check=True)
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            else:
                print("Could not find VS Code executable. Please open manually.")
        except Exception as e:
            print(f"Failed to open VS Code: {e}")

    print(f"Project '{args.project_name}' created successfully at {project_path}")
    print(f"To get started, navigate to the project directory: cd '{project_path}'")

def main():
    """Main entry point."""
    args = parse_arguments()
    create_project(args)

if __name__ == "__main__":
    main()
