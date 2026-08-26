# Project Organization and Automation Guide

## Overview

This guide outlines the standardized process for creating, organizing, and managing projects using the automation tools provided. Following these practices ensures consistency, improves productivity, and makes project management more efficient.

## Project Creation Options

### PowerShell Script (Windows)

```powershell
.\scripts\new-project.ps1 -ProjectName "MyNewProject" -Template "python" -Description "Description of the project" -OpenVSCode
```

Parameters:
- `ProjectName`: Name of the new project (required)
- `Template`: Template to use (default, python, web, research, data)
- `Path`: Location to create the project (defaults to ~/Projects)
- `Description`: Brief description of the project
- `OpenVSCode`: Switch to open in VS Code when done

### Python Script (Cross-platform)

```bash
python scripts/new-project.py "MyNewProject" --template python --description "Description of the project" --open-vscode
```

Parameters:
- `project_name`: Name of the new project (required)
- `--template`: Template to use (default, python, web, research, data)
- `--path`: Location to create the project (defaults to ~/Projects)
- `--description`: Brief description of the project
- `--open-vscode`: Flag to open in VS Code when done

## Project Templates

The scripts create projects using templates stored in `~/Templates/ProjectTemplates/`. Available templates:

1. **default**: Basic structure suitable for any project
   - README.md, LICENSE, .gitignore
   - src/, tests/, docs/ directories
   - GitHub workflow configuration

2. **python**: Python project template
   - Python-specific .gitignore
   - Requirements files
   - Sample package structure

3. **web**: Web development template
   - HTML/CSS/JS starter files
   - Node.js configuration

4. **research**: Research project template
   - Data directories
   - Notebook setup
   - Reference management

5. **data**: Data analysis project template
   - Data processing pipeline structure
   - Visualization directory
   - Analysis notebooks

## Project Organization

### Standard Project Structure

```
ProjectName/
├── .vscode/               # VS Code settings
├── .github/               # GitHub configurations
│   └── workflows/         # CI/CD workflows
├── src/                   # Source code
├── tests/                 # Test files
├── docs/                  # Documentation
├── README.md              # Project overview
├── LICENSE                # License information
├── .gitignore             # Git ignore file
├── .project-info.json     # Project metadata
└── ProjectName.code-workspace  # VS Code workspace file
```

### Project Tracking

Projects are tracked in the master index file at `~/Projects/project-index.md`. This file includes:

- Project name
- Description
- Template used
- Creation date
- Project path

## Best Practices

1. **Consistent Naming**: Use clear, descriptive names for projects
   - Format: `purpose-projectname-date` (e.g., `ai-imageprocessor-2025`)

2. **Documentation**: Always maintain an up-to-date README.md with:
   - Project purpose
   - Setup instructions
   - Usage examples
   - Author information

3. **Version Control**: All projects are automatically initialized with Git
   - Create a remote repository on GitHub after local creation
   - Push the initial commit to the remote repository

4. **Project Metadata**: The `.project-info.json` file contains important metadata
   - Do not delete this file as it's used by management tools

5. **VS Code Integration**: Use the generated workspace file for best experience
   - Customized settings for each project type
   - Extension recommendations included

## Customizing Templates

To create or modify templates:

1. Navigate to `~/Templates/ProjectTemplates/`
2. Create a new directory for your template or modify existing ones
3. Add the files and directory structure you want in new projects
4. The template will be available for use in the creation scripts

## Adding to Existing Projects

For existing projects not created with these tools:

1. Create a `.project-info.json` file with project metadata
2. Add an entry to the master index file
3. Standardize the directory structure to match templates

## Support

For help with these tools or to suggest improvements, contact Stephen Bilodeau.

---

*This guide was created as part of the project automation system for Stephen Bilodeau's workflows.*
