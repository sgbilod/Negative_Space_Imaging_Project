# Project Organization and Automation System

A comprehensive system for automating project creation, organization, and management, designed to maximize ease of use and consistency across all platforms.

## Features

- **Standardized Project Creation**: Create new projects with consistent structure and organization
- **Multiple Platform Support**: Run from PowerShell, Python, or batch file interfaces
- **Template System**: Choose from multiple project templates or create your own
- **Project Dashboard**: Visual overview of all your projects
- **Integration with VS Code**: Seamless editor integration with workspace files
- **Git Initialization**: Automatic git setup for all new projects
- **Project Tracking**: Master index of all projects with metadata

## Getting Started

### Prerequisites

- Windows with PowerShell 5.0+ (for PowerShell script)
- Python 3.6+ (for Python script)
- Git installed and accessible from command line
- Visual Studio Code installed (optional but recommended)

### Installation

1. Clone or copy these scripts to your system
2. Ensure the scripts directory is accessible
3. For PowerShell script, you may need to allow script execution:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### Quick Start

The easiest way to get started is to use the batch file interface:

1. Navigate to the `scripts` directory
2. Run `project_manager.bat`
3. Follow the interactive prompts

Alternatively, use the PowerShell or Python scripts directly as described in the [Project Organization Guide](PROJECT_ORGANIZATION_GUIDE.md).

## Usage Examples

### Create a New Python Project

PowerShell:
```powershell
.\scripts\new-project.ps1 -ProjectName "ImageProcessor" -Template "python" -Description "Image processing library" -OpenVSCode
```

Python:
```bash
python scripts/new-project.py "ImageProcessor" --template python --description "Image processing library" --open-vscode
```

### Generate a Project Dashboard

```bash
python scripts/project_dashboard.py
```

This will create an HTML dashboard showing all your projects and open it in your default browser.

## Project Structure

The system creates projects with the following structure:

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

## Templates

The system includes several templates:

- **default**: Basic structure for any project
- **python**: Python-specific project structure
- **web**: Web development project structure
- **research**: Research project structure
- **data**: Data analysis project structure

Custom templates can be added to `~/Templates/ProjectTemplates/`.

## Documentation

See the [Project Organization Guide](PROJECT_ORGANIZATION_GUIDE.md) for detailed documentation on:

- Project creation process
- Template customization
- Best practices for organization
- Project tracking system

## License

MIT License - Copyright (c) 2025 Stephen Bilodeau

## Author

Stephen Bilodeau
