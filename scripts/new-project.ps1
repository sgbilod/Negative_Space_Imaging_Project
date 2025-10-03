#requires -Version 5.0
<#
.SYNOPSIS
    Creates a new project from templates with standardized structure.
.DESCRIPTION
    This PowerShell script automates project creation for Stephen Bilodeau
    using configurable templates and consistent organization.
.PARAMETER ProjectName
    The name of the new project.
.PARAMETER Template
    The template to use (default, python, web, research, data).
.PARAMETER Path
    The path where the project should be created. Defaults to ~/Projects.
.PARAMETER Description
    A brief description of the project.
.PARAMETER OpenVSCode
    Whether to open the project in VS Code after creation.
.EXAMPLE
    .\new-project.ps1 -ProjectName "MyProject" -Template "python" -Description "A new Python project" -OpenVSCode
.NOTES
    Author: Stephen Bilodeau
    Created: August 2025
#>

param (
    [Parameter(Mandatory=$true)]
    [string]$ProjectName,

    [Parameter(Mandatory=$false)]
    [ValidateSet("default", "python", "web", "research", "data")]
    [string]$Template = "default",

    [Parameter(Mandatory=$false)]
    [string]$Path = "$env:USERPROFILE\Projects",

    [Parameter(Mandatory=$false)]
    [string]$Description = "Project created with automatic project generator",

    [Parameter(Mandatory=$false)]
    [switch]$OpenVSCode
)

# Configuration
$templatePath = "$PSScriptRoot\templates"
$gitUsername = "StephenBilodeau"
$gitEmail = "stephenbilodeau@example.com"

# Ensure templates path exists
if (-not (Test-Path $templatePath)) {
    Write-Host "Creating templates directory: $templatePath" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $templatePath -Force | Out-Null

    # Create basic template structure if it doesn't exist
    $defaultTemplatePath = Join-Path $templatePath "default"
    if (-not (Test-Path $defaultTemplatePath)) {
        Write-Host "Creating default template structure..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $defaultTemplatePath -Force | Out-Null
        New-Item -ItemType Directory -Path "$defaultTemplatePath\src" -Force | Out-Null
        New-Item -ItemType Directory -Path "$defaultTemplatePath\docs" -Force | Out-Null
        New-Item -ItemType Directory -Path "$defaultTemplatePath\tests" -Force | Out-Null
        New-Item -ItemType Directory -Path "$defaultTemplatePath\.github\workflows" -Force | Out-Null

        # Create README.md
        @"
# Project Title

$Description

## Description

A detailed description of the project.

## Getting Started

Instructions on setting up and running the project.

## Author

Stephen Bilodeau
"@ | Out-File -FilePath "$defaultTemplatePath\README.md" -Encoding utf8

        # Create .gitignore
        @"
# Common ignored files
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
"@ | Out-File -FilePath "$defaultTemplatePath\.gitignore" -Encoding utf8

        # Create LICENSE file
        @"
MIT License

Copyright (c) $(Get-Date -Format yyyy) Stephen Bilodeau

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
"@ | Out-File -FilePath "$defaultTemplatePath\LICENSE" -Encoding utf8
    }
}

# Check if the specific template exists
$selectedTemplatePath = Join-Path $templatePath $Template
if (-not (Test-Path $selectedTemplatePath)) {
    Write-Host "Template '$Template' not found. Using default template instead." -ForegroundColor Yellow
    $Template = "default"
    $selectedTemplatePath = Join-Path $templatePath $Template
}

# Ensure the destination path exists
if (-not (Test-Path $Path)) {
    Write-Host "Creating projects directory: $Path" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

# Create the full project path
$projectPath = Join-Path $Path $ProjectName
if (Test-Path $projectPath) {
    Write-Host "Error: Project '$ProjectName' already exists at $projectPath" -ForegroundColor Red
    exit 1
}

# Create the project directory
Write-Host "Creating project: $ProjectName using $Template template..." -ForegroundColor Green
New-Item -ItemType Directory -Path $projectPath -Force | Out-Null

# Copy template files to the new project
Copy-Item -Path "$selectedTemplatePath\*" -Destination $projectPath -Recurse

# Update README.md with the project name and description
$readmePath = Join-Path $projectPath "README.md"
if (Test-Path $readmePath) {
    $readmeContent = Get-Content -Path $readmePath -Raw
    $readmeContent = $readmeContent -replace "# Project Title", "# $ProjectName"
    $readmeContent = $readmeContent -replace "A detailed description of the project.", $Description
    $readmeContent | Out-File -FilePath $readmePath -Encoding utf8
}

# Create project-specific VS Code settings
$vscodePath = Join-Path $projectPath ".vscode"
if (-not (Test-Path $vscodePath)) {
    New-Item -ItemType Directory -Path $vscodePath -Force | Out-Null
}

# Create VS Code workspace file
$workspaceConfig = @{
    folders = @(
        @{
            path = "."
        }
    )
    settings = @{
        "editor.formatOnSave" = $true
        "files.autoSave" = "afterDelay"
        "workbench.colorTheme" = "Default Dark+"
    }
}

$workspaceJson = $workspaceConfig | ConvertTo-Json -Depth 10
$workspaceFile = Join-Path $projectPath "$ProjectName.code-workspace"
$workspaceJson | Out-File -FilePath $workspaceFile -Encoding utf8

# Initialize git repository
Write-Host "Initializing Git repository..." -ForegroundColor Green
Push-Location $projectPath
git init
git config user.name $gitUsername
git config user.email $gitEmail
git add .
git commit -m "Initial commit - Project structure created by automation script"
Pop-Location

# Create project metadata file
$metadataFile = Join-Path $projectPath ".project-info.json"
$metadata = @{
    name = $ProjectName
    description = $Description
    template = $Template
    created = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    author = "Stephen Bilodeau"
    path = $projectPath
}
$metadataJson = $metadata | ConvertTo-Json
$metadataJson | Out-File -FilePath $metadataFile -Encoding utf8

# Add project to the master index
$masterIndexPath = Join-Path $Path "project-index.md"
if (-not (Test-Path $masterIndexPath)) {
    @"
# Project Index

List of all projects created with the project generator script.

| Project | Description | Template | Created | Path |
|---------|-------------|----------|---------|------|
"@ | Out-File -FilePath $masterIndexPath -Encoding utf8
}

$indexLine = "| $ProjectName | $Description | $Template | $(Get-Date -Format 'yyyy-MM-dd') | $projectPath |"
Add-Content -Path $masterIndexPath -Value $indexLine

# Open in VS Code if requested
if ($OpenVSCode) {
    Write-Host "Opening project in VS Code..." -ForegroundColor Green

    # Try different possible paths to VS Code
    $vscodePaths = @(
        "code", # If in PATH
        "$env:LOCALAPPDATA\Programs\Microsoft VS Code\bin\code.cmd",
        "$env:PROGRAMFILES\Microsoft VS Code\bin\code.cmd",
        "${env:PROGRAMFILES(X86)}\Microsoft VS Code\bin\code.cmd"
    )

    $codeFound = $false
    foreach ($vscodePath in $vscodePaths) {
        try {
            if ($vscodePath -eq "code") {
                Start-Process code -ArgumentList $workspaceFile -ErrorAction Stop
            } else {
                if (Test-Path $vscodePath) {
                    Start-Process $vscodePath -ArgumentList $workspaceFile -ErrorAction Stop
                } else {
                    continue
                }
            }
            $codeFound = $true
            break
        } catch {
            continue
        }
    }

    if (-not $codeFound) {
        Write-Host "Could not find VS Code executable. Please open manually." -ForegroundColor Yellow
    }
}

Write-Host "Project '$ProjectName' created successfully at $projectPath" -ForegroundColor Green
Write-Host "To get started, navigate to the project directory: cd '$projectPath'" -ForegroundColor Cyan
