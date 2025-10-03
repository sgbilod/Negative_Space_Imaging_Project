# Windows Environment Setup Script for Negative Space Imaging Project
# Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

$ErrorActionPreference = "Stop"

# Helper function to check if a command exists
function Test-Command($command) {
    try {
        Get-Command $command -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

# Helper function to download and install packages
function Install-Package($url, $outputFile, $installArgs) {
    Write-Host "Downloading $outputFile..."
    Invoke-WebRequest -Uri $url -OutFile $outputFile
    Write-Host "Installing $outputFile..."
    Start-Process -FilePath $outputFile -ArgumentList $installArgs -Wait
}

Write-Host "Starting Windows environment setup for Negative Space Imaging Project..."

# Check for administrative privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "This script requires administrative privileges. Please run as administrator."
    exit 1
}

# Install Visual Studio Build Tools if not present
if (-not (Test-Command cl)) {
    $buildToolsUrl = "https://aka.ms/vs/17/release/vs_buildtools.exe"
    $buildToolsInstaller = "vs_buildtools.exe"
    Install-Package $buildToolsUrl $buildToolsInstaller "--quiet --wait --norestart --nocache --installPath C:\BuildTools --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
}

# Install OpenSSL if not present
if (-not (Test-Command openssl)) {
    # Using a light version of OpenSSL for Windows
    $opensslUrl = "https://slproweb.com/download/Win64OpenSSL_Light-3_1_4.exe"
    $opensslInstaller = "Win64OpenSSL_Light.exe"
    Install-Package $opensslUrl $opensslInstaller "/silent /verysilent /sp- /suppressmsgboxes"

    # Add OpenSSL to system PATH
    $opensslPath = "C:\Program Files\OpenSSL-Win64\bin"
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    if (-not $currentPath.Contains($opensslPath)) {
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$opensslPath", "Machine")
    }
}

# Install Python if not present or update if needed
if (-not (Test-Command python)) {
    $pythonUrl = "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe"
    $pythonInstaller = "python-3.11.0-amd64.exe"
    Install-Package $pythonUrl $pythonInstaller "/quiet InstallAllUsers=1 PrependPath=1"
}

# Verify installations
$verificationResults = @()

# Check Visual C++
if (Test-Command cl) {
    $verificationResults += "✅ Visual C++ Build Tools installed"
}
else {
    $verificationResults += "❌ Visual C++ Build Tools installation failed"
}

# Check OpenSSL
if (Test-Command openssl) {
    $opensslVersion = (openssl version)
    $verificationResults += "✅ OpenSSL installed: $opensslVersion"
}
else {
    $verificationResults += "❌ OpenSSL installation failed"
}

# Check Python
if (Test-Command python) {
    $pythonVersion = (python --version)
    $verificationResults += "✅ Python installed: $pythonVersion"
}
else {
    $verificationResults += "❌ Python installation failed"
}

# Display results
Write-Host "`nVerification Results:"
$verificationResults | ForEach-Object { Write-Host $_ }

# Run Python environment validator
$validatorPath = Join-Path $PSScriptRoot "validate_environment.py"
if (Test-Path $validatorPath) {
    Write-Host "`nRunning Python environment validator..."
    python $validatorPath
}
else {
    Write-Host "`nWarning: Python environment validator script not found at $validatorPath"
}

Write-Host "`nSetup completed. Please check the verification results above for any issues."
