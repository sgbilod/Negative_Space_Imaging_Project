# Development Environment Setup Guide
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This document explains the development environment setup process for the Negative Space Imaging Project.

## Prerequisites

### Windows
- Windows 10 or later
- Administrator privileges for installing system components
- PowerShell 5.1 or later

### Linux/macOS
- sudo privileges for installing system components
- bash shell

## Automatic Setup

The project includes automated setup scripts that will install and configure all necessary components:

1. **Windows Setup**:
   ```powershell
   # Open PowerShell as Administrator
   cd scripts
   .\setup_windows_env.ps1
   ```

2. **Linux/macOS Setup**:
   ```bash
   # Open terminal
   cd scripts
   ./setup_unix_env.sh
   ```

## Manual Setup

If you prefer to set up components manually or the automatic setup fails:

1. Install system prerequisites:
   - Visual C++ Build Tools (Windows)
   - OpenSSL 1.1.0 or later
   - Python 3.13.6 (canonical version)

2. Create and activate virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   (For Linux/macOS, use `source .venv/bin/activate`)

3. Upgrade pip and install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Capture environment snapshot (for reproducibility):
   ```powershell
   python -m pip freeze > environment_snapshot.txt
   ```

## Verification

To verify your setup:

```powershell
python environment_verification.py
```

This will check:
- System prerequisites
- Python environment
- Critical package installations
- Security component configurations

## Troubleshooting

If you encounter issues:

1. Check environment_verification.json for detailed status
2. Review environment_snapshot.txt for installed packages
2. Ensure system prerequisites are properly installed
3. Try reinstalling problematic packages individually
4. Check the logs in logs/ directory

## Security Notes
## Demo Script Execution

To run the demo:
```powershell
python demo.py
```

The setup process includes installation of cryptographic components that are essential for:
- Secure image processing
- Multi-signature verification
- HIPAA compliance features

Ensure all security components are properly installed and verified before using the system.
