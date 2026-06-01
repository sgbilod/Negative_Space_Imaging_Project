# Documentation for license_checker.py

```python
"""
Copyright (c) 2025 Negative Space Imaging Project
All rights reserved.

This source code is protected by international copyright laws and treaties.
Unauthorized reproduction or distribution of this code, or any portion of it,
may result in severe civil and criminal penalties, and will be prosecuted
to the maximum extent possible under law.

File: license_checker.py
Last Modified: 2025-08-06T02:06:31.708972
"""
"""
License compliance checker.
This script checks for license compliance in project dependencies.
"""
import os
import subprocess

def check_licenses():
    try:
        result = subprocess.run(["pip", "licenses"], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Error checking licenses: {e}")

if __name__ == "__main__":
    check_licenses()

```