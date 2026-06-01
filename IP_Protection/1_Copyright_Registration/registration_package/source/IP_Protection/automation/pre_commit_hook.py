"""
Copyright (c) 2025 Negative Space Imaging Project
All rights reserved.

This source code is protected by international copyright laws and treaties.
Unauthorized reproduction or distribution of this code, or any portion of it,
may result in severe civil and criminal penalties, and will be prosecuted
to the maximum extent possible under law.

File: pre_commit_hook.py
Last Modified: 2025-08-06T02:06:31.711701
"""
"""
Pre-commit hook for IP protection.
This script ensures all files are scanned and protected before committing.
"""
import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ip_protection import IPProtectionSystem

def run_pre_commit():
    root_dir = os.getcwd()
    protection_system = IPProtectionSystem(root_dir)
    
    print("Scanning codebase...")
    inventory = protection_system.scan_codebase()
    
    protected_count = 0
    for file_path in inventory:
        if protection_system.protect_source_file(file_path):
            protected_count += 1
            protection_system.create_audit_record("protected", file_path)
    
    print(f"Scanned {len(inventory)} files")
    print(f"Protected {protected_count} files")
    
    return protected_count

if __name__ == "__main__":
    run_pre_commit()
