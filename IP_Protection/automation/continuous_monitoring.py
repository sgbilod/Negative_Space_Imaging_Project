"""
Copyright (c) 2025 Negative Space Imaging Project
All rights reserved.

This source code is protected by international copyright laws and treaties.
Unauthorized reproduction or distribution of this code, or any portion of it,
may result in severe civil and criminal penalties, and will be prosecuted
to the maximum extent possible under law.

File: continuous_monitoring.py
Last Modified: 2025-08-06T02:06:31.703366
"""
"""
Continuous monitoring system for file changes.
This script logs all file modifications in the project directory.
"""
import os
import time
import hashlib
from datetime import datetime

def monitor_changes(root_dir):
    file_hashes = {}

    while True:
        for root, _, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(('.py', '.md', '.json')):
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        file_hash = hashlib.sha256(content).hexdigest()

                    if file_path not in file_hashes:
                        file_hashes[file_path] = file_hash
                        print(f"New file detected: {file_path}")
                    elif file_hashes[file_path] != file_hash:
                        file_hashes[file_path] = file_hash
                        print(f"File modified: {file_path} at {datetime.now().isoformat()}")

        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    monitor_changes("c:\\Users\\sgbil\\OneDrive\\Desktop\\Negative_Space_Imaging_Project")
