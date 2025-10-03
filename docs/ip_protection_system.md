# Documentation for ip_protection_system.py

```python
"""
Automated Source Code Documentation and IP Protection Script

This script performs the following tasks:
1. Scans all source code files
2. Generates documentation
3. Creates IP protection headers
4. Tracks file modifications
5. Maintains an audit trail
"""

import os
import datetime
import hashlib
from typing import Dict, List

class IPProtectionSystem:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.file_registry = {}
        self.audit_trail = []
        
    def scan_codebase(self) -> Dict[str, dict]:
        """Scan codebase and create inventory."""
        inventory = {}
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.md')):
                    path = os.path.join(root, file)
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        file_hash = hashlib.sha256(content.encode()).hexdigest()
                        inventory[path] = {
                            'hash': file_hash,
                            'last_modified': datetime.datetime.fromtimestamp(
                                os.path.getmtime(path)
                            ).isoformat(),
                            'size': os.path.getsize(path)
                        }
        return inventory

    def generate_ip_header(self, file_path: str) -> str:
        """Generate IP protection header for source files."""
        return f'''"""
Copyright (c) 2025 Negative Space Imaging Project
All rights reserved.

This source code is protected by international copyright laws and treaties.
Unauthorized reproduction or distribution of this code, or any portion of it,
may result in severe civil and criminal penalties, and will be prosecuted
to the maximum extent possible under law.

File: {os.path.basename(file_path)}
Last Modified: {datetime.datetime.now().isoformat()}
"""
'''

    def protect_source_file(self, file_path: str) -> None:
        """Add IP protection header to source file if not present."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if '"""Copyright (c)' not in content:
            header = self.generate_ip_header(file_path)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(header + content)
                
    def create_audit_record(self, action: str, file_path: str) -> None:
        """Create audit record for file modifications."""
        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'action': action,
            'file': file_path,
            'hash': hashlib.sha256(
                open(file_path, 'rb').read()
            ).hexdigest()
        }
        self.audit_trail.append(record)

if __name__ == "__main__":
    protection_system = IPProtectionSystem("path/to/codebase")
    # Add implementation of main execution logic

```