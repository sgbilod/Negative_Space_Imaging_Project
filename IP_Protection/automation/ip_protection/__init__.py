"""
Base module for IP protection system
"""

class IPProtectionSystem:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.audit_trail = []

    def scan_codebase(self):
        """Scan and inventory all source files."""
        import os
        import hashlib
        from datetime import datetime
        
        inventory = {}
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.md')):
                    try:
                        path = os.path.join(root, file)
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            file_hash = hashlib.sha256(content.encode()).hexdigest()
                            inventory[path] = {
                                'hash': file_hash,
                                'last_modified': datetime.fromtimestamp(
                                    os.path.getmtime(path)
                                ).isoformat(),
                                'size': os.path.getsize(path)
                            }
                    except UnicodeDecodeError:
                        # If UTF-8 fails, try with a more permissive encoding
                        with open(path, 'r', encoding='latin-1') as f:
                            content = f.read()
                            file_hash = hashlib.sha256(content.encode()).hexdigest()
                            inventory[path] = {
                                'hash': file_hash,
                                'last_modified': datetime.fromtimestamp(
                                    os.path.getmtime(path)
                                ).isoformat(),
                                'size': os.path.getsize(path)
                            }
        return inventory

    def protect_source_file(self, file_path: str) -> bool:
        """Add IP protection header to source file if not present."""
        import os
        from datetime import datetime

        header = f'''"""
Copyright (c) 2025 Negative Space Imaging Project
All rights reserved.

This source code is protected by international copyright laws and treaties.
Unauthorized reproduction or distribution of this code, or any portion of it,
may result in severe civil and criminal penalties, and will be prosecuted
to the maximum extent possible under law.

File: {os.path.basename(file_path)}
Last Modified: {datetime.now().isoformat()}
"""
'''
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()

        if '"""Copyright (c)' not in content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(header + content)
            return True
        return False

    def create_audit_record(self, action: str, file_path: str) -> None:
        """Create audit record for file modifications."""
        import hashlib
        from datetime import datetime
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                file_hash = hashlib.sha256(content).hexdigest()
        except Exception as e:
            file_hash = f"Error: {str(e)}"

        record = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'file': file_path,
            'hash': file_hash
        }
        self.audit_trail.append(record)
