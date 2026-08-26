"""
Git Hooks for IP Protection

This script implements Git hooks to ensure IP protection measures are maintained:
1. Pre-commit hook to verify file headers
2. Pre-push hook to check for sensitive information
3. Commit-msg hook to enforce documentation standards
"""

import os
import sys
import re
import hashlib
from datetime import datetime
from typing import List, Dict

class IPProtectionHooks:
    def __init__(self):
        self.sensitive_patterns = [
            r'password\s*=',
            r'api_key\s*=',
            r'secret\s*=',
            r'private_key\s*='
        ]
        
    def check_file_header(self, file_path: str) -> bool:
        """Verify that source files have proper copyright headers."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(500)  # Read header portion
            return '"""Copyright (c)' in content
            
    def scan_for_sensitive_info(self, file_path: str) -> List[str]:
        """Scan for sensitive information in files."""
        violations = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for pattern in self.sensitive_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(f"Sensitive pattern found: {pattern}")
        return violations
        
    def verify_commit_message(self, message: str) -> bool:
        """Verify commit message follows documentation standards."""
        required_patterns = [
            r'\[FEATURE\]|\[FIX\]|\[DOC\]|\[REFACTOR\]',
            r'JIRA-\d+',
            r'.{10,}'  # Minimum length
        ]
        return all(re.search(pattern, message) for pattern in required_patterns)

    def run_pre_commit_hook(self) -> bool:
        """Pre-commit hook implementation."""
        staged_files = self.get_staged_files()
        for file_path in staged_files:
            if not self.check_file_header(file_path):
                print(f"Error: Missing copyright header in {file_path}")
                return False
            violations = self.scan_for_sensitive_info(file_path)
            if violations:
                print(f"Error: Found sensitive information in {file_path}")
                for violation in violations:
                    print(f"  - {violation}")
                return False
        return True

    def get_staged_files(self) -> List[str]:
        """Get list of staged files."""
        # Implementation would use git commands to get staged files
        return []

if __name__ == "__main__":
    hooks = IPProtectionHooks()
    if len(sys.argv) > 1:
        hook_type = sys.argv[1]
        if hook_type == "pre-commit":
            sys.exit(0 if hooks.run_pre_commit_hook() else 1)
        # Add other hook implementations as needed
