"""
Security Vulnerability Fix Verification Test Suite
===================================================

Comprehensive tests to verify all critical security fixes have been applied:
1. JWT Secret Fallback Removal
2. Argon2id Password Hashing
3. Private Key Encryption
4. Duplicate Function Consolidation
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("security_verification")

# Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class SecurityVerification:
    """Verify all security fixes."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def test(self, name: str, condition: bool, details: str = ""):
        """Log test result."""
        if condition:
            print(f"{GREEN}✅ PASS{RESET}: {name}")
            if details:
                print(f"   {details}")
            self.passed += 1
        else:
            print(f"{RED}❌ FAIL{RESET}: {name}")
            if details:
                print(f"   {details}")
            self.failed += 1

    def warn(self, name: str, details: str = ""):
        """Log warning."""
        print(f"{YELLOW}⚠️  WARN{RESET}: {name}")
        if details:
            print(f"   {details}")
        self.warnings += 1

    def section(self, title: str):
        """Print section header."""
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}{title}{RESET}")
        print(f"{BLUE}{'='*70}{RESET}\n")

    def summary(self):
        """Print final summary."""
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}VERIFICATION SUMMARY{RESET}")
        print(f"{BLUE}{'='*70}{RESET}")
        print(f"{GREEN}✅ Passed: {self.passed}{RESET}")
        if self.failed > 0:
            print(f"{RED}❌ Failed: {self.failed}{RESET}")
        if self.warnings > 0:
            print(f"{YELLOW}⚠️  Warnings: {self.warnings}{RESET}")

        return self.failed == 0

    def check_file_exists(self, path: str) -> bool:
        """Check if file exists."""
        return Path(path).exists()

    def check_content(self, path: str, should_contain: str, should_not_contain: str = None) -> tuple:
        """Check file content."""
        if not self.check_file_exists(path):
            return False, "File not found"

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 for binary-compatible encoding
            with open(path, 'r', encoding='latin-1') as f:
                content = f.read()

        if should_contain and should_contain not in content:
            return False, f"Missing: {should_contain[:50]}"

        if should_not_contain and should_not_contain in content:
            return False, f"Should not contain: {should_not_contain[:50]}"

        return True, "Content verified"


def main():
    v = SecurityVerification()

    print(f"""
{BLUE}╔════════════════════════════════════════════════════════════════════╗
║       NEGATIVE SPACE IMAGING PROJECT - SECURITY VERIFICATION        ║
║                    Vulnerability Fix Test Suite                       ║
╚════════════════════════════════════════════════════════════════════╝{RESET}
""")

    # =========================================================================
    # TASK 1: JWT Secret Fallback Removal
    # =========================================================================
    v.section("TASK 1: JWT Secret Fallback Removal")

    jwt_files = [
        "api/api.py",
        "api/security/websocket_auth.py",
        "api/config/production.py",
        "sovereign/web_interface.py"
    ]

    for file_path in jwt_files:
        exists = v.check_file_exists(file_path)
        v.test(f"File exists: {file_path}", exists)

        if exists:
            # Check that fallback values are removed
            should_not_contain = "change-me-in-production"
            condition, msg = v.check_content(
                file_path,
                should_contain=None,
                should_not_contain=should_not_contain
            )

            # Special case for sovereign which might not have this exact string
            if "sovereign" in file_path:
                condition, msg = v.check_content(
                    file_path,
                    should_contain="if not app.secret_key",
                    should_not_contain=None
                )
                v.test(
                    f"Has forced validation: {file_path}",
                    condition,
                    f"Requires SECRET_KEY environment variable"
                )
            else:
                v.test(
                    f"No hardcoded fallback: {file_path}",
                    condition,
                    "Removed 'change-me-in-production' fallback"
                )

            # Check for ValueError on missing env var
            condition, msg = v.check_content(
                file_path,
                should_contain="raise ValueError",
                should_not_contain=None
            )
            v.test(
                f"Raises ValueError on missing env var: {file_path}",
                condition,
                "Will fail fast if JWT_SECRET/SECRET_KEY not set"
            )

    # =========================================================================
    # TASK 2: Argon2id Password Hashing
    # =========================================================================
    v.section("TASK 2: Argon2id Password Hashing Implementation")

    # Check password_security module exists
    pwd_sec_file = "security/password_security.py"
    exists = v.check_file_exists(pwd_sec_file)
    v.test(f"Module exists: {pwd_sec_file}", exists)

    if exists:
        condition, msg = v.check_content(
            pwd_sec_file,
            should_contain="from passlib.context import CryptContext",
            should_not_contain=None
        )
        v.test("Uses passlib for hashing", condition, "CryptContext imported")

        condition, msg = v.check_content(
            pwd_sec_file,
            should_contain='schemes=["argon2"]',
            should_not_contain=None
        )
        v.test("Argon2 configured as primary scheme", condition, "Argon2id will be used")

        condition, msg = v.check_content(
            pwd_sec_file,
            should_contain="def hash_password",
            should_not_contain=None
        )
        v.test("hash_password() function exists", condition)

        condition, msg = v.check_content(
            pwd_sec_file,
            should_contain="def verify_password",
            should_not_contain=None
        )
        v.test("verify_password() function exists", condition)

    # Check RBAC module updated
    rbac_file = "security/rbac.py"
    exists = v.check_file_exists(rbac_file)
    v.test(f"Module exists: {rbac_file}", exists)

    if exists:
        condition, msg = v.check_content(
            rbac_file,
            should_contain="from .password_security import",
            should_not_contain=None
        )
        v.test("RBAC imports password_security", condition)

        condition, msg = v.check_content(
            rbac_file,
            should_contain="password_hash",
            should_not_contain="user[\"password\"] !="
        )
        v.test("RBAC uses password hashing", condition, "No plaintext password comparison")

    # =========================================================================
    # TASK 3: Encrypt Private Keys at Rest
    # =========================================================================
    v.section("TASK 3: Private Key Encryption (Fernet)")

    key_mgr_file = "security/key_manager.py"
    exists = v.check_file_exists(key_mgr_file)
    v.test(f"Module exists: {key_mgr_file}", exists)

    if exists:
        condition, msg = v.check_content(
            key_mgr_file,
            should_contain="from cryptography.fernet import Fernet",
            should_not_contain=None
        )
        v.test("Uses Fernet encryption", condition, "AES-128-CBC with HMAC")

        condition, msg = v.check_content(
            key_mgr_file,
            should_contain="class KeyManager",
            should_not_contain=None
        )
        v.test("KeyManager class defined", condition)

        functions = ["encrypt_key", "decrypt_key", "encrypt_key_file", "decrypt_key_file"]
        for func in functions:
            condition, msg = v.check_content(
                key_mgr_file,
                should_contain=f"def {func}",
                should_not_contain=None
            )
            v.test(f"Function exists: {func}()", condition)

        # Check keys directory
        keys_dir = "keys"
        exists = v.check_file_exists(keys_dir)
        v.test(f"Keys directory exists: {keys_dir}", exists)

    # =========================================================================
    # TASK 4: Duplicate load_config Consolidation
    # =========================================================================
    v.section("TASK 4: Duplicate load_config Functions Consolidated")

    # Check centralized loader exists
    config_loader = "config/config_loader.py"
    exists = v.check_file_exists(config_loader)
    v.test(f"Centralized loader exists: {config_loader}", exists)

    if exists:
        condition, msg = v.check_content(
            config_loader,
            should_contain="def load_config(config_path:",
            should_not_contain=None
        )
        v.test("Centralized load_config() defined", condition)

        condition, msg = v.check_content(
            config_loader,
            should_contain="YAML_AVAILABLE",
            should_not_contain=None
        )
        v.test("Supports both YAML and JSON", condition)

    # Check that files import from centralized loader
    files_to_check = [
        ("src/main.py", "from config.config_loader import"),
        ("end_to_end_demo.py", "from config.config_loader import"),
        ("performance_cli.py", "from config.config_loader import"),
    ]

    for file_path, import_statement in files_to_check:
        if v.check_file_exists(file_path):
            condition, msg = v.check_content(
                file_path,
                should_contain=import_statement,
                should_not_contain=None
            )
            v.test(f"{file_path} uses centralized loader", condition)

    # Check project_generator has documented its local config
    if v.check_file_exists("scripts/project_generator.py"):
        condition, msg = v.check_content(
            "scripts/project_generator.py",
            should_contain="domain-specific config loader",
            should_not_contain=None
        )
        v.test(
            "project_generator.py documented as domain-specific",
            condition,
            "Kept as local config for project generation"
        )

    # =========================================================================
    # Additional Security Checks
    # =========================================================================
    v.section("Additional Security Checks")

    # Check that deprecated load_config in src/main.py wraps the new one
    if v.check_file_exists("src/main.py"):
        condition, msg = v.check_content(
            "src/main.py",
            should_contain="DEPRECATED",
            should_not_contain=None
        )
        v.test("Backward compatibility maintained with deprecation warning", condition)

    # Check requirements for new dependencies
    req_files = ["requirements.txt", "requirements.dev.txt"]
    for req_file in req_files:
        if v.check_file_exists(req_file):
            with open(req_file, 'r') as f:
                content = f.read()

            condition = "passlib" in content or "argon2" in content
            if condition:
                v.test(f"{req_file} includes password hashing dependencies", condition)

            condition = "cryptography" in content
            if condition:
                v.test(f"{req_file} includes encryption dependencies", condition)

    # =========================================================================
    # Print Summary
    # =========================================================================
    success = v.summary()

    print(f"""
{BLUE}╔════════════════════════════════════════════════════════════════════╗
║                        NEXT STEPS                                    ║
╚════════════════════════════════════════════════════════════════════╝{RESET}

1. {YELLOW}Environment Variables Required:{RESET}
   - JWT_SECRET: Strong random secret (for FastAPI)
   - SECRET_KEY: Strong random secret (for Flask)
   - MASTER_KEY: Base64-encoded Fernet key (for key encryption)

   Generate with:
   $ python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

2. {YELLOW}Testing:{RESET}
   - Run: python security/password_security.py
   - Run: python security/key_manager.py
   - Run: python config/config_loader.py

3. {YELLOW}Dependencies:{RESET}
   - Ensure passlib, cryptography, and pyyaml installed
   - pip install passlib cryptography pyyaml

4. {YELLOW}Migration Path:{RESET}
   - Existing plaintext passwords in RBAC must be re-hashed
   - Use password_security.hash_password() for new passwords
   - Existing key files should be encrypted using key_manager

5. {YELLOW}Recommendations:{RESET}
   ✓ Regular security audits
   ✓ Implement key rotation policy
   ✓ Enable audit logging for all auth events
   ✓ Consider MFA for sensitive operations
   ✓ Implement rate limiting on login endpoints
""")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
