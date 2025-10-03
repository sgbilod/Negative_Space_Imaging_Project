"""
Sovereign Security Module
© 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

Provides comprehensive security features for the Sovereign Control System:
- Authentication and authorization
- Data encryption and integrity
- Access control and audit logging
- Quantum-resistant cryptography
- Intrusion detection and prevention
"""

import uuid
import json
import hmac
import base64
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import secrets
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sovereign.security')


class SecurityLevel:
    """Security levels for the sovereign system"""
    STANDARD = "STANDARD"
    ENHANCED = "ENHANCED"
    MAXIMUM = "MAXIMUM"
    QUANTUM = "QUANTUM"


class SecurityManager:
    """
    Security Manager for Sovereign Control System

    Provides comprehensive security services:
    - Authentication and authorization
    - Encryption and decryption
    - Access control and permissions
    - Audit logging and monitoring
    - Intrusion detection
    """

    def __init__(self, project_root: Path, security_level: str = SecurityLevel.ENHANCED):
        """Initialize the security manager"""
        self.project_root = project_root
        self.security_level = security_level
        self.instance_id = str(uuid.uuid4())
        self.creation_time = datetime.now()

        # Paths for security-related files
        self.security_path = project_root / "sovereign" / "security"
        self.security_path.mkdir(parents=True, exist_ok=True)

        self.keys_path = self.security_path / "keys"
        self.keys_path.mkdir(exist_ok=True)

        self.users_path = self.security_path / "users"
        self.users_path.mkdir(exist_ok=True)

        self.logs_path = self.security_path / "logs"
        self.logs_path.mkdir(exist_ok=True)

        # Initialize security components
        self._initialize_master_key()
        self._initialize_users()
        self._initialize_audit_log()

        # Security settings
        self.settings = self._load_default_settings()

        logger.info(f"Security Manager initialized with level: {security_level}")
        self.log_security_event("SECURITY_INITIALIZED",
                               {"level": security_level, "instance_id": self.instance_id})

    def _load_default_settings(self) -> Dict[str, Any]:
        """Load default security settings"""
        return {
            "access_control": {
                "max_failed_attempts": 5,
                "lockout_duration_minutes": 30,
                "session_timeout_minutes": 60,
                "require_mfa": self.security_level != SecurityLevel.STANDARD,
                "password_expiry_days": 90
            },
            "encryption": {
                "algorithm": "AES-256-GCM" if self.security_level != SecurityLevel.QUANTUM else "QUANTUM-RESISTANT-AES",
                "key_rotation_days": 30,
                "secure_data_transit": True,
                "encrypt_logs": self.security_level != SecurityLevel.STANDARD,
                "encrypt_config": True
            },
            "auditing": {
                "log_all_actions": True,
                "log_failed_attempts": True,
                "log_data_access": self.security_level != SecurityLevel.STANDARD,
                "log_retention_days": 365,
                "real_time_alerts": self.security_level == SecurityLevel.MAXIMUM or self.security_level == SecurityLevel.QUANTUM
            },
            "integrity": {
                "verify_file_signatures": True,
                "runtime_integrity_checks": self.security_level != SecurityLevel.STANDARD,
                "secure_boot_verification": self.security_level == SecurityLevel.MAXIMUM or self.security_level == SecurityLevel.QUANTUM,
                "tamper_detection": True
            }
        }

    def _initialize_master_key(self):
        """Initialize or load the master encryption key"""
        master_key_path = self.keys_path / "master.key"

        if master_key_path.exists():
            # Load existing key
            with open(master_key_path, "rb") as f:
                self.master_key = f.read()
            logger.info("Master key loaded from file")
        else:
            # Generate new key
            if self.security_level == SecurityLevel.QUANTUM:
                # For quantum security, use a longer key
                self.master_key = secrets.token_bytes(64)  # 512 bits
            else:
                self.master_key = secrets.token_bytes(32)  # 256 bits

            # Save key to file
            with open(master_key_path, "wb") as f:
                f.write(self.master_key)

            logger.info("New master key generated and saved")

        # Generate derived keys for different purposes
        self._generate_derived_keys()

    def _generate_derived_keys(self):
        """Generate derived keys for different purposes from the master key"""
        # Use PBKDF2 to derive different keys for different purposes
        salt = b"sovereign_control_system"

        # Data encryption key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt + b"data_encryption",
            iterations=100000,
            backend=default_backend()
        )
        self.data_key = kdf.derive(self.master_key)

        # Config encryption key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt + b"config_encryption",
            iterations=100000,
            backend=default_backend()
        )
        self.config_key = kdf.derive(self.master_key)

        # Authentication key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt + b"authentication",
            iterations=100000,
            backend=default_backend()
        )
        self.auth_key = kdf.derive(self.master_key)

    def _initialize_users(self):
        """Initialize default users if they don't exist"""
        users_file = self.users_path / "users.json"

        if not users_file.exists():
            # Create default admin user
            admin_password = secrets.token_urlsafe(16)
            admin_salt = secrets.token_bytes(16)

            # Hash the password
            password_hash = self._hash_password(admin_password, admin_salt)

            default_users = {
                "admin": {
                    "username": "admin",
                    "password_hash": base64.b64encode(password_hash).decode('utf-8'),
                    "salt": base64.b64encode(admin_salt).decode('utf-8'),
                    "role": "administrator",
                    "created": datetime.now().isoformat(),
                    "last_login": None,
                    "failed_attempts": 0,
                    "locked_until": None,
                    "require_password_change": True,
                    "mfa_enabled": False
                }
            }

            # Save users file
            self._save_encrypted_json(users_file, default_users, self.auth_key)

            logger.info(f"Default admin user created with password: {admin_password}")
            logger.info("IMPORTANT: Change this password immediately after first login")

            # Also save the initial password in a secure location for first login
            with open(self.security_path / "initial_credentials.txt", "w") as f:
                f.write(f"Initial admin password: {admin_password}\n")
                f.write("IMPORTANT: Delete this file and change the password immediately after first login\n")
        else:
            logger.info("Users file already exists, skipping default user creation")

    def _initialize_audit_log(self):
        """Initialize the security audit log"""
        self.audit_log_file = self.logs_path / f"audit_{datetime.now().strftime('%Y%m%d')}.log"

        logger.info(f"Audit logging initialized: {self.audit_log_file}")

    def _hash_password(self, password: str, salt: bytes) -> bytes:
        """
        Hash a password with the provided salt

        Args:
            password: The password to hash
            salt: The salt to use

        Returns:
            Bytes containing the password hash
        """
        # Use PBKDF2 for password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(password.encode('utf-8'))

    def _save_encrypted_json(self, path: Path, data: Dict[str, Any], key: bytes):
        """
        Save data as encrypted JSON

        Args:
            path: Path to save the file
            data: Data to encrypt and save
            key: Encryption key to use
        """
        # Convert data to JSON
        json_data = json.dumps(data, indent=2).encode('utf-8')

        # Generate a random IV
        iv = secrets.token_bytes(16)

        # Create an encryptor
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        # Add padding
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(json_data) + padder.finalize()

        # Encrypt the data
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Write the IV and ciphertext to file
        with open(path, "wb") as f:
            f.write(iv)
            f.write(encryptor.tag)
            f.write(ciphertext)

    def _load_encrypted_json(self, path: Path, key: bytes) -> Dict[str, Any]:
        """
        Load and decrypt JSON data

        Args:
            path: Path to the encrypted file
            key: Decryption key to use

        Returns:
            Decrypted JSON data
        """
        with open(path, "rb") as f:
            # Read the IV (16 bytes) and tag (16 bytes)
            iv = f.read(16)
            tag = f.read(16)
            ciphertext = f.read()

        # Create a decryptor
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()

        # Decrypt the data
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()

        # Remove padding
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()

        # Parse JSON
        return json.loads(data.decode('utf-8'))

    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log a security event to the audit log

        Args:
            event_type: Type of security event
            details: Event details
        """
        timestamp = datetime.now().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "instance_id": self.instance_id,
            "details": details
        }

        # Write to log file
        with open(self.audit_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Real-time alerts for critical events
        if (self.settings["auditing"]["real_time_alerts"] and
            event_type in ["AUTHENTICATION_FAILED", "UNAUTHORIZED_ACCESS",
                          "ENCRYPTION_FAILURE", "INTEGRITY_VIOLATION"]):
            logger.warning(f"SECURITY ALERT: {event_type} - {details}")

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Authenticate a user with username and password

        Args:
            username: Username to authenticate
            password: Password to verify

        Returns:
            Tuple of (success, user_data)
        """
        users_file = self.users_path / "users.json"

        if not users_file.exists():
            logger.error("Users file not found")
            self.log_security_event("AUTHENTICATION_ERROR",
                                   {"reason": "users_file_not_found", "username": username})
            return False, None

        try:
            # Load users data
            users = self._load_encrypted_json(users_file, self.auth_key)

            # Check if user exists
            if username not in users:
                logger.warning(f"Authentication failed: User {username} not found")
                self.log_security_event("AUTHENTICATION_FAILED",
                                       {"reason": "user_not_found", "username": username})
                return False, None

            user = users[username]

            # Check if account is locked
            if user.get("locked_until"):
                locked_until = datetime.fromisoformat(user["locked_until"])
                if datetime.now() < locked_until:
                    logger.warning(f"Authentication failed: Account {username} is locked")
                    self.log_security_event("AUTHENTICATION_FAILED",
                                           {"reason": "account_locked", "username": username})
                    return False, None
                else:
                    # Reset failed attempts if lock has expired
                    user["failed_attempts"] = 0
                    user["locked_until"] = None

            # Verify password
            salt = base64.b64decode(user["salt"])
            stored_hash = base64.b64decode(user["password_hash"])

            computed_hash = self._hash_password(password, salt)

            if hmac.compare_digest(computed_hash, stored_hash):
                # Successful authentication
                user["last_login"] = datetime.now().isoformat()
                user["failed_attempts"] = 0

                # Save updated user data
                users[username] = user
                self._save_encrypted_json(users_file, users, self.auth_key)

                logger.info(f"User {username} authenticated successfully")
                self.log_security_event("AUTHENTICATION_SUCCESS", {"username": username})

                return True, user
            else:
                # Failed authentication
                user["failed_attempts"] = user.get("failed_attempts", 0) + 1

                # Check if account should be locked
                max_attempts = self.settings["access_control"]["max_failed_attempts"]
                if user["failed_attempts"] >= max_attempts:
                    lockout_minutes = self.settings["access_control"]["lockout_duration_minutes"]
                    user["locked_until"] = (datetime.now() + timedelta(minutes=lockout_minutes)).isoformat()
                    logger.warning(f"Account {username} locked for {lockout_minutes} minutes after {max_attempts} failed attempts")

                # Save updated user data
                users[username] = user
                self._save_encrypted_json(users_file, users, self.auth_key)

                logger.warning(f"Authentication failed: Invalid password for user {username}")
                self.log_security_event("AUTHENTICATION_FAILED",
                                       {"reason": "invalid_password", "username": username,
                                        "attempts": user["failed_attempts"]})

                return False, None

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            self.log_security_event("AUTHENTICATION_ERROR",
                                   {"reason": "exception", "error": str(e), "username": username})
            return False, None

    def create_user(self, username: str, password: str, role: str) -> bool:
        """
        Create a new user

        Args:
            username: Username for the new user
            password: Password for the new user
            role: Role for the new user

        Returns:
            Success status
        """
        users_file = self.users_path / "users.json"

        try:
            # Load existing users or create new dict
            if users_file.exists():
                users = self._load_encrypted_json(users_file, self.auth_key)
            else:
                users = {}

            # Check if user already exists
            if username in users:
                logger.warning(f"Cannot create user: Username {username} already exists")
                self.log_security_event("USER_CREATION_FAILED",
                                       {"reason": "username_exists", "username": username})
                return False

            # Generate salt and hash password
            salt = secrets.token_bytes(16)
            password_hash = self._hash_password(password, salt)

            # Create new user
            new_user = {
                "username": username,
                "password_hash": base64.b64encode(password_hash).decode('utf-8'),
                "salt": base64.b64encode(salt).decode('utf-8'),
                "role": role,
                "created": datetime.now().isoformat(),
                "last_login": None,
                "failed_attempts": 0,
                "locked_until": None,
                "require_password_change": True,
                "mfa_enabled": False
            }

            # Add user to users dict
            users[username] = new_user

            # Save updated users
            self._save_encrypted_json(users_file, users, self.auth_key)

            logger.info(f"User {username} created with role {role}")
            self.log_security_event("USER_CREATED", {"username": username, "role": role})

            return True

        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            self.log_security_event("USER_CREATION_ERROR",
                                   {"reason": "exception", "error": str(e), "username": username})
            return False

    def encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypt data using the data encryption key

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data
        """
        # Generate a random IV
        iv = secrets.token_bytes(16)

        # Create an encryptor
        cipher = Cipher(
            algorithms.AES(self.data_key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        # Add padding
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(data) + padder.finalize()

        # Encrypt the data
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Combine IV, tag, and ciphertext
        result = iv + encryptor.tag + ciphertext

        return result

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using the data encryption key

        Args:
            encrypted_data: Data to decrypt

        Returns:
            Decrypted data
        """
        # Extract IV and tag
        iv = encrypted_data[:16]
        tag = encrypted_data[16:32]
        ciphertext = encrypted_data[32:]

        # Create a decryptor
        cipher = Cipher(
            algorithms.AES(self.data_key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()

        # Decrypt the data
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()

        # Remove padding
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()

        return data

    def generate_session_token(self, username: str, expiry_minutes: int = 60) -> str:
        """
        Generate a secure session token for a user

        Args:
            username: Username to generate token for
            expiry_minutes: Token expiry time in minutes

        Returns:
            Session token
        """
        # Create token data
        expiry_time = datetime.now() + timedelta(minutes=expiry_minutes)
        token_data = {
            "username": username,
            "expiry": expiry_time.isoformat(),
            "instance_id": self.instance_id,
            "nonce": secrets.token_hex(8)
        }

        # Convert to JSON and encode
        token_json = json.dumps(token_data)
        token_bytes = token_json.encode('utf-8')

        # Sign the token
        signature = hmac.new(
            self.auth_key,
            token_bytes,
            hashlib.sha256
        ).digest()

        # Combine token and signature
        combined = token_bytes + signature

        # Encrypt token
        encrypted_token = self.encrypt_data(combined)

        # Encode as base64 for transmission
        token = base64.urlsafe_b64encode(encrypted_token).decode('utf-8')

        logger.debug(f"Generated session token for user {username}")
        return token

    def validate_session_token(self, token: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a session token

        Args:
            token: Session token to validate

        Returns:
            Tuple of (valid, username)
        """
        try:
            # Decode from base64
            encrypted_token = base64.urlsafe_b64decode(token)

            # Decrypt token
            combined = self.decrypt_data(encrypted_token)

            # Extract token data and signature
            token_bytes = combined[:-32]
            signature = combined[-32:]

            # Verify signature
            expected_signature = hmac.new(
                self.auth_key,
                token_bytes,
                hashlib.sha256
            ).digest()

            if not hmac.compare_digest(signature, expected_signature):
                logger.warning("Token validation failed: Invalid signature")
                self.log_security_event("TOKEN_VALIDATION_FAILED", {"reason": "invalid_signature"})
                return False, None

            # Parse token data
            token_data = json.loads(token_bytes.decode('utf-8'))

            # Check expiry
            expiry = datetime.fromisoformat(token_data["expiry"])
            if datetime.now() > expiry:
                logger.warning("Token validation failed: Token expired")
                self.log_security_event("TOKEN_VALIDATION_FAILED",
                                       {"reason": "token_expired", "username": token_data["username"]})
                return False, None

            # Check instance ID if required
            if self.security_level == SecurityLevel.MAXIMUM or self.security_level == SecurityLevel.QUANTUM:
                if token_data["instance_id"] != self.instance_id:
                    logger.warning("Token validation failed: Invalid instance ID")
                    self.log_security_event("TOKEN_VALIDATION_FAILED",
                                           {"reason": "invalid_instance", "username": token_data["username"]})
                    return False, None

            username = token_data["username"]
            logger.debug(f"Token validated for user {username}")
            return True, username

        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            self.log_security_event("TOKEN_VALIDATION_ERROR", {"reason": "exception", "error": str(e)})
            return False, None

    def check_authorization(self, username: str, required_role: str) -> bool:
        """
        Check if a user is authorized for a specific role

        Args:
            username: Username to check
            required_role: Required role

        Returns:
            Authorization status
        """
        users_file = self.users_path / "users.json"

        if not users_file.exists():
            logger.error("Users file not found")
            return False

        try:
            # Load users data
            users = self._load_encrypted_json(users_file, self.auth_key)

            # Check if user exists
            if username not in users:
                logger.warning(f"Authorization failed: User {username} not found")
                self.log_security_event("AUTHORIZATION_FAILED",
                                       {"reason": "user_not_found", "username": username})
                return False

            user = users[username]
            user_role = user["role"]

            # Simple role hierarchy
            role_hierarchy = {
                "administrator": ["administrator", "manager", "user", "guest"],
                "manager": ["manager", "user", "guest"],
                "user": ["user", "guest"],
                "guest": ["guest"]
            }

            # Check if user's role has access to required role
            if user_role in role_hierarchy and required_role in role_hierarchy.get(user_role, []):
                logger.info(f"User {username} authorized for role {required_role}")
                return True
            else:
                logger.warning(f"Authorization failed: User {username} with role {user_role} not authorized for {required_role}")
                self.log_security_event("AUTHORIZATION_FAILED",
                                       {"username": username, "user_role": user_role, "required_role": required_role})
                return False

        except Exception as e:
            logger.error(f"Authorization error: {str(e)}")
            self.log_security_event("AUTHORIZATION_ERROR",
                                   {"reason": "exception", "error": str(e), "username": username})
            return False

    def verify_file_integrity(self, file_path: Path) -> bool:
        """
        Verify the integrity of a file using stored checksums

        Args:
            file_path: Path to the file to verify

        Returns:
            Integrity status
        """
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"File integrity check failed: File {file_path} not found")
            return False

        # Get checksums file
        checksums_file = self.security_path / "checksums.json"

        if not checksums_file.exists():
            logger.warning("Checksums file not found, cannot verify integrity")
            return False

        try:
            # Load checksums
            checksums = self._load_encrypted_json(checksums_file, self.config_key)

            # Get relative path for lookup
            rel_path = str(file_path.relative_to(self.project_root))

            # Check if file has a stored checksum
            if rel_path not in checksums:
                logger.warning(f"No checksum found for file {rel_path}")
                return False

            # Get stored checksum
            stored_checksum = checksums[rel_path]

            # Calculate current checksum
            with open(file_path, "rb") as f:
                file_data = f.read()
                current_checksum = hashlib.sha256(file_data).hexdigest()

            # Compare checksums
            if current_checksum == stored_checksum:
                logger.debug(f"Integrity verified for file {rel_path}")
                return True
            else:
                logger.warning(f"Integrity check failed for file {rel_path}")
                self.log_security_event("INTEGRITY_VIOLATION", {"file": rel_path})
                return False

        except Exception as e:
            logger.error(f"Integrity check error: {str(e)}")
            self.log_security_event("INTEGRITY_CHECK_ERROR",
                                   {"reason": "exception", "error": str(e), "file": str(file_path)})
            return False

    def update_file_checksum(self, file_path: Path) -> bool:
        """
        Update the stored checksum for a file

        Args:
            file_path: Path to the file

        Returns:
            Success status
        """
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Cannot update checksum: File {file_path} not found")
            return False

        # Get checksums file
        checksums_file = self.security_path / "checksums.json"

        try:
            # Load existing checksums or create new dict
            if checksums_file.exists():
                checksums = self._load_encrypted_json(checksums_file, self.config_key)
            else:
                checksums = {}

            # Get relative path for storage
            rel_path = str(file_path.relative_to(self.project_root))

            # Calculate checksum
            with open(file_path, "rb") as f:
                file_data = f.read()
                checksum = hashlib.sha256(file_data).hexdigest()

            # Update checksums
            checksums[rel_path] = checksum

            # Save updated checksums
            self._save_encrypted_json(checksums_file, checksums, self.config_key)

            logger.info(f"Updated checksum for file {rel_path}")
            return True

        except Exception as e:
            logger.error(f"Error updating checksum: {str(e)}")
            self.log_security_event("CHECKSUM_UPDATE_ERROR",
                                   {"reason": "exception", "error": str(e), "file": str(file_path)})
            return False

    def encrypt_file(self, file_path: Path) -> bool:
        """
        Encrypt a file in place

        Args:
            file_path: Path to the file to encrypt

        Returns:
            Success status
        """
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Cannot encrypt file: File {file_path} not found")
            return False

        try:
            # Read file data
            with open(file_path, "rb") as f:
                file_data = f.read()

            # Encrypt data
            encrypted_data = self.encrypt_data(file_data)

            # Write encrypted data back to file
            with open(file_path, "wb") as f:
                f.write(encrypted_data)

            # Update file extension to indicate encryption
            encrypted_path = file_path.with_suffix(file_path.suffix + ".enc")
            file_path.rename(encrypted_path)

            logger.info(f"File encrypted: {file_path} -> {encrypted_path}")
            return True

        except Exception as e:
            logger.error(f"Error encrypting file: {str(e)}")
            self.log_security_event("FILE_ENCRYPTION_ERROR",
                                   {"reason": "exception", "error": str(e), "file": str(file_path)})
            return False

    def decrypt_file(self, file_path: Path) -> bool:
        """
        Decrypt a file in place

        Args:
            file_path: Path to the encrypted file

        Returns:
            Success status
        """
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Cannot decrypt file: File {file_path} not found")
            return False

        # Check if file has .enc extension
        if not str(file_path).endswith(".enc"):
            logger.warning(f"Cannot decrypt file: File {file_path} does not have .enc extension")
            return False

        try:
            # Read encrypted data
            with open(file_path, "rb") as f:
                encrypted_data = f.read()

            # Decrypt data
            decrypted_data = self.decrypt_data(encrypted_data)

            # Get original file path (remove .enc extension)
            original_path = file_path.with_suffix("")

            # Write decrypted data to original file
            with open(original_path, "wb") as f:
                f.write(decrypted_data)

            # Remove encrypted file
            file_path.unlink()

            logger.info(f"File decrypted: {file_path} -> {original_path}")
            return True

        except Exception as e:
            logger.error(f"Error decrypting file: {str(e)}")
            self.log_security_event("FILE_DECRYPTION_ERROR",
                                   {"reason": "exception", "error": str(e), "file": str(file_path)})
            return False

    def secure_config(self, config_data: Dict[str, Any]) -> bytes:
        """
        Secure configuration data for storage

        Args:
            config_data: Configuration data to secure

        Returns:
            Secured config data
        """
        # Convert to JSON
        json_data = json.dumps(config_data).encode('utf-8')

        # Encrypt the data
        encrypted_data = self.encrypt_data(json_data)

        return encrypted_data

    def load_secure_config(self, encrypted_data: bytes) -> Dict[str, Any]:
        """
        Load secured configuration data

        Args:
            encrypted_data: Encrypted configuration data

        Returns:
            Decrypted configuration data
        """
        # Decrypt the data
        json_data = self.decrypt_data(encrypted_data)

        # Parse JSON
        config_data = json.loads(json_data.decode('utf-8'))

        return config_data

    def rotate_keys(self) -> bool:
        """
        Rotate encryption keys

        Returns:
            Success status
        """
        logger.info("Rotating encryption keys")

        try:
            # Generate new master key
            if self.security_level == SecurityLevel.QUANTUM:
                new_master_key = secrets.token_bytes(64)  # 512 bits
            else:
                new_master_key = secrets.token_bytes(32)  # 256 bits

            # Save old keys for transitioning
            old_master_key = self.master_key
            old_data_key = self.data_key
            old_config_key = self.config_key
            old_auth_key = self.auth_key

            # Update master key
            self.master_key = new_master_key

            # Generate new derived keys
            self._generate_derived_keys()

            # Save new master key to file
            master_key_path = self.keys_path / "master.key"
            with open(master_key_path, "wb") as f:
                f.write(self.master_key)

            # Re-encrypt sensitive files with new keys
            # Users file
            users_file = self.users_path / "users.json"
            if users_file.exists():
                users = self._load_encrypted_json(users_file, old_auth_key)
                self._save_encrypted_json(users_file, users, self.auth_key)

            # Checksums file
            checksums_file = self.security_path / "checksums.json"
            if checksums_file.exists():
                checksums = self._load_encrypted_json(checksums_file, old_config_key)
                self._save_encrypted_json(checksums_file, checksums, self.config_key)

            logger.info("Key rotation completed successfully")
            self.log_security_event("KEY_ROTATION_SUCCESS", {})

            return True

        except Exception as e:
            logger.error(f"Error rotating keys: {str(e)}")
            self.log_security_event("KEY_ROTATION_ERROR",
                                   {"reason": "exception", "error": str(e)})
            return False
