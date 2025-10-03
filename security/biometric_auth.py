"""
Biometric Authentication System for Negative Space Imaging
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import hashlib
import hmac
import os
import json
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from base64 import b64encode, b64decode
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('security.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BiometricAuth")


class BiometricVerifier:
    """Handles biometric verification and secure template storage."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        storage_path: Optional[str] = None
    ):
        self.config_path = config_path or 'security_config.json'
        self.storage_path = storage_path or 'security_store.json'
        self.config = self._load_config()
        self._init_secure_storage()
        self.key = self._generate_key()
        self.fernet = Fernet(self.key)
        logger.info("Biometric verification system initialized")

    def _load_config(self) -> Dict:
        """Load security configuration, fallback to default if corrupted."""
        default_config = {
            'min_confidence': 0.95,
            'max_attempts': 3,
            'lockout_duration': 300,  # 5 minutes
            'template_expiry': 86400,  # 24 hours
            'salt_length': 32,
            'hash_iterations': 100000,
            'quantum': {
                'key_size': 32,
                'nonce_size': 12,
                'salt_size': 16,
                'info_string': 'cXVhbnR1bV9lbmhhbmNlZF9lbmNyeXB0aW9u',
                'key_rotation_interval': 86400,
                'curve': 'secp521r1'
            }
        }
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    try:
                        config = json.load(f)
                        # Ensure quantum section exists
                        if 'quantum' not in config:
                            config['quantum'] = default_config['quantum']
                            with open(self.config_path, 'w') as fw:
                                json.dump(config, fw, indent=4)
                        return config
                    except Exception:
                        # Corrupted file, overwrite with default
                        with open(self.config_path, 'w') as fw:
                            json.dump(default_config, fw, indent=4)
                        return default_config
            # File does not exist, create default
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _init_secure_storage(self):
        """Initialize secure template storage."""
        try:
            if Path(self.storage_path).exists():
                return
            storage = {
                'templates': {},
                'attempts': {},
                'lockouts': {}
            }
            with open(self.storage_path, 'w') as f:
                json.dump(storage, f, indent=4)
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
            raise

    def _generate_key(self) -> bytes:
        """Generate encryption key using PBKDF2."""
        try:
            salt = os.urandom(self.config['salt_length'])
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.config['hash_iterations']
            )
            key = kdf.derive(os.urandom(32))
            return b64encode(key)
        except Exception as e:
            logger.error(f"Error generating key: {e}")
            raise

    def _secure_hash(
        self,
        data: bytes,
        salt: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """Create secure hash of biometric data."""
        try:
            if salt is None:
                salt = os.urandom(self.config['salt_length'])
            hasher = hashlib.sha3_512()
            hasher.update(salt + data)
            return hasher.digest(), salt
        except Exception as e:
            logger.error(f"Error creating hash: {e}")
            raise

    def _verify_hash(
        self,
        data: bytes,
        stored_hash: bytes,
        salt: bytes
    ) -> bool:
        """Verify hash matches stored template."""
        try:
            test_hash, _ = self._secure_hash(data, salt)
            return hmac.compare_digest(test_hash, stored_hash)
        except Exception as e:
            logger.error(f"Error verifying hash: {e}")
            return False

    def enroll_biometric(
        self,
        user_id: str,
        biometric_data: bytes
    ) -> bool:
        """Enroll new biometric template."""
        try:
            template_hash, salt = self._secure_hash(biometric_data)
            encrypted_template = self.fernet.encrypt(template_hash)
            with open(self.storage_path, 'r+') as f:
                storage = json.load(f)
                storage['templates'][user_id] = {
                    'template': b64encode(encrypted_template).decode(),
                    'salt': b64encode(salt).decode(),
                    'created': int(time.time())
                }
                f.seek(0)
                json.dump(storage, f, indent=4)
                f.truncate()
            logger.info(f"Enrolled biometric template for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error enrolling biometric: {e}")
            return False

    def verify_biometric(
        self,
        user_id: str,
        biometric_data: bytes
    ) -> bool:
        """Verify biometric against stored template."""
        try:
            if self._is_locked_out(user_id):
                logger.warning(f"User {user_id} is locked out")
                return False
            with open(self.storage_path, 'r') as f:
                storage = json.load(f)
            if user_id not in storage['templates']:
                logger.warning(f"No template found for user {user_id}")
                return False
            template_data = storage['templates'][user_id]
            if self._is_template_expired(template_data['created']):
                logger.warning(f"Template expired for user {user_id}")
                return False
            encrypted_template = b64decode(template_data['template'])
            stored_template = self.fernet.decrypt(encrypted_template)
            salt = b64decode(template_data['salt'])
            if self._verify_hash(biometric_data, stored_template, salt):
                self._clear_attempts(user_id)
                logger.info(f"Successful biometric verification for user {user_id}")
                return True
            self._record_failed_attempt(user_id)
            logger.warning(f"Failed biometric verification for user {user_id}")
            return False
        except Exception as e:
            logger.error(f"Error verifying biometric: {e}")
            return False

    def _is_template_expired(self, created_time: int) -> bool:
        """Check if template has expired."""
        return (
            time.time() - created_time >
            self.config['template_expiry']
        )

    def _is_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out."""
        try:
            with open(self.storage_path, 'r') as f:
                storage = json.load(f)
            if user_id in storage['lockouts']:
                lockout_time = storage['lockouts'][user_id]
                if (
                    time.time() - lockout_time <
                    self.config['lockout_duration']
                ):
                    return True
                self._clear_lockout(user_id)
            return False
        except Exception as e:
            logger.error(f"Error checking lockout: {e}")
            return False

    def _record_failed_attempt(self, user_id: str):
        """Record failed verification attempt."""
        try:
            with open(self.storage_path, 'r+') as f:
                storage = json.load(f)
                if user_id not in storage['attempts']:
                    storage['attempts'][user_id] = {
                        'count': 0,
                        'first_attempt': int(time.time())
                    }
                attempts = storage['attempts'][user_id]
                attempts['count'] += 1
                if attempts['count'] >= self.config['max_attempts']:
                    storage['lockouts'][user_id] = int(time.time())
                    logger.warning(f"User {user_id} locked out")
                f.seek(0)
                json.dump(storage, f, indent=4)
                f.truncate()
        except Exception as e:
            logger.error(f"Error recording attempt: {e}")

    def _clear_attempts(self, user_id: str):
        """Clear failed attempts after successful verification."""
        try:
            with open(self.storage_path, 'r+') as f:
                storage = json.load(f)
                if user_id in storage['attempts']:
                    del storage['attempts'][user_id]
                f.seek(0)
                json.dump(storage, f, indent=4)
                f.truncate()
        except Exception as e:
            logger.error(f"Error clearing attempts: {e}")

    def _clear_lockout(self, user_id: str):
        """Clear lockout after expiry."""
        try:
            with open(self.storage_path, 'r+') as f:
                storage = json.load(f)
                if user_id in storage['lockouts']:
                    del storage['lockouts'][user_id]
                f.seek(0)
                json.dump(storage, f, indent=4)
                f.truncate()
        except Exception as e:
            logger.error(f"Error clearing lockout: {e}")

    def remove_template(self, user_id: str) -> bool:
        """Remove stored biometric template."""
        try:
            with open(self.storage_path, 'r+') as f:
                storage = json.load(f)
                if user_id in storage['templates']:
                    del storage['templates'][user_id]
                f.seek(0)
                json.dump(storage, f, indent=4)
                f.truncate()
            logger.info(f"Removed biometric template for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing template: {e}")
            return False
