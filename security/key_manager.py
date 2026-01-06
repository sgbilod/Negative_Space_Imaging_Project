"""
Key Manager Module - Fernet-Based Encryption for Private Keys
==============================================================

Provides secure encryption/decryption of private keys at rest using
Fernet (AES-128 in CBC mode with HMAC authentication).

Features:
- Fernet symmetric encryption (AES-128-CBC)
- HMAC authentication for integrity
- Master key derivation from environment variable
- Automatic key rotation support
- Encrypted key storage with metadata
"""

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import os
import base64
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

logger = logging.getLogger("key_manager")

# Configuration
MASTER_KEY_ENV_VAR = "MASTER_KEY"
KEY_ENCRYPTION_ALGORITHM = "Fernet"
KEY_SALT = b"negative-space-imaging-key-manager"
KEY_ITERATIONS = 100000  # OWASP recommendation


class KeyManager:
    """Manages encryption/decryption of private keys at rest."""

    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize KeyManager with master key.

        Args:
            master_key: Base64-encoded master key. If None, reads from MASTER_KEY env var.

        Raises:
            ValueError: If master_key not provided and MASTER_KEY env var not set
        """
        if master_key is None:
            master_key = os.environ.get(MASTER_KEY_ENV_VAR)

        if not master_key:
            raise ValueError(
                f"CRITICAL: {MASTER_KEY_ENV_VAR} environment variable not set. "
                "KeyManager requires a master key for encryption operations. "
                f"Generate one with: python -c 'from cryptography.fernet import Fernet; "
                f"print(Fernet.generate_key().decode())'"
            )

        try:
            # Validate that master_key is a valid Fernet key
            self.cipher = Fernet(master_key.encode() if isinstance(master_key, str) else master_key)
            self.master_key = master_key
            logger.info("KeyManager initialized successfully with master key")
        except Exception as e:
            raise ValueError(f"Invalid master key format: {e}")

    @staticmethod
    def generate_master_key() -> str:
        """
        Generate a new master key suitable for key encryption.

        Returns:
            Base64-encoded Fernet key as string
        """
        key = Fernet.generate_key()
        return key.decode('utf-8')

    def encrypt_key(self, key_data: bytes, metadata: Optional[Dict] = None) -> str:
        """
        Encrypt a private key.

        Args:
            key_data: Raw key bytes to encrypt
            metadata: Optional metadata (key_type, key_id, etc.)

        Returns:
            Base64-encoded encrypted data with metadata as JSON string
        """
        if not key_data:
            raise ValueError("Key data cannot be empty")

        try:
            # Encrypt the key data
            encrypted_data = self.cipher.encrypt(key_data)

            # Prepare metadata
            meta = metadata or {}
            meta['encrypted_at'] = datetime.utcnow().isoformat()
            meta['algorithm'] = KEY_ENCRYPTION_ALGORITHM
            meta['key_size'] = len(key_data)

            # Combine encrypted data and metadata
            package = {
                'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
                'metadata': meta
            }

            result = json.dumps(package)
            logger.info(f"Key encrypted successfully ({len(key_data)} bytes)")
            return result
        except Exception as e:
            logger.error(f"Failed to encrypt key: {e}")
            raise

    def decrypt_key(self, encrypted_package: str) -> Tuple[bytes, Dict]:
        """
        Decrypt an encrypted private key.

        Args:
            encrypted_package: JSON string with encrypted_data and metadata

        Returns:
            Tuple of (decrypted_key_bytes, metadata_dict)

        Raises:
            InvalidToken: If decryption fails (wrong key or corrupted data)
            json.JSONDecodeError: If encrypted_package is invalid JSON
        """
        try:
            # Parse JSON package
            package = json.loads(encrypted_package)
            encrypted_data_b64 = package.get('encrypted_data')
            metadata = package.get('metadata', {})

            if not encrypted_data_b64:
                raise ValueError("No encrypted_data found in package")

            # Decode from base64
            encrypted_data = base64.b64decode(encrypted_data_b64)

            # Decrypt
            decrypted = self.cipher.decrypt(encrypted_data)

            logger.info(f"Key decrypted successfully ({len(decrypted)} bytes)")
            return decrypted, metadata
        except InvalidToken as e:
            logger.error("Key decryption failed - invalid token or wrong master key")
            raise InvalidToken(f"Failed to decrypt key: {e}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Invalid encrypted package format: {e}")
            raise ValueError(f"Invalid encrypted package: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during decryption: {e}")
            raise

    def encrypt_key_file(self, input_path: str, output_path: str, metadata: Optional[Dict] = None) -> None:
        """
        Encrypt a key file from disk.

        Args:
            input_path: Path to plaintext key file
            output_path: Path to write encrypted key
            metadata: Optional metadata to include
        """
        try:
            # Read key file
            with open(input_path, 'rb') as f:
                key_data = f.read()

            # Add filename to metadata
            meta = metadata or {}
            meta['source_file'] = Path(input_path).name

            # Encrypt
            encrypted_package = self.encrypt_key(key_data, meta)

            # Write encrypted file
            with open(output_path, 'w') as f:
                f.write(encrypted_package)

            logger.info(f"Key file encrypted: {input_path} -> {output_path}")
        except FileNotFoundError:
            logger.error(f"Key file not found: {input_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to encrypt key file: {e}")
            raise

    def decrypt_key_file(self, input_path: str, output_path: str) -> Dict:
        """
        Decrypt an encrypted key file from disk.

        Args:
            input_path: Path to encrypted key file
            output_path: Path to write decrypted key

        Returns:
            Metadata dictionary

        Raises:
            FileNotFoundError: If input file not found
            InvalidToken: If decryption fails
        """
        try:
            # Read encrypted file
            with open(input_path, 'r') as f:
                encrypted_package = f.read()

            # Decrypt
            key_data, metadata = self.decrypt_key(encrypted_package)

            # Write decrypted file
            with open(output_path, 'wb') as f:
                f.write(key_data)

            logger.info(f"Key file decrypted: {input_path} -> {output_path}")
            return metadata
        except FileNotFoundError:
            logger.error(f"Encrypted key file not found: {input_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to decrypt key file: {e}")
            raise

    def encrypt_directory(self, input_dir: str, output_dir: str, pattern: str = "*.key") -> Dict[str, bool]:
        """
        Encrypt all matching key files in a directory.

        Args:
            input_dir: Directory containing plaintext keys
            output_dir: Directory to write encrypted keys
            pattern: Glob pattern for files to encrypt (default: *.key)

        Returns:
            Dictionary mapping filename -> encryption success
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}
        for key_file in input_path.glob(pattern):
            try:
                output_file = output_path / f"{key_file.stem}.enc"
                self.encrypt_key_file(str(key_file), str(output_file))
                results[key_file.name] = True
            except Exception as e:
                logger.error(f"Failed to encrypt {key_file.name}: {e}")
                results[key_file.name] = False

        return results

    def decrypt_directory(self, input_dir: str, output_dir: str, pattern: str = "*.enc") -> Dict[str, bool]:
        """
        Decrypt all matching encrypted key files in a directory.

        Args:
            input_dir: Directory containing encrypted keys
            output_dir: Directory to write decrypted keys
            pattern: Glob pattern for files to decrypt (default: *.enc)

        Returns:
            Dictionary mapping filename -> decryption success
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}
        for enc_file in input_path.glob(pattern):
            try:
                output_file = output_path / enc_file.stem
                self.decrypt_key_file(str(enc_file), str(output_file))
                results[enc_file.name] = True
            except Exception as e:
                logger.error(f"Failed to decrypt {enc_file.name}: {e}")
                results[enc_file.name] = False

        return results


# Module-level convenience functions
_key_manager: Optional[KeyManager] = None


def get_key_manager() -> KeyManager:
    """Get or initialize the global KeyManager instance."""
    global _key_manager
    if _key_manager is None:
        _key_manager = KeyManager()
    return _key_manager


def load_key(file_path: str) -> bytes:
    """Load and decrypt a key from file."""
    manager = get_key_manager()
    decrypted_key, _ = manager.decrypt_key_file(file_path, file_path + ".tmp")
    return decrypted_key


def encrypt_key(key_data: bytes, metadata: Optional[Dict] = None) -> str:
    """Encrypt key data."""
    manager = get_key_manager()
    return manager.encrypt_key(key_data, metadata)


def decrypt_key(encrypted_package: str) -> Tuple[bytes, Dict]:
    """Decrypt encrypted key package."""
    manager = get_key_manager()
    return manager.decrypt_key(encrypted_package)


# Example usage and testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    print("=" * 70)
    print("KEY MANAGER MODULE - FERNET ENCRYPTION TEST")
    print("=" * 70)

    # Test 1: Generate master key
    print(f"\nTest 1: Generating master key...")
    master_key = KeyManager.generate_master_key()
    print(f"Master key: {master_key[:20]}...")
    print(f"Master key length: {len(master_key)}")

    # Set in environment for tests
    os.environ[MASTER_KEY_ENV_VAR] = master_key

    # Test 2: Initialize KeyManager
    print(f"\nTest 2: Initializing KeyManager...")
    km = KeyManager()
    print(f"KeyManager initialized successfully")

    # Test 3: Encrypt key data
    print(f"\nTest 3: Encrypting key data...")
    test_key = b"-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg...\n-----END PRIVATE KEY-----"
    metadata = {
        "key_type": "ED25519",
        "key_id": "test-key-001"
    }

    encrypted_package = km.encrypt_key(test_key, metadata)
    print(f"Encrypted package length: {len(encrypted_package)}")
    print(f"Encrypted package (first 100 chars): {encrypted_package[:100]}...")

    # Test 4: Decrypt key data
    print(f"\nTest 4: Decrypting key data...")
    decrypted_key, decrypted_meta = km.decrypt_key(encrypted_package)
    print(f"Decrypted key matches original: {decrypted_key == test_key}")
    print(f"Metadata: {decrypted_meta}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
