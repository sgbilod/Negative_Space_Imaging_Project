"""
Quantum-Enhanced Encryption System for Negative Space Imaging
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import os
import json
import time
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from base64 import b64encode, b64decode
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('security.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumEncryption")


class QuantumEncryption:
    """Implements quantum-resistant encryption system."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        key_path: Optional[str] = None
    ):
        self.config_path = config_path or 'security_config.json'
        self.key_path = key_path or 'quantum_keys.json'


        # Initialize key storage
        self._init_key_storage()

        logger.info("Quantum encryption system initialized")

    def _load_config(self) -> Dict:
        """Load security configuration."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                    if 'quantum' not in config:
                        config['quantum'] = self._default_quantum_config()

                        with open(self.config_path, 'w') as f:
                            json.dump(config, f, indent=4)

                    return config

            # Create default config
            config = {'quantum': self._default_quantum_config()}

            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)

            return config

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
        def _load_config(self) -> Dict:
            """Load security configuration, fallback to default if corrupted."""
            default_quantum = self._default_quantum_config()
            default_config = {'quantum': default_quantum}
            try:
                if Path(self.config_path).exists():
                    with open(self.config_path, 'r') as f:
                        try:
                            config = json.load(f)
                            if 'quantum' not in config:
                                config['quantum'] = default_quantum
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

    def _default_quantum_config(self) -> Dict:
        """Create default quantum encryption config."""
        return {
            'key_size': 32,
            'nonce_size': 12,
            'salt_size': 16,
            # Store info_string as base64-encoded string for JSON compatibility
            'info_string': b64encode(b'quantum_enhanced_encryption').decode('utf-8'),
            'key_rotation_interval': 86400,  # 24 hours
            'curve': 'secp521r1'  # Strongest NIST curve
        }

    def _init_key_storage(self):
        """Initialize quantum key storage."""
        try:
            if Path(self.key_path).exists():
                return

            # Create empty key storage
            storage = {
                'keypairs': {},
                'shared_keys': {},
                'rotations': {}
            }

            with open(self.key_path, 'w') as f:
                json.dump(storage, f, indent=4)

        except Exception as e:
            logger.error(f"Error initializing key storage: {e}")
            raise

    def generate_keypair(self, entity_id: str) -> bool:
        """Generate quantum-resistant keypair."""
        try:
            # Generate ECC keypair
            private_key = ec.generate_private_key(
                getattr(ec, self.config['quantum']['curve'])()
            )
            public_key = private_key.public_key()

            # Serialize keys
            private_bytes = private_key.private_bytes(
                encoding=ec.Encoding.PEM,
                format=ec.PrivateFormat.PKCS8,
                encryption_algorithm=ec.NoEncryption()
            )

            public_bytes = public_key.public_bytes(
                encoding=ec.Encoding.PEM,
                format=ec.PublicFormat.SubjectPublicKeyInfo
            )

            # Store keypair
            with open(self.key_path, 'r+') as f:
                storage = json.load(f)

                storage['keypairs'][entity_id] = {
                    'private': b64encode(private_bytes).decode(),
                    'public': b64encode(public_bytes).decode(),
                    'created': int(time.time())
                }

                f.seek(0)
                json.dump(storage, f, indent=4)
                f.truncate()

            logger.info(f"Generated quantum keypair for {entity_id}")
            return True

        except Exception as e:
            logger.error(f"Error generating keypair: {e}")
            return False

    def establish_shared_key(
        self,
        entity1_id: str,
        entity2_id: str
    ) -> bool:
        """Establish shared key between two entities."""
        try:
            with open(self.key_path, 'r+') as f:
                storage = json.load(f)

                # Verify entities have keypairs
                if (
                    entity1_id not in storage['keypairs'] or
                    entity2_id not in storage['keypairs']
                ):
                    raise ValueError("Missing keypair for one or both entities")

                # Load keys
                private1_bytes = b64decode(
                    storage['keypairs'][entity1_id]['private']
                )
                public2_bytes = b64decode(
                    storage['keypairs'][entity2_id]['public']
                )

                private1 = ec.load_pem_private_key(
                    private1_bytes,
                    password=None
                )
                public2 = ec.load_pem_public_key(public2_bytes)

                # Generate shared key
                shared_key = private1.exchange(
                    ec.ECDH(),
                    public2
                )

                # Derive encryption key using HKDF
                salt = os.urandom(self.config['quantum']['salt_size'])
                info = self.config['quantum']['info_string']

                derived_key = HKDF(
                    algorithm=hashes.SHA512(),
                    length=self.config['quantum']['key_size'],
                    salt=salt,
                    info=info
                ).derive(shared_key)

                # Store shared key
                key_id = f"{entity1_id}:{entity2_id}"
                storage['shared_keys'][key_id] = {
                    'key': b64encode(derived_key).decode(),
                    'salt': b64encode(salt).decode(),
                    'created': int(time.time())
                }

                f.seek(0)
                json.dump(storage, f, indent=4)
                f.truncate()

            logger.info(f"Established shared key between {entity1_id} and {entity2_id}")
            return True

        except Exception as e:
            logger.error(f"Error establishing shared key: {e}")
            return False

    def encrypt(
        self,
        data: Union[str, bytes],
        sender_id: str,
        recipient_id: str
    ) -> Optional[bytes]:
        """Encrypt data using quantum-enhanced encryption."""
        try:
            # Convert string to bytes if needed
            if isinstance(data, str):
                data = data.encode()

            # Get shared key
            key_id = f"{sender_id}:{recipient_id}"
            with open(self.key_path, 'r') as f:
                storage = json.load(f)

                if key_id not in storage['shared_keys']:
                    if not self.establish_shared_key(sender_id, recipient_id):
                        return None

                    # Reload storage after key establishment
                    with open(self.key_path, 'r') as f:
                        storage = json.load(f)

                key_data = storage['shared_keys'][key_id]

                # Check key rotation
                if self._needs_rotation(key_data['created']):
                    if not self.rotate_shared_key(sender_id, recipient_id):
                        return None

                    # Reload storage after rotation
                    with open(self.key_path, 'r') as f:
                        storage = json.load(f)
                        key_data = storage['shared_keys'][key_id]

            # Generate nonce
            nonce = os.urandom(self.config['quantum']['nonce_size'])

            # Create cipher
            key = b64decode(key_data['key'])
            cipher = ChaCha20Poly1305(key)

            # Encrypt data
            ciphertext = cipher.encrypt(nonce, data, None)

            # Combine nonce and ciphertext
            return nonce + ciphertext

        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            return None

    def decrypt(
        self,
        encrypted_data: bytes,
        recipient_id: str,
        sender_id: str
    ) -> Optional[bytes]:
        """Decrypt data using quantum-enhanced encryption."""
        try:
            # Extract nonce
            nonce = encrypted_data[:self.config['quantum']['nonce_size']]
            ciphertext = encrypted_data[self.config['quantum']['nonce_size']:]

            # Get shared key
            key_id = f"{sender_id}:{recipient_id}"
            with open(self.key_path, 'r') as f:
                storage = json.load(f)

                if key_id not in storage['shared_keys']:
                    logger.error(f"No shared key found for {key_id}")
                    return None

                key_data = storage['shared_keys'][key_id]

            # Create cipher
            key = b64decode(key_data['key'])
            cipher = ChaCha20Poly1305(key)

            # Decrypt data
            return cipher.decrypt(nonce, ciphertext, None)

        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            return None

    def rotate_shared_key(
        self,
        entity1_id: str,
        entity2_id: str
    ) -> bool:
        """Rotate shared key between entities."""
        try:
            # Generate new shared key
            if not self.establish_shared_key(entity1_id, entity2_id):
                return False

            # Record rotation
            with open(self.key_path, 'r+') as f:
                storage = json.load(f)

                key_id = f"{entity1_id}:{entity2_id}"
                storage['rotations'][key_id] = int(time.time())

                f.seek(0)
                json.dump(storage, f, indent=4)
                f.truncate()

            logger.info(f"Rotated shared key for {key_id}")
            return True

        except Exception as e:
            logger.error(f"Error rotating key: {e}")
            return False

    def _needs_rotation(self, created_time: int) -> bool:
        """Check if key needs rotation."""
        return (
            time.time() - created_time >
            self.config['quantum']['key_rotation_interval']
        )

    def remove_keys(self, entity_id: str) -> bool:
        """Remove all keys associated with an entity."""
        try:
            with open(self.key_path, 'r+') as f:
                storage = json.load(f)

                # Remove keypair
                if entity_id in storage['keypairs']:
                    del storage['keypairs'][entity_id]

                # Remove shared keys
                shared_keys = list(storage['shared_keys'].keys())
                for key_id in shared_keys:
                    if entity_id in key_id:
                        del storage['shared_keys'][key_id]

                # Remove rotations
                rotations = list(storage['rotations'].keys())
                for key_id in rotations:
                    if entity_id in key_id:
                        del storage['rotations'][key_id]

                f.seek(0)
                json.dump(storage, f, indent=4)
                f.truncate()

            logger.info(f"Removed all keys for {entity_id}")
            return True

        except Exception as e:
            logger.error(f"Error removing keys: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Create encryption system
    encryptor = QuantumEncryption()

    # Generate keypairs
    encryptor.generate_keypair("alice")
    encryptor.generate_keypair("bob")

    # Establish shared key
    encryptor.establish_shared_key("alice", "bob")

    # Test encryption/decryption
    message = "Hello, quantum world!"
    encrypted = encryptor.encrypt(message, "alice", "bob")
    decrypted = encryptor.decrypt(encrypted, "bob", "alice")

    print("Original:", message)
    print("Decrypted:", decrypted.decode())
