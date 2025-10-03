"""
Quantum Encryption System
Copyright (c) 2025 Stephen Bilodeau
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import hashlib
import hmac


class QuantumEncryption:
    """Quantum-secure encryption system"""

    def __init__(self):
        self.dimension = 128
        self.key_space = 1024
        self.quantum_keys = {}

    def generate_quantum_key(
        self,
        quantum_state: np.ndarray,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate quantum encryption key"""
        if not timestamp:
            timestamp = datetime.now()

        # Normalize quantum state
        quantum_state = np.clip(quantum_state, 0, 1)
        if np.sum(quantum_state) == 0:
            quantum_state = np.ones_like(quantum_state) / quantum_state.size
        else:
            quantum_state = quantum_state / np.sum(quantum_state)

        # Generate quantum components
        temporal_key = self._generate_temporal_key(timestamp)
        spatial_key = self._generate_spatial_key(quantum_state)
        entropy_key = self._generate_entropy_key(quantum_state)

        # Combine keys
        combined_key = self._combine_quantum_keys(
            temporal_key,
            spatial_key,
            entropy_key
        )

        # Store key
        key_id = hashlib.sha256(combined_key).hexdigest()
        self.quantum_keys[key_id] = {
            'key': combined_key,
            'timestamp': timestamp,
            'state_hash': self._hash_quantum_state(quantum_state)
        }

        return {
            'key_id': key_id,
            'timestamp': timestamp,
            'dimension': self.dimension
        }

    def encrypt_quantum_signature(
        self,
        signature: Dict[str, Any],
        key_id: str
    ) -> Dict[str, Any]:
        """Encrypt quantum signature"""
        if key_id not in self.quantum_keys:
            raise ValueError("Invalid quantum key")

        key = self.quantum_keys[key_id]['key']

        # Encrypt signature components
        encrypted_data = self._quantum_encrypt(
            signature['signature'],
            key
        )

        # Generate authentication tag
        auth_tag = self._generate_auth_tag(encrypted_data, key)

        return {
            'encrypted_signature': encrypted_data,
            'auth_tag': auth_tag,
            'key_id': key_id,
            'timestamp': signature['timestamp']
        }

    def decrypt_quantum_signature(
        self,
        encrypted_signature: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decrypt quantum signature"""
        key_id = encrypted_signature['key_id']
        if key_id not in self.quantum_keys:
            raise ValueError("Invalid quantum key")

        key = self.quantum_keys[key_id]['key']

        # Verify authentication tag
        if not self._verify_auth_tag(
            encrypted_signature['encrypted_signature'],
            encrypted_signature['auth_tag'],
            key
        ):
            raise ValueError("Signature authentication failed")

        # Decrypt signature
        decrypted_data = self._quantum_decrypt(
            encrypted_signature['encrypted_signature'],
            key
        )

        return {
            'signature': decrypted_data,
            'timestamp': encrypted_signature['timestamp'],
            'verified': True
        }

    def _generate_temporal_key(self, timestamp: datetime) -> np.ndarray:
        """Generate temporal component of quantum key"""
        # Use timestamp as seed (ensure 32-bit integer)
        seed = int(timestamp.timestamp() * 1000) & 0xFFFFFFFF
        np.random.seed(seed)
        temporal_key = np.random.random(self.dimension)
        return temporal_key / np.sum(temporal_key)  # Normalize

    def _generate_spatial_key(self, quantum_state: np.ndarray) -> np.ndarray:
        """Generate spatial component of quantum key"""
        # Calculate spatial features with proper axis spacing
        spatial_features = np.mean(quantum_state, axis=(0, 1, 2))
        # Ensure non-zero
        if np.sum(spatial_features) == 0:
            spatial_features = np.ones_like(spatial_features)
        # Expand to full dimension
        repeat_count = self.dimension // len(spatial_features) + 1
        expanded = np.tile(spatial_features, repeat_count)[:self.dimension]
        return expanded / np.sum(expanded)  # Normalize

    def _generate_entropy_key(self, quantum_state: np.ndarray) -> np.ndarray:
        """Generate entropy component of quantum key"""
        # Calculate quantum entropy with bounds
        bounded_state = np.clip(quantum_state, 1e-10, 1.0)
        entropy = -np.sum(bounded_state * np.log2(bounded_state), axis=-1)
        entropy_flat = entropy.flatten()

        # Handle zero/infinite entropy
        if np.any(np.isinf(entropy_flat)) or np.sum(entropy_flat) == 0:
            entropy_flat = np.ones_like(entropy_flat)

        # Expand to full dimension
        repeat_count = self.dimension // len(entropy_flat) + 1
        expanded = np.tile(entropy_flat, repeat_count)[:self.dimension]
        return expanded / np.sum(expanded)  # Normalize

    def _combine_quantum_keys(
        self,
        temporal: np.ndarray,
        spatial: np.ndarray,
        entropy: np.ndarray
    ) -> bytes:
        """Combine quantum key components with proper normalization"""
        # Ensure each component is properly normalized
        temporal = temporal / np.sum(temporal)
        spatial = spatial / np.sum(spatial)
        entropy = entropy / np.sum(entropy)

        # Weighted combination
        combined = 0.4 * temporal + 0.3 * spatial + 0.3 * entropy
        combined = combined / np.sum(combined)  # Final normalization
        return combined.tobytes()

    def _hash_quantum_state(self, quantum_state: np.ndarray) -> str:
        """Generate hash of normalized quantum state"""
        normalized_state = quantum_state / np.sum(quantum_state)
        return hashlib.sha256(normalized_state.tobytes()).hexdigest()

    def _quantum_encrypt(
        self,
        data: np.ndarray,
        key: bytes
    ) -> bytes:
        """Encrypt data using quantum key with proper normalization"""
        # Normalize input data safely
        data_sum = np.sum(data)
        if np.isfinite(data_sum) and data_sum != 0:
            data = data / data_sum
        else:
            data = np.ones_like(data) / data.size

        # Convert normalized data to bytes
        data_bytes = data.tobytes()

        # Generate one-time pad
        pad = hashlib.shake_256(key).digest(len(data_bytes))

        # Apply one-time pad
        encrypted = bytes(a ^ b for a, b in zip(data_bytes, pad))
        return encrypted

    def _quantum_decrypt(
        self,
        encrypted_data: bytes,
        key: bytes
    ) -> np.ndarray:
        """Decrypt data using quantum key and normalize output"""
        # Regenerate one-time pad
        pad = hashlib.shake_256(key).digest(len(encrypted_data))

        # Remove one-time pad and convert back to array
        decrypted_bytes = bytes(a ^ b for a, b in zip(encrypted_data, pad))
        decrypted = np.frombuffer(decrypted_bytes, dtype=np.float64)

        # Normalize output
        if np.sum(decrypted) != 0:
            decrypted = decrypted / np.sum(decrypted)
        return decrypted

    def _generate_auth_tag(self, data: bytes, key: bytes) -> bytes:
        """Generate authentication tag using HMAC"""
        return hmac.new(key, data, hashlib.sha256).digest()

    def _verify_auth_tag(
        self,
        data: bytes,
        auth_tag: bytes,
        key: bytes
    ) -> bool:
        """Verify authentication tag using constant-time comparison"""
        expected_tag = self._generate_auth_tag(data, key)
        return hmac.compare_digest(auth_tag, expected_tag)
