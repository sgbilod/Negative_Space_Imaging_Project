"""
Key Rotation Scheduler Module
=============================

Implements automatic cryptographic key rotation with:
- Schedule-based key generation
- Graceful old key deprecation
- Dual-key support (current + previous)
- Zero-downtime rotation
- Complete audit trail

Philosophy: Keys have lifecycles. Rotate them regularly to minimize damage from compromise.

Author: @CIPHER - Advanced Cryptography & Security
Date: December 2025
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64
import secrets

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False
    logger = logging.getLogger("key_rotation_scheduler")
    logger.warning("schedule module not installed. Use APScheduler instead.")

logger = logging.getLogger("key_rotation_scheduler")


class KeyStatus(Enum):
    """Status of a cryptographic key in its lifecycle."""
    ACTIVE = "active"           # Currently in use for encryption/signing
    DEPRECATED = "deprecated"   # Can decrypt but should not encrypt
    RETIRED = "retired"         # No longer used, kept for audit trail
    COMPROMISED = "compromised" # Immediately revoked
    PENDING = "pending"         # Scheduled for activation


class KeyRotationPolicy(Enum):
    """Key rotation policy types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


@dataclass
class KeyMetadata:
    """Metadata for a stored key."""
    key_id: str
    algorithm: str
    key_size: int
    created_at: datetime
    activated_at: Optional[datetime]
    deprecated_at: Optional[datetime]
    status: KeyStatus
    rotation_count: int = 0
    parent_key_id: Optional[str] = None  # Key used to encrypt this key

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d['status'] = self.status.value
        d['created_at'] = self.created_at.isoformat()
        d['activated_at'] = self.activated_at.isoformat() if self.activated_at else None
        d['deprecated_at'] = self.deprecated_at.isoformat() if self.deprecated_at else None
        return d

    @staticmethod
    def from_dict(d: Dict) -> 'KeyMetadata':
        """Create from JSON dict."""
        d = d.copy()
        d['status'] = KeyStatus(d['status'])
        d['created_at'] = datetime.fromisoformat(d['created_at'])
        d['activated_at'] = datetime.fromisoformat(d['activated_at']) if d.get('activated_at') else None
        d['deprecated_at'] = datetime.fromisoformat(d['deprecated_at']) if d.get('deprecated_at') else None
        return KeyMetadata(**d)


class KeyRotationScheduler:
    """
    Manages automatic key rotation with zero-downtime deployment.

    Strategy:
    1. Generate new key with PENDING status
    2. Use new key for encryption going forward
    3. Keep old key to decrypt existing data
    4. Mark old key as DEPRECATED
    5. Eventually retire old key (keep for audit trail)
    """

    def __init__(
        self,
        storage_dir: str = "./security/keys",
        policy: KeyRotationPolicy = KeyRotationPolicy.MONTHLY,
        key_size: int = 32  # 256 bits for Fernet
    ):
        """
        Initialize key rotation scheduler.

        Args:
            storage_dir: Directory for key storage
            policy: Rotation policy (daily, weekly, monthly, etc.)
            key_size: Key size in bytes (default 256 bits)
        """
        self.storage_dir = Path(storage_dir)
        self.policy = policy
        self.key_size = key_size

        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.keys_file = self.storage_dir / "keys.json"
        self.metadata_file = self.storage_dir / "metadata.json"

        # In-memory key cache
        self._keys: Dict[str, str] = {}  # key_id -> key_material
        self._metadata: Dict[str, KeyMetadata] = {}  # key_id -> metadata

        # Thread safety
        self._lock = threading.RLock()

        # Scheduler thread
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Load existing keys
        self._load_keys()

        logger.info(
            f"Initialized KeyRotationScheduler: policy={policy.value}, "
            f"key_size={key_size} bytes"
        )

    def _load_keys(self) -> None:
        """Load keys and metadata from disk."""
        with self._lock:
            if self.metadata_file.exists():
                try:
                    with open(self.metadata_file, 'r') as f:
                        metadata_dict = json.load(f)
                        self._metadata = {
                            k: KeyMetadata.from_dict(v)
                            for k, v in metadata_dict.items()
                        }
                    logger.info(f"Loaded {len(self._metadata)} keys from disk")
                except Exception as e:
                    logger.error(f"Error loading keys: {e}")

            if self.keys_file.exists():
                try:
                    with open(self.keys_file, 'r') as f:
                        self._keys = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading key material: {e}")

    def _save_keys(self) -> None:
        """Save keys and metadata to disk."""
        with self._lock:
            try:
                # Save metadata
                metadata_dict = {
                    k: v.to_dict() for k, v in self._metadata.items()
                }
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata_dict, f, indent=2)

                # Save keys (in encrypted form in production)
                with open(self.keys_file, 'w') as f:
                    json.dump(self._keys, f)

                logger.info("Keys saved to disk")
            except Exception as e:
                logger.error(f"Error saving keys: {e}")

    def generate_key(self) -> Tuple[str, str]:
        """
        Generate a new cryptographic key.

        Returns:
            Tuple of (key_id, key_material)
        """
        # Generate key material
        key_material = base64.b64encode(secrets.token_bytes(self.key_size)).decode()

        # Generate key ID
        key_id = f"key_{int(datetime.utcnow().timestamp())}_{secrets.token_hex(4)}"

        # Create metadata
        metadata = KeyMetadata(
            key_id=key_id,
            algorithm="Fernet",
            key_size=self.key_size,
            created_at=datetime.utcnow(),
            activated_at=None,
            deprecated_at=None,
            status=KeyStatus.PENDING,
            rotation_count=0
        )

        with self._lock:
            self._keys[key_id] = key_material
            self._metadata[key_id] = metadata

        logger.info(f"Generated new key: {key_id}")
        return key_id, key_material

    def rotate_key(self) -> Tuple[str, str]:
        """
        Perform a key rotation.

        Steps:
        1. Deprecate current active key
        2. Generate new key
        3. Activate new key
        4. Log rotation event

        Returns:
            Tuple of (new_key_id, new_key_material)
        """
        with self._lock:
            # Deprecate current active key
            active_key = self.get_active_key()
            if active_key:
                active_key_id = active_key[0]
                if active_key_id in self._metadata:
                    self._metadata[active_key_id].status = KeyStatus.DEPRECATED
                    self._metadata[active_key_id].deprecated_at = datetime.utcnow()
                    logger.info(f"Deprecated key: {active_key_id}")

            # Generate new key
            new_key_id, new_key_material = self.generate_key()

            # Activate new key
            self._metadata[new_key_id].status = KeyStatus.ACTIVE
            self._metadata[new_key_id].activated_at = datetime.utcnow()
            self._metadata[new_key_id].rotation_count = (active_key[2] + 1) if active_key else 1

            logger.info(f"Activated new key: {new_key_id}")

            self._save_keys()

        return new_key_id, new_key_material

    def get_active_key(self) -> Optional[Tuple[str, str, int]]:
        """
        Get the currently active key.

        Returns:
            Tuple of (key_id, key_material, rotation_count) or None
        """
        with self._lock:
            for key_id, metadata in self._metadata.items():
                if metadata.status == KeyStatus.ACTIVE:
                    key_material = self._keys.get(key_id)
                    if key_material:
                        return (key_id, key_material, metadata.rotation_count)
            return None

    def get_key(self, key_id: str) -> Optional[str]:
        """
        Get key material by ID (for decryption of old data).

        Args:
            key_id: Key identifier

        Returns:
            Key material or None if not found
        """
        with self._lock:
            return self._keys.get(key_id)

    def get_decryption_keys(self) -> Dict[str, str]:
        """
        Get all non-retired keys for decryption.

        Returns decryption keys in priority order:
        1. Active key (newest)
        2. Deprecated keys (older)
        3. NOT compromised or retired
        """
        with self._lock:
            keys = {}
            for key_id, metadata in self._metadata.items():
                if metadata.status not in [KeyStatus.RETIRED, KeyStatus.COMPROMISED]:
                    key_material = self._keys.get(key_id)
                    if key_material:
                        keys[key_id] = key_material
            return keys

    def mark_key_deprecated(self, key_id: str) -> bool:
        """
        Mark a key as deprecated (can decrypt but not encrypt).

        Args:
            key_id: Key identifier

        Returns:
            True if successful
        """
        with self._lock:
            if key_id in self._metadata:
                self._metadata[key_id].status = KeyStatus.DEPRECATED
                self._metadata[key_id].deprecated_at = datetime.utcnow()
                self._save_keys()
                logger.info(f"Marked key as deprecated: {key_id}")
                return True
            return False

    def mark_key_compromised(self, key_id: str) -> bool:
        """
        Mark a key as compromised (immediately revoke).

        Args:
            key_id: Key identifier

        Returns:
            True if successful
        """
        with self._lock:
            if key_id in self._metadata:
                self._metadata[key_id].status = KeyStatus.COMPROMISED
                self._save_keys()
                logger.critical(f"Key marked as compromised: {key_id}")
                return True
            return False

    def validate_key_age(self, max_age_days: int = 30) -> Dict[str, bool]:
        """
        Validate that keys are not too old.

        Returns:
            Dictionary of {key_id: is_valid}
        """
        with self._lock:
            result = {}
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

            for key_id, metadata in self._metadata.items():
                is_valid = metadata.created_at > cutoff_date
                result[key_id] = is_valid

                if not is_valid and metadata.status == KeyStatus.ACTIVE:
                    logger.warning(f"Active key is too old: {key_id} ({metadata.created_at})")

            return result

    def schedule_rotation(self) -> None:
        """Start automatic key rotation scheduling."""
        if not HAS_SCHEDULE:
            logger.warning("schedule module not installed. Use APScheduler instead.")
            return

        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("Scheduler already running")
            return

        def rotation_worker():
            """Worker thread for scheduled rotations."""
            while not self._stop_event.is_set():
                try:
                    # Schedule rotation based on policy
                    if self.policy == KeyRotationPolicy.DAILY:
                        schedule.every().day.at("02:00").do(self.rotate_key)
                    elif self.policy == KeyRotationPolicy.WEEKLY:
                        schedule.every().monday.at("02:00").do(self.rotate_key)
                    elif self.policy == KeyRotationPolicy.MONTHLY:
                        schedule.every().month.do(self.rotate_key)
                    elif self.policy == KeyRotationPolicy.QUARTERLY:
                        schedule.every(91).days.do(self.rotate_key)
                    elif self.policy == KeyRotationPolicy.ANNUAL:
                        schedule.every(365).days.do(self.rotate_key)

                    # Run pending jobs
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute

                except Exception as e:
                    logger.error(f"Error in rotation worker: {e}")
                    time.sleep(60)

        self._scheduler_thread = threading.Thread(
            target=rotation_worker,
            daemon=True,
            name="KeyRotationScheduler"
        )
        self._scheduler_thread.start()
        logger.info("Key rotation scheduler started")

    def stop_rotation(self) -> None:
        """Stop the rotation scheduler."""
        self._stop_event.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logger.info("Key rotation scheduler stopped")

    def get_key_status_report(self) -> Dict:
        """
        Get current status of all keys.

        Returns:
            Comprehensive key status report
        """
        with self._lock:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "policy": self.policy.value,
                "total_keys": len(self._metadata),
                "keys_by_status": {},
                "active_key": None,
                "key_details": []
            }

            # Group by status
            for status in KeyStatus:
                report["keys_by_status"][status.value] = sum(
                    1 for m in self._metadata.values() if m.status == status
                )

            # Get active key
            active_key = self.get_active_key()
            if active_key:
                report["active_key"] = active_key[0]

            # Key details
            for key_id, metadata in self._metadata.items():
                report["key_details"].append({
                    "key_id": key_id,
                    "status": metadata.status.value,
                    "created_at": metadata.created_at.isoformat(),
                    "activated_at": metadata.activated_at.isoformat() if metadata.activated_at else None,
                    "deprecated_at": metadata.deprecated_at.isoformat() if metadata.deprecated_at else None,
                    "age_days": (datetime.utcnow() - metadata.created_at).days,
                    "rotation_count": metadata.rotation_count
                })

            return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create scheduler
    scheduler = KeyRotationScheduler(
        storage_dir="./test_keys",
        policy=KeyRotationPolicy.DAILY
    )

    print("\n=== Key Rotation Demonstration ===\n")

    # Generate initial key
    print("1. Generating initial key...")
    key_id, key_material = scheduler.generate_key()
    print(f"   Key ID: {key_id}")
    print(f"   Key: {key_material[:16]}...")

    # Activate it
    print("\n2. Activating key...")
    scheduler._metadata[key_id].status = KeyStatus.ACTIVE
    scheduler._metadata[key_id].activated_at = datetime.utcnow()
    scheduler._save_keys()

    # Get active key
    print("\n3. Retrieving active key...")
    active = scheduler.get_active_key()
    if active:
        print(f"   Active Key ID: {active[0]}")
        print(f"   Rotation Count: {active[2]}")

    # Perform rotation
    print("\n4. Performing key rotation...")
    new_key_id, new_key = scheduler.rotate_key()
    print(f"   New Key ID: {new_key_id}")

    # Get status report
    print("\n5. Key Status Report:")
    report = scheduler.get_key_status_report()
    print(f"   Total Keys: {report['total_keys']}")
    print(f"   Active Key: {report['active_key']}")
    print(f"   Status Distribution: {report['keys_by_status']}")

    # Validate key age
    print("\n6. Validating key age (max 30 days)...")
    age_valid = scheduler.validate_key_age(max_age_days=30)
    for key_id, is_valid in age_valid.items():
        status = "✅ Valid" if is_valid else "❌ Too Old"
        print(f"   {key_id}: {status}")

    print("\n=== Demonstration Complete ===")
