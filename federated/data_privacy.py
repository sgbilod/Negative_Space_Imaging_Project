"""
Data Privacy Management for Federated Learning
Local-only data handling, anonymization, and privacy audit logging.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import hashlib
import json
from datetime import datetime
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class DataCategory(Enum):
    """Data sensitivity categories."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class PrivacyAuditLog:
    """Audit log for data access and operations."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    operation: str = ""
    user_id: str = ""
    data_category: str = ""
    action: str = ""
    status: str = "success"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "user_id": self.user_id,
            "data_category": self.data_category,
            "action": self.action,
            "status": self.status,
            "details": self.details,
        }


class DataPrivacyManager:
    """
    Manages local-only data with privacy guarantees.
    Ensures raw data never leaves client.
    """

    def __init__(
        self,
        client_id: str,
        data_category: DataCategory = DataCategory.CONFIDENTIAL,
        enable_audit_logging: bool = True,
    ):
        """
        Initialize data privacy manager.

        Args:
            client_id: Unique client identifier
            data_category: Data sensitivity level
            enable_audit_logging: Enable privacy audit logging
        """
        self.client_id = client_id
        self.data_category = data_category
        self.enable_audit_logging = enable_audit_logging

        self.local_data = None
        self.data_hash = None
        self.data_size = 0
        self.audit_logs: List[PrivacyAuditLog] = []

        logger.info(
            f"Data Privacy Manager initialized for client {client_id} | "
            f"Category: {data_category.value}"
        )

    def load_local_data(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
    ) -> bool:
        """
        Load local data from file (stays on client).

        Args:
            data_path: Path to local data file
            max_samples: Maximum samples to load

        Returns:
            True if successful
        """
        try:
            import pickle

            with open(data_path, 'rb') as f:
                self.local_data = pickle.load(f)

            if max_samples:
                self.local_data = self.local_data[:max_samples]

            self.data_size = len(self.local_data)
            self.data_hash = self._compute_hash()

            self._log_audit(
                operation="load_data",
                action="local_data_load",
                details={
                    "data_path": data_path,
                    "num_samples": self.data_size,
                    "hash": self.data_hash,
                },
            )

            logger.info(
                f"Local data loaded: {self.data_size} samples | Hash: {self.data_hash}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load local data: {e}")
            self._log_audit(
                operation="load_data",
                action="local_data_load",
                status="failure",
                details={"error": str(e)},
            )
            return False

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about local data without exposing raw data.

        Args:
            Returns:
            Statistics dictionary
        """
        if self.local_data is None:
            return {}

        stats = {
            "num_samples": self.data_size,
            "data_hash": self.data_hash,
            "data_category": self.data_category.value,
            "timestamp": datetime.now().isoformat(),
        }

        # Compute aggregate statistics safely
        if isinstance(self.local_data, np.ndarray):
            stats.update({
                "shape": self.local_data.shape,
                "dtype": str(self.local_data.dtype),
                "min": float(np.min(self.local_data)),
                "max": float(np.max(self.local_data)),
                "mean": float(np.mean(self.local_data)),
                "std": float(np.std(self.local_data)),
            })

        return stats

    def validate_data_distribution(
        self,
        other_stats: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Validate data distribution skew with other clients.

        Args:
            other_stats: Statistics from other client

        Returns:
            (is_valid, message) tuple
        """
        local_stats = self.get_data_statistics()

        if not local_stats or not other_stats:
            return True, "Insufficient statistics"

        # Check for extreme distribution skew
        local_mean = local_stats.get("mean", 0)
        other_mean = other_stats.get("mean", 0)

        if abs(local_mean - other_mean) > 10:
            return False, f"Distribution skew detected: means differ by {abs(local_mean - other_mean)}"

        return True, "Distribution acceptable"

    def anonymize_batch(
        self,
        batch: np.ndarray,
        anonymization_level: int = 1,
    ) -> np.ndarray:
        """
        Apply anonymization to batch before any transmission.

        Args:
            batch: Input batch
            anonymization_level: Level of anonymization (1-5)

        Returns:
            Anonymized batch
        """
        if anonymization_level == 0:
            return batch

        # Add small noise proportional to anonymization level
        noise_std = 0.01 * anonymization_level
        noise = np.random.normal(0, noise_std, batch.shape)

        anonymized = batch + noise

        return anonymized

    def split_into_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> List[np.ndarray]:
        """
        Split local data into batches for training.
        Data never leaves client in raw form.

        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data

        Returns:
            List of batches
        """
        if self.local_data is None:
            return []

        data = np.array(self.local_data)

        if shuffle:
            np.random.shuffle(data)

        batches = [
            data[i:i + batch_size]
            for i in range(0, len(data), batch_size)
        ]

        return batches

    def create_data_summary(self) -> Dict[str, Any]:
        """
        Create privacy-preserving summary of data.

        Args:
            Returns:
            Summary dictionary (safe to transmit)
        """
        return {
            "client_id": self.client_id,
            "num_samples": self.data_size,
            "data_hash": self.data_hash,
            "data_category": self.data_category.value,
            "timestamp": datetime.now().isoformat(),
        }

    def _compute_hash(self) -> str:
        """Compute hash of local data."""
        if self.local_data is None:
            return ""

        data_bytes = json.dumps(
            str(self.local_data),
            default=str,
        ).encode()

        return hashlib.sha256(data_bytes).hexdigest()

    def _log_audit(
        self,
        operation: str,
        action: str,
        status: str = "success",
        details: Optional[Dict] = None,
    ):
        """Log audit entry."""
        if not self.enable_audit_logging:
            return

        log_entry = PrivacyAuditLog(
            operation=operation,
            user_id=self.client_id,
            data_category=self.data_category.value,
            action=action,
            status=status,
            details=details or {},
        )

        self.audit_logs.append(log_entry)

    def get_audit_logs(self) -> List[Dict]:
        """Get audit logs."""
        return [log.to_dict() for log in self.audit_logs]


class DataValidator:
    """Validates data quality and privacy properties."""

    @staticmethod
    def validate_data_shape(
        data: np.ndarray,
        expected_shape: Tuple,
    ) -> Tuple[bool, str]:
        """
        Validate data shape.

        Args:
            data: Input data
            expected_shape: Expected shape

        Returns:
            (is_valid, message) tuple
        """
        if data.shape != expected_shape:
            return False, f"Shape mismatch: {data.shape} != {expected_shape}"
        return True, "Shape valid"

    @staticmethod
    def validate_data_range(
        data: np.ndarray,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        Validate data is within expected range.

        Args:
            data: Input data
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            (is_valid, message) tuple
        """
        if np.any(data < min_val) or np.any(data > max_val):
            return False, f"Data out of range [{min_val}, {max_val}]"
        return True, "Data range valid"

    @staticmethod
    def validate_no_nan_inf(
        data: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Validate data has no NaN or Inf values.

        Args:
            data: Input data

        Returns:
            (is_valid, message) tuple
        """
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return False, "Data contains NaN or Inf values"
        return True, "Data valid (no NaN/Inf)"

    @staticmethod
    def validate_sufficient_samples(
        data: np.ndarray,
        min_samples: int = 10,
    ) -> Tuple[bool, str]:
        """
        Validate data has sufficient samples.

        Args:
            data: Input data
            min_samples: Minimum required samples

        Returns:
            (is_valid, message) tuple
        """
        if len(data) < min_samples:
            return False, f"Insufficient samples: {len(data)} < {min_samples}"
        return True, "Sufficient samples"

    @staticmethod
    def check_differential_privacy_feasibility(
        dataset_size: int,
        sensitivity: float,
        target_epsilon: float,
    ) -> Tuple[bool, str]:
        """
        Check if DP is feasible for dataset size and epsilon.

        Args:
            dataset_size: Size of dataset
            sensitivity: Sensitivity of function
            target_epsilon: Target epsilon

        Returns:
            (is_feasible, message) tuple
        """
        # Rough heuristic: need at least ~100 samples per epsilon point
        min_samples = int(100 / target_epsilon)

        if dataset_size < min_samples:
            return False, (
                f"Dataset too small for DP: {dataset_size} < {min_samples} "
                f"(needed for ε={target_epsilon})"
            )

        return True, f"DP feasible (ε={target_epsilon} achievable)"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    manager = DataPrivacyManager(
        client_id="hospital_001",
        data_category=DataCategory.RESTRICTED,
    )

    # Create dummy data
    dummy_data = np.random.randn(100, 28, 28)

    # Validate
    validator = DataValidator()
    valid, msg = validator.validate_no_nan_inf(dummy_data)
    print(f"Data validation: {msg}")

    valid, msg = validator.validate_sufficient_samples(dummy_data, min_samples=50)
    print(f"Sufficient samples: {msg}")
