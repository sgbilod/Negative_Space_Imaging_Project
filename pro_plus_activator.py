#!/usr/bin/env python
"""
Pro-Plus License Activator
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module handles Pro-Plus license activation and validation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LicenseType(Enum):
    """License types."""
    BASIC = "basic"
    PROFESSIONAL = "professional"
    PRO_PLUS = "pro_plus"
    ENTERPRISE = "enterprise"


class LicenseStatus(Enum):
    """License status."""
    VALID = "valid"
    EXPIRED = "expired"
    INVALID = "invalid"
    NOT_ACTIVATED = "not_activated"


@dataclass
class LicenseInfo:
    """License information container."""
    license_type: LicenseType = LicenseType.BASIC
    status: LicenseStatus = LicenseStatus.NOT_ACTIVATED
    license_key: Optional[str] = None
    activation_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    features: Dict[str, bool] = field(default_factory=dict)
    organization: Optional[str] = None
    max_users: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "license_type": self.license_type.value,
            "status": self.status.value,
            "license_key": self.license_key[:8] + "..." if self.license_key else None,
            "activation_date": self.activation_date.isoformat() if self.activation_date else None,
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "features": self.features,
            "organization": self.organization,
            "max_users": self.max_users,
        }

    def is_valid(self) -> bool:
        """Check if license is valid."""
        if self.status != LicenseStatus.VALID:
            return False
        if self.expiration_date and datetime.utcnow() > self.expiration_date:
            return False
        return True


class ProPlusActivator:
    """
    Pro-Plus license activation and validation.
    
    Features unlocked with Pro-Plus:
    - Advanced negative space detection algorithms
    - GPU acceleration
    - HPC cluster integration
    - Priority support
    - Custom model training
    """

    PRO_PLUS_FEATURES = {
        "advanced_detection": "Advanced negative space detection algorithms",
        "gpu_acceleration": "GPU-accelerated processing",
        "hpc_integration": "HPC cluster integration",
        "priority_support": "Priority technical support",
        "custom_training": "Custom model training",
        "batch_processing": "Unlimited batch processing",
        "api_access": "Full API access",
        "export_formats": "All export formats",
    }

    def __init__(self, license_file: str = ".license"):
        """
        Initialize Pro-Plus activator.
        
        Args:
            license_file: Path to license file
        """
        self.license_file = Path(license_file)
        self._license_info: Optional[LicenseInfo] = None
        self._load_license()

    def _load_license(self) -> None:
        """Load license from file."""
        if self.license_file.exists():
            try:
                with open(self.license_file, "r") as f:
                    data = json.load(f)
                self._license_info = LicenseInfo(
                    license_type=LicenseType(data.get("license_type", "basic")),
                    status=LicenseStatus(data.get("status", "not_activated")),
                    license_key=data.get("license_key"),
                    activation_date=datetime.fromisoformat(data["activation_date"]) if data.get("activation_date") else None,
                    expiration_date=datetime.fromisoformat(data["expiration_date"]) if data.get("expiration_date") else None,
                    features=data.get("features", {}),
                    organization=data.get("organization"),
                    max_users=data.get("max_users", 1),
                )
                self._validate_license()
            except Exception as e:
                logger.warning(f"Failed to load license: {e}")
                self._license_info = LicenseInfo()
        else:
            self._license_info = LicenseInfo()

    def _save_license(self) -> None:
        """Save license to file."""
        if self._license_info:
            data = {
                "license_type": self._license_info.license_type.value,
                "status": self._license_info.status.value,
                "license_key": self._license_info.license_key,
                "activation_date": self._license_info.activation_date.isoformat() if self._license_info.activation_date else None,
                "expiration_date": self._license_info.expiration_date.isoformat() if self._license_info.expiration_date else None,
                "features": self._license_info.features,
                "organization": self._license_info.organization,
                "max_users": self._license_info.max_users,
            }
            with open(self.license_file, "w") as f:
                json.dump(data, f, indent=2)

    def _validate_license(self) -> bool:
        """Validate loaded license."""
        if not self._license_info or not self._license_info.license_key:
            return False

        # Check expiration
        if self._license_info.expiration_date:
            if datetime.utcnow() > self._license_info.expiration_date:
                self._license_info.status = LicenseStatus.EXPIRED
                return False

        return True

    def activate(
        self,
        license_key: str,
        organization: Optional[str] = None,
    ) -> bool:
        """
        Activate Pro-Plus license.
        
        Args:
            license_key: License key to activate
            organization: Organization name
            
        Returns:
            True if activation successful
        """
        # Validate key format (simple validation)
        if not self._validate_key_format(license_key):
            logger.error("Invalid license key format")
            return False

        # In a real implementation, this would verify with a license server
        # For demo purposes, we accept keys starting with "PROPLUS-"
        if not license_key.startswith("PROPLUS-"):
            logger.error("Invalid Pro-Plus license key")
            return False

        # Activate license
        self._license_info = LicenseInfo(
            license_type=LicenseType.PRO_PLUS,
            status=LicenseStatus.VALID,
            license_key=license_key,
            activation_date=datetime.utcnow(),
            expiration_date=datetime.utcnow() + timedelta(days=365),
            features={feature: True for feature in self.PRO_PLUS_FEATURES},
            organization=organization,
            max_users=10,
        )

        self._save_license()
        logger.info("Pro-Plus license activated successfully")
        return True

    def deactivate(self) -> bool:
        """Deactivate current license."""
        self._license_info = LicenseInfo()
        if self.license_file.exists():
            self.license_file.unlink()
        logger.info("License deactivated")
        return True

    def _validate_key_format(self, key: str) -> bool:
        """Validate license key format."""
        if not key or len(key) < 10:
            return False
        # Expected format: PROPLUS-XXXX-XXXX-XXXX
        parts = key.split("-")
        return len(parts) >= 2

    def get_license_info(self) -> LicenseInfo:
        """Get current license information."""
        return self._license_info or LicenseInfo()

    def is_pro_plus(self) -> bool:
        """Check if Pro-Plus is active."""
        return (
            self._license_info is not None
            and self._license_info.license_type == LicenseType.PRO_PLUS
            and self._license_info.is_valid()
        )

    def has_feature(self, feature: str) -> bool:
        """Check if a specific feature is available."""
        if not self._license_info:
            return False
        return self._license_info.features.get(feature, False)

    def get_available_features(self) -> Dict[str, str]:
        """Get list of available features."""
        if not self._license_info or not self._license_info.is_valid():
            return {}
        
        return {
            feature: description
            for feature, description in self.PRO_PLUS_FEATURES.items()
            if self._license_info.features.get(feature, False)
        }


# Global activator instance
_activator: Optional[ProPlusActivator] = None


def get_activator() -> ProPlusActivator:
    """Get the global Pro-Plus activator instance."""
    global _activator
    if _activator is None:
        _activator = ProPlusActivator()
    return _activator


def is_pro_plus_active() -> bool:
    """Check if Pro-Plus is active."""
    return get_activator().is_pro_plus()


def require_pro_plus(func):
    """Decorator to require Pro-Plus license."""
    def wrapper(*args, **kwargs):
        if not is_pro_plus_active():
            raise PermissionError(
                "This feature requires a Pro-Plus license. "
                "Visit https://negative-space.io/pro-plus for more information."
            )
        return func(*args, **kwargs)
    return wrapper


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    activator = ProPlusActivator()
    
    print("Pro-Plus License Activator")
    print("=" * 40)
    
    info = activator.get_license_info()
    print(f"\nCurrent License:")
    print(f"  Type: {info.license_type.value}")
    print(f"  Status: {info.status.value}")
    
    print("\nPro-Plus Features:")
    for feature, description in ProPlusActivator.PRO_PLUS_FEATURES.items():
        print(f"  - {feature}: {description}")
