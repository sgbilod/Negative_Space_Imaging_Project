"""
Ephemeral One-Time-Pad Encryption Service (Project "NyxCom")

This package provides theoretically unbreakable, "one-time-pad" encryption for ultra-secure
communications using ever-changing negative space configurations.
"""

from .ephemeral_encryption_service import (
    EphemeralKeyStream,
    SecureDataEscrow,
    EphemeralEncryptionService
)

__all__ = [
    'EphemeralKeyStream',
    'SecureDataEscrow',
    'EphemeralEncryptionService'
]
