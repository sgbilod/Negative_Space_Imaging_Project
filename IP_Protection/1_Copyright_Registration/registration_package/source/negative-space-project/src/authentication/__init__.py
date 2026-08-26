"""
Authentication Module for Negative Space Signatures

This module provides authentication mechanisms using negative space signatures,
including both single-signature and multi-signature approaches.

Modules:
    multi_signature: Multi-signature authentication with various combination methods
"""

from .multi_signature import (
    MultiSignatureManager,
    SignatureCombiner,
    ThresholdVerifier,
    HierarchicalVerifier
)

__all__ = [
    'MultiSignatureManager',
    'SignatureCombiner',
    'ThresholdVerifier',
    'HierarchicalVerifier'
]
