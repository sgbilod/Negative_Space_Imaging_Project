#!/usr/bin/env python
"""Test imports for security module."""
try:
    import cryptography
    print(f"cryptography version: {cryptography.__version__}")
    from cryptography.hazmat.primitives import hashes
    print("hashes module imported successfully")
    from cryptography.hazmat.primitives.asymmetric import padding
    print("padding module imported successfully")
except ImportError as e:
    print(f"Import error: {e}")

try:
    from secure_imaging_workflow import SecureImagingWorkflow
    print("SecureImagingWorkflow imported successfully")
except ImportError as e:
    print(f"Import error: {e}")

try:
    from multi_signature_demo import SignatureMode, SignatureStatus, Role, Signer
    print("All signature modules imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
