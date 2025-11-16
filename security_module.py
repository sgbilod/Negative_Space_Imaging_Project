#!/usr/bin/env python
"""Security Module (Track E Phase 1)

Provides hashing, digital signing (Ed25519), verification, and provenance
utilities for the Negative Space Imaging Project.

Prototype Notes:
 - Keys stored locally (NOT production-secure)
 - Ed25519 chosen for speed & modern security properties
 - Hash algorithm default SHA-256
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import hashlib
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey
)
from cryptography.hazmat.primitives import serialization


KEY_DIR_DEFAULT = "keys"


class SecurityError(Exception):
    """Base security exception."""


@dataclass
class SignedArtifact:
    name: str
    path: str
    hash_algorithm: str
    artifact_hash: str
    signature_b64: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_ed25519_keypair() -> Tuple[
    Ed25519PrivateKey, Ed25519PublicKey, str
]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    # key_id: first 16 hex of public key bytes hash
    pub_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    key_id = hashlib.sha256(pub_bytes).hexdigest()[:16]
    return private_key, public_key, key_id


def load_or_create_keys(
    key_dir: str = KEY_DIR_DEFAULT
) -> Tuple[Ed25519PrivateKey, Ed25519PublicKey, str]:
    os.makedirs(key_dir, exist_ok=True)
    priv_path = Path(key_dir) / "ed25519_private.key"
    pub_path = Path(key_dir) / "ed25519_public.key"

    if priv_path.exists() and pub_path.exists():
        with open(priv_path, "rb") as f:
            private_key = Ed25519PrivateKey.from_private_bytes(f.read())
        with open(pub_path, "rb") as f:
            public_key = Ed25519PublicKey.from_public_bytes(f.read())
        pub_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        key_id = hashlib.sha256(pub_bytes).hexdigest()[:16]
        return private_key, public_key, key_id

    private_key, public_key, key_id = generate_ed25519_keypair()
    with open(priv_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        ))
    with open(pub_path, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        ))
    return private_key, public_key, key_id


def compute_hash(file_path: str, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sign_bytes(private_key: Ed25519PrivateKey, data: bytes) -> bytes:
    return private_key.sign(data)


def verify_signature(
    public_key: Ed25519PublicKey,
    data: bytes,
    signature: bytes
) -> bool:
    try:
        public_key.verify(signature, data)
        return True
    except Exception:  # noqa: BLE001
        return False


def sign_file(
    private_key: Ed25519PrivateKey,
    file_path: str,
    algorithm: str = "sha256"
) -> Tuple[str, str]:
    file_hash = compute_hash(file_path, algorithm)
    signature = sign_bytes(private_key, bytes.fromhex(file_hash))
    return file_hash, base64.b64encode(signature).decode("utf-8")


def verify_file(
    public_key: Ed25519PublicKey,
    file_path: str,
    expected_hash: str,
    signature_b64: str
) -> bool:
    current_hash = compute_hash(file_path)
    if current_hash != expected_hash:
        return False
    try:
        signature = base64.b64decode(signature_b64)
    except Exception:
        return False
    return verify_signature(
        public_key,
        bytes.fromhex(expected_hash),
        signature
    )


def build_provenance_entry(
    stage: str,
    artifact_path: str,
    parent_hash: Optional[str] = None
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "artifact": artifact_path,
        "timestamp": _timestamp(),
        "parent_hash": parent_hash,
    }


def sign_artifact(
    private_key: Ed25519PrivateKey,
    name: str,
    path: str,
    algorithm: str = "sha256"
) -> SignedArtifact:
    h, sig_b64 = sign_file(private_key, path, algorithm)
    return SignedArtifact(
        name=name,
        path=path,
        hash_algorithm=algorithm,
        artifact_hash=h,
        signature_b64=sig_b64,
        timestamp=_timestamp()
    )


def export_security_manifest(
    artifacts: List[SignedArtifact],
    key_id: str,
    output_dir: str
) -> str:
    manifest = {
        "key_id": key_id,
        "generated_at": _timestamp(),
        "artifacts": [a.to_dict() for a in artifacts]
    }
    path = Path(output_dir) / "security_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return str(path)


__all__ = [
    "SecurityError",
    "SignedArtifact",
    "load_or_create_keys",
    "compute_hash",
    "sign_file",
    "verify_file",
    "sign_artifact",
    "export_security_manifest",
    "build_provenance_entry",
]
