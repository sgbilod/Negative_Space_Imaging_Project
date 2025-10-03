#!/usr/bin/env python
"""
Multi-Signature Verification System for Negative Space Imaging
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements a multi-signature verification system to ensure the integrity
and authenticity of image processing results. Multiple authorized parties must sign off
on processing results before they are accepted as valid, enhancing security and compliance
with HIPAA regulations.

Modes:
- threshold: Requires m-of-n signatures (e.g., 3 of 5 authorized signers)
- sequential: Requires signatures in a specific sequence (e.g., analyst → supervisor → compliance officer)
- role-based: Requires signatures from specific roles (e.g., one from each: analyst, doctor, administrator)

Usage:
  python multi_signature_demo.py --mode threshold --signatures 5 --threshold 3
  python multi_signature_demo.py --mode sequential --signatures 3
  python multi_signature_demo.py --mode role-based --roles analyst,physician,admin
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

# Constants
KEY_SIZE = 2048
HASH_ALGORITHM = hashes.SHA256()
SIGNATURE_TIMEOUT = 3600  # 1 hour


class SignatureMode(Enum):
    THRESHOLD = "threshold"
    SEQUENTIAL = "sequential"
    ROLE_BASED = "role-based"


class SignatureStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    COMPLETE = "complete"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Role:
    id: str
    name: str
    description: str
    priority: int
    required: bool


@dataclass
class Signer:
    id: str
    name: str
    public_key: rsa.RSAPublicKey
    role: Optional[Role] = None

    @classmethod
    def generate(cls, id: str, name: str, role: Optional[Role] = None) -> Tuple["Signer", rsa.RSAPrivateKey]:
        """Generate a new signer with a key pair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=KEY_SIZE, backend=default_backend()
        )
        public_key = private_key.public_key()
        return cls(id=id, name=name, public_key=public_key, role=role), private_key


@dataclass
class Signature:
    signer_id: str
    timestamp: float
    signature: bytes
    role_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "signer_id": self.signer_id,
            "timestamp": self.timestamp,
            "signature": base64.b64encode(self.signature).decode("utf-8"),
            "role_id": self.role_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Signature":
        return cls(
            signer_id=data["signer_id"],
            timestamp=data["timestamp"],
            signature=base64.b64decode(data["signature"]),
            role_id=data.get("role_id"),
        )


class SignatureRequest:
    def __init__(
        self,
        data: bytes,
        mode: SignatureMode,
        signers: List[Signer],
        threshold: Optional[int] = None,
        roles: Optional[List[Role]] = None,
    ):
        self.id = hashlib.sha256(data + str(time.time()).encode()).hexdigest()[:16]
        self.data = data
        self.data_hash = hashlib.sha256(data).digest()
        self.mode = mode
        self.signers = {signer.id: signer for signer in signers}
        self.threshold = threshold
        self.roles = {role.id: role for role in roles} if roles else None
        self.signatures: Dict[str, Signature] = {}
        self.created_at = time.time()
        self.status = SignatureStatus.PENDING

    def sign(self, signer_id: str, signature: bytes, role_id: Optional[str] = None) -> bool:
        """Add a signature to the request."""
        if signer_id not in self.signers:
            print(f"Error: Unknown signer ID: {signer_id}")
            return False

        if signer_id in self.signatures:
            print(f"Error: Signer {signer_id} has already signed")
            return False

        signer = self.signers[signer_id]

        # Verify signature
        try:
            signer.public_key.verify(
                signature,
                self.data_hash,
                padding.PSS(
                    mgf=padding.MGF1(HASH_ALGORITHM),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                HASH_ALGORITHM,
            )
        except Exception as e:
            print(f"Error: Invalid signature from {signer_id}: {e}")
            return False

        # Check role if in role-based mode
        if self.mode == SignatureMode.ROLE_BASED and role_id:
            if role_id not in self.roles:
                print(f"Error: Unknown role ID: {role_id}")
                return False
            if signer.role and signer.role.id != role_id:
                print(f"Error: Signer {signer_id} does not have role {role_id}")
                return False

        # Check sequential order if in sequential mode
        if self.mode == SignatureMode.SEQUENTIAL:
            expected_signer_index = len(self.signatures)
            actual_signer_index = list(self.signers.keys()).index(signer_id)
            if actual_signer_index != expected_signer_index:
                print(f"Error: Out of sequence signing. Expected signer index {expected_signer_index}, got {actual_signer_index}")
                return False

        # Add signature
        self.signatures[signer_id] = Signature(
            signer_id=signer_id,
            timestamp=time.time(),
            signature=signature,
            role_id=role_id,
        )

        # Update status
        self._update_status()
        return True

    def _update_status(self) -> None:
        """Update the status of the signature request based on collected signatures."""
        # Check if expired
        if time.time() - self.created_at > SIGNATURE_TIMEOUT:
            self.status = SignatureStatus.EXPIRED
            return

        if self.mode == SignatureMode.THRESHOLD:
            if len(self.signatures) >= self.threshold:
                self.status = SignatureStatus.COMPLETE
            elif len(self.signatures) > 0:
                self.status = SignatureStatus.PARTIAL
            else:
                self.status = SignatureStatus.PENDING

        elif self.mode == SignatureMode.SEQUENTIAL:
            if len(self.signatures) == len(self.signers):
                self.status = SignatureStatus.COMPLETE
            elif len(self.signatures) > 0:
                self.status = SignatureStatus.PARTIAL
            else:
                self.status = SignatureStatus.PENDING

        elif self.mode == SignatureMode.ROLE_BASED:
            signed_roles = set(sig.role_id for sig in self.signatures.values() if sig.role_id)
            required_roles = set(role.id for role in self.roles.values() if role.required)

            if required_roles.issubset(signed_roles):
                self.status = SignatureStatus.COMPLETE
            elif len(self.signatures) > 0:
                self.status = SignatureStatus.PARTIAL
            else:
                self.status = SignatureStatus.PENDING

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "data_hash": base64.b64encode(self.data_hash).decode("utf-8"),
            "mode": self.mode.value,
            "threshold": self.threshold,
            "created_at": self.created_at,
            "status": self.status.value,
            "signatures": {
                signer_id: signature.to_dict()
                for signer_id, signature in self.signatures.items()
            }
        }

    @property
    def is_complete(self) -> bool:
        return self.status == SignatureStatus.COMPLETE


def generate_sample_data() -> bytes:
    """Generate sample image processing data for demonstration."""
    # Simulate image processing results with some metadata
    result = {
        "timestamp": time.time(),
        "processing_id": os.urandom(8).hex(),
        "negative_space_analysis": {
            "detected_regions": [
                {"x": 120, "y": 145, "width": 50, "height": 30, "confidence": 0.92},
                {"x": 210, "y": 280, "width": 25, "height": 40, "confidence": 0.87},
            ],
            "algorithm_version": "2.3.0",
            "processing_time_ms": 238,
        },
        "image_metadata": {
            "width": 1024,
            "height": 768,
            "format": "DICOM",
            "acquisition_date": time.strftime("%Y-%m-%d"),
        },
    }
    return json.dumps(result, indent=2).encode()


def demonstrate_threshold_signatures(num_signers: int, threshold: int) -> None:
    """Demonstrate threshold signature mode (m-of-n)."""
    print(f"\n=== Demonstrating Threshold Signature ({threshold} of {num_signers}) ===\n")

    # Generate signers with key pairs
    signers = []
    private_keys = {}

    for i in range(num_signers):
        signer_id = f"signer_{i+1}"
        signer, private_key = Signer.generate(signer_id, f"Signer {i+1}")
        signers.append(signer)
        private_keys[signer_id] = private_key

    # Create sample data and signature request
    data = generate_sample_data()
    request = SignatureRequest(
        data=data,
        mode=SignatureMode.THRESHOLD,
        signers=signers,
        threshold=threshold,
    )

    print(f"Created signature request {request.id} for data hash {request.data_hash.hex()[:16]}...")
    print(f"Requires {threshold} of {num_signers} signatures")
    print(f"Status: {request.status.value}")

    # Sign with different signers
    for i in range(threshold):
        signer_id = f"signer_{i+1}"
        private_key = private_keys[signer_id]

        # Sign the data hash
        signature = private_key.sign(
            request.data_hash,
            padding.PSS(
                mgf=padding.MGF1(HASH_ALGORITHM),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            HASH_ALGORITHM,
        )

        print(f"\nSigner {i+1} is signing the request...")
        success = request.sign(signer_id, signature)

        if success:
            print(f"✓ Signature from {signer_id} accepted")
            print(f"Status: {request.status.value}")
        else:
            print(f"✗ Signature from {signer_id} rejected")

    # Final status
    print(f"\nFinal status: {request.status.value}")
    if request.is_complete:
        print("✓ Signature requirements met - processing authorized!")
    else:
        print("✗ Signature requirements not met - processing denied!")

    # Visualize the threshold signatures
    visualize_threshold_signatures(request, num_signers, threshold)


def demonstrate_sequential_signatures(num_signers: int) -> None:
    """Demonstrate sequential signature mode."""
    print(f"\n=== Demonstrating Sequential Signature (Chain of {num_signers}) ===\n")

    # Generate signers with key pairs
    signers = []
    private_keys = {}

    for i in range(num_signers):
        signer_id = f"signer_{i+1}"
        signer, private_key = Signer.generate(
            signer_id,
            f"{'Analyst' if i==0 else 'Supervisor' if i==1 else 'Compliance Officer' if i==2 else f'Signer {i+1}'}"
        )
        signers.append(signer)
        private_keys[signer_id] = private_key

    # Create sample data and signature request
    data = generate_sample_data()
    request = SignatureRequest(
        data=data,
        mode=SignatureMode.SEQUENTIAL,
        signers=signers,
    )

    print(f"Created signature request {request.id} for data hash {request.data_hash.hex()[:16]}...")
    print(f"Requires sequential signatures from {num_signers} signers")
    print(f"Status: {request.status.value}")

    # Attempt out-of-order signing to demonstrate sequential enforcement
    if num_signers >= 3:
        print("\nAttempting out-of-order signing (should fail)...")

        wrong_signer_id = f"signer_3"  # Try to sign with the third signer first
        wrong_private_key = private_keys[wrong_signer_id]

        wrong_signature = wrong_private_key.sign(
            request.data_hash,
            padding.PSS(
                mgf=padding.MGF1(HASH_ALGORITHM),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            HASH_ALGORITHM,
        )

        success = request.sign(wrong_signer_id, wrong_signature)

        if not success:
            print(f"✓ Correctly rejected out-of-order signature from {wrong_signer_id}")
        else:
            print(f"✗ Incorrectly accepted out-of-order signature from {wrong_signer_id}")

    # Sign in correct sequence
    print("\nSigning in correct sequence...")
    for i in range(num_signers):
        signer_id = f"signer_{i+1}"
        private_key = private_keys[signer_id]

        # Sign the data hash
        signature = private_key.sign(
            request.data_hash,
            padding.PSS(
                mgf=padding.MGF1(HASH_ALGORITHM),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            HASH_ALGORITHM,
        )

        print(f"\nSigner {i+1} is signing the request...")
        success = request.sign(signer_id, signature)

        if success:
            print(f"✓ Signature from {signer_id} accepted")
            print(f"Status: {request.status.value}")
        else:
            print(f"✗ Signature from {signer_id} rejected")

    # Final status
    print(f"\nFinal status: {request.status.value}")
    if request.is_complete:
        print("✓ Signature requirements met - processing authorized!")
    else:
        print("✗ Signature requirements not met - processing denied!")


def demonstrate_role_based_signatures() -> None:
    """Demonstrate role-based signature mode."""
    print(f"\n=== Demonstrating Role-Based Signature ===\n")

    # Define roles
    roles = [
        Role(id="analyst", name="Analyst", description="Image processing analyst", priority=1, required=True),
        Role(id="physician", name="Physician", description="Medical doctor", priority=2, required=True),
        Role(id="admin", name="Administrator", description="System administrator", priority=3, required=True),
        Role(id="auditor", name="Auditor", description="Compliance auditor", priority=4, required=False),
    ]

    # Generate signers with key pairs and roles
    signers = []
    private_keys = {}

    # Create multiple signers for each role to demonstrate flexibility
    for role in roles:
        for i in range(2):  # 2 signers per role
            signer_id = f"{role.id}_{i+1}"
            signer, private_key = Signer.generate(
                signer_id,
                f"{role.name} {i+1}",
                role
            )
            signers.append(signer)
            private_keys[signer_id] = private_key

    # Create sample data and signature request
    data = generate_sample_data()
    request = SignatureRequest(
        data=data,
        mode=SignatureMode.ROLE_BASED,
        signers=signers,
        roles=roles,
    )

    print(f"Created signature request {request.id} for data hash {request.data_hash.hex()[:16]}...")
    print(f"Requires signatures from roles: {', '.join(role.name for role in roles if role.required)}")
    print(f"Optional signatures from: {', '.join(role.name for role in roles if not role.required)}")
    print(f"Status: {request.status.value}")

    # Sign with one person from each required role
    signed_roles = set()

    for role in roles:
        if role.required and role.id not in signed_roles:
            signer_id = f"{role.id}_1"  # Choose the first signer from this role
            private_key = private_keys[signer_id]

            # Sign the data hash
            signature = private_key.sign(
                request.data_hash,
                padding.PSS(
                    mgf=padding.MGF1(HASH_ALGORITHM),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                HASH_ALGORITHM,
            )

            print(f"\n{request.signers[signer_id].name} ({role.name}) is signing the request...")
            success = request.sign(signer_id, signature, role.id)

            if success:
                print(f"✓ Signature from {signer_id} ({role.name}) accepted")
                print(f"Status: {request.status.value}")
                signed_roles.add(role.id)
            else:
                print(f"✗ Signature from {signer_id} ({role.name}) rejected")

    # Final status
    print(f"\nFinal status: {request.status.value}")
    if request.is_complete:
        print("✓ Signature requirements met - processing authorized!")
    else:
        print("✗ Signature requirements not met - processing denied!")


def visualize_threshold_signatures(request, num_signers, threshold):
    """Generate a visual representation of the threshold signature status."""
    plt.figure(figsize=(10, 6))

    # Create a grid of signer blocks
    for i in range(num_signers):
        signer_id = f"signer_{i+1}"
        signed = signer_id in request.signatures

        color = 'green' if signed else 'lightgray'
        plt.fill_between([i, i+0.8], [0, 0], [1, 1], color=color, alpha=0.5)
        plt.text(i+0.4, 0.5, f"Signer\n{i+1}", ha='center', va='center', fontsize=12)

    # Draw threshold line
    plt.axvline(x=threshold-0.1, color='red', linestyle='--', linewidth=2)
    plt.text(threshold-0.1, 1.1, f"Threshold ({threshold})", color='red', ha='center')

    # Labels and styling
    plt.xlim(-0.5, num_signers+0.5)
    plt.ylim(0, 1.3)
    plt.title(f"Threshold Signature Status: {request.status.value.upper()}")
    plt.xticks([])
    plt.yticks([])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.5, label='Signed'),
        Patch(facecolor='lightgray', alpha=0.5, label='Unsigned'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig("threshold_signature_visualization.png")
    print("\nVisualization saved to 'threshold_signature_visualization.png'")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Signature Verification System Demo")
    parser.add_argument(
        "--mode",
        choices=["threshold", "sequential", "role-based"],
        default="threshold",
        help="Signature mode to demonstrate",
    )
    parser.add_argument(
        "--signatures",
        type=int,
        default=5,
        help="Number of signers (for threshold and sequential modes)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Signature threshold (for threshold mode)",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default="analyst,physician,admin",
        help="Comma-separated list of roles (for role-based mode)",
    )
    return parser.parse_args()


def main():
    """Main entry point for the demonstration."""
    args = parse_arguments()

    print("🔐 Multi-Signature Verification System for Negative Space Imaging 🔐")
    print("===============================================================")

    if args.mode == "threshold":
        demonstrate_threshold_signatures(args.signatures, args.threshold)
    elif args.mode == "sequential":
        demonstrate_sequential_signatures(args.signatures)
    elif args.mode == "role-based":
        demonstrate_role_based_signatures()

    print("\nDemo completed successfully! 🎉")


if __name__ == "__main__":
    main()
