#!/usr/bin/env python
"""
Secure Imaging Workflow for Negative Space Imaging

Copyright (c) 2025 Stephen Bilodeau
All rights reserved.

This module demonstrates a complete secure workflow that integrates:
1. Image acquisition from demo_acquisition.py
2. Negative space processing (simulated)
3. Multi-signature verification from multi_signature_demo.py

This creates an end-to-end secure pipeline that ensures:
- Image data integrity through cryptographic hashing
- Processing authenticity via multi-signature verification
- HIPAA compliance with proper role-based authorizations
- Audit trail for all operations

Usage:
  python secure_imaging_workflow.py --mode threshold --signatures 5 --threshold 3
  python secure_imaging_workflow.py --mode sequential --signatures 3
  python secure_imaging_workflow.py --mode role-based
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

# Import from our own modules
from multi_signature_demo import (
    SignatureMode,
    SignatureStatus,
    Role,
    Signer,
    SignatureRequest,
    generate_sample_data
)

try:
    from demo_acquisition import acquire_image, get_acquisition_metadata
    from image_acquisition import ImageFormat, AcquisitionMode
    REAL_ACQUISITION = True
except ImportError:
    print("Warning: demo_acquisition.py or image_acquisition.py not found or has errors.")
    print("Will use simulated acquisition instead.")
    REAL_ACQUISITION = False


class SecureImagingWorkflow:
    """
    Orchestrates a secure imaging workflow with multi-signature verification.
    """

    def __init__(
        self,
        signature_mode: SignatureMode,
        num_signers: int = 5,
        threshold: int = 3,
        roles: Optional[List[str]] = None,
    ):
        self.signature_mode = signature_mode
        self.num_signers = num_signers
        self.threshold = threshold
        self.roles = self._setup_roles(roles) if roles else None

        # Set up signers with their keys
        self.signers, self.private_keys = self._setup_signers()

        # Tracking data
        self.workflow_id = os.urandom(8).hex()
        self.audit_log = []
        self.start_time = time.time()

    def _setup_roles(self, role_names: List[str]) -> List[Role]:
        """Set up roles for role-based signatures."""
        roles = []
        for i, name in enumerate(role_names):
            roles.append(
                Role(
                    id=name.lower(),
                    name=name.capitalize(),
                    description=f"{name.capitalize()} role",
                    priority=i+1,
                    required=True
                )
            )
        return roles

    def _setup_signers(self) -> Tuple[List[Signer], Dict[str, any]]:
        """Set up signers with their key pairs."""
        signers = []
        private_keys = {}

        if self.signature_mode == SignatureMode.ROLE_BASED and self.roles:
            # Create signers for each role
            for role in self.roles:
                for i in range(2):  # 2 signers per role
                    signer_id = f"{role.id}_{i+1}"
                    signer, private_key = Signer.generate(
                        signer_id,
                        f"{role.name} {i+1}",
                        role
                    )
                    signers.append(signer)
                    private_keys[signer_id] = private_key
        else:
            # Create regular signers
            for i in range(self.num_signers):
                signer_id = f"signer_{i+1}"
                if self.signature_mode == SignatureMode.SEQUENTIAL:
                    name = f"{'Analyst' if i==0 else 'Supervisor' if i==1 else 'Compliance Officer' if i==2 else f'Signer {i+1}'}"
                else:
                    name = f"Signer {i+1}"

                signer, private_key = Signer.generate(signer_id, name)
                signers.append(signer)
                private_keys[signer_id] = private_key

        return signers, private_keys

    def _log_event(self, event_type: str, details: Dict) -> None:
        """Add an event to the audit log."""
        self.audit_log.append({
            "timestamp": time.time(),
            "event_type": event_type,
            "workflow_id": self.workflow_id,
            "details": details
        })

    def simulate_acquisition(self) -> Tuple[Dict, bytes]:
        """
        Simulate image acquisition if the real modules aren't available.

        Returns:
            Tuple[Dict, bytes]: Metadata dictionary and image data as bytes
        """
        print("Simulating image acquisition...")

        # Create a simple test image (grayscale square with a circle)
        size = 512
        image = np.ones((size, size)) * 200  # Light gray background

        # Add a dark circle in the middle
        y, x = np.ogrid[:size, :size]
        center = size // 2
        mask = (x - center)**2 + (y - center)**2 <= (size//4)**2
        image[mask] = 50  # Dark circle

        # Add some noise
        image += np.random.normal(0, 10, (size, size))
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Generate a unique image ID
        image_id = os.urandom(8).hex()

        # Convert image to bytes
        image_bytes = image.tobytes()

        # Create metadata
        metadata = {
            "image_id": image_id,
            "timestamp": time.time(),
            "source": "simulation",
            "width": size,
            "height": size,
            "size_bytes": size * size,
            "negative_space_regions": 3,
            "hash": hashlib.sha256(image_bytes).hexdigest(),
            "format": "RAW",
            "bit_depth": 8,
            "acquisition_params": {
                "exposure": 100,
                "gain": 1.0,
                "binning": 1
            }
        }

        return metadata, image_bytes

    def acquire_image(self) -> Tuple[Dict, bytes]:
        """Acquire an image using the real acquisition module or simulation."""
        if REAL_ACQUISITION:
            try:
                # Use the real acquisition code
                image_data = acquire_image(
                    exposure=100,
                    gain=1.0,
                    binning=1,
                    format=ImageFormat.RAW,
                    mode=AcquisitionMode.NORMAL
                )
                metadata = get_acquisition_metadata()

                # Combine image data and metadata
                return metadata, image_data
            except Exception as e:
                print(f"Error during real acquisition: {e}")
                print("Falling back to simulated acquisition...")
                return self.simulate_acquisition()
        else:
            return self.simulate_acquisition()

    def process_image(self, image_data: bytes, metadata: Dict) -> Dict:
        """
        Process the image to detect negative space (simulated).
        In a real implementation, this would call into the core negative space
        detection algorithms.
        """
        print("Processing image for negative space detection...")

        # Here we would normally process the actual image data
        # For this demo, we'll create a simulated result

        # Calculate a hash of the image data for integrity
        image_hash = hashlib.sha256(image_data).hexdigest()

        # Simulated negative space detection results
        processing_result = {
            "timestamp": time.time(),
            "processing_id": os.urandom(8).hex(),
            "image_hash": image_hash,
            "image_metadata": metadata,
            "negative_space_analysis": {
                "detected_regions": [
                    {"x": 120, "y": 145, "width": 50, "height": 30, "confidence": 0.92},
                    {"x": 210, "y": 280, "width": 25, "height": 40, "confidence": 0.87},
                    {"x": 315, "y": 210, "width": 35, "height": 22, "confidence": 0.79},
                ],
                "algorithm_version": "2.3.0",
                "processing_time_ms": 238,
            }
        }

        self._log_event("image_processing", {
            "processing_id": processing_result["processing_id"],
            "image_hash": image_hash,
            "algorithm_version": processing_result["negative_space_analysis"]["algorithm_version"]
        })

        return processing_result

    def verify_processing_result(self, processing_result: Dict) -> bool:
        """
        Verify the processing result using multi-signature verification.
        """
        print(f"\nInitiating multi-signature verification ({self.signature_mode.value})...")

        # Convert processing result to bytes for signing
        result_bytes = json.dumps(processing_result, indent=2, sort_keys=True).encode()

        # Create signature request
        request = SignatureRequest(
            data=result_bytes,
            mode=self.signature_mode,
            signers=self.signers,
            threshold=self.threshold if self.signature_mode == SignatureMode.THRESHOLD else None,
            roles=self.roles if self.signature_mode == SignatureMode.ROLE_BASED else None,
        )

        print(f"Created signature request {request.id} for processing result...")
        self._log_event("signature_request_created", {
            "request_id": request.id,
            "mode": self.signature_mode.value,
            "num_signers": len(self.signers),
            "threshold": self.threshold if self.signature_mode == SignatureMode.THRESHOLD else None,
        })

        # Perform signatures based on the mode
        if self.signature_mode == SignatureMode.THRESHOLD:
            return self._perform_threshold_signatures(request, result_bytes)
        elif self.signature_mode == SignatureMode.SEQUENTIAL:
            return self._perform_sequential_signatures(request, result_bytes)
        elif self.signature_mode == SignatureMode.ROLE_BASED:
            return self._perform_role_based_signatures(request, result_bytes)

        return False

    def _perform_threshold_signatures(self, request, data_bytes) -> bool:
        """Perform threshold signature verification."""
        print(f"Performing threshold signature verification (requires {self.threshold} of {self.num_signers} signatures)...")

        # Sign with enough signers to meet the threshold
        for i in range(self.threshold):
            signer_id = f"signer_{i+1}"
            private_key = self.private_keys[signer_id]

            # Sign the data hash
            signature = private_key.sign(
                request.data_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            print(f"  Signer {i+1} is signing the request...")
            success = request.sign(signer_id, signature)

            if success:
                print(f"  ✓ Signature from {signer_id} accepted")
                self._log_event("signature_added", {
                    "request_id": request.id,
                    "signer_id": signer_id,
                    "success": True
                })
            else:
                print(f"  ✗ Signature from {signer_id} rejected")
                self._log_event("signature_added", {
                    "request_id": request.id,
                    "signer_id": signer_id,
                    "success": False
                })

        print(f"\nSignature verification status: {request.status.value}")
        if request.is_complete:
            print("✓ Signature requirements met - processing result verified!")
            return True
        else:
            print("✗ Signature requirements not met - processing result rejected!")
            return False

    def _perform_sequential_signatures(self, request, data_bytes) -> bool:
        """Perform sequential signature verification."""
        print(f"Performing sequential signature verification (requires {self.num_signers} signatures in sequence)...")

        # Sign in correct sequence
        for i in range(self.num_signers):
            signer_id = f"signer_{i+1}"
            private_key = self.private_keys[signer_id]

            # Sign the data hash
            signature = private_key.sign(
                request.data_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            print(f"  {request.signers[signer_id].name} is signing the request...")
            success = request.sign(signer_id, signature)

            if success:
                print(f"  ✓ Signature from {request.signers[signer_id].name} accepted")
                self._log_event("signature_added", {
                    "request_id": request.id,
                    "signer_id": signer_id,
                    "success": True
                })
            else:
                print(f"  ✗ Signature from {request.signers[signer_id].name} rejected")
                self._log_event("signature_added", {
                    "request_id": request.id,
                    "signer_id": signer_id,
                    "success": False
                })
                return False

        print(f"\nSignature verification status: {request.status.value}")
        if request.is_complete:
            print("✓ Signature requirements met - processing result verified!")
            return True
        else:
            print("✗ Signature requirements not met - processing result rejected!")
            return False

    def _perform_role_based_signatures(self, request, data_bytes) -> bool:
        """Perform role-based signature verification."""
        print(f"Performing role-based signature verification...")
        print(f"Required roles: {', '.join(role.name for role in self.roles if role.required)}")

        # Sign with one person from each required role
        for role in self.roles:
            if role.required:
                signer_id = f"{role.id}_1"  # Choose the first signer from this role
                private_key = self.private_keys[signer_id]

                # Sign the data hash
                signature = private_key.sign(
                    request.data_hash,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    hashes.SHA256(),
                )

                print(f"  {request.signers[signer_id].name} ({role.name}) is signing the request...")
                success = request.sign(signer_id, signature, role.id)

                if success:
                    print(f"  ✓ Signature from {signer_id} ({role.name}) accepted")
                    self._log_event("signature_added", {
                        "request_id": request.id,
                        "signer_id": signer_id,
                        "role_id": role.id,
                        "success": True
                    })
                else:
                    print(f"  ✗ Signature from {signer_id} ({role.name}) rejected")
                    self._log_event("signature_added", {
                        "request_id": request.id,
                        "signer_id": signer_id,
                        "role_id": role.id,
                        "success": False
                    })
                    return False

        print(f"\nSignature verification status: {request.status.value}")
        if request.is_complete:
            print("✓ Signature requirements met - processing result verified!")
            return True
        else:
            print("✗ Signature requirements not met - processing result rejected!")
            return False

    def save_audit_log(self, filename="security_audit.json"):
        """Save the audit log to a file."""
        with open(filename, "w") as f:
            json.dump({
                "workflow_id": self.workflow_id,
                "start_time": self.start_time,
                "end_time": time.time(),
                "signature_mode": self.signature_mode.value,
                "events": self.audit_log
            }, f, indent=2)
        print(f"Audit log saved to {filename}")

    def run(self) -> bool:
        """Run the complete secure imaging workflow."""
        try:
            print("="*80)
            print("🔒 Secure Negative Space Imaging Workflow 🔒")
            print("="*80)

            # 1. Acquire image
            print("\n[1/3] Acquiring image...")
            metadata, image_data = self.acquire_image()
            image_hash = hashlib.sha256(image_data).hexdigest()[:16]
            print(f"Image acquired: {len(image_data)} bytes, hash: {image_hash}...")
            self._log_event("image_acquired", {
                "image_size": len(image_data),
                "image_hash": image_hash,
                "metadata": metadata
            })

            # 2. Process image
            print("\n[2/3] Processing image...")
            processing_result = self.process_image(image_data, metadata)
            print(f"Image processed: {len(processing_result['negative_space_analysis']['detected_regions'])} negative space regions detected")

            # 3. Verify processing result with multi-signature
            print("\n[3/3] Verifying processing result...")
            verification_success = self.verify_processing_result(processing_result)
            self._log_event("verification_completed", {
                "success": verification_success,
                "processing_id": processing_result["processing_id"]
            })

            # 4. Save audit log
            self.save_audit_log()

            print("\n" + "="*80)
            if verification_success:
                print("✅ Secure workflow completed successfully!")
                print("All steps executed with proper authorization and verification.")
            else:
                print("❌ Secure workflow failed!")
                print("Verification requirements were not met.")
            print("="*80)

            return verification_success

        except Exception as e:
            print(f"Error in secure workflow: {str(e)}")
            import traceback
            traceback.print_exc()
            self._log_event("workflow_error", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            self.save_audit_log("security_audit_error.json")
            return False


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Secure Imaging Workflow")
    parser.add_argument(
        "--mode",
        choices=["threshold", "sequential", "role-based"],
        default="threshold",
        help="Signature mode to use",
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
    """Main entry point for the secure workflow."""
    args = parse_arguments()

    # Set up signature mode
    if args.mode == "threshold":
        signature_mode = SignatureMode.THRESHOLD
        workflow = SecureImagingWorkflow(
            signature_mode=signature_mode,
            num_signers=args.signatures,
            threshold=args.threshold
        )
    elif args.mode == "sequential":
        signature_mode = SignatureMode.SEQUENTIAL
        workflow = SecureImagingWorkflow(
            signature_mode=signature_mode,
            num_signers=args.signatures
        )
    elif args.mode == "role-based":
        signature_mode = SignatureMode.ROLE_BASED
        roles = args.roles.split(",")
        workflow = SecureImagingWorkflow(
            signature_mode=signature_mode,
            roles=roles
        )

    # Run the workflow
    workflow.run()


if __name__ == "__main__":
    # Make sure we have the cryptography import
    try:
        from cryptography.hazmat.primitives.asymmetric import padding
        main()
    except ImportError:
        print("Error: cryptography package is required for this script.")
        print("Please install it with: pip install cryptography")
        sys.exit(1)
