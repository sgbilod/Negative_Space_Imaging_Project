#!/usr/bin/env python
"""
Tiny helper to verify acquisition and reconstruction signatures
without quoting issues.

Usage:
    pwsh> python scripts/check_security.py

Options:
    --source URI           Source URI (default: simulation://test_pattern)
    --width N              Width for simulation (default: 64)
    --height N             Height for simulation (default: 64)
    --output-dir PATH      Output dir (default: security_check_outputs)
    --skip-recon           Only verify acquisition, skip reconstruction
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path
import sys as _sys

# Ensure project root on sys.path for direct execution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from acquisition_service import acquire_sync  # noqa: E402
from advanced_reconstructor import AdvancedReconstructor  # noqa: E402
from security_module import load_or_create_keys, verify_file  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import (  # noqa: E402
    ed25519 as _ed25519,
)


def verify_acquisition_signature(
    data: bytes,
    security: dict,
    public_key: _ed25519.Ed25519PublicKey
) -> bool:
    try:
        algo = security.get("hash_algorithm", "sha256").lower()
        expected_hash = security.get("artifact_hash", "")
        signature_b64 = security.get("signature", "")
        if not expected_hash or not signature_b64:
            print("[acq] Missing expected_hash or signature in security block")
            return False

        actual_hash = hashlib.new(algo)
        actual_hash.update(data)
        actual_hex = actual_hash.hexdigest()
        if actual_hex != expected_hash:
            print(
                f"[acq] Hash mismatch: actual={actual_hex} "
                f"expected={expected_hash}"
            )
            return False

        # Verify signature over the hash bytes
        signature = base64.b64decode(signature_b64)
        try:
            public_key.verify(signature, bytes.fromhex(expected_hash))
            return True
        except Exception as e:
            print(f"[acq] Signature verification failed: {e}")
            return False
    except Exception as e:  # noqa: BLE001
        print(f"[acq] Verification error: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check security signatures end-to-end"
    )
    parser.add_argument("--source", default="simulation://test_pattern")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--output-dir", default="security_check_outputs")
    parser.add_argument("--skip-recon", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Acquire
    result = acquire_sync(args.source, width=args.width, height=args.height)
    if not result.success:
        print(f"[acq] Acquisition failed: {result.error}")
        return 2

    print("[acq] Security block:")
    print(json.dumps(result.metadata.security, indent=2))

    # Load public key
    _, public_key, key_id = load_or_create_keys()
    print(f"[keys] Using key_id: {key_id}")

    # Verify acquisition signature
    ok_acq = verify_acquisition_signature(
        result.data,
        result.metadata.security,
        public_key
    )
    print(f"[acq] Signature valid: {ok_acq}")

    ok_recon = True
    if not args.skip_recon:
        # Save acquired bytes as a grayscale PNG for reconstructor
        try:
            arr = np.frombuffer(result.data, dtype=np.uint8).reshape(
                (args.height, args.width)
            )
            img_path = out_dir / "acq_image.png"
            Image.fromarray(arr, mode="L").save(img_path)
        except Exception as e:
            print(f"[recon] Failed to save acquired bytes as image: {e}")
            return 3

        # Run reconstruction; artifacts are signed inside AdvancedReconstructor
        recon = AdvancedReconstructor()
        recon_out_dir = out_dir / "recon_outputs"
        recon_result = recon.reconstruct(str(img_path), str(recon_out_dir))

        # Verify reconstruction artifact signatures
        sec = (recon_result.provenance.security or {})
        artifacts = sec.get("artifacts", [])
        if not artifacts:
            print("[recon] No artifact signatures found to verify")
            ok_recon = False
        else:
            print("[recon] Artifact signatures:")
            print(json.dumps(artifacts, indent=2))
            for art in artifacts:
                art_ok = verify_file(
                    public_key,
                    art.get("path", ""),
                    art.get("artifact_hash", ""),
                    art.get("signature_b64", ""),
                )
                print(f"[recon] {art.get('name', '?')}: valid={art_ok}")
                ok_recon = ok_recon and art_ok

    all_ok = ok_acq and ok_recon
    print(
        f"[summary] acquisition_ok={ok_acq} "
        f"reconstruction_ok={ok_recon} all_ok={all_ok}"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
