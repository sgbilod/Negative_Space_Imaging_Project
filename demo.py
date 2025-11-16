import sys
import argparse
import json
from pathlib import Path


def run_demo(image_path: str) -> None:
    try:
        from src.processing.negative_space_detection import (
            detect_negative_space,
        )
        from src.utils.image_loader import load_image
    except ImportError:
        # Fallback for direct execution in root
        from processing.negative_space_detection import detect_negative_space
        from utils.image_loader import load_image

    image = load_image(image_path)
    result = detect_negative_space(image)
    print(f"Negative space detection result: {result}")


def verify_security_end_to_end(width: int = 64, height: int = 64) -> int:
    """Acquire -> reconstruct -> verify all signatures, print summary."""
    import hashlib
    import base64
    import numpy as np
    from PIL import Image

    from acquisition_service import acquire_sync
    from advanced_reconstructor import AdvancedReconstructor
    from security_module import load_or_create_keys, verify_file

    # Acquire simulated image
    acq = acquire_sync('simulation://test_pattern', width=width, height=height)
    if not acq.success:
        print(f"[acq] Acquisition failed: {acq.error}")
        return 1

    # Verify acquisition signature
    sec = acq.metadata.security or {}
    algo = sec.get('hash_algorithm', 'sha256').lower()
    expected_hash = sec.get('artifact_hash', '')
    signature_b64 = sec.get('signature', '')

    _, public_key, key_id = load_or_create_keys()
    print(f"[keys] key_id={key_id}")

    h = hashlib.new(algo)
    h.update(acq.data)
    actual = h.hexdigest()
    if actual != expected_hash:
        print(f"[acq] hash mismatch actual={actual} expected={expected_hash}")
        return 2
    try:
        sig = base64.b64decode(signature_b64)
        public_key.verify(sig, bytes.fromhex(expected_hash))
        print("[acq] signature valid: True")
    except Exception as e:
        print(f"[acq] signature valid: False ({e})")
        return 3

    # Save to PNG for reconstruction
    out_dir = Path('demo_security_outputs')
    out_dir.mkdir(exist_ok=True)
    img_path = out_dir / 'acq_image.png'
    arr = np.frombuffer(acq.data, dtype=np.uint8).reshape((height, width))
    Image.fromarray(arr, mode='L').save(img_path)

    # Reconstruct with artifact signing
    recon = AdvancedReconstructor()
    recon_out = out_dir / 'recon_outputs'
    result = recon.reconstruct(str(img_path), str(recon_out))

    # Verify artifact signatures
    artifacts = (result.provenance.security or {}).get('artifacts', [])
    if not artifacts:
        print("[recon] no artifacts to verify")
        return 4
    all_ok = True
    for art in artifacts:
        ok = verify_file(
            public_key,
            art.get('path', ''),
            art.get('artifact_hash', ''),
            art.get('signature_b64', ''),
        )
        print(f"[recon] {art.get('name', '?')}: valid={ok}")
        all_ok = all_ok and ok

    print(json.dumps({
        'acquisition_ok': True,
        'reconstruction_ok': all_ok
    }, indent=2))
    return 0 if all_ok else 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NSIP demo and security verifier"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to image for detection"
    )
    parser.add_argument(
        "--verify-security",
        action="store_true",
        help="Run acquisition+reconstruction signature verification"
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    args = parser.parse_args()

    if args.verify_security:
        code = verify_security_end_to_end(width=args.width, height=args.height)
        sys.exit(code)

    if not args.image_path:
        print("Usage: python demo.py <image_path> [--verify-security]")
        sys.exit(1)

    run_demo(args.image_path)
    # ...expand with visualization and multiple algorithms
