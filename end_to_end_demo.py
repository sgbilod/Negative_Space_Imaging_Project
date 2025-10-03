#!/usr/bin/env python
"""
Negative Space Imaging - End-to-End Demo
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This script demonstrates the complete Negative Space Imaging pipeline from
acquisition to secure processing and verification. It uses:
1. demo_acquisition.py for image acquisition and preprocessing
2. secure_imaging_workflow.py for cryptographic verification

The demo uses Hoag's object, which is a perfect example of negative space
in astronomical imagery.
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Import from demo_acquisition
from demo_acquisition import ImageAcquisition


def acquire_and_process_image(image_path, output_dir):
    """
    Acquire and process an image using the demo_acquisition module.

    Args:
        image_path: Path to the input image
        output_dir: Directory to save processed outputs

    Returns:
        Tuple of (processed_image_path, metadata_path)
    """
    print(f"Processing image: {image_path}")

    # Initialize acquisition with the file source
    acquisition = ImageAcquisition(
        source_type="file",
        source_path=image_path,
        output_dir=output_dir
    )

    # Acquire the image
    _ = acquisition.acquire()

    # Preprocess the image
    _ = acquisition.preprocess(
        resize=True,
        enhance=True,
        denoise=True
    )

    # Save the processed image
    output_path = acquisition.save(format="jpg")

    # Get metadata path
    metadata_path = os.path.join(
        output_dir, f'{acquisition.image_id}_metadata.json'
    )

    print(f"Image acquisition complete")
    print(f"Processed image: {output_path}")
    print(f"Metadata: {metadata_path}")

    return output_path, metadata_path


def secure_verify_image(image_path, num_signatures=5, threshold=3):
    """
    Apply secure verification to the processed image using
    the secure_imaging_workflow module.

    Args:
        image_path: Path to the processed image
        num_signatures: Number of signatures to generate
        threshold: Threshold for signature verification

    Returns:
        Result of the verification (True/False)
    """
    print(f"\nApplying secure verification to: {image_path}")
    print(f"Using threshold signatures (k={threshold} of n={num_signatures})")

    # Call the secure workflow as a subprocess
    cmd = [
        "python", "secure_imaging_workflow.py",
        "--mode", "threshold",
        "--signatures", str(num_signatures),
        "--threshold", str(threshold),
        "--image", image_path
    ]

    print(f"Running: {' '.join(cmd)}")

    # Run the secure verification
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        print("\nSecure verification output:")
        print(result.stdout)

        # Check if verification was successful
        if "Verification successful" in result.stdout:
            return True
        else:
            print(f"Verification failed: {result.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Error during secure verification: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main function to run the end-to-end demo."""
    parser = argparse.ArgumentParser(
        description="Negative Space Imaging - End-to-End Demo"
    )
    parser.add_argument(
        "--image",
        default="Hoag's_object.jpg",
        help="Path to input image (default: Hoag's_object.jpg)"
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--signatures",
        type=int,
        default=5,
        help="Number of signatures for verification (default: 5)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Threshold for signature verification (default: 3)"
    )

    args = parser.parse_args()

    # Ensure image path is absolute
    if not os.path.isabs(args.image):
        image_path = os.path.join(os.path.dirname(__file__), args.image)
    else:
        image_path = args.image

    # Ensure output directory exists
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("NEGATIVE SPACE IMAGING - END-TO-END DEMO")
    print("=" * 80)
    print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input Image: {image_path}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    try:
        # Step 1: Acquire and process the image
        print("\nSTEP 1: IMAGE ACQUISITION AND PROCESSING")
        print("-" * 50)
        output_path, metadata_path = acquire_and_process_image(
            image_path, output_dir
        )

        # Step 2: Apply secure verification
        print("\nSTEP 2: SECURE VERIFICATION")
        print("-" * 50)
        verification_result = secure_verify_image(
            output_path,
            num_signatures=args.signatures,
            threshold=args.threshold
        )

        # Step 3: Display final results
        print("\nSTEP 3: FINAL RESULTS")
        print("-" * 50)

        if verification_result:
            print("✅ The image has been successfully processed and verified")
            print("✅ The integrity and authenticity of the image is confirmed")
        else:
            print("❌ Image verification failed")
            print("❌ The image may have been tampered with or corrupted")

        # Load and display metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print("\nIMAGE METADATA:")
        print(f"  - ID: {metadata.get('image_id', 'Unknown')}")
        print(f"  - Original: {metadata.get('original_filename', 'Unknown')}")

        # Show dimensions if available
        dimensions = metadata.get('original_dimensions', 'Unknown')
        print(f"  - Dimensions: {dimensions}")

        # Show processing info
        preprocessing = metadata.get('preprocessing', {})
        print(f"  - Processing: Enhanced={preprocessing.get('enhanced', False)}, "
              f"Denoised={preprocessing.get('denoised', False)}")

        print("\nDemo completed successfully!")
        return 0

    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
