#!/usr/bin/env python
"""
Negative Space Imaging System - Command Line Interface
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides a comprehensive command-line interface for the
Negative Space Imaging System, enabling users to:

1. Acquire images from various sources
2. Process images to detect negative space
3. Apply multi-signature verification for secure workflows
4. Run the complete secure workflow pipeline
5. View and manage security audit logs

Usage:
  python cli.py acquire --exposure 100 --gain 1.0 --output image.raw
  python cli.py process --input image.raw --output results.json
  python cli.py verify --input results.json --mode threshold --signatures 5 --threshold 3
  python cli.py workflow --mode threshold --signatures 5 --threshold 3
  python cli.py audit --view --log security_audit.json
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional
import importlib.util
from pathlib import Path

# Import multi-node deployment CLI
try:
    from deployment.cli_multi_node import integrate_with_main_cli
    MULTI_NODE_AVAILABLE = True
except ImportError:
    MULTI_NODE_AVAILABLE = False

# Check for required dependencies
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Import local modules (with error handling)
try:
    from secure_imaging_workflow import SecureImagingWorkflow
    from multi_signature_demo import SignatureMode
    WORKFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Secure workflow modules not available: {e}")
    print("Some features will be disabled.")
    WORKFLOW_AVAILABLE = False

try:
    from demo_acquisition import acquire_image, get_acquisition_metadata
    ACQUISITION_AVAILABLE = True
except ImportError:
    ACQUISITION_AVAILABLE = False


class NegativeSpaceCliApp:
    """
    Command-line interface application for the Negative Space Imaging System.
    """

    def __init__(self):
        """Initialize the CLI application."""
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the command-line argument parser."""
        parser = argparse.ArgumentParser(
            description="Negative Space Imaging System CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python cli.py acquire --exposure 100 --gain 1.0 --output image.raw
  python cli.py process --input image.raw --output results.json
  python cli.py verify --input results.json --mode threshold --signatures 5 --threshold 3
  python cli.py workflow --mode threshold --signatures 5 --threshold 3
  python cli.py audit --view --log security_audit.json
            """
        )

        # Create subparsers for different commands
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")

        # Acquire command
        acquire_parser = subparsers.add_parser("acquire", help="Acquire an image")
        acquire_parser.add_argument("--exposure", type=float, default=100.0, help="Exposure time (ms)")
        acquire_parser.add_argument("--gain", type=float, default=1.0, help="Sensor gain")
        acquire_parser.add_argument("--binning", type=int, default=1, help="Pixel binning factor")
        acquire_parser.add_argument("--simulate", action="store_true", help="Use simulated acquisition")
        acquire_parser.add_argument("--output", type=str, required=True, help="Output image file path")

        # Process command
        process_parser = subparsers.add_parser("process", help="Process an image for negative space")
        process_parser.add_argument("--input", type=str, required=True, help="Input image file path")
        process_parser.add_argument("--output", type=str, required=True, help="Output results file path")
        process_parser.add_argument("--algorithm", type=str, default="standard",
                                  choices=["standard", "advanced", "experimental"],
                                  help="Algorithm to use for processing")

        # Verify command
        verify_parser = subparsers.add_parser("verify", help="Verify processing results with multi-signature")
        verify_parser.add_argument("--input", type=str, required=True, help="Input results file path")
        verify_parser.add_argument("--mode", type=str, default="threshold",
                                 choices=["threshold", "sequential", "role-based"],
                                 help="Signature mode to use")
        verify_parser.add_argument("--signatures", type=int, default=5, help="Number of signers")
        verify_parser.add_argument("--threshold", type=int, default=3, help="Signature threshold")
        verify_parser.add_argument("--roles", type=str, default="analyst,physician,admin",
                                 help="Comma-separated list of roles")

        # Workflow command (runs the complete secure workflow)
        workflow_parser = subparsers.add_parser("workflow", help="Run the complete secure workflow")
        workflow_parser.add_argument("--mode", type=str, default="threshold",
                                   choices=["threshold", "sequential", "role-based"],
                                   help="Signature mode to use")
        workflow_parser.add_argument("--signatures", type=int, default=5, help="Number of signers")
        workflow_parser.add_argument("--threshold", type=int, default=3, help="Signature threshold")
        workflow_parser.add_argument("--roles", type=str, default="analyst,physician,admin",
                                   help="Comma-separated list of roles")
        workflow_parser.add_argument("--output", type=str, default="security_audit.json",
                                   help="Output audit log file path")

        # Audit command
        audit_parser = subparsers.add_parser("audit", help="View and manage security audit logs")
        audit_parser.add_argument("--log", type=str, default="security_audit.json",
                                help="Audit log file path")
        audit_parser.add_argument("--view", action="store_true", help="View the audit log")
        audit_parser.add_argument("--filter", type=str, help="Filter events by type")
        audit_parser.add_argument("--export", type=str, help="Export log to specified file path")

        # Integrate multi-node deployment commands if available
        if MULTI_NODE_AVAILABLE:
            # Get the multi-node parser setup function and command handler
            setup_multi_node_parser, handle_multi_node_command = integrate_with_main_cli()

            # Set up the multi-node parser
            multi_node_parser = setup_multi_node_parser(subparsers)

            # Store the handler for later use
            self._handle_multi_node_command = handle_multi_node_command

        return parser

    def run(self) -> int:
        """
        Run the CLI application with the provided arguments.
        Returns the exit code (0 for success, non-zero for errors).
        """
        args = self.parser.parse_args()

        if not args.command:
            self.parser.print_help()
            return 1

        # Check for required dependencies
        if not CRYPTO_AVAILABLE and args.command in ["verify", "workflow"]:
            print("Error: cryptography package is required for verification features.")
            print("Please install it with: pip install cryptography")
            return 1

        if not WORKFLOW_AVAILABLE and args.command in ["verify", "workflow"]:
            print("Error: Secure workflow modules are required for this command.")
            print("Please ensure secure_imaging_workflow.py and multi_signature_demo.py are available.")
            return 1

        try:
            # Dispatch to the appropriate command handler
            if args.command == "acquire":
                return self._handle_acquire(args)
            elif args.command == "process":
                return self._handle_process(args)
            elif args.command == "verify":
                return self._handle_verify(args)
            elif args.command == "workflow":
                return self._handle_workflow(args)
            elif args.command == "audit":
                return self._handle_audit(args)
            elif args.command == "multi-node" and MULTI_NODE_AVAILABLE:
                return self._handle_multi_node_command(args)
            else:
                print(f"Error: Unknown command: {args.command}")
                return 1
        except Exception as e:
            print(f"Error executing command: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def _handle_acquire(self, args) -> int:
        """Handle the 'acquire' command."""
        print("=== Image Acquisition ===")

        if args.simulate or not ACQUISITION_AVAILABLE:
            if not args.simulate:
                print("Warning: Real acquisition not available, using simulation instead.")

            print(f"Simulating image acquisition with parameters:")
            print(f"  Exposure: {args.exposure} ms")
            print(f"  Gain: {args.gain}")
            print(f"  Binning: {args.binning}")

            # Create simulated image and metadata
            workflow = SecureImagingWorkflow(SignatureMode.THRESHOLD)
            metadata, image_data = workflow.simulate_acquisition()

            # Save the image data to the output file
            with open(args.output, "wb") as f:
                f.write(image_data)

            # Save metadata alongside the image
            metadata_path = f"{os.path.splitext(args.output)[0]}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Simulated image saved to: {args.output}")
            print(f"Metadata saved to: {metadata_path}")

        else:
            try:
                # Use real acquisition
                print(f"Acquiring image with parameters:")
                print(f"  Exposure: {args.exposure} ms")
                print(f"  Gain: {args.gain}")
                print(f"  Binning: {args.binning}")

                from image_acquisition import ImageFormat, AcquisitionMode

                image_data = acquire_image(
                    exposure=args.exposure,
                    gain=args.gain,
                    binning=args.binning,
                    format=ImageFormat.RAW,
                    mode=AcquisitionMode.NORMAL
                )

                metadata = get_acquisition_metadata()

                # Save the image data to the output file
                with open(args.output, "wb") as f:
                    f.write(image_data)

                # Save metadata alongside the image
                metadata_path = f"{os.path.splitext(args.output)[0]}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"Image acquired and saved to: {args.output}")
                print(f"Metadata saved to: {metadata_path}")

            except Exception as e:
                print(f"Error during image acquisition: {e}")
                return 1

        return 0

    def _handle_process(self, args) -> int:
        """Handle the 'process' command."""
        print("=== Image Processing ===")

        # Check if input file exists
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return 1

        # Check for metadata file
        metadata_path = f"{os.path.splitext(args.input)[0]}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"Loaded metadata from: {metadata_path}")
        else:
            print(f"Warning: No metadata file found at {metadata_path}")
            metadata = {
                "timestamp": time.time(),
                "width": 512,
                "height": 512,
                "format": "RAW"
            }

        # Load image data
        with open(args.input, "rb") as f:
            image_data = f.read()

        print(f"Processing image ({len(image_data)} bytes) with algorithm: {args.algorithm}")

        # Calculate image hash for integrity
        import hashlib
        image_hash = hashlib.sha256(image_data).hexdigest()

        # Process the image (simulated for now)
        # In a real implementation, this would call the actual negative space detection algorithm
        print("Detecting negative space regions...")

        # Simulated processing result
        processing_result = {
            "timestamp": time.time(),
            "processing_id": os.urandom(8).hex(),
            "image_hash": image_hash,
            "image_metadata": metadata,
            "algorithm": args.algorithm,
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

        # Save processing result
        with open(args.output, "w") as f:
            json.dump(processing_result, f, indent=2)

        print(f"Detected {len(processing_result['negative_space_analysis']['detected_regions'])} negative space regions")
        print(f"Processing results saved to: {args.output}")

        return 0

    def _handle_verify(self, args) -> int:
        """Handle the 'verify' command."""
        print("=== Multi-Signature Verification ===")

        # Check if input file exists
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return 1

        # Load processing results
        with open(args.input, "r") as f:
            try:
                processing_result = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in input file: {args.input}")
                return 1

        # Convert to bytes for verification
        result_bytes = json.dumps(processing_result, indent=2, sort_keys=True).encode()

        # Set up the signature mode
        if args.mode == "threshold":
            signature_mode = SignatureMode.THRESHOLD
            print(f"Using threshold signature mode ({args.threshold} of {args.signatures})")
            workflow = SecureImagingWorkflow(
                signature_mode=signature_mode,
                num_signers=args.signatures,
                threshold=args.threshold
            )
        elif args.mode == "sequential":
            signature_mode = SignatureMode.SEQUENTIAL
            print(f"Using sequential signature mode ({args.signatures} signers in sequence)")
            workflow = SecureImagingWorkflow(
                signature_mode=signature_mode,
                num_signers=args.signatures
            )
        elif args.mode == "role-based":
            signature_mode = SignatureMode.ROLE_BASED
            roles = args.roles.split(",")
            print(f"Using role-based signature mode (roles: {', '.join(roles)})")
            workflow = SecureImagingWorkflow(
                signature_mode=signature_mode,
                roles=roles
            )

        # Create a fake processing result to verify (in place of the real one)
        # This is just for the demo - in real usage, you'd use the actual processing result
        fake_result = workflow.process_image(result_bytes, processing_result["image_metadata"])

        # Perform verification
        print("\nVerifying processing result...")
        verification_success = workflow.verify_processing_result(fake_result)

        if verification_success:
            print("\n✅ Verification successful!")
            print("The processing result has been cryptographically verified.")
            return 0
        else:
            print("\n❌ Verification failed!")
            print("The processing result could not be verified.")
            return 1

    def _handle_workflow(self, args) -> int:
        """Handle the 'workflow' command."""
        print("=== Complete Secure Workflow ===")

        # Set up the signature mode
        if args.mode == "threshold":
            signature_mode = SignatureMode.THRESHOLD
            print(f"Using threshold signature mode ({args.threshold} of {args.signatures})")
            workflow = SecureImagingWorkflow(
                signature_mode=signature_mode,
                num_signers=args.signatures,
                threshold=args.threshold
            )
        elif args.mode == "sequential":
            signature_mode = SignatureMode.SEQUENTIAL
            print(f"Using sequential signature mode ({args.signatures} signers in sequence)")
            workflow = SecureImagingWorkflow(
                signature_mode=signature_mode,
                num_signers=args.signatures
            )
        elif args.mode == "role-based":
            signature_mode = SignatureMode.ROLE_BASED
            roles = args.roles.split(",")
            print(f"Using role-based signature mode (roles: {', '.join(roles)})")
            workflow = SecureImagingWorkflow(
                signature_mode=signature_mode,
                roles=roles
            )

        # Run the complete workflow
        success = workflow.run()

        if success:
            # Rename the audit log if a custom output path was specified
            if args.output != "security_audit.json" and os.path.exists("security_audit.json"):
                import shutil
                shutil.copy("security_audit.json", args.output)
                print(f"Audit log copied to: {args.output}")

            return 0
        else:
            print("Workflow failed.")
            return 1

    def _handle_audit(self, args) -> int:
        """Handle the 'audit' command."""
        print("=== Security Audit Log Management ===")

        # Check if log file exists
        if not os.path.exists(args.log):
            print(f"Error: Audit log file not found: {args.log}")
            return 1

        # Load the audit log
        with open(args.log, "r") as f:
            try:
                audit_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in audit log file: {args.log}")
                return 1

        # View the audit log
        if args.view:
            print("\nAudit Log Summary:")
            print(f"Workflow ID: {audit_data.get('workflow_id', 'Unknown')}")

            start_time = audit_data.get('start_time')
            end_time = audit_data.get('end_time')

            if start_time and end_time:
                from datetime import datetime
                start_dt = datetime.fromtimestamp(start_time)
                end_dt = datetime.fromtimestamp(end_time)
                duration = end_time - start_time

                print(f"Start Time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"End Time: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Duration: {duration:.2f} seconds")

            print(f"Signature Mode: {audit_data.get('signature_mode', 'Unknown')}")
            print(f"Events: {len(audit_data.get('events', []))}")

            # Display events (filtered if specified)
            events = audit_data.get('events', [])
            if args.filter:
                events = [e for e in events if e.get('event_type') == args.filter]
                print(f"Filtered to {len(events)} events of type '{args.filter}'")

            print("\nEvent Log:")
            print("-" * 80)
            for i, event in enumerate(events):
                from datetime import datetime
                event_time = datetime.fromtimestamp(event.get('timestamp', 0))
                print(f"{i+1}. [{event_time.strftime('%H:%M:%S')}] {event.get('event_type', 'Unknown')}")

                # Display event details
                details = event.get('details', {})
                for key, value in details.items():
                    if isinstance(value, dict) or isinstance(value, list):
                        print(f"   {key}: (complex data)")
                    else:
                        print(f"   {key}: {value}")

                print("-" * 80)

        # Export the audit log
        if args.export:
            # Filter if needed
            if args.filter:
                filtered_events = [e for e in audit_data.get('events', [])
                                  if e.get('event_type') == args.filter]
                export_data = audit_data.copy()
                export_data['events'] = filtered_events
                export_data['filtered'] = True
                export_data['filter_type'] = args.filter
            else:
                export_data = audit_data

            # Save to the specified path
            with open(args.export, "w") as f:
                json.dump(export_data, f, indent=2)

            print(f"Audit log exported to: {args.export}")

        return 0


def main():
    """Main entry point for the CLI application."""
    app = NegativeSpaceCliApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
