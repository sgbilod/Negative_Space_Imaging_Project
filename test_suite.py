#!/usr/bin/env python
"""
Comprehensive Test Suite for Secure Negative Space Imaging System
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides comprehensive testing for the entire Negative Space Imaging
workflow, including:
1. Image acquisition
2. Negative space detection
3. Multi-signature verification
4. Security audit logging

Tests include:
- Unit tests for individual components
- Integration tests for component interactions
- End-to-end workflow tests
- Security and compliance tests
- Performance benchmarks

Usage:
  python test_suite.py --all
  python test_suite.py --unit
  python test_suite.py --integration
  python test_suite.py --security
  python test_suite.py --performance
"""

import argparse
import hashlib
import json
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Import local modules (with error handling for missing dependencies)
try:
    from secure_imaging_workflow import SecureImagingWorkflow
    from multi_signature_demo import SignatureMode, SignatureStatus, Role, Signer
    WORKFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules are not available: {e}")
    print("Some tests will be skipped.")
    WORKFLOW_AVAILABLE = False


class TestImageAcquisition(unittest.TestCase):
    """Test cases for image acquisition functionality."""

    def setUp(self):
        """Set up test environment before each test."""
        pass

    def test_simulated_acquisition(self):
        """Test the simulated image acquisition process."""
        if not WORKFLOW_AVAILABLE:
            self.skipTest("SecureImagingWorkflow not available")

        workflow = SecureImagingWorkflow(SignatureMode.THRESHOLD, 3, 2)
        metadata, image_data = workflow.simulate_acquisition()

        # Verify the returned data
        self.assertIsInstance(metadata, dict)
        self.assertIsInstance(image_data, bytes)
        self.assertTrue(len(image_data) > 0)

        # Check metadata fields
        required_fields = ["timestamp", "image_id", "width", "height", "format"]
        for field in required_fields:
            self.assertIn(field, metadata)

    @unittest.skipIf(not os.path.exists("demo_acquisition.py"), "demo_acquisition.py not available")
    def test_real_acquisition(self):
        """Test the real image acquisition process if available."""
        try:
            from demo_acquisition import acquire_image, get_acquisition_metadata
            from image_acquisition import ImageFormat, AcquisitionMode

            # Test acquisition with parameters
            image_data = acquire_image(
                exposure=100,
                gain=1.0,
                binning=1,
                mode="simulation",  # Use simulation mode for reliable testing
                width=512,
                height=512
            )

            # Verify the returned data
            self.assertIsNotNone(image_data)
            self.assertIsInstance(image_data, bytes)
            self.assertTrue(len(image_data) > 0)

        except ImportError:
            self.skipTest("image_acquisition module not available")


class TestMultiSignatureVerification(unittest.TestCase):
    """Test cases for multi-signature verification functionality."""

    def setUp(self):
        """Set up test environment before each test."""
        if not WORKFLOW_AVAILABLE:
            self.skipTest("Required modules not available")

    def test_threshold_signatures(self):
        """Test threshold signature mode."""
        # Create signers and keys
        signers = []
        private_keys = {}

        for i in range(5):
            signer_id = f"test_signer_{i+1}"
            signer, private_key = Signer.generate(signer_id, f"Test Signer {i+1}")
            signers.append(signer)
            private_keys[signer_id] = private_key

        # Create a sample data payload
        test_data = {
            "timestamp": time.time(),
            "test_id": os.urandom(8).hex(),
            "content": "Test payload for multi-signature verification"
        }
        data_bytes = json.dumps(test_data).encode()

        # Create signature request with threshold mode
        from multi_signature_demo import SignatureRequest
        request = SignatureRequest(
            data=data_bytes,
            mode=SignatureMode.THRESHOLD,
            signers=signers,
            threshold=3
        )

        # Sign with 3 signers (should meet threshold)
        for i in range(3):
            signer_id = f"test_signer_{i+1}"
            private_key = private_keys[signer_id]

            # Create signature
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding

            signature = private_key.sign(
                request.data_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            # Add signature to request
            success = request.sign(signer_id, signature)
            self.assertTrue(success)

        # Verify signature status
        self.assertEqual(request.status, SignatureStatus.COMPLETE)
        self.assertTrue(request.is_complete)

    def test_sequential_signatures(self):
        """Test sequential signature mode."""
        # Create signers and keys
        signers = []
        private_keys = {}

        for i in range(3):
            signer_id = f"test_signer_{i+1}"
            signer, private_key = Signer.generate(signer_id, f"Test Signer {i+1}")
            signers.append(signer)
            private_keys[signer_id] = private_key

        # Create a sample data payload
        test_data = {
            "timestamp": time.time(),
            "test_id": os.urandom(8).hex(),
            "content": "Test payload for multi-signature verification"
        }
        data_bytes = json.dumps(test_data).encode()

        # Create signature request with sequential mode
        from multi_signature_demo import SignatureRequest
        request = SignatureRequest(
            data=data_bytes,
            mode=SignatureMode.SEQUENTIAL,
            signers=signers
        )

        # Sign in correct sequence
        for i in range(3):
            signer_id = f"test_signer_{i+1}"
            private_key = private_keys[signer_id]

            # Create signature
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding

            signature = private_key.sign(
                request.data_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            # Add signature to request
            success = request.sign(signer_id, signature)
            self.assertTrue(success)

        # Verify signature status
        self.assertEqual(request.status, SignatureStatus.COMPLETE)
        self.assertTrue(request.is_complete)

    def test_out_of_order_signing(self):
        """Test that out-of-order signing fails in sequential mode."""
        # Create signers and keys
        signers = []
        private_keys = {}

        for i in range(3):
            signer_id = f"test_signer_{i+1}"
            signer, private_key = Signer.generate(signer_id, f"Test Signer {i+1}")
            signers.append(signer)
            private_keys[signer_id] = private_key

        # Create a sample data payload
        test_data = {
            "timestamp": time.time(),
            "test_id": os.urandom(8).hex(),
            "content": "Test payload for multi-signature verification"
        }
        data_bytes = json.dumps(test_data).encode()

        # Create signature request with sequential mode
        from multi_signature_demo import SignatureRequest
        request = SignatureRequest(
            data=data_bytes,
            mode=SignatureMode.SEQUENTIAL,
            signers=signers
        )

        # Try to sign with the second signer first (should fail)
        signer_id = "test_signer_2"
        private_key = private_keys[signer_id]

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        signature = private_key.sign(
            request.data_hash,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        # Add signature to request (should fail)
        success = request.sign(signer_id, signature)
        self.assertFalse(success)

        # Verify status is still PENDING
        self.assertEqual(request.status, SignatureStatus.PENDING)


class TestSecureWorkflow(unittest.TestCase):
    """Test cases for the complete secure imaging workflow."""

    def setUp(self):
        """Set up test environment before each test."""
        if not WORKFLOW_AVAILABLE:
            self.skipTest("Required modules not available")

    @patch('secure_imaging_workflow.SecureImagingWorkflow.simulate_acquisition')
    def test_complete_workflow(self, mock_acquire):
        """Test the complete secure imaging workflow."""
        # Mock the acquisition to return a test image
        test_size = 256
        test_image = np.ones((test_size, test_size)) * 200
        y, x = np.ogrid[:test_size, :test_size]
        center = test_size // 2
        mask = (x - center)**2 + (y - center)**2 <= (test_size//4)**2
        test_image[mask] = 50
        test_image_bytes = test_image.tobytes()

        mock_metadata = {
            "timestamp": time.time(),
            "image_id": "test_image_001",
            "width": test_size,
            "height": test_size,
            "format": "RAW",
            "bit_depth": 8
        }

        mock_acquire.return_value = (mock_metadata, test_image_bytes)

        # Create and run the workflow
        workflow = SecureImagingWorkflow(
            signature_mode=SignatureMode.THRESHOLD,
            num_signers=3,
            threshold=2
        )

        # Run the workflow with threshold signatures
        result = workflow.run()

        # Verify the workflow completed successfully
        self.assertTrue(result)

        # Verify audit log was created
        self.assertTrue(os.path.exists("security_audit.json"))

        # Check the contents of the audit log
        with open("security_audit.json", "r") as f:
            audit_data = json.load(f)
            self.assertIn("events", audit_data)
            self.assertIn("workflow_id", audit_data)

            # Verify the events in the log
            events = [event["event_type"] for event in audit_data["events"]]
            self.assertIn("image_acquired", events)
            self.assertIn("image_processing", events)
            self.assertIn("signature_request_created", events)
            self.assertIn("verification_completed", events)


class TestSecurity(unittest.TestCase):
    """Test cases focused on security aspects."""

    def setUp(self):
        """Set up test environment before each test."""
        if not WORKFLOW_AVAILABLE:
            self.skipTest("Required modules not available")

    def test_tampered_data_rejection(self):
        """Test that tampered data is correctly rejected during verification."""
        # Create signers and keys
        signers = []
        private_keys = {}

        for i in range(3):
            signer_id = f"test_signer_{i+1}"
            signer, private_key = Signer.generate(signer_id, f"Test Signer {i+1}")
            signers.append(signer)
            private_keys[signer_id] = private_key

        # Create original data
        original_data = {
            "timestamp": time.time(),
            "test_id": "test_123",
            "content": "Original content"
        }
        original_bytes = json.dumps(original_data).encode()

        # Create signature request
        from multi_signature_demo import SignatureRequest
        request = SignatureRequest(
            data=original_bytes,
            mode=SignatureMode.THRESHOLD,
            signers=signers,
            threshold=2
        )

        # Generate signature for the original data
        signer_id = "test_signer_1"
        private_key = private_keys[signer_id]

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        signature = private_key.sign(
            request.data_hash,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        # Now create tampered data
        tampered_data = {
            "timestamp": time.time(),
            "test_id": "test_123",
            "content": "Tampered content"  # Changed content
        }
        tampered_bytes = json.dumps(tampered_data).encode()

        # Create a new request with the tampered data
        tampered_request = SignatureRequest(
            data=tampered_bytes,
            mode=SignatureMode.THRESHOLD,
            signers=signers,
            threshold=2
        )

        # Try to verify the signature created for original data with the tampered data
        # This should fail because the signature doesn't match the tampered data
        success = tampered_request.sign(signer_id, signature)
        self.assertFalse(success)


class TestPerformance(unittest.TestCase):
    """Performance benchmark tests."""

    def setUp(self):
        """Set up test environment before each test."""
        if not WORKFLOW_AVAILABLE:
            self.skipTest("Required modules not available")

    def test_signature_performance(self):
        """Benchmark the performance of the signature verification process."""
        # Skip detailed timing tests in regular test runs
        self.skipTest("Performance tests are disabled by default")

        # Number of signatures to test
        num_signatures = [5, 10, 20, 50]

        # Results storage
        results = []

        for num in num_signatures:
            # Create signers and keys
            signers = []
            private_keys = {}

            for i in range(num):
                signer_id = f"bench_signer_{i+1}"
                signer, private_key = Signer.generate(signer_id, f"Bench Signer {i+1}")
                signers.append(signer)
                private_keys[signer_id] = private_key

            # Create test data
            test_data = {
                "timestamp": time.time(),
                "bench_id": os.urandom(8).hex(),
                "content": "Benchmark test data" * 100  # Larger payload
            }
            data_bytes = json.dumps(test_data).encode()

            # Measure time to create signature request
            start_time = time.time()

            from multi_signature_demo import SignatureRequest
            request = SignatureRequest(
                data=data_bytes,
                mode=SignatureMode.THRESHOLD,
                signers=signers,
                threshold=num // 2
            )

            request_time = time.time() - start_time

            # Measure time to sign and verify
            total_sign_time = 0

            for i in range(num // 2):
                signer_id = f"bench_signer_{i+1}"
                private_key = private_keys[signer_id]

                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import padding

                # Time the signature creation
                sign_start = time.time()
                signature = private_key.sign(
                    request.data_hash,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    hashes.SHA256(),
                )

                # Time the verification
                success = request.sign(signer_id, signature)
                sign_end = time.time()

                total_sign_time += (sign_end - sign_start)

            # Record results
            results.append({
                "num_signers": num,
                "threshold": num // 2,
                "request_creation_time": request_time,
                "avg_signature_time": total_sign_time / (num // 2),
                "total_time": request_time + total_sign_time
            })

        # Print results
        print("\nPerformance Benchmark Results:")
        print("-" * 80)
        print(f"{'Signers':<10} {'Threshold':<10} {'Request Time (s)':<20} {'Avg Sign Time (s)':<20} {'Total Time (s)':<15}")
        print("-" * 80)

        for result in results:
            print(f"{result['num_signers']:<10} {result['threshold']:<10} {result['request_creation_time']:<20.6f} {result['avg_signature_time']:<20.6f} {result['total_time']:<15.6f}")


def run_all_tests():
    """Run all test cases."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestImageAcquisition))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestMultiSignatureVerification))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestSecureWorkflow))
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestSecurity))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)


def run_performance_tests():
    """Run performance benchmark tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add performance test cases
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestPerformance))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test Suite for Secure Negative Space Imaging")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--security", action="store_true", help="Run security tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    return parser.parse_args()


def main():
    """Main entry point for the test suite."""
    args = parse_arguments()

    print("="*80)
    print("🧪 Secure Negative Space Imaging Test Suite 🧪")
    print("="*80)

    if args.all or (not any([args.unit, args.integration, args.security, args.performance])):
        print("\nRunning all tests...")
        run_all_tests()

    if args.unit:
        print("\nRunning unit tests...")
        unit_suite = unittest.TestSuite()
        unit_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestImageAcquisition))
        unit_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestMultiSignatureVerification))
        unittest.TextTestRunner(verbosity=2).run(unit_suite)

    if args.integration:
        print("\nRunning integration tests...")
        integration_suite = unittest.TestSuite()
        integration_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestSecureWorkflow))
        unittest.TextTestRunner(verbosity=2).run(integration_suite)

    if args.security:
        print("\nRunning security tests...")
        security_suite = unittest.TestSuite()
        security_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestSecurity))
        unittest.TextTestRunner(verbosity=2).run(security_suite)

    if args.performance:
        print("\nRunning performance tests...")
        run_performance_tests()

    print("\n" + "="*80)
    print("Test suite execution completed.")
    print("="*80)


if __name__ == "__main__":
    main()
