#!/usr/bin/env python3
"""
Deployment Test Script for Negative Space Imaging Project

This script runs integration tests against a deployed instance of the
Negative Space Imaging Project to verify functionality.
"""

import argparse
import json
import os
import sys
import logging
import time
import requests
import subprocess
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("test-deployment")

class DeploymentTester:
    """Class to test the deployment functionality"""

    def __init__(self, deployment_type, base_url=None, namespace=None):
        """
        Initialize deployment tester

        Args:
            deployment_type: The type of deployment ('docker-compose' or 'kubernetes')
            base_url: Base URL for API requests (auto-detected if None)
            namespace: Kubernetes namespace (for Kubernetes deployment)
        """
        self.deployment_type = deployment_type
        self.namespace = namespace or "negative-space-imaging"
        self.base_url = base_url
        self.deployment_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        # Initialize test results
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "deployment_type": deployment_type,
            "tests": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
        }

        # Detect base URL if not provided
        if not self.base_url:
            self._detect_base_url()

    def _detect_base_url(self):
        """Detect the base URL for API requests"""
        if self.deployment_type == "docker-compose":
            # For Docker Compose, typically localhost
            self.base_url = "http://localhost:8080"
        elif self.deployment_type == "kubernetes":
            # For Kubernetes, get service endpoint
            try:
                result = subprocess.run(
                    [
                        "kubectl", "get", "service", "imaging-service",
                        "-n", self.namespace, "-o", "jsonpath={.spec.ports[0].nodePort}"
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )

                if result.stdout.strip():
                    node_port = result.stdout.strip()
                    # Get the first node's external IP
                    result = subprocess.run(
                        [
                            "kubectl", "get", "nodes",
                            "-o", "jsonpath={.items[0].status.addresses[?(@.type==\"ExternalIP\")].address}"
                        ],
                        check=True,
                        capture_output=True,
                        text=True
                    )

                    if result.stdout.strip():
                        node_ip = result.stdout.strip()
                        self.base_url = f"http://{node_ip}:{node_port}"
                    else:
                        # Fallback to localhost if no external IP
                        logger.warning("No external IP found for Kubernetes node, using localhost")
                        self.base_url = f"http://localhost:{node_port}"
                else:
                    # If no NodePort, try port-forwarding
                    logger.warning("No NodePort found for service, using port-forwarding")
                    self.base_url = "http://localhost:8080"

                    # Start port-forwarding in the background
                    logger.info("Starting port-forwarding...")
                    subprocess.Popen(
                        [
                            "kubectl", "port-forward", "service/imaging-service",
                            "8080:80", "-n", self.namespace
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    # Give it a moment to establish
                    time.sleep(3)
            except Exception as e:
                logger.warning(f"Failed to detect Kubernetes service URL: {str(e)}")
                self.base_url = "http://localhost:8080"
        else:
            logger.warning(f"Unknown deployment type: {self.deployment_type}")
            self.base_url = "http://localhost:8080"

        logger.info(f"Using base URL: {self.base_url}")

    def _log_test_result(self, test_name, passed, message=None, data=None):
        """
        Log a test result

        Args:
            test_name: The name of the test
            passed: Whether the test passed
            message: Optional message about the test result
            data: Optional data associated with the test
        """
        result = {
            "passed": passed,
            "timestamp": datetime.now().isoformat()
        }

        if message:
            result["message"] = message

        if data:
            result["data"] = data

        self.results["tests"][test_name] = result

        # Update summary
        self.results["summary"]["total"] += 1
        if passed:
            self.results["summary"]["passed"] += 1
            logger.info(f"✅ {test_name}: PASSED")
            if message:
                logger.info(f"   {message}")
        else:
            self.results["summary"]["failed"] += 1
            logger.error(f"❌ {test_name}: FAILED")
            if message:
                logger.error(f"   {message}")

    def test_api_connectivity(self):
        """
        Test basic API connectivity

        Returns:
            bool: Whether the test passed
        """
        test_name = "API Connectivity"

        try:
            url = f"{self.base_url}/api/health"
            logger.info(f"Testing API connectivity: {url}")

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                self._log_test_result(
                    test_name,
                    True,
                    f"API responded with status code {response.status_code}"
                )
                return True
            else:
                self._log_test_result(
                    test_name,
                    False,
                    f"API responded with unexpected status code: {response.status_code}"
                )
                return False
        except requests.RequestException as e:
            self._log_test_result(
                test_name,
                False,
                f"Failed to connect to API: {str(e)}"
            )
            return False

    def test_database_connectivity(self):
        """
        Test database connectivity through the API

        Returns:
            bool: Whether the test passed
        """
        test_name = "Database Connectivity"

        try:
            url = f"{self.base_url}/api/database/health"
            logger.info(f"Testing database connectivity: {url}")

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                self._log_test_result(
                    test_name,
                    True,
                    "Database connection successful"
                )
                return True
            else:
                self._log_test_result(
                    test_name,
                    False,
                    f"Database connection failed with status code: {response.status_code}"
                )
                return False
        except requests.RequestException as e:
            self._log_test_result(
                test_name,
                False,
                f"Failed to test database connectivity: {str(e)}"
            )
            return False

    def test_monitoring_services(self):
        """
        Test monitoring services

        Returns:
            bool: Whether the test passed
        """
        prometheus_passed = self._test_prometheus()
        grafana_passed = self._test_grafana()

        return prometheus_passed and grafana_passed

    def _test_prometheus(self):
        """
        Test Prometheus service

        Returns:
            bool: Whether the test passed
        """
        test_name = "Prometheus Service"

        try:
            if self.deployment_type == "docker-compose":
                url = "http://localhost:9090/api/v1/status/config"
            else:  # kubernetes
                # Start port-forwarding in the background
                port_forward_process = subprocess.Popen(
                    [
                        "kubectl", "port-forward", "service/prometheus",
                        "9090:9090", "-n", self.namespace
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # Give it a moment to establish
                time.sleep(3)
                url = "http://localhost:9090/api/v1/status/config"

            logger.info(f"Testing Prometheus service: {url}")

            response = requests.get(url, timeout=10)

            # Clean up port-forwarding if needed
            if self.deployment_type == "kubernetes":
                port_forward_process.terminate()

            if response.status_code == 200:
                self._log_test_result(
                    test_name,
                    True,
                    "Prometheus service is running"
                )
                return True
            else:
                self._log_test_result(
                    test_name,
                    False,
                    f"Prometheus service returned unexpected status code: {response.status_code}"
                )
                return False
        except requests.RequestException as e:
            # Clean up port-forwarding if needed
            if self.deployment_type == "kubernetes" and 'port_forward_process' in locals():
                port_forward_process.terminate()

            self._log_test_result(
                test_name,
                False,
                f"Failed to connect to Prometheus service: {str(e)}"
            )
            return False

    def _test_grafana(self):
        """
        Test Grafana service

        Returns:
            bool: Whether the test passed
        """
        test_name = "Grafana Service"

        try:
            if self.deployment_type == "docker-compose":
                url = "http://localhost:3000/api/health"
            else:  # kubernetes
                # Start port-forwarding in the background
                port_forward_process = subprocess.Popen(
                    [
                        "kubectl", "port-forward", "service/grafana",
                        "3000:3000", "-n", self.namespace
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # Give it a moment to establish
                time.sleep(3)
                url = "http://localhost:3000/api/health"

            logger.info(f"Testing Grafana service: {url}")

            response = requests.get(url, timeout=10)

            # Clean up port-forwarding if needed
            if self.deployment_type == "kubernetes":
                port_forward_process.terminate()

            if response.status_code == 200:
                self._log_test_result(
                    test_name,
                    True,
                    "Grafana service is running"
                )
                return True
            else:
                self._log_test_result(
                    test_name,
                    False,
                    f"Grafana service returned unexpected status code: {response.status_code}"
                )
                return False
        except requests.RequestException as e:
            # Clean up port-forwarding if needed
            if self.deployment_type == "kubernetes" and 'port_forward_process' in locals():
                port_forward_process.terminate()

            self._log_test_result(
                test_name,
                False,
                f"Failed to connect to Grafana service: {str(e)}"
            )
            return False

    def test_logging_services(self):
        """
        Test logging services

        Returns:
            bool: Whether the test passed
        """
        loki_passed = self._test_loki()

        return loki_passed

    def _test_loki(self):
        """
        Test Loki service

        Returns:
            bool: Whether the test passed
        """
        test_name = "Loki Service"

        try:
            if self.deployment_type == "docker-compose":
                url = "http://localhost:3100/ready"
            else:  # kubernetes
                # Start port-forwarding in the background
                port_forward_process = subprocess.Popen(
                    [
                        "kubectl", "port-forward", "service/loki",
                        "3100:3100", "-n", self.namespace
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # Give it a moment to establish
                time.sleep(3)
                url = "http://localhost:3100/ready"

            logger.info(f"Testing Loki service: {url}")

            response = requests.get(url, timeout=10)

            # Clean up port-forwarding if needed
            if self.deployment_type == "kubernetes":
                port_forward_process.terminate()

            if response.status_code == 200:
                self._log_test_result(
                    test_name,
                    True,
                    "Loki service is running"
                )
                return True
            else:
                self._log_test_result(
                    test_name,
                    False,
                    f"Loki service returned unexpected status code: {response.status_code}"
                )
                return False
        except requests.RequestException as e:
            # Clean up port-forwarding if needed
            if self.deployment_type == "kubernetes" and 'port_forward_process' in locals():
                port_forward_process.terminate()

            self._log_test_result(
                test_name,
                False,
                f"Failed to connect to Loki service: {str(e)}"
            )
            return False

    def test_basic_workflow(self):
        """
        Test basic imaging workflow

        Returns:
            bool: Whether the test passed
        """
        test_name = "Basic Imaging Workflow"

        try:
            # Test image processing endpoint
            url = f"{self.base_url}/api/process-image"
            logger.info(f"Testing image processing workflow: {url}")

            # Create test payload
            payload = {
                "mode": "threshold",
                "parameters": {
                    "threshold": 3,
                    "signatures": 5
                }
            }

            response = requests.post(url, json=payload, timeout=30)

            if response.status_code in (200, 202):
                self._log_test_result(
                    test_name,
                    True,
                    "Image processing workflow completed successfully",
                    data={"response": response.json() if response.content else None}
                )
                return True
            else:
                self._log_test_result(
                    test_name,
                    False,
                    f"Image processing workflow failed with status code: {response.status_code}",
                    data={"response": response.text}
                )
                return False
        except requests.RequestException as e:
            self._log_test_result(
                test_name,
                False,
                f"Failed to test image processing workflow: {str(e)}"
            )
            return False

    def run_all_tests(self):
        """
        Run all deployment tests

        Returns:
            bool: Whether all tests passed
        """
        logger.info(f"Running all tests against {self.deployment_type} deployment")

        # Run tests
        api_ok = self.test_api_connectivity()

        # Only proceed with other tests if API is accessible
        if api_ok:
            self.test_database_connectivity()
            self.test_monitoring_services()
            self.test_logging_services()
            self.test_basic_workflow()
        else:
            logger.warning("Skipping further tests due to API connectivity failure")
            # Mark other tests as skipped
            self._log_test_result("Database Connectivity", False, "Skipped due to API failure")
            self._log_test_result("Prometheus Service", False, "Skipped due to API failure")
            self._log_test_result("Grafana Service", False, "Skipped due to API failure")
            self._log_test_result("Loki Service", False, "Skipped due to API failure")
            self._log_test_result("Basic Imaging Workflow", False, "Skipped due to API failure")

            # Update skipped count
            self.results["summary"]["skipped"] += 5
            self.results["summary"]["failed"] -= 5

        # Return overall result
        return self.results["summary"]["passed"] == self.results["summary"]["total"]

    def display_results(self):
        """Display test results"""
        print("\n=== Negative Space Imaging Project Deployment Test Results ===")
        print(f"Timestamp: {datetime.fromisoformat(self.results['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Deployment Type: {self.results['deployment_type']}")
        print(f"Base URL: {self.base_url}")
        print("\nTest Results:")

        for test_name, result in self.results["tests"].items():
            status = "PASSED" if result["passed"] else "FAILED"
            print(f"\n{test_name}: {status}")

            if "message" in result:
                print(f"  Message: {result['message']}")

        print("\nSummary:")
        print(f"  Total Tests: {self.results['summary']['total']}")
        print(f"  Passed: {self.results['summary']['passed']}")
        print(f"  Failed: {self.results['summary']['failed']}")
        print(f"  Skipped: {self.results['summary']['skipped']}")

        overall = "PASSED" if self.results["summary"]["passed"] == self.results["summary"]["total"] else "FAILED"
        print(f"\nOverall Result: {overall}")

    def export_results(self, file_path):
        """
        Export test results to a file

        Args:
            file_path: The path to save the results to

        Returns:
            bool: Whether the export was successful
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.results, f, indent=2)

            logger.info(f"Test results exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export test results: {str(e)}")
            return False

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Negative Space Imaging Project Deployment Test"
    )

    parser.add_argument("--type", "-t", choices=["docker-compose", "kubernetes"],
                        default="docker-compose",
                        help="Deployment type (default: docker-compose)")

    parser.add_argument("--url", "-u",
                        help="Base URL for API requests (auto-detected if not provided)")

    parser.add_argument("--namespace", "-n", default="negative-space-imaging",
                        help="Kubernetes namespace (default: negative-space-imaging)")

    parser.add_argument("--export", "-e",
                        help="Export the test results to a file")

    parser.add_argument("--skip-monitoring", action="store_true",
                        help="Skip monitoring service tests")

    parser.add_argument("--skip-logging", action="store_true",
                        help="Skip logging service tests")

    args = parser.parse_args()

    # Run tests
    tester = DeploymentTester(args.type, args.url, args.namespace)

    # Always test API connectivity
    api_ok = tester.test_api_connectivity()

    # Only proceed with other tests if API is accessible
    if api_ok:
        tester.test_database_connectivity()

        if not args.skip_monitoring:
            tester.test_monitoring_services()
        else:
            logger.info("Skipping monitoring service tests")

        if not args.skip_logging:
            tester.test_logging_services()
        else:
            logger.info("Skipping logging service tests")

        tester.test_basic_workflow()
    else:
        logger.warning("Skipping further tests due to API connectivity failure")

    # Display and export results
    tester.display_results()

    if args.export:
        tester.export_results(args.export)

    # Return exit code based on test results
    return 0 if tester.results["summary"]["failed"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
