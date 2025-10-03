#!/usr/bin/env python3
"""
Deployment Health Check Script for Negative Space Imaging Project
This script checks the health of deployed services and provides diagnostics
"""

import argparse
import json
import subprocess
import sys
import logging
import time
import re
from datetime import datetime
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("healthcheck")

class HealthCheck:
    """Class to handle health checks for different deployment types"""

    def __init__(self, deployment_type="auto", namespace="negative-space-imaging"):
        """
        Initialize health check

        Args:
            deployment_type: The type of deployment ('docker-compose', 'kubernetes', or 'auto')
            namespace: The Kubernetes namespace (for Kubernetes deployment)
        """
        self.namespace = namespace
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "deployment_type": None,
            "overall_status": "unknown",
            "services": {}
        }

        # Determine deployment type
        if deployment_type == "auto":
            self.deployment_type = self._detect_deployment_type()
        else:
            self.deployment_type = deployment_type

        self.report["deployment_type"] = self.deployment_type

    def _detect_deployment_type(self):
        """
        Detect the type of deployment

        Returns:
            str: The type of deployment ('docker-compose', 'kubernetes', or None)
        """
        # Check for Docker Compose deployment
        try:
            result = subprocess.run(
                ["docker-compose", "-f", "deployment/docker-compose.yaml", "ps", "--quiet"],
                check=False,
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                logger.info("Docker Compose deployment detected")
                return "docker-compose"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Check for Kubernetes deployment
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", self.namespace],
                check=False,
                capture_output=True,
                text=True
            )
            if "No resources found" not in result.stderr and result.returncode == 0:
                logger.info("Kubernetes deployment detected")
                return "kubernetes"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        logger.warning("No active deployment detected")
        return None

    def check_docker_compose_services(self):
        """
        Check the health of Docker Compose services

        Returns:
            bool: Whether all services are healthy
        """
        logger.info("Checking Docker Compose services...")

        # Get service status
        try:
            result = subprocess.run(
                ["docker-compose", "-f", "deployment/docker-compose.yaml", "ps"],
                check=True,
                capture_output=True,
                text=True
            )

            # Parse services and status
            services = {}
            lines = result.stdout.strip().split('\n')
            if len(lines) <= 1:
                logger.warning("No Docker Compose services found")
                return False

            # Skip header line
            for line in lines[1:]:
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 3:
                    service_name = parts[0].split('_')[-1]  # Extract service name from container name
                    status = "healthy" if "Up" in parts[2] else "unhealthy"
                    services[service_name] = {
                        "status": status,
                        "container_id": parts[0],
                        "container_status": parts[2]
                    }

            # Check container health
            for service, info in services.items():
                try:
                    # Get container health status if available
                    health_result = subprocess.run(
                        ["docker", "inspect", "--format", "{{.State.Health.Status}}", info["container_id"]],
                        check=False,
                        capture_output=True,
                        text=True
                    )
                    health_status = health_result.stdout.strip()

                    if health_status and health_status != "<nil>":
                        info["health_status"] = health_status
                        if health_status == "healthy":
                            info["status"] = "healthy"
                        else:
                            info["status"] = "unhealthy"
                except Exception as e:
                    logger.warning(f"Failed to check health for {service}: {str(e)}")

            # Add to report
            self.report["services"] = services

            # Check overall status
            all_healthy = all(info["status"] == "healthy" for info in services.values())
            self.report["overall_status"] = "healthy" if all_healthy else "unhealthy"

            return all_healthy

        except Exception as e:
            logger.error(f"Failed to check Docker Compose services: {str(e)}")
            self.report["overall_status"] = "error"
            return False

    def check_kubernetes_services(self):
        """
        Check the health of Kubernetes services

        Returns:
            bool: Whether all services are healthy
        """
        logger.info(f"Checking Kubernetes services in namespace {self.namespace}...")

        # Get pod status
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", self.namespace, "-o", "json"],
                check=True,
                capture_output=True,
                text=True
            )

            pods = json.loads(result.stdout)
            services = {}

            for pod in pods.get("items", []):
                pod_name = pod["metadata"]["name"]
                service_name = "-".join(pod_name.split("-")[:-2]) if "-" in pod_name else pod_name

                # Extract status
                phase = pod["status"]["phase"]
                container_statuses = pod["status"].get("containerStatuses", [])

                ready = all(status.get("ready", False) for status in container_statuses)
                restarts = sum(status.get("restartCount", 0) for status in container_statuses)

                if service_name not in services:
                    services[service_name] = {
                        "pods": [],
                        "status": "unknown"
                    }

                pod_info = {
                    "name": pod_name,
                    "phase": phase,
                    "ready": ready,
                    "restarts": restarts
                }

                # Determine status
                if phase == "Running" and ready:
                    pod_info["status"] = "healthy"
                elif phase == "Pending":
                    pod_info["status"] = "starting"
                else:
                    pod_info["status"] = "unhealthy"

                services[service_name]["pods"].append(pod_info)

            # Determine service status based on pod status
            for service_name, service_info in services.items():
                if all(pod["status"] == "healthy" for pod in service_info["pods"]):
                    service_info["status"] = "healthy"
                elif any(pod["status"] == "starting" for pod in service_info["pods"]):
                    service_info["status"] = "starting"
                else:
                    service_info["status"] = "unhealthy"

            # Add to report
            self.report["services"] = services

            # Check overall status
            all_healthy = all(info["status"] == "healthy" for info in services.values())
            self.report["overall_status"] = "healthy" if all_healthy else "unhealthy"

            return all_healthy

        except Exception as e:
            logger.error(f"Failed to check Kubernetes services: {str(e)}")
            self.report["overall_status"] = "error"
            return False

    def check_health(self):
        """
        Check the health of all services based on deployment type

        Returns:
            bool: Whether all services are healthy
        """
        if not self.deployment_type:
            logger.error("No deployment detected")
            return False

        if self.deployment_type == "docker-compose":
            return self.check_docker_compose_services()
        elif self.deployment_type == "kubernetes":
            return self.check_kubernetes_services()
        else:
            logger.error(f"Unsupported deployment type: {self.deployment_type}")
            return False

    def display_report(self):
        """Display the health check report"""
        print("\n=== Negative Space Imaging Project Health Check ===")
        print(f"Timestamp: {datetime.fromisoformat(self.report['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Deployment Type: {self.report['deployment_type']}")
        print(f"Overall Status: {self.report['overall_status'].upper()}")
        print("\nServices:")

        if self.deployment_type == "docker-compose":
            # Format Docker Compose service report
            table_data = []
            for service_name, service_info in self.report["services"].items():
                status = service_info["status"].upper()
                health = service_info.get("health_status", "N/A")
                container_status = service_info.get("container_status", "N/A")

                table_data.append([
                    service_name,
                    status,
                    health,
                    container_status
                ])

            print(tabulate(
                table_data,
                headers=["Service", "Status", "Health", "Container Status"],
                tablefmt="grid"
            ))

        elif self.deployment_type == "kubernetes":
            # Format Kubernetes service report
            for service_name, service_info in self.report["services"].items():
                print(f"\n{service_name}: {service_info['status'].upper()}")

                pod_data = []
                for pod in service_info["pods"]:
                    pod_data.append([
                        pod["name"],
                        pod["phase"],
                        "Yes" if pod["ready"] else "No",
                        pod["restarts"],
                        pod["status"].upper()
                    ])

                print(tabulate(
                    pod_data,
                    headers=["Pod", "Phase", "Ready", "Restarts", "Status"],
                    tablefmt="grid"
                ))

    def export_report(self, file_path):
        """
        Export the health check report to a file

        Args:
            file_path: The path to save the report to

        Returns:
            bool: Whether the export was successful
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.report, f, indent=2)

            logger.info(f"Report exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export report: {str(e)}")
            return False

def main():
    """Main entry point for the script"""
    try:
        # Check if tabulate is installed
        import tabulate
    except ImportError:
        print("Error: The 'tabulate' package is required.")
        print("Please install it using: pip install tabulate")
        return 1

    parser = argparse.ArgumentParser(description="Negative Space Imaging Project Health Check")

    parser.add_argument("--deployment-type", "-t", choices=["docker-compose", "kubernetes", "auto"],
                        default="auto", help="The type of deployment")

    parser.add_argument("--namespace", "-n", default="negative-space-imaging",
                        help="Kubernetes namespace (for Kubernetes deployment)")

    parser.add_argument("--export", "-e", help="Export the report to a file")

    parser.add_argument("--watch", "-w", action="store_true",
                        help="Watch mode - continuously check health")

    parser.add_argument("--interval", "-i", type=int, default=30,
                        help="Interval in seconds for watch mode (default: 30)")

    args = parser.parse_args()

    # Run health check
    health_checker = HealthCheck(args.deployment_type, args.namespace)

    if args.watch:
        try:
            print(f"Watching health status (Ctrl+C to stop, interval: {args.interval}s)")
            while True:
                health_checker.check_health()
                health_checker.display_report()

                if args.export:
                    health_checker.export_report(args.export)

                print(f"\nNext check in {args.interval} seconds (Ctrl+C to stop)...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nWatch mode stopped")
    else:
        health_checker.check_health()
        health_checker.display_report()

        if args.export:
            health_checker.export_report(args.export)

    return 0 if health_checker.report["overall_status"] == "healthy" else 1

if __name__ == "__main__":
    sys.exit(main())
