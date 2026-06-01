#!/usr/bin/env python3
"""
Deployment Automation Script for Negative Space Imaging Project

This script automates the process of deploying the Negative Space Imaging Project
across different environments using Docker Compose or Kubernetes.
"""

import argparse
import os
import sys
import subprocess
import logging
import time
import json
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("deploy-auto")

class DeploymentAutomation:
    """Class to automate deployment processes"""

    def __init__(self, deployment_dir=None):
        """
        Initialize deployment automation

        Args:
            deployment_dir: Path to the deployment directory
        """
        # Set deployment directory
        if deployment_dir is None:
            # Assume script is in the deployment directory
            self.deployment_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.deployment_dir = Path(deployment_dir)

        # Initialize deployment status
        self.status = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "docker_compose": {
                "deployed": False,
                "services": {}
            },
            "kubernetes": {
                "deployed": False,
                "resources": {}
            }
        }

    def deploy_docker_compose(self, detached=True, build=False):
        """
        Deploy using Docker Compose

        Args:
            detached: Whether to run in detached mode
            build: Whether to build images before starting

        Returns:
            bool: Whether the deployment was successful
        """
        logger.info("Deploying with Docker Compose...")

        docker_compose_file = self.deployment_dir / "docker-compose.yaml"
        if not docker_compose_file.exists():
            logger.error(f"Docker Compose file not found: {docker_compose_file}")
            return False

        try:
            # Build images if requested
            if build:
                logger.info("Building Docker images...")
                result = subprocess.run(
                    ["docker-compose", "-f", str(docker_compose_file), "build"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info("Docker images built successfully")

            # Start services
            cmd = ["docker-compose", "-f", str(docker_compose_file), "up"]
            if detached:
                cmd.append("-d")

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            logger.info("Docker Compose services started successfully")

            # Get service status
            result = subprocess.run(
                ["docker-compose", "-f", str(docker_compose_file), "ps", "--format", "json"],
                check=True,
                capture_output=True,
                text=True
            )

            # Parse services
            try:
                services = json.loads(result.stdout)
                for service in services:
                    name = service.get("Service", "unknown")
                    self.status["docker_compose"]["services"][name] = {
                        "state": service.get("State", "unknown"),
                        "health": service.get("Health", "unknown")
                    }
            except json.JSONDecodeError:
                # Fallback for older Docker Compose versions that don't support --format json
                result = subprocess.run(
                    ["docker-compose", "-f", str(docker_compose_file), "ps"],
                    check=True,
                    capture_output=True,
                    text=True
                )

                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) >= 3:
                            name = parts[0].split('_')[-1]  # Extract service name
                            state = "running" if "Up" in line else "stopped"
                            self.status["docker_compose"]["services"][name] = {
                                "state": state,
                                "health": "unknown"
                            }

            self.status["docker_compose"]["deployed"] = True
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker Compose deployment failed: {e}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Docker Compose deployment failed: {str(e)}")
            return False

    def deploy_kubernetes(self, namespace="negative-space-imaging", create_namespace=True):
        """
        Deploy using Kubernetes

        Args:
            namespace: Kubernetes namespace
            create_namespace: Whether to create the namespace if it doesn't exist

        Returns:
            bool: Whether the deployment was successful
        """
        logger.info(f"Deploying with Kubernetes to namespace '{namespace}'...")

        k8s_dir = self.deployment_dir / "kubernetes"
        if not k8s_dir.exists() or not k8s_dir.is_dir():
            logger.error(f"Kubernetes directory not found: {k8s_dir}")
            return False

        try:
            # Create namespace if needed
            if create_namespace:
                logger.info(f"Ensuring namespace '{namespace}' exists...")
                result = subprocess.run(
                    ["kubectl", "get", "namespace", namespace],
                    check=False,
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    logger.info(f"Creating namespace '{namespace}'...")
                    result = subprocess.run(
                        ["kubectl", "create", "namespace", namespace],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    logger.info(f"Namespace '{namespace}' created successfully")

            # Deploy all YAML files
            yaml_files = sorted(k8s_dir.glob("**/*.yaml"))
            if not yaml_files:
                logger.error("No Kubernetes YAML files found")
                return False

            # Deploy infrastructure resources first (namespace, storage, etc.)
            for yaml_file in yaml_files:
                if "infrastructure" in str(yaml_file):
                    logger.info(f"Applying infrastructure resource: {yaml_file.name}")
                    result = subprocess.run(
                        ["kubectl", "apply", "-f", str(yaml_file), "-n", namespace],
                        check=True,
                        capture_output=True,
                        text=True
                    )

            # Then deploy services and deployments
            for yaml_file in yaml_files:
                if "infrastructure" not in str(yaml_file):
                    logger.info(f"Applying resource: {yaml_file.name}")
                    result = subprocess.run(
                        ["kubectl", "apply", "-f", str(yaml_file), "-n", namespace],
                        check=True,
                        capture_output=True,
                        text=True
                    )

            logger.info("Kubernetes resources applied successfully")

            # Wait for deployments to be ready
            logger.info("Waiting for deployments to be ready...")
            time.sleep(5)  # Give Kubernetes some time to process

            result = subprocess.run(
                ["kubectl", "get", "deployments", "-n", namespace, "-o", "json"],
                check=True,
                capture_output=True,
                text=True
            )

            deployments = json.loads(result.stdout).get("items", [])
            for deployment in deployments:
                name = deployment["metadata"]["name"]
                status = deployment["status"]

                ready = status.get("readyReplicas", 0)
                total = status.get("replicas", 0)

                self.status["kubernetes"]["resources"][name] = {
                    "type": "deployment",
                    "ready": f"{ready}/{total}",
                    "available": status.get("availableReplicas", 0)
                }

            # Get services
            result = subprocess.run(
                ["kubectl", "get", "services", "-n", namespace, "-o", "json"],
                check=True,
                capture_output=True,
                text=True
            )

            services = json.loads(result.stdout).get("items", [])
            for service in services:
                name = service["metadata"]["name"]
                spec = service["spec"]

                self.status["kubernetes"]["resources"][name] = {
                    "type": "service",
                    "cluster_ip": spec.get("clusterIP", "None"),
                    "ports": spec.get("ports", [])
                }

            self.status["kubernetes"]["deployed"] = True
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {str(e)}")
            return False

    def check_deployment(self, deployment_type):
        """
        Check the status of deployed services

        Args:
            deployment_type: The type of deployment ('docker-compose' or 'kubernetes')

        Returns:
            bool: Whether the deployment is healthy
        """
        if deployment_type == "docker-compose":
            return self._check_docker_compose()
        elif deployment_type == "kubernetes":
            return self._check_kubernetes()
        else:
            logger.error(f"Unsupported deployment type: {deployment_type}")
            return False

    def _check_docker_compose(self):
        """
        Check the status of Docker Compose services

        Returns:
            bool: Whether all services are running and healthy
        """
        logger.info("Checking Docker Compose services...")

        docker_compose_file = self.deployment_dir / "docker-compose.yaml"
        if not docker_compose_file.exists():
            logger.error(f"Docker Compose file not found: {docker_compose_file}")
            return False

        try:
            # Get service status
            result = subprocess.run(
                ["docker-compose", "-f", str(docker_compose_file), "ps"],
                check=True,
                capture_output=True,
                text=True
            )

            # Check if services are running
            lines = result.stdout.strip().split('\n')
            if len(lines) <= 1:
                logger.warning("No Docker Compose services running")
                return False

            all_running = True
            for line in lines[1:]:
                if "Up" not in line:
                    all_running = False
                    break

            return all_running
        except Exception as e:
            logger.error(f"Failed to check Docker Compose services: {str(e)}")
            return False

    def _check_kubernetes(self, namespace="negative-space-imaging"):
        """
        Check the status of Kubernetes resources

        Args:
            namespace: Kubernetes namespace

        Returns:
            bool: Whether all deployments are ready
        """
        logger.info(f"Checking Kubernetes resources in namespace '{namespace}'...")

        try:
            # Check deployments
            result = subprocess.run(
                ["kubectl", "get", "deployments", "-n", namespace, "-o", "json"],
                check=True,
                capture_output=True,
                text=True
            )

            deployments = json.loads(result.stdout).get("items", [])
            if not deployments:
                logger.warning("No Kubernetes deployments found")
                return False

            all_ready = True
            for deployment in deployments:
                name = deployment["metadata"]["name"]
                status = deployment["status"]

                ready = status.get("readyReplicas", 0)
                total = status.get("replicas", 0)

                if ready < total:
                    all_ready = False
                    logger.warning(f"Deployment '{name}' not fully ready: {ready}/{total}")

            return all_ready
        except Exception as e:
            logger.error(f"Failed to check Kubernetes resources: {str(e)}")
            return False

    def cleanup_docker_compose(self):
        """
        Remove Docker Compose deployment

        Returns:
            bool: Whether the cleanup was successful
        """
        logger.info("Cleaning up Docker Compose deployment...")

        docker_compose_file = self.deployment_dir / "docker-compose.yaml"
        if not docker_compose_file.exists():
            logger.error(f"Docker Compose file not found: {docker_compose_file}")
            return False

        try:
            # Stop and remove containers
            result = subprocess.run(
                ["docker-compose", "-f", str(docker_compose_file), "down", "--volumes", "--remove-orphans"],
                check=True,
                capture_output=True,
                text=True
            )

            logger.info("Docker Compose deployment cleaned up successfully")
            self.status["docker_compose"]["deployed"] = False
            self.status["docker_compose"]["services"] = {}

            return True
        except Exception as e:
            logger.error(f"Failed to clean up Docker Compose deployment: {str(e)}")
            return False

    def cleanup_kubernetes(self, namespace="negative-space-imaging", delete_namespace=False):
        """
        Remove Kubernetes deployment

        Args:
            namespace: Kubernetes namespace
            delete_namespace: Whether to delete the namespace

        Returns:
            bool: Whether the cleanup was successful
        """
        logger.info(f"Cleaning up Kubernetes deployment in namespace '{namespace}'...")

        k8s_dir = self.deployment_dir / "kubernetes"
        if not k8s_dir.exists() or not k8s_dir.is_dir():
            logger.error(f"Kubernetes directory not found: {k8s_dir}")
            return False

        try:
            # Delete all resources in reverse order
            yaml_files = sorted(k8s_dir.glob("**/*.yaml"), reverse=True)
            if not yaml_files:
                logger.warning("No Kubernetes YAML files found")

                # Still try to clean up based on namespace
                if delete_namespace:
                    logger.info(f"Deleting namespace '{namespace}'...")
                    subprocess.run(
                        ["kubectl", "delete", "namespace", namespace],
                        check=False,
                        capture_output=True,
                        text=True
                    )
                    logger.info(f"Namespace '{namespace}' deleted")

                return True

            # Delete services and deployments first
            for yaml_file in yaml_files:
                if "infrastructure" not in str(yaml_file):
                    logger.info(f"Deleting resource: {yaml_file.name}")
                    subprocess.run(
                        ["kubectl", "delete", "-f", str(yaml_file), "-n", namespace],
                        check=False,
                        capture_output=True,
                        text=True
                    )

            # Then delete infrastructure resources
            for yaml_file in yaml_files:
                if "infrastructure" in str(yaml_file):
                    logger.info(f"Deleting infrastructure resource: {yaml_file.name}")
                    subprocess.run(
                        ["kubectl", "delete", "-f", str(yaml_file), "-n", namespace],
                        check=False,
                        capture_output=True,
                        text=True
                    )

            # Delete namespace if requested
            if delete_namespace:
                logger.info(f"Deleting namespace '{namespace}'...")
                subprocess.run(
                    ["kubectl", "delete", "namespace", namespace],
                    check=False,
                    capture_output=True,
                    text=True
                )
                logger.info(f"Namespace '{namespace}' deleted")

            logger.info("Kubernetes deployment cleaned up successfully")
            self.status["kubernetes"]["deployed"] = False
            self.status["kubernetes"]["resources"] = {}

            return True
        except Exception as e:
            logger.error(f"Failed to clean up Kubernetes deployment: {str(e)}")
            return False

    def display_status(self):
        """Display deployment status"""
        print("\n=== Negative Space Imaging Project Deployment Status ===")
        print(f"Timestamp: {self.status['timestamp']}")

        # Docker Compose status
        if self.status["docker_compose"]["deployed"]:
            print("\nDocker Compose: DEPLOYED")
            print("Services:")
            for service, info in self.status["docker_compose"]["services"].items():
                print(f"  - {service}: {info['state'].upper()}")
        else:
            print("\nDocker Compose: NOT DEPLOYED")

        # Kubernetes status
        if self.status["kubernetes"]["deployed"]:
            print("\nKubernetes: DEPLOYED")
            print("Resources:")
            for resource, info in self.status["kubernetes"]["resources"].items():
                resource_type = info.get("type", "unknown")
                if resource_type == "deployment":
                    print(f"  - {resource} (Deployment): Ready {info['ready']}")
                elif resource_type == "service":
                    ports = info.get("ports", [])
                    port_str = ", ".join([f"{p.get('port', 'N/A')}/{p.get('protocol', 'TCP')}" for p in ports])
                    print(f"  - {resource} (Service): ClusterIP {info['cluster_ip']}, Ports: {port_str}")
                else:
                    print(f"  - {resource}: {info}")
        else:
            print("\nKubernetes: NOT DEPLOYED")

    def export_status(self, file_path):
        """
        Export deployment status to a file

        Args:
            file_path: The path to save the status to

        Returns:
            bool: Whether the export was successful
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.status, f, indent=2)

            logger.info(f"Status exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export status: {str(e)}")
            return False

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Negative Space Imaging Project Deployment Automation"
    )

    parser.add_argument("action", choices=["deploy", "status", "cleanup"],
                        help="Action to perform")

    parser.add_argument("--type", "-t", choices=["docker-compose", "kubernetes"],
                        default="docker-compose",
                        help="Deployment type (default: docker-compose)")

    parser.add_argument("--deployment-dir", "-d",
                        help="Path to the deployment directory")

    parser.add_argument("--namespace", "-n", default="negative-space-imaging",
                        help="Kubernetes namespace (default: negative-space-imaging)")

    parser.add_argument("--delete-namespace", action="store_true",
                        help="Delete Kubernetes namespace during cleanup")

    parser.add_argument("--build", action="store_true",
                        help="Build Docker images before deploying (Docker Compose only)")

    parser.add_argument("--no-detach", action="store_true",
                        help="Do not run in detached mode (Docker Compose only)")

    parser.add_argument("--export", "-e",
                        help="Export the deployment status to a file")

    args = parser.parse_args()

    # Initialize deployment automation
    automation = DeploymentAutomation(args.deployment_dir)

    # Perform the requested action
    if args.action == "deploy":
        if args.type == "docker-compose":
            success = automation.deploy_docker_compose(
                detached=not args.no_detach,
                build=args.build
            )
        else:  # kubernetes
            success = automation.deploy_kubernetes(
                namespace=args.namespace,
                create_namespace=True
            )

        if success:
            print("Deployment successful!")
        else:
            print("Deployment failed!")
            return 1

    elif args.action == "status":
        if args.type == "docker-compose":
            automation.check_deployment("docker-compose")
        else:  # kubernetes
            automation.check_deployment("kubernetes")

    elif args.action == "cleanup":
        if args.type == "docker-compose":
            success = automation.cleanup_docker_compose()
        else:  # kubernetes
            success = automation.cleanup_kubernetes(
                namespace=args.namespace,
                delete_namespace=args.delete_namespace
            )

        if success:
            print("Cleanup successful!")
        else:
            print("Cleanup failed!")
            return 1

    # Display and export status
    automation.display_status()

    if args.export:
        automation.export_status(args.export)

    return 0

if __name__ == "__main__":
    sys.exit(main())
