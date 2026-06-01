#!/usr/bin/env python3
"""
Deployment script for Negative Space Imaging Project
This script automates the deployment process using Docker Compose or Kubernetes
"""

import argparse
import os
import subprocess
import sys
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("deployment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("deployment")

def check_prerequisites():
    """
    Check if all prerequisites are met for deployment
    """
    logger.info("Checking deployment prerequisites...")

    # Check if Docker is installed
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        logger.info("Docker is installed")
        docker_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Docker is not installed or not available in PATH")
        docker_available = False

    # Check if Docker Compose is installed
    try:
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
        logger.info("Docker Compose is installed")
        docker_compose_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Docker Compose is not installed or not available in PATH")
        docker_compose_available = False

    # Check if kubectl is installed (for Kubernetes deployment)
    try:
        subprocess.run(["kubectl", "version", "--client"], check=True, capture_output=True)
        logger.info("kubectl is installed")
        k8s_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("kubectl is not installed or not available in PATH")
        k8s_available = False

    # Check if required files exist
    docker_compose_file = Path("deployment/docker-compose.yaml")
    if not docker_compose_file.exists():
        logger.error(f"Docker Compose file not found: {docker_compose_file}")
        return False

    # Return overall status
    if not (docker_available and docker_compose_available) and not k8s_available:
        logger.error("Neither Docker with Docker Compose nor Kubernetes is available")
        return False

    return True

def deploy_docker_compose(environment="development", build=False):
    """
    Deploy using Docker Compose

    Args:
        environment: The deployment environment (development, staging, production)
        build: Whether to build the images before deployment

    Returns:
        bool: Whether the deployment was successful
    """
    logger.info(f"Deploying with Docker Compose in {environment} environment")

    # Define environment variables
    env_vars = {
        "ENVIRONMENT": environment,
        "TAG": environment,
        **os.environ
    }

    try:
        # Change to the project root directory
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Create directories for Docker volumes if they don't exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("processing", exist_ok=True)

        # Build images if requested
        if build:
            logger.info("Building Docker images...")
            subprocess.run(
                ["docker-compose", "-f", "deployment/docker-compose.yaml", "build"],
                check=True,
                env=env_vars
            )

        # Start the services
        logger.info("Starting services...")
        subprocess.run(
            ["docker-compose", "-f", "deployment/docker-compose.yaml", "up", "-d"],
            check=True,
            env=env_vars
        )

        # Check if services are running
        result = subprocess.run(
            ["docker-compose", "-f", "deployment/docker-compose.yaml", "ps"],
            check=True,
            capture_output=True,
            text=True,
            env=env_vars
        )

        logger.info("Docker Compose deployment completed successfully")
        logger.info("Service status:")
        logger.info(result.stdout)

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker Compose deployment failed: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Docker Compose deployment failed: {str(e)}")
        return False

def deploy_kubernetes(environment="development", namespace="negative-space-imaging"):
    """
    Deploy using Kubernetes

    Args:
        environment: The deployment environment (development, staging, production)
        namespace: The Kubernetes namespace to deploy to

    Returns:
        bool: Whether the deployment was successful
    """
    logger.info(f"Deploying with Kubernetes in {environment} environment to namespace {namespace}")

    try:
        # Change to the project root directory
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Check if namespace exists, create if it doesn't
        result = subprocess.run(
            ["kubectl", "get", "namespace", namespace],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.info(f"Creating namespace {namespace}...")
            subprocess.run(
                ["kubectl", "create", "namespace", namespace],
                check=True
            )

        # Apply Kubernetes manifests
        manifest_dir = Path("deployment/kubernetes/manifests")
        if manifest_dir.exists():
            logger.info("Applying Kubernetes manifests...")
            for manifest in sorted(manifest_dir.glob("*.yaml")):
                logger.info(f"Applying {manifest}...")
                subprocess.run(
                    ["kubectl", "apply", "-f", str(manifest), "-n", namespace],
                    check=True
                )
        else:
            logger.error(f"Kubernetes manifest directory not found: {manifest_dir}")
            return False

        # Check if pods are running
        logger.info("Waiting for pods to be ready...")
        subprocess.run(
            ["kubectl", "wait", "--for=condition=Ready", "pods", "--all", "-n", namespace, "--timeout=300s"],
            check=True
        )

        # Get pod status
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace],
            check=True,
            capture_output=True,
            text=True
        )

        logger.info("Kubernetes deployment completed successfully")
        logger.info("Pod status:")
        logger.info(result.stdout)

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Kubernetes deployment failed: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Kubernetes deployment failed: {str(e)}")
        return False

def stop_deployment(deployment_type="docker-compose", namespace="negative-space-imaging"):
    """
    Stop and clean up the deployment

    Args:
        deployment_type: The type of deployment to stop (docker-compose or kubernetes)
        namespace: The Kubernetes namespace to clean up (only for kubernetes)

    Returns:
        bool: Whether the cleanup was successful
    """
    logger.info(f"Stopping {deployment_type} deployment")

    try:
        # Change to the project root directory
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if deployment_type == "docker-compose":
            # Stop and remove containers, networks, and volumes
            subprocess.run(
                ["docker-compose", "-f", "deployment/docker-compose.yaml", "down", "--volumes"],
                check=True
            )
            logger.info("Docker Compose deployment stopped successfully")
        elif deployment_type == "kubernetes":
            # Delete all resources in the namespace
            subprocess.run(
                ["kubectl", "delete", "all", "--all", "-n", namespace],
                check=True
            )
            logger.info(f"Kubernetes deployment in namespace {namespace} stopped successfully")
        else:
            logger.error(f"Unsupported deployment type: {deployment_type}")
            return False

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to stop {deployment_type} deployment: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Failed to stop {deployment_type} deployment: {str(e)}")
        return False

def display_status(deployment_type="docker-compose", namespace="negative-space-imaging"):
    """
    Display the status of the deployment

    Args:
        deployment_type: The type of deployment to check (docker-compose or kubernetes)
        namespace: The Kubernetes namespace to check (only for kubernetes)

    Returns:
        bool: Whether the status check was successful
    """
    logger.info(f"Checking status of {deployment_type} deployment")

    try:
        # Change to the project root directory
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if deployment_type == "docker-compose":
            # Check the status of the containers
            result = subprocess.run(
                ["docker-compose", "-f", "deployment/docker-compose.yaml", "ps"],
                check=True,
                capture_output=True,
                text=True
            )
            print("\nDocker Compose Status:")
            print("=====================")
            print(result.stdout)

            # Check logs (last few lines for each service)
            services = ["api-gateway", "image-processing", "data-storage", "distributed-computing", "security", "database"]
            print("\nRecent Logs:")
            print("============")
            for service in services:
                print(f"\n{service}:")
                print("-" * len(service))
                try:
                    result = subprocess.run(
                        ["docker-compose", "-f", "deployment/docker-compose.yaml", "logs", "--tail=10", service],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    print(result.stdout)
                except:
                    print("Logs not available")

        elif deployment_type == "kubernetes":
            # Check the status of the pods
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", namespace],
                check=True,
                capture_output=True,
                text=True
            )
            print("\nKubernetes Pods:")
            print("===============")
            print(result.stdout)

            # Check services
            result = subprocess.run(
                ["kubectl", "get", "services", "-n", namespace],
                check=True,
                capture_output=True,
                text=True
            )
            print("\nKubernetes Services:")
            print("===================")
            print(result.stdout)

            # Check recent events
            result = subprocess.run(
                ["kubectl", "get", "events", "--sort-by=.metadata.creationTimestamp", "-n", namespace],
                check=True,
                capture_output=True,
                text=True
            )
            print("\nRecent Events:")
            print("=============")
            print(result.stdout)

        else:
            logger.error(f"Unsupported deployment type: {deployment_type}")
            return False

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to check {deployment_type} status: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Failed to check {deployment_type} status: {str(e)}")
        return False

def main():
    """Main entry point for the deployment script"""
    parser = argparse.ArgumentParser(description="Negative Space Imaging Project Deployment")

    # Main arguments
    parser.add_argument("--check-only", action="store_true", help="Only check prerequisites, don't deploy")
    parser.add_argument("--environment", "-e", choices=["development", "staging", "production"],
                        default="development", help="Deployment environment")
    parser.add_argument("--type", "-t", choices=["docker-compose", "kubernetes"],
                        default="docker-compose", help="Deployment type")

    # Docker Compose specific arguments
    parser.add_argument("--build", "-b", action="store_true", help="Build images before deployment (Docker Compose only)")

    # Kubernetes specific arguments
    parser.add_argument("--namespace", "-n", default="negative-space-imaging",
                        help="Kubernetes namespace (Kubernetes only)")

    # Action arguments
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--deploy", "-d", action="store_true", help="Deploy the application")
    action_group.add_argument("--stop", "-s", action="store_true", help="Stop and clean up the deployment")
    action_group.add_argument("--status", action="store_true", help="Check the status of the deployment")

    args = parser.parse_args()

    # Set default action if none specified
    if not (args.check_only or args.deploy or args.stop or args.status):
        args.deploy = True

    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed")
        return 1

    if args.check_only:
        logger.info("Prerequisites check passed")
        return 0

    # Execute the requested action
    if args.deploy:
        if args.type == "docker-compose":
            if not deploy_docker_compose(args.environment, args.build):
                return 1
        else:  # kubernetes
            if not deploy_kubernetes(args.environment, args.namespace):
                return 1
    elif args.stop:
        if not stop_deployment(args.type, args.namespace):
            return 1
    elif args.status:
        if not display_status(args.type, args.namespace):
            return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
