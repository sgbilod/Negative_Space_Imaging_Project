#!/usr/bin/env python3
"""
Monitoring Dashboard Access Script for Negative Space Imaging Project
This script provides easy access to monitoring dashboards for the deployment
"""

import argparse
import os
import subprocess
import sys
import webbrowser
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("monitoring")

def check_deployment_type():
    """
    Check what type of deployment is running (Docker Compose or Kubernetes)

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
            ["kubectl", "get", "pods", "-n", "negative-space-imaging"],
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

def setup_port_forwarding(service, port, namespace="negative-space-imaging"):
    """
    Set up port forwarding for a Kubernetes service

    Args:
        service: The name of the service to forward
        port: The port to forward
        namespace: The Kubernetes namespace

    Returns:
        subprocess.Popen: The process running the port forwarding
    """
    logger.info(f"Setting up port forwarding for {service} on port {port}")

    # Check if the service exists
    result = subprocess.run(
        ["kubectl", "get", "service", service, "-n", namespace],
        check=False,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"Service {service} not found in namespace {namespace}")
        return None

    # Start port forwarding
    process = subprocess.Popen(
        ["kubectl", "port-forward", f"service/{service}", f"{port}:{port}", "-n", namespace],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait a moment for port forwarding to establish
    time.sleep(2)

    # Check if process is still running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        logger.error(f"Port forwarding failed: {stderr}")
        return None

    logger.info(f"Port forwarding established for {service} on port {port}")
    return process

def open_dashboard(url, dashboard_name):
    """
    Open a dashboard URL in the default web browser

    Args:
        url: The URL to open
        dashboard_name: The name of the dashboard

    Returns:
        bool: Whether the operation was successful
    """
    logger.info(f"Opening {dashboard_name} dashboard: {url}")

    try:
        webbrowser.open(url)
        logger.info(f"{dashboard_name} dashboard opened in browser")
        return True
    except Exception as e:
        logger.error(f"Failed to open {dashboard_name} dashboard: {str(e)}")
        print(f"\nPlease manually open the {dashboard_name} dashboard at: {url}")
        return False

def access_grafana(deployment_type, namespace="negative-space-imaging"):
    """
    Access the Grafana dashboard

    Args:
        deployment_type: The type of deployment ('docker-compose' or 'kubernetes')
        namespace: The Kubernetes namespace (for Kubernetes deployment)

    Returns:
        tuple: (bool for success, process for port forwarding if applicable)
    """
    if deployment_type == "docker-compose":
        # For Docker Compose, Grafana should be directly accessible
        return open_dashboard("http://localhost:3000", "Grafana"), None
    elif deployment_type == "kubernetes":
        # For Kubernetes, set up port forwarding
        port_forward_process = setup_port_forwarding("grafana", 3000, namespace)
        if port_forward_process:
            return open_dashboard("http://localhost:3000", "Grafana"), port_forward_process
        return False, None
    return False, None

def access_prometheus(deployment_type, namespace="negative-space-imaging"):
    """
    Access the Prometheus dashboard

    Args:
        deployment_type: The type of deployment ('docker-compose' or 'kubernetes')
        namespace: The Kubernetes namespace (for Kubernetes deployment)

    Returns:
        tuple: (bool for success, process for port forwarding if applicable)
    """
    if deployment_type == "docker-compose":
        # For Docker Compose, Prometheus should be directly accessible
        return open_dashboard("http://localhost:9090", "Prometheus"), None
    elif deployment_type == "kubernetes":
        # For Kubernetes, set up port forwarding
        port_forward_process = setup_port_forwarding("prometheus", 9090, namespace)
        if port_forward_process:
            return open_dashboard("http://localhost:9090", "Prometheus"), port_forward_process
        return False, None
    return False, None

def access_loki(deployment_type, namespace="negative-space-imaging"):
    """
    Access the Loki dashboard (via Grafana)

    Args:
        deployment_type: The type of deployment ('docker-compose' or 'kubernetes')
        namespace: The Kubernetes namespace (for Kubernetes deployment)

    Returns:
        tuple: (bool for success, process for port forwarding if applicable)
    """
    # For both deployment types, Loki is accessed through Grafana
    success, process = access_grafana(deployment_type, namespace)
    if success:
        print("\nTo access Loki logs:")
        print("1. Log in to Grafana")
        print("2. Go to Explore")
        print("3. Select 'Loki' as the data source")
        print("4. Start querying logs")
    return success, process

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Negative Space Imaging Project Monitoring Dashboard Access")

    parser.add_argument("--dashboard", "-d", choices=["grafana", "prometheus", "loki", "all"],
                        default="all", help="The dashboard to access")

    parser.add_argument("--namespace", "-n", default="negative-space-imaging",
                        help="Kubernetes namespace (for Kubernetes deployment)")

    parser.add_argument("--deployment-type", "-t", choices=["docker-compose", "kubernetes"],
                        help="Manually specify the deployment type")

    args = parser.parse_args()

    # Determine deployment type if not specified
    deployment_type = args.deployment_type
    if not deployment_type:
        deployment_type = check_deployment_type()
        if not deployment_type:
            logger.error("No active deployment detected and no deployment type specified")
            return 1

    # Track port forwarding processes
    port_forward_processes = []

    try:
        # Access requested dashboards
        if args.dashboard in ["grafana", "all"]:
            success, process = access_grafana(deployment_type, args.namespace)
            if process:
                port_forward_processes.append(process)

        if args.dashboard in ["prometheus", "all"]:
            success, process = access_prometheus(deployment_type, args.namespace)
            if process:
                port_forward_processes.append(process)

        if args.dashboard in ["loki", "all"]:
            success, process = access_loki(deployment_type, args.namespace)
            if process:
                port_forward_processes.append(process)

        # If using Kubernetes with port forwarding, keep the script running
        if deployment_type == "kubernetes" and port_forward_processes:
            print("\nPort forwarding is active. Press Ctrl+C to stop.")
            try:
                # Keep the script running
                while all(p.poll() is None for p in port_forward_processes):
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping port forwarding...")
            finally:
                # Clean up port forwarding processes
                for process in port_forward_processes:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        # Clean up port forwarding processes
        for process in port_forward_processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
