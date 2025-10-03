#!/usr/bin/env python3
"""
Multi-Node Deployment Manager for Negative Space Imaging Project
Handles automated deployment and configuration of multi-node clusters
"""

import os
import sys
import time
import logging
import argparse
import yaml
import json
import subprocess
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("multi_node_deployment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiNodeDeploymentManager:
    """Manages the deployment of Negative Space Imaging system across multiple nodes"""

    def __init__(self, config_path: str = "deployment/multi_node_config.yaml"):
        """
        Initialize the deployment manager

        Args:
            config_path: Path to the multi-node configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.cluster_name = self.config["cluster"]["name"]
        self.environment = self.config["cluster"]["environment"]
        self.orchestration = self.config["cluster"]["orchestration"]

        # Set up paths
        self.base_dir = Path.cwd()
        self.deployment_dir = self.base_dir / "deployment"
        self.template_dir = self.deployment_dir / "templates"
        self.output_dir = self.deployment_dir / "output"

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize deployment status
        self.deployment_status = {
            "state": "initialized",
            "progress": 0,
            "last_action": "initialized",
            "errors": [],
            "start_time": time.time(),
            "end_time": None
        }

        logger.info(f"Deployment manager initialized for cluster '{self.cluster_name}' using {self.orchestration}")

    def _load_config(self) -> Dict[str, Any]:
        """Load the multi-node configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            sys.exit(1)

    def _update_status(self, state: str, progress: int, action: str, error: Optional[str] = None):
        """Update the deployment status"""
        self.deployment_status["state"] = state
        self.deployment_status["progress"] = progress
        self.deployment_status["last_action"] = action

        if error:
            self.deployment_status["errors"].append({
                "time": time.time(),
                "message": error
            })

        # Save status to file
        with open(self.output_dir / "deployment_status.json", 'w') as f:
            json.dump(self.deployment_status, f, indent=2)

        logger.info(f"Deployment status: {state} ({progress}%) - {action}")

    def _run_command(self, command: List[str], check: bool = True) -> Tuple[int, str, str]:
        """Run a shell command and return the exit code, stdout, and stderr"""
        logger.debug(f"Running command: {' '.join(command)}")
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            exit_code = process.returncode

            if exit_code != 0 and check:
                logger.error(f"Command failed with exit code {exit_code}: {stderr}")
                self._update_status("error", self.deployment_status["progress"], f"Command failed: {' '.join(command)}", stderr)

            return exit_code, stdout, stderr
        except Exception as e:
            logger.error(f"Failed to run command: {str(e)}")
            self._update_status("error", self.deployment_status["progress"], f"Exception running command: {' '.join(command)}", str(e))
            return 1, "", str(e)

    def _render_template(self, template_name: str, context: Dict[str, Any], output_name: str) -> Path:
        """Render a template with the provided context and save to output directory"""
        try:
            template_path = self.template_dir / template_name
            output_path = self.output_dir / output_name

            with open(template_path, 'r') as f:
                template_content = f.read()

            # Simple template rendering (replace placeholders with context values)
            for key, value in context.items():
                placeholder = f"{{{{ {key} }}}}"
                if isinstance(value, str):
                    template_content = template_content.replace(placeholder, value)
                else:
                    template_content = template_content.replace(placeholder, json.dumps(value))

            with open(output_path, 'w') as f:
                f.write(template_content)

            logger.info(f"Rendered template {template_name} to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to render template: {str(e)}")
            self._update_status("error", self.deployment_status["progress"], f"Template rendering failed: {template_name}", str(e))
            raise

    def setup_kubernetes(self):
        """Set up a Kubernetes cluster"""
        self._update_status("running", 10, "Setting up Kubernetes cluster")

        # Generate Kubernetes YAML files from config
        self._generate_kubernetes_files()

        # Check if cluster already exists
        exit_code, stdout, _ = self._run_command(["kubectl", "get", "cluster", self.cluster_name], check=False)
        if exit_code == 0:
            logger.info(f"Cluster {self.cluster_name} already exists, updating configuration")
            self._update_status("running", 20, "Updating existing cluster")
        else:
            logger.info(f"Creating new cluster {self.cluster_name}")
            self._update_status("running", 20, "Creating new cluster")

            # Create the cluster
            cluster_config_path = self.output_dir / "cluster.yaml"
            exit_code, _, stderr = self._run_command(["kubectl", "apply", "-f", str(cluster_config_path)])
            if exit_code != 0:
                logger.error(f"Failed to create cluster: {stderr}")
                self._update_status("error", 20, "Cluster creation failed", stderr)
                return False

        self._update_status("running", 30, "Deploying node configuration")

        # Deploy node configuration
        node_config_path = self.output_dir / "nodes.yaml"
        exit_code, _, stderr = self._run_command(["kubectl", "apply", "-f", str(node_config_path)])
        if exit_code != 0:
            logger.error(f"Failed to deploy node configuration: {stderr}")
            self._update_status("error", 30, "Node configuration deployment failed", stderr)
            return False

        # Wait for nodes to be ready
        self._update_status("running", 40, "Waiting for nodes to be ready")
        logger.info("Waiting for nodes to be ready")
        time.sleep(60)  # Give nodes some time to start up

        # Check node status
        exit_code, stdout, _ = self._run_command(["kubectl", "get", "nodes", "-o", "json"])
        if exit_code != 0:
            logger.error("Failed to get node status")
            self._update_status("error", 40, "Node status check failed")
            return False

        # Deploy storage configuration
        self._update_status("running", 50, "Deploying storage configuration")
        storage_config_path = self.output_dir / "storage.yaml"
        exit_code, _, stderr = self._run_command(["kubectl", "apply", "-f", str(storage_config_path)])
        if exit_code != 0:
            logger.error(f"Failed to deploy storage configuration: {stderr}")
            self._update_status("error", 50, "Storage configuration deployment failed", stderr)
            return False

        # Deploy networking configuration
        self._update_status("running", 60, "Deploying networking configuration")
        network_config_path = self.output_dir / "networking.yaml"
        exit_code, _, stderr = self._run_command(["kubectl", "apply", "-f", str(network_config_path)])
        if exit_code != 0:
            logger.error(f"Failed to deploy networking configuration: {stderr}")
            self._update_status("error", 60, "Networking configuration deployment failed", stderr)
            return False

        # Deploy services
        self._update_status("running", 70, "Deploying services")
        services_config_path = self.output_dir / "services.yaml"
        exit_code, _, stderr = self._run_command(["kubectl", "apply", "-f", str(services_config_path)])
        if exit_code != 0:
            logger.error(f"Failed to deploy services: {stderr}")
            self._update_status("error", 70, "Services deployment failed", stderr)
            return False

        # Deploy monitoring configuration
        self._update_status("running", 80, "Deploying monitoring configuration")
        monitoring_config_path = self.output_dir / "monitoring.yaml"
        exit_code, _, stderr = self._run_command(["kubectl", "apply", "-f", str(monitoring_config_path)])
        if exit_code != 0:
            logger.error(f"Failed to deploy monitoring configuration: {stderr}")
            self._update_status("error", 80, "Monitoring configuration deployment failed", stderr)
            return False

        # Check deployment status
        self._update_status("running", 90, "Checking deployment status")
        exit_code, stdout, _ = self._run_command(["kubectl", "get", "pods", "--all-namespaces"])
        if exit_code != 0:
            logger.error("Failed to check pod status")
            self._update_status("error", 90, "Pod status check failed")
            return False

        # Update deployment status
        self._update_status("completed", 100, "Kubernetes cluster deployment completed")
        self.deployment_status["end_time"] = time.time()

        # Save final status
        with open(self.output_dir / "deployment_status.json", 'w') as f:
            json.dump(self.deployment_status, f, indent=2)

        logger.info("Kubernetes cluster deployment completed successfully")
        return True

    def _generate_kubernetes_files(self):
        """Generate Kubernetes YAML files from configuration"""
        # Generate cluster configuration
        cluster_context = {
            "name": self.cluster_name,
            "environment": self.environment,
            "version": self.config.get("version", "1.0")
        }
        self._render_template("cluster.yaml.tmpl", cluster_context, "cluster.yaml")

        # Generate node configuration
        node_context = {
            "cluster_name": self.cluster_name,
            "nodes": self.config["nodes"],
            "auto_scaling": self.config["cluster"]["auto_scaling"]
        }
        self._render_template("nodes.yaml.tmpl", node_context, "nodes.yaml")

        # Generate storage configuration
        storage_context = {
            "cluster_name": self.cluster_name,
            "storage": self.config["storage"]
        }
        self._render_template("storage.yaml.tmpl", storage_context, "storage.yaml")

        # Generate networking configuration
        network_context = {
            "cluster_name": self.cluster_name,
            "networking": self.config["networking"]
        }
        self._render_template("networking.yaml.tmpl", network_context, "networking.yaml")

        # Generate services configuration
        services_context = {
            "cluster_name": self.cluster_name,
            "services": self.config["services"]
        }
        self._render_template("services.yaml.tmpl", services_context, "services.yaml")

        # Generate monitoring configuration
        monitoring_context = {
            "cluster_name": self.cluster_name,
            "monitoring": self.config["monitoring"],
            "logging": self.config["logging"]
        }
        self._render_template("monitoring.yaml.tmpl", monitoring_context, "monitoring.yaml")

    def setup_docker_swarm(self):
        """Set up a Docker Swarm cluster"""
        self._update_status("running", 10, "Setting up Docker Swarm cluster")

        # Check if swarm already exists
        exit_code, stdout, _ = self._run_command(["docker", "info", "--format", "{{.Swarm.LocalNodeState}}"])
        if exit_code != 0:
            logger.error("Failed to get Docker Swarm status")
            self._update_status("error", 10, "Docker Swarm status check failed")
            return False

        is_swarm_active = "active" in stdout.strip()

        if is_swarm_active:
            logger.info("Docker Swarm already initialized")
        else:
            # Initialize swarm
            logger.info("Initializing Docker Swarm")
            exit_code, _, stderr = self._run_command(["docker", "swarm", "init"])
            if exit_code != 0:
                logger.error(f"Failed to initialize Docker Swarm: {stderr}")
                self._update_status("error", 20, "Docker Swarm initialization failed", stderr)
                return False

        self._update_status("running", 30, "Creating Docker networks")

        # Create overlay networks
        networks = self.config["networking"].get("networks", ["frontend", "backend", "data"])
        for network in networks:
            exit_code, _, stderr = self._run_command(
                ["docker", "network", "create", "--driver", "overlay", network]
            )
            if exit_code != 0 and "already exists" not in stderr:
                logger.error(f"Failed to create network {network}: {stderr}")
                self._update_status("error", 30, f"Network creation failed: {network}", stderr)
                return False

        self._update_status("running", 40, "Creating Docker volumes")

        # Create volumes
        for storage_name, storage_config in self.config["storage"].items():
            if storage_name != "persistence":
                volume_name = f"{self.cluster_name}-{storage_name}"
                exit_code, _, stderr = self._run_command(
                    ["docker", "volume", "create", volume_name]
                )
                if exit_code != 0 and "already exists" not in stderr:
                    logger.error(f"Failed to create volume {volume_name}: {stderr}")
                    self._update_status("error", 40, f"Volume creation failed: {volume_name}", stderr)
                    return False

        self._update_status("running", 50, "Deploying services")

        # Generate docker-compose file
        compose_file = self._generate_docker_compose()

        # Deploy stack
        exit_code, _, stderr = self._run_command(
            ["docker", "stack", "deploy", "-c", str(compose_file), self.cluster_name]
        )
        if exit_code != 0:
            logger.error(f"Failed to deploy stack: {stderr}")
            self._update_status("error", 50, "Stack deployment failed", stderr)
            return False

        # Wait for services to start
        self._update_status("running", 70, "Waiting for services to start")
        logger.info("Waiting for services to start")
        time.sleep(30)

        # Check service status
        exit_code, stdout, _ = self._run_command(["docker", "service", "ls", "--format", "{{.Name}}: {{.Replicas}}"])
        if exit_code != 0:
            logger.error("Failed to check service status")
            self._update_status("error", 80, "Service status check failed")
            return False

        logger.info(f"Service status:\n{stdout}")

        # Update deployment status
        self._update_status("completed", 100, "Docker Swarm deployment completed")
        self.deployment_status["end_time"] = time.time()

        # Save final status
        with open(self.output_dir / "deployment_status.json", 'w') as f:
            json.dump(self.deployment_status, f, indent=2)

        logger.info("Docker Swarm deployment completed successfully")
        return True

    def _generate_docker_compose(self) -> Path:
        """Generate a docker-compose file for the cluster"""
        compose_data = {
            "version": "3.8",
            "services": {},
            "networks": {},
            "volumes": {}
        }

        # Add services
        for service_name, service_config in self.config["services"].items():
            compose_service = {
                "image": service_config["image"],
                "deploy": {
                    "replicas": service_config["replicas"],
                    "resources": {
                        "limits": service_config["resources"]["limits"],
                        "reservations": service_config["resources"]["requests"]
                    }
                },
                "environment": []
            }

            # Add environment variables
            for env_var in service_config.get("environment_variables", []):
                compose_service["environment"].append(f"{env_var['name']}={env_var['value']}")

            # Add health check if defined
            if "health_check" in service_config:
                health = service_config["health_check"]
                compose_service["healthcheck"] = {
                    "test": f"curl -f http://localhost:{health['port']}{health['path']} || exit 1",
                    "interval": f"{health['period_seconds']}s",
                    "timeout": "10s",
                    "retries": 3,
                    "start_period": f"{health['initial_delay_seconds']}s"
                }

            # Add volume mounts if defined
            if "volume_mounts" in service_config:
                compose_service["volumes"] = []
                for volume in service_config["volume_mounts"]:
                    volume_name = f"{self.cluster_name}-{volume['name']}"
                    compose_service["volumes"].append(f"{volume_name}:{volume['mount_path']}")

            # Add service to compose file
            compose_data["services"][service_name] = compose_service

        # Add networks
        networks = self.config["networking"].get("networks", ["frontend", "backend", "data"])
        for network in networks:
            compose_data["networks"][network] = {
                "driver": "overlay",
                "attachable": True
            }

        # Add volumes
        for storage_name, storage_config in self.config["storage"].items():
            if storage_name != "persistence":
                volume_name = f"{self.cluster_name}-{storage_name}"
                compose_data["volumes"][volume_name] = {
                    "driver": "local"
                }

        # Write docker-compose file
        compose_file = self.output_dir / "docker-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(compose_data, f, default_flow_style=False)

        logger.info(f"Generated docker-compose file at {compose_file}")
        return compose_file

    def deploy(self):
        """Deploy the multi-node cluster"""
        logger.info(f"Starting deployment of {self.cluster_name} in {self.environment} environment")
        self._update_status("running", 5, "Starting deployment")

        # Check prerequisites
        if not self._check_prerequisites():
            logger.error("Prerequisites check failed")
            self._update_status("error", 5, "Prerequisites check failed")
            return False

        # Deploy based on orchestration type
        if self.orchestration == "kubernetes":
            return self.setup_kubernetes()
        elif self.orchestration == "docker-swarm":
            return self.setup_docker_swarm()
        else:
            logger.error(f"Unsupported orchestration type: {self.orchestration}")
            self._update_status("error", 5, f"Unsupported orchestration type: {self.orchestration}")
            return False

    def _check_prerequisites(self) -> bool:
        """Check if all prerequisites for deployment are met"""
        self._update_status("running", 5, "Checking prerequisites")

        # Check if configuration is valid
        if not self._validate_config():
            logger.error("Configuration validation failed")
            return False

        # Check for required tools
        if self.orchestration == "kubernetes":
            tools = ["kubectl"]
        elif self.orchestration == "docker-swarm":
            tools = ["docker"]
        else:
            tools = []

        for tool in tools:
            exit_code, _, _ = self._run_command([tool, "--version"], check=False)
            if exit_code != 0:
                logger.error(f"Required tool not found: {tool}")
                self._update_status("error", 5, f"Required tool not found: {tool}")
                return False

        # Check if template directory exists
        if not self.template_dir.exists():
            logger.error(f"Template directory not found: {self.template_dir}")
            self._update_status("error", 5, "Template directory not found")
            return False

        return True

    def _validate_config(self) -> bool:
        """Validate the configuration file"""
        required_sections = ["cluster", "nodes", "networking", "services", "storage", "monitoring", "logging"]

        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required section in config: {section}")
                self._update_status("error", 5, f"Missing required section in config: {section}")
                return False

        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Multi-Node Deployment Manager")
    parser.add_argument("--config", "-c", default="deployment/multi_node_config.yaml", help="Path to configuration file")
    parser.add_argument("--templates", "-t", default=None, help="Path to template directory")
    args = parser.parse_args()

    try:
        # Create deployment manager
        deployment_manager = MultiNodeDeploymentManager(args.config)

        # Start deployment
        success = deployment_manager.deploy()

        if success:
            logger.info("Deployment completed successfully")
            return 0
        else:
            logger.error("Deployment failed")
            return 1
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
