#!/usr/bin/env python3
"""
Deployment Verification Script for Negative Space Imaging Project

This script verifies configuration consistency across services and checks for
potential issues in the deployment configuration.
"""

import argparse
import json
import os
import sys
import logging
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("verify")

class DeploymentVerifier:
    """Class to verify deployment configuration consistency"""

    def __init__(self, deployment_dir=None):
        """
        Initialize deployment verifier

        Args:
            deployment_dir: Path to the deployment directory
        """
        # Set deployment directory
        if deployment_dir is None:
            # Assume script is in the deployment directory
            self.deployment_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.deployment_dir = Path(deployment_dir)

        # Initialize verification results
        self.results = {
            "docker_compose": {
                "verified": False,
                "issues": []
            },
            "kubernetes": {
                "verified": False,
                "issues": []
            },
            "monitoring": {
                "verified": False,
                "issues": []
            },
            "logging": {
                "verified": False,
                "issues": []
            },
            "database": {
                "verified": False,
                "issues": []
            }
        }

        # Initialize data structures
        self.docker_compose_config = None
        self.kubernetes_manifests = []
        self.monitoring_config = {}
        self.logging_config = {}
        self.database_config = {}

    def load_docker_compose(self):
        """Load Docker Compose configuration"""
        docker_compose_path = self.deployment_dir / "docker-compose.yaml"
        if not docker_compose_path.exists():
            logger.warning(f"Docker Compose file not found: {docker_compose_path}")
            self.results["docker_compose"]["issues"].append(
                "Docker Compose file not found"
            )
            return False

        try:
            with open(docker_compose_path, 'r') as f:
                self.docker_compose_config = yaml.safe_load(f)
            logger.info("Docker Compose configuration loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load Docker Compose config: {str(e)}")
            self.results["docker_compose"]["issues"].append(
                f"Failed to load Docker Compose config: {str(e)}"
            )
            return False

    def load_kubernetes_manifests(self):
        """Load Kubernetes manifests"""
        k8s_dir = self.deployment_dir / "kubernetes"
        if not k8s_dir.exists() or not k8s_dir.is_dir():
            logger.warning(f"Kubernetes directory not found: {k8s_dir}")
            self.results["kubernetes"]["issues"].append(
                "Kubernetes directory not found"
            )
            return False

        try:
            # Load all YAML files in the kubernetes directory
            for yaml_file in k8s_dir.glob("**/*.yaml"):
                try:
                    with open(yaml_file, 'r') as f:
                        # Split YAML files with multiple documents
                        docs = list(yaml.safe_load_all(f))
                        for doc in docs:
                            if doc:  # Skip empty documents
                                self.kubernetes_manifests.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to load Kubernetes manifest {yaml_file}: {str(e)}")
                    self.results["kubernetes"]["issues"].append(
                        f"Failed to load {yaml_file.name}: {str(e)}"
                    )

            logger.info(f"Loaded {len(self.kubernetes_manifests)} Kubernetes manifests")
            return len(self.kubernetes_manifests) > 0
        except Exception as e:
            logger.error(f"Failed to load Kubernetes manifests: {str(e)}")
            self.results["kubernetes"]["issues"].append(
                f"Failed to load Kubernetes manifests: {str(e)}"
            )
            return False

    def load_monitoring_config(self):
        """Load monitoring configuration"""
        monitoring_dir = self.deployment_dir / "monitoring"
        if not monitoring_dir.exists() or not monitoring_dir.is_dir():
            logger.warning(f"Monitoring directory not found: {monitoring_dir}")
            self.results["monitoring"]["issues"].append(
                "Monitoring directory not found"
            )
            return False

        try:
            # Load Prometheus config
            prometheus_config_path = monitoring_dir / "prometheus.yml"
            if prometheus_config_path.exists():
                with open(prometheus_config_path, 'r') as f:
                    self.monitoring_config["prometheus"] = yaml.safe_load(f)

            # Load Grafana datasources
            datasources_path = monitoring_dir / "datasources" / "datasources.yaml"
            if datasources_path.exists():
                with open(datasources_path, 'r') as f:
                    self.monitoring_config["datasources"] = yaml.safe_load(f)

            # Load dashboard providers
            dashboard_provider_path = monitoring_dir / "dashboard-providers" / "dashboard-provider.yaml"
            if dashboard_provider_path.exists():
                with open(dashboard_provider_path, 'r') as f:
                    self.monitoring_config["dashboard_provider"] = yaml.safe_load(f)

            # Check if we loaded any monitoring config
            if not self.monitoring_config:
                logger.warning("No monitoring configuration found")
                self.results["monitoring"]["issues"].append(
                    "No monitoring configuration found"
                )
                return False

            logger.info("Monitoring configuration loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load monitoring config: {str(e)}")
            self.results["monitoring"]["issues"].append(
                f"Failed to load monitoring config: {str(e)}"
            )
            return False

    def load_logging_config(self):
        """Load logging configuration"""
        logging_dir = self.deployment_dir / "logging"
        if not logging_dir.exists() or not logging_dir.is_dir():
            logger.warning(f"Logging directory not found: {logging_dir}")
            self.results["logging"]["issues"].append(
                "Logging directory not found"
            )
            return False

        try:
            # Load Loki config
            loki_config_path = logging_dir / "loki-config.yaml"
            if loki_config_path.exists():
                with open(loki_config_path, 'r') as f:
                    self.logging_config["loki"] = yaml.safe_load(f)

            # Load Promtail config
            promtail_config_path = logging_dir / "promtail-config.yaml"
            if promtail_config_path.exists():
                with open(promtail_config_path, 'r') as f:
                    self.logging_config["promtail"] = yaml.safe_load(f)

            # Check if we loaded any logging config
            if not self.logging_config:
                logger.warning("No logging configuration found")
                self.results["logging"]["issues"].append(
                    "No logging configuration found"
                )
                return False

            logger.info("Logging configuration loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load logging config: {str(e)}")
            self.results["logging"]["issues"].append(
                f"Failed to load logging config: {str(e)}"
            )
            return False

    def load_database_config(self):
        """Load database configuration"""
        database_dir = self.deployment_dir / "database"
        if not database_dir.exists() or not database_dir.is_dir():
            logger.warning(f"Database directory not found: {database_dir}")
            self.results["database"]["issues"].append(
                "Database directory not found"
            )
            return False

        try:
            # Check for SQL init scripts
            self.database_config["init_scripts"] = []
            for sql_file in sorted(database_dir.glob("*.sql")):
                self.database_config["init_scripts"].append(sql_file.name)

            # Check for database init script
            init_script_path = database_dir / "init-database.sh"
            if init_script_path.exists():
                self.database_config["init_script"] = True
            else:
                self.database_config["init_script"] = False
                self.results["database"]["issues"].append(
                    "Database initialization script not found"
                )

            # Check if we found any database files
            if not self.database_config["init_scripts"] and not self.database_config["init_script"]:
                logger.warning("No database configuration found")
                self.results["database"]["issues"].append(
                    "No database configuration found"
                )
                return False

            logger.info("Database configuration loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load database config: {str(e)}")
            self.results["database"]["issues"].append(
                f"Failed to load database config: {str(e)}"
            )
            return False

    def verify_docker_compose(self):
        """Verify Docker Compose configuration"""
        if not self.docker_compose_config:
            return False

        try:
            # Check for version
            if "version" not in self.docker_compose_config:
                self.results["docker_compose"]["issues"].append(
                    "Docker Compose version not specified"
                )

            # Check for services
            if "services" not in self.docker_compose_config:
                self.results["docker_compose"]["issues"].append(
                    "No services defined in Docker Compose"
                )
                return False

            services = self.docker_compose_config["services"]

            # Check for required services
            required_services = ["imaging-service", "database", "prometheus", "grafana"]
            for service in required_services:
                if service not in services:
                    self.results["docker_compose"]["issues"].append(
                        f"Required service '{service}' not defined"
                    )

            # Check for service dependencies
            for service_name, service_config in services.items():
                # Check for image or build
                if "image" not in service_config and "build" not in service_config:
                    self.results["docker_compose"]["issues"].append(
                        f"Service '{service_name}' has no image or build specified"
                    )

                # Check for depends_on consistency
                if "depends_on" in service_config:
                    for dependency in service_config["depends_on"]:
                        if dependency not in services:
                            self.results["docker_compose"]["issues"].append(
                                f"Service '{service_name}' depends on non-existent service '{dependency}'"
                            )

            # Check for networks
            if "networks" not in self.docker_compose_config:
                self.results["docker_compose"]["issues"].append(
                    "No networks defined in Docker Compose"
                )

            # Check for volumes
            if "volumes" not in self.docker_compose_config:
                self.results["docker_compose"]["issues"].append(
                    "No volumes defined in Docker Compose"
                )

            # Docker Compose is verified if there are no issues
            self.results["docker_compose"]["verified"] = len(self.results["docker_compose"]["issues"]) == 0
            return self.results["docker_compose"]["verified"]
        except Exception as e:
            logger.error(f"Failed to verify Docker Compose config: {str(e)}")
            self.results["docker_compose"]["issues"].append(
                f"Failed to verify Docker Compose config: {str(e)}"
            )
            return False

    def verify_kubernetes(self):
        """Verify Kubernetes manifests"""
        if not self.kubernetes_manifests:
            return False

        try:
            # Check for required resources
            required_resources = {
                "Deployment": set(["imaging-service", "database", "prometheus", "grafana"]),
                "Service": set(["imaging-service", "database", "prometheus", "grafana"]),
                "ConfigMap": set(),
                "Secret": set(),
                "PersistentVolumeClaim": set()
            }

            found_resources = {
                "Deployment": set(),
                "Service": set(),
                "ConfigMap": set(),
                "Secret": set(),
                "PersistentVolumeClaim": set()
            }

            # Process all manifests
            for manifest in self.kubernetes_manifests:
                if not manifest:
                    continue

                kind = manifest.get("kind")
                if not kind:
                    self.results["kubernetes"]["issues"].append(
                        "Manifest missing 'kind' field"
                    )
                    continue

                # Extract name
                metadata = manifest.get("metadata", {})
                name = metadata.get("name", "")

                # Add to found resources
                if kind in found_resources:
                    found_resources[kind].add(name)

            # Check for missing required resources
            for kind, required_names in required_resources.items():
                found_names = found_resources.get(kind, set())
                missing_names = required_names - found_names

                for name in missing_names:
                    self.results["kubernetes"]["issues"].append(
                        f"Required {kind} '{name}' not found"
                    )

            # Check namespace consistency
            namespaces = set()
            for manifest in self.kubernetes_manifests:
                if not manifest:
                    continue

                metadata = manifest.get("metadata", {})
                namespace = metadata.get("namespace")

                if namespace:
                    namespaces.add(namespace)

            if len(namespaces) > 1:
                self.results["kubernetes"]["issues"].append(
                    f"Multiple namespaces found: {', '.join(namespaces)}"
                )

            # Kubernetes is verified if there are no issues
            self.results["kubernetes"]["verified"] = len(self.results["kubernetes"]["issues"]) == 0
            return self.results["kubernetes"]["verified"]
        except Exception as e:
            logger.error(f"Failed to verify Kubernetes manifests: {str(e)}")
            self.results["kubernetes"]["issues"].append(
                f"Failed to verify Kubernetes manifests: {str(e)}"
            )
            return False

    def verify_monitoring(self):
        """Verify monitoring configuration"""
        if not self.monitoring_config:
            return False

        try:
            # Check Prometheus config
            prometheus_config = self.monitoring_config.get("prometheus")
            if prometheus_config:
                # Check for scrape configs
                scrape_configs = prometheus_config.get("scrape_configs", [])
                if not scrape_configs:
                    self.results["monitoring"]["issues"].append(
                        "No scrape_configs in Prometheus configuration"
                    )

                # Check for required scrape targets
                required_targets = ["prometheus", "imaging-service"]
                found_targets = set()

                for config in scrape_configs:
                    job_name = config.get("job_name", "")
                    found_targets.add(job_name)

                for target in required_targets:
                    if target not in found_targets:
                        self.results["monitoring"]["issues"].append(
                            f"Required Prometheus target '{target}' not configured"
                        )
            else:
                self.results["monitoring"]["issues"].append(
                    "Prometheus configuration not found"
                )

            # Check Grafana datasources
            datasources_config = self.monitoring_config.get("datasources")
            if datasources_config:
                datasources = datasources_config.get("datasources", [])
                if not datasources:
                    self.results["monitoring"]["issues"].append(
                        "No datasources in Grafana configuration"
                    )

                # Check for Prometheus datasource
                found_prometheus = False
                for datasource in datasources:
                    if datasource.get("type") == "prometheus":
                        found_prometheus = True
                        break

                if not found_prometheus:
                    self.results["monitoring"]["issues"].append(
                        "Prometheus datasource not configured in Grafana"
                    )
            else:
                self.results["monitoring"]["issues"].append(
                    "Grafana datasources configuration not found"
                )

            # Monitoring is verified if there are no issues
            self.results["monitoring"]["verified"] = len(self.results["monitoring"]["issues"]) == 0
            return self.results["monitoring"]["verified"]
        except Exception as e:
            logger.error(f"Failed to verify monitoring config: {str(e)}")
            self.results["monitoring"]["issues"].append(
                f"Failed to verify monitoring config: {str(e)}"
            )
            return False

    def verify_logging(self):
        """Verify logging configuration"""
        if not self.logging_config:
            return False

        try:
            # Check Loki config
            loki_config = self.logging_config.get("loki")
            if not loki_config:
                self.results["logging"]["issues"].append(
                    "Loki configuration not found"
                )

            # Check Promtail config
            promtail_config = self.logging_config.get("promtail")
            if promtail_config:
                # Check for clients (Loki connection)
                clients = promtail_config.get("clients", [])
                if not clients:
                    self.results["logging"]["issues"].append(
                        "No clients configured in Promtail"
                    )

                # Check for scrape configs
                scrape_configs = promtail_config.get("scrape_configs", [])
                if not scrape_configs:
                    self.results["logging"]["issues"].append(
                        "No scrape_configs in Promtail configuration"
                    )
            else:
                self.results["logging"]["issues"].append(
                    "Promtail configuration not found"
                )

            # Logging is verified if there are no issues
            self.results["logging"]["verified"] = len(self.results["logging"]["issues"]) == 0
            return self.results["logging"]["verified"]
        except Exception as e:
            logger.error(f"Failed to verify logging config: {str(e)}")
            self.results["logging"]["issues"].append(
                f"Failed to verify logging config: {str(e)}"
            )
            return False

    def verify_database(self):
        """Verify database configuration"""
        if not self.database_config:
            return False

        try:
            # Check for init scripts
            if not self.database_config.get("init_scripts"):
                self.results["database"]["issues"].append(
                    "No database initialization SQL scripts found"
                )

            # Check for init script
            if not self.database_config.get("init_script"):
                self.results["database"]["issues"].append(
                    "Database initialization script not found"
                )

            # Database is verified if there are no issues
            self.results["database"]["verified"] = len(self.results["database"]["issues"]) == 0
            return self.results["database"]["verified"]
        except Exception as e:
            logger.error(f"Failed to verify database config: {str(e)}")
            self.results["database"]["issues"].append(
                f"Failed to verify database config: {str(e)}"
            )
            return False

    def verify_deployment(self):
        """Verify all deployment components"""
        # Load configurations
        self.load_docker_compose()
        self.load_kubernetes_manifests()
        self.load_monitoring_config()
        self.load_logging_config()
        self.load_database_config()

        # Verify configurations
        self.verify_docker_compose()
        self.verify_kubernetes()
        self.verify_monitoring()
        self.verify_logging()
        self.verify_database()

        # Check cross-component consistency
        self._check_cross_component_consistency()

        # Return overall verification result
        return all([
            self.results["docker_compose"]["verified"],
            self.results["kubernetes"]["verified"],
            self.results["monitoring"]["verified"],
            self.results["logging"]["verified"],
            self.results["database"]["verified"]
        ])

    def _check_cross_component_consistency(self):
        """Check consistency across components"""
        # Compare services between Docker Compose and Kubernetes
        if self.docker_compose_config and self.kubernetes_manifests:
            docker_services = set(self.docker_compose_config.get("services", {}).keys())

            k8s_services = set()
            for manifest in self.kubernetes_manifests:
                if manifest and manifest.get("kind") == "Service":
                    name = manifest.get("metadata", {}).get("name", "")
                    if name:
                        k8s_services.add(name)

            # Find services in Docker Compose but not in Kubernetes
            missing_in_k8s = docker_services - k8s_services
            for service in missing_in_k8s:
                self.results["kubernetes"]["issues"].append(
                    f"Service '{service}' defined in Docker Compose but missing in Kubernetes"
                )

        # Check monitoring consistency
        if self.docker_compose_config and self.monitoring_config.get("prometheus"):
            docker_services = set(self.docker_compose_config.get("services", {}).keys())

            # Get services being monitored
            monitored_services = set()
            for config in self.monitoring_config["prometheus"].get("scrape_configs", []):
                job_name = config.get("job_name", "")
                if job_name:
                    monitored_services.add(job_name)

            # Check for important services that should be monitored
            important_services = {"imaging-service", "database"}
            for service in important_services:
                if service in docker_services and service not in monitored_services:
                    self.results["monitoring"]["issues"].append(
                        f"Important service '{service}' is not being monitored by Prometheus"
                    )

    def display_results(self):
        """Display verification results"""
        print("\n=== Negative Space Imaging Project Deployment Verification ===")

        overall_result = all([
            self.results["docker_compose"]["verified"],
            self.results["kubernetes"]["verified"],
            self.results["monitoring"]["verified"],
            self.results["logging"]["verified"],
            self.results["database"]["verified"]
        ])

        if overall_result:
            print("\n✅ Overall Verification: PASSED")
        else:
            print("\n❌ Overall Verification: FAILED")

        for component, result in self.results.items():
            if result["verified"]:
                print(f"\n✅ {component.replace('_', ' ').title()}: VERIFIED")
            else:
                print(f"\n❌ {component.replace('_', ' ').title()}: NOT VERIFIED")

                if result["issues"]:
                    print("   Issues:")
                    for issue in result["issues"]:
                        print(f"   - {issue}")

    def export_results(self, file_path):
        """
        Export verification results to a file

        Args:
            file_path: The path to save the results to

        Returns:
            bool: Whether the export was successful
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.results, f, indent=2)

            logger.info(f"Results exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")
            return False

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Negative Space Imaging Project Deployment Verification"
    )

    parser.add_argument("--deployment-dir", "-d",
                        help="Path to the deployment directory")

    parser.add_argument("--export", "-e",
                        help="Export the verification results to a file")

    args = parser.parse_args()

    # Run verification
    verifier = DeploymentVerifier(args.deployment_dir)
    verifier.verify_deployment()
    verifier.display_results()

    if args.export:
        verifier.export_results(args.export)

    # Return exit code based on verification result
    overall_result = all([
        verifier.results["docker_compose"]["verified"],
        verifier.results["kubernetes"]["verified"],
        verifier.results["monitoring"]["verified"],
        verifier.results["logging"]["verified"],
        verifier.results["database"]["verified"]
    ])

    return 0 if overall_result else 1

if __name__ == "__main__":
    sys.exit(main())
