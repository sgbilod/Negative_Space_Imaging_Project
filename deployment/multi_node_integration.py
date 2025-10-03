"""
Multi-Node Integration Module for Negative Space Imaging Project
Integrates multi-node deployment with existing image processing and security components
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
import importlib.util
import subprocess
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("multi_node_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
def import_module_from_path(module_name, file_path):
    """Import a module from file path"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            logger.error(f"Failed to load spec for {module_name} from {file_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Failed to import {module_name} from {file_path}: {str(e)}")
        return None

# Get project root directory
project_root = Path.cwd()

# Import necessary project modules
try:
    # Import security module
    security_module = import_module_from_path("security", project_root / "security_components.py")

    # Import image acquisition module
    image_acquisition_module = import_module_from_path("image_acquisition", project_root / "image_acquisition.py")

    # Import HPC module
    hpc_module = import_module_from_path("hpc", project_root / "hpc_integration.py")

    # Import distributed computing module
    distributed_computing_module = import_module_from_path("distributed_computing", project_root / "distributed_computing.py")

    modules_loaded = all([
        security_module,
        image_acquisition_module,
        hpc_module,
        distributed_computing_module
    ])

    if not modules_loaded:
        logger.warning("Some required modules could not be loaded")
except Exception as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    modules_loaded = False

class MultiNodeIntegration:
    """Integrates multi-node deployment with existing system components"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the multi-node integration

        Args:
            config_path: Path to the multi-node configuration file
        """
        self.project_root = project_root
        self.config_path = config_path or str(self.project_root / "deployment" / "multi_node_config.yaml")
        self.config = self._load_config()

        # Initialize components
        self.security_provider = None
        self.image_processor = None
        self.hpc_manager = None
        self.distributed_computing = None

        if modules_loaded:
            self._initialize_components()

    def _load_config(self) -> Dict[str, Any]:
        """Load the multi-node configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return {}

    def _initialize_components(self):
        """Initialize system components"""
        try:
            # Initialize security provider
            if security_module:
                logger.info("Initializing security provider")
                self.security_provider = security_module.EnhancedSecurityProvider()

            # Initialize image processor
            if image_acquisition_module:
                logger.info("Initializing image processor")
                self.image_processor = image_acquisition_module.ImageAcquisition()

            # Initialize HPC manager
            if hpc_module:
                logger.info("Initializing HPC manager")
                hpc_config_path = str(self.project_root / "hpc_config.yaml")
                self.hpc_manager = hpc_module.HPCIntegration(config_path=hpc_config_path)

            # Initialize distributed computing
            if distributed_computing_module:
                logger.info("Initializing distributed computing")
                self.distributed_computing = distributed_computing_module.DistributedComputing()

            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")

    def adapt_security_config(self):
        """Adapt security configuration for multi-node deployment"""
        if not self.security_provider:
            logger.error("Security provider not initialized")
            return False

        try:
            logger.info("Adapting security configuration for multi-node deployment")

            # Load current security config
            security_config_path = self.project_root / "security_config.json"
            with open(security_config_path, 'r') as f:
                security_config = json.load(f)

            # Update for multi-node deployment
            security_config["distributed_mode"] = True
            security_config["node_verification"] = True
            security_config["inter_node_encryption"] = True

            # Add node-specific security settings
            if "nodes" in self.config:
                node_security = {}

                for node_type in ["master", "compute", "storage", "edge"]:
                    if node_type in self.config["nodes"]:
                        for node in self.config["nodes"][node_type]:
                            node_name = node["name"]
                            node_security[node_name] = {
                                "role": node_type,
                                "access_level": self._get_access_level_for_node_type(node_type),
                                "allowed_operations": self._get_allowed_operations_for_node_type(node_type)
                            }

                security_config["node_security"] = node_security

            # Update security config
            with open(security_config_path, 'w') as f:
                json.dump(security_config, f, indent=2)

            logger.info("Security configuration updated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to adapt security configuration: {str(e)}")
            return False

    def _get_access_level_for_node_type(self, node_type: str) -> str:
        """Get access level for node type"""
        access_levels = {
            "master": "admin",
            "compute": "processor",
            "storage": "storage",
            "edge": "api"
        }
        return access_levels.get(node_type, "restricted")

    def _get_allowed_operations_for_node_type(self, node_type: str) -> List[str]:
        """Get allowed operations for node type"""
        allowed_operations = {
            "master": ["admin", "monitor", "process", "store", "retrieve", "configure"],
            "compute": ["process", "monitor", "retrieve"],
            "storage": ["store", "retrieve", "monitor"],
            "edge": ["api", "monitor", "retrieve"]
        }
        return allowed_operations.get(node_type, ["monitor"])

    def adapt_hpc_config(self):
        """Adapt HPC configuration for multi-node deployment"""
        if not self.hpc_manager:
            logger.error("HPC manager not initialized")
            return False

        try:
            logger.info("Adapting HPC configuration for multi-node deployment")

            # Load current HPC config
            hpc_config_path = self.project_root / "hpc_config.yaml"
            with open(hpc_config_path, 'r') as f:
                hpc_config = yaml.safe_load(f)

            # Update for multi-node deployment
            hpc_config["distributed_mode"] = True

            # Configure compute nodes
            compute_nodes = []
            if "nodes" in self.config and "compute" in self.config["nodes"]:
                for node in self.config["nodes"]["compute"]:
                    compute_node = {
                        "name": node["name"],
                        "address": f"{node['name']}.internal",
                        "resources": {
                            "cpu": node["resources"]["cpu"],
                            "memory": node["resources"]["memory"]
                        }
                    }

                    # Add GPU if available
                    if "gpu" in node["resources"]:
                        compute_node["resources"]["gpu"] = node["resources"]["gpu"]

                    compute_nodes.append(compute_node)

            hpc_config["compute_nodes"] = compute_nodes

            # Configure storage nodes
            storage_nodes = []
            if "nodes" in self.config and "storage" in self.config["nodes"]:
                for node in self.config["nodes"]["storage"]:
                    storage_node = {
                        "name": node["name"],
                        "address": f"{node['name']}.internal",
                        "resources": {
                            "storage": node["resources"]["storage"]
                        }
                    }
                    storage_nodes.append(storage_node)

            hpc_config["storage_nodes"] = storage_nodes

            # Update HPC config
            with open(hpc_config_path, 'w') as f:
                yaml.dump(hpc_config, f, default_flow_style=False)

            logger.info("HPC configuration updated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to adapt HPC configuration: {str(e)}")
            return False

    def adapt_distributed_computing_config(self):
        """Adapt distributed computing configuration for multi-node deployment"""
        if not self.distributed_computing:
            logger.error("Distributed computing not initialized")
            return False

        try:
            logger.info("Adapting distributed computing configuration for multi-node deployment")

            # Create distributed computing config
            distributed_config = {
                "cluster": {
                    "name": self.config["cluster"]["name"],
                    "scheduler_address": f"{self.config['cluster']['name']}-distributed-computing.internal:8787"
                },
                "workers": []
            }

            # Configure workers
            if "nodes" in self.config and "compute" in self.config["nodes"]:
                for node in self.config["nodes"]["compute"]:
                    worker = {
                        "name": f"worker-{node['name']}",
                        "address": f"{node['name']}.internal",
                        "resources": {
                            "cpu": node["resources"]["cpu"],
                            "memory": node["resources"]["memory"]
                        }
                    }

                    # Add GPU if available
                    if "gpu" in node["resources"]:
                        worker["resources"]["gpu"] = node["resources"]["gpu"]

                    distributed_config["workers"].append(worker)

            # Save distributed computing config
            distributed_config_path = self.project_root / "distributed_computing_config.yaml"
            with open(distributed_config_path, 'w') as f:
                yaml.dump(distributed_config, f, default_flow_style=False)

            logger.info("Distributed computing configuration updated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to adapt distributed computing configuration: {str(e)}")
            return False

    def generate_service_definitions(self):
        """Generate service definitions for multi-node deployment"""
        try:
            logger.info("Generating service definitions for multi-node deployment")

            service_definitions = {}

            # Generate API Gateway service
            service_definitions["api_gateway"] = {
                "image": "negativespaceimagingproject/api-gateway:latest",
                "replicas": 2,
                "resources": {
                    "limits": {
                        "cpu": "1",
                        "memory": "2Gi"
                    },
                    "requests": {
                        "cpu": "0.5",
                        "memory": "1Gi"
                    }
                },
                "environment_variables": [
                    {
                        "name": "API_PORT",
                        "value": "8080"
                    },
                    {
                        "name": "LOG_LEVEL",
                        "value": "INFO"
                    },
                    {
                        "name": "SECURITY_SERVICE_URL",
                        "value": "http://security-service:8443"
                    }
                ]
            }

            # Generate Image Processing service
            service_definitions["image_processing"] = {
                "image": "negativespaceimagingproject/image-processing:latest",
                "replicas": 2,
                "resources": {
                    "limits": {
                        "cpu": "4",
                        "memory": "16Gi",
                        "gpu": "1"
                    },
                    "requests": {
                        "cpu": "2",
                        "memory": "8Gi",
                        "gpu": "1"
                    }
                },
                "environment_variables": [
                    {
                        "name": "PROCESSING_PORT",
                        "value": "9000"
                    },
                    {
                        "name": "GPU_ENABLED",
                        "value": "true"
                    },
                    {
                        "name": "DATA_SERVICE_URL",
                        "value": "http://data-service:8000"
                    }
                ],
                "volume_mounts": [
                    {
                        "name": "models",
                        "mount_path": "/app/models"
                    },
                    {
                        "name": "processing",
                        "mount_path": "/app/processing"
                    }
                ]
            }

            # Generate Data Storage service
            service_definitions["data_storage"] = {
                "image": "negativespaceimagingproject/data-storage:latest",
                "replicas": 1,
                "resources": {
                    "limits": {
                        "cpu": "2",
                        "memory": "8Gi"
                    },
                    "requests": {
                        "cpu": "1",
                        "memory": "4Gi"
                    }
                },
                "environment_variables": [
                    {
                        "name": "STORAGE_PORT",
                        "value": "8000"
                    },
                    {
                        "name": "DATA_PATH",
                        "value": "/app/data"
                    }
                ],
                "volume_mounts": [
                    {
                        "name": "data",
                        "mount_path": "/app/data"
                    }
                ]
            }

            # Generate Distributed Computing service
            service_definitions["distributed_computing"] = {
                "image": "negativespaceimagingproject/distributed-computing:latest",
                "replicas": 1,
                "resources": {
                    "limits": {
                        "cpu": "2",
                        "memory": "8Gi"
                    },
                    "requests": {
                        "cpu": "1",
                        "memory": "4Gi"
                    }
                },
                "environment_variables": [
                    {
                        "name": "SCHEDULER_PORT",
                        "value": "8787"
                    },
                    {
                        "name": "NUM_WORKERS",
                        "value": "4"
                    }
                ],
                "health_check": {
                    "path": "/health",
                    "port": 8787,
                    "initial_delay_seconds": 30,
                    "period_seconds": 10
                }
            }

            # Generate Security service
            service_definitions["security"] = {
                "image": "negativespaceimagingproject/security:latest",
                "replicas": 1,
                "resources": {
                    "limits": {
                        "cpu": "1",
                        "memory": "2Gi"
                    },
                    "requests": {
                        "cpu": "0.5",
                        "memory": "1Gi"
                    }
                },
                "environment_variables": [
                    {
                        "name": "SECURITY_PORT",
                        "value": "8443"
                    },
                    {
                        "name": "CONFIG_PATH",
                        "value": "/app/config"
                    }
                ],
                "volume_mounts": [
                    {
                        "name": "security-config",
                        "mount_path": "/app/config"
                    }
                ],
                "health_check": {
                    "path": "/health",
                    "port": 8443,
                    "initial_delay_seconds": 30,
                    "period_seconds": 10
                }
            }

            # Save service definitions
            services_path = self.project_root / "deployment" / "services.yaml"
            with open(services_path, 'w') as f:
                yaml.dump(service_definitions, f, default_flow_style=False)

            logger.info(f"Service definitions saved to {services_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate service definitions: {str(e)}")
            return False

    def integrate(self):
        """Integrate multi-node deployment with existing system"""
        logger.info("Starting multi-node integration")

        # Check if modules are loaded
        if not modules_loaded:
            logger.error("Required modules not loaded, cannot proceed with integration")
            return False

        # Adapt security configuration
        if not self.adapt_security_config():
            logger.warning("Failed to adapt security configuration")

        # Adapt HPC configuration
        if not self.adapt_hpc_config():
            logger.warning("Failed to adapt HPC configuration")

        # Adapt distributed computing configuration
        if not self.adapt_distributed_computing_config():
            logger.warning("Failed to adapt distributed computing configuration")

        # Generate service definitions
        if not self.generate_service_definitions():
            logger.warning("Failed to generate service definitions")

        logger.info("Multi-node integration completed")
        return True

def main():
    """Main entry point"""
    try:
        # Create multi-node integration
        integration = MultiNodeIntegration()

        # Run integration
        success = integration.integrate()

        if success:
            logger.info("Integration completed successfully")
            return 0
        else:
            logger.error("Integration failed")
            return 1
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
