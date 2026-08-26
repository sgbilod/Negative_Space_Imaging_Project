"""
Kubernetes Manifest Generator for Negative Space Imaging Project
Automatically generates Kubernetes manifests for multi-node deployment
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("kubernetes_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KubernetesManifestGenerator:
    """Generates Kubernetes manifests for multi-node deployment"""

    def __init__(self, config_path: str = "deployment/multi_node_config.yaml"):
        """
        Initialize the Kubernetes manifest generator

        Args:
            config_path: Path to the multi-node configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

        # Set up paths
        self.base_dir = Path.cwd()
        self.deployment_dir = self.base_dir / "deployment"
        self.k8s_dir = self.deployment_dir / "kubernetes"

        # Ensure directories exist
        self.k8s_dir.mkdir(parents=True, exist_ok=True)

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

    def generate_namespace(self) -> Dict[str, Any]:
        """Generate namespace manifest"""
        namespace = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.config["cluster"]["name"],
                "labels": {
                    "name": self.config["cluster"]["name"],
                    "environment": self.config["cluster"]["environment"]
                }
            }
        }

        return namespace

    def generate_deployments(self) -> List[Dict[str, Any]]:
        """Generate deployment manifests"""
        deployments = []

        for service_name, service_config in self.config["services"].items():
            deployment = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": f"{self.config['cluster']['name']}-{service_name}",
                    "namespace": self.config["cluster"]["name"],
                    "labels": {
                        "app": service_name,
                        "cluster": self.config["cluster"]["name"]
                    }
                },
                "spec": {
                    "replicas": service_config["replicas"],
                    "selector": {
                        "matchLabels": {
                            "app": service_name
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": service_name,
                                "cluster": self.config["cluster"]["name"]
                            }
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": service_name,
                                    "image": service_config["image"],
                                    "resources": {
                                        "limits": service_config["resources"]["limits"],
                                        "requests": service_config["resources"]["requests"]
                                    },
                                    "env": [
                                        {
                                            "name": env_var["name"],
                                            "value": env_var["value"]
                                        }
                                        for env_var in service_config.get("environment_variables", [])
                                    ]
                                }
                            ]
                        }
                    }
                }
            }

            # Add health check if defined
            if "health_check" in service_config:
                health = service_config["health_check"]
                health_check = {
                    "livenessProbe": {
                        "httpGet": {
                            "path": health["path"],
                            "port": health["port"]
                        },
                        "initialDelaySeconds": health["initial_delay_seconds"],
                        "periodSeconds": health["period_seconds"]
                    },
                    "readinessProbe": {
                        "httpGet": {
                            "path": health["path"],
                            "port": health["port"]
                        },
                        "initialDelaySeconds": health["initial_delay_seconds"],
                        "periodSeconds": health["period_seconds"]
                    }
                }
                deployment["spec"]["template"]["spec"]["containers"][0].update(health_check)

            # Add volume mounts if defined
            if "volume_mounts" in service_config:
                volume_mounts = []
                volumes = []

                for volume in service_config["volume_mounts"]:
                    volume_name = volume["name"]
                    volume_mounts.append({
                        "name": volume_name,
                        "mountPath": volume["mount_path"]
                    })

                    # Add volume definition
                    if volume_name == "data":
                        volumes.append({
                            "name": volume_name,
                            "persistentVolumeClaim": {
                                "claimName": f"{self.config['cluster']['name']}-data"
                            }
                        })
                    elif volume_name == "models":
                        volumes.append({
                            "name": volume_name,
                            "persistentVolumeClaim": {
                                "claimName": f"{self.config['cluster']['name']}-models"
                            }
                        })
                    elif volume_name == "processing":
                        volumes.append({
                            "name": volume_name,
                            "emptyDir": {}
                        })
                    elif volume_name == "security-config":
                        volumes.append({
                            "name": volume_name,
                            "configMap": {
                                "name": f"{self.config['cluster']['name']}-security-config"
                            }
                        })
                    else:
                        volumes.append({
                            "name": volume_name,
                            "emptyDir": {}
                        })

                deployment["spec"]["template"]["spec"]["containers"][0]["volumeMounts"] = volume_mounts
                deployment["spec"]["template"]["spec"]["volumes"] = volumes

            # Add node selector if appropriate
            if service_name == "api-gateway":
                deployment["spec"]["template"]["spec"]["nodeSelector"] = {"nodeType": "edge"}
            elif service_name == "image-processing":
                deployment["spec"]["template"]["spec"]["nodeSelector"] = {"nodeType": "compute"}
            elif service_name == "data-storage":
                deployment["spec"]["template"]["spec"]["nodeSelector"] = {"nodeType": "storage"}
            elif service_name == "security":
                deployment["spec"]["template"]["spec"]["nodeSelector"] = {"nodeType": "master"}

            deployments.append(deployment)

        return deployments

    def generate_services(self) -> List[Dict[str, Any]]:
        """Generate service manifests"""
        services = []

        for service_name, service_config in self.config["services"].items():
            ports = []

            # Add ports based on service type
            if service_name == "api-gateway":
                ports = [
                    {"name": "http", "port": 80, "targetPort": 8080},
                    {"name": "https", "port": 443, "targetPort": 8443}
                ]
            elif service_name == "image-processing":
                ports = [
                    {"name": "grpc", "port": 9000, "targetPort": 9000},
                    {"name": "http", "port": 8080, "targetPort": 8080}
                ]
            elif service_name == "data-storage":
                ports = [
                    {"name": "http", "port": 8000, "targetPort": 8000}
                ]
            elif service_name == "distributed-computing":
                ports = [
                    {"name": "scheduler", "port": 8787, "targetPort": 8787},
                    {"name": "http", "port": 8080, "targetPort": 8080}
                ]
            elif service_name == "security":
                ports = [
                    {"name": "https", "port": 8443, "targetPort": 8443},
                    {"name": "http", "port": 8080, "targetPort": 8080}
                ]

            # Determine service type
            service_type = "ClusterIP"
            if service_name == "api-gateway":
                service_type = "LoadBalancer"

            service = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"{self.config['cluster']['name']}-{service_name}",
                    "namespace": self.config["cluster"]["name"],
                    "labels": {
                        "app": service_name,
                        "cluster": self.config["cluster"]["name"]
                    }
                },
                "spec": {
                    "selector": {
                        "app": service_name
                    },
                    "ports": ports,
                    "type": service_type
                }
            }

            services.append(service)

        return services

    def generate_persistent_volume_claims(self) -> List[Dict[str, Any]]:
        """Generate persistent volume claim manifests"""
        pvcs = []

        # Add data PVC
        data_pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{self.config['cluster']['name']}-data",
                "namespace": self.config["cluster"]["name"]
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "resources": {
                    "requests": {
                        "storage": self.config["storage"]["data"]["capacity"]
                    }
                },
                "storageClassName": f"{self.config['cluster']['name']}-persistent"
            }
        }
        pvcs.append(data_pvc)

        # Add models PVC
        models_pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{self.config['cluster']['name']}-models",
                "namespace": self.config["cluster"]["name"]
            },
            "spec": {
                "accessModes": ["ReadOnlyMany"],
                "resources": {
                    "requests": {
                        "storage": self.config["storage"]["models"]["capacity"]
                    }
                },
                "storageClassName": f"{self.config['cluster']['name']}-persistent"
            }
        }
        pvcs.append(models_pvc)

        return pvcs

    def generate_storage_classes(self) -> List[Dict[str, Any]]:
        """Generate storage class manifests"""
        storage_classes = []

        # Persistent storage class
        persistent_sc = {
            "apiVersion": "storage.k8s.io/v1",
            "kind": "StorageClass",
            "metadata": {
                "name": f"{self.config['cluster']['name']}-persistent"
            },
            "provisioner": "kubernetes.io/aws-ebs",
            "parameters": {
                "type": "gp2",
                "fsType": "ext4"
            },
            "reclaimPolicy": "Retain"
        }
        storage_classes.append(persistent_sc)

        # Fast local storage class
        fast_local_sc = {
            "apiVersion": "storage.k8s.io/v1",
            "kind": "StorageClass",
            "metadata": {
                "name": f"{self.config['cluster']['name']}-fast-local"
            },
            "provisioner": "kubernetes.io/local-storage",
            "parameters": {
                "type": "SSD"
            },
            "reclaimPolicy": "Retain",
            "volumeBindingMode": "WaitForFirstConsumer"
        }
        storage_classes.append(fast_local_sc)

        # High-performance storage class
        high_perf_sc = {
            "apiVersion": "storage.k8s.io/v1",
            "kind": "StorageClass",
            "metadata": {
                "name": f"{self.config['cluster']['name']}-high-performance"
            },
            "provisioner": "kubernetes.io/local-storage",
            "parameters": {
                "type": "NVMe"
            },
            "reclaimPolicy": "Retain",
            "volumeBindingMode": "WaitForFirstConsumer"
        }
        storage_classes.append(high_perf_sc)

        return storage_classes

    def generate_config_maps(self) -> List[Dict[str, Any]]:
        """Generate config map manifests"""
        config_maps = []

        # Security config map
        security_config = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.config['cluster']['name']}-security-config",
                "namespace": self.config["cluster"]["name"]
            },
            "data": {
                "security_config.json": json.dumps({
                    "distributed_mode": True,
                    "node_verification": True,
                    "inter_node_encryption": True,
                    "authentication": {
                        "jwt_secret": "${JWT_SECRET}",
                        "token_expiry": 3600
                    },
                    "authorization": {
                        "role_based_access": True,
                        "default_role": "viewer"
                    },
                    "audit": {
                        "enabled": True,
                        "log_level": "INFO",
                        "retention": "30d"
                    }
                }, indent=2)
            }
        }
        config_maps.append(security_config)

        return config_maps

    def generate_network_policies(self) -> List[Dict[str, Any]]:
        """Generate network policy manifests"""
        network_policies = []

        # Internal network policy
        internal_policy = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{self.config['cluster']['name']}-internal",
                "namespace": self.config["cluster"]["name"]
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "cluster": self.config["cluster"]["name"]
                    }
                },
                "ingress": [
                    {
                        "from": [
                            {
                                "podSelector": {
                                    "matchLabels": {
                                        "cluster": self.config["cluster"]["name"]
                                    }
                                }
                            }
                        ]
                    }
                ]
            }
        }
        network_policies.append(internal_policy)

        # API network policy
        api_policy = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{self.config['cluster']['name']}-api",
                "namespace": self.config["cluster"]["name"]
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": "api-gateway"
                    }
                },
                "ingress": [
                    {
                        "from": [
                            {
                                "ipBlock": {
                                    "cidr": self.config["networking"]["api_access_cidr"]
                                }
                            }
                        ],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 443
                            },
                            {
                                "protocol": "TCP",
                                "port": 80
                            }
                        ]
                    }
                ]
            }
        }
        network_policies.append(api_policy)

        return network_policies

    def generate_all_manifests(self):
        """Generate all Kubernetes manifests"""
        try:
            # Generate namespace
            namespace = self.generate_namespace()
            namespace_path = self.k8s_dir / "00-namespace.yaml"
            with open(namespace_path, 'w') as f:
                yaml.dump(namespace, f, default_flow_style=False)
            logger.info(f"Generated namespace manifest: {namespace_path}")

            # Generate storage classes
            storage_classes = self.generate_storage_classes()
            storage_classes_path = self.k8s_dir / "01-storage-classes.yaml"
            with open(storage_classes_path, 'w') as f:
                yaml.dump_all(storage_classes, f, default_flow_style=False)
            logger.info(f"Generated storage class manifests: {storage_classes_path}")

            # Generate persistent volume claims
            pvcs = self.generate_persistent_volume_claims()
            pvcs_path = self.k8s_dir / "02-persistent-volume-claims.yaml"
            with open(pvcs_path, 'w') as f:
                yaml.dump_all(pvcs, f, default_flow_style=False)
            logger.info(f"Generated persistent volume claim manifests: {pvcs_path}")

            # Generate config maps
            config_maps = self.generate_config_maps()
            config_maps_path = self.k8s_dir / "03-config-maps.yaml"
            with open(config_maps_path, 'w') as f:
                yaml.dump_all(config_maps, f, default_flow_style=False)
            logger.info(f"Generated config map manifests: {config_maps_path}")

            # Generate deployments
            deployments = self.generate_deployments()
            deployments_path = self.k8s_dir / "04-deployments.yaml"
            with open(deployments_path, 'w') as f:
                yaml.dump_all(deployments, f, default_flow_style=False)
            logger.info(f"Generated deployment manifests: {deployments_path}")

            # Generate services
            services = self.generate_services()
            services_path = self.k8s_dir / "05-services.yaml"
            with open(services_path, 'w') as f:
                yaml.dump_all(services, f, default_flow_style=False)
            logger.info(f"Generated service manifests: {services_path}")

            # Generate network policies
            network_policies = self.generate_network_policies()
            network_policies_path = self.k8s_dir / "06-network-policies.yaml"
            with open(network_policies_path, 'w') as f:
                yaml.dump_all(network_policies, f, default_flow_style=False)
            logger.info(f"Generated network policy manifests: {network_policies_path}")

            # Create README file
            readme_path = self.k8s_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(f"""# Kubernetes Manifests for {self.config["cluster"]["name"]}

This directory contains Kubernetes manifests for deploying the Negative Space Imaging Project.

## Manifests

1. **00-namespace.yaml**: Namespace for the deployment
2. **01-storage-classes.yaml**: Storage classes for persistent storage
3. **02-persistent-volume-claims.yaml**: Persistent volume claims for data and models
4. **03-config-maps.yaml**: Configuration maps for services
5. **04-deployments.yaml**: Deployments for services
6. **05-services.yaml**: Services for deployments
7. **06-network-policies.yaml**: Network policies for security

## Deployment

To deploy these manifests, run:

```bash
kubectl apply -f ./
```

To delete the deployment, run:

```bash
kubectl delete -f ./
```
""")
            logger.info(f"Generated README: {readme_path}")

            return True
        except Exception as e:
            logger.error(f"Failed to generate manifests: {str(e)}")
            return False

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Kubernetes Manifest Generator")
    parser.add_argument("--config", "-c", default="deployment/multi_node_config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    try:
        # Create manifest generator
        generator = KubernetesManifestGenerator(args.config)

        # Generate all manifests
        if generator.generate_all_manifests():
            logger.info("Kubernetes manifests generated successfully")
            return 0
        else:
            logger.error("Failed to generate Kubernetes manifests")
            return 1
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
