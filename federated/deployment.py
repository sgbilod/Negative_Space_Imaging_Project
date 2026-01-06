"""
Deployment Configuration for Federated Learning
Docker, Kubernetes, and production setup.
"""

import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DockerConfig:
    """Docker configuration for federated learning."""

    image_name: str = "federated-learning"
    image_tag: str = "latest"
    base_image: str = "python:3.10-slim"
    working_dir: str = "/app"

    def to_dockerfile(self) -> str:
        """Generate Dockerfile content."""
        return f"""FROM {self.base_image}

WORKDIR {self.working_dir}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run application
CMD ["python", "-m", "federated.main"]
"""


@dataclass
class KubernetesConfig:
    """Kubernetes configuration."""

    namespace: str = "federated-learning"
    server_replicas: int = 1
    client_replicas: int = 5
    cpu_request: str = "500m"
    cpu_limit: str = "2"
    memory_request: str = "512Mi"
    memory_limit: str = "2Gi"

    def to_deployment_yaml(self, component: str = "server") -> str:
        """Generate Kubernetes deployment YAML."""
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: federated-{component}
  namespace: {self.namespace}
spec:
  replicas: {self.server_replicas if component == "server" else self.client_replicas}
  selector:
    matchLabels:
      app: federated-{component}
  template:
    metadata:
      labels:
        app: federated-{component}
    spec:
      containers:
      - name: {component}
        image: federated-learning:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: {self.cpu_request}
            memory: {self.memory_request}
          limits:
            cpu: {self.cpu_limit}
            memory: {self.memory_limit}
        env:
        - name: ROLE
          value: "{component}"
        - name: SERVER_ADDRESS
          value: "federated-server:8080"
"""


class DeploymentManager:
    """Manages deployment of federated learning system."""

    def __init__(self, deployment_dir: Optional[str] = None):
        """
        Initialize deployment manager.

        Args:
            deployment_dir: Directory for deployment files
        """
        self.deployment_dir = Path(deployment_dir or "./deployment")
        self.deployment_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DeploymentManager initialized: {self.deployment_dir}")

    def generate_dockerfile(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate Dockerfile for federated learning.

        Args:
            output_path: Path to save Dockerfile

        Returns:
            Dockerfile content
        """
        config = DockerConfig()
        dockerfile = config.to_dockerfile()

        if output_path:
            Path(output_path).write_text(dockerfile)
            logger.info(f"Dockerfile written to {output_path}")

        return dockerfile

    def generate_kubernetes_manifests(
        self,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate Kubernetes manifests.

        Args:
            output_dir: Directory to save manifests

        Returns:
            Dictionary of manifest names to content
        """
        config = KubernetesConfig()

        manifests = {
            "namespace.yaml": self._generate_namespace_yaml(config.namespace),
            "server-deployment.yaml": config.to_deployment_yaml("server"),
            "client-deployment.yaml": config.to_deployment_yaml("client"),
            "service.yaml": self._generate_service_yaml(),
        }

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            for name, content in manifests.items():
                (output_path / name).write_text(content)
                logger.info(f"Manifest written to {output_path / name}")

        return manifests

    def _generate_namespace_yaml(self, namespace: str) -> str:
        """Generate Kubernetes namespace YAML."""
        return f"""apiVersion: v1
kind: Namespace
metadata:
  name: {namespace}
"""

    def _generate_service_yaml(self) -> str:
        """Generate Kubernetes service YAML."""
        return """apiVersion: v1
kind: Service
metadata:
  name: federated-server
  namespace: federated-learning
spec:
  selector:
    app: federated-server
  ports:
  - protocol: TCP
    port: 8080
    targetPort: 8080
  type: LoadBalancer
"""

    def generate_docker_compose(
        self,
        num_clients: int = 5,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate Docker Compose configuration.

        Args:
            num_clients: Number of client containers
            output_path: Path to save docker-compose.yml

        Returns:
            Docker Compose content
        """
        services = {
            "server": {
                "image": "federated-learning:latest",
                "ports": ["8080:8080"],
                "environment": ["ROLE=server"],
                "volumes": ["./data:/app/data"],
            }
        }

        # Add clients
        for i in range(num_clients):
            services[f"client_{i}"] = {
                "image": "federated-learning:latest",
                "environment": [
                    f"ROLE=client",
                    f"CLIENT_ID=client_{i}",
                    "SERVER_ADDRESS=server:8080",
                ],
                "depends_on": ["server"],
                "volumes": [f"./data/client_{i}:/app/data"],
            }

        compose = {
            "version": "3.8",
            "services": services,
            "volumes": {
                "data": {}
            }
        }

        content = self._dict_to_yaml(compose)

        if output_path:
            Path(output_path).write_text(content)
            logger.info(f"Docker Compose written to {output_path}")

        return content

    def _dict_to_yaml(self, d: Dict) -> str:
        """Convert dict to YAML format (simple version)."""
        def format_dict(d, indent=0):
            lines = []
            for k, v in d.items():
                if isinstance(v, dict):
                    lines.append("  " * indent + f"{k}:")
                    lines.append(format_dict(v, indent + 1))
                elif isinstance(v, list):
                    lines.append("  " * indent + f"{k}:")
                    for item in v:
                        if isinstance(item, dict):
                            for sk, sv in item.items():
                                lines.append("  " * (indent + 1) + f"- {sk}: {sv}")
                        else:
                            lines.append("  " * (indent + 1) + f"- {item}")
                else:
                    lines.append("  " * indent + f"{k}: {v}")
            return "\n".join(lines)

        return format_dict(d)

    def generate_requirements_txt(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate requirements.txt for deployment.

        Args:
            output_path: Path to save requirements.txt

        Returns:
            Requirements content
        """
        requirements = """torch==2.0.0
torchvision==0.15.0
flwr==1.4.1
numpy==1.24.0
opacus==1.4.0
pydantic==2.0.0
pyyaml==6.0
requests==2.31.0
protobuf==4.24.0
grpcio==1.56.0
cryptography==41.0.0
"""

        if output_path:
            Path(output_path).write_text(requirements)
            logger.info(f"Requirements written to {output_path}")

        return requirements

    def generate_docker_compose_prod(
        self,
        num_clients: int = 10,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate production Docker Compose with health checks.

        Args:
            num_clients: Number of client containers
            output_path: Path to save docker-compose.yml

        Returns:
            Docker Compose content
        """
        content = f"""version: '3.8'

services:
  server:
    image: federated-learning:latest
    container_name: federated-server
    ports:
      - "8080:8080"
    environment:
      - ROLE=server
      - LOG_LEVEL=INFO
    volumes:
      - ./data/server:/app/data
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
    networks:
      - federated

  """

        for i in range(num_clients):
            content += f"""client_{i}:
    image: federated-learning:latest
    container_name: federated-client-{i}
    environment:
      - ROLE=client
      - CLIENT_ID=client_{i}
      - SERVER_ADDRESS=server:8080
      - LOG_LEVEL=INFO
    volumes:
      - ./data/client_{i}:/app/data
      - ./logs/client_{i}:/app/logs
    depends_on:
      server:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import socket; socket.create_connection(('server', 8080))"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    networks:
      - federated

  """

        content += """networks:
  federated:
    driver: bridge

volumes:
  server_data:
  logs:
"""

        if output_path:
            Path(output_path).write_text(content)
            logger.info(f"Production Docker Compose written to {output_path}")

        return content


class HealthCheckManager:
    """Manages health checks for deployed components."""

    @staticmethod
    def check_server_health(server_address: str) -> bool:
        """Check if server is healthy."""
        import socket

        try:
            host, port = server_address.split(":")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, int(port)))
            sock.close()
            return result == 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    @staticmethod
    def check_client_health(client_id: str) -> bool:
        """Check if client is responsive."""
        # Placeholder for client health check
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    manager = DeploymentManager()

    # Generate Dockerfile
    manager.generate_dockerfile("./Dockerfile")

    # Generate Kubernetes manifests
    manager.generate_kubernetes_manifests("./k8s")

    # Generate Docker Compose
    manager.generate_docker_compose(num_clients=5, output_path="./docker-compose.yml")

    # Generate requirements
    manager.generate_requirements_txt("./requirements.txt")

    print("Deployment configuration generated")
