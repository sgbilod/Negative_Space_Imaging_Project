#!/usr/bin/env python3
"""
Docker Image Builder for Negative Space Imaging Project

Builds and pushes Docker images for the multi-node deployment
"""

import os
import sys
import subprocess
import argparse
import logging
import yaml
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("docker_build.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Docker image definitions
IMAGE_DEFINITIONS = {
    "api-gateway": {
        "context": ".",
        "dockerfile": "deployment/dockerfiles/api-gateway.Dockerfile",
        "build_args": {
            "PORT": "8080"
        }
    },
    "image-processing": {
        "context": ".",
        "dockerfile": "deployment/dockerfiles/image-processing.Dockerfile",
        "build_args": {
            "GPU_ENABLED": "true"
        }
    },
    "data-storage": {
        "context": ".",
        "dockerfile": "deployment/dockerfiles/data-storage.Dockerfile",
        "build_args": {
            "STORAGE_PORT": "8000"
        }
    },
    "distributed-computing": {
        "context": ".",
        "dockerfile": "deployment/dockerfiles/distributed-computing.Dockerfile",
        "build_args": {
            "SCHEDULER_PORT": "8787"
        }
    },
    "security": {
        "context": ".",
        "dockerfile": "deployment/dockerfiles/security.Dockerfile",
        "build_args": {
            "SECURITY_PORT": "8443"
        }
    }
}

def run_command(command, check=True):
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

        return exit_code, stdout, stderr
    except Exception as e:
        logger.error(f"Failed to run command: {str(e)}")
        return 1, "", str(e)

def build_image(image_name, image_config, tag=None, push=False, registry=None):
    """Build a Docker image"""
    # Determine the image tag
    if registry:
        full_tag = f"{registry}/negativespaceimagingproject/{image_name}"
    else:
        full_tag = f"negativespaceimagingproject/{image_name}"

    if tag:
        full_tag = f"{full_tag}:{tag}"
    else:
        full_tag = f"{full_tag}:latest"

    # Build the Docker command
    cmd = ["docker", "build", "-t", full_tag]

    # Add build arguments
    for arg_name, arg_value in image_config.get("build_args", {}).items():
        cmd.extend(["--build-arg", f"{arg_name}={arg_value}"])

    # Add context
    cmd.append("-f")
    cmd.append(image_config["dockerfile"])
    cmd.append(image_config["context"])

    # Run the build
    logger.info(f"Building image: {full_tag}")
    exit_code, stdout, stderr = run_command(cmd)

    if exit_code != 0:
        logger.error(f"Failed to build image: {full_tag}")
        logger.error(stderr)
        return False

    logger.info(f"Successfully built image: {full_tag}")

    # Push the image if requested
    if push:
        logger.info(f"Pushing image: {full_tag}")
        exit_code, stdout, stderr = run_command(["docker", "push", full_tag])

        if exit_code != 0:
            logger.error(f"Failed to push image: {full_tag}")
            logger.error(stderr)
            return False

        logger.info(f"Successfully pushed image: {full_tag}")

    return True

def create_dockerfile_directory():
    """Create the dockerfiles directory if it doesn't exist"""
    dockerfile_dir = Path("deployment/dockerfiles")
    if not dockerfile_dir.exists():
        logger.info(f"Creating directory: {dockerfile_dir}")
        dockerfile_dir.mkdir(parents=True, exist_ok=True)

    return dockerfile_dir

def create_sample_dockerfiles():
    """Create sample Dockerfiles for the various services"""
    logger.info("Creating sample Dockerfiles")

    dockerfile_dir = create_dockerfile_directory()

    # API Gateway Dockerfile
    api_gateway_dockerfile = dockerfile_dir / "api-gateway.Dockerfile"
    if not api_gateway_dockerfile.exists():
        with open(api_gateway_dockerfile, "w") as f:
            f.write("""FROM python:3.9-slim

ARG PORT=8080
ENV API_PORT=$PORT

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$API_PORT/health || exit 1

EXPOSE $API_PORT

CMD ["python", "-m", "uvicorn", "api_gateway:app", "--host", "0.0.0.0", "--port", "$API_PORT"]
""")
        logger.info(f"Created {api_gateway_dockerfile}")

    # Image Processing Dockerfile
    image_processing_dockerfile = dockerfile_dir / "image-processing.Dockerfile"
    if not image_processing_dockerfile.exists():
        with open(image_processing_dockerfile, "w") as f:
            f.write("""FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04

ARG GPU_ENABLED=true
ENV GPU_ENABLED=$GPU_ENABLED

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 9000
EXPOSE 8080

CMD ["python3", "image_acquisition.py", "--server"]
""")
        logger.info(f"Created {image_processing_dockerfile}")

    # Data Storage Dockerfile
    data_storage_dockerfile = dockerfile_dir / "data-storage.Dockerfile"
    if not data_storage_dockerfile.exists():
        with open(data_storage_dockerfile, "w") as f:
            f.write("""FROM python:3.9-slim

ARG STORAGE_PORT=8000
ENV STORAGE_PORT=$STORAGE_PORT

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directory
RUN mkdir -p /app/data

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$STORAGE_PORT/health || exit 1

EXPOSE $STORAGE_PORT

CMD ["python", "data_storage_server.py"]
""")
        logger.info(f"Created {data_storage_dockerfile}")

    # Distributed Computing Dockerfile
    distributed_computing_dockerfile = dockerfile_dir / "distributed-computing.Dockerfile"
    if not distributed_computing_dockerfile.exists():
        with open(distributed_computing_dockerfile, "w") as f:
            f.write("""FROM python:3.9-slim

ARG SCHEDULER_PORT=8787
ENV SCHEDULER_PORT=$SCHEDULER_PORT

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt dask distributed

COPY . .

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE $SCHEDULER_PORT
EXPOSE 8080

CMD ["python", "distributed_computing.py", "--scheduler"]
""")
        logger.info(f"Created {distributed_computing_dockerfile}")

    # Security Dockerfile
    security_dockerfile = dockerfile_dir / "security.Dockerfile"
    if not security_dockerfile.exists():
        with open(security_dockerfile, "w") as f:
            f.write("""FROM python:3.9-slim

ARG SECURITY_PORT=8443
ENV SECURITY_PORT=$SECURITY_PORT

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE $SECURITY_PORT
EXPOSE 8080

CMD ["python", "security_service.py"]
""")
        logger.info(f"Created {security_dockerfile}")

    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Docker Image Builder")
    parser.add_argument("--build", action="store_true", help="Build the Docker images")
    parser.add_argument("--push", action="store_true", help="Push the Docker images")
    parser.add_argument("--tag", type=str, default="latest", help="Image tag")
    parser.add_argument("--registry", type=str, help="Docker registry")
    parser.add_argument("--image", type=str, help="Specific image to build")
    args = parser.parse_args()

    # Create sample Dockerfiles
    if not create_sample_dockerfiles():
        logger.error("Failed to create sample Dockerfiles")
        return 1

    # Check if building images
    if args.build:
        if args.image:
            # Build a specific image
            if args.image not in IMAGE_DEFINITIONS:
                logger.error(f"Unknown image: {args.image}")
                return 1

            if not build_image(args.image, IMAGE_DEFINITIONS[args.image], args.tag, args.push, args.registry):
                logger.error(f"Failed to build image: {args.image}")
                return 1
        else:
            # Build all images
            success = True
            for image_name, image_config in IMAGE_DEFINITIONS.items():
                if not build_image(image_name, image_config, args.tag, args.push, args.registry):
                    logger.error(f"Failed to build image: {image_name}")
                    success = False

            if not success:
                return 1
    else:
        logger.info("No actions specified. Use --build to build images.")
        parser.print_help()

    return 0

if __name__ == "__main__":
    sys.exit(main())
