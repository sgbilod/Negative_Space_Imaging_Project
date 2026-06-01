#!/usr/bin/env python3
"""
Multi-Node Deployment Setup Script for Negative Space Imaging Project
Prepares the environment for multi-node deployment with advanced configuration
options and automatic dependency resolution.
"""

import os
import sys
import logging
import argparse
import subprocess
import shutil
import json
import yaml
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import secrets
import string
import datetime
import re

# Configure logging with advanced formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("deployment_setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(command: List[str], check: bool = True, env: Optional[Dict[str, str]] = None,
             cwd: Optional[str] = None, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    """
    Run a shell command and return the exit code, stdout, and stderr

    Args:
        command: The command to run as a list of strings
        check: Whether to raise an exception if the command fails
        env: Environment variables to pass to the command
        cwd: Working directory to run the command in
        timeout: Timeout in seconds for the command

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    command_str = ' '.join(command)
    logger.debug(f"Running command: {command_str}")

    # Prepare environment
    command_env = os.environ.copy()
    if env:
        command_env.update(env)

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=command_env,
            cwd=cwd
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = process.returncode

            if exit_code != 0 and check:
                logger.error(f"Command failed with exit code {exit_code}")
                logger.error(f"Command: {command_str}")
                logger.error(f"STDERR: {stderr}")
                logger.error(f"STDOUT: {stdout}")

            return exit_code, stdout, stderr
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            logger.error(f"Command timed out after {timeout} seconds")
            logger.error(f"Command: {command_str}")
            if check:
                raise
            return 124, stdout, f"Timeout after {timeout} seconds"

    except Exception as e:
        logger.error(f"Failed to run command: {str(e)}")
        logger.error(f"Command: {command_str}")
        if check:
            raise
        return 1, "", str(e)

def check_prerequisites():
    """
    Check if all prerequisites for deployment are met
    Validates directories, dependencies, and required tools

    Returns:
        bool: True if all prerequisites are met, False otherwise
    """
    logger.info("Checking prerequisites...")

    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False

    logger.debug(f"Python version: {platform.python_version()}")

    # Check for required tools
    required_tools = ["docker", "kubectl", "helm"]
    for tool in required_tools:
        try:
            exit_code, stdout, stderr = run_command([tool, "--version"], check=False)
            if exit_code != 0:
                logger.warning(f"{tool} not found in PATH")
            else:
                logger.debug(f"{tool} version: {stdout.strip()}")
        except Exception as e:
            logger.warning(f"Failed to check {tool}: {str(e)}")

    # Check for required directories
    required_dirs = ["deployment", "deployment/kubernetes", "deployment/templates", "deployment/config"]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Directory exists or created: {dir_path}")

    # Check for required Python packages
    required_packages = ["pyyaml", "jinja2", "kubernetes", "psycopg2-binary", "cryptography"]
    missing_packages = []

    for package in required_packages:
        try:
            # Try to import the package
            exec(f"import {package.split('-')[0]}")
            logger.debug(f"Package {package} is installed")
        except ImportError:
            logger.warning(f"Package {package} is not installed")
            missing_packages.append(package)

    if missing_packages:
        logger.warning(f"Missing Python packages: {', '.join(missing_packages)}")
        logger.info("You can install them with: pip install " + " ".join(missing_packages))
        # Continue anyway, as they might be installed later

    logger.info("Prerequisites check completed")
    return True

def setup_database_files():
    """
    Set up database files for multi-node deployment
    Creates database initialization scripts and schema files

    Returns:
        bool: True if setup was successful, False otherwise
    """
    logger.info("Setting up database files...")

    deployment_dir = Path("deployment")
    database_dir = deployment_dir / "database"

    # Ensure database directory exists
    database_dir.mkdir(parents=True, exist_ok=True)

    # Define database files
    db_files = {
        "01-init-schema.sql": """-- Database initialization script for Negative Space Imaging Project
-- Creates the required schema and tables for the application

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- System Events Table
CREATE TABLE IF NOT EXISTS system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- System Nodes Table
CREATE TABLE IF NOT EXISTS system_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    node_type VARCHAR(50) DEFAULT 'compute',
    details JSONB,
    last_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Data Storage Table
CREATE TABLE IF NOT EXISTS data_storage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    storage_id VARCHAR(100) NOT NULL UNIQUE,
    storage_type VARCHAR(50) NOT NULL,
    location TEXT NOT NULL,
    capacity BIGINT NOT NULL,
    used BIGINT NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Imaging Jobs Table
CREATE TABLE IF NOT EXISTS imaging_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 5,
    source_data TEXT,
    output_location TEXT,
    parameters JSONB,
    node_id UUID REFERENCES system_nodes(id),
    progress DECIMAL(5,2) DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Job Results Table
CREATE TABLE IF NOT EXISTS job_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES imaging_jobs(id),
    result_type VARCHAR(50) NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_system_events_event_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_system_events_created_at ON system_events(created_at);
CREATE INDEX IF NOT EXISTS idx_system_nodes_node_type ON system_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_system_nodes_status ON system_nodes(status);
CREATE INDEX IF NOT EXISTS idx_imaging_jobs_status ON imaging_jobs(status);
CREATE INDEX IF NOT EXISTS idx_imaging_jobs_priority ON imaging_jobs(priority);
CREATE INDEX IF NOT EXISTS idx_job_results_job_id ON job_results(job_id);
""",

        "02-init-data.sql": """-- Initial data for Negative Space Imaging Project
-- Inserts base configuration and required data

-- Insert system initialization event
INSERT INTO system_events (event_type, severity, message, details)
VALUES ('system.initialization', 'info', 'Database initialized',
       '{"initialized_at": "' || NOW()::TEXT || '", "initialized_by": "setup_script"}')
ON CONFLICT DO NOTHING;

-- Insert default storage locations
INSERT INTO data_storage (storage_id, storage_type, location, capacity, status, details)
VALUES
    ('primary-storage', 'persistent', '/data/primary', 1099511627776, 'active',
     '{"filesystem": "ext4", "redundancy": "raid10", "description": "Primary data storage"}'),
    ('fast-storage', 'local-ssd', '/data/fast', 549755813888, 'active',
     '{"filesystem": "ext4", "description": "Fast processing storage"}'),
    ('archive-storage', 'object', 's3://negative-space-imaging-archive', 10995116277760, 'active',
     '{"provider": "aws", "region": "us-west-2", "description": "Long-term archival storage"}')
ON CONFLICT DO NOTHING;
"""
    }

    # Create database files
    success = True
    for filename, content in db_files.items():
        file_path = database_dir / filename

        try:
            with open(file_path, 'w') as f:
                f.write(content)
            logger.info(f"Created database file: {filename}")
        except Exception as e:
            logger.error(f"Failed to create database file {filename}: {str(e)}")
            success = False

    # Create migrations directory
    migrations_dir = database_dir / "migrations"
    migrations_dir.mkdir(exist_ok=True)

    # Create backups directory
    backups_dir = database_dir / "backups"
    backups_dir.mkdir(exist_ok=True)

    if success:
        logger.info("Database files setup completed successfully")
    else:
        logger.warning("Database files setup completed with errors")

    return success

def setup_database_integration(force_recreate=False):
    """
    Set up database integration for multi-node deployment

    Args:
        force_recreate: If True, recreate configuration files even if they exist

    Returns:
        bool: True if setup was successful, False otherwise
    """
    logger.info("Setting up database integration...")

    deployment_dir = Path("deployment")
    database_dir = deployment_dir / "database"

    # Ensure database directory exists
    database_dir.mkdir(parents=True, exist_ok=True)

    # Create or update database connection configuration
    config_dir = deployment_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    db_config_path = config_dir / "database.yaml"
    if not db_config_path.exists() or force_recreate:
        logger.info("Creating database configuration file...")
        db_config = {
            "database": {
                "host": "db.negative-space-imaging.svc.cluster.local",
                "port": 5432,
                "dbname": "negative_space_imaging",
                "user": "postgres",
                "password": "postgres",  # This should be replaced with a secure password
                "timeout": 30,
                "min_connections": 1,
                "max_connections": 10,
                "schema_file": "deployment/database/01-init-schema.sql",
                "data_file": "deployment/database/02-init-data.sql",
                "migrations_dir": "deployment/database/migrations",
                "backup_dir": "deployment/database/backups",
                "ssl_mode": "prefer"
            }
        }

        try:
            import yaml
            with open(db_config_path, 'w') as f:
                yaml.dump(db_config, f, default_flow_style=False)
            logger.info(f"Created database configuration: {db_config_path}")
        except Exception as e:
            logger.error(f"Failed to create database configuration: {str(e)}")
            return False

    # Copy database deployment scripts
    logger.info("Setting up database deployment scripts...")

    # Source files
    source_files = [
        ("database_connection.py", deployment_dir / "database_connection.py"),
        ("database_deploy.py", deployment_dir / "database_deploy.py"),
        ("hpc_database_integration.py", deployment_dir / "hpc_database_integration.py"),
        ("test_database_deployment.py", deployment_dir / "test_database_deployment.py")
    ]

    # Copy files if they exist
    for source_name, dest_path in source_files:
        source_path = deployment_dir / source_name
        if source_path.exists() and not dest_path.exists():
            try:
                import shutil
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied {source_name} to {dest_path}")
            except Exception as e:
                logger.error(f"Failed to copy {source_name}: {str(e)}")
                # Continue even if one file fails

    # Create database integration script for Kubernetes
    k8s_db_script_path = deployment_dir / "kubernetes" / "deploy-database.yaml"
    os.makedirs(os.path.dirname(k8s_db_script_path), exist_ok=True)

    k8s_db_script_content = """apiVersion: v1
kind: ConfigMap
metadata:
  name: db-init-scripts
data:
  01-init-schema.sql: |
    -- Include schema creation script here
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    -- Create tables if not exists
    CREATE TABLE IF NOT EXISTS system_nodes (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        node_id VARCHAR(100) NOT NULL UNIQUE,
        name VARCHAR(100),
        status VARCHAR(20) NOT NULL DEFAULT 'active',
        node_type VARCHAR(50) DEFAULT 'compute',
        details JSONB,
        last_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
    );

    -- More tables would be included here

  02-init-data.sql: |
    -- Include initial data script here
    INSERT INTO system_events (event_type, severity, message, details)
    VALUES ('system.initialization', 'info', 'Kubernetes database initialized',
           '{"initialized_at": "' || NOW()::TEXT || '", "environment": "kubernetes"}')
    ON CONFLICT DO NOTHING;
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: "postgres"
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:13
        ports:
        - containerPort: 5432
          name: postgres
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_DB
          value: postgres
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        - name: init-scripts
          mountPath: /docker-entrypoint-initdb.d
      volumes:
      - name: init-scripts
        configMap:
          name: db-init-scripts
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: db
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  clusterIP: None
---
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
data:
  password: cG9zdGdyZXM=  # "postgres" in base64, should be replaced in production
"""

    try:
        with open(k8s_db_script_path, 'w') as f:
            f.write(k8s_db_script_content)
        logger.info(f"Created Kubernetes database deployment script: {k8s_db_script_path}")
    except Exception as e:
        logger.error(f"Failed to create Kubernetes database script: {str(e)}")
        return False

    logger.info("Database integration setup completed")
    return True

def setup_template_files(force_recreate=False):
    """
    Set up template files for deployment

    Args:
        force_recreate: If True, recreate all template files even if they exist

    Returns:
        bool: True if setup was successful, False otherwise
    """
    logger.info("Setting up deployment template files...")

    deployment_dir = Path("deployment")
    template_dir = deployment_dir / "templates"

    # Ensure template directory exists
    template_dir.mkdir(parents=True, exist_ok=True)

    # Define template files and their content
    templates = {
        "cluster.yaml.tmpl": """apiVersion: v1
kind: Cluster
metadata:
  name: {{ name }}
  labels:
    environment: {{ environment }}
    version: {{ version }}
spec:
  environment: {{ environment }}
  version: {{ version }}
  description: "Negative Space Imaging Project Multi-Node Cluster"
  managedBy: "Negative Space Imaging Project Deployment Manager"
""",

        "nodes.yaml.tmpl": """apiVersion: v1
kind: NodeSet
metadata:
  name: {{ cluster_name }}-nodes
  labels:
    cluster: {{ cluster_name }}
spec:
  clusterName: {{ cluster_name }}
  nodes:
    # Master nodes
    {{#nodes.master}}
    - name: {{ name }}
      role: master
      resources:
        cpu: {{ resources.cpu }}
        memory: {{ resources.memory }}
      labels:
        nodeType: master
        zone: {{ zone }}
      taints:
        - key: node-role.kubernetes.io/master
          effect: NoSchedule
    {{/nodes.master}}

    # Compute nodes
    {{#nodes.compute}}
    - name: {{ name }}
      role: worker
      resources:
        cpu: {{ resources.cpu }}
        memory: {{ resources.memory }}
        gpu: {{ resources.gpu }}
      labels:
        nodeType: compute
        zone: {{ zone }}
        accelerator: {{ accelerator_type }}
    {{/nodes.compute}}

    # Storage nodes
    {{#nodes.storage}}
    - name: {{ name }}
      role: worker
      resources:
        cpu: {{ resources.cpu }}
        memory: {{ resources.memory }}
        storage: {{ resources.storage }}
      labels:
        nodeType: storage
        zone: {{ zone }}
        storageType: {{ storage_type }}
    {{/nodes.storage}}

    # Edge nodes
    {{#nodes.edge}}
    - name: {{ name }}
      role: worker
      resources:
        cpu: {{ resources.cpu }}
        memory: {{ resources.memory }}
      labels:
        nodeType: edge
        zone: {{ zone }}
        ingress: "true"
    {{/nodes.edge}}

  # Auto-scaling configuration
  autoScaling:
    enabled: {{ auto_scaling.enabled }}
    minNodes: {{ auto_scaling.min_nodes }}
    maxNodes: {{ auto_scaling.max_nodes }}
    scaleUpThreshold: {{ auto_scaling.scale_up_threshold }}
    scaleDownThreshold: {{ auto_scaling.scale_down_threshold }}
""",

        "storage.yaml.tmpl": """apiVersion: v1
kind: StorageClass
metadata:
  name: {{ cluster_name }}-storage
  labels:
    cluster: {{ cluster_name }}
spec:
  # Main storage configuration
  storageClasses:
    # Fast local storage for processing
    - name: {{ cluster_name }}-fast-local
      provisioner: kubernetes.io/local-storage
      reclaimPolicy: Retain
      volumeBindingMode: WaitForFirstConsumer
      parameters:
        type: SSD

    # Persistent network storage
    - name: {{ cluster_name }}-persistent
      provisioner: kubernetes.io/aws-ebs
      reclaimPolicy: Retain
      parameters:
        type: gp2
        fsType: ext4

    # High-performance storage for GPU processing
    - name: {{ cluster_name }}-high-performance
      provisioner: kubernetes.io/local-storage
      reclaimPolicy: Retain
      volumeBindingMode: WaitForFirstConsumer
      parameters:
        type: NVMe

  # Persistent Volume Claims
  persistentVolumeClaims:
    # Data storage
    - name: {{ cluster_name }}-data
      storageClassName: {{ cluster_name }}-persistent
      accessModes:
        - ReadWriteMany
      resources:
        requests:
          storage: {{ storage.data.capacity }}

    # Model storage
    - name: {{ cluster_name }}-models
      storageClassName: {{ cluster_name }}-persistent
      accessModes:
        - ReadOnlyMany
      resources:
        requests:
          storage: {{ storage.models.capacity }}

    # Cache storage
    - name: {{ cluster_name }}-cache
      storageClassName: {{ cluster_name }}-fast-local
      accessModes:
        - ReadWriteMany
      resources:
        requests:
          storage: {{ storage.cache.capacity }}

    # Processing storage
    - name: {{ cluster_name }}-processing
      storageClassName: {{ cluster_name }}-high-performance
      accessModes:
        - ReadWriteMany
      resources:
        requests:
          storage: {{ storage.processing.capacity }}
""",

        "networking.yaml.tmpl": """apiVersion: v1
kind: NetworkPolicy
metadata:
  name: {{ cluster_name }}-network
  labels:
    cluster: {{ cluster_name }}
spec:
  # Network configuration
  networkPolicies:
    # Internal network for communication between services
    - name: {{ cluster_name }}-internal
      podSelector:
        matchLabels:
          cluster: {{ cluster_name }}
      ingress:
        - from:
            - podSelector:
                matchLabels:
                  cluster: {{ cluster_name }}

    # API network for external API access
    - name: {{ cluster_name }}-api
      podSelector:
        matchLabels:
          role: api
      ingress:
        - from:
            - ipBlock:
                cidr: {{ networking.api_access_cidr }}
          ports:
            - protocol: TCP
              port: 443

    # Data transfer network for high-bandwidth internal transfers
    - name: {{ cluster_name }}-data-transfer
      podSelector:
        matchLabels:
          role: data-processor
      ingress:
        - from:
            - podSelector:
                matchLabels:
                  role: data-storage
          ports:
            - protocol: TCP
              port: 8000

  # Service mesh configuration
  serviceMesh:
    enabled: {{ networking.service_mesh.enabled }}
    implementation: {{ networking.service_mesh.implementation }}
    autoInject: {{ networking.service_mesh.auto_inject }}
    mtls:
      enabled: {{ networking.service_mesh.mtls.enabled }}
      mode: {{ networking.service_mesh.mtls.mode }}

  # Load balancer configuration
  loadBalancer:
    enabled: {{ networking.load_balancer.enabled }}
    type: {{ networking.load_balancer.type }}
    implementation: {{ networking.load_balancer.implementation }}
    config:
      healthCheckPath: /health
      healthCheckPort: 8080
      healthCheckInterval: 30s
      sessionAffinity: {{ networking.load_balancer.session_affinity }}
""",

        "services.yaml.tmpl": """apiVersion: v1
kind: ServiceDeployment
metadata:
  name: {{ cluster_name }}-services
  labels:
    cluster: {{ cluster_name }}
spec:
  services:
    # API Gateway Service
    - name: {{ cluster_name }}-api-gateway
      image: {{ services.api_gateway.image }}
      replicas: {{ services.api_gateway.replicas }}
      resources:
        limits:
          cpu: {{ services.api_gateway.resources.limits.cpu }}
          memory: {{ services.api_gateway.resources.limits.memory }}
        requests:
          cpu: {{ services.api_gateway.resources.requests.cpu }}
          memory: {{ services.api_gateway.resources.requests.memory }}
      ports:
        - name: http
          containerPort: 8080
          servicePort: 80
        - name: https
          containerPort: 8443
          servicePort: 443
      environment:
        {{#services.api_gateway.environment_variables}}
        - name: {{ name }}
          value: {{ value }}
        {{/services.api_gateway.environment_variables}}
      healthCheck:
        path: /health
        port: 8080
        initialDelaySeconds: 30
        periodSeconds: 10
      nodeSelector:
        nodeType: edge

    # Image Processing Service
    - name: {{ cluster_name }}-image-processing
      image: {{ services.image_processing.image }}
      replicas: {{ services.image_processing.replicas }}
      resources:
        limits:
          cpu: {{ services.image_processing.resources.limits.cpu }}
          memory: {{ services.image_processing.resources.limits.memory }}
          nvidia.com/gpu: {{ services.image_processing.resources.limits.gpu }}
        requests:
          cpu: {{ services.image_processing.resources.requests.cpu }}
          memory: {{ services.image_processing.resources.requests.memory }}
          nvidia.com/gpu: {{ services.image_processing.resources.requests.gpu }}
      ports:
        - name: grpc
          containerPort: 9000
          servicePort: 9000
      environment:
        {{#services.image_processing.environment_variables}}
        - name: {{ name }}
          value: {{ value }}
        {{/services.image_processing.environment_variables}}
      volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        - name: processing
          mountPath: /app/processing
        - name: data
          mountPath: /app/data
      healthCheck:
        path: /health
        port: 8080
        initialDelaySeconds: 60
        periodSeconds: 15
      nodeSelector:
        nodeType: compute

    # Additional services omitted for brevity
""",

        "monitoring.yaml.tmpl": """apiVersion: v1
kind: MonitoringConfig
metadata:
  name: {{ cluster_name }}-monitoring
  labels:
    cluster: {{ cluster_name }}
spec:
  # Prometheus configuration
  prometheus:
    enabled: {{ monitoring.prometheus.enabled }}
    retention: {{ monitoring.prometheus.retention }}
    scrapeInterval: {{ monitoring.prometheus.scrape_interval }}
    resources:
      limits:
        cpu: {{ monitoring.prometheus.resources.limits.cpu }}
        memory: {{ monitoring.prometheus.resources.limits.memory }}
      requests:
        cpu: {{ monitoring.prometheus.resources.requests.cpu }}
        memory: {{ monitoring.prometheus.resources.requests.memory }}
    storage:
      size: {{ monitoring.prometheus.storage.size }}
      storageClassName: {{ cluster_name }}-persistent
    alerting:
      enabled: {{ monitoring.prometheus.alerting.enabled }}
      rules:
        - name: HighCpuUsage
          expr: avg(rate(container_cpu_usage_seconds_total{namespace="{{ cluster_name }}"}[5m])) by (pod) > 0.8
          for: 5m
          severity: warning
          summary: High CPU usage detected

  # Grafana configuration
  grafana:
    enabled: {{ monitoring.grafana.enabled }}
    adminPassword: "{{ monitoring.grafana.admin_password }}"
    resources:
      limits:
        cpu: {{ monitoring.grafana.resources.limits.cpu }}
        memory: {{ monitoring.grafana.resources.limits.memory }}
      requests:
        cpu: {{ monitoring.grafana.resources.requests.cpu }}
        memory: {{ monitoring.grafana.resources.requests.memory }}
    dashboards:
      - name: ClusterOverview
        configMap: {{ cluster_name }}-grafana-dashboards
        fileName: cluster-overview.json
"""
    }

    # Create each template file
    success = True
    for template_name, template_content in templates.items():
        template_path = template_dir / template_name

        if template_path.exists() and not force_recreate:
            logger.info(f"Template file {template_name} already exists, skipping")
            continue

        try:
            with open(template_path, 'w') as f:
                f.write(template_content)
            logger.info(f"Created template file: {template_name}")
        except Exception as e:
            logger.error(f"Failed to create template file {template_name}: {str(e)}")
            success = False

    if success:
        logger.info("All template files created successfully")
    else:
        logger.warning("Some template files could not be created")

    return success

def create_example_config():
    """
    Create an example configuration file for deployment

    Returns:
        bool: True if creation was successful, False otherwise
    """
    logger.info("Creating example configuration file...")

    deployment_dir = Path("deployment")
    config_dir = deployment_dir / "config"

    # Ensure config directory exists
    config_dir.mkdir(parents=True, exist_ok=True)

    # Define example configuration
    example_config = {
        "cluster": {
            "name": "negative-space-imaging",
            "environment": "development",
            "version": "1.0.0",
            "region": "us-west-2",
            "description": "Negative Space Imaging Multi-Node Cluster"
        },
        "nodes": {
            "master": [
                {
                    "name": "master-1",
                    "resources": {
                        "cpu": "2",
                        "memory": "4Gi"
                    },
                    "zone": "us-west-2a"
                }
            ],
            "compute": [
                {
                    "name": "compute-1",
                    "resources": {
                        "cpu": "4",
                        "memory": "16Gi",
                        "gpu": "1"
                    },
                    "zone": "us-west-2b",
                    "accelerator_type": "nvidia-tesla-t4"
                },
                {
                    "name": "compute-2",
                    "resources": {
                        "cpu": "8",
                        "memory": "32Gi",
                        "gpu": "2"
                    },
                    "zone": "us-west-2c",
                    "accelerator_type": "nvidia-tesla-v100"
                }
            ],
            "storage": [
                {
                    "name": "storage-1",
                    "resources": {
                        "cpu": "2",
                        "memory": "8Gi",
                        "storage": "1Ti"
                    },
                    "zone": "us-west-2a",
                    "storage_type": "fast-ssd"
                }
            ],
            "edge": [
                {
                    "name": "edge-1",
                    "resources": {
                        "cpu": "2",
                        "memory": "4Gi"
                    },
                    "zone": "us-west-2a"
                }
            ]
        },
        "auto_scaling": {
            "enabled": True,
            "min_nodes": 3,
            "max_nodes": 10,
            "scale_up_threshold": 75,
            "scale_down_threshold": 30
        },
        "storage": {
            "data": {
                "capacity": "500Gi"
            },
            "models": {
                "capacity": "100Gi"
            },
            "cache": {
                "capacity": "50Gi"
            },
            "processing": {
                "capacity": "200Gi"
            }
        },
        "networking": {
            "api_access_cidr": "0.0.0.0/0",
            "service_mesh": {
                "enabled": True,
                "implementation": "istio",
                "auto_inject": True,
                "mtls": {
                    "enabled": True,
                    "mode": "STRICT"
                }
            },
            "load_balancer": {
                "enabled": True,
                "type": "LoadBalancer",
                "implementation": "cloud",
                "session_affinity": "ClientIP"
            }
        },
        "services": {
            "api_gateway": {
                "image": "negative-space-imaging/api-gateway:latest",
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
                        "name": "PORT",
                        "value": "8080"
                    },
                    {
                        "name": "LOG_LEVEL",
                        "value": "info"
                    }
                ]
            },
            "image_processing": {
                "image": "negative-space-imaging/image-processing:latest",
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
                        "name": "CUDA_VISIBLE_DEVICES",
                        "value": "0"
                    },
                    {
                        "name": "MODEL_PATH",
                        "value": "/app/models"
                    }
                ]
            }
        },
        "monitoring": {
            "prometheus": {
                "enabled": True,
                "retention": "15d",
                "scrape_interval": "15s",
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
                "storage": {
                    "size": "50Gi"
                },
                "alerting": {
                    "enabled": True
                }
            },
            "grafana": {
                "enabled": True,
                "admin_password": "admin",
                "resources": {
                    "limits": {
                        "cpu": "0.5",
                        "memory": "1Gi"
                    },
                    "requests": {
                        "cpu": "0.2",
                        "memory": "512Mi"
                    }
                }
            },
            "node_exporter": {
                "enabled": True,
                "resources": {
                    "limits": {
                        "cpu": "0.2",
                        "memory": "256Mi"
                    },
                    "requests": {
                        "cpu": "0.1",
                        "memory": "128Mi"
                    }
                }
            },
            "application_metrics": {
                "enabled": True,
                "scrape_interval": "15s"
            }
        },
        "logging": {
            "loki": {
                "enabled": True,
                "retention": "7d",
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
                "storage": {
                    "size": "50Gi"
                }
            },
            "fluentd": {
                "enabled": True,
                "resources": {
                    "limits": {
                        "cpu": "0.5",
                        "memory": "1Gi"
                    },
                    "requests": {
                        "cpu": "0.2",
                        "memory": "512Mi"
                    }
                }
            }
        },
        "backup": {
            "enabled": True,
            "schedule": "0 0 * * *",
            "retention": "30d",
            "storage": {
                "provider": "s3",
                "bucket": "negative-space-imaging-backups",
                "region": "us-west-2"
            }
        },
        "security": {
            "rbac": {
                "enabled": True
            },
            "network_policies": {
                "enabled": True,
                "default_deny": True
            },
            "pod_security_policies": {
                "enabled": True
            },
            "secrets_encryption": {
                "enabled": True,
                "provider": "vault"
            }
        }
    }

    example_config_path = config_dir / "example-config.yaml"
    try:
        import yaml
        with open(example_config_path, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False)
        logger.info(f"Created example configuration: {example_config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create example configuration: {str(e)}")
        return False

def main():
    """
    Main function to set up the multi-node deployment environment
    Parses command line arguments and runs the setup process

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(description='Set up multi-node deployment for Negative Space Imaging Project')
    parser.add_argument('--check-only', action='store_true', help='Only check prerequisites without setting up')
    parser.add_argument('--force', action='store_true', help='Force recreation of existing files')
    parser.add_argument('--skip-database', action='store_true', help='Skip database setup')
    parser.add_argument('--skip-db-integration', action='store_true', help='Skip database integration setup')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    logger.info("Starting multi-node deployment setup")

    try:
        # Check prerequisites
        if not check_prerequisites():
            logger.error("Prerequisites check failed")
            return 1

        if args.check_only:
            logger.info("Prerequisites check completed successfully")
            return 0

        # Set up template files
        if not setup_template_files(force_recreate=args.force):
            logger.error("Template files setup failed")
            return 1

        # Setup database files
        if not args.skip_database:
            if not setup_database_files():
                logger.error("Database files setup failed")
                return 1
        else:
            logger.info("Skipping database files setup")

        # Setup database integration
        if not args.skip_database and not args.skip_db_integration:
            if not setup_database_integration(force_recreate=args.force):
                logger.error("Database integration setup failed")
                return 1
        else:
            logger.info("Skipping database integration setup")

        # Create example configuration
        if not create_example_config():
            logger.error("Example configuration creation failed")
            return 1

        logger.info("Deployment setup completed successfully")
        logger.info("You can now create your own configuration file based on the example")
        logger.info("and run the deployment with: python deployment/multi_node_deploy.py")

        # Print database information if database was set up
        if not args.skip_database:
            logger.info("\nDatabase Integration:")
            logger.info("- Database configuration is at: deployment/config/database.yaml")
            logger.info("- Use the database deployment tools:")
            logger.info("  * python deployment/database_deploy.py --deploy    : Deploy the database")
            logger.info("  * python deployment/database_deploy.py --verify    : Verify the database")
            logger.info("  * python deployment/database_deploy.py --migrate   : Run migrations")
            logger.info("  * python deployment/database_deploy.py --backup    : Backup the database")
            logger.info("  * python deployment/test_database_deployment.py    : Test database setup")

        return 0
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
