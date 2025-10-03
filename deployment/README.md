# Multi-Node Deployment System for Negative Space Imaging Project

This system enables the deployment of the Negative Space Imaging Project across multiple nodes, providing high availability, scalability, and performance optimization for large-scale deployments.

## Overview

The multi-node deployment system provides:

- Automated deployment of Negative Space Imaging components across a cluster
- Support for Kubernetes and Docker Swarm orchestration
- Configuration-driven setup with customizable node types
- Integration with existing security and HPC components
- Monitoring, logging, and alerting infrastructure
- Disaster recovery and backup capabilities

## Components

The multi-node deployment system consists of the following components:

1. **Multi-Node Configuration**: YAML file defining the cluster architecture
2. **Deployment Manager**: Python script that handles the deployment process
3. **Template Files**: Templates for generating Kubernetes/Docker configurations
4. **Integration Module**: Integrates with existing system components
5. **CLI Integration**: Command-line interface for managing deployments
6. **Docker Build System**: Builds container images for deployment

## Node Types

The system supports the following node types:

- **Master Nodes**: Control plane nodes that manage the cluster
- **Compute Nodes**: High-performance nodes with optional GPU acceleration
- **Storage Nodes**: Optimized for data storage and retrieval
- **Edge Nodes**: Handle external API requests and user interfaces

## Getting Started

### Prerequisites

- Python 3.7+
- Docker (for Docker Swarm deployment)
- Kubernetes tools (for Kubernetes deployment)
- PyYAML package

### Setup

1. Run the setup script to prepare the environment:

```bash
python cli.py multi-node setup
```

2. Create or modify the configuration file:

```bash
# Modify the example configuration
nano deployment/example_multi_node_config.yaml

# Save as your own configuration
cp deployment/example_multi_node_config.yaml deployment/multi_node_config.yaml
```

3. Build Docker images for deployment:

```bash
python deployment/docker_build.py --build
```

### Docker Compose Deployment

For development or small-scale deployments, you can use Docker Compose:

```bash
# Deploy with Docker Compose
python deployment/deploy.py --type docker-compose --environment development

# Build images before deployment
python deployment/deploy.py --type docker-compose --build

# Check deployment status
python deployment/deploy.py --type docker-compose --status

# Stop and clean up deployment
python deployment/deploy.py --type docker-compose --stop
```

### Kubernetes Deployment

For production or large-scale deployments, use Kubernetes:

```bash
# Deploy with Kubernetes
python deployment/deploy.py --type kubernetes --environment production --namespace nsi-prod

# Check deployment status
python deployment/deploy.py --type kubernetes --status --namespace nsi-prod

# Stop and clean up deployment
python deployment/deploy.py --type kubernetes --stop --namespace nsi-prod
```

### Integration

Integrate with existing system components:

```bash
python cli.py multi-node integrate
```

### Status

Check the status of the deployment:

```bash
python cli.py multi-node status
```

## Configuration Reference

The `multi_node_config.yaml` file contains the following sections:

- `cluster`: General cluster configuration
- `nodes`: Node configuration for different types
- `storage`: Storage configuration
- `networking`: Network and service mesh configuration
- `services`: Service definitions
- `monitoring`: Monitoring and alerting configuration
- `logging`: Logging configuration
- `backup`: Backup configuration
- `disaster_recovery`: Disaster recovery configuration
- `security_settings`: Security configuration

## Monitoring and Logging

The deployment includes comprehensive monitoring and logging:

- **Prometheus**: Collects metrics from all services
- **Grafana**: Provides visualization and dashboards
- **Loki**: Aggregates logs from all services
- **Promtail**: Collects and forwards logs to Loki

Access the monitoring interfaces at:
- Grafana: http://localhost:3000 (default admin/changeme)
- Prometheus: http://localhost:9090

### Dashboard Access

After deployment, access the monitoring dashboard through:

```bash
# Open in browser (Kubernetes)
kubectl port-forward -n negative-space-imaging svc/grafana 3000:3000

# Direct access (Docker Compose)
# Open http://localhost:3000 in your browser
```

## Advanced Usage

### Custom Templates

You can create custom templates in the `deployment/templates` directory to customize the deployment:

- `cluster.yaml.tmpl`: Cluster configuration template
- `nodes.yaml.tmpl`: Node configuration template
- `storage.yaml.tmpl`: Storage configuration template
- `networking.yaml.tmpl`: Networking configuration template
- `services.yaml.tmpl`: Services configuration template
- `monitoring.yaml.tmpl`: Monitoring configuration template

### Custom Docker Images

You can customize the Docker images by modifying the Dockerfiles in the `deployment/dockerfiles` directory:

- `api-gateway.Dockerfile`: API Gateway service
- `image-processing.Dockerfile`: Image processing service
- `data-storage.Dockerfile`: Data storage service
- `distributed-computing.Dockerfile`: Distributed computing service
- `security.Dockerfile`: Security service

## Troubleshooting

Common issues and solutions:

1. **Deployment Fails**:
   - Check the deployment logs: `deployment.log` and `deployment_setup.log`
   - Verify all prerequisites are installed: `python deployment/deploy.py --check-only`
   - Ensure configuration files are valid: `python deployment/setup_multi_node.py --validate`

2. **Container Startup Issues**:
   - Check container logs: `docker-compose -f deployment/docker-compose.yaml logs <service>`
   - Check Kubernetes pod logs: `kubectl logs -n <namespace> <pod-name>`
   - Verify resource limits are appropriate for your environment

3. **Networking Issues**:
   - Verify service connectivity: `python deployment/deploy.py --status`
   - Check network policies and firewall rules
   - Ensure ports are accessible: `curl -v http://localhost:<port>/health`

4. **Monitoring Issues**:
   - Verify Prometheus is running: `curl -v http://localhost:9090/-/healthy`
   - Check Grafana connectivity: `curl -v http://localhost:3000/api/health`
   - Ensure metrics are being scraped in Prometheus targets

5. **Performance Issues**:
   - Review Grafana dashboards for bottlenecks
   - Check resource utilization: `python deployment/deploy.py --status`
   - Adjust resource allocations in configuration file

## Contributing

Contributions to the multi-node deployment system are welcome. Please follow the guidelines in `CONTRIBUTING.md`.
