# Negative Space Imaging Project - Deployment System

This directory contains the deployment system for the Negative Space Imaging Project, which provides tools for deploying, monitoring, testing, and managing the application in both development and production environments.

## Overview

The deployment system provides:

- Docker Compose configuration for local development
- Kubernetes manifests for production deployment
- Monitoring infrastructure with Prometheus and Grafana
- Centralized logging with Loki and Promtail
- Database configuration and initialization
- Deployment automation scripts
- Health and verification tools
- Testing utilities

## Directory Structure

```
deployment/
├── docker-compose.yaml            # Docker Compose configuration
├── deploy.py                      # Main deployment script
├── deploy_auto.py                 # Automated deployment script
├── health_check.py                # Deployment health check tool
├── verify_deployment.py           # Configuration verification tool
├── test_deployment.py             # Integration test script
├── monitoring_dashboard.py        # Monitoring dashboard access tool
├── dockerfiles/                   # Docker image definitions
├── kubernetes/                    # Kubernetes manifests
├── monitoring/                    # Monitoring configuration
│   ├── prometheus.yml             # Prometheus configuration
│   ├── dashboards/                # Grafana dashboards
│   ├── datasources/               # Grafana datasources
│   └── dashboard-providers/       # Grafana dashboard providers
├── logging/                       # Logging configuration
│   ├── loki-config.yaml           # Loki configuration
│   └── promtail-config.yaml       # Promtail configuration
├── database/                      # Database configuration
│   ├── 01-init-schema.sql         # Database schema initialization
│   ├── 02-init-data.sql           # Initial data population
│   └── init-database.sh           # Database initialization script
└── secrets/                       # Secret management
    └── db_password.txt            # Database password (example)
```

## Getting Started

### Prerequisites

- Docker and Docker Compose for local development
- Kubernetes cluster for production deployment
- `kubectl` command-line tool for Kubernetes management
- Python 3.7+ for running the deployment scripts

### Deployment Options

#### Local Development with Docker Compose

```bash
# Deploy all services
python deploy.py deploy --type docker-compose

# Check deployment status
python health_check.py --deployment-type docker-compose

# Access monitoring dashboards
python monitoring_dashboard.py --type docker-compose

# Test the deployment
python test_deployment.py --type docker-compose

# Clean up deployment
python deploy.py cleanup --type docker-compose
```

#### Production Deployment with Kubernetes

```bash
# Deploy to Kubernetes
python deploy.py deploy --type kubernetes --namespace negative-space-imaging

# Check deployment health
python health_check.py --deployment-type kubernetes --namespace negative-space-imaging

# Access monitoring dashboards
python monitoring_dashboard.py --type kubernetes --namespace negative-space-imaging

# Test the deployment
python test_deployment.py --type kubernetes --namespace negative-space-imaging

# Clean up deployment
python deploy.py cleanup --type kubernetes --namespace negative-space-imaging
```

## Key Features

### Deployment Automation

The `deploy.py` and `deploy_auto.py` scripts provide automated deployment capabilities:

- Support for both Docker Compose and Kubernetes
- Environment-specific configuration
- Service dependencies management
- Cleanup and status reporting

### Health Monitoring

The `health_check.py` script provides real-time health information about the deployed services:

- Real-time service status monitoring
- Detailed health reports
- Watch mode for continuous monitoring
- Support for both Docker Compose and Kubernetes

### Configuration Verification

The `verify_deployment.py` script validates the deployment configuration:

- Checks for configuration consistency
- Validates required services and dependencies
- Ensures monitoring and logging configuration
- Cross-component verification

### Integration Testing

The `test_deployment.py` script runs integration tests against a deployed instance:

- API connectivity testing
- Database connectivity verification
- Monitoring service validation
- Logging service validation
- Basic workflow testing

### Monitoring Dashboard Access

The `monitoring_dashboard.py` script provides easy access to monitoring dashboards:

- Automatic URL detection
- Port forwarding for Kubernetes
- Browser launch for dashboard access

## Advanced Configuration

### Customizing Docker Compose

Edit the `docker-compose.yaml` file to customize:

- Service configurations
- Resource limits
- Volumes and networks
- Environment variables

### Customizing Kubernetes Deployment

Edit the Kubernetes manifests in the `kubernetes/` directory to customize:

- Deployment configurations
- Service definitions
- Resource quotas
- ConfigMaps and Secrets

### Monitoring Configuration

Edit the monitoring configuration files to customize:

- Prometheus scrape targets
- Grafana dashboards
- Alert rules
- Retention policies

### Logging Configuration

Edit the logging configuration files to customize:

- Log collection rules
- Log retention
- Log formats

### Database Configuration

Edit the database initialization scripts to customize:

- Database schema
- Initial data
- User permissions

## Troubleshooting

### Common Issues

1. **Docker Compose services fail to start**
   - Check service dependencies
   - Verify port availability
   - Check resource constraints

2. **Kubernetes pods stuck in pending state**
   - Check resource quotas
   - Verify persistent volume claims
   - Check node availability

3. **Monitoring dashboards not accessible**
   - Verify service connectivity
   - Check port forwarding
   - Ensure proper configuration

4. **Database connectivity issues**
   - Check database initialization
   - Verify credentials
   - Check network connectivity

### Diagnostic Tools

- Use `health_check.py` to identify service health issues
- Use `verify_deployment.py` to validate configuration
- Use `test_deployment.py` to test system functionality
- Check service logs for detailed error information

## Maintenance

### Updating Deployments

To update a deployment:

1. Update the configuration files
2. Verify the configuration with `verify_deployment.py`
3. Apply the changes with `deploy.py`
4. Check the health with `health_check.py`
5. Test the deployment with `test_deployment.py`

### Backup and Restore

- Database data is persisted in Docker volumes or Kubernetes persistent volumes
- Use standard database backup and restore procedures
- Configuration files should be version controlled

## Security Considerations

- Sensitive information is stored in the `secrets/` directory
- Database passwords and API keys should be properly secured
- Follow the principle of least privilege for service accounts
- Use network policies to restrict service communication

## Further Reading

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/grafana/latest/)
- [Loki Documentation](https://grafana.com/docs/loki/latest/)
