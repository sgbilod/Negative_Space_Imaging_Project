# Performance Monitoring System for Negative Space Imaging Project

This directory contains the configuration and deployment files for the comprehensive performance monitoring system. The system provides real-time insights into system resources, application performance, database operations, and high-performance computing workloads.

## Quick Start

To get started with the performance monitoring system, run:

```bash
python performance_cli.py start
```

Then access the dashboards at:
- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090

## Components

The performance monitoring system consists of the following components:

1. **Docker-based Monitoring Stack**
   - Prometheus for metrics collection and storage
   - Grafana for dashboards and visualization
   - Alert Manager for notifications

2. **Performance CLI**
   - Command-line interface for managing the monitoring system
   - View real-time metrics from the terminal
   - Export metrics to JSON or CSV

3. **Configuration Files**
   - `performance_monitoring_config.yaml`: Main configuration file
   - `docker-compose.performance.yml`: Docker Compose configuration
   - Prometheus and Grafana configuration files

## Available Commands

The Performance CLI provides the following commands:

```bash
# Check if monitoring system is running
python performance_cli.py status

# Start monitoring system
python performance_cli.py start

# Stop monitoring system
python performance_cli.py stop

# Restart monitoring system
python performance_cli.py restart

# View current metrics
python performance_cli.py metrics

# Export metrics to JSON
python performance_cli.py metrics --export json --output metrics.json

# Export metrics to CSV
python performance_cli.py metrics --export csv --output metrics.csv
```

## System Requirements

- Docker and Docker Compose
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 10GB free disk space

## Detailed Documentation

For more detailed information about the performance monitoring system, refer to:

- [Performance Monitoring System Documentation](PERFORMANCE_MONITORING.md)
- [Docker Compose Configuration](docker-compose.performance.yml)
- [Configuration File Reference](performance_monitoring_config.yaml)

## Integration with Deployment

The performance monitoring system integrates with the main deployment process. When deploying the system, you can enable performance monitoring by setting `use_docker_monitoring: true` in the monitoring section of your project configuration.

## Troubleshooting

If you encounter issues with the performance monitoring system:

1. Check if Docker services are running: `docker-compose -f docker-compose.performance.yml ps`
2. View Docker container logs: `docker-compose -f docker-compose.performance.yml logs monitoring`
3. Verify that Prometheus is accessible: http://localhost:9090/-/healthy
4. Check Grafana health: http://localhost:3001/api/health

## License

This performance monitoring system is licensed under the same terms as the main Negative Space Imaging Project.
