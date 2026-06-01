# Docker Deployment Guide - Negative Space Imaging Project

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Development Deployment](#development-deployment)
3. [Production Deployment](#production-deployment)
4. [Health Checks & Verification](#health-checks--verification)
5. [Troubleshooting](#troubleshooting)
6. [Scaling & Performance](#scaling--performance)

---

## Prerequisites

### System Requirements
- **Docker:** 20.10+ (`docker --version`)
- **Docker Compose:** 1.29+ (`docker-compose --version`)
- **Memory:** 4GB minimum (8GB recommended)
- **Disk Space:** 20GB for images and volumes
- **OS:** Linux, macOS, or Windows (with WSL2)

### Port Requirements
Ensure these ports are available:
- **80, 443:** Frontend (Nginx)
- **3000:** Express API
- **3001:** Grafana
- **5432:** PostgreSQL
- **6379:** Redis
- **8000:** Python Service
- **9090:** Prometheus

### Environment Setup
```bash
# Clone or navigate to project
cd Negative_Space_Imaging_Project

# Copy environment template
cp .env.example .env

# For production, copy and update
cp .env.example .env.prod

# Create required directories
mkdir -p uploads shared_data logs monitoring
```

---

## Development Deployment

### Quick Start (5 minutes)

```bash
# 1. Initialize environment
./scripts/docker-init.sh

# 2. Build images
docker-compose build

# 3. Start services
./scripts/docker-up.sh

# 4. Verify health
./scripts/docker-test.sh

# 5. Access services
# Frontend:     http://localhost
# API:          http://localhost:3000
# Grafana:      http://localhost:3001
```

### Detailed Steps

#### 1. Environment Initialization
```bash
./scripts/docker-init.sh
# This will:
# - Check Docker/Docker Compose installation
# - Create .env file from template
# - Create required directories
# - Validate docker-compose configuration
```

#### 2. Build Docker Images
```bash
# Build all images
docker-compose build

# Build specific image
docker-compose build python_service
docker-compose build api
docker-compose build frontend

# Build without cache
docker-compose build --no-cache
```

#### 3. Start Services
```bash
# Start in detached mode (background)
docker-compose up -d

# Start in foreground with logs
docker-compose up

# Start specific service
docker-compose up -d postgres redis
```

#### 4. Monitor Services
```bash
# View all services
docker-compose ps

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api

# View last 100 lines
docker-compose logs --tail=100 api

# View real-time resource usage
docker stats
```

#### 5. Development Workflows

**Restarting a service after code changes:**
```bash
# Rebuild and restart (with code changes)
docker-compose up -d --build api

# Just restart without rebuild
docker-compose restart api

# Rebuild from scratch
docker-compose down -v
docker-compose build
docker-compose up -d
```

**Debugging a service:**
```bash
# Enter container shell
docker exec -it nsi_api /bin/sh

# View detailed logs with timestamps
docker-compose logs --timestamps --tail=1000 api

# Inspect network connectivity
docker exec nsi_api curl http://python_service:8000/health
```

---

## Production Deployment

### Pre-Deployment Checklist
- [ ] Update `.env.prod` with production values
- [ ] Set strong passwords for DB_PASSWORD, REDIS_PASSWORD, JWT_SECRET
- [ ] Configure DOMAIN variable
- [ ] Set PYTHON_WORKERS based on CPU cores
- [ ] Review resource limits in docker-compose.prod.yml
- [ ] Set up SSL certificates
- [ ] Configure backup strategy
- [ ] Plan monitoring and alerting

### Production Deployment Steps

#### 1. Prepare Production Environment
```bash
# Copy and configure production environment
cp .env.example .env.prod

# Edit with production values
nano .env.prod

# Required values to set:
# - DB_PASSWORD_PROD
# - REDIS_PASSWORD
# - JWT_SECRET
# - GRAFANA_PASSWORD
# - DOMAIN
```

#### 2. Build Production Images
```bash
# Build with production tags
./scripts/docker-build.sh prod

# Verify images
docker images | grep nsi-
```

#### 3. Start Production Services
```bash
# Start production environment (with health checks)
./scripts/docker-up.sh prod

# Verify all services are healthy
docker-compose -f docker-compose.yml -f docker-compose.prod.yml ps

# Run comprehensive tests
./scripts/docker-test.sh --verbose
```

#### 4. Verify Production Deployment

**Check service status:**
```bash
# List all services with health status
docker-compose -f docker-compose.yml -f docker-compose.prod.yml ps

# Check individual service health
curl https://negative-space.local/health
curl https://negative-space.local/api/health
```

**Verify database:**
```bash
# Connect to PostgreSQL
docker exec -it nsi_postgres psql -U nsi_prod_user -d negative_space_prod

# Check tables
\dt

# Exit
\q
```

**Check monitoring:**
```bash
# Access Grafana
# https://monitoring.negative-space.local
# Username: admin
# Password: ${GRAFANA_PASSWORD}

# Access Prometheus
# https://negative-space.local:9090
```

### Production Configuration Details

#### Resource Limits (from docker-compose.prod.yml)

| Service | CPU Limit | Memory Limit | CPU Reserved | Memory Reserved |
|---------|-----------|--------------|--------------|-----------------|
| PostgreSQL | 2 | 2GB | 1 | 1GB |
| Redis | 1 | 1GB | 0.5 | 512MB |
| Python | 4 | 4GB | 2 | 2GB |
| API | 2 | 2GB | 1 | 1GB |
| Frontend | 1 | 512MB | 0.5 | 256MB |
| Prometheus | 1 | 512MB | 0.5 | 256MB |
| Grafana | 1 | 512MB | 0.5 | 256MB |

#### Health Check Configuration

All services use the following health check pattern:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:PORT/health"]
  interval: 60s           # Check every 60s in production
  timeout: 10s            # Wait 10s for response
  retries: 3              # Fail after 3 retries
  start_period: 30s       # Give 30s startup time
```

#### Logging Configuration

Production logging uses JSON driver with rotation:
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "100m"      # Rotate at 100MB
    max-file: "10"        # Keep 10 rotated files (1GB total)
```

---

## Health Checks & Verification

### Manual Health Verification

```bash
# Test all services
./scripts/docker-test.sh

# Verbose output with resource usage
./scripts/docker-test.sh --verbose

# Quick endpoint checks
curl http://localhost/health              # Frontend
curl http://localhost:3000/health         # API
curl http://localhost:8000/health         # Python
curl http://localhost:9090/-/healthy      # Prometheus
curl http://localhost:3001/api/health     # Grafana
```

### Service Communication Tests

```bash
# API can reach Python service
docker exec nsi_api curl http://python_service:8000/health

# API can reach database
docker exec nsi_api bash -c 'psql -h postgres -U nsi_admin -c "SELECT 1"'

# API can reach Redis
docker exec nsi_api redis-cli -h redis ping

# Frontend can reach API
docker exec nsi_frontend curl http://api:3000/health
```

### Performance Monitoring

```bash
# Real-time resource usage
docker stats

# Container event log
docker events

# Network inspection
docker network inspect nsi_network

# Volume inspection
docker volume inspect nsi_postgres_data
```

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port
lsof -i :3000

# Kill process
kill -9 <PID>

# Or change port in .env
PORT=3001
docker-compose restart api
```

#### 2. Database Connection Failed
```bash
# Check PostgreSQL is healthy
docker-compose ps postgres

# View PostgreSQL logs
docker-compose logs postgres

# Verify credentials in .env
grep DB_ .env

# Test connection
docker exec nsi_postgres pg_isready
```

#### 3. Redis Connection Issues
```bash
# Check Redis is running
docker-compose ps redis

# Test Redis connection
docker exec nsi_redis redis-cli ping

# Check Redis config
docker exec nsi_redis redis-cli CONFIG GET maxmemory
```

#### 4. Service Won't Start
```bash
# View detailed logs
docker-compose logs <service_name>

# Check resource availability
docker stats

# Check free disk space
df -h

# Rebuild service
docker-compose build --no-cache <service_name>
```

#### 5. Memory/CPU Issues
```bash
# Monitor resource usage
docker stats --no-stream

# Increase resource limits in docker-compose.prod.yml
# Or add Docker memory limit: docker run --memory=2gb

# View which containers are using most resources
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

### Debugging Commands

```bash
# Enter service container
docker exec -it nsi_api /bin/sh

# View environment variables
docker exec nsi_api env | grep DB_

# Check network connectivity
docker network inspect nsi_network

# View container logs with timestamps
docker-compose logs --timestamps api

# Monitor container startup
docker events --filter type=container

# Clean up and start fresh
docker-compose down -v
docker system prune -a
docker-compose up -d
```

---

## Scaling & Performance

### Horizontal Scaling

#### Multiple API Instances (Production)
```yaml
# In docker-compose.prod.yml
api:
  deploy:
    replicas: 3    # Run 3 instances
    resources:
      limits:
        cpus: '2'
        memory: 2G
```

#### Database Connection Pooling
```bash
# In .env.prod
DB_POOL_MAX=50      # Maximum connections
DB_POOL_MIN=10      # Minimum connections
```

### Performance Optimization

#### Python Service Optimization
```bash
# Increase worker count based on CPU cores
# In .env.prod
PYTHON_WORKERS=8    # For 4-core system: cores * 2

# Monitor Python service performance
docker exec nsi_python ps aux | grep gunicorn
```

#### Redis Optimization
```bash
# Monitor Redis memory usage
docker exec nsi_redis redis-cli INFO memory

# Check Redis eviction policy
docker exec nsi_redis redis-cli CONFIG GET maxmemory-policy

# Set max memory
docker exec nsi_redis redis-cli CONFIG SET maxmemory 512mb
```

#### PostgreSQL Optimization
```bash
# Check connection count
docker exec nsi_postgres psql -U nsi_admin -d negative_space -c "SELECT count(*) FROM pg_stat_activity;"

# View slow queries (if configured)
docker exec nsi_postgres psql -U nsi_admin -d negative_space -c "SELECT * FROM pg_stat_statements LIMIT 10;"
```

### Monitoring Best Practices

```bash
# Set up Prometheus scraping
# Configure targets in monitoring/prometheus.yml

# Create Grafana dashboards
# Import pre-built dashboards for:
# - Docker Container metrics
# - PostgreSQL metrics
# - Redis metrics
# - Python application metrics

# Set up alerting
# Configure alert rules in Prometheus
# Send to Slack, PagerDuty, or email
```

---

## Maintenance

### Regular Tasks

#### Weekly
```bash
# Check disk usage
docker system df

# Review logs for errors
docker-compose logs --since 7d | grep ERROR

# Verify backups
ls -lh backups/
```

#### Monthly
```bash
# Clean up unused images/volumes
docker system prune -a

# Update base images
docker pull postgres:15-alpine
docker pull redis:7-alpine
docker pull node:20-alpine

# Rebuild images with latest base
docker-compose build --pull
```

#### Quarterly
```bash
# Full backup of all data
docker-compose exec postgres pg_dump -U nsi_admin -d negative_space > backup_$(date +%Y%m%d).sql

# Test restore from backup
psql -U nsi_admin -d test_restore < backup_20250101.sql

# Review and update security policies
# - Rotate secrets (JWT_SECRET, passwords)
# - Update Docker base images
# - Review firewall rules
```

### Backup Strategy

```bash
# Automated daily PostgreSQL backup
docker exec nsi_postgres \
  pg_dump -U nsi_admin -d negative_space \
  > /backup/db_backup_$(date +%Y%m%d).sql

# Backup Redis data
docker exec nsi_redis redis-cli BGSAVE

# Copy Redis dump
docker cp nsi_redis:/data/dump.rdb /backup/redis_dump_$(date +%Y%m%d).rdb
```

---

## Support & Documentation

- **GitHub Repository:** https://github.com/sgbilod/Negative_Space_Imaging_Project
- **Docker Documentation:** https://docs.docker.com/
- **Docker Compose Reference:** https://docs.docker.com/compose/compose-file/
- **Project Issues:** GitHub Issues
- **Community:** Docker Community Forums

---

**Last Updated:** November 8, 2025
**Version:** 1.0
**Status:** Production Ready
