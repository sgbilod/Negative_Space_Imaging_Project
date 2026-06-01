# Docker & Container Orchestration Guide

## Negative Space Imaging Project - Phase 9 (Production Deployment)

**Last Updated:** October 17, 2025
**Status:** ✅ Production-Ready
**Version:** 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Dockerfiles](#dockerfiles)
5. [Docker Compose](#docker-compose)
6. [Deployment](#deployment)
7. [Security](#security)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Overview

This Docker setup provides production-grade containerization for the Negative Space Imaging Project with:

- ✅ Multi-stage builds for minimal image sizes
- ✅ Alpine base images for security and efficiency
- ✅ Non-root user execution for security
- ✅ Health checks for all services
- ✅ Complete observability with Prometheus + Grafana
- ✅ PostgreSQL database with automatic backups
- ✅ Redis caching layer
- ✅ Nginx reverse proxy with SSL support
- ✅ Python analyzer microservice
- ✅ Comprehensive logging with JSON format

---

## Architecture

### Service Topology

```
┌─────────────────────────────────────────────────────────┐
│                    Nginx Reverse Proxy                  │
│                  (Port 80, 443)                         │
└─────────────┬──────────────────────────────┬────────────┘
              │                              │
              ▼                              ▼
    ┌──────────────────┐        ┌──────────────────────┐
    │    Frontend      │        │   API Server         │
    │  (React/Nginx)   │        │   (Express.js)       │
    │  (Port 3001)     │        │   (Port 3000)        │
    └──────────────────┘        └──────────┬───────────┘
              │                             │
              └─────────────┬───────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
    ┌────────┐         ┌────────┐         ┌──────────┐
    │   DB   │         │ Cache  │         │ Analyzer │
    │(PgSQL) │         │(Redis) │         │(Python)  │
    │:5432   │         │:6379   │         │:5000     │
    └────────┘         └────────┘         └──────────┘

    ┌─────────────────────────────────────────┐
    │   Observability Stack                   │
    ├──────────────┬──────────────────────────┤
    │ Prometheus   │   Grafana                │
    │ (:9090)      │   (:3002)                │
    └──────────────┴──────────────────────────┘
```

### Services

| Service             | Image                  | Port   | Purpose            |
| ------------------- | ---------------------- | ------ | ------------------ |
| **postgres**        | postgres:16-alpine     | 5432   | Primary database   |
| **redis**           | redis:7-alpine         | 6379   | Cache layer        |
| **api**             | nsip-api:latest        | 3000   | Express API        |
| **frontend**        | nsip-frontend:latest   | 3001   | React UI           |
| **python-analyzer** | nsip-analyzer:latest   | 5000   | Analysis service   |
| **prometheus**      | prom/prometheus:latest | 9090   | Metrics collection |
| **grafana**         | grafana/grafana:latest | 3002   | Dashboards         |
| **nginx**           | nginx:1.25-alpine      | 80/443 | Reverse proxy      |

---

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 20GB disk space

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd negative-space-imaging

# Create environment file
cp .env.example .env

# Start services
./scripts/docker-deploy.sh up

# View status
./scripts/docker-deploy.sh status

# Check health
./scripts/docker-deploy.sh health
```

### Access Services

- Frontend: http://localhost:3001
- API: http://localhost:3000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3002 (admin/admin)

### Stop Services

```bash
./scripts/docker-deploy.sh down
```

---

## Dockerfiles

### Dockerfile.api (Express Server)

**Purpose:** Multi-stage build for Node.js Express API
**Base Image:** node:20-alpine
**Final Size:** ~250MB

**Build Stages:**

1. Builder: Compiles TypeScript, installs dependencies
2. Runtime: Minimal image with only production dependencies

**Key Features:**

- Multi-stage for size efficiency
- Non-root user (node:node)
- Health check endpoint
- Signal handling with dumb-init

### Dockerfile.frontend (React with Nginx)

**Purpose:** Production-grade React SPA with Nginx
**Base Images:** node:20-alpine (build), nginx:1.25-alpine (runtime)
**Final Size:** ~100MB

**Build Stages:**

1. Builder: Builds React app with Vite
2. Runtime: Serves with Nginx with optimized caching

**Features:**

- SPA routing support
- Static asset caching (1 year)
- API proxy to backend
- Gzip compression enabled

### Dockerfile.python (Python Analyzer)

**Purpose:** Python-based image analysis service
**Base Images:** python:3.11-slim (build), python:3.11-slim (runtime)
**Final Size:** ~400MB

**Build Stages:**

1. Builder: Creates virtual environment, installs packages
2. Runtime: Lean runtime with venv only

**Features:**

- Virtual environment for isolation
- Gunicorn WSGI server
- Non-root user execution
- 4 worker processes

---

## Docker Compose

### Development (docker-compose.yml)

Features:

- Hot reload with volume mounts
- All services in one network
- Health checks enabled
- JSON logging driver
- Database seeding on startup

### Production (docker-compose.prod.yml)

Features:

- AWS CloudWatch logging
- Restricted port binding (127.0.0.1)
- Enhanced security:
  - no-new-privileges flag
  - Dropped all capabilities
  - Added only NET_BIND_SERVICE
- AWS CloudWatch integration
- Database backup configuration
- 90-day log retention

---

## Deployment

### Building Images

```bash
# Build all images (development)
./scripts/docker-build.sh

# Build specific version
./scripts/docker-build.sh --version 1.0.0

# Multi-platform build (requires buildx)
./scripts/docker-build.sh --multi-platform

# Build and push to registry
./scripts/docker-build.sh --push --version 1.0.0
```

### Starting Services

```bash
# Development
./scripts/docker-deploy.sh up

# Production
./scripts/docker-deploy.sh up --env production

# Specific service only
./scripts/docker-deploy.sh up --service api
```

### Database Backup

```bash
# Create backup
./scripts/docker-deploy.sh backup

# Restore from backup
./scripts/docker-deploy.sh restore ./logs/backup-20250101-120000.sql
```

### Environment Variables

Create `.env` file:

```bash
# Environment
NODE_ENV=production
ENVIRONMENT=production

# Database
DB_USER=postgres
DB_PASSWORD=secure_password_here
DB_NAME=negative_space

# Redis
REDIS_PASSWORD=redis_password_here
REDIS_MAX_MEMORY=256mb

# Security
JWT_SECRET=your_jwt_secret_key_here
CORS_ORIGIN=https://yourdomain.com

# Domain
DOMAIN=yourdomain.com

# Monitoring
GRAFANA_USER=admin
GRAFANA_PASSWORD=grafana_password_here
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project

# AWS (for CloudWatch logging)
AWS_REGION=us-east-1
```

---

## Security

### Best Practices Implemented

1. **Non-root Users**
   - All services run as non-root users
   - node:node for Node services
   - python user for Python services

2. **Minimal Attack Surface**
   - Alpine base images (~5MB)
   - Multi-stage builds reduce final size
   - Only runtime dependencies included

3. **Resource Limits**
   - Memory limits per service
   - CPU limits for resource-heavy services
   - Disk quota enforcement

4. **Network Isolation**
   - Custom bridge network (172.20.0.0/16)
   - Service discovery via DNS
   - Ports bound to 127.0.0.1 in production

5. **Security Options**
   - `no-new-privileges: true`
   - Dropped all capabilities
   - Added only required capabilities

6. **SSL/TLS**
   - Nginx terminates SSL
   - Certificates via Let's Encrypt
   - HTTP/2 support

### Secret Management

**Development:**

- Use `.env` file (add to `.gitignore`)
- Docker secrets for sensitive data

**Production:**

- AWS Secrets Manager
- Docker Swarm secrets
- Kubernetes secrets (for K8s deployment)
- HashiCorp Vault integration possible

---

## Monitoring

### Prometheus

Collects metrics from:

- Node exporter (system metrics)
- API application metrics
- Nginx metrics
- PostgreSQL metrics

Configuration: `monitoring/prometheus.yml`

### Grafana

Pre-configured dashboards:

- System overview
- API performance
- Database statistics
- Error rates

Access: http://localhost:3002
Default credentials: admin/admin

### Health Checks

All services include health checks:

```yaml
healthcheck:
  test: ['CMD', 'curl', '-f', 'http://localhost:PORT/health']
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s
```

View status:

```bash
./scripts/docker-deploy.sh health
```

### Logging

JSON-format logs for easy parsing:

```bash
# View logs
./scripts/docker-deploy.sh logs

# Follow specific service
./scripts/docker-deploy.sh logs --service api --follow

# Show last 100 lines with timestamps
./scripts/docker-deploy.sh logs --lines 100 --timestamps
```

---

## Troubleshooting

### Service won't start

```bash
# Check logs
docker logs nsip-api

# Check dependencies
docker-compose ps

# Restart service
./scripts/docker-deploy.sh restart --service api
```

### Database connection issues

```bash
# Check postgres health
docker-compose exec postgres pg_isready

# Check redis
docker-compose exec redis redis-cli ping

# View database logs
docker-compose logs postgres
```

### High memory usage

```bash
# Check container stats
docker stats

# Reduce max memory
# Edit .env: REDIS_MAX_MEMORY=128mb
./scripts/docker-deploy.sh restart --service redis
```

### SSL certificate issues

```bash
# Check certificate
docker exec nsip-nginx openssl s_client -connect localhost:443

# Renew certificate
# Using Let's Encrypt through Nginx
# See monitoring/nginx.prod.conf
```

---

## Advanced Usage

### Scaling Services

```bash
# Scale to 3 instances (requires load balancer)
docker-compose up -d --scale api=3
```

### Custom Networks

```bash
# Create network
docker network create nsip-network

# Inspect network
docker network inspect nsip-network
```

### Volume Management

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect negative-space-imaging_postgres-data

# Backup volume
docker run --rm -v nsip-postgres-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/postgres-backup.tar.gz /data
```

### Build Optimization

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Build with progress
docker build --progress=plain -f Dockerfile.api .

# Cache busting
docker build --no-cache -f Dockerfile.api .
```

### Registry Integration

```bash
# Login to registry
docker login docker.io

# Tag image
docker tag nsip-api:latest docker.io/username/nsip-api:1.0.0

# Push to registry
docker push docker.io/username/nsip-api:1.0.0

# Pull image
docker pull docker.io/username/nsip-api:1.0.0
```

### Kubernetes Deployment

Images are Kubernetes-ready:

```yaml
# Example Pod definition
apiVersion: v1
kind: Pod
metadata:
  name: nsip-api
spec:
  containers:
    - name: api
      image: docker.io/username/nsip-api:latest
      ports:
        - containerPort: 3000
      env:
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: nsip-config
              key: db_host
      livenessProbe:
        httpGet:
          path: /health
          port: 3000
        initialDelaySeconds: 10
        periodSeconds: 30
      readinessProbe:
        httpGet:
          path: /health
          port: 3000
        initialDelaySeconds: 5
        periodSeconds: 10
```

---

## Performance Optimization

### Image Size Reduction

| Service  | Optimization        | Size   |
| -------- | ------------------- | ------ |
| API      | Multi-stage, Alpine | ~250MB |
| Frontend | Nginx + gzip        | ~100MB |
| Analyzer | Python venv         | ~400MB |

### Build Speed

- Leverage layer caching
- Order Dockerfile commands by changeability
- Use `.dockerignore` to exclude files
- BuildKit with parallel stages

### Runtime Performance

- Redis for caching
- Connection pooling (database)
- Nginx compression
- Browser caching (1 year for static assets)

---

## Maintenance

### Regular Tasks

```bash
# Weekly
./scripts/docker-deploy.sh backup

# Monthly
./scripts/docker-deploy.sh prune

# When updating dependencies
./scripts/docker-build.sh --version 1.1.0 --push
```

### Upgrade Procedure

```bash
# 1. Pull new images
docker pull nsip-api:latest

# 2. Update docker-compose
git pull origin main

# 3. Rebuild services
./scripts/docker-build.sh

# 4. Recreate containers
./scripts/docker-deploy.sh restart

# 5. Verify health
./scripts/docker-deploy.sh health
```

---

## Support

For issues or questions:

1. Check troubleshooting section
2. Review logs: `./scripts/docker-deploy.sh logs`
3. Open GitHub issue with:
   - Docker version
   - Docker Compose version
   - Relevant logs
   - Steps to reproduce

---

**Last Updated:** October 17, 2025
**Maintainer:** Negative Space Systems Team
**License:** MIT
