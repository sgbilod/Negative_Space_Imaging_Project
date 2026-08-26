# Docker Quick Start - 5 Minute Guide

## One Command Start

```bash
# Development (all defaults)
./scripts/docker-init.sh && \
docker-compose build && \
./scripts/docker-up.sh && \
./scripts/docker-test.sh
```

## Service URLs (Development)

| Service | URL | User |
|---------|-----|------|
| **Frontend** | `localhost` | - |
| **API** | `localhost:3000` | - |
| **Python Service** | `localhost:8000` | - |
| **Prometheus** | `localhost:9090` | - |
| **Grafana** | `localhost:3001` | admin / admin |
| **Database** | `localhost:5432` | nsi_admin / password |
| **Cache** | `localhost:6379` | - |

## Essential Commands

### Lifecycle
```bash
./scripts/docker-init.sh           # Initialize environment
docker-compose build               # Build images
./scripts/docker-up.sh             # Start services
./scripts/docker-test.sh           # Verify health
./scripts/docker-down.sh           # Stop services
./scripts/docker-down.sh --volumes # Stop + cleanup
```

### Development
```bash
docker-compose logs -f api         # Watch logs
docker-compose restart api         # Restart service
docker exec -it nsi_api /bin/sh    # Shell access
docker-compose ps                  # Service status
```

### Troubleshooting
```bash
./scripts/docker-test.sh --verbose # Full diagnostics
docker stats                       # Resource usage
docker-compose logs                # All service logs
docker system prune -a             # Deep cleanup
```

### Production
```bash
./scripts/docker-up.sh prod        # Start production
docker-compose -f docker-compose.yml \
  -f docker-compose.prod.yml ps    # Check prod status
```

## Health Checks

```bash
# All services at once
./scripts/docker-test.sh

# Individual services
curl http://localhost/health              # Frontend
curl http://localhost:3000/health         # API
curl http://localhost:8000/health         # Python
curl http://localhost:9090/-/healthy      # Prometheus
```

## Environment Setup

```bash
# Development (default)
cp .env.example .env

# Production (requires secret values)
cp .env.example .env.prod
nano .env.prod    # Edit with production values
```

## Ports Reference

| Port | Service |
|------|---------|
| **80** | Frontend (Nginx) |
| **443** | Frontend (HTTPS) |
| **3000** | Express API |
| **3001** | Grafana |
| **5432** | PostgreSQL |
| **6379** | Redis |
| **8000** | Python Service |
| **9090** | Prometheus |

## Database Access

```bash
# PostgreSQL
docker exec -it nsi_postgres psql -U nsi_admin -d negative_space

# Redis
docker exec -it nsi_redis redis-cli

# Commands
SELECT 1;         # PostgreSQL: test connection
INFO              # Redis: server info
```

## Logs & Debugging

```bash
# Follow all logs
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api

# With timestamps
docker-compose logs --timestamps api

# Since specific time
docker-compose logs --since 30m api
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Port in use | `lsof -i :3000` then `kill -9 <PID>` |
| DB won't connect | `docker-compose restart postgres redis` |
| Services fail to start | `docker-compose logs` check errors |
| Out of memory | `docker stats` check usage, increase limits |
| Slow performance | Scale services: modify replicas in compose file |

## Cleanup Levels

```bash
# Soft: Stop services, keep data
docker-compose down

# Medium: Stop services, remove data
docker-compose down -v

# Hard: Remove everything (images too)
docker system prune -a -f

# Full reset (nuclear option)
docker system prune -a -f
rm -rf ./uploads ./shared_data
docker-compose up -d --build
```

## Performance Monitoring

```bash
# Real-time resource usage
docker stats

# Container-specific stats
docker stats nsi_api

# Export stats to file
docker stats --no-stream > resources.txt
```

## File Permissions

```bash
# Make scripts executable
chmod +x scripts/docker-*.sh

# Set proper directory permissions
chmod 755 uploads shared_data logs monitoring
chmod 666 uploads shared_data/*.* 2>/dev/null || true
```

## Network Debugging

```bash
# Test service-to-service communication
docker exec nsi_api curl http://python_service:8000/health

# Inspect network
docker network inspect nsi_network

# DNS resolution
docker exec nsi_api nslookup postgres
```

## Data Backup

```bash
# Database dump
docker exec nsi_postgres pg_dump -U nsi_admin -d negative_space > backup.sql

# Restore
docker exec -i nsi_postgres psql -U nsi_admin -d negative_space < backup.sql

# Redis dump
docker exec nsi_redis redis-cli BGSAVE
docker cp nsi_redis:/data/dump.rdb ./redis_backup.rdb
```

## System Requirements

- **Docker:** 20.10+
- **Docker Compose:** 1.29+
- **RAM:** 4GB minimum (8GB recommended)
- **Disk:** 20GB for images/data
- **Ports:** 80, 443, 3000, 3001, 5432, 6379, 8000, 9090

---

**Quick Reference Version:** 1.0
**Last Updated:** November 8, 2025
