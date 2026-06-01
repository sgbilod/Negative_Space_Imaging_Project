# Docker Health Check & Initialization Scripts

This directory contains comprehensive Docker health checking and initialization scripts for the Negative Space Imaging Project.

## Overview

The scripts provide production-ready health monitoring and service initialization with:

- ✅ Comprehensive error handling
- ✅ Retry logic with exponential backoff
- ✅ Color-coded status reporting
- ✅ Timestamped logging to files
- ✅ Service-specific recovery commands
- ✅ JSON report generation
- ✅ Multi-platform support (Linux, macOS, Windows/WSL2)

## Files

### 1. `docker-health-check.sh`

Bash script for comprehensive Docker container and service health checking.

**Features:**

- Docker daemon status verification
- Container health status checks (app, redis, postgres, monitoring, grafana)
- Service connectivity tests (port accessibility, service responsiveness)
- Resource usage monitoring
- Error log analysis
- Automatic repair capability
- Timestamped logging with JSON metrics

**Requirements:**

- Bash 4.0+
- Docker CLI
- Docker Compose
- net-tools (for networking checks)

**Usage:**

```bash
# Basic health check
./docker-health-check.sh

# Verbose mode with logging
./docker-health-check.sh --verbose

# Attempt automatic repair
./docker-health-check.sh --repair

# Custom log file location
./docker-health-check.sh --log-file /var/log/docker-health.log

# All options combined
./docker-health-check.sh --verbose --repair --log-file custom.log
```

**Exit Codes:**

- `0` - All services healthy
- `1` - One or more services degraded/failed
- `2` - Docker daemon not running
- `3` - Script execution error

**Output Files:**

- `logs/docker-health-{YYYYMMDD_HHMMSS}.log` - Detailed log file
- `logs/docker-metrics-{YYYYMMDD_HHMMSS}.json` - JSON metrics report

### 2. `docker-init.js`

Node.js script for Docker service initialization and health verification.

**Features:**

- Environment variable validation
- PostgreSQL connectivity testing
- Redis connection verification
- Express server health endpoint checking
- Connection retry logic with exponential backoff
- Formatted JSON health report
- Service-specific error handling
- Test mode for validation without modifications

**Requirements:**

- Node.js 14+
- `pg` package (optional, for full PostgreSQL testing)
- `redis` package (optional, for full Redis testing)

**Installation:**

```bash
npm install pg redis
```

**Usage:**

```javascript
// Basic initialization
node docker-init.js

// Verbose mode with logging
node docker-init.js --verbose --log-file docker-init.log

// Test mode (no modifications)
node docker-init.js --test-mode

// Custom timeout and retry
node docker-init.js --timeout 60 --retry-attempts 5

// All options
node docker-init.js \
  --verbose \
  --log-file logs/docker-init.log \
  --timeout 45 \
  --retry-attempts 3
```

**Exit Codes:**

- `0` - All services initialized successfully
- `1` - One or more services failed initialization
- `2` - Environment validation failed
- `3` - Script execution error

**Output Files:**

- `logs/docker-init-{TIMESTAMP}.json` - JSON health report
- Custom log file if specified with `--log-file`

### 3. `docker-health-check.ps1`

PowerShell wrapper for Windows/WSL2 environments.

**Features:**

- Automatic Docker Desktop detection
- WSL2 integration handling
- Cross-platform script execution
- Log file management
- Colored console output

**Requirements:**

- PowerShell 5.1+
- Docker Desktop with WSL2 backend (Windows)
- Bash via WSL2 or Git Bash
- Node.js installed

**Usage:**

```powershell
# Health check with verbose output
.\docker-health-check.ps1 -ScriptType health -Verbose

# Initialization with logging
.\docker-health-check.ps1 -ScriptType init -LogFile "logs\docker-init.log"

# Health check with automatic repair
.\docker-health-check.ps1 -ScriptType health -Repair

# Custom timeout and retries
.\docker-health-check.ps1 -ScriptType init `
  -Timeout 60 `
  -RetryAttempts 5
```

## Service Configurations

### PostgreSQL

- **Port:** 5432
- **Host:** localhost
- **Database:** negative_space
- **User:** postgres
- **Test Query:** SELECT 1 and SELECT NOW()

### Redis

- **Port:** 6379
- **Host:** localhost
- **Test Command:** PING

### Express Application

- **Port:** 8000 (Docker) / 3000 (Direct)
- **Host:** localhost
- **Health Endpoint:** /health
- **Test:** HTTP GET with 200 response

### Prometheus

- **Port:** 9090
- **Host:** localhost

### Grafana

- **Port:** 3000
- **Host:** localhost

## Environment Variables

Required for full health checks:

```
DATABASE_URL=postgres://postgres:postgres@localhost:5432/negative_space
REDIS_URL=redis://localhost:6379
NODE_ENV=development
JWT_SECRET=your-secret-key
```

Optional:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=negative_space
DB_USER=postgres
DB_PASSWORD=postgres
REDIS_HOST=localhost
REDIS_PORT=6379
EXPRESS_HOST=localhost
EXPRESS_PORT=3000
LOG_LEVEL=info
```

## Typical Workflows

### Daily Health Check

```bash
# Check all services
./docker-health-check.sh --verbose

# View results
tail -f logs/docker-health-*.log
```

### Initial Deployment

```bash
# Start containers
docker-compose up -d

# Wait for services to initialize
sleep 10

# Run initialization
node docker-init.js --verbose

# Verify health
./docker-health-check.sh
```

### Troubleshooting Failed Services

```bash
# Get detailed status with repair attempt
./docker-health-check.sh --verbose --repair

# Check specific service logs
docker logs negative-space-imaging-project_postgres_1

# Get metrics report
tail -f logs/docker-metrics-*.json
```

### Windows Development Setup

```powershell
# First time setup
docker-compose up -d

# Initialize services
.\docker-health-check.ps1 -ScriptType init -Verbose

# Daily health check
.\docker-health-check.ps1 -ScriptType health

# With issues - attempt repair
.\docker-health-check.ps1 -ScriptType health -Repair
```

## Error Handling

### Docker Daemon Not Running

**Error:** `Docker daemon is not running`
**Solutions:**

- Linux: `sudo systemctl start docker`
- Docker Desktop: Open Docker Desktop application
- WSL2: Ensure WSL2 backend is enabled in Docker Desktop

### PostgreSQL Connection Failed

**Error:** `PostgreSQL port 5432 is not accessible`
**Solutions:**

- Check if container is running: `docker ps | grep postgres`
- Restart service: `docker-compose restart postgres`
- Check logs: `docker logs negative-space-imaging-project_postgres_1`

### Redis Connection Failed

**Error:** `Redis port 6379 is not accessible`
**Solutions:**

- Check if container is running: `docker ps | grep redis`
- Restart service: `docker-compose restart redis`
- Test connection: `docker exec -it negative-space-imaging-project_redis_1 redis-cli ping`

### Express Server Not Responding

**Error:** `Application port 8000 is not accessible`
**Solutions:**

- Check if app is running: `docker ps | grep app`
- Restart application: `docker-compose restart app`
- Check application logs: `docker logs negative-space-imaging-project_app_1`

## Performance Considerations

### Timeout Settings

- Default: 30 seconds
- For slow networks: Increase with `--timeout 60`
- For fast networks: Can reduce to `--timeout 15`

### Retry Attempts

- Default: 3 attempts
- For unstable networks: Increase to `--retry-attempts 5`
- For stable networks: Can reduce to `--retry-attempts 2`

### Connection Pooling

- PostgreSQL pool size: 20 (configurable via `DATABASE_POOL_SIZE`)
- Connection timeout: 5 seconds
- Idle timeout: 30 seconds

## Security Considerations

- ✅ No hardcoded credentials (uses environment variables)
- ✅ Credentials never logged to files
- ✅ Local port checks only (no external network exposure)
- ✅ Connection timeouts to prevent hanging
- ✅ Error messages sanitized to avoid exposing internal details

## Logging and Monitoring

### Log Levels

- **ERROR:** Service failures, connection errors
- **WARN:** Degraded service, retries
- **INFO:** General progress, service status
- **OK:** Successful checks and operations
- **DEBUG:** Detailed diagnostic information (verbose mode only)

### Log Retention

- Logs stored in `logs/` directory
- Timestamped for easy archive
- Consider rotating logs for long-running systems:
  ```bash
  # Archive old logs
  find logs/ -name "docker-*.log" -mtime +30 -exec gzip {} \;
  ```

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Docker Health Check
  run: |
    chmod +x scripts/docker-health-check.sh
    scripts/docker-health-check.sh --verbose

- name: Docker Initialization
  run: |
    node scripts/docker-init.js --log-file ci-init.log
```

### GitLab CI Example

```yaml
health_check:
  script:
    - chmod +x scripts/docker-health-check.sh
    - scripts/docker-health-check.sh --verbose
  artifacts:
    paths:
      - logs/
```

## Contributing

When modifying these scripts:

1. Maintain backwards compatibility
2. Add tests for new functionality
3. Update this README with new options
4. Follow existing code style and conventions
5. Add comprehensive error handling
6. Include detailed comments for complex logic

## Support

For issues or questions:

1. Check the log files for detailed error messages
2. Review the "Error Handling" section above
3. Enable verbose mode: `--verbose` flag
4. Check Docker Desktop logs for daemon issues
5. Verify environment variables are set correctly

## Version History

### v1.0.0 (2025-10-17)

- Initial release
- Docker daemon and container checks
- Service connectivity verification
- PostgreSQL and Redis health checks
- Express server health verification
- Automatic repair capability
- Multi-platform support
- Comprehensive error handling
- JSON metrics and reporting

## License

These scripts are part of the Negative Space Imaging Project and are subject to the project's LICENSE.
