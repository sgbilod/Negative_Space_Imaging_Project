# Docker Scripts Implementation Guide

## Complete Overview

You now have a comprehensive Docker health monitoring and initialization system for the Negative Space Imaging Project. This document provides detailed guidance on implementation and usage.

## 📦 Deliverables

### 1. **docker-health-check.sh** (650+ lines)

Production-ready bash script with:

- Docker daemon verification
- Multi-container health checks (app, redis, postgres, monitoring, grafana)
- Service connectivity verification
- Automatic repair capability
- Timestamped logging with JSON metrics
- Color-coded console output
- Comprehensive error handling

### 2. **docker-init.js** (550+ lines)

Production-ready Node.js script with:

- Environment variable validation
- PostgreSQL connection testing with retry logic
- Redis connectivity verification
- Express server health endpoint checking
- Connection pooling support
- Exponential backoff retry mechanism
- JSON report generation
- Test mode support

### 3. **docker-health-check.ps1** (250+ lines)

PowerShell wrapper for Windows users featuring:

- Docker Desktop detection
- WSL2 integration
- Cross-platform script execution
- Colored console output
- Log file management

### 4. **docker-quick-start.sh** (300+ lines)

Interactive menu system with:

- Quick access to all common operations
- Log viewing and management
- Service status inspection
- Docker system diagnostics
- One-command full setup

## 🚀 Quick Start

### For Linux/macOS Users

```bash
cd scripts

# Make scripts executable
chmod +x docker-health-check.sh docker-quick-start.sh

# Run basic health check
./docker-health-check.sh

# Or use interactive menu
./docker-quick-start.sh
```

### For Windows Users

```powershell
cd scripts

# Run health check
.\docker-health-check.ps1 -ScriptType health -Verbose

# Run initialization
.\docker-health-check.ps1 -ScriptType init -LogFile "logs\docker-init.log"
```

### For Node.js Users

```bash
# Install optional dependencies (for full testing)
npm install pg redis

# Run initialization
node docker-init.js --verbose --log-file logs/docker-init.log
```

## 🔍 Key Features

### Error Handling

- **Automatic Retries**: Exponential backoff with configurable attempts
- **Timeout Protection**: Prevents hanging connections
- **Fallback Mechanisms**: Graceful degradation when services unavailable
- **Detailed Logging**: All errors logged with stack traces (verbose mode)

### Health Checks Performed

#### Docker Infrastructure

- ✓ Docker daemon is running
- ✓ Docker Compose availability
- ✓ Container existence and status
- ✓ Health endpoint responses

#### Database Services

- ✓ PostgreSQL port accessibility
- ✓ PostgreSQL connection establishment
- ✓ Test query execution (SELECT 1, SELECT NOW())
- ✓ Connection pooling validation

#### Cache Services

- ✓ Redis port accessibility
- ✓ Redis connection establishment
- ✓ PING/PONG verification
- ✓ Memory usage monitoring

#### Application Services

- ✓ Express server port accessibility
- ✓ /health endpoint response
- ✓ HTTP status code verification
- ✓ Response time measurement

#### Infrastructure Services

- ✓ Prometheus endpoint connectivity
- ✓ Grafana dashboard availability
- ✓ Monitoring stack status

### Logging System

Each script generates timestamped logs:

```
logs/
├── docker-health-20251017_154325.log    # Bash health check
├── docker-health-20251017_154326.log    # Bash health check (automated repair)
├── docker-metrics-20251017_154325.json  # JSON metrics from bash
├── docker-init-1729172605000.json       # JSON report from Node.js
└── docker-init.log                      # Custom log file (if specified)
```

## 📊 Configuration Reference

### Environment Variables Required

```bash
# Database Configuration
DATABASE_URL=postgres://postgres:postgres@localhost:5432/negative_space
DB_HOST=localhost
DB_PORT=5432
DB_NAME=negative_space
DB_USER=postgres
DB_PASSWORD=postgres

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_HOST=localhost
REDIS_PORT=6379

# Application Configuration
NODE_ENV=development
EXPRESS_HOST=localhost
EXPRESS_PORT=3000
PORT=3000

# Security
JWT_SECRET=your-secure-secret
JWT_EXPIRES_IN=1d
BCRYPT_SALT_ROUNDS=12
COOKIE_SECRET=your-secure-cookie-secret
CSRF_ENABLED=true

# Monitoring
LOG_LEVEL=info
LOG_DIR=logs
LOG_FILENAME=app.log
```

### Service Port Mapping

| Service    | Port      | Container Name                            | Docker Network |
| ---------- | --------- | ----------------------------------------- | -------------- |
| PostgreSQL | 5432      | negative-space-imaging-project_postgres_1 | app-network    |
| Redis      | 6379      | negative-space-imaging-project_redis_1    | app-network    |
| App        | 8000/3000 | negative-space-imaging-project_app_1      | app-network    |
| Prometheus | 9090      | monitoring                                | app-network    |
| Grafana    | 3000      | grafana                                   | app-network    |

## 🛠️ Common Operations

### Daily Health Check

```bash
# Quick status check
./docker-health-check.sh

# Detailed check with logging
./docker-health-check.sh --verbose --log-file daily-check.log

# Check and repair if needed
./docker-health-check.sh --repair
```

### New Deployment Setup

```bash
# 1. Start containers
docker-compose up -d

# 2. Wait for initialization
sleep 15

# 3. Run Node.js initialization
node docker-init.js --verbose

# 4. Verify health
./docker-health-check.sh --verbose
```

### Troubleshooting

```bash
# Get detailed diagnostic info
./docker-health-check.sh --verbose

# View service logs
docker logs <container-name> --tail 50 --follow

# Test specific port
nc -zv localhost 5432

# Check resource usage
docker stats

# View JSON metrics report
cat logs/docker-metrics-*.json | jq .
```

### Log Management

```bash
# View latest health check log
tail -f logs/docker-health-*.log

# Search for errors
grep ERROR logs/docker-health-*.log

# Archive old logs
find logs/ -name "docker-*.log" -mtime +7 -exec gzip {} \;

# Clean up old logs (interactive)
./docker-quick-start.sh  # Select option 9
```

## 🔐 Security Features

### Implemented

- ✅ No hardcoded credentials
- ✅ Environment variable based configuration
- ✅ Sensitive data never logged
- ✅ Connection timeouts (prevent DOS)
- ✅ Error sanitization (no internal details exposed)
- ✅ Local-only port checks
- ✅ No privileged operations required

### Best Practices

- Store sensitive configs in `.env` file (never commit)
- Use different credentials for prod/dev/test
- Rotate Redis and PostgreSQL passwords regularly
- Monitor logs for authentication failures
- Use VPN/firewalls for production access

## 📈 Performance Tuning

### Connection Pooling

```javascript
// From docker-init.js
DATABASE_POOL_SIZE = 20; // Concurrent connections
DATABASE_MAX_OVERFLOW = 30; // Max overflow connections
connectionTimeoutMillis = 5000; // Connection timeout
idleTimeoutMillis = 30000; // Idle timeout
```

### Timeout Configuration

```bash
# Health check with custom timeout (45 seconds)
./docker-health-check.sh --timeout 45

# Node.js init with custom timeout (60 seconds)
node docker-init.js --timeout 60 --retry-attempts 5
```

### Batch Operations

```bash
# Run multiple checks in sequence
for i in {1..5}; do
  echo "Health check #$i"
  ./docker-health-check.sh --quiet
  sleep 10
done
```

## 🐛 Troubleshooting Guide

### "Docker daemon is not running"

```bash
# Linux
sudo systemctl start docker

# macOS
open /Applications/Docker.app

# Windows
# Open Docker Desktop from Start Menu
```

### "PostgreSQL port is not accessible"

```bash
# Check if container is running
docker ps | grep postgres

# Restart service
docker-compose restart postgres

# Check logs
docker logs -f negative-space-imaging-project_postgres_1
```

### "Redis connection failed"

```bash
# Verify Redis is running
docker exec -it negative-space-imaging-project_redis_1 redis-cli ping

# Check Redis logs
docker logs negative-space-imaging-project_redis_1

# Reset Redis cache
docker-compose down redis
docker-compose up -d redis
```

### "Express server not responding"

```bash
# Check app container
docker ps | grep app

# View app logs
docker logs -f negative-space-imaging-project_app_1

# Check if app is crashing
docker logs negative-space-imaging-project_app_1 | tail -100

# Restart app
docker-compose restart app
```

## 📋 Integration Examples

### GitHub Actions CI/CD

```yaml
name: Docker Health Check
on: [push, pull_request]

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Start Docker services
        run: docker-compose up -d

      - name: Wait for services
        run: sleep 10

      - name: Run health check
        run: |
          chmod +x scripts/docker-health-check.sh
          scripts/docker-health-check.sh --verbose

      - name: Upload logs
        if: failure()
        uses: actions/upload-artifact@v2
        with:
          name: docker-logs
          path: logs/
```

### GitLab CI

```yaml
docker-health-check:
  script:
    - chmod +x scripts/docker-health-check.sh
    - docker-compose up -d
    - sleep 10
    - scripts/docker-health-check.sh --verbose
  artifacts:
    paths:
      - logs/
    expire_in: 1 week
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    stages {
        stage('Health Check') {
            steps {
                sh '''
                    chmod +x scripts/docker-health-check.sh
                    docker-compose up -d
                    sleep 10
                    scripts/docker-health-check.sh --verbose
                '''
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'logs/**/*.log', allowEmptyArchive: true
        }
    }
}
```

## 📚 API Reference

### Bash Script Functions (docker-health-check.sh)

```bash
check_docker_daemon()           # Verify Docker is running
check_container_health()        # Check individual container
check_port_accessibility()      # Test port connectivity
check_postgres_health()         # PostgreSQL specific tests
check_redis_health()            # Redis specific tests
check_app_health()              # Application tests
repair_services()               # Attempt auto-repair
```

### Node.js Script Functions (docker-init.js)

```javascript
validateEnvironment(); // Check required env vars
checkPostgresHealth(); // PostgreSQL connection test
checkRedisHealth(); // Redis connection test
checkExpressHealth(); // Express server test
checkPort(); // Generic port check
retryWithBackoff(); // Retry with exponential backoff
```

## 🎯 Best Practices

1. **Run daily**: Schedule health checks at predictable times
2. **Monitor logs**: Set up log aggregation/monitoring
3. **Alert on failures**: Configure alerting when health checks fail
4. **Automate repair**: Use `--repair` flag for automatic recovery
5. **Archive logs**: Keep logs for audit/compliance
6. **Test recovery**: Regularly test failure and recovery scenarios
7. **Update scripts**: Keep scripts synchronized with infrastructure changes
8. **Document changes**: Track modifications to scripts

## 📞 Support & Maintenance

### Updating Scripts

When Docker setup changes:

1. Update `SERVICE_CONFIG` in docker-init.js
2. Update port mappings in docker-health-check.sh
3. Update environment variables documentation
4. Test all scenarios before deployment
5. Document changes in version history

### Extending Functionality

To add new service checks:

1. Add service to `SERVICES` array
2. Create `check_<service>_health()` function
3. Add test in main execution phase
4. Update status reporting
5. Add to README documentation

## 📝 Version History

### v1.0.0 - October 17, 2025

- Initial release
- Docker daemon and container checks
- Service connectivity verification
- PostgreSQL and Redis health checks
- Express server health verification
- Automatic repair capability
- Multi-platform support (Linux, macOS, Windows)
- Comprehensive error handling
- JSON metrics and reporting

---

**For detailed usage information, see DOCKER_HEALTH_CHECK_README.md**
