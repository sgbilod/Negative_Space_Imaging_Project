# Docker Scripts - Quick Reference Card

## 📋 Essential Commands

### Linux/macOS Users

```bash
# Make executable
chmod +x scripts/*.sh

# Basic health check
./scripts/docker-health-check.sh

# Verbose with auto-repair
./scripts/docker-health-check.sh --verbose --repair

# Interactive menu
./scripts/docker-quick-start.sh

# Verify installation
./scripts/verify-installation.sh --fix

# Node.js initialization
node scripts/docker-init.js --verbose
```

### Windows/PowerShell Users

```powershell
# Health check
.\scripts\docker-health-check.ps1 -ScriptType health -Verbose

# Initialization
.\scripts\docker-health-check.ps1 -ScriptType init

# With logging
.\scripts\docker-health-check.ps1 -ScriptType health -LogFile "logs\check.log"

# Custom timeout
.\scripts\docker-health-check.ps1 -ScriptType init -Timeout 60
```

## 🔧 Common Scenarios

### First Time Setup

```bash
# 1. Start containers
docker-compose up -d

# 2. Wait for initialization
sleep 15

# 3. Run health check
./scripts/docker-health-check.sh --verbose

# 4. Run initialization
node scripts/docker-init.js --verbose
```

### Daily Operations

```bash
# Quick health check
./scripts/docker-health-check.sh

# View latest logs
tail -f logs/docker-health-*.log

# View metrics report
cat logs/docker-metrics-*.json | jq .

# Full diagnostics
./scripts/docker-quick-start.sh  # Use menu option 8
```

### Troubleshooting

```bash
# Detailed diagnostics
./scripts/docker-health-check.sh --verbose

# Attempt automatic repair
./scripts/docker-health-check.sh --verbose --repair

# Check specific service
docker logs negative-space-imaging-project_postgres_1

# Verify environment
node scripts/docker-init.js --verbose
```

## 📊 Exit Codes

| Code | Meaning     | Action                     |
| ---- | ----------- | -------------------------- |
| 0    | Success ✓   | Continue normal operations |
| 1    | Failed ✗    | Investigate errors         |
| 2    | Docker down | Start Docker daemon        |
| 3    | Error       | Check script logs          |

## 🗂️ File Guide

| File                    | Purpose           | Platform    |
| ----------------------- | ----------------- | ----------- |
| docker-health-check.sh  | Main health check | Linux/macOS |
| docker-init.js          | Initialization    | All         |
| docker-health-check.ps1 | Windows wrapper   | Windows     |
| docker-quick-start.sh   | Interactive menu  | Linux/macOS |
| verify-installation.sh  | Verification      | Linux/macOS |

## 🔍 What Gets Checked

### Services Verified

- ✓ PostgreSQL (port 5432)
- ✓ Redis (port 6379)
- ✓ Express App (port 8000)
- ✓ Prometheus (port 9090)
- ✓ Grafana (port 3000)

### Tests Performed

- ✓ Port accessibility
- ✓ Container status
- ✓ Database connectivity
- ✓ Query execution
- ✓ Health endpoints
- ✓ Environment variables

## 🔐 Environment Variables Required

```bash
DATABASE_URL=postgres://postgres:postgres@localhost:5432/negative_space
REDIS_URL=redis://localhost:6379
NODE_ENV=development
JWT_SECRET=your-secret-key
```

## 📝 Flag Reference

### docker-health-check.sh

- `--verbose` - Detailed output with debug info
- `--repair` - Attempt automatic service repair
- `--log-file FILE` - Custom log location
- `--quiet` - Suppress console output
- `--help` - Show help message

### docker-init.js

- `--verbose` - Detailed output
- `--log-file FILE` - Custom log location
- `--test-mode` - Test without modifications
- `--timeout SECONDS` - Connection timeout
- `--retry-attempts NUM` - Number of retries
- `--help` - Show help message

### docker-health-check.ps1

- `-ScriptType health|init` - Which script to run
- `-Verbose` - Detailed output
- `-LogFile FILE` - Log file path
- `-Repair` - Attempt repair
- `-Timeout SECONDS` - Timeout value
- `-RetryAttempts NUM` - Retry attempts

## 🐛 Quick Troubleshooting

### Docker not running

```bash
# Linux
sudo systemctl start docker

# macOS
open /Applications/Docker.app

# Windows
# Open Docker Desktop from Start Menu
```

### PostgreSQL not accessible

```bash
# Check container
docker ps | grep postgres

# Check logs
docker logs negative-space-imaging-project_postgres_1

# Restart service
docker-compose restart postgres
```

### Redis not accessible

```bash
# Check connection
docker exec -it negative-space-imaging-project_redis_1 redis-cli ping

# Restart service
docker-compose restart redis
```

### Express not responding

```bash
# Check logs
docker logs negative-space-imaging-project_app_1

# Restart app
docker-compose restart app

# Check port
netstat -tlnp | grep 8000
```

## 📚 Documentation Map

- **Quick Start** → INDEX.md
- **Complete Guide** → DOCKER_HEALTH_CHECK_README.md
- **Advanced Topics** → IMPLEMENTATION_GUIDE.md
- **Project Summary** → DELIVERABLES_SUMMARY.txt
- **Project Status** → COMPLETION_REPORT.txt

## 🎯 Menu Options (docker-quick-start.sh)

```
1 - Basic health check
2 - Verbose health check with debug info
3 - Health check with automatic repair
4 - Run service initialization
5 - Full setup (start + init + check)
6 - View recent log files
7 - Show current service status
8 - Docker system information
9 - Cleanup old logs
0 - Exit menu
```

## 💡 Pro Tips

1. **Enable verbose mode** for detailed troubleshooting
2. **Use `--repair` flag** to auto-fix common issues
3. **Check logs first** when problems occur
4. **Schedule daily checks** for production monitoring
5. **Keep logs archived** for historical analysis
6. **Set up alerting** on health check failures
7. **Test recovery** procedures regularly
8. **Update scripts** when infrastructure changes

## 🚨 Error Recovery Commands

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart postgres

# Rebuild containers
docker-compose down
docker-compose up -d

# View detailed logs
docker logs -f service-name

# Cleanup and reset
docker-compose down -v
docker-compose up -d
```

## 📊 Log Files Location

```
logs/
├── docker-health-{TIMESTAMP}.log      # Health check logs
├── docker-metrics-{TIMESTAMP}.json    # Health metrics
└── docker-init-{TIMESTAMP}.json       # Init reports
```

## 🔄 Typical Deployment Flow

```
1. Start containers
   docker-compose up -d

2. Wait for services (15s)
   sleep 15

3. Run initialization
   node docker-init.js --verbose

4. Verify health
   ./docker-health-check.sh --verbose

5. View results
   cat logs/docker-health-*.log

6. If issues
   ./docker-health-check.sh --repair

7. Confirm fix
   ./docker-health-check.sh
```

## 📞 Getting Help

1. Check documentation in scripts/ directory
2. Enable `--verbose` flag for details
3. Review logs in logs/ directory
4. Run `verify-installation.sh` to diagnose
5. Check Docker Desktop/daemon status

---

**Version**: 1.0.0 | **Date**: October 17, 2025 | **Status**: Production Ready ✓
