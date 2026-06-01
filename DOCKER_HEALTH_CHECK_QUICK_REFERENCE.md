# Docker Health Check System - Quick Reference

## 📋 What You Have

A complete production-grade health monitoring system with **6 deliverables**:

1. **Bash Script** (`scripts/docker-health-check.sh`) - Linux/macOS
2. **PowerShell Script** (`scripts/docker-health-check.ps1`) - Windows
3. **Node.js Module** (`src/health-checks/health-check.js`) - JavaScript/TypeScript
4. **Python Module** (`src/health-checks/health_monitor.py`) - Python
5. **Interactive Dashboard** (`public/health-dashboard.html`) - Web UI
6. **Documentation** (`scripts/DOCKER_HEALTH_CHECK_README.md`) - Complete guide

---

## 🚀 Quick Start (Choose Your Platform)

### Linux/macOS
```bash
chmod +x scripts/docker-health-check.sh
./scripts/docker-health-check.sh
```

### Windows (PowerShell)
```powershell
.\scripts\docker-health-check.ps1 -Verbose
```

### Node.js
```bash
node src/health-checks/health-check.js
```

### Python
```bash
pip install aiohttp psutil
python src/health-checks/health_monitor.py
```

### Web Dashboard
```bash
# Start a web server
python -m http.server 8000
# Open: http://localhost:8000/public/health-dashboard.html
```

---

## ✨ What It Monitors

| Service | Check | Location |
|---------|-------|----------|
| PostgreSQL | Port connectivity | localhost:5432 |
| Redis | Port connectivity | localhost:6379 |
| Node.js | Port + API endpoint | localhost:3000/api/health |
| Python | Port + API endpoint | localhost:5000/health |
| React | Port + frontend access | localhost:3001/ |
| System | CPU/Memory/Disk | System-wide |

---

## 📊 Output Examples

### Console Output
```
✓ PostgreSQL: HEALTHY
✓ Redis: HEALTHY
✓ Node.js: HEALTHY
✓ Python: HEALTHY
✓ React: HEALTHY
✓ CPU Usage: HEALTHY - 45.2%
✓ Memory Usage: HEALTHY - 62.8%
✓ Disk Usage: HEALTHY - 38.4%

================================================================================
DOCKER HEALTH CHECK SUMMARY
================================================================================
Overall Status: HEALTHY
Total Services: 10
Healthy: 10
Warnings: 0
Failures: 0
================================================================================
```

### JSON Report
```json
{
  "timestamp": "2025-11-08T12:00:00Z",
  "overall_status": "HEALTHY",
  "check_duration_seconds": 2.34,
  "summary": {
    "total_services": 10,
    "healthy": 10,
    "warnings": 0,
    "failures": 0
  }
}
```

---

## 🔧 Configuration

Set environment variables in `.env.local`:

```bash
# Services
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
REDIS_HOST=localhost
REDIS_PORT=6379
NODE_HOST=localhost
NODE_PORT=3000
PYTHON_HOST=localhost
PYTHON_PORT=5000
REACT_HOST=localhost
REACT_PORT=3001

# Logging
LOG_DIR=./logs
```

---

## 🔌 Integration Examples

### Docker Compose Health Check
```yaml
healthcheck:
  test: ["CMD", "bash", "scripts/docker-health-check.sh"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Express.js Middleware
```javascript
const { healthCheckMiddleware } = require('./src/health-checks/health-check');
app.get('/api/health', healthCheckMiddleware(checker));
```

### Flask Blueprint
```python
from health_monitor import create_flask_blueprint
app.register_blueprint(create_flask_blueprint(), url_prefix='/api')
```

### Kubernetes Probe
```yaml
livenessProbe:
  httpGet:
    path: /api/health
    port: 3000
  initialDelaySeconds: 30
  periodSeconds: 10
```

---

## 📁 File Locations

```
Project Root/
├── scripts/
│   ├── docker-health-check.sh          ← Bash script
│   ├── docker-health-check.ps1         ← PowerShell script
│   └── DOCKER_HEALTH_CHECK_README.md   ← Full documentation
│
├── src/health-checks/
│   ├── health-check.js                 ← Node.js module
│   └── health_monitor.py               ← Python module
│
├── public/
│   └── health-dashboard.html           ← Interactive dashboard
│
└── logs/
    ├── health-check-YYYYMMDD.log       ← Daily logs
    └── health-report-TIMESTAMP.json    ← Reports
```

---

## ⚡ Common Commands

### Run all health checks with details
```bash
# Bash
./scripts/docker-health-check.sh -v

# PowerShell
.\scripts\docker-health-check.ps1 -Verbose

# Python
python src/health-checks/health_monitor.py
```

### View latest report
```bash
cat logs/health-report-*.json | tail -1 | jq .
```

### Watch health checks continuously
```bash
watch -n 5 './scripts/docker-health-check.sh'
```

### Schedule automated checks (Linux/macOS)
```bash
# Add to crontab (every 5 minutes)
*/5 * * * * /path/to/docker-health-check.sh >> /path/to/logs/cron.log 2>&1
```

### Download dashboard as standalone
```bash
# Open in browser
file:///path/to/public/health-dashboard.html
```

---

## 🎯 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All services healthy ✓ |
| 1 | One or more services failed ✗ |
| 2 | Configuration or execution error |

---

## 🔍 Troubleshooting

### "Port not responding"
```bash
# Check if service is running
netstat -tuln | grep PORTNUMBER
ss -tuln | grep PORTNUMBER

# Test connectivity
telnet localhost PORTNUMBER
```

### "Connection refused"
```bash
# Verify service is listening
curl http://localhost:PORT/health
```

### "No such file or directory"
```bash
# Make script executable
chmod +x scripts/docker-health-check.sh
```

### Missing Python dependencies
```bash
pip install aiohttp psutil
```

---

## 📈 Next Steps

1. **Test It**: Run health checks on your services
2. **Integrate**: Add to Docker Compose or Kubernetes
3. **Monitor**: Set up automated checks with cron or Docker
4. **Alert**: Configure notifications on failures
5. **Track**: Archive reports for trend analysis

---

## 📚 Full Documentation

For comprehensive documentation, configuration details, and advanced usage:

```
scripts/DOCKER_HEALTH_CHECK_README.md
```

Contains:
- Detailed installation instructions
- All configuration options
- Advanced integration patterns
- Security best practices
- Troubleshooting guide
- Roadmap for future features

---

## ✅ Verification Checklist

- [ ] All services are running (PostgreSQL, Redis, Node.js, Python, React)
- [ ] Health check script is executable
- [ ] Environment variables are set (or defaults work)
- [ ] Logs directory exists with write permissions
- [ ] First health check runs without errors
- [ ] JSON report is generated
- [ ] Dashboard opens in browser
- [ ] All 5 services show as HEALTHY

---

## 🎉 You're All Set!

All 6 deliverables are production-ready. Choose your preferred platform and start monitoring!

**Questions?** Check the full documentation or troubleshooting guide.

---

**Version**: 1.0.0
**Last Updated**: 2025-11-08
**Status**: ✅ Production Ready
