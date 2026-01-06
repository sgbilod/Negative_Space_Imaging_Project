# Deployment Guide

Complete guide for deploying the Negative Space Imaging Project to production environments.

## Pre-Deployment Checklist

- [ ] All tests passing (100% CI/CD pipeline success)
- [ ] Code review completed and approved
- [ ] Security scanning passed (no critical vulnerabilities)
- [ ] Performance testing completed
- [ ] Database migrations tested
- [ ] Environment variables configured
- [ ] Backups configured
- [ ] Monitoring set up
- [ ] Runbooks prepared
- [ ] Team notified of deployment window

## Deployment Environments

### Development
- Local Docker Compose
- Mock data
- Debug logging
- Hot reload enabled

### Staging
- Kubernetes cluster (minikube or cloud)
- Production-like data
- Standard logging
- Monitoring enabled
- Pre-production testing

### Production
- Kubernetes cluster (cloud provider)
- Real user data
- Minimal logging
- Comprehensive monitoring
- Automatic backups
- High availability

## Docker Deployment

### Building Images

```bash
# Build Node/API backend
docker build -f Dockerfile.api -t nsi-backend:latest .
docker tag nsi-backend:latest nsi-backend:$(git rev-parse --short HEAD)

# Build Python service
docker build -f Dockerfile.python -t nsi-python:latest .
docker tag nsi-python:latest nsi-python:$(git rev-parse --short HEAD)

# Push to registry
docker push nsi-backend:latest
docker push nsi-python:latest
```

### Running Containers

```bash
# Start with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose -f docker-compose.prod.yml down

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
```

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
kubectl version --client

# Configure cluster access
kubectl config use-context production

# Verify access
kubectl get nodes
```

### Deployment Process

```bash
# Create namespace
kubectl create namespace nsi-production

# Create secrets
kubectl create secret generic db-credentials \
  --from-literal=password=$DB_PASSWORD \
  -n nsi-production

# Apply configurations
kubectl apply -f k8s/configmap.yml -n nsi-production
kubectl apply -f k8s/secrets.yml -n nsi-production
kubectl apply -f k8s/postgres.yml -n nsi-production
kubectl apply -f k8s/redis.yml -n nsi-production
kubectl apply -f k8s/backend.yml -n nsi-production
kubectl apply -f k8s/python-service.yml -n nsi-production
kubectl apply -f k8s/frontend.yml -n nsi-production

# Verify deployments
kubectl get deployments -n nsi-production
kubectl get pods -n nsi-production
```

### Scaling

```bash
# Scale deployment
kubectl scale deployment backend --replicas=3 -n nsi-production

# Auto-scaling
kubectl apply -f k8s/autoscale.yml -n nsi-production
kubectl get hpa -n nsi-production
```

## Database Migrations

### Before Deployment

```bash
# Test migrations locally
npm run migrate:test

# Generate migration
npm run migrate:generate "AddNewColumn"

# Review migration
cat migrations/[timestamp]_AddNewColumn.sql

# Commit migration
git add migrations/
git commit -m "feat(db): add migration description"
```

### During Deployment

```bash
# Run migrations
kubectl exec -it deployment/backend -n nsi-production -- npm run migrate

# Verify migrations
npm run migrate:status

# Rollback if needed
npm run migrate:rollback
```

## Blue-Green Deployment

Safe deployment strategy with zero downtime.

```yaml
# k8s/backend-blue-green.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
      version: blue
  template:
    metadata:
      labels:
        app: backend
        version: blue
    spec:
      containers:
      - name: backend
        image: nsi-backend:v1.0.0
        ports:
        - containerPort: 3000

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
      version: green
  template:
    metadata:
      labels:
        app: backend
        version: green
    spec:
      containers:
      - name: backend
        image: nsi-backend:v1.1.0
        ports:
        - containerPort: 3000

---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: backend
    version: blue  # Switch to 'green' when ready
  ports:
  - port: 80
    targetPort: 3000
```

### Deployment Steps

1. Deploy new version as "green"
2. Run smoke tests against green
3. Switch service selector to green
4. Monitor for issues
5. Remove old "blue" deployment

## Rollback Procedure

### Automatic Rollback

```bash
# Set rollout history
kubectl rollout history deployment/backend -n nsi-production

# Automatic rollback on failed health checks
kubectl set env deployment/backend ROLLBACK_ON_ERROR=true
```

### Manual Rollback

```bash
# View deployment history
kubectl rollout history deployment/backend -n nsi-production

# Rollback to previous version
kubectl rollout undo deployment/backend -n nsi-production

# Rollback to specific revision
kubectl rollout undo deployment/backend --to-revision=2 -n nsi-production

# Watch rollback progress
kubectl rollout status deployment/backend -n nsi-production
```

## Monitoring & Logging

### Prometheus Metrics

```yaml
# k8s/prometheus-config.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'backend'
    static_configs:
      - targets: ['localhost:3000']
  - job_name: 'python-service'
    static_configs:
      - targets: ['localhost:5000']
```

### Log Aggregation

```bash
# View pod logs
kubectl logs deployment/backend -n nsi-production

# Stream logs
kubectl logs -f deployment/backend -n nsi-production

# View logs from all pods
kubectl logs -l app=backend -n nsi-production --all-containers=true
```

### Alerting

```yaml
# alerting-rules.yml
groups:
  - name: application
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, http_request_duration_seconds) > 2
        for: 5m
        annotations:
          summary: "High response time detected"
```

## Security

### Container Image Scanning with Trivy

Container image vulnerability scanning is integrated into the CI/CD pipeline via Trivy, an open-source vulnerability scanner. All Docker images are automatically scanned for vulnerabilities during the build process.

#### Configuration

Trivy is configured via `.github/trivy-config.yaml`:

```yaml
# Severity levels to fail on
severity:
  - CRITICAL
  - HIGH

# Exit with non-zero code on vulnerabilities
exit-code: 1

# Output format for GitHub Security integration
format: sarif

# Skip database updates for faster CI runs
skip-db-update: true

# Scan both vulnerabilities and secrets
scanners:
  - vuln
  - secret
```

#### Scanned Components

The CI/CD pipeline (`.github/workflows/build-deploy.yml`) scans:

1. **Dockerfile.api** - Express.js API server
2. **Dockerfile.python** - Python analyzer service
3. **Dockerfile.frontend** - React frontend with Nginx
4. **Dockerfile.monitoring** - Prometheus & Grafana stack
5. **Python dependencies** - `requirements.txt`
6. **Node.js dependencies** - `package.json`

#### Local Testing

Test images locally before deployment:

```bash
# Install Trivy (macOS)
brew install trivy

# Install Trivy (Ubuntu/Debian)
sudo apt-get install trivy

# Scan Docker image
docker build -f Dockerfile.api -t nsi-api:test .
trivy image --severity CRITICAL,HIGH nsi-api:test

# Scan with config file
trivy image --config .github/trivy-config.yaml nsi-api:test

# Generate SARIF report locally
trivy image --format sarif --output report.sarif nsi-api:test

# Scan filesystem for vulnerabilities
trivy fs --severity CRITICAL,HIGH .

# Scan for secrets
trivy fs --scanners secret .
```

#### GitHub Integration

Vulnerability reports are automatically uploaded to GitHub's Security tab:

1. Navigate to repository **Security** tab
2. Select **Code scanning** section
3. View detailed vulnerability information
4. Set vulnerability severity thresholds

#### Responding to Vulnerabilities

**CRITICAL vulnerabilities:**
- Block deployment immediately
- Create emergency patch branch
- Test thoroughly before re-scanning
- Escalate to security team if unable to patch

**HIGH vulnerabilities:**
- Must be resolved before production deployment
- Update dependencies: `pip install --upgrade package`
- Rebuild image and re-scan
- Document exception if no patch available

**MEDIUM/LOW vulnerabilities:**
- Track in issue tracker
- Include in next release cycle
- Monitor for security updates

#### Base Image Updates

Regularly update base images to receive security patches:

```bash
# Update base images in Dockerfiles
# FROM node:20-alpine -> FROM node:20.10-alpine
# FROM python:3.11-slim -> FROM python:3.13-slim
# FROM nginx:1.25-alpine -> FROM nginx:1.27-alpine

# Rebuild and re-scan all images
docker compose build
trivy image --config .github/trivy-config.yaml nsi-api:latest
trivy image --config .github/trivy-config.yaml nsi-python:latest
trivy image --config .github/trivy-config.yaml nsi-frontend:latest
trivy image --config .github/trivy-config.yaml nsi-monitoring:latest
```

#### CI/CD Pipeline Integration

The security scanning stage in `.github/workflows/build-deploy.yml`:

```yaml
# STAGE 4: SCAN - Container Image Vulnerability Scanning
scan:
  name: Security Scanning
  runs-on: ubuntu-latest
  needs: build

  steps:
    # Scan all Docker images with Trivy
    - name: Run Trivy scanner
      uses: aquasecurity/trivy-action@master
      with:
        input: '/tmp/api-image.tar'
        format: 'sarif'
        output: 'trivy-api.sarif'
        severity: 'CRITICAL,HIGH'

    # Upload to GitHub Security tab
    - name: Upload SARIF results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-*.sarif'
```

#### False Positives

If a vulnerability is a false positive:

1. Document the CVE ID
2. Create `.trivyignore` file in project root:

```txt
# CVE-2021-12345: False positive - package not actually used
CVE-2021-12345
```

3. Commit and push for review

### Secrets Management

```bash
# Create secrets
kubectl create secret generic app-secrets \
  --from-literal=jwt-secret=$(openssl rand -base64 32) \
  --from-literal=db-password=$(openssl rand -base64 32) \
  -n nsi-production

# Use in deployment
env:
  - name: JWT_SECRET
    valueFrom:
      secretKeyRef:
        name: app-secrets
        key: jwt-secret
```

### Network Policies

```yaml
# k8s/network-policy.yml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-policy
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: frontend
      ports:
        - protocol: TCP
          port: 3000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
```

## Performance Optimization

### Resource Limits

```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Backup & Recovery

### Database Backups

```bash
# Create backup
kubectl exec -it pod/postgres-0 -n nsi-production -- \
  pg_dump -U admin negative_space > backup.sql

# Scheduled backups (CronJob)
kubectl apply -f k8s/backup-cronjob.yml -n nsi-production

# Restore from backup
kubectl exec -it pod/postgres-0 -n nsi-production -- \
  psql -U admin negative_space < backup.sql
```

## Maintenance

### Update Dependencies

```bash
# Check for outdated packages
npm outdated
pip list --outdated

# Update packages
npm update
pip install --upgrade pip -r requirements.txt

# Test thoroughly before deploying
```

### Node Maintenance

```bash
# Drain node (move workloads away)
kubectl drain node-name --ignore-daemonsets

# Perform maintenance
# ...

# Resume scheduling
kubectl uncordon node-name
```

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n nsi-production

# View logs
kubectl logs <pod-name> -n nsi-production

# Check resource availability
kubectl describe nodes
```

### Service Not Accessible

```bash
# Check service
kubectl get svc -n nsi-production
kubectl describe svc backend-service -n nsi-production

# Check endpoints
kubectl get endpoints -n nsi-production

# Test connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -- sh
wget http://backend-service:80/health
```

### Performance Issues

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n nsi-production

# Check metrics
kubectl get hpa -n nsi-production

# View metrics details
kubectl describe hpa backend-hpa -n nsi-production
```

---

Last Updated: November 2025

**Deployment Readiness:** ✅ All systems ready for production deployment
