# Negative Space Imaging Project - Deployment Automation Guide

This guide provides detailed instructions for automating the deployment of the Negative Space Imaging Project using the deployment tools provided in this repository.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup Requirements](#setup-requirements)
3. [Docker Compose Deployment](#docker-compose-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Continuous Integration/Continuous Deployment](#continuous-integrationcontinuous-deployment)
6. [Advanced Deployment Scenarios](#advanced-deployment-scenarios)
7. [Troubleshooting](#troubleshooting)

## Introduction

The Negative Space Imaging Project deployment system provides a comprehensive set of tools to automate the deployment, monitoring, and testing of the application across different environments. This guide focuses specifically on the automation aspects to help you integrate the deployment into your CI/CD pipelines or automation scripts.

## Setup Requirements

Before you begin, ensure you have the following prerequisites installed:

### For Development Environment (Docker Compose)

- Docker Engine (19.03.0+)
- Docker Compose (1.27.0+)
- Python 3.7+
- Required Python packages (install using `pip install -r deployment/requirements.txt`)

### For Production Environment (Kubernetes)

- Kubernetes cluster (1.19+)
- kubectl configured to access your cluster
- Helm (optional, for additional deployments)
- Python 3.7+
- Required Python packages (install using `pip install -r deployment/requirements.txt`)

## Docker Compose Deployment

### Step 1: Prepare the Environment

Create a `.env` file in the deployment directory with the required environment variables:

```bash
# Create .env file with default values
cat << EOF > deployment/.env
DB_PASSWORD=secure_password_here
IMAGE_TAG=latest
DEBUG_MODE=false
EOF
```

### Step 2: Automated Deployment

Use the `deploy_auto.py` script to automate the deployment:

```bash
# Deploy with Docker Compose
python deployment/deploy_auto.py deploy --type docker-compose --build

# Check deployment status
python deployment/deploy_auto.py status --type docker-compose

# Export deployment status to JSON
python deployment/deploy_auto.py status --type docker-compose --export status.json
```

### Step 3: Verify Deployment

Use the verification and health check tools to ensure the deployment is working correctly:

```bash
# Run health check
python deployment/health_check.py --deployment-type docker-compose

# Verify deployment configuration
python deployment/verify_deployment.py --deployment-dir ./deployment

# Run integration tests
python deployment/test_deployment.py --type docker-compose
```

### Step 4: Automated Cleanup

When you're done, clean up the deployment:

```bash
# Clean up Docker Compose deployment
python deployment/deploy_auto.py cleanup --type docker-compose
```

### Example: Complete Deployment Script

Here's a complete example script for automating Docker Compose deployment:

```bash
#!/bin/bash
set -e

# Navigate to project directory
cd /path/to/negative-space-imaging-project

# Set environment variables
export DB_PASSWORD=$(openssl rand -base64 16)
export IMAGE_TAG=$(git rev-parse --short HEAD)

# Save environment variables to .env file
cat << EOF > deployment/.env
DB_PASSWORD=$DB_PASSWORD
IMAGE_TAG=$IMAGE_TAG
DEBUG_MODE=false
EOF

# Deploy with Docker Compose
python deployment/deploy_auto.py deploy --type docker-compose --build

# Wait for services to start
sleep 10

# Check deployment health
python deployment/health_check.py --deployment-type docker-compose --export health_report.json

# Run integration tests
python deployment/test_deployment.py --type docker-compose --export test_results.json

# Exit with the test result status code
exit $?
```

## Kubernetes Deployment

### Step 1: Prepare the Kubernetes Environment

Ensure your kubectl is configured correctly and create a namespace:

```bash
# Create namespace
kubectl create namespace negative-space-imaging

# Create secrets
kubectl create secret generic db-credentials \
  --from-literal=password=$(openssl rand -base64 16) \
  --namespace negative-space-imaging
```

### Step 2: Automated Deployment

Use the `deploy_auto.py` script to automate the deployment:

```bash
# Deploy to Kubernetes
python deployment/deploy_auto.py deploy --type kubernetes --namespace negative-space-imaging

# Check deployment status
python deployment/deploy_auto.py status --type kubernetes --namespace negative-space-imaging

# Export deployment status to JSON
python deployment/deploy_auto.py status --type kubernetes --namespace negative-space-imaging --export k8s-status.json
```

### Step 3: Verify Kubernetes Deployment

Use the verification and health check tools to ensure the deployment is working correctly:

```bash
# Run health check
python deployment/health_check.py --deployment-type kubernetes --namespace negative-space-imaging

# Verify deployment configuration
python deployment/verify_deployment.py --deployment-dir ./deployment

# Run integration tests
python deployment/test_deployment.py --type kubernetes --namespace negative-space-imaging
```

### Step 4: Automated Cleanup

When you're done, clean up the deployment:

```bash
# Clean up Kubernetes deployment
python deployment/deploy_auto.py cleanup --type kubernetes --namespace negative-space-imaging
```

### Example: Complete Kubernetes Deployment Script

Here's a complete example script for automating Kubernetes deployment:

```bash
#!/bin/bash
set -e

# Navigate to project directory
cd /path/to/negative-space-imaging-project

# Set environment variables
export NAMESPACE=negative-space-imaging
export IMAGE_TAG=$(git rev-parse --short HEAD)

# Ensure namespace exists
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Create or update secrets
kubectl create secret generic db-credentials \
  --from-literal=password=$(openssl rand -base64 16) \
  --namespace $NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy to Kubernetes
python deployment/deploy_auto.py deploy --type kubernetes --namespace $NAMESPACE

# Wait for deployments to be ready
sleep 30

# Check deployment health
python deployment/health_check.py --deployment-type kubernetes --namespace $NAMESPACE --export k8s_health_report.json

# Run integration tests
python deployment/test_deployment.py --type kubernetes --namespace $NAMESPACE --export k8s_test_results.json

# Exit with the test result status code
exit $?
```

## Continuous Integration/Continuous Deployment

You can integrate the deployment automation into your CI/CD pipeline using GitHub Actions, Jenkins, GitLab CI, or other CI/CD tools.

### GitHub Actions Example

Here's an example GitHub Actions workflow file:

```yaml
name: Deploy Negative Space Imaging Project

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r deployment/requirements.txt

    - name: Set up Docker
      uses: docker/setup-buildx-action@v1

    - name: Build and deploy with Docker Compose
      run: |
        python deployment/deploy_auto.py deploy --type docker-compose --build

    - name: Run health check
      run: |
        python deployment/health_check.py --deployment-type docker-compose --export health_report.json

    - name: Run integration tests
      run: |
        python deployment/test_deployment.py --type docker-compose --export test_results.json

    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: |
          health_report.json
          test_results.json

    - name: Clean up deployment
      if: always()
      run: |
        python deployment/deploy_auto.py cleanup --type docker-compose
```

## Advanced Deployment Scenarios

### Blue-Green Deployment

For zero-downtime deployments, you can implement a blue-green deployment strategy:

```bash
#!/bin/bash
set -e

# Navigate to project directory
cd /path/to/negative-space-imaging-project

# Set environment variables
export NAMESPACE=negative-space-imaging
export NEW_COLOR=blue
export OLD_COLOR=green

# Determine current active deployment
ACTIVE_DEPLOYMENT=$(kubectl get service main-service -n $NAMESPACE -o jsonpath='{.spec.selector.deployment}' 2>/dev/null || echo "none")

# Switch colors if needed
if [ "$ACTIVE_DEPLOYMENT" == "$NEW_COLOR" ]; then
  NEW_COLOR=green
  OLD_COLOR=blue
fi

echo "Current active deployment: $ACTIVE_DEPLOYMENT"
echo "New deployment color: $NEW_COLOR"
echo "Old deployment color: $OLD_COLOR"

# Deploy new version
python deployment/deploy_auto.py deploy --type kubernetes --namespace $NAMESPACE --color $NEW_COLOR

# Wait for new deployment to be ready
sleep 30

# Run health check on new deployment
python deployment/health_check.py --deployment-type kubernetes --namespace $NAMESPACE --color $NEW_COLOR

# Run integration tests on new deployment
TESTS_PASSED=$(python deployment/test_deployment.py --type kubernetes --namespace $NAMESPACE --color $NEW_COLOR)

# Switch traffic to new deployment if tests passed
if [ $TESTS_PASSED -eq 0 ]; then
  echo "Tests passed, switching traffic to $NEW_COLOR deployment"
  kubectl patch service main-service -n $NAMESPACE --type=json -p='[{"op": "replace", "path": "/spec/selector/deployment", "value":"'$NEW_COLOR'"}]'

  # Clean up old deployment after waiting period
  sleep 60
  python deployment/deploy_auto.py cleanup --type kubernetes --namespace $NAMESPACE --color $OLD_COLOR
else
  echo "Tests failed, keeping traffic on $OLD_COLOR deployment"
  # Clean up failed deployment
  python deployment/deploy_auto.py cleanup --type kubernetes --namespace $NAMESPACE --color $NEW_COLOR
  exit 1
fi
```

### Multi-Region Deployment

For high availability, you can deploy to multiple regions:

```bash
#!/bin/bash
set -e

# Define regions
REGIONS=("us-east-1" "eu-west-1" "ap-southeast-1")

# Deploy to each region
for REGION in "${REGIONS[@]}"; do
  echo "Deploying to region $REGION"

  # Set kubectl context to the region
  kubectl config use-context $REGION

  # Deploy to the region
  python deployment/deploy_auto.py deploy --type kubernetes --namespace negative-space-imaging --region $REGION

  # Verify deployment
  python deployment/health_check.py --deployment-type kubernetes --namespace negative-space-imaging --region $REGION
done

# Run integration tests against all regions
for REGION in "${REGIONS[@]}"; do
  echo "Testing deployment in region $REGION"

  # Set kubectl context to the region
  kubectl config use-context $REGION

  # Run tests
  python deployment/test_deployment.py --type kubernetes --namespace negative-space-imaging --region $REGION
done
```

## Troubleshooting

### Common Deployment Issues

1. **Docker Compose Services Fail to Start**

   Check the service logs:

   ```bash
   docker-compose -f deployment/docker-compose.yaml logs <service-name>
   ```

   Run the health check for detailed diagnostics:

   ```bash
   python deployment/health_check.py --deployment-type docker-compose --watch
   ```

2. **Kubernetes Pods Stuck in Pending State**

   Check pod events:

   ```bash
   kubectl describe pod <pod-name> -n negative-space-imaging
   ```

   Check node resources:

   ```bash
   kubectl describe nodes
   ```

3. **Database Initialization Failures**

   Check database logs:

   ```bash
   # For Docker Compose
   docker-compose -f deployment/docker-compose.yaml logs database

   # For Kubernetes
   kubectl logs <database-pod-name> -n negative-space-imaging
   ```

4. **Integration Tests Failing**

   Run tests with verbose logging:

   ```bash
   python deployment/test_deployment.py --type <deployment-type> --verbose
   ```

5. **Monitoring Services Not Accessible**

   Check service status:

   ```bash
   # For Docker Compose
   docker-compose -f deployment/docker-compose.yaml ps

   # For Kubernetes
   kubectl get pods -n negative-space-imaging
   ```

   Use the monitoring dashboard script:

   ```bash
   python deployment/monitoring_dashboard.py --type <deployment-type> --debug
   ```

### Collecting Diagnostic Information

When troubleshooting, collect comprehensive diagnostic information:

```bash
# Create a diagnostics directory
mkdir -p diagnostics

# Collect deployment status
python deployment/deploy_auto.py status --type <deployment-type> --export diagnostics/deployment-status.json

# Run health check
python deployment/health_check.py --deployment-type <deployment-type> --export diagnostics/health-check.json

# Verify configuration
python deployment/verify_deployment.py --deployment-dir ./deployment --export diagnostics/verification-results.json

# Collect logs
# For Docker Compose
docker-compose -f deployment/docker-compose.yaml logs > diagnostics/docker-compose-logs.txt

# For Kubernetes
kubectl logs -l app=imaging-service -n negative-space-imaging > diagnostics/imaging-service-logs.txt
kubectl logs -l app=database -n negative-space-imaging > diagnostics/database-logs.txt
kubectl logs -l app=prometheus -n negative-space-imaging > diagnostics/prometheus-logs.txt
kubectl logs -l app=grafana -n negative-space-imaging > diagnostics/grafana-logs.txt

# Package the diagnostics
tar -czf diagnostics.tar.gz diagnostics/
```

With this comprehensive information, you'll be better equipped to diagnose and resolve deployment issues.
