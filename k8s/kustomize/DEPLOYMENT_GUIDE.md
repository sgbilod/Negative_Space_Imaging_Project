# Kustomize Deployment Guide

## Quick Start

### Deploy to Development (Local)

```bash
# Navigate to project root
cd k8s/kustomize

# Build and view manifests
kustomize build overlays/dev/

# Deploy to cluster
kubectl apply -k overlays/dev/

# Verify deployment
kubectl get pods -n negative-space
kubectl describe svc -n negative-space
```

**Expected Result:**
- 1 API pod, 1 ML processor pod, 1 web viewer pod
- All services as ClusterIP
- DEBUG logging, metrics disabled
- Resource limits: CPU 250-500m, Memory 256Mi-512Mi

### Deploy to Staging (Testing)

```bash
# Prerequisites
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.0/deploy/static/provider/cloud/deploy.yaml

# Deploy
kubectl apply -k overlays/staging/

# Verify
kubectl get pods -n negative-space
kubectl get svc -n negative-space
kubectl get ingress -n negative-space

# Test LoadBalancer access
STAGING_IP=$(kubectl get svc negative-space-api -n negative-space -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$STAGING_IP:8000/health/ready
```

**Expected Result:**
- 2 API pods, 2 ML processor pods, 2 web viewer pods
- API and Web services as LoadBalancer
- INFO logging, metrics enabled
- Ingress configured for staging.negative-space.dev
- HPA active (min 2, max 5 replicas, CPU 70%)

### Deploy to Production (HA)

```bash
# Prerequisites
# 1. Ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.0/deploy/static/provider/cloud/deploy.yaml

# 2. Cert-manager for TLS
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true

# 3. Network policies must be enabled on cluster
# Contact cluster admin if not enabled

# 4. Verify cluster capacity (6+ CPU, 8Gi memory minimum)
kubectl top nodes
kubectl top pods --all-namespaces

# Deploy with caution - consider phased approach
kubectl apply -k overlays/prod/

# Monitor rollout
kubectl rollout status deployment/negative-space-api -n negative-space

# Verify all components
kubectl get all -n negative-space
kubectl get networkpolicies -n negative-space
kubectl get poddisruptionbudgets -n negative-space
```

**Expected Result:**
- 3 API pods, 3 ML processor pods, 3 web viewer pods
- Services as LoadBalancer (externalTrafficPolicy: Local)
- Network policies enforcing traffic rules
- Pod Disruption Budgets protecting 2 replicas minimum
- HPA aggressive (min 3, max 10, CPU 60%)
- Pods spread across nodes (pod anti-affinity)
- TLS Ingress configured

## Environment Comparison Matrix

| Aspect | Dev | Staging | Production |
|--------|-----|---------|------------|
| **Replicas** | 1 | 2 | 3+ |
| **CPU Request** | 250m | 500m | 1000m |
| **Memory Request** | 256Mi | 512Mi | 1Gi |
| **Image Policy** | Always | IfNotPresent | IfNotPresent |
| **Logging Level** | DEBUG | INFO | WARN |
| **Metrics** | Disabled | Enabled | Enabled |
| **Service Type** | ClusterIP | LoadBalancer | LoadBalancer |
| **Ingress** | None | Yes | Yes + TLS |
| **HPA** | Disabled | Enabled (5) | Enabled (10) |
| **Network Policies** | None | None | Strict |
| **Pod Anti-Affinity** | None | None | Required |
| **PDB** | None | None | minAvailable=2 |
| **Domain** | local | staging.negative-space.dev | api.negative-space.io |

## Configuration Inheritance

Kustomize uses base + overlays pattern:

```
┌─────────────────────────────┐
│     base/                   │
│  (Common for all envs)      │
│                             │
│  - namespace                │
│  - rbac                     │
│  - configmap (base)         │
│  - secrets (template)       │
│  - deployments (generic)    │
│  - services (ClusterIP)     │
│  - hpa (template)           │
└─────────────────────────────┘
         ▲  ▲  ▲
         │  │  │
    ┌────┘  │  └────┐
    │       │       │
    ▼       ▼       ▼
┌──────┐ ┌────────┐ ┌──────┐
│ dev/ │ │staging/│ │ prod/│
└──────┘ └────────┘ └──────┘
```

Each overlay inherits from base and applies patches:
- Replica count patches
- Resource limits patches
- Environment variables patches
- Service type patches (staging/prod)
- Affinity patches (prod only)

## Customization Examples

### Change Replica Count

Edit `overlays/prod/kustomization.yaml`:

```yaml
replicas:
- name: negative-space-api
  count: 5  # Change from 3 to 5
- name: negative-space-ml-processor
  count: 5
- name: negative-space-web-viewer
  count: 5
```

Apply: `kubectl apply -k overlays/prod/`

### Update Image Tags

Edit `overlays/prod/kustomization.yaml`:

```yaml
images:
- name: negative-space-api
  newTag: v1.0.1  # Update version
- name: negative-space-ml-processor
  newTag: v1.0.1
- name: negative-space-web-viewer
  newTag: v1.0.1
```

Apply: `kubectl apply -k overlays/prod/`

### Modify Resource Limits

Edit `overlays/prod/patches/resource-limits-patch.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: negative-space-api
spec:
  template:
    spec:
      containers:
      - name: api
        resources:
          requests:
            cpu: "1500m"  # Increased from 1000m
            memory: "1.5Gi"
          limits:
            cpu: "3000m"  # Increased from 2000m
            memory: "3Gi"
```

Apply: `kubectl apply -k overlays/prod/`

### Add Environment Variables

Edit `overlays/prod/patches/env-vars-patch.yaml`:

```yaml
env:
- name: NEW_VAR
  value: "new_value"
- name: FEATURE_FLAG
  value: "true"
```

Apply: `kubectl apply -k overlays/prod/`

## Validation Workflows

### Pre-deployment Validation

```bash
#!/bin/bash
set -e

echo "=== Validating Kustomize Configuration ==="

# 1. Validate YAML syntax
echo "1. Checking YAML syntax..."
kustomize build overlays/prod/ > /tmp/prod-manifest.yaml
kubectl apply -f /tmp/prod-manifest.yaml --dry-run=client

# 2. Check for required fields
echo "2. Checking required fields..."
grep -q "image:" /tmp/prod-manifest.yaml || echo "WARNING: No images found"
grep -q "resources:" /tmp/prod-manifest.yaml || echo "WARNING: No resources defined"
grep -q "livenessProbe:" /tmp/prod-manifest.yaml || echo "WARNING: No liveness probes"

# 3. Validate image tags (no :latest in prod)
echo "3. Checking image tags..."
if grep -q ":latest" /tmp/prod-manifest.yaml; then
  echo "ERROR: Found :latest tags in production manifest"
  exit 1
fi

# 4. Check resource requests/limits
echo "4. Checking resource definitions..."
if ! grep -q "requests:" /tmp/prod-manifest.yaml; then
  echo "ERROR: Resource requests not defined"
  exit 1
fi

echo "✓ All validations passed"
```

### Dry-Run Deployment

```bash
# View what would be deployed
kubectl apply -k overlays/prod/ --dry-run=client -o yaml

# Test actual API with validation
kubectl apply -k overlays/prod/ --dry-run=server

# Check for API deprecations
kubectl api-resources | grep -E "v1beta|deprecated"
```

### Post-deployment Verification

```bash
#!/bin/bash

echo "=== Verifying Production Deployment ==="

# 1. Check pod status
echo "1. Checking pod status..."
kubectl get pods -n negative-space

# 2. Check service endpoints
echo "2. Checking service endpoints..."
kubectl get svc -n negative-space -o wide

# 3. Verify HPA status
echo "3. Checking HPA status..."
kubectl get hpa -n negative-space

# 4. Check network policies
echo "4. Checking network policies..."
kubectl get networkpolicies -n negative-space

# 5. Verify pod disruption budgets
echo "5. Checking Pod Disruption Budgets..."
kubectl get poddisruptionbudgets -n negative-space

# 6. Test connectivity
echo "6. Testing service connectivity..."
API_POD=$(kubectl get pod -n negative-space -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n negative-space $API_POD -- curl -s http://localhost:8000/health/ready || echo "Health check failed"

echo "✓ Verification complete"
```

## Troubleshooting Guide

### Issue: Pods in ImagePullBackOff

```bash
# Check image availability
kubectl describe pod <pod-name> -n negative-space

# Solutions:
# 1. Verify image repository access
# 2. Check imagePullSecrets in serviceAccount
# 3. Ensure registry credentials are configured
kubectl get secrets -n negative-space
```

### Issue: CrashLoopBackOff

```bash
# Check pod logs
kubectl logs <pod-name> -n negative-space
kubectl logs <pod-name> -n negative-space --previous

# Check resource constraints
kubectl top pod <pod-name> -n negative-space
kubectl describe node <node-name>
```

### Issue: Pods not spreading across nodes

```bash
# Check node affinity rules
kubectl get pods -n negative-space -o wide

# Verify anti-affinity configuration
kubectl get pod <pod-name> -n negative-space -o yaml | grep -A 10 affinity

# Solutions for production:
# 1. Ensure cluster has 3+ nodes
# 2. Check node labels for affinity matching
# 3. Verify nodeAffinity requirements
```

### Issue: HPA not scaling

```bash
# Check HPA status
kubectl describe hpa negative-space-api-hpa -n negative-space

# Check metrics availability
kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes
kubectl get --raw /apis/metrics.k8s.io/v1beta1/namespaces/negative-space/pods

# Ensure metrics-server is running
kubectl get deployment metrics-server -n kube-system
kubectl logs -n kube-system -l k8s-app=metrics-server
```

## Production Deployment Checklist

- [ ] Cluster has 3+ worker nodes (for HA)
- [ ] Network policies enabled on cluster
- [ ] Ingress controller deployed (nginx-ingress)
- [ ] cert-manager installed and configured
- [ ] TLS certificates valid and not expiring soon
- [ ] Database credentials stored in sealed-secrets
- [ ] All image tags are specific versions (not :latest)
- [ ] Resource requests/limits validated against cluster capacity
- [ ] Pod disruption budgets configured
- [ ] Network policies reviewed and tested
- [ ] Pod anti-affinity rules tested (pods spread across nodes)
- [ ] HPA configured with appropriate thresholds
- [ ] Monitoring and alerting configured
- [ ] Backup and disaster recovery plan in place
- [ ] Security scan completed
- [ ] Load testing completed
- [ ] Rollback procedure documented
- [ ] Change management approval obtained

## Rollback Procedures

### Immediate Rollback

```bash
# Rollback previous Kubernetes rollout
kubectl rollout undo deployment/negative-space-api -n negative-space
kubectl rollout undo deployment/negative-space-ml-processor -n negative-space
kubectl rollout undo deployment/negative-space-web-viewer -n negative-space

# Monitor rollback
kubectl rollout status deployment/negative-space-api -n negative-space
```

### Full Environment Rollback

```bash
# Restore previous Kustomize build
git checkout HEAD~1 k8s/kustomize/overlays/prod/

# Reapply previous configuration
kubectl apply -k overlays/prod/

# Verify
kubectl get pods -n negative-space
```

## References

- Kustomize official docs: https://kustomize.io/
- Kubernetes Deployments: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- HPA: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- Network Policies: https://kubernetes.io/docs/concepts/services-networking/network-policies/

---

**Version:** 1.0
**Last Updated:** December 2024
