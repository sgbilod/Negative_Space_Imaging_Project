# Kustomize Configuration Management for Negative Space Imaging

## Overview

This directory contains Kubernetes manifests organized using Kustomize for managing environment-specific configurations across development, staging, and production environments.

## Architecture

```
k8s/kustomize/
├── base/                          # Base manifests (all common configurations)
│   ├── kustomization.yaml         # Base kustomization configuration
│   ├── namespace.yaml             # Kubernetes namespace definition
│   ├── rbac.yaml                  # ServiceAccount, ClusterRole, ClusterRoleBinding
│   ├── configmap.yaml             # Common application configuration
│   ├── secrets.yaml               # Secret templates (values from overlays)
│   ├── deployment.yaml            # Deployments for all services (API, ML, Web)
│   ├── service.yaml               # Service definitions (ClusterIP)
│   └── hpa.yaml                   # Horizontal Pod Autoscaler templates
│
└── overlays/                      # Environment-specific customizations
    ├── dev/                       # Development environment
    │   ├── kustomization.yaml     # Dev-specific overrides
    │   └── patches/               # Dev patches
    │       ├── replica-patch.yaml
    │       ├── resource-limits-patch.yaml
    │       └── env-vars-patch.yaml
    │
    ├── staging/                   # Staging environment
    │   ├── kustomization.yaml     # Staging-specific overrides
    │   ├── ingress.yaml          # Staging ingress configuration
    │   └── patches/               # Staging patches
    │       ├── replica-patch.yaml
    │       ├── resource-limits-patch.yaml
    │       ├── env-vars-patch.yaml
    │       └── service-patch.yaml
    │
    └── prod/                      # Production environment
        ├── kustomization.yaml     # Prod-specific overrides
        ├── ingress.yaml          # Production ingress (TLS, domains)
        ├── network-policy.yaml    # Network policies for traffic control
        ├── pod-disruption-budget.yaml # HA and disruption protection
        └── patches/               # Production patches
            ├── replica-patch.yaml
            ├── resource-limits-patch.yaml
            ├── env-vars-patch.yaml
            ├── service-patch.yaml
            └── affinity-patch.yaml

```

## Environment Specifications

### Development Environment

**Purpose:** Local development with minimal resources

| Component | Configuration |
|-----------|---------------|
| **Replicas** | 1 per service |
| **CPU Request** | 250m (API), 500m (ML), 100m (Web) |
| **Memory Request** | 256Mi (API), 512Mi (ML), 128Mi (Web) |
| **CPU Limit** | 500m (API), 1000m (ML), 250m (Web) |
| **Memory Limit** | 512Mi (API), 1Gi (ML), 256Mi (Web) |
| **Image Pull** | Always (latest dev images) |
| **Service Type** | ClusterIP (internal only) |
| **Logging** | DEBUG level, verbose output |
| **Metrics** | Disabled (reduce overhead) |
| **HPA** | Disabled (minReplicas=1) |
| **Domain** | local-dev (internal) |

**Use Case:** Developer laptops, local Kubernetes (minikube, kind, Docker Desktop)

### Staging Environment

**Purpose:** Pre-production testing with HA capabilities

| Component | Configuration |
|-----------|---------------|
| **Replicas** | 2 per service (HA testing) |
| **CPU Request** | 500m (API), 750m (ML), 250m (Web) |
| **Memory Request** | 512Mi (API), 768Mi (ML), 256Mi (Web) |
| **CPU Limit** | 1000m (API), 1500m (ML), 500m (Web) |
| **Memory Limit** | 1Gi (API), 1.5Gi (ML), 512Mi (Web) |
| **Image Pull** | IfNotPresent (use release tags) |
| **Service Type** | LoadBalancer (external access for testing) |
| **Logging** | INFO level, structured logs |
| **Metrics** | Enabled (monitoring test) |
| **HPA** | Enabled (minReplicas=2, maxReplicas=5, CPU 70%) |
| **Ingress** | staging.negative-space.dev (TLS) |
| **Domain** | staging.negative-space.dev |

**Use Case:** QA testing, staging servers, pre-release validation

### Production Environment

**Purpose:** High-availability production deployment

| Component | Configuration |
|-----------|---------------|
| **Replicas** | 3+ per service (HA) |
| **CPU Request** | 1000m (API), 1500m (ML), 500m (Web) |
| **Memory Request** | 1Gi (API), 1.5Gi (ML), 512Mi (Web) |
| **CPU Limit** | 2000m (API), 2500m (ML), 1000m (Web) |
| **Memory Limit** | 2Gi (API), 2.5Gi (ML), 1Gi (Web) |
| **Image Pull** | IfNotPresent (specific release tags) |
| **Service Type** | LoadBalancer + Ingress (external traffic) |
| **Logging** | WARN level, minimal overhead |
| **Metrics** | Enabled (full monitoring stack) |
| **HPA** | Aggressive (minReplicas=3, maxReplicas=10, CPU 60%) |
| **Pod Affinity** | Required anti-affinity (spread across nodes) |
| **Network Policies** | Strict traffic control between services |
| **Ingress** | api.negative-space.io, viewer.negative-space.io (TLS) |
| **PDB** | minAvailable=2 (disruption budget) |
| **Domain** | Production domains with TLS/SSL |

**Use Case:** Customer-facing production, SLA requirements, full HA

## Building and Deploying

### Build Base Configuration

Validate the base configuration:

```bash
kustomize build base/
# Output: All K8s manifests for base configuration
```

### Build Development Overlay

Generate dev-specific manifests:

```bash
kustomize build overlays/dev/
# Output: Development environment manifests (1 replica, debug logging, no metrics)
```

### Build Staging Overlay

Generate staging-specific manifests:

```bash
kustomize build overlays/staging/
# Output: Staging manifests (2 replicas, info logging, LoadBalancer, Ingress)
```

### Build Production Overlay

Generate production-specific manifests:

```bash
kustomize build overlays/prod/
# Output: Production manifests (3+ replicas, warn logging, full HA, network policies)
```

## Deployment Instructions

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl installed
- kustomize installed (v4.0+)
- Ingress controller (nginx-ingress for Staging/Prod)
- cert-manager (for TLS certificates)

### Deploy to Development

```bash
# Dry-run
kubectl kustomize overlays/dev/ | kubectl apply --dry-run=client -f -

# Deploy
kubectl apply -k overlays/dev/

# Verify
kubectl get pods -n negative-space
kubectl logs -n negative-space -l app.kubernetes.io/component=api
```

### Deploy to Staging

```bash
# Prerequisites: Ingress controller installed

# Dry-run
kustomize build overlays/staging/ | kubectl apply --dry-run=client -f -

# Deploy
kubectl apply -k overlays/staging/

# Verify
kubectl get svc -n negative-space
kubectl get ingress -n negative-space
```

### Deploy to Production

```bash
# Prerequisites:
# - Ingress controller installed
# - cert-manager installed
# - Network policies enabled
# - PDB requirements met

# Dry-run
kustomize build overlays/prod/ | kubectl apply --dry-run=client -f -

# Deploy with caution (recommend phased rollout)
kubectl apply -k overlays/prod/

# Verify
kubectl get pods -n negative-space -o wide
kubectl get svc,ingress -n negative-space
kubectl get networkpolicies -n negative-space
kubectl get poddisruptionbudgets -n negative-space
```

## Configuration Management

### Secrets Management

Secrets are templated in `base/secrets.yaml` with placeholder values. For actual deployments:

**Option 1: SealedSecrets** (Recommended for GitOps)

```bash
# Install sealed-secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/controller.yaml

# Encrypt a secret
echo -n 'actual-password' | kubectl create secret generic \
  negative-space-secrets --dry-run=client \
  --from-file=DB_PASSWORD=/dev/stdin -o yaml | \
  kubeseal -f - > overlays/prod/secrets-sealed.yaml

# Apply sealed secret
kubectl apply -f overlays/prod/secrets-sealed.yaml
```

**Option 2: ExternalSecrets Operator**

```bash
# Install external-secrets
helm repo add external-secrets https://external-secrets.github.io/charts
helm install external-secrets external-secrets/external-secrets -n external-secrets-system

# Create ExternalSecret referencing AWS Secrets Manager, Vault, etc.
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: negative-space-secrets
  namespace: negative-space
spec:
  secretStoreRef:
    name: aws-secrets
    kind: SecretStore
  target:
    name: negative-space-secrets
    creationPolicy: Owner
  data:
  - secretKey: DB_PASSWORD
    remoteRef:
      key: negative-space/db/password
```

**Option 3: Manual Secret Creation**

```bash
# For non-GitOps environments
kubectl create secret generic negative-space-secrets \
  -n negative-space \
  --from-literal=DB_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=API_SECRET_KEY=$(openssl rand -base64 32) \
  --from-literal=JWT_SECRET=$(openssl rand -base64 32)
```

### ConfigMap Customization

ConfigMaps are customized per environment in kustomization.yaml files:

```yaml
# overlays/prod/kustomization.yaml
configMapGenerator:
- name: negative-space-config
  behavior: replace
  literals:
  - LOG_LEVEL=WARN
  - ENVIRONMENT=prod
  - METRICS_ENABLED=true
  - ENABLE_TRACING=true
```

### Image Tag Management

Update image tags in overlay kustomization files:

```yaml
# overlays/prod/kustomization.yaml
images:
- name: negative-space-api
  newTag: v1.0.0           # Production release tag
- name: negative-space-ml-processor
  newTag: v1.0.0
- name: negative-space-web-viewer
  newTag: v1.0.0
```

## Network Policies (Production)

Production environment enforces strict network policies:

- **API Container**: Accepts traffic from web-viewer, ingress-nginx
- **ML Processor**: Accepts traffic from API, connects to databases
- **Web Viewer**: Accepts traffic from ingress-nginx, connects to API
- **Egress**: Limited to required services and database connections

## Production Readiness Checklist

- [ ] All images tagged with release versions (not :latest)
- [ ] Database credentials stored in sealed secrets or external secret store
- [ ] TLS certificates configured (cert-manager, letsencrypt-prod)
- [ ] Network policies validated
- [ ] Pod Disruption Budgets tested
- [ ] HPA metrics and thresholds validated
- [ ] Load testing performed
- [ ] Logging aggregation configured
- [ ] Monitoring and alerting in place
- [ ] Backup and disaster recovery tested
- [ ] Security scan results reviewed
- [ ] Compliance requirements verified

## Monitoring and Troubleshooting

### View Resource Usage

```bash
kubectl top nodes -n negative-space
kubectl top pods -n negative-space
```

### Check HPA Status

```bash
kubectl get hpa -n negative-space -w
kubectl describe hpa negative-space-api-hpa -n negative-space
```

### Inspect Network Policies

```bash
kubectl get networkpolicies -n negative-space
kubectl describe networkpolicy negative-space-api-netpol -n negative-space
```

### View Pod Logs

```bash
# API logs
kubectl logs -n negative-space -l app.kubernetes.io/component=api -f

# ML Processor logs
kubectl logs -n negative-space -l app.kubernetes.io/component=ml-processor -f

# Web Viewer logs
kubectl logs -n negative-space -l app.kubernetes.io/component=web-viewer -f
```

### Verify Affinity (Production)

```bash
# Check node distribution (should be spread across nodes)
kubectl get pods -n negative-space -o wide
```

## Common Tasks

### Scale Services Manually

```bash
# Override HPA for manual scaling
kubectl patch deployment negative-space-api \
  -p '{"spec":{"replicas":5}}' \
  -n negative-space
```

### Perform Rolling Update

```bash
# Update image tag
kubectl set image deployment/negative-space-api \
  api=negative-space-api:v1.0.1 \
  -n negative-space

# Monitor rollout
kubectl rollout status deployment/negative-space-api -n negative-space
```

### Roll Back Deployment

```bash
kubectl rollout undo deployment/negative-space-api -n negative-space
kubectl rollout history deployment/negative-space-api -n negative-space
```

### Drain Node (Production Maintenance)

```bash
# Safely drain node with PDB respect
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Re-enable node
kubectl uncordon <node-name>
```

## Best Practices

1. **Image Management**: Use specific version tags, never :latest in production
2. **Resource Requests**: Always set requests/limits to enable proper scheduling
3. **HealthChecks**: Liveness and readiness probes for reliability
4. **Security**: Run as non-root, use read-only filesystems, drop ALL capabilities
5. **Logging**: Structured JSON logging, appropriate log levels per environment
6. **Secrets**: Use sealed-secrets or external-secrets, never commit plaintext secrets
7. **Networking**: Production uses network policies to restrict traffic
8. **High Availability**: Production uses pod anti-affinity and disruption budgets
9. **Monitoring**: Prometheus metrics, distributed tracing in production
10. **Testing**: Always run dry-run before applying to production

## Troubleshooting Common Issues

### Pod Cannot Schedule

```bash
# Check events and node availability
kubectl describe pod <pod-name> -n negative-space
kubectl get nodes -o wide
kubectl top nodes
```

**Solutions:**
- Check resource requests match node capacity
- Verify node selectors and affinity rules
- Check for tainted nodes in production

### HPA Not Scaling

```bash
# Check HPA status
kubectl get hpa negative-space-api-hpa -n negative-space
kubectl describe hpa negative-space-api-hpa -n negative-space

# Check metrics server
kubectl get deployment metrics-server -n kube-system
```

**Solutions:**
- Ensure metrics-server is deployed
- Wait for metrics to be collected (30+ seconds)
- Check CPU/memory requests are set
- Verify HPA min/maxReplicas settings

### Network Policy Blocking Traffic

```bash
# Check network policies
kubectl get networkpolicies -n negative-space
kubectl describe networkpolicy negative-space-api-netpol -n negative-space

# Test connectivity (from pod)
kubectl exec -it <pod-name> -n negative-space -- \
  wget -O- http://negative-space-api.negative-space.svc.cluster.local:8000/health/ready
```

**Solutions:**
- Review ingress/egress rules
- Ensure correct label selectors
- Check namespace labels for cross-namespace traffic
- Validate port numbers

## References

- [Kustomize Documentation](https://kustomize.io/)
- [Kubernetes Manifests Best Practices](https://kubernetes.io/docs/concepts/services-networking/)
- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [Pod Disruption Budgets](https://kubernetes.io/docs/tasks/run-application/configure-pdb/)

## Support and Contribution

For issues, questions, or contributions, please refer to the main project documentation.

---

**Last Updated:** December 2024
**Kustomize Version:** v4.0+
**Kubernetes Version:** 1.20+
