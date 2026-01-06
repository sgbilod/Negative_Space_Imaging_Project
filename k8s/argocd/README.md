# ArgoCD GitOps Implementation Guide

## Overview

This directory contains a complete ArgoCD installation and configuration for GitOps-based continuous deployment across development, staging, and production environments.

**Key Features:**
- ✅ Multi-environment support (dev/staging/prod)
- ✅ Automatic sync for dev/staging, manual approval for prod
- ✅ Comprehensive RBAC and access control
- ✅ Webhook integration with GitHub
- ✅ Deployment sync waves and ordering
- ✅ Pre/post-sync hooks (backup, health checks, rollback)
- ✅ Slack/Email notifications
- ✅ Production-grade monitoring and alerts

---

## 📁 Directory Structure

```
k8s/argocd/
├── argocd-namespace.yaml          # Namespace, ServiceAccounts, RBAC
├── argocd-install.yaml            # Core deployments (server, controller, repo-server, dex, notifications)
├── argocd-config.yaml             # ConfigMaps with ArgoCD settings
├── argocd-ingress.yaml            # Ingress for UI and Dex (HTTPS)
├── argocd-rbac.yaml               # Additional RBAC configuration
│
├── applications/
│   ├── negative-space-api-dev.yaml      # Dev environment Application CRD
│   ├── negative-space-api-staging.yaml  # Staging environment Application CRD
│   └── negative-space-api-prod.yaml     # Production environment Application CRD (manual sync)
│
├── projects/
│   └── projects.yaml              # AppProject definitions (dev, staging, prod)
│
├── sync-waves/
│   └── waves.yaml                 # Deployment ordering and hooks
│
├── webhooks/
│   └── webhooks.yaml              # GitHub webhook integration
│
├── rbac/
│   └── rbac-config.yaml           # RBAC policies and service accounts
│
├── notifications/
│   └── notifications-config.yaml  # Slack/Email notification setup
│
└── README.md                      # This file
```

---

## 🚀 Installation

### Prerequisites

- Kubernetes cluster (1.24+)
- `kubectl` configured
- NGINX Ingress Controller
- cert-manager (for TLS certificates)
- GitHub account with OAuth app (optional)

### Step 1: Install cert-manager

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### Step 2: Install NGINX Ingress Controller

```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace
```

### Step 3: Create Let's Encrypt ClusterIssuer

```bash
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Step 4: Install ArgoCD

```bash
# Apply all manifests in order
kubectl apply -f k8s/argocd/argocd-namespace.yaml
kubectl apply -f k8s/argocd/argocd-install.yaml
kubectl apply -f k8s/argocd/argocd-config.yaml
kubectl apply -f k8s/argocd/argocd-rbac.yaml
kubectl apply -f k8s/argocd/argocd-ingress.yaml

# Create target namespaces
kubectl create namespace negative-space-dev
kubectl create namespace negative-space-staging
kubectl create namespace negative-space-prod

# Label namespaces for network policies
kubectl label namespace ingress-nginx name=ingress-nginx
```

### Step 5: Install AppProjects and Applications

```bash
# Deploy projects
kubectl apply -f k8s/argocd/projects/projects.yaml

# Deploy applications
kubectl apply -f k8s/argocd/applications/negative-space-api-dev.yaml
kubectl apply -f k8s/argocd/applications/negative-space-api-staging.yaml
kubectl apply -f k8s/argocd/applications/negative-space-api-prod.yaml

# Deploy sync waves and webhooks
kubectl apply -f k8s/argocd/sync-waves/waves.yaml
kubectl apply -f k8s/argocd/webhooks/webhooks.yaml
kubectl apply -f k8s/argocd/rbac/rbac-config.yaml
kubectl apply -f k8s/argocd/notifications/notifications-config.yaml
```

### Step 6: Verify Installation

```bash
# Check all pods are running
kubectl -n argocd get pods

# Check ingress is ready
kubectl -n argocd get ingress
```

---

## 🔐 Initial Configuration

### 1. Reset Admin Password

```bash
# Port forward to ArgoCD server
kubectl port-forward -n argocd svc/argocd-server 8080:80 &

# Reset password
kubectl -n argocd exec -it deployment/argocd-server -- argocd admin initial-password

# Navigate to http://localhost:8080
# Login with username: admin and password from above
```

### 2. Update ArgoCD Settings

Edit `argocd-config.yaml` and update:

- **Repository URL:** Replace `https://github.com/yourusername/Negative_Space_Imaging_Project`
- **OAuth Client ID/Secret:** GitHub OAuth app credentials
- **Ingress hosts:** Replace `argocd.example.com`, `dex.example.com`, `webhook.example.com`
- **Slack webhook:** Update Slack channel webhook URL
- **Email SMTP:** Update email server settings

### 3. Create GitHub OAuth Application

1. Go to GitHub Settings → Developer settings → OAuth Apps
2. Create new OAuth App with:
   - **Authorization callback URL:** `https://argocd.example.com/api/dex/callback`
3. Copy Client ID and Client Secret
4. Update `argocd-config.yaml` with credentials

### 4. Configure GitHub Webhook

1. Go to your repository Settings → Webhooks
2. Add webhook with:
   - **Payload URL:** `https://webhook.example.com/api/webhook`
   - **Content type:** `application/json`
   - **Events:** Push events, Pull requests
   - **Secret:** Match `github-webhook-secret` in `webhooks.yaml`

### 5. Setup Slack Integration

1. Create Slack App and get webhook URL
2. Update secret in `notifications/notifications-config.yaml`:

```bash
kubectl -n argocd create secret generic argocd-notifications-slack \
  --from-literal=slack-token='https://hooks.slack.com/services/T.../B.../X...' \
  --dry-run=client -o yaml | kubectl apply -f -
```

---

## 📊 Deployment Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Git Repository                            │
│           (Kustomize overlays in k8s/kustomize/)            │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  GitHub Webhook Push                         │
│                (on main/staging/develop)                    │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           ArgoCD Webhook Receiver (Pod)                     │
│     Triggers sync of corresponding Application              │
└─────────────────────────────────────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │                                        │
        ▼                                        ▼
┌───────────────────┐              ┌────────────────────┐
│  Auto-Sync (Dev)  │              │  Manual Sync (Prod)│
│  Auto-Sync (Staging)             │  Requires Approval │
└───────────────────┘              └────────────────────┘
        │                                        │
        ▼                                        ▼
┌──────────────────────────────────────────────────────────────┐
│              Application Controller                           │
│  1. Fetch desired state from Git                             │
│  2. Compare with cluster state                               │
│  3. Execute sync waves (if configured)                       │
└──────────────────────────────────────────────────────────────┘
        │
        ├─ Wave 0: Infrastructure (RBAC, namespaces)
        │
        ├─ Wave 1: Data Layer (DBs, Secrets, ConfigMaps)
        │     • Pre-sync hooks: Backup database
        │     • Health checks: Verify DB connectivity
        │
        ├─ Wave 2: Core Services (API, ML Processor)
        │
        ├─ Wave 3: Frontend Services (Web UI, Gateway)
        │
        └─ Wave 4: Observability (Prometheus, Grafana)
              • Post-sync hooks: Run smoke tests
              • Verify deployment: Health checks
              • Failure hooks: Automatic rollback
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                   Deployment Complete                        │
│  Notifications sent to Slack/Email/GitHub                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔄 Sync Policies

### Development Environment
- **Sync Policy:** Automatic
- **Prune:** Enabled (delete resources not in Git)
- **Self-Heal:** Enabled (reconcile drift)
- **Approval:** Not required
- **Use Case:** Rapid iteration, testing

### Staging Environment
- **Sync Policy:** Automatic
- **Prune:** Enabled
- **Self-Heal:** Enabled
- **Approval:** Not required (but use sync waves)
- **Use Case:** Pre-production testing, validation

### Production Environment
- **Sync Policy:** Manual
- **Prune:** Not automatic (requires confirmation)
- **Self-Heal:** Not enabled (prevent surprises)
- **Approval:** Required before sync
- **Rollback:** Automatic if health checks fail
- **Use Case:** Controlled, audited deployments

---

## ✅ RBAC Security Model

### Roles

| Role | Permissions | Environments |
|------|-------------|--------------|
| `admin` | Full access to all resources | All |
| `deployer` | Can sync applications | All |
| `dev-admin` | Can fully manage dev apps | Dev |
| `staging-deployer` | Can sync staging apps | Staging |
| `staging-reviewer` | Read-only access to staging | Staging |
| `prod-admin` | Full prod access (manual sync only) | Prod |
| `prod-viewer` | Read-only access to prod | Prod |
| `readonly` | Read-only access | All |

### Team Assignments

```csv
platform-engineers → role:admin
dev-team → role:dev-admin
qa-team → role:staging-reviewer
devops-team → role:deployer
sre-team → role:prod-admin
developers → role:readonly
```

### OIDC/GitHub Team Integration

ArgoCD syncs GitHub team memberships via OIDC. Teams are automatically mapped to roles:

```bash
# GitHub team: org:platform-engineers
# ArgoCD role: role:admin
```

---

## 🔔 Notifications

### Slack Notifications

Configured events:
- ✅ **Sync Succeeded** → `#deployments` channel
- ❌ **Sync Failed** → `#alerts` channel
- ⚠️ **Health Degraded** → `#alerts` channel
- ⏳ **Sync Running** → `#deployments` channel

### Email Notifications

Recipients per environment:
- **Production failures:** ops-team@, devops@, on-call@
- **Staging failures:** ops-team@, devops@
- **Development:** devops@ only

### GitHub Status Updates

Commit status is updated on GitHub:
- ✅ Successful sync → Green checkmark
- ❌ Failed sync → Red X
- ⏳ Running sync → Yellow circle

---

## 🔗 Webhook Integration

### GitHub Webhook Configuration

**Payload URL:** `https://webhook.example.com/api/webhook`

**Events triggered by:**
- Push to `main` → Sync `prod` (manual)
- Push to `staging` → Sync `staging` (auto)
- Push to `develop` → Sync `dev` (auto)

**Branch-specific actions:**

```yaml
main:
  app: negative-space-api-prod
  auto_sync: false
  notify: true

staging:
  app: negative-space-api-staging
  auto_sync: true
  notify: true

develop:
  app: negative-space-api-dev
  auto_sync: true
  notify: true
```

---

## 🔄 Manual Sync Procedure (Production)

### Via ArgoCD UI

1. Navigate to `https://argocd.example.com`
2. Click on `negative-space-api-prod` application
3. Review changes in Git vs cluster
4. Click **SYNC** button
5. Confirm deployment
6. Monitor sync progress
7. Verify health checks

### Via ArgoCD CLI

```bash
# Login to ArgoCD
argocd login argocd.example.com

# View application status
argocd app get negative-space-api-prod

# Sync application (manual approval required)
argocd app sync negative-space-api-prod

# Wait for sync to complete
argocd app wait negative-space-api-prod
```

### Via kubectl

```bash
# Trigger sync by updating Application CRD
kubectl -n argocd patch application negative-space-api-prod \
  --type merge \
  --patch '{"operation":"Sync"}'
```

---

## 🔙 Rollback Procedures

### Automatic Rollback (Production)

Pre-configured for:
- ❌ Application health degradation
- 🔄 Deployment replicas not ready after 10min
- 💥 Container OOMKilled
- 🌀 CrashLoopBackOff detected

```bash
# View rollback history
kubectl -n argocd logs -f deployment/argocd-application-controller | grep rollback
```

### Manual Rollback

```bash
# Via ArgoCD UI
# Click app → HISTORY → Select previous revision → ROLLBACK

# Via CLI
argocd app rollback negative-space-api-prod <revision>

# Via Git (GitOps way - revert commit and push)
git revert <commit-hash>
git push
# ArgoCD will sync to previous state automatically
```

---

## 📊 Pre-Sync Hooks

### Database Backup

Runs before deployment to production:

```bash
kubectl -n negative-space-prod logs job/pre-sync-backup
```

### Health Check

Verifies current system is healthy before sync:

```bash
# Checks: /health endpoint returns 200
kubectl -n negative-space-prod logs job/pre-sync-health-check
```

---

## ✔️ Post-Sync Verification

### Health Checks

```bash
# 1. Deployment readiness (all replicas ready)
kubectl -n negative-space-prod rollout status deployment/negative-space-api

# 2. Service endpoints
kubectl -n negative-space-prod get endpoints

# 3. Pod logs for errors
kubectl -n negative-space-prod logs -f deployment/negative-space-api
```

### Smoke Tests

```bash
# Test API health endpoint
curl https://api.example.com/health

# Test API version
curl https://api.example.com/api/v1/version

# Test database connectivity (from pod)
kubectl -n negative-space-prod exec -it deployment/negative-space-api -- \
  curl postgresql:5432/psql
```

---

## 🔍 Troubleshooting

### Application Not Syncing

```bash
# Check application status
kubectl -n argocd get application negative-space-api-prod -o yaml

# View sync status details
argocd app describe negative-space-api-prod

# Check controller logs
kubectl -n argocd logs deployment/argocd-application-controller

# Check repository access
kubectl -n argocd exec deployment/argocd-repo-server -- \
  git ls-remote https://github.com/yourusername/Negative_Space_Imaging_Project
```

### Webhook Not Triggering

```bash
# Check webhook service
kubectl -n argocd get service argocd-webhook

# Test webhook endpoint
curl -X POST https://webhook.example.com/api/webhook \
  -H "Content-Type: application/json" \
  -d '{"repository":"test"}'

# Check webhook logs
kubectl -n argocd logs deployment/argocd-server | grep webhook
```

### RBAC Access Denied

```bash
# Check current user permissions
argocd account can-i get applications '*/*'

# View user's roles
kubectl -n argocd get configmap argocd-rbac-cm -o jsonpath='{.data.policy\.csv}'

# Check service account permissions
kubectl auth can-i create deployments --as=system:serviceaccount:argocd:argocd-cicd-sa -n negative-space-dev
```

---

## 📈 Monitoring & Observability

### ArgoCD Metrics

Exposed on port 8082/8083/8084/8085 (Prometheus format):

```bash
# Application controller metrics
curl http://localhost:8085/metrics | grep argocd

# Key metrics to monitor
- argocd_app_sync_total (total syncs)
- argocd_app_sync_duration_seconds (sync latency)
- argocd_app_reconcile_bucket (reconciliation duration)
```

### Setting Up Prometheus Scraping

```yaml
- job_name: 'argocd-app-controller'
  kubernetes_sd_configs:
  - role: pod
    namespaces:
      names:
      - argocd
  relabel_configs:
  - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_name]
    action: keep
    regex: argocd-application-controller
```

---

## 🔐 Production Readiness Checklist

- [ ] All manifests validated with `kubectl apply --dry-run=client`
- [ ] Ingress DNS records configured (A records pointing to LB)
- [ ] TLS certificates from cert-manager working
- [ ] GitHub OAuth app created and configured
- [ ] GitHub webhook secret configured
- [ ] Slack integration tested
- [ ] Database backups verified
- [ ] RBAC policies tested
- [ ] Network policies applied
- [ ] Pod disruption budgets configured
- [ ] Resource quotas set
- [ ] Monitoring/alerting enabled
- [ ] Disaster recovery plan documented
- [ ] On-call runbook created
- [ ] Team trained on ArgoCD operations

---

## 📚 Additional Resources

- [ArgoCD Official Docs](https://argoproj.github.io/argo-cd/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/overview/working-with-objects/kubernetes-objects/)
- [GitOps Guide](https://www.weave.works/technologies/gitops/)
- [Kustomize Documentation](https://kustomize.io/)

---

**Last Updated:** December 2024
**Maintained By:** Platform Engineering Team
**Contact:** devops@example.com
