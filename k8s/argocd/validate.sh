#!/bin/bash

# ArgoCD Installation Validation Script
# Validates all YAML manifests and checks deployment readiness

set -e

echo "═════════════════════════════════════════════════════════════"
echo "  ArgoCD Installation Validation Script"
echo "═════════════════════════════════════════════════════════════"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0
SUCCESS=0

# Function to validate YAML file
validate_yaml() {
    local file=$1
    echo -n "Validating $file ... "

    if kubectl apply -f "$file" --dry-run=client -o yaml > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        ((SUCCESS++))
    else
        echo -e "${RED}✗${NC}"
        echo "  Error in $file:"
        kubectl apply -f "$file" --dry-run=client -o yaml 2>&1 | head -5
        ((ERRORS++))
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo -e "${BLUE}[1] Checking Prerequisites${NC}"
echo "───────────────────────────────────────────────────────────"

if command_exists kubectl; then
    echo -e "${GREEN}✓${NC} kubectl found"
    kubectl version --client --short
else
    echo -e "${RED}✗${NC} kubectl not found"
    ((ERRORS++))
fi

if command_exists kustomize; then
    echo -e "${GREEN}✓${NC} kustomize found"
else
    echo -e "${YELLOW}⚠${NC} kustomize not found (optional for advanced usage)"
    ((WARNINGS++))
fi

echo ""
echo -e "${BLUE}[2] Validating YAML Manifests${NC}"
echo "───────────────────────────────────────────────────────────"

# Find all YAML files in k8s/argocd
YAML_FILES=(
    "k8s/argocd/argocd-namespace.yaml"
    "k8s/argocd/argocd-config.yaml"
    "k8s/argocd/argocd-install.yaml"
    "k8s/argocd/argocd-ingress.yaml"
    "k8s/argocd/argocd-rbac.yaml"
    "k8s/argocd/projects/projects.yaml"
    "k8s/argocd/applications/negative-space-api-dev.yaml"
    "k8s/argocd/applications/negative-space-api-staging.yaml"
    "k8s/argocd/applications/negative-space-api-prod.yaml"
    "k8s/argocd/sync-waves/waves.yaml"
    "k8s/argocd/webhooks/webhooks.yaml"
    "k8s/argocd/rbac/rbac-config.yaml"
    "k8s/argocd/notifications/notifications-config.yaml"
)

for yaml_file in "${YAML_FILES[@]}"; do
    if [ -f "$yaml_file" ]; then
        validate_yaml "$yaml_file"
    else
        echo -e "${YELLOW}⚠${NC} File not found: $yaml_file"
        ((WARNINGS++))
    fi
done

echo ""
echo -e "${BLUE}[3] Checking Kubernetes Cluster Access${NC}"
echo "───────────────────────────────────────────────────────────"

if kubectl cluster-info > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Connected to Kubernetes cluster"
    kubectl cluster-info | head -2
else
    echo -e "${RED}✗${NC} Cannot connect to Kubernetes cluster"
    echo "  Make sure kubectl is configured correctly"
    ((ERRORS++))
fi

echo ""
echo -e "${BLUE}[4] Checking Required Namespaces${NC}"
echo "───────────────────────────────────────────────────────────"

NAMESPACES=("ingress-nginx" "cert-manager" "argocd")

for ns in "${NAMESPACES[@]}"; do
    if kubectl get namespace "$ns" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Namespace exists: $ns"
    else
        echo -e "${YELLOW}⚠${NC} Namespace not found: $ns (will be created by manifests)"
        ((WARNINGS++))
    fi
done

echo ""
echo -e "${BLUE}[5] Checking Storage Classes${NC}"
echo "───────────────────────────────────────────────────────────"

STORAGE_CLASSES=("fast-ssd" "prod-ssd" "backup-archive" "default")

for sc in "${STORAGE_CLASSES[@]}"; do
    if kubectl get storageclass "$sc" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Storage class exists: $sc"
    else
        echo -e "${YELLOW}⚠${NC} Storage class not found: $sc"
        ((WARNINGS++))
    fi
done

echo ""
echo -e "${BLUE}[6] Configuration Checks${NC}"
echo "───────────────────────────────────────────────────────────"

# Check if GitHub settings are configured
ARGOCD_CONFIG="k8s/argocd/argocd-config.yaml"
if grep -q "yourusername" "$ARGOCD_CONFIG"; then
    echo -e "${YELLOW}⚠${NC} GitHub username placeholder not replaced"
    echo "  Update 'yourusername' in argocd-config.yaml"
    ((WARNINGS++))
else
    echo -e "${GREEN}✓${NC} GitHub repository configured"
fi

if grep -q "example.com" "$ARGOCD_CONFIG"; then
    echo -e "${YELLOW}⚠${NC} Domain placeholders not replaced"
    echo "  Update 'example.com' with your actual domain"
    ((WARNINGS++))
else
    echo -e "${GREEN}✓${NC} Domain configured"
fi

echo ""
echo -e "${BLUE}[7] Security Validation${NC}"
echo "───────────────────────────────────────────────────────────"

# Check RBAC is defined
if grep -q "role:admin" "k8s/argocd/rbac/rbac-config.yaml"; then
    echo -e "${GREEN}✓${NC} RBAC roles defined"
else
    echo -e "${RED}✗${NC} RBAC roles not found"
    ((ERRORS++))
fi

# Check network policies exist
if [ -f "k8s/kustomize/overlays/prod/network-policy.yaml" ]; then
    echo -e "${GREEN}✓${NC} Network policies configured"
else
    echo -e "${YELLOW}⚠${NC} Network policies not found"
    ((WARNINGS++))
fi

echo ""
echo "═════════════════════════════════════════════════════════════"
echo "  VALIDATION SUMMARY"
echo "═════════════════════════════════════════════════════════════"

echo -e "${GREEN}✓ Successful: $SUCCESS${NC}"
echo -e "${YELLOW}⚠ Warnings: $WARNINGS${NC}"
echo -e "${RED}✗ Errors: $ERRORS${NC}"

echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All validations passed! Ready to deploy.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Update placeholders in k8s/argocd/argocd-config.yaml"
    echo "2. Run: kubectl apply -k k8s/argocd/"
    echo "3. Wait for ArgoCD pods to be ready: kubectl -n argocd get pods"
    echo "4. Access ArgoCD: kubectl port-forward -n argocd svc/argocd-server 8080:80"
    exit 0
else
    echo -e "${RED}Validation failed! Fix the errors above before deploying.${NC}"
    exit 1
fi
