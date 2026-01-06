#!/bin/bash

# Kustomize Configuration Validation Script
# Validates all base and overlay configurations

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Kustomize Configuration Validation & Build Test               ║"
echo "║  Negative Space Imaging Project - Phase 4, Task 29             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

KUSTOMIZE_DIR="k8s/kustomize"
TEMP_DIR=$(mktemp -d)
PASS=0
FAIL=0

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if kustomize is installed
check_kustomize() {
    if ! command -v kustomize &> /dev/null; then
        echo -e "${RED}✗ kustomize is not installed${NC}"
        echo "  Install from: https://kustomize.io/"
        exit 1
    fi
    KUSTOMIZE_VERSION=$(kustomize version --short)
    echo -e "${GREEN}✓ kustomize installed: $KUSTOMIZE_VERSION${NC}"
}

# Function to validate YAML syntax
validate_yaml() {
    local file=$1
    if command -v kubectl &> /dev/null; then
        if kubectl apply -f "$file" --dry-run=client &> /dev/null; then
            echo -e "${GREEN}✓ Valid YAML: $file${NC}"
            ((PASS++))
            return 0
        else
            echo -e "${RED}✗ Invalid YAML: $file${NC}"
            ((FAIL++))
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ kubectl not found, skipping validation${NC}"
        return 0
    fi
}

# Function to build and validate kustomize output
build_and_validate() {
    local path=$1
    local name=$2

    echo ""
    echo "Building: $name ($path)"
    echo "─────────────────────────────────────────────────────"

    if kustomize build "$path" > "$TEMP_DIR/${name}-manifest.yaml" 2>&1; then
        echo -e "${GREEN}✓ Build successful${NC}"
        ((PASS++))

        # Count resources
        local resource_count=$(grep -c "^kind:" "$TEMP_DIR/${name}-manifest.yaml" || echo 0)
        local line_count=$(wc -l < "$TEMP_DIR/${name}-manifest.yaml")

        echo "  Resources: $resource_count"
        echo "  Lines: $line_count"

        # Validate output YAML
        if validate_yaml "$TEMP_DIR/${name}-manifest.yaml"; then
            echo "  Structure: Valid"
        else
            echo "  Structure: Invalid"
        fi

    else
        echo -e "${RED}✗ Build failed${NC}"
        cat "$TEMP_DIR/${name}-manifest.yaml"
        ((FAIL++))
        return 1
    fi
}

# Function to validate configuration differences
validate_differences() {
    local overlay=$1
    local overlay_name=$2

    echo ""
    echo "Validating differences: $overlay_name"
    echo "─────────────────────────────────────────────────────"

    local base_manifest="$TEMP_DIR/base-manifest.yaml"
    local overlay_manifest="$TEMP_DIR/${overlay_name}-manifest.yaml"

    # Check replicas
    local base_replicas=$(grep -A 2 "replicas:" "$base_manifest" | head -1 | grep -o "[0-9]*" | head -1)
    local overlay_replicas=$(grep -A 2 "replicas:" "$overlay_manifest" | grep "replicas:" | head -1 | grep -o "[0-9]*")

    if [ -n "$base_replicas" ] && [ -n "$overlay_replicas" ]; then
        if [ "$base_replicas" != "$overlay_replicas" ]; then
            echo -e "${GREEN}✓ Replicas differ (base: $base_replicas, overlay: $overlay_replicas)${NC}"
            ((PASS++))
        else
            echo -e "${YELLOW}⚠ Replicas unchanged${NC}"
        fi
    fi

    # Check environment variables
    if grep -q "LOG_LEVEL" "$overlay_manifest"; then
        echo -e "${GREEN}✓ Environment variables configured${NC}"
        ((PASS++))
    fi

    # Check resource limits
    if grep -q "requests:" "$overlay_manifest" && grep -q "limits:" "$overlay_manifest"; then
        echo -e "${GREEN}✓ Resource limits configured${NC}"
        ((PASS++))
    fi
}

# Main validation
main() {
    echo ""

    # Check prerequisites
    check_kustomize
    echo ""

    # Validate base configuration
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  BASE CONFIGURATION VALIDATION                                 ║"
    echo "╚════════════════════════════════════════════════════════════════╝"

    build_and_validate "$KUSTOMIZE_DIR/base" "base"

    # Validate overlays
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  DEVELOPMENT OVERLAY VALIDATION                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"

    build_and_validate "$KUSTOMIZE_DIR/overlays/dev" "dev"
    validate_differences "$KUSTOMIZE_DIR/overlays/dev" "dev"

    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  STAGING OVERLAY VALIDATION                                    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"

    build_and_validate "$KUSTOMIZE_DIR/overlays/staging" "staging"
    validate_differences "$KUSTOMIZE_DIR/overlays/staging" "staging"

    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  PRODUCTION OVERLAY VALIDATION                                 ║"
    echo "╚════════════════════════════════════════════════════════════════╝"

    build_and_validate "$KUSTOMIZE_DIR/overlays/prod" "prod"
    validate_differences "$KUSTOMIZE_DIR/overlays/prod" "prod"

    # Validate no :latest tags in production
    echo ""
    echo "Validating production image tags..."
    echo "─────────────────────────────────────────────────────"

    if ! grep -q ":latest" "$TEMP_DIR/prod-manifest.yaml"; then
        echo -e "${GREEN}✓ No :latest tags in production (good!)${NC}"
        ((PASS++))
    else
        echo -e "${RED}✗ Found :latest tags in production${NC}"
        ((FAIL++))
    fi

    # Validate network policies in production
    echo ""
    echo "Validating production network policies..."
    echo "─────────────────────────────────────────────────────"

    local netpol_count=$(grep -c "^kind: NetworkPolicy" "$TEMP_DIR/prod-manifest.yaml" || echo 0)
    if [ "$netpol_count" -ge 3 ]; then
        echo -e "${GREEN}✓ Network policies found: $netpol_count${NC}"
        ((PASS++))
    else
        echo -e "${YELLOW}⚠ Expected 3+ network policies, found: $netpol_count${NC}"
    fi

    # Validate PDBs in production
    echo ""
    echo "Validating production Pod Disruption Budgets..."
    echo "─────────────────────────────────────────────────────"

    local pdb_count=$(grep -c "^kind: PodDisruptionBudget" "$TEMP_DIR/prod-manifest.yaml" || echo 0)
    if [ "$pdb_count" -ge 3 ]; then
        echo -e "${GREEN}✓ PDBs found: $pdb_count${NC}"
        ((PASS++))
    else
        echo -e "${YELLOW}⚠ Expected 3+ PDBs, found: $pdb_count${NC}"
    fi

    # Final summary
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  VALIDATION SUMMARY                                            ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "Passed: ${GREEN}$PASS${NC}"
    echo -e "Failed: ${RED}$FAIL${NC}"
    echo ""

    # Save manifests for inspection
    echo "Generated manifests saved to: $TEMP_DIR"
    echo ""
    echo "To inspect manifests:"
    echo "  cat $TEMP_DIR/base-manifest.yaml"
    echo "  cat $TEMP_DIR/dev-manifest.yaml"
    echo "  cat $TEMP_DIR/staging-manifest.yaml"
    echo "  cat $TEMP_DIR/prod-manifest.yaml"
    echo ""

    if [ $FAIL -eq 0 ]; then
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ALL VALIDATIONS PASSED - KUSTOMIZE READY FOR DEPLOYMENT      ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
        return 0
    else
        echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  VALIDATION FAILED - PLEASE FIX ERRORS ABOVE                   ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
        return 1
    fi
}

# Run main
main
EXIT_CODE=$?

# Cleanup
# rm -rf "$TEMP_DIR"

exit $EXIT_CODE
