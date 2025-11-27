#!/bin/bash
# =============================================================================
# Negative Space Imaging Project - Edge Deployment Script
# Automated deployment for Raspberry Pi 4/5 (ARM64)
# =============================================================================
#
# This script handles complete edge node deployment:
# - System prerequisite checks
# - Docker and Docker Compose installation
# - TLS certificate generation
# - Container image building/pulling
# - Service deployment and verification
#
# Usage:
#   ./deploy.sh [OPTIONS]
#
# Options:
#   --build       Build images locally (instead of pulling)
#   --dev         Deploy in development mode
#   --no-tls      Skip TLS certificate generation
#   --clean       Remove existing deployment first
#   --verify-only Run verification without deploying
#
# Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${SCRIPT_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
BUILD_IMAGES=false
DEV_MODE=false
GENERATE_TLS=true
CLEAN_FIRST=false
VERIFY_ONLY=false
NODE_ID="${NSI_NODE_ID:-edge-node-$(hostname -s)}"

# Minimum requirements
MIN_MEMORY_MB=2048
MIN_DISK_GB=10
MIN_DOCKER_VERSION="20.10.0"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
    echo ""
}

version_compare() {
    # Compare two version strings
    # Returns 0 if $1 >= $2, 1 otherwise
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                BUILD_IMAGES=true
                shift
                ;;
            --dev)
                DEV_MODE=true
                shift
                ;;
            --no-tls)
                GENERATE_TLS=false
                shift
                ;;
            --clean)
                CLEAN_FIRST=true
                shift
                ;;
            --verify-only)
                VERIFY_ONLY=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
Negative Space Imaging - Edge Deployment Script

Usage: ./deploy.sh [OPTIONS]

Options:
    --build       Build images locally (instead of pulling)
    --dev         Deploy in development mode
    --no-tls      Skip TLS certificate generation
    --clean       Remove existing deployment first
    --verify-only Run verification without deploying
    -h, --help    Show this help message

Environment Variables:
    NSI_NODE_ID       Node identifier (default: edge-node-<hostname>)
    NSI_JURISDICTION  Data sovereignty jurisdiction (default: LOCAL)
    NSI_TLS_ENABLED   Enable TLS (default: true)

Examples:
    ./deploy.sh                    # Standard deployment
    ./deploy.sh --build            # Build images locally
    ./deploy.sh --clean --build    # Clean and rebuild
    ./deploy.sh --verify-only      # Only run verification checks
EOF
}

# =============================================================================
# Prerequisite Checks
# =============================================================================

check_prerequisites() {
    log_section "Checking Prerequisites"

    local checks_passed=true

    # Check architecture
    log_info "Checking system architecture..."
    local arch=$(uname -m)
    if [[ "$arch" != "aarch64" && "$arch" != "arm64" ]]; then
        log_warning "Non-ARM64 architecture detected: $arch"
        log_warning "This deployment is optimized for Raspberry Pi 4/5 (ARM64)"
    else
        log_success "ARM64 architecture detected"
    fi

    # Check memory
    log_info "Checking available memory..."
    local mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local mem_mb=$((mem_kb / 1024))
    if [[ $mem_mb -lt $MIN_MEMORY_MB ]]; then
        log_error "Insufficient memory: ${mem_mb}MB (minimum: ${MIN_MEMORY_MB}MB)"
        checks_passed=false
    else
        log_success "Memory check passed: ${mem_mb}MB available"
    fi

    # Check disk space
    log_info "Checking disk space..."
    local disk_avail=$(df -BG "${PROJECT_ROOT}" | tail -1 | awk '{print $4}' | tr -d 'G')
    if [[ $disk_avail -lt $MIN_DISK_GB ]]; then
        log_error "Insufficient disk space: ${disk_avail}GB (minimum: ${MIN_DISK_GB}GB)"
        checks_passed=false
    else
        log_success "Disk space check passed: ${disk_avail}GB available"
    fi

    # Check Docker
    log_info "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        log_info "Install with: curl -fsSL https://get.docker.com | sh"
        checks_passed=false
    else
        local docker_version=$(docker --version | grep -oP '\d+\.\d+\.\d+' | head -1)
        if version_compare "$docker_version" "$MIN_DOCKER_VERSION"; then
            log_success "Docker version $docker_version installed"
        else
            log_error "Docker version $docker_version is below minimum ($MIN_DOCKER_VERSION)"
            checks_passed=false
        fi
    fi

    # Check Docker Compose
    log_info "Checking Docker Compose..."
    if docker compose version &> /dev/null; then
        local compose_version=$(docker compose version --short)
        log_success "Docker Compose version $compose_version installed"
    elif command -v docker-compose &> /dev/null; then
        local compose_version=$(docker-compose --version | grep -oP '\d+\.\d+\.\d+')
        log_success "Docker Compose (standalone) version $compose_version installed"
    else
        log_error "Docker Compose is not installed"
        checks_passed=false
    fi

    # Check Docker daemon
    log_info "Checking Docker daemon..."
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running or current user lacks permissions"
        log_info "Try: sudo systemctl start docker && sudo usermod -aG docker \$USER"
        checks_passed=false
    else
        log_success "Docker daemon is running"
    fi

    # Check required files
    log_info "Checking required files..."
    local required_files=(
        "${DEPLOYMENT_DIR}/docker-compose.edge.yml"
        "${DEPLOYMENT_DIR}/Dockerfile.arm64"
        "${PROJECT_ROOT}/requirements-edge.txt"
    )

    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            checks_passed=false
        fi
    done

    if [[ -f "${DEPLOYMENT_DIR}/docker-compose.edge.yml" ]]; then
        log_success "All required files present"
    fi

    if [[ "$checks_passed" != true ]]; then
        log_error "Prerequisite checks failed. Please resolve the issues above."
        exit 1
    fi

    log_success "All prerequisite checks passed"
}

# =============================================================================
# TLS Certificate Generation
# =============================================================================

generate_tls_certificates() {
    log_section "Generating TLS Certificates"

    local cert_dir="${DEPLOYMENT_DIR}/config/certs"
    mkdir -p "$cert_dir"

    # Check if certificates already exist
    if [[ -f "${cert_dir}/server.crt" && -f "${cert_dir}/server.key" ]]; then
        log_info "Existing certificates found. Checking validity..."

        # Check certificate expiry
        local expiry=$(openssl x509 -enddate -noout -in "${cert_dir}/server.crt" 2>/dev/null | cut -d= -f2)
        local expiry_epoch=$(date -d "$expiry" +%s 2>/dev/null || echo 0)
        local now_epoch=$(date +%s)
        local days_left=$(( (expiry_epoch - now_epoch) / 86400 ))

        if [[ $days_left -gt 30 ]]; then
            log_success "Existing certificates valid for $days_left more days"
            return 0
        else
            log_warning "Certificates expire in $days_left days. Regenerating..."
        fi
    fi

    log_info "Generating new self-signed certificates..."

    # Generate CA key and certificate
    openssl genrsa -out "${cert_dir}/ca.key" 4096 2>/dev/null
    openssl req -new -x509 -days 3650 -key "${cert_dir}/ca.key" \
        -out "${cert_dir}/ca.crt" \
        -subj "/C=US/ST=Local/L=Local/O=NSI Edge/OU=Edge CA/CN=NSI Edge CA" \
        2>/dev/null

    # Generate server key and CSR
    openssl genrsa -out "${cert_dir}/server.key" 2048 2>/dev/null
    openssl req -new -key "${cert_dir}/server.key" \
        -out "${cert_dir}/server.csr" \
        -subj "/C=US/ST=Local/L=Local/O=NSI Edge/OU=Edge Server/CN=${NODE_ID}" \
        2>/dev/null

    # Create extensions file for SAN
    cat > "${cert_dir}/server.ext" << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = ${NODE_ID}
DNS.3 = nsi-imaging-processor
DNS.4 = nsi-verification
IP.1 = 127.0.0.1
IP.2 = 172.28.0.1
EOF

    # Sign server certificate
    openssl x509 -req -in "${cert_dir}/server.csr" \
        -CA "${cert_dir}/ca.crt" \
        -CAkey "${cert_dir}/ca.key" \
        -CAcreateserial \
        -out "${cert_dir}/server.crt" \
        -days 365 \
        -extfile "${cert_dir}/server.ext" \
        2>/dev/null

    # Set permissions
    chmod 600 "${cert_dir}"/*.key
    chmod 644 "${cert_dir}"/*.crt

    # Clean up temporary files
    rm -f "${cert_dir}/server.csr" "${cert_dir}/server.ext" "${cert_dir}/ca.srl"

    log_success "TLS certificates generated successfully"
    log_info "  CA Certificate: ${cert_dir}/ca.crt"
    log_info "  Server Certificate: ${cert_dir}/server.crt"
    log_info "  Server Key: ${cert_dir}/server.key"
}

# =============================================================================
# Clean Existing Deployment
# =============================================================================

clean_deployment() {
    log_section "Cleaning Existing Deployment"

    log_info "Stopping existing containers..."
    cd "${DEPLOYMENT_DIR}"

    if docker compose -f docker-compose.edge.yml ps -q 2>/dev/null | grep -q .; then
        docker compose -f docker-compose.edge.yml down -v --remove-orphans || true
        log_success "Containers stopped and removed"
    else
        log_info "No running containers found"
    fi

    # Remove dangling images
    log_info "Removing dangling images..."
    docker image prune -f 2>/dev/null || true

    log_success "Cleanup completed"
}

# =============================================================================
# Build Docker Images
# =============================================================================

build_images() {
    log_section "Building Docker Images"

    cd "${PROJECT_ROOT}"

    log_info "Building ARM64 image..."
    docker build \
        -t nsi-edge:latest \
        -f "${DEPLOYMENT_DIR}/Dockerfile.arm64" \
        --build-arg SERVICE_MODE=processor \
        . 2>&1 | while read line; do
            echo "  $line"
        done

    log_success "Docker image built successfully"

    # Tag for verification service
    docker tag nsi-edge:latest nsi-verification:latest
    log_success "Tagged nsi-verification:latest"
}

# =============================================================================
# Deploy Services
# =============================================================================

deploy_services() {
    log_section "Deploying Services"

    cd "${DEPLOYMENT_DIR}"

    # Create data directories
    log_info "Creating data directories..."
    mkdir -p "${PROJECT_ROOT}/data"/{input,output,cache,memory,verification}

    # Set environment variables
    export NSI_NODE_ID="${NODE_ID}"
    export NSI_TLS_ENABLED="${GENERATE_TLS}"
    export NSI_JURISDICTION="${NSI_JURISDICTION:-LOCAL}"

    log_info "Deploying with node ID: ${NODE_ID}"
    log_info "TLS enabled: ${GENERATE_TLS}"
    log_info "Jurisdiction: ${NSI_JURISDICTION:-LOCAL}"

    # Pull or use local images
    if [[ "$BUILD_IMAGES" != true ]]; then
        log_info "Pulling base images..."
        docker compose -f docker-compose.edge.yml pull --ignore-pull-failures || true
    fi

    # Start services
    log_info "Starting services..."
    docker compose -f docker-compose.edge.yml up -d

    log_success "Services deployed"

    # Wait for services to be healthy
    log_info "Waiting for services to become healthy..."
    local max_wait=120
    local waited=0

    while [[ $waited -lt $max_wait ]]; do
        local healthy=$(docker compose -f docker-compose.edge.yml ps --format json 2>/dev/null | \
            grep -c '"Health": "healthy"' || echo 0)
        local total=$(docker compose -f docker-compose.edge.yml ps --format json 2>/dev/null | \
            grep -c '"Service"' || echo 0)

        if [[ $healthy -eq $total && $total -gt 0 ]]; then
            log_success "All $total services are healthy"
            break
        fi

        echo -n "."
        sleep 5
        waited=$((waited + 5))
    done

    if [[ $waited -ge $max_wait ]]; then
        log_warning "Timeout waiting for services to be healthy"
        log_info "Checking service status..."
        docker compose -f docker-compose.edge.yml ps
    fi
}

# =============================================================================
# Verification
# =============================================================================

verify_deployment() {
    log_section "Verifying Deployment"

    cd "${DEPLOYMENT_DIR}"

    local all_passed=true

    # Check container status
    log_info "Checking container status..."
    docker compose -f docker-compose.edge.yml ps

    # Check processor health endpoint
    log_info "Checking imaging processor health..."
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
        log_success "Imaging processor is responding"
    else
        log_warning "Imaging processor health check failed"
        all_passed=false
    fi

    # Check verification service
    log_info "Checking verification service health..."
    if curl -sf http://localhost:8081/health > /dev/null 2>&1; then
        log_success "Verification service is responding"
    else
        log_warning "Verification service health check failed (may not be fully started)"
    fi

    # Check Redis
    log_info "Checking Redis connection..."
    if docker compose -f docker-compose.edge.yml exec -T cache redis-cli ping 2>/dev/null | grep -q PONG; then
        log_success "Redis is responding"
    else
        log_warning "Redis check failed"
        all_passed=false
    fi

    # Check Prometheus
    log_info "Checking Prometheus..."
    if curl -sf http://localhost:9090/-/ready > /dev/null 2>&1; then
        log_success "Prometheus is ready"
    else
        log_warning "Prometheus check failed"
    fi

    # Check metrics endpoint
    log_info "Checking metrics endpoint..."
    if curl -sf http://localhost:9100/metrics > /dev/null 2>&1; then
        log_success "Metrics endpoint is responding"
    else
        log_warning "Metrics endpoint check failed"
    fi

    # Display service URLs
    echo ""
    log_info "Service Endpoints:"
    echo "  - Imaging Processor: http://localhost:8080"
    echo "  - Verification:      http://localhost:8081"
    echo "  - Prometheus:        http://localhost:9090"
    echo "  - Metrics:           http://localhost:9100/metrics"

    if [[ "$all_passed" == true ]]; then
        log_success "Deployment verification completed successfully"
    else
        log_warning "Some verification checks failed. Review logs with:"
        echo "  docker compose -f docker-compose.edge.yml logs"
    fi

    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════════════╗"
    echo "║     Negative Space Imaging - Edge Deployment Script                       ║"
    echo "║     Raspberry Pi 4/5 (ARM64) Deployment                                   ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════╝"
    echo ""

    parse_args "$@"

    # Run prerequisite checks
    check_prerequisites

    # Verify-only mode
    if [[ "$VERIFY_ONLY" == true ]]; then
        verify_deployment
        exit 0
    fi

    # Clean if requested
    if [[ "$CLEAN_FIRST" == true ]]; then
        clean_deployment
    fi

    # Generate TLS certificates
    if [[ "$GENERATE_TLS" == true ]]; then
        generate_tls_certificates
    fi

    # Build images if requested
    if [[ "$BUILD_IMAGES" == true ]]; then
        build_images
    fi

    # Deploy services
    deploy_services

    # Verify deployment
    verify_deployment

    log_section "Deployment Complete"
    log_success "Edge node ${NODE_ID} is now running"
    echo ""
    echo "Useful commands:"
    echo "  View logs:     docker compose -f ${DEPLOYMENT_DIR}/docker-compose.edge.yml logs -f"
    echo "  Stop services: docker compose -f ${DEPLOYMENT_DIR}/docker-compose.edge.yml down"
    echo "  Restart:       docker compose -f ${DEPLOYMENT_DIR}/docker-compose.edge.yml restart"
    echo ""
}

# Run main function
main "$@"
