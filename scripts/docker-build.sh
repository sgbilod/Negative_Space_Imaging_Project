#!/bin/bash

# ============================================================================
# Docker Build & Push Script
# Builds multi-stage Docker images and pushes to registry
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGISTRY="${DOCKER_REGISTRY:-docker.io}"
PROJECT_NAME="${PROJECT_NAME:-negative-space-imaging}"
VERSION="${VERSION:-$(git describe --tags --always)}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Parse arguments
BUILD_ONLY=false
PUSH=false
SKIP_TESTS=false
PLATFORMS="linux/amd64"

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --multi-platform)
            PLATFORMS="linux/amd64,linux/arm64"
            shift
            ;;
        --version)
            VERSION=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Functions
print_header() {
    echo -e "\n${BLUE}=================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker found: $(docker --version)"

    # Check Docker buildx for multi-platform
    if [[ "$PLATFORMS" == *","* ]]; then
        if ! docker buildx version &> /dev/null; then
            print_error "Docker buildx is not available for multi-platform builds"
            exit 1
        fi
        print_success "Docker buildx available"
    fi

    # Check Git
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed"
        exit 1
    fi
    print_success "Git found: $(git --version)"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        print_warning "Skipping tests (--skip-tests flag set)"
        return
    fi

    print_header "Running Tests"

    if [[ -f "package.json" ]]; then
        print_info "Running Node.js tests..."
        npm ci --prefer-offline --no-audit || print_warning "Node tests failed (continuing)"
        npm run test || print_warning "Some tests failed"
    fi
}

# Build Docker images
build_images() {
    print_header "Building Docker Images"

    # Determine build command
    local BUILD_CMD="docker build"
    if [[ "$PLATFORMS" == *","* ]]; then
        BUILD_CMD="docker buildx build --push"
    fi

    # Build API
    print_info "Building API image..."
    $BUILD_CMD \
        --file Dockerfile.api \
        --tag "$REGISTRY/$PROJECT_NAME/api:$VERSION" \
        --tag "$REGISTRY/$PROJECT_NAME/api:latest" \
        --label "build.date=$BUILD_DATE" \
        --label "vcs.ref=$GIT_COMMIT" \
        --label "vcs.branch=$GIT_BRANCH" \
        --label "version=$VERSION" \
        --platform "$PLATFORMS" \
        .
    print_success "API image built: $REGISTRY/$PROJECT_NAME/api:$VERSION"

    # Build Frontend
    print_info "Building Frontend image..."
    $BUILD_CMD \
        --file Dockerfile.frontend \
        --tag "$REGISTRY/$PROJECT_NAME/frontend:$VERSION" \
        --tag "$REGISTRY/$PROJECT_NAME/frontend:latest" \
        --label "build.date=$BUILD_DATE" \
        --label "vcs.ref=$GIT_COMMIT" \
        --label "vcs.branch=$GIT_BRANCH" \
        --label "version=$VERSION" \
        --platform "$PLATFORMS" \
        .
    print_success "Frontend image built: $REGISTRY/$PROJECT_NAME/frontend:$VERSION"

    # Build Python Analyzer
    print_info "Building Python Analyzer image..."
    $BUILD_CMD \
        --file Dockerfile.python \
        --tag "$REGISTRY/$PROJECT_NAME/analyzer:$VERSION" \
        --tag "$REGISTRY/$PROJECT_NAME/analyzer:latest" \
        --label "build.date=$BUILD_DATE" \
        --label "vcs.ref=$GIT_COMMIT" \
        --label "vcs.branch=$GIT_BRANCH" \
        --label "version=$VERSION" \
        --platform "$PLATFORMS" \
        .
    print_success "Python Analyzer image built: $REGISTRY/$PROJECT_NAME/analyzer:$VERSION"
}

# Push images
push_images() {
    if [[ "$PUSH" != true ]]; then
        return
    fi

    print_header "Pushing Images to Registry"

    # Push API
    print_info "Pushing API image..."
    docker push "$REGISTRY/$PROJECT_NAME/api:$VERSION"
    docker push "$REGISTRY/$PROJECT_NAME/api:latest"
    print_success "API image pushed"

    # Push Frontend
    print_info "Pushing Frontend image..."
    docker push "$REGISTRY/$PROJECT_NAME/frontend:$VERSION"
    docker push "$REGISTRY/$PROJECT_NAME/frontend:latest"
    print_success "Frontend image pushed"

    # Push Analyzer
    print_info "Pushing Analyzer image..."
    docker push "$REGISTRY/$PROJECT_NAME/analyzer:$VERSION"
    docker push "$REGISTRY/$PROJECT_NAME/analyzer:latest"
    print_success "Analyzer image pushed"
}

# Show image info
show_image_info() {
    print_header "Built Images"

    docker images | grep "$PROJECT_NAME" || print_warning "No images found"

    echo ""
    print_info "Image tags:"
    echo "  API:             $REGISTRY/$PROJECT_NAME/api:$VERSION"
    echo "  Frontend:        $REGISTRY/$PROJECT_NAME/frontend:$VERSION"
    echo "  Analyzer:        $REGISTRY/$PROJECT_NAME/analyzer:$VERSION"
}

# Main execution
main() {
    print_header "Docker Build & Push Script"
    echo "Registry:     $REGISTRY"
    echo "Project:      $PROJECT_NAME"
    echo "Version:      $VERSION"
    echo "Platforms:    $PLATFORMS"
    echo "Push:         $PUSH"
    echo "Build Date:   $BUILD_DATE"
    echo "Git Commit:   $GIT_COMMIT"
    echo "Git Branch:   $GIT_BRANCH"

    check_prerequisites
    run_tests
    build_images
    push_images
    show_image_info

    print_header "Build Complete"
    print_success "All done!"
}

main "$@"
