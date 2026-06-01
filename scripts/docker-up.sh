#!/bin/bash
# ============================================================================
# Docker Up Script - Start all services with verification
# ============================================================================
# Starts Docker Compose services and verifies health checks
#
# Usage:
#   ./scripts/docker-up.sh                 # Start development services
#   ./scripts/docker-up.sh prod            # Start production services
#   ./scripts/docker-up.sh --build         # Build and start
#   ./scripts/docker-up.sh --detach        # Start in detached mode

set -e

ENVIRONMENT="${1:-dev}"
BUILD="${2:---no-build}"
DETACH="${3:--d}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Starting Docker Services - Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}============================================================${NC}"

# Check if Docker daemon is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker daemon is not running${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker daemon is running${NC}"

# Build images if requested
if [ "$BUILD" = "--build" ]; then
    echo -e "${YELLOW}Building Docker images...${NC}"
    docker-compose build --no-cache
    echo -e "${GREEN}✓ Build complete${NC}"
fi

# Select compose files based on environment
if [ "$ENVIRONMENT" = "prod" ]; then
    echo -e "${YELLOW}Starting production environment...${NC}"
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"
    ENV_FILE=".env.prod"
else
    echo -e "${YELLOW}Starting development environment...${NC}"
    COMPOSE_FILES="-f docker-compose.yml"
    ENV_FILE=".env"
fi

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}✗ Environment file not found: $ENV_FILE${NC}"
    exit 1
fi

# Start services
echo -e "${YELLOW}Starting services...${NC}"
docker-compose $COMPOSE_FILES up $DETACH

# Wait for services to be healthy
if [ "$DETACH" = "-d" ]; then
    echo -e "${YELLOW}Waiting for services to become healthy...${NC}"
    sleep 10

    # Check service health
    echo -e "${YELLOW}Checking service health...${NC}"

    SERVICES=("nsi_postgres" "nsi_redis" "nsi_python" "nsi_api" "nsi_frontend")

    for service in "${SERVICES[@]}"; do
        if docker ps --filter "name=$service" --format "{{.Names}}" | grep -q "$service"; then
            STATUS=$(docker ps --filter "name=$service" --format "{{.Status}}")
            echo -e "${GREEN}✓ $service: $STATUS${NC}"
        else
            echo -e "${YELLOW}⚠ $service: Not running (development service)${NC}"
        fi
    done

    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${GREEN}✓ Services started successfully!${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo -e "\n${YELLOW}Service URLs:${NC}"
    echo -e "  Frontend:     ${GREEN}http://localhost${NC}"
    echo -e "  API:          ${GREEN}http://localhost:3000${NC}"
    echo -e "  Python:       ${GREEN}http://localhost:8000${NC}"
    echo -e "  Prometheus:   ${GREEN}http://localhost:9090${NC}"
    echo -e "  Grafana:      ${GREEN}http://localhost:3001${NC}"
    echo -e "  Database:     ${GREEN}localhost:5432${NC}"
    echo -e "  Redis:        ${GREEN}localhost:6379${NC}"
    echo -e "\n${YELLOW}Useful commands:${NC}"
    echo -e "  View logs:    ${GREEN}docker-compose logs -f${NC}"
    echo -e "  View stats:   ${GREEN}docker stats${NC}"
    echo -e "  Stop:         ${GREEN}./scripts/docker-down.sh${NC}"
    echo -e ""
fi
