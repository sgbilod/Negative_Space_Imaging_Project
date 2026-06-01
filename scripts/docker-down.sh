#!/bin/bash
# ============================================================================
# Docker Down Script - Gracefully stop all services
# ============================================================================
# Stops Docker Compose services with optional cleanup
#
# Usage:
#   ./scripts/docker-down.sh              # Stop services, keep volumes
#   ./scripts/docker-down.sh --volumes    # Stop services and remove volumes
#   ./scripts/docker-down.sh --all        # Stop, remove volumes, images, and prune
#   ./scripts/docker-down.sh --force      # Force stop without graceful shutdown

set -e

CLEANUP="${1:---no-cleanup}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Stopping Docker Services${NC}"
echo -e "${BLUE}Cleanup: ${CLEANUP}${NC}"
echo -e "${BLUE}============================================================${NC}"

# Check if Docker daemon is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Docker daemon is not running${NC}"
    exit 0
fi

# Check if any containers are running
if [ -z "$(docker-compose ps -q 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠ No containers are currently running${NC}"
    exit 0
fi

# Stop services based on cleanup mode
case "$CLEANUP" in
    --volumes)
        echo -e "${YELLOW}Stopping services and removing volumes...${NC}"
        docker-compose down -v
        echo -e "${GREEN}✓ Services stopped and volumes removed${NC}"
        ;;
    --all)
        echo -e "${YELLOW}Stopping services, removing volumes, images, and pruning...${NC}"
        docker-compose down -v --remove-orphans
        docker system prune -a -f
        echo -e "${GREEN}✓ Full cleanup complete${NC}"
        ;;
    --force)
        echo -e "${YELLOW}Force stopping containers...${NC}"
        docker-compose kill
        docker-compose down
        echo -e "${GREEN}✓ Containers forcefully stopped${NC}"
        ;;
    *)
        echo -e "${YELLOW}Stopping services (gracefully)...${NC}"
        docker-compose down --timeout=30
        echo -e "${GREEN}✓ Services stopped${NC}"
        echo -e "${YELLOW}⚠ Volumes retained. Use --volumes to remove them${NC}"
        ;;
esac

# Show remaining containers
if [ -n "$(docker ps -q)" ]; then
    echo -e "\n${YELLOW}⚠ Some containers still running:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}"
else
    echo -e "\n${GREEN}✓ All containers stopped${NC}"
fi

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}✓ Shutdown complete!${NC}"
echo -e "${BLUE}============================================================${NC}"
