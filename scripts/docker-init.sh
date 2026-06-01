#!/bin/bash
# ============================================================================
# Docker Initialization Script - Setup environment and validate configuration
# ============================================================================
# This script sets up the Docker environment for the project
#
# Usage:
#   ./scripts/docker-init.sh              # Setup development environment
#   ./scripts/docker-init.sh prod         # Setup production environment
#   ./scripts/docker-init.sh --clean      # Clean and reinitialize

set -e

ENVIRONMENT="${1:-dev}"
CLEAN="${2:---no-clean}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Docker Environment Initialization${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}============================================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker found: $(docker --version)${NC}"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}✗ Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker Compose found: $(docker-compose --version)${NC}"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env from .env.example...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env created${NC}"
else
    echo -e "${GREEN}✓ .env already exists${NC}"
fi

# Create required directories
echo -e "${YELLOW}Creating required directories...${NC}"
mkdir -p uploads shared_data logs monitoring
chmod 755 uploads shared_data logs monitoring
echo -e "${GREEN}✓ Directories created${NC}"

# Validate docker-compose.yml
echo -e "${YELLOW}Validating docker-compose configuration...${NC}"
if docker-compose config > /dev/null 2>&1; then
    echo -e "${GREEN}✓ docker-compose.yml is valid${NC}"
else
    echo -e "${RED}✗ docker-compose.yml validation failed${NC}"
    exit 1
fi

# Production setup
if [ "$ENVIRONMENT" = "prod" ]; then
    echo -e "${YELLOW}Setting up production environment...${NC}"

    # Check for production .env file
    if [ ! -f .env.prod ]; then
        echo -e "${YELLOW}Creating .env.prod...${NC}"
        cp .env.example .env.prod
        echo -e "${YELLOW}⚠ Please update .env.prod with production values${NC}"
    fi

    # Validate production compose
    if docker-compose -f docker-compose.yml -f docker-compose.prod.yml config > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Production docker-compose configuration is valid${NC}"
    else
        echo -e "${RED}✗ Production configuration validation failed${NC}"
        exit 1
    fi
fi

# Clean up if requested
if [ "$CLEAN" = "--clean" ]; then
    echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
    docker-compose down -v 2>/dev/null || true
    docker system prune -f 2>/dev/null || true
    echo -e "${GREEN}✓ Cleanup complete${NC}"
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}✓ Docker environment initialized successfully!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  1. Review and update .env configuration"
echo -e "  2. Run: ${GREEN}docker-compose build${NC}"
echo -e "  3. Run: ${GREEN}docker-compose up -d${NC}"
echo -e "  4. Check: ${GREEN}docker-compose ps${NC}"
echo -e ""
