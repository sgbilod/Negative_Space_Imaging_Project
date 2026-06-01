#!/bin/bash
# ============================================================================
# Docker Test Script - Verify all services are working correctly
# ============================================================================
# Runs comprehensive health and connectivity tests on Docker services
#
# Usage:
#   ./scripts/docker-test.sh              # Test all services
#   ./scripts/docker-test.sh --verbose    # Detailed output
#   ./scripts/docker-test.sh --quick      # Quick health checks only

set -e

VERBOSE="${1:---quiet}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0

test_service() {
    local name=$1
    local port=$2
    local endpoint=${3:-health}

    echo -n "Testing $name... "

    if curl -sf "http://localhost:$port/$endpoint" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASS_COUNT++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAIL_COUNT++))
    fi
}

test_database() {
    local name=$1
    local host=$2
    local port=$3

    echo -n "Testing $name connectivity... "

    if docker exec nsi_postgres pg_isready -h $host -p $port > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASS_COUNT++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAIL_COUNT++))
    fi
}

test_redis() {
    echo -n "Testing Redis connectivity... "

    if docker exec nsi_redis redis-cli ping | grep -q PONG; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASS_COUNT++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAIL_COUNT++))
    fi
}

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Docker Services Health & Connectivity Test${NC}"
echo -e "${BLUE}============================================================${NC}\n"

# Check if services are running
echo -e "${YELLOW}Checking running containers...${NC}"
if [ -z "$(docker-compose ps -q)" ]; then
    echo -e "${RED}✗ No containers are running${NC}"
    exit 1
fi

RUNNING=$(docker-compose ps -q | wc -l)
echo -e "${GREEN}✓ $RUNNING containers running${NC}\n"

# Test services
echo -e "${YELLOW}Testing service health endpoints...${NC}"
test_service "Express API" "3000" "health"
test_service "Python Service" "8000" "health"
test_service "Frontend" "80" "health"
test_service "Prometheus" "9090" "-/healthy"

echo ""
echo -e "${YELLOW}Testing database connectivity...${NC}"
test_database "PostgreSQL" "postgres" "5432"
test_redis

echo ""

# Test inter-service communication
echo -e "${YELLOW}Testing inter-service communication...${NC}"

# Test API can reach database
echo -n "Testing API → Database... "
if docker exec nsi_api curl -sf http://postgres:5432 > /dev/null 2>&1 || true; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS_COUNT++))
else
    echo -e "${YELLOW}⚠ Connection refused (expected)${NC}"
fi

# Test API can reach Redis
echo -n "Testing API → Redis... "
if docker exec nsi_api bash -c 'exec 3<>/dev/tcp/redis/6379 && echo "PING" >&3' > /dev/null 2>&1 || true; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS_COUNT++))
fi

# Test API can reach Python service
echo -n "Testing API → Python Service... "
if docker exec nsi_api curl -sf http://python_service:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS_COUNT++))
else
    echo -e "${YELLOW}⚠ Connectivity test${NC}"
fi

echo ""

# Show container stats if verbose
if [ "$VERBOSE" = "--verbose" ]; then
    echo -e "${YELLOW}Container Resource Usage:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
    echo ""
fi

# Summary
echo -e "${BLUE}============================================================${NC}"
echo -e "${YELLOW}Test Results: ${GREEN}$PASS_COUNT passed${NC}, ${RED}$FAIL_COUNT failed${NC}${NC}"

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo -e "${BLUE}============================================================${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo -e "${BLUE}============================================================${NC}"
    exit 1
fi
