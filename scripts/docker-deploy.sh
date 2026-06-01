#!/bin/bash

# ============================================================================
# Docker Compose Deployment Script
# Manages containerized deployment lifecycle
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ENVIRONMENT="${ENVIRONMENT:-development}"
COMPOSE_FILE="docker-compose.yml"
LOG_DIR="./logs"

if [[ "$ENVIRONMENT" == "production" ]]; then
    COMPOSE_FILE="docker-compose.prod.yml"
fi

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

# Show usage
show_usage() {
    cat << EOF
Usage: docker-deploy.sh [COMMAND] [OPTIONS]

Commands:
    up              Start all services
    down            Stop all services
    restart         Restart services
    logs            Show service logs
    status          Show service status
    health          Check service health
    backup          Backup database
    restore         Restore database
    clean           Remove containers and volumes
    ps              List running containers
    exec            Execute command in service
    prune           Remove unused Docker resources
    version         Show version information

Options:
    --env ENV       Set environment (development, production)
    --service SVC   Specific service name
    --follow        Follow logs
    --timestamps    Include timestamps in logs
    --lines N       Show last N lines of logs
    --help          Show this help message

Examples:
    docker-deploy.sh up
    docker-deploy.sh logs --service api --follow
    docker-deploy.sh restart --env production
    docker-deploy.sh health
    docker-deploy.sh backup

EOF
}

# Parse arguments
COMMAND="${1:-help}"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT=$2
            COMPOSE_FILE="docker-compose${ENVIRONMENT:+.${ENVIRONMENT}}.yml"
            shift 2
            ;;
        --service)
            SERVICE=$2
            shift 2
            ;;
        --follow)
            FOLLOW_LOGS=true
            shift
            ;;
        --timestamps)
            LOG_TIMESTAMPS=true
            shift
            ;;
        --lines)
            LOG_LINES=$2
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Setup
mkdir -p "$LOG_DIR"
set +e # Continue on error for some commands

# Commands
cmd_up() {
    print_header "Starting Services ($ENVIRONMENT)"

    # Build images
    print_info "Building images..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache || print_warning "Build completed with warnings"

    # Start services
    print_info "Starting containers..."
    docker-compose -f "$COMPOSE_FILE" up -d || {
        print_error "Failed to start services"
        exit 1
    }

    # Wait for services to be healthy
    print_info "Waiting for services to become healthy..."
    sleep 5

    docker-compose -f "$COMPOSE_FILE" ps
    print_success "Services started successfully"
}

cmd_down() {
    print_header "Stopping Services"

    docker-compose -f "$COMPOSE_FILE" down || {
        print_error "Failed to stop services"
        exit 1
    }

    print_success "Services stopped"
}

cmd_restart() {
    print_header "Restarting Services"

    docker-compose -f "$COMPOSE_FILE" restart ${SERVICE:-} || {
        print_error "Failed to restart services"
        exit 1
    }

    print_success "Services restarted"
}

cmd_logs() {
    print_header "Showing Logs"

    local OPTS="-f"
    [[ -z "${FOLLOW_LOGS:-}" ]] && OPTS=""
    [[ -n "${LOG_TIMESTAMPS:-}" ]] && OPTS="$OPTS --timestamps"
    [[ -n "${LOG_LINES:-}" ]] && OPTS="$OPTS --tail ${LOG_LINES}"

    docker-compose -f "$COMPOSE_FILE" logs $OPTS ${SERVICE:-}
}

cmd_status() {
    print_header "Service Status"
    docker-compose -f "$COMPOSE_FILE" ps
}

cmd_health() {
    print_header "Health Check"

    services=$(docker-compose -f "$COMPOSE_FILE" config --services)

    for service in $services; do
        status=$(docker-compose -f "$COMPOSE_FILE" ps "$service" | tail -1 | awk '{print $NF}')

        if [[ "$status" == *"Up"* ]]; then
            print_success "$service is UP"
        else
            print_warning "$service is $status"
        fi
    done
}

cmd_backup() {
    print_header "Database Backup"

    BACKUP_FILE="$LOG_DIR/backup-$(date +%Y%m%d-%H%M%S).sql"

    print_info "Backing up database to $BACKUP_FILE..."

    docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump \
        -U "${DB_USER:-postgres}" \
        "${DB_NAME:-negative_space}" > "$BACKUP_FILE" || {
        print_error "Backup failed"
        exit 1
    }

    print_success "Backup created: $BACKUP_FILE"
    ls -lh "$BACKUP_FILE"
}

cmd_restore() {
    BACKUP_FILE="${1:-$LOG_DIR/latest-backup.sql}"

    if [[ ! -f "$BACKUP_FILE" ]]; then
        print_error "Backup file not found: $BACKUP_FILE"
        exit 1
    fi

    print_header "Database Restore"
    print_warning "This will overwrite the current database!"
    read -p "Continue? (yes/no): " -r

    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        print_info "Restore cancelled"
        exit 0
    fi

    print_info "Restoring from $BACKUP_FILE..."

    docker-compose -f "$COMPOSE_FILE" exec -T postgres psql \
        -U "${DB_USER:-postgres}" \
        "${DB_NAME:-negative_space}" < "$BACKUP_FILE" || {
        print_error "Restore failed"
        exit 1
    }

    print_success "Database restored"
}

cmd_clean() {
    print_header "Cleaning Up"
    print_warning "This will remove containers and volumes!"
    read -p "Continue? (yes/no): " -r

    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        print_info "Cleanup cancelled"
        exit 0
    fi

    print_info "Removing containers and volumes..."
    docker-compose -f "$COMPOSE_FILE" down -v || {
        print_error "Cleanup failed"
        exit 1
    }

    print_success "Cleanup complete"
}

cmd_ps() {
    docker ps -a --filter "label=com.docker.compose.project" || docker ps -a
}

cmd_exec() {
    SERVICE="${SERVICE:-api}"
    COMMAND="${1:-/bin/bash}"

    docker-compose -f "$COMPOSE_FILE" exec "$SERVICE" $COMMAND
}

cmd_prune() {
    print_header "Docker System Prune"
    print_warning "This will remove unused Docker resources!"
    read -p "Continue? (yes/no): " -r

    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        print_info "Prune cancelled"
        exit 0
    fi

    docker system prune -af --volumes || {
        print_error "Prune failed"
        exit 1
    }

    print_success "Prune complete"
}

cmd_version() {
    print_header "Version Information"

    echo "Docker Compose: $(docker-compose --version)"
    echo "Docker: $(docker --version)"
    echo "Environment: $ENVIRONMENT"
    echo "Compose File: $COMPOSE_FILE"
}

# Execute command
case "$COMMAND" in
    up)
        cmd_up
        ;;
    down)
        cmd_down
        ;;
    restart)
        cmd_restart
        ;;
    logs)
        cmd_logs
        ;;
    status|ps)
        cmd_status
        ;;
    health)
        cmd_health
        ;;
    backup)
        cmd_backup
        ;;
    restore)
        cmd_restore "${1:-}"
        ;;
    clean)
        cmd_clean
        ;;
    exec)
        cmd_exec "${1:-}"
        ;;
    prune)
        cmd_prune
        ;;
    version)
        cmd_version
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac

set -e
