#!/bin/bash

################################################################################
# DOCKER HEALTH CHECK SCRIPT FOR NEGATIVE SPACE IMAGING PROJECT
#
# Purpose:
#   Comprehensive health check for all Docker containers and services
#   Validates Docker daemon, container status, service connectivity, and logs
#
# Usage:
#   ./docker-health-check.sh [OPTIONS]
#
# Options:
#   -v, --verbose         Enable verbose output with detailed debug info
#   -l, --log-file FILE   Custom log file path (default: logs/docker-health-{timestamp}.log)
#   -q, --quiet           Suppress console output (logs only)
#   -r, --repair          Attempt automatic repair of failed services
#   -h, --help            Display this help message
#
# Exit Codes:
#   0 - All services healthy
#   1 - One or more services degraded/failed
#   2 - Docker daemon not running
#   3 - Script execution error
#
# Author: DevOps Team
# Date: 2025-10-17
# Version: 1.0.0
#
################################################################################

set -euo pipefail

# ============================================================================
# COLOR AND FORMATTING CONSTANTS
# ============================================================================

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color
readonly BOLD='\033[1m'

# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================

VERBOSE_MODE=false
QUIET_MODE=false
REPAIR_MODE=false
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/docker-health-${TIMESTAMP}.log"
METRICS_FILE="${LOG_DIR}/docker-metrics-${TIMESTAMP}.json"

# Service configuration
declare -A SERVICES=(
  [app]="8000"
  [redis]="6379"
  [postgres]="5432"
  [monitoring]="9090"
  [grafana]="3000"
)

# Health check thresholds
CONTAINER_TIMEOUT=30
CONNECTION_TIMEOUT=5
MAX_RESTART_ATTEMPTS=3

# Status tracking
OVERALL_STATUS=0
SERVICES_FAILED=()
SERVICES_WARNING=()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

##
# Print colored status message
# Arguments:
#   $1 - status level (OK, WARN, ERROR, INFO, DEBUG)
#   $2 - message
##
log_message() {
  local level="$1"
  local message="$2"
  local timestamp
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')

  local color_code=""
  case "$level" in
    OK)     color_code="$GREEN" ;;
    WARN)   color_code="$YELLOW" ;;
    ERROR)  color_code="$RED" ;;
    INFO)   color_code="$BLUE" ;;
    DEBUG)  color_code="$CYAN" ;;
    *)      color_code="$NC" ;;
  esac

  local formatted_msg="${color_code}[${timestamp}] [${level}]${NC} ${message}"

  # Log to file
  echo "[${timestamp}] [${level}] ${message}" >> "$LOG_FILE"

  # Output to console unless quiet mode
  if [[ "$QUIET_MODE" == false ]]; then
    echo -e "$formatted_msg"
  fi

  # Debug output in verbose mode
  if [[ "$VERBOSE_MODE" == true && "$level" == "DEBUG" ]]; then
    echo -e "$formatted_msg" >&2
  fi
}

##
# Create necessary directories and initialize logging
##
initialize_environment() {
  # Create logs directory if it doesn't exist
  mkdir -p "$LOG_DIR"

  # Initialize log file with header
  {
    echo "========================================================================"
    echo "DOCKER HEALTH CHECK LOG"
    echo "========================================================================"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Hostname: $(hostname)"
    echo "User: $(whoami)"
    echo "Script Version: 1.0.0"
    echo "========================================================================"
    echo ""
  } > "$LOG_FILE"

  log_message "INFO" "Health check initialized - Log: ${LOG_FILE}"
}

##
# Display usage information
##
show_usage() {
  cat << EOF
${BOLD}Docker Health Check Script${NC}

${BOLD}Usage:${NC}
  ./docker-health-check.sh [OPTIONS]

${BOLD}Options:${NC}
  -v, --verbose         Enable verbose output with detailed debug information
  -l, --log-file FILE   Specify custom log file path
  -q, --quiet           Suppress console output (logs to file only)
  -r, --repair          Attempt automatic repair of failed services
  -h, --help            Display this help message

${BOLD}Examples:${NC}
  # Run standard health check
  ./docker-health-check.sh

  # Verbose mode with automatic repair
  ./docker-health-check.sh --verbose --repair

  # Quiet mode with custom log file
  ./docker-health-check.sh --quiet --log-file /var/log/docker-health.log

${BOLD}Exit Codes:${NC}
  0 - All services healthy
  1 - One or more services degraded/failed
  2 - Docker daemon not running
  3 - Script execution error

EOF
}

##
# Parse command-line arguments
##
parse_arguments() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      -v|--verbose)
        VERBOSE_MODE=true
        shift
        ;;
      -q|--quiet)
        QUIET_MODE=true
        shift
        ;;
      -r|--repair)
        REPAIR_MODE=true
        shift
        ;;
      -l|--log-file)
        LOG_FILE="$2"
        shift 2
        ;;
      -h|--help)
        show_usage
        exit 0
        ;;
      *)
        log_message "ERROR" "Unknown option: $1"
        show_usage
        exit 3
        ;;
    esac
  done
}

# ============================================================================
# DOCKER DAEMON CHECKS
# ============================================================================

##
# Check if Docker daemon is running
# Returns: 0 if running, 1 if not
##
check_docker_daemon() {
  log_message "INFO" "Checking Docker daemon status..."

  if ! docker info >/dev/null 2>&1; then
    log_message "ERROR" "Docker daemon is not running"
    log_message "INFO" "Recovery commands:"
    log_message "INFO" "  - On Linux: systemctl start docker"
    log_message "INFO" "  - On Docker Desktop: Open Docker Desktop application"
    log_message "INFO" "  - On WSL2: wsl --list --verbose (check if WSL2 backend is enabled)"
    return 1
  fi

  log_message "OK" "Docker daemon is running"
  return 0
}

##
# Get Docker version and system info
##
get_docker_system_info() {
  log_message "INFO" "Retrieving Docker system information..."

  local docker_version
  docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")

  log_message "INFO" "Docker Version: ${docker_version}"

  # Get containers running count
  local running_count
  running_count=$(docker ps --quiet | wc -l)

  log_message "INFO" "Running containers: ${running_count}"
}

# ============================================================================
# CONTAINER HEALTH CHECKS
# ============================================================================

##
# Check if a container exists
# Arguments:
#   $1 - container name
# Returns: 0 if exists, 1 if not
##
container_exists() {
  local container_name="$1"
  docker ps --all --filter "name=^${container_name}$" --quiet >/dev/null 2>&1
}

##
# Check container running status
# Arguments:
#   $1 - container name
# Returns: 0 if running, 1 if not
##
is_container_running() {
  local container_name="$1"
  docker ps --filter "name=^${container_name}$" --quiet >/dev/null 2>&1
}

##
# Get container health status
# Arguments:
#   $1 - container name
# Returns: "healthy", "unhealthy", "none", or "missing"
##
get_container_health() {
  local container_name="$1"

  if ! container_exists "$container_name"; then
    echo "missing"
    return
  fi

  # Try to get health status from inspect
  local health_status
  health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "none")

  echo "$health_status"
}

##
# Check individual container status
# Arguments:
#   $1 - container name
#   $2 - expected port
##
check_container_health() {
  local container_name="$1"
  local port="$2"

  log_message "INFO" "Checking container: ${BOLD}${container_name}${NC} (port ${port})"

  # Check if container exists
  if ! container_exists "$container_name"; then
    log_message "ERROR" "Container '${container_name}' does not exist"
    SERVICES_FAILED+=("$container_name")
    OVERALL_STATUS=1
    return 1
  fi

  # Check if running
  if ! is_container_running "$container_name"; then
    log_message "WARN" "Container '${container_name}' is not running"
    SERVICES_WARNING+=("$container_name")
    OVERALL_STATUS=1

    # Provide recovery command
    log_message "INFO" "Recovery: docker-compose up -d ${container_name}"
    return 1
  fi

  # Check health status
  local health
  health=$(get_container_health "$container_name")

  case "$health" in
    healthy|none)
      log_message "OK" "Container '${container_name}' is running (health: ${health})"
      return 0
      ;;
    unhealthy)
      log_message "WARN" "Container '${container_name}' is unhealthy"
      SERVICES_WARNING+=("$container_name")
      OVERALL_STATUS=1
      return 1
      ;;
    *)
      log_message "DEBUG" "Container '${container_name}' health check not available"
      return 0
      ;;
  esac
}

##
# Get container resource usage
# Arguments:
#   $1 - container name
##
get_container_stats() {
  local container_name="$1"

  if ! is_container_running "$container_name"; then
    return 1
  fi

  local stats
  stats=$(docker stats --no-stream --format "{{.MemUsage}}" "$container_name" 2>/dev/null || echo "N/A")

  log_message "DEBUG" "Container ${container_name} memory usage: ${stats}"
}

##
# Check container logs for errors
# Arguments:
#   $1 - container name
##
check_container_logs() {
  local container_name="$1"

  if ! is_container_running "$container_name"; then
    return 1
  fi

  log_message "DEBUG" "Recent logs for ${container_name}:"

  # Get last 5 lines of logs (suppress in quiet mode)
  if [[ "$VERBOSE_MODE" == true ]]; then
    docker logs --tail 5 "$container_name" 2>/dev/null | while read -r line; do
      log_message "DEBUG" "  [${container_name}] ${line}"
    done
  fi
}

# ============================================================================
# SERVICE CONNECTIVITY CHECKS
# ============================================================================

##
# Check if a port is accessible
# Arguments:
#   $1 - hostname
#   $2 - port
# Returns: 0 if accessible, 1 if not
##
check_port_accessibility() {
  local host="$1"
  local port="$2"

  if timeout "$CONNECTION_TIMEOUT" bash -c "cat < /dev/null > /dev/tcp/${host}/${port}" 2>/dev/null; then
    return 0
  fi
  return 1
}

##
# Test PostgreSQL connectivity
##
check_postgres_health() {
  log_message "INFO" "Testing PostgreSQL connectivity..."

  if ! check_port_accessibility "localhost" "5432"; then
    log_message "WARN" "PostgreSQL port 5432 is not accessible"
    SERVICES_WARNING+=("postgres")
    OVERALL_STATUS=1
    return 1
  fi

  # Try to connect with postgres
  if ! docker exec negative-space-imaging-project_postgres_1 \
    psql -U postgres -d negative_space -c "SELECT 1" >/dev/null 2>&1; then
    log_message "WARN" "PostgreSQL query test failed"
    SERVICES_WARNING+=("postgres")
    OVERALL_STATUS=1
    return 1
  fi

  log_message "OK" "PostgreSQL is accessible and responding"
  return 0
}

##
# Test Redis connectivity
##
check_redis_health() {
  log_message "INFO" "Testing Redis connectivity..."

  if ! check_port_accessibility "localhost" "6379"; then
    log_message "WARN" "Redis port 6379 is not accessible"
    SERVICES_WARNING+=("redis")
    OVERALL_STATUS=1
    return 1
  fi

  # Try to ping redis
  if ! docker exec negative-space-imaging-project_redis_1 \
    redis-cli ping >/dev/null 2>&1; then
    log_message "WARN" "Redis ping test failed"
    SERVICES_WARNING+=("redis")
    OVERALL_STATUS=1
    return 1
  fi

  log_message "OK" "Redis is accessible and responding"
  return 0
}

##
# Test application/API connectivity
##
check_app_health() {
  log_message "INFO" "Testing application health..."

  if ! check_port_accessibility "localhost" "8000"; then
    log_message "WARN" "Application port 8000 is not accessible"
    SERVICES_WARNING+=("app")
    OVERALL_STATUS=1
    return 1
  fi

  log_message "OK" "Application is accessible on port 8000"
  return 0
}

##
# Test Prometheus connectivity
##
check_prometheus_health() {
  log_message "INFO" "Testing Prometheus connectivity..."

  if ! check_port_accessibility "localhost" "9090"; then
    log_message "WARN" "Prometheus port 9090 is not accessible"
    SERVICES_WARNING+=("monitoring")
    OVERALL_STATUS=1
    return 1
  fi

  log_message "OK" "Prometheus is accessible on port 9090"
  return 0
}

##
# Test Grafana connectivity
##
check_grafana_health() {
  log_message "INFO" "Testing Grafana connectivity..."

  if ! check_port_accessibility "localhost" "3000"; then
    log_message "WARN" "Grafana port 3000 is not accessible"
    SERVICES_WARNING+=("grafana")
    OVERALL_STATUS=1
    return 1
  fi

  log_message "OK" "Grafana is accessible on port 3000"
  return 0
}

# ============================================================================
# REPAIR AND RECOVERY FUNCTIONS
# ============================================================================

##
# Attempt to repair failed services
##
repair_services() {
  if [[ "$REPAIR_MODE" == false ]]; then
    return 0
  fi

  log_message "INFO" "Attempting to repair failed services..."

  for service in "${SERVICES_WARNING[@]}"; do
    log_message "INFO" "Restarting service: ${service}"

    if docker-compose restart "$service" >/dev/null 2>&1; then
      log_message "OK" "Successfully restarted ${service}"
    else
      log_message "ERROR" "Failed to restart ${service}"
    fi

    # Wait for service to start
    sleep 2
  done

  # Re-run health checks
  log_message "INFO" "Running post-repair health checks..."
  sleep 5
}

# ============================================================================
# REPORTING AND SUMMARY
# ============================================================================

##
# Generate JSON metrics report
##
generate_metrics_report() {
  local uptime
  uptime=$(docker stats --no-stream --format "{{json .}}" 2>/dev/null || echo "{}")

  {
    echo "{"
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"overall_status\": \"$([ $OVERALL_STATUS -eq 0 ] && echo 'healthy' || echo 'degraded')\","
    echo "  \"docker_version\": \"$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 'unknown')\","
    echo "  \"containers_running\": $(docker ps --quiet | wc -l),"
    echo "  \"containers_total\": $(docker ps --all --quiet | wc -l),"
    echo "  \"services_checked\": $(( ${#SERVICES[@]} )),"
    echo "  \"services_failed\": $(( ${#SERVICES_FAILED[@]} )),"
    echo "  \"services_warning\": $(( ${#SERVICES_WARNING[@]} ))"
    echo "}"
  } > "$METRICS_FILE"

  log_message "INFO" "Metrics report saved to: ${METRICS_FILE}"
}

##
# Print health check summary
##
print_summary() {
  echo ""
  echo -e "${BOLD}========================================================================"
  echo -e "DOCKER HEALTH CHECK SUMMARY"
  echo -e "========================================================================${NC}"
  echo ""

  echo -e "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
  echo -e "Overall Status: $([ $OVERALL_STATUS -eq 0 ] && echo -e "${GREEN}HEALTHY${NC}" || echo -e "${RED}DEGRADED${NC}")"
  echo ""

  echo -e "${BOLD}Container Status:${NC}"
  for service in "${!SERVICES[@]}"; do
    if is_container_running "$service"; then
      echo -e "  ${GREEN}✓${NC} ${service} (${SERVICES[$service]})"
    else
      echo -e "  ${RED}✗${NC} ${service} (${SERVICES[$service]})"
    fi
  done

  echo ""

  if [[ ${#SERVICES_FAILED[@]} -gt 0 ]]; then
    echo -e "${RED}${BOLD}Failed Services:${NC}"
    for service in "${SERVICES_FAILED[@]}"; do
      echo -e "  ${RED}•${NC} ${service}"
    done
    echo ""
  fi

  if [[ ${#SERVICES_WARNING[@]} -gt 0 ]]; then
    echo -e "${YELLOW}${BOLD}Warning Services:${NC}"
    for service in "${SERVICES_WARNING[@]}"; do
      echo -e "  ${YELLOW}•${NC} ${service}"
    done
    echo ""
  fi

  echo -e "${BOLD}Logs:${NC}"
  echo -e "  Full Log: ${LOG_FILE}"
  echo -e "  Metrics:  ${METRICS_FILE}"
  echo ""

  if [[ ${#SERVICES_FAILED[@]} -gt 0 ]] || [[ ${#SERVICES_WARNING[@]} -gt 0 ]]; then
    echo -e "${YELLOW}${BOLD}Recovery Commands:${NC}"
    echo -e "  Restart all containers: ${CYAN}docker-compose restart${NC}"
    echo -e "  View logs: ${CYAN}tail -f ${LOG_FILE}${NC}"
    echo ""
  fi

  echo -e "${BOLD}========================================================================"
  echo -e "End of Report${NC}"
  echo ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

##
# Main health check function
##
main() {
  trap 'log_message "ERROR" "Script interrupted"; exit 3' INT TERM

  # Parse arguments
  parse_arguments "$@"

  # Initialize environment
  initialize_environment

  log_message "INFO" "Starting comprehensive Docker health check..."
  log_message "DEBUG" "Verbose Mode: ${VERBOSE_MODE}"
  log_message "DEBUG" "Repair Mode: ${REPAIR_MODE}"

  # ========================================================================
  # PHASE 1: DOCKER DAEMON CHECK
  # ========================================================================

  if ! check_docker_daemon; then
    log_message "ERROR" "Cannot proceed: Docker daemon not running"
    print_summary
    exit 2
  fi

  get_docker_system_info

  # ========================================================================
  # PHASE 2: CONTAINER HEALTH CHECKS
  # ========================================================================

  log_message "INFO" "Phase 2: Checking container status..."
  for service in "${!SERVICES[@]}"; do
    check_container_health "$service" "${SERVICES[$service]}"
    get_container_stats "$service"
    check_container_logs "$service"
  done

  # ========================================================================
  # PHASE 3: SERVICE CONNECTIVITY CHECKS
  # ========================================================================

  log_message "INFO" "Phase 3: Testing service connectivity..."
  check_postgres_health
  check_redis_health
  check_app_health
  check_prometheus_health
  check_grafana_health

  # ========================================================================
  # PHASE 4: REPAIR AND RECOVERY (if enabled)
  # ========================================================================

  if [[ ${#SERVICES_WARNING[@]} -gt 0 ]] && [[ "$REPAIR_MODE" == true ]]; then
    log_message "INFO" "Phase 4: Attempting repair..."
    repair_services
  fi

  # ========================================================================
  # FINAL REPORTING
  # ========================================================================

  generate_metrics_report
  print_summary

  log_message "INFO" "Health check completed with exit code: ${OVERALL_STATUS}"

  exit "$OVERALL_STATUS"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
