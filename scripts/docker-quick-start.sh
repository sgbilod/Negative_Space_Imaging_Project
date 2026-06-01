#!/bin/bash

################################################################################
# DOCKER QUICK START & USAGE GUIDE
#
# This script demonstrates common use cases and commands for the Docker
# health check and initialization scripts.
#
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_section() {
  echo -e "\n${BLUE}${1}${NC}"
  echo -e "${BLUE}$(printf '=%.0s' {1..70})${NC}\n"
}

print_command() {
  echo -e "${CYAN}$ ${1}${NC}"
}

print_success() {
  echo -e "${GREEN}✓ ${1}${NC}"
}

print_info() {
  echo -e "${YELLOW}ℹ ${1}${NC}"
}

# ============================================================================
# MAIN MENU
# ============================================================================

show_menu() {
  clear
  echo -e "${CYAN}"
  cat << "EOF"
╔════════════════════════════════════════════════════════════════════════╗
║          Docker Health Check & Initialization Quick Start             ║
║                Negative Space Imaging Project                          ║
╚════════════════════════════════════════════════════════════════════════╝
EOF
  echo -e "${NC}"

  echo "Select an option:"
  echo ""
  echo "  1) Run Basic Health Check"
  echo "  2) Run Verbose Health Check"
  echo "  3) Run Health Check with Auto-Repair"
  echo "  4) Run Initialization"
  echo "  5) Run Full Setup (Start + Init + Health)"
  echo "  6) View Recent Logs"
  echo "  7) View Service Status"
  echo "  8) Docker System Info"
  echo "  9) Cleanup Old Logs"
  echo "  0) Exit"
  echo ""
  echo -n "Choice [0-9]: "
}

# ============================================================================
# SCRIPT OPERATIONS
# ============================================================================

run_basic_health_check() {
  print_section "Running Basic Health Check"
  print_command "./docker-health-check.sh"
  ./docker-health-check.sh

  print_section "Health Check Summary"
  print_info "Check logs directory for detailed reports:"
  print_command "ls -lah logs/docker-health-*.log"
  ls -lah logs/docker-health-*.log | head -5
}

run_verbose_health_check() {
  print_section "Running Verbose Health Check"
  print_command "./docker-health-check.sh --verbose"
  ./docker-health-check.sh --verbose
}

run_repair_health_check() {
  print_section "Running Health Check with Auto-Repair"
  print_command "./docker-health-check.sh --verbose --repair"
  ./docker-health-check.sh --verbose --repair
}

run_initialization() {
  print_section "Running Initialization"

  if ! command -v node &> /dev/null; then
    print_info "Node.js not found. Attempting to use available method..."
    if command -v npm &> /dev/null; then
      print_command "npm run health-init"
      npm run health-init 2>/dev/null || print_info "npm script not configured"
    fi
  else
    print_command "node docker-init.js --verbose"
    node docker-init.js --verbose
  fi
}

run_full_setup() {
  print_section "Full Setup: Start Containers + Initialize + Health Check"

  print_section "Step 1: Starting Docker Containers"
  print_command "docker-compose up -d"
  docker-compose up -d

  print_section "Step 2: Waiting for services to initialize"
  print_info "Waiting 15 seconds for services to start..."
  for i in {15..1}; do
    echo -ne "\r  Waiting: ${i}s remaining...  "
    sleep 1
  done
  echo ""

  print_section "Step 3: Running Initialization"
  if command -v node &> /dev/null; then
    print_command "node docker-init.js --verbose"
    node docker-init.js --verbose
  else
    print_info "Node.js not available, skipping detailed init"
  fi

  print_section "Step 4: Running Health Check"
  print_command "./docker-health-check.sh --verbose"
  ./docker-health-check.sh --verbose

  print_success "Full setup completed!"
}

view_recent_logs() {
  print_section "Recent Health Check Logs"

  if [ ! -d "logs" ] || [ -z "$(find logs -name 'docker-*.log' 2>/dev/null)" ]; then
    print_info "No logs found. Run a health check first:"
    print_command "./docker-health-check.sh"
    return
  fi

  local latest_log=$(find logs -name 'docker-health-*.log' -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

  if [ -z "$latest_log" ]; then
    print_info "No health check logs found"
    return
  fi

  echo -e "${CYAN}Latest log: ${latest_log}${NC}\n"

  echo "Preview (last 50 lines):"
  echo -e "${CYAN}$(printf '=%.0s' {1..70})${NC}"
  tail -50 "$latest_log"
  echo -e "${CYAN}$(printf '=%.0s' {1..70})${NC}"

  echo -e "\n${YELLOW}To view full log:${NC}"
  print_command "cat ${latest_log}"

  echo -e "\n${YELLOW}To follow log in real-time:${NC}"
  print_command "tail -f ${latest_log}"
}

view_service_status() {
  print_section "Docker Service Status"

  print_command "docker-compose ps"
  docker-compose ps

  print_section "Container Details"
  echo -e "${CYAN}Running containers:${NC}"
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

  print_section "Network Status"
  echo -e "${CYAN}Testing port accessibility:${NC}"

  local ports=(8000 5432 6379 9090 3000)
  for port in "${ports[@]}"; do
    if timeout 2 bash -c "cat < /dev/null > /dev/tcp/localhost/${port}" 2>/dev/null; then
      print_success "Port ${port} is accessible"
    else
      print_info "Port ${port} is not accessible"
    fi
  done
}

docker_system_info() {
  print_section "Docker System Information"

  print_command "docker version --format 'Server Version: {{.Server.Version}}'"
  docker version --format 'Server Version: {{.Server.Version}}'

  print_section "Docker Disk Usage"
  print_command "docker system df"
  docker system df

  print_section "Docker Network Status"
  print_command "docker network ls"
  docker network ls
}

cleanup_old_logs() {
  print_section "Cleanup Old Logs"

  if [ ! -d "logs" ]; then
    print_info "No logs directory found"
    return
  fi

  local log_count=$(find logs -name 'docker-*.log' -type f 2>/dev/null | wc -l)

  if [ "$log_count" -eq 0 ]; then
    print_info "No logs to cleanup"
    return
  fi

  echo -e "${YELLOW}Found ${log_count} log files${NC}\n"

  echo -n "Keep logs from last N days (default 7): "
  read -r days
  days=${days:-7}

  local files_to_delete=$(find logs -name 'docker-*.log' -type f -mtime +${days} 2>/dev/null | wc -l)

  if [ "$files_to_delete" -eq 0 ]; then
    print_info "No logs older than ${days} days found"
    return
  fi

  echo -e "\n${YELLOW}Files to delete (older than ${days} days): ${files_to_delete}${NC}"
  find logs -name 'docker-*.log' -type f -mtime +${days} -exec ls -lh {} \;

  echo -n "Delete these files? (y/n): "
  read -r confirm

  if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    find logs -name 'docker-*.log' -type f -mtime +${days} -delete
    print_success "Cleanup completed"
  else
    print_info "Cleanup cancelled"
  fi
}

# ============================================================================
# MAIN LOOP
# ============================================================================

main() {
  cd "$(dirname "$0")" || exit

  # Make scripts executable
  chmod +x docker-health-check.sh 2>/dev/null || true

  while true; do
    show_menu
    read -r choice

    case $choice in
      1) run_basic_health_check ;;
      2) run_verbose_health_check ;;
      3) run_repair_health_check ;;
      4) run_initialization ;;
      5) run_full_setup ;;
      6) view_recent_logs ;;
      7) view_service_status ;;
      8) docker_system_info ;;
      9) cleanup_old_logs ;;
      0)
        echo -e "\n${GREEN}Goodbye!${NC}\n"
        exit 0
        ;;
      *)
        print_info "Invalid choice. Please try again."
        sleep 2
        ;;
    esac

    echo -e "\n${YELLOW}Press Enter to continue...${NC}"
    read -r
  done
}

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
  main "$@"
fi
