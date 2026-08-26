#!/bin/bash

################################################################################
# DOCKER SCRIPTS VERIFICATION SCRIPT
#
# Verifies that all Docker health check and initialization scripts are
# properly installed and functional.
#
# Usage: ./verify-installation.sh [--fix]
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

# Counters
PASSED=0
FAILED=0
WARNINGS=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

check() {
  local description="$1"
  local command="$2"

  if eval "$command" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} $description"
    ((PASSED++))
    return 0
  else
    echo -e "${RED}✗${NC} $description"
    ((FAILED++))
    return 1
  fi
}

warn() {
  echo -e "${YELLOW}⚠${NC} $1"
  ((WARNINGS++))
}

info() {
  echo -e "${BLUE}ℹ${NC} $1"
}

print_header() {
  echo ""
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${CYAN}$1${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
}

# ============================================================================
# VERIFICATION CHECKS
# ============================================================================

verify_files() {
  print_header "File Existence Checks"

  check "docker-health-check.sh exists" "[ -f '$SCRIPT_DIR/docker-health-check.sh' ]"
  check "docker-init.js exists" "[ -f '$SCRIPT_DIR/docker-init.js' ]"
  check "docker-health-check.ps1 exists" "[ -f '$SCRIPT_DIR/docker-health-check.ps1' ]"
  check "docker-quick-start.sh exists" "[ -f '$SCRIPT_DIR/docker-quick-start.sh' ]"
  check "DOCKER_HEALTH_CHECK_README.md exists" "[ -f '$SCRIPT_DIR/DOCKER_HEALTH_CHECK_README.md' ]"
  check "IMPLEMENTATION_GUIDE.md exists" "[ -f '$SCRIPT_DIR/IMPLEMENTATION_GUIDE.md' ]"
}

verify_permissions() {
  print_header "File Permissions Checks"

  check "docker-health-check.sh is executable" "[ -x '$SCRIPT_DIR/docker-health-check.sh' ]"
  check "docker-quick-start.sh is executable" "[ -x '$SCRIPT_DIR/docker-quick-start.sh' ]"
}

verify_file_sizes() {
  print_header "File Size Checks"

  local bash_size=$(wc -l < "$SCRIPT_DIR/docker-health-check.sh" 2>/dev/null || echo "0")
  local node_size=$(wc -l < "$SCRIPT_DIR/docker-init.js" 2>/dev/null || echo "0")

  if [ "$bash_size" -ge 600 ]; then
    echo -e "${GREEN}✓${NC} docker-health-check.sh has $bash_size lines (expected 650+)"
    ((PASSED++))
  else
    echo -e "${RED}✗${NC} docker-health-check.sh has only $bash_size lines (expected 650+)"
    ((FAILED++))
  fi

  if [ "$node_size" -ge 500 ]; then
    echo -e "${GREEN}✓${NC} docker-init.js has $node_size lines (expected 550+)"
    ((PASSED++))
  else
    echo -e "${RED}✗${NC} docker-init.js has only $node_size lines (expected 550+)"
    ((FAILED++))
  fi
}

verify_bash_syntax() {
  print_header "Bash Syntax Checks"

  if bash -n "$SCRIPT_DIR/docker-health-check.sh" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} docker-health-check.sh syntax is valid"
    ((PASSED++))
  else
    echo -e "${RED}✗${NC} docker-health-check.sh has syntax errors"
    ((FAILED++))
  fi

  if bash -n "$SCRIPT_DIR/docker-quick-start.sh" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} docker-quick-start.sh syntax is valid"
    ((PASSED++))
  else
    echo -e "${RED}✗${NC} docker-quick-start.sh has syntax errors"
    ((FAILED++))
  fi
}

verify_node_syntax() {
  print_header "Node.js Syntax Checks"

  if command -v node >/dev/null 2>&1; then
    if node -c "$SCRIPT_DIR/docker-init.js" >/dev/null 2>&1; then
      echo -e "${GREEN}✓${NC} docker-init.js syntax is valid"
      ((PASSED++))
    else
      echo -e "${RED}✗${NC} docker-init.js has syntax errors"
      ((FAILED++))
    fi
  else
    warn "Node.js not installed - skipping syntax check for docker-init.js"
  fi
}

verify_shell_commands() {
  print_header "Shell Commands Availability"

  check "bash is available" "command -v bash"
  check "docker command available" "command -v docker"
  check "grep command available" "command -v grep"
  check "awk command available" "command -v awk"
}

verify_optional_commands() {
  print_header "Optional Commands"

  if command -v node >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Node.js is installed ($(node --version))"
    ((PASSED++))
  else
    warn "Node.js not installed - some features may be limited"
  fi

  if command -v npm >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} npm is installed ($(npm --version))"
    ((PASSED++))
  else
    warn "npm not installed - cannot auto-install dependencies"
  fi

  if command -v pwsh >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} PowerShell is available"
    ((PASSED++))
  else
    warn "PowerShell not available - Windows scripts will be limited"
  fi
}

verify_docker_compose() {
  print_header "Docker Compose Configuration"

  check "docker-compose.yml exists" "[ -f '$PROJECT_ROOT/docker-compose.yml' ]"

  if command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} docker-compose is available"
    ((PASSED++))
  else
    warn "docker-compose command not found - health checks may fail"
  fi
}

verify_documentation() {
  print_header "Documentation"

  if grep -q "docker-health-check.sh" "$SCRIPT_DIR/DOCKER_HEALTH_CHECK_README.md" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} README contains docker-health-check.sh documentation"
    ((PASSED++))
  else
    echo -e "${RED}✗${NC} README missing docker-health-check.sh documentation"
    ((FAILED++))
  fi

  if grep -q "docker-init.js" "$SCRIPT_DIR/DOCKER_HEALTH_CHECK_README.md" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} README contains docker-init.js documentation"
    ((PASSED++))
  else
    echo -e "${RED}✗${NC} README missing docker-init.js documentation"
    ((FAILED++))
  fi
}

verify_environment() {
  print_header "Environment Configuration"

  if [ -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${GREEN}✓${NC} .env file exists"
    ((PASSED++))

    if grep -q "DATABASE_URL\|DB_HOST" "$PROJECT_ROOT/.env" 2>/dev/null; then
      echo -e "${GREEN}✓${NC} Database configuration found in .env"
      ((PASSED++))
    else
      warn ".env file missing database configuration"
    fi
  else
    warn ".env file not found - you may need to create it"
  fi

  if [ -f "$PROJECT_ROOT/.env.example" ]; then
    echo -e "${GREEN}✓${NC} .env.example template exists"
    ((PASSED++))
  else
    warn ".env.example template not found"
  fi
}

verify_logs_dir() {
  print_header "Logs Directory"

  if [ -d "$SCRIPT_DIR/logs" ]; then
    echo -e "${GREEN}✓${NC} logs directory exists"
    ((PASSED++))

    local log_count=$(find "$SCRIPT_DIR/logs" -name "docker-*.log" 2>/dev/null | wc -l)
    if [ "$log_count" -gt 0 ]; then
      echo -e "${GREEN}✓${NC} Found $log_count existing log files"
      ((PASSED++))
    else
      info "No existing logs - will be created on first run"
    fi
  else
    info "logs directory will be created on first run"
  fi
}

# ============================================================================
# FIX FUNCTIONS
# ============================================================================

fix_permissions() {
  echo -e "\n${YELLOW}Fixing file permissions...${NC}\n"

  chmod +x "$SCRIPT_DIR/docker-health-check.sh" 2>/dev/null && \
    echo -e "${GREEN}✓${NC} Fixed permissions for docker-health-check.sh" || \
    echo -e "${RED}✗${NC} Failed to fix permissions for docker-health-check.sh"

  chmod +x "$SCRIPT_DIR/docker-quick-start.sh" 2>/dev/null && \
    echo -e "${GREEN}✓${NC} Fixed permissions for docker-quick-start.sh" || \
    echo -e "${RED}✗${NC} Failed to fix permissions for docker-quick-start.sh"

  chmod +x "$SCRIPT_DIR/verify-installation.sh" 2>/dev/null && \
    echo -e "${GREEN}✓${NC} Fixed permissions for verify-installation.sh" || \
    echo -e "${RED}✗${NC} Failed to fix permissions for verify-installation.sh"
}

create_logs_dir() {
  if [ ! -d "$SCRIPT_DIR/logs" ]; then
    mkdir -p "$SCRIPT_DIR/logs" && \
      echo -e "${GREEN}✓${NC} Created logs directory" || \
      echo -e "${RED}✗${NC} Failed to create logs directory"
  fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
  clear

  echo -e "${CYAN}"
  cat << "EOF"
╔════════════════════════════════════════════════════════════════════════╗
║              Docker Scripts Installation Verification                 ║
║                Negative Space Imaging Project                          ║
╚════════════════════════════════════════════════════════════════════════╝
EOF
  echo -e "${NC}"

  info "Script Directory: $SCRIPT_DIR"
  info "Project Root: $PROJECT_ROOT"
  echo ""

  # Run all verification checks
  verify_files
  verify_permissions
  verify_file_sizes
  verify_bash_syntax
  verify_node_syntax
  verify_shell_commands
  verify_optional_commands
  verify_docker_compose
  verify_documentation
  verify_environment
  verify_logs_dir

  # Summary
  print_header "Verification Summary"

  local total=$((PASSED + FAILED + WARNINGS))

  echo -e "${GREEN}Passed:${NC}   $PASSED"
  echo -e "${RED}Failed:${NC}   $FAILED"
  echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
  echo -e "${BLUE}Total:${NC}    $total"
  echo ""

  # Handle --fix flag
  if [[ "$1" == "--fix" ]]; then
    fix_permissions
    create_logs_dir
  elif [ "$FAILED" -gt 0 ]; then
    echo -e "${YELLOW}Some issues were found. Run with --fix flag to attempt repairs:${NC}"
    echo -e "${CYAN}./verify-installation.sh --fix${NC}"
    echo ""
  fi

  # Final status
  echo ""
  if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✓ Installation verification PASSED${NC}"
    echo ""
    echo "Next steps:"
    echo -e "  ${CYAN}./docker-quick-start.sh${NC}            # Interactive menu"
    echo -e "  ${CYAN}./docker-health-check.sh${NC}          # Quick health check"
    echo -e "  ${CYAN}node docker-init.js --verbose${NC}     # Full initialization"
    echo ""
    return 0
  else
    echo -e "${RED}✗ Installation verification FAILED${NC}"
    echo ""
    echo "Issues found. Please review the output above."
    echo "If using --fix flag didn't resolve issues, check:"
    echo "  1. File permissions are correct"
    echo "  2. Docker is installed and running"
    echo "  3. All dependencies are available"
    echo ""
    return 1
  fi
}

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
