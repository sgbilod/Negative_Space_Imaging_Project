#!/bin/bash
# ============================================================================
# HPC Setup Script for Linux/macOS
# Negative Space Imaging Project
# Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
# ============================================================================
# This script sets up the HPC environment for the Negative Space Imaging Project.
# It installs dependencies, configures the environment, and validates the setup.
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
PYTHON_MIN_VERSION="3.9"
VENV_DIR="${SCRIPT_DIR}/.venv-hpc"
LOG_FILE="${SCRIPT_DIR}/hpc_setup.log"

# Functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
    echo "[SUCCESS] $1" >> "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    echo "[WARNING] $1" >> "$LOG_FILE"
}

error() {
    echo -e "${RED}✗${NC} $1"
    echo "[ERROR] $1" >> "$LOG_FILE"
}

check_python() {
    log "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        error "Python not found. Please install Python ${PYTHON_MIN_VERSION}+"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    
    if [[ $(echo "$PYTHON_VERSION >= $PYTHON_MIN_VERSION" | bc -l) -eq 1 ]]; then
        success "Python $PYTHON_VERSION found"
    else
        error "Python $PYTHON_MIN_VERSION+ required, found $PYTHON_VERSION"
        exit 1
    fi
}

check_cuda() {
    log "Checking CUDA availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        success "CUDA available (Driver: $CUDA_VERSION)"
        HAS_CUDA=true
    else
        warning "CUDA not available. GPU acceleration will be disabled."
        HAS_CUDA=false
    fi
}

check_mpi() {
    log "Checking MPI availability..."
    
    if command -v mpirun &> /dev/null; then
        MPI_VERSION=$(mpirun --version 2>&1 | head -1)
        success "MPI available: $MPI_VERSION"
        HAS_MPI=true
    elif command -v srun &> /dev/null; then
        success "SLURM MPI available"
        HAS_MPI=true
    else
        warning "MPI not available. Distributed computing features may be limited."
        HAS_MPI=false
    fi
}

create_virtual_env() {
    log "Creating virtual environment..."
    
    if [[ -d "$VENV_DIR" ]]; then
        warning "Virtual environment already exists at $VENV_DIR"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            success "Using existing virtual environment"
            return
        fi
    fi
    
    $PYTHON_CMD -m venv "$VENV_DIR"
    success "Virtual environment created at $VENV_DIR"
}

activate_venv() {
    log "Activating virtual environment..."
    
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        source "$VENV_DIR/bin/activate"
        success "Virtual environment activated"
    else
        error "Virtual environment not found"
        exit 1
    fi
}

install_base_requirements() {
    log "Installing base requirements..."
    
    pip install --upgrade pip setuptools wheel
    
    if [[ -f "${SCRIPT_DIR}/requirements.txt" ]]; then
        pip install -r "${SCRIPT_DIR}/requirements.txt"
        success "Base requirements installed"
    else
        warning "requirements.txt not found, skipping base requirements"
    fi
}

install_hpc_requirements() {
    log "Installing HPC requirements..."
    
    # Core HPC packages
    pip install \
        dask[distributed]>=2023.1.0 \
        ray>=2.5.0 \
        aiohttp>=3.8.0 \
        pyarrow>=12.0.0
    
    # MPI support (if available)
    if [[ "$HAS_MPI" == true ]]; then
        pip install mpi4py>=3.1.0
        success "MPI Python bindings installed"
    fi
    
    # CUDA support (if available)
    if [[ "$HAS_CUDA" == true ]]; then
        pip install cupy-cuda12x || warning "Failed to install CuPy, continuing..."
        success "CUDA Python packages installed"
    fi
    
    # Optional but recommended
    pip install \
        prefect>=2.10.0 \
        prometheus-client>=0.17.0 \
        psutil>=5.9.0
    
    success "HPC requirements installed"
}

configure_environment() {
    log "Configuring HPC environment..."
    
    # Create HPC config directory
    HPC_CONFIG_DIR="${SCRIPT_DIR}/config/hpc"
    mkdir -p "$HPC_CONFIG_DIR"
    
    # Generate environment file
    cat > "${SCRIPT_DIR}/.env.hpc" << ENVFILE
# HPC Environment Configuration
# Generated by setup_hpc.sh on $(date)

# Cluster settings
HPC_BACKEND=auto
HPC_CLUSTER_NAME=negative-space-hpc
HPC_ENVIRONMENT=development

# Resource defaults
HPC_DEFAULT_CPUS=4
HPC_DEFAULT_MEMORY_GB=16
HPC_DEFAULT_WALL_TIME=04:00:00

# GPU settings
HPC_GPUS_ENABLED=${HAS_CUDA}

# MPI settings
HPC_MPI_ENABLED=${HAS_MPI}

# Logging
HPC_LOG_LEVEL=INFO

# Paths
HPC_SCRATCH_PATH=/tmp/nsi_scratch
HPC_OUTPUT_PATH=${SCRIPT_DIR}/output
ENVFILE
    
    success "Environment configuration created at .env.hpc"
}

validate_installation() {
    log "Validating HPC installation..."
    
    # Run validation script
    if [[ -f "${SCRIPT_DIR}/install_hpc_requirements.py" ]]; then
        python "${SCRIPT_DIR}/install_hpc_requirements.py" --validate
    fi
    
    # Test imports
    python -c "
import sys
try:
    import dask
    import ray
    import aiohttp
    import pyarrow
    print('Core HPC packages: OK')
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
"
    
    success "HPC installation validated"
}

print_summary() {
    echo
    echo "=============================================="
    echo "HPC Setup Complete"
    echo "=============================================="
    echo
    echo "Virtual Environment: $VENV_DIR"
    echo "CUDA Support: $HAS_CUDA"
    echo "MPI Support: $HAS_MPI"
    echo
    echo "To activate the HPC environment:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  source .env.hpc"
    echo
    echo "To run HPC examples:"
    echo "  python hpc_integration_example.py"
    echo
}

# Main execution
main() {
    echo "=============================================="
    echo "Negative Space Imaging Project"
    echo "HPC Setup Script"
    echo "=============================================="
    echo
    
    # Initialize log file
    echo "HPC Setup Log - $(date)" > "$LOG_FILE"
    
    # Run setup steps
    check_python
    check_cuda
    check_mpi
    create_virtual_env
    activate_venv
    install_base_requirements
    install_hpc_requirements
    configure_environment
    validate_installation
    print_summary
    
    log "Setup completed successfully!"
}

# Run main function
main "$@"
