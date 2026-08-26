#!/usr/bin/env python
"""
System Management Tools Installation Script
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import os
import sys
import json
import shutil
from pathlib import Path

def setup_directories(project_root: Path):
    """Create required directories."""
    dirs = [
        "logs",
        "config",
        "reports",
        "data",
        "cache",
        "temp"
    ]

    for dir_name in dirs:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")

def setup_config(project_root: Path):
    """Set up configuration files."""
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)

    # Create orchestrator config if it doesn't exist
    orchestrator_config = config_dir / "orchestrator.json"
    if not orchestrator_config.exists():
        config = {
            "system": {
                "monitoring_interval": 300,
                "log_level": "INFO",
                "startup_validation_level": "COMPLETE",
                "monitoring_validation_level": "BASIC"
            },
            "security": {
                "require_multi_signature": True,
                "min_signatures": 3,
                "signature_threshold": 2,
                "encryption_algorithm": "AES-256-GCM",
                "key_rotation_days": 30
            },
            "performance": {
                "max_threads": 8,
                "gpu_enabled": True,
                "distributed_mode": False,
                "benchmark_interval": 3600
            },
            "logging": {
                "file_rotation": "1 day",
                "max_size": "100MB",
                "backup_count": 30,
                "compression": True
            },
            "paths": {
                "data_dir": "data",
                "cache_dir": "cache",
                "temp_dir": "temp",
                "log_dir": "logs"
            }
        }
        orchestrator_config.write_text(json.dumps(config, indent=2))
        print(f"Created configuration: {orchestrator_config}")

def setup_logging(project_root: Path):
    """Set up logging configuration."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create log files
    log_files = [
        "system.log",
        "monitoring.log",
        "validation.log",
        "performance.log"
    ]

    for log_file in log_files:
        log_path = log_dir / log_file
        if not log_path.exists():
            log_path.touch()
            print(f"Created log file: {log_path}")

def verify_dependencies():
    """Verify required Python packages."""
    required_packages = {
        "psutil": "5.8.0",
        "cryptography": "35.0.0",
        "pyyaml": "5.4.1"
    }

    import pkg_resources

    for package, version in required_packages.items():
        try:
            pkg_resources.require(f"{package}>={version}")
            print(f"✓ {package} >= {version}")
        except pkg_resources.VersionConflict:
            print(f"! {package} version conflict")
            sys.exit(1)
        except pkg_resources.DistributionNotFound:
            print(f"✗ {package} not found")
            sys.exit(1)

def main():
    """Main installation function."""
    project_root = Path(__file__).parent

    print("Setting up System Management Tools...")
    print("====================================")

    # Verify Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

    # Setup steps
    verify_dependencies()
    setup_directories(project_root)
    setup_config(project_root)
    setup_logging(project_root)

    print("\nInstallation completed successfully!")
    print("You can now run the system using:")
    print("python scripts/orchestrate_system.py")

if __name__ == '__main__':
    main()
