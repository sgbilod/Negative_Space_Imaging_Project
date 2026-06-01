#!/usr/bin/env python3
"""
Test script to validate setup_multi_node.py functions
"""
import os
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions from setup_multi_node.py
from deployment.setup_multi_node import (
    check_prerequisites,
    setup_database_files,
    setup_database_integration,
    setup_template_files,
    create_example_config,
    main
)

def test_functions():
    """Test each function in setup_multi_node.py"""
    print("Testing check_prerequisites...")
    assert check_prerequisites() == True
    print("check_prerequisites: PASSED")

    print("\nTesting setup_template_files...")
    assert setup_template_files(force_recreate=True) == True
    print("setup_template_files: PASSED")

    print("\nTesting setup_database_files...")
    assert setup_database_files() == True
    print("setup_database_files: PASSED")

    print("\nTesting setup_database_integration...")
    assert setup_database_integration(force_recreate=True) == True
    print("setup_database_integration: PASSED")

    print("\nTesting create_example_config...")
    assert create_example_config() == True
    print("create_example_config: PASSED")

    # Check that files were created
    template_dir = Path("deployment/templates")
    config_dir = Path("deployment/config")
    database_dir = Path("deployment/database")

    print("\nVerifying created files:")

    # Check template files
    template_files = list(template_dir.glob("*.tmpl"))
    print(f"Template files created: {len(template_files)}")
    assert len(template_files) >= 5

    # Check database files
    database_files = list(database_dir.glob("*.sql"))
    print(f"Database files created: {len(database_files)}")
    assert len(database_files) >= 2

    # Check config files
    config_files = list(config_dir.glob("*.yaml"))
    print(f"Config files created: {len(config_files)}")
    assert len(config_files) >= 2

    # Check example config specifically
    example_config_path = config_dir / "example-config.yaml"
    assert example_config_path.exists()
    print(f"Example config created: {example_config_path}")

    print("\nAll tests PASSED!")
    return 0

if __name__ == "__main__":
    sys.exit(test_functions())
