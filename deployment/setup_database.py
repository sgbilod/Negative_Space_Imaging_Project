#!/usr/bin/env python3
"""
Setup Database Integration Environment for Negative Space Imaging Project
Initializes the database deployment and integration environment.
"""

import os
import sys
import logging
import argparse
import subprocess
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("database_setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Setup Database Integration Environment"
    )

    parser.add_argument("--config", type=str,
                        default="deployment/config/database.yaml",
                        help="Path to database configuration file")
    parser.add_argument("--install-deps", action="store_true",
                        help="Install dependencies")
    parser.add_argument("--create-dirs", action="store_true",
                        help="Create directory structure")
    parser.add_argument("--init-db", action="store_true",
                        help="Initialize database")
    parser.add_argument("--test", action="store_true",
                        help="Run tests after setup")
    parser.add_argument("--all", action="store_true",
                        help="Perform all setup steps")
    parser.add_argument("--force", action="store_true",
                        help="Force installation even if components exist")

    return parser.parse_args()

def ensure_dirs_exist():
    """Ensure required directories exist"""
    logger.info("Creating directory structure...")

    dirs = [
        "deployment/config",
        "deployment/database",
        "deployment/database/migrations",
        "deployment/database/backups",
        "deployment/logs"
    ]

    for dir_path in dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

    return True

def check_postgres():
    """Check if PostgreSQL is installed and accessible"""
    logger.info("Checking PostgreSQL installation...")

    try:
        # Check if psql is available
        result = subprocess.run(
            ["psql", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            logger.info(f"PostgreSQL client found: {result.stdout.strip()}")
            return True
        else:
            logger.error("PostgreSQL client not found")
            return False
    except Exception as e:
        logger.error(f"Error checking PostgreSQL: {str(e)}")
        return False

def install_dependencies():
    """Install required Python dependencies"""
    logger.info("Installing Python dependencies...")

    required_packages = [
        "psycopg2-binary",
        "pyyaml",
        "psutil"
    ]

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade"] + required_packages,
            check=True
        )
        logger.info("Successfully installed Python dependencies")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {str(e)}")
        return False

def create_config_file(config_path):
    """Create a default configuration file if it doesn't exist"""
    logger.info(f"Creating default configuration file: {config_path}")

    config_dir = os.path.dirname(config_path)
    os.makedirs(config_dir, exist_ok=True)

    if os.path.exists(config_path) and not args.force:
        logger.info(f"Configuration file already exists: {config_path}")
        return True

    config_content = """# Database Configuration for Negative Space Imaging Project
# deployment/config/database.yaml

database:
  # Connection settings
  host: localhost
  port: 5432
  dbname: negative_space_imaging
  user: postgres
  password: postgres
  timeout: 30

  # Pool settings
  min_connections: 1
  max_connections: 10

  # Schema settings
  schema_file: "deployment/database/01-init-schema.sql"
  data_file: "deployment/database/02-init-data.sql"
  migrations_dir: "deployment/database/migrations"

  # Backup settings
  backup_dir: "deployment/database/backups"
  backup_retention_days: 30
  backup_schedule: "0 0 * * *"  # Daily at midnight (cron format)

  # Security settings
  ssl_mode: "prefer"
  verify_ca: false
  client_cert: ""
  client_key: ""
  ca_cert: ""

  # Monitoring settings
  enable_monitoring: true
  log_slow_queries: true
  slow_query_threshold_ms: 1000
  log_all_queries: false
"""

    try:
        with open(config_path, 'w') as f:
            f.write(config_content)
        logger.info(f"Created configuration file: {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create configuration file: {str(e)}")
        return False

def init_database(args):
    """Initialize the database"""
    logger.info("Initializing database...")

    # Run database deployment script
    try:
        cmd = [
            sys.executable,
            "deployment/database_deploy.py",
            "--deploy",
            "--config", args.config
        ]

        if args.force:
            cmd.append("--force")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            logger.info("Database initialization successful")
            return True
        else:
            logger.error(f"Database initialization failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        return False

def run_tests(args):
    """Run database tests"""
    logger.info("Running database tests...")

    try:
        cmd = [
            sys.executable,
            "deployment/test_database_deployment.py",
            "--config", args.config
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print(result.stdout)

        if result.returncode == 0:
            logger.info("Database tests passed")
            return True
        else:
            logger.error(f"Database tests failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Failed to run tests: {str(e)}")
        return False

def main():
    """Main function"""
    args = parse_args()

    steps = []

    if args.all or args.create_dirs:
        steps.append(("Create directories", ensure_dirs_exist))

    if args.all or args.install_deps:
        steps.append(("Check PostgreSQL", check_postgres))
        steps.append(("Install dependencies", install_dependencies))

    if args.all:
        steps.append(("Create config file", lambda: create_config_file(args.config)))

    if args.all or args.init_db:
        steps.append(("Initialize database", lambda: init_database(args)))

    if args.all or args.test:
        steps.append(("Run tests", lambda: run_tests(args)))

    if not steps:
        print("No actions specified. Use --help to see available options.")
        return 0

    # Execute steps
    print("\n=== Setting up Database Integration Environment ===\n")

    for step_name, step_func in steps:
        print(f"--- {step_name} ---")
        if step_func():
            print(f"✓ {step_name}: SUCCESS")
        else:
            print(f"✗ {step_name}: FAILED")
            print("\nSetup incomplete. Please check the log for details.")
            return 1
        print()

    print("=== Database Integration Environment Setup Complete ===")
    print("\nYou can now use the database deployment and integration tools:")
    print("- deployment/database_deploy.py: Deploy, migrate, backup, and restore the database")
    print("- deployment/hpc_database_integration.py: Integrate database with HPC computations")
    print("- deployment/test_database_deployment.py: Test database functionality")

    return 0

if __name__ == "__main__":
    args = parse_args()
    sys.exit(main())
