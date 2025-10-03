#!/usr/bin/env python3
"""
Database Deployment Manager for Negative Space Imaging Project
Handles database deployment, migrations, and verification.
"""

import os
import sys
import logging
import argparse
import yaml
import time
import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path to import database_connection
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deployment.database_connection import (
    setup_database,
    check_database_connection,
    init_db_pool,
    execute_query
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("database_deployment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Database Deployment Manager")

    # Main commands
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy database")
    parser.add_argument("--verify", action="store_true",
                        help="Verify database setup")
    parser.add_argument("--migrate", action="store_true",
                        help="Run database migrations")
    parser.add_argument("--backup", action="store_true",
                        help="Backup database")
    parser.add_argument("--restore", action="store_true",
                        help="Restore database from backup")

    # Configuration options
    parser.add_argument("--config", type=str, default="deployment/config/database.yaml",
                        help="Path to database configuration file")
    parser.add_argument("--schema", type=str,
                        default="deployment/database/01-init-schema.sql",
                        help="Path to schema SQL file")
    parser.add_argument("--data", type=str,
                        default="deployment/database/02-init-data.sql",
                        help="Path to data SQL file")
    parser.add_argument("--migrations", type=str,
                        default="deployment/database/migrations",
                        help="Path to migrations directory")
    parser.add_argument("--backup-dir", type=str,
                        default="deployment/database/backups",
                        help="Path to backup directory")
    parser.add_argument("--backup-file", type=str,
                        help="Specific backup file to restore (for --restore)")

    # Additional options
    parser.add_argument("--force", action="store_true",
                        help="Force operation even if validation fails")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout in seconds for operations")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    return parser.parse_args()

def deploy_database(args):
    """Deploy database schema and initial data"""
    logger.info("Starting database deployment")

    # Ensure directories exist
    ensure_directories(args)

    # Setup database
    success = setup_database(
        config_file=args.config,
        schema_file=args.schema,
        data_file=args.data
    )

    if not success:
        logger.error("Database deployment failed")
        return False

    # Run migrations if available
    if args.migrations and Path(args.migrations).exists():
        if not run_migrations(args):
            logger.warning("Some migrations failed, but deployment will continue")

    # Verify deployment
    if not verify_database(args) and not args.force:
        logger.error("Database verification failed")
        return False

    logger.info("Database deployment completed successfully")
    return True

def verify_database(args):
    """Verify database setup"""
    logger.info("Verifying database setup")

    # Check connection
    if not check_database_connection(args.config):
        logger.error("Database connection failed")
        return False

    # Initialize connection pool
    if not init_db_pool(args.config):
        logger.error("Failed to initialize connection pool")
        return False

    # Check required tables
    required_tables = [
        "users", "projects", "images", "computations",
        "security_logs", "system_events"
    ]

    for table in required_tables:
        # Check if table exists
        result, success = execute_query(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = %s)",
            (table,),
            fetch_one=True
        )

        if not success or not result or not result.get("exists", False):
            logger.error(f"Required table '{table}' not found")
            return False

        logger.info(f"Table '{table}' exists")

    # Check admin user
    result, success = execute_query(
        "SELECT COUNT(*) FROM users WHERE username = 'admin'",
        fetch_one=True
    )

    if not success or not result or result.get("count", 0) == 0:
        logger.error("Admin user not found")
        return False

    logger.info("Admin user exists")

    # Check default project
    result, success = execute_query(
        "SELECT COUNT(*) FROM projects WHERE name = 'Default Project'",
        fetch_one=True
    )

    if not success or not result or result.get("count", 0) == 0:
        logger.error("Default project not found")
        return False

    logger.info("Default project exists")

    logger.info("Database verification completed successfully")
    return True

def run_migrations(args):
    """Run database migrations"""
    logger.info("Running database migrations")

    migrations_dir = Path(args.migrations)
    if not migrations_dir.exists():
        logger.warning(f"Migrations directory not found: {args.migrations}")
        return False

    # Get list of migration files
    migration_files = sorted([
        f for f in migrations_dir.glob("*.sql")
        if f.name.startswith("V") and f.name.endswith(".sql")
    ])

    if not migration_files:
        logger.info("No migration files found")
        return True

    # Initialize connection pool
    if not init_db_pool(args.config):
        logger.error("Failed to initialize connection pool")
        return False

    # Create migrations table if it doesn't exist
    result, success = execute_query(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(50) PRIMARY KEY,
            applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            description TEXT,
            success BOOLEAN DEFAULT TRUE
        )
        """
    )

    if not success:
        logger.error("Failed to create migrations table")
        return False

    # Get applied migrations
    result, success = execute_query(
        "SELECT version FROM schema_migrations WHERE success = TRUE",
        fetch_all=True
    )

    if not success:
        logger.error("Failed to get applied migrations")
        return False

    applied_migrations = set(r.get("version", "") for r in result if r)

    # Run pending migrations
    all_succeeded = True
    for migration_file in migration_files:
        # Extract version and description from filename
        # Format: V{version}__{description}.sql
        # Example: V1.0.0__initial_schema.sql
        filename = migration_file.name
        version = filename.split("__")[0][1:]  # Remove V prefix
        description = filename.split("__")[1][:-4]  # Remove .sql suffix

        # Skip if already applied
        if version in applied_migrations:
            logger.info(f"Migration {version} already applied, skipping")
            continue

        logger.info(f"Applying migration {version}: {description}")

        # Read migration file
        try:
            with open(migration_file, 'r') as f:
                migration_sql = f.read()

            # Execute migration
            result, success = execute_query(migration_sql)

            if success:
                # Record successful migration
                result, success = execute_query(
                    """
                    INSERT INTO schema_migrations (version, description, success)
                    VALUES (%s, %s, TRUE)
                    """,
                    (version, description)
                )

                if success:
                    logger.info(f"Migration {version} applied successfully")
                else:
                    logger.error(f"Failed to record migration {version}")
                    all_succeeded = False
            else:
                # Record failed migration
                result, success = execute_query(
                    """
                    INSERT INTO schema_migrations (version, description, success)
                    VALUES (%s, %s, FALSE)
                    """,
                    (version, description)
                )

                logger.error(f"Migration {version} failed")
                all_succeeded = False

                if not args.force:
                    logger.error("Aborting migrations due to failure")
                    return False
        except Exception as e:
            logger.error(f"Error applying migration {version}: {str(e)}")
            all_succeeded = False

            if not args.force:
                logger.error("Aborting migrations due to error")
                return False

    logger.info("Database migrations completed")
    return all_succeeded

def backup_database(args):
    """Backup database to file"""
    logger.info("Starting database backup")

    # Ensure backup directory exists
    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {args.config}")
        return False

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        db_config = config.get("database", {})
        dbname = db_config.get("dbname", "negative_space_imaging")
        user = db_config.get("user", "postgres")
        host = db_config.get("host", "localhost")
        port = db_config.get("port", 5432)

        # Generate backup filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{dbname}_backup_{timestamp}.sql"

        # Use pg_dump to create backup
        from subprocess import run, PIPE

        env = os.environ.copy()
        if "password" in db_config:
            env["PGPASSWORD"] = db_config["password"]

        cmd = [
            "pg_dump",
            f"--host={host}",
            f"--port={port}",
            f"--username={user}",
            "--format=c",  # Custom format (compressed)
            f"--file={backup_file}",
            dbname
        ]

        logger.info(f"Running pg_dump to create backup: {backup_file}")
        result = run(cmd, env=env, stdout=PIPE, stderr=PIPE, text=True)

        if result.returncode != 0:
            logger.error(f"Database backup failed: {result.stderr}")
            return False

        logger.info(f"Database backup completed successfully: {backup_file}")
        return True
    except Exception as e:
        logger.error(f"Database backup failed: {str(e)}")
        return False

def restore_database(args):
    """Restore database from backup file"""
    logger.info("Starting database restore")

    # Determine backup file
    backup_file = args.backup_file
    if not backup_file:
        # Use latest backup
        backup_dir = Path(args.backup_dir)
        if not backup_dir.exists():
            logger.error(f"Backup directory not found: {args.backup_dir}")
            return False

        backup_files = sorted(backup_dir.glob("*_backup_*.sql"), reverse=True)
        if not backup_files:
            logger.error("No backup files found")
            return False

        backup_file = str(backup_files[0])
        logger.info(f"Using latest backup file: {backup_file}")

    if not Path(backup_file).exists():
        logger.error(f"Backup file not found: {backup_file}")
        return False

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {args.config}")
        return False

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        db_config = config.get("database", {})
        dbname = db_config.get("dbname", "negative_space_imaging")
        user = db_config.get("user", "postgres")
        host = db_config.get("host", "localhost")
        port = db_config.get("port", 5432)

        # Use pg_restore to restore backup
        from subprocess import run, PIPE

        env = os.environ.copy()
        if "password" in db_config:
            env["PGPASSWORD"] = db_config["password"]

        # Drop existing database
        if args.force:
            logger.warning(f"Dropping existing database: {dbname}")

            # Connect to postgres database to drop the target database
            drop_cmd = [
                "psql",
                f"--host={host}",
                f"--port={port}",
                f"--username={user}",
                "--dbname=postgres",
                "-c", f"DROP DATABASE IF EXISTS {dbname}"
            ]

            result = run(drop_cmd, env=env, stdout=PIPE, stderr=PIPE, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to drop database: {result.stderr}")
                return False

            # Create fresh database
            create_cmd = [
                "psql",
                f"--host={host}",
                f"--port={port}",
                f"--username={user}",
                "--dbname=postgres",
                "-c", f"CREATE DATABASE {dbname}"
            ]

            result = run(create_cmd, env=env, stdout=PIPE, stderr=PIPE, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to create database: {result.stderr}")
                return False

        # Restore from backup
        restore_cmd = [
            "pg_restore",
            f"--host={host}",
            f"--port={port}",
            f"--username={user}",
            "--clean",  # Clean (drop) database objects before recreating
            "--if-exists",  # Don't error if objects don't exist
            "--no-owner",  # Don't include ownership in restore
            "--dbname", dbname,
            backup_file
        ]

        logger.info(f"Running pg_restore to restore database from: {backup_file}")
        result = run(restore_cmd, env=env, stdout=PIPE, stderr=PIPE, text=True)

        # Note: pg_restore might return non-zero even on successful restore
        # with some warnings, so we check stderr for serious errors
        if result.returncode != 0 and "error:" in result.stderr.lower():
            logger.error(f"Database restore failed: {result.stderr}")
            return False

        logger.info("Database restore completed successfully")
        return True
    except Exception as e:
        logger.error(f"Database restore failed: {str(e)}")
        return False

def ensure_directories(args):
    """Ensure required directories exist"""
    dirs = [
        Path(os.path.dirname(args.config)),
        Path(os.path.dirname(args.schema)),
        Path(args.migrations),
        Path(args.backup_dir)
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

def main():
    """Main function"""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Execute command
    if args.deploy:
        success = deploy_database(args)
    elif args.verify:
        success = verify_database(args)
    elif args.migrate:
        success = run_migrations(args)
    elif args.backup:
        success = backup_database(args)
    elif args.restore:
        success = restore_database(args)
    else:
        print("No command specified. Use --help for available commands.")
        return 0

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
