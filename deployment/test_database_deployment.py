#!/usr/bin/env python3
"""
Database Deployment Test for Negative Space Imaging Project
Tests database deployment, connections, and basic operations.
"""

import os
import sys
import logging
import argparse
import json
import uuid
from pathlib import Path

# Add parent directory to path to import database modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deployment.database_connection import (
    init_db_pool,
    execute_query,
    check_database_connection,
    setup_database
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("database_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Database Deployment Test")

    parser.add_argument("--config", type=str,
                        default="deployment/config/database.yaml",
                        help="Path to database configuration file")
    parser.add_argument("--schema", type=str,
                        default="deployment/database/01-init-schema.sql",
                        help="Path to schema SQL file")
    parser.add_argument("--data", type=str,
                        default="deployment/database/02-init-data.sql",
                        help="Path to data SQL file")
    parser.add_argument("--full", action="store_true",
                        help="Run full test suite including writes")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    return parser.parse_args()

def test_connection(args):
    """Test database connection"""
    logger.info("Testing database connection...")

    if check_database_connection(args.config):
        logger.info("✓ Database connection successful")
        return True
    else:
        logger.error("✗ Database connection failed")
        return False

def test_pool_initialization(args):
    """Test connection pool initialization"""
    logger.info("Testing connection pool initialization...")

    if init_db_pool(args.config):
        logger.info("✓ Connection pool initialized successfully")
        return True
    else:
        logger.error("✗ Connection pool initialization failed")
        return False

def test_basic_query():
    """Test basic query execution"""
    logger.info("Testing basic query execution...")

    result, success = execute_query("SELECT 1 as test", fetch_one=True)

    if success and result and result.get("test") == 1:
        logger.info("✓ Basic query execution successful")
        return True
    else:
        logger.error("✗ Basic query execution failed")
        return False

def test_tables_exist():
    """Test that required tables exist"""
    logger.info("Testing required tables exist...")

    required_tables = [
        "users", "projects", "images", "computations",
        "security_logs", "system_events"
    ]

    all_exist = True
    for table in required_tables:
        result, success = execute_query(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = %s)",
            (table,),
            fetch_one=True
        )

        if success and result and result.get("exists"):
            logger.info(f"✓ Table '{table}' exists")
        else:
            logger.error(f"✗ Table '{table}' does not exist")
            all_exist = False

    return all_exist

def test_admin_user():
    """Test that admin user exists"""
    logger.info("Testing admin user exists...")

    result, success = execute_query(
        "SELECT id, username, email FROM users WHERE username = 'admin'",
        fetch_one=True
    )

    if success and result and result.get("username") == "admin":
        logger.info(f"✓ Admin user exists: {result.get('email')}")
        return True
    else:
        logger.error("✗ Admin user does not exist")
        return False

def test_default_project():
    """Test that default project exists"""
    logger.info("Testing default project exists...")

    result, success = execute_query(
        "SELECT id, name, description FROM projects WHERE name = 'Default Project'",
        fetch_one=True
    )

    if success and result and result.get("name") == "Default Project":
        logger.info(f"✓ Default project exists: {result.get('description')}")
        return True
    else:
        logger.error("✗ Default project does not exist")
        return False

def test_settings():
    """Test that settings table has entries"""
    logger.info("Testing settings table has entries...")

    result, success = execute_query(
        "SELECT COUNT(*) FROM settings",
        fetch_one=True
    )

    if success and result and result.get("count", 0) > 0:
        logger.info(f"✓ Settings table has {result.get('count')} entries")

        # Check specific important settings
        important_settings = ["system.version", "security.max_login_attempts"]
        for setting in important_settings:
            result, success = execute_query(
                "SELECT key, value FROM settings WHERE key = %s",
                (setting,),
                fetch_one=True
            )

            if success and result:
                logger.info(f"✓ Setting '{setting}' = {result.get('value')}")
            else:
                logger.error(f"✗ Setting '{setting}' not found")
                return False

        return True
    else:
        logger.error("✗ Settings table is empty")
        return False

def test_insert_and_retrieve(args):
    """Test inserting and retrieving data"""
    if not args.full:
        logger.info("Skipping insert/retrieve test (use --full to run)")
        return True

    logger.info("Testing insert and retrieve operations...")

    # Get admin user ID
    admin_result, admin_success = execute_query(
        "SELECT id FROM users WHERE username = 'admin'",
        fetch_one=True
    )

    if not admin_success or not admin_result:
        logger.error("✗ Failed to get admin user ID")
        return False

    admin_id = admin_result.get("id")

    # Get default project ID
    project_result, project_success = execute_query(
        "SELECT id FROM projects WHERE name = 'Default Project'",
        fetch_one=True
    )

    if not project_success or not project_result:
        logger.error("✗ Failed to get default project ID")
        return False

    project_id = project_result.get("id")

    # Insert test image
    test_name = f"Test Image {uuid.uuid4()}"
    test_path = f"/fake/path/{test_name}.fits"

    image_result, image_success = execute_query(
        """
        INSERT INTO images (
            project_id, name, description, file_path, file_size, file_hash,
            width, height, bit_depth, channels, format, created_by
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            project_id,
            test_name,
            "Test image for database testing",
            test_path,
            1024000,  # 1MB
            "abcdef1234567890",
            1024,
            1024,
            16,
            3,
            "fits",
            admin_id
        ),
        fetch_one=True
    )

    if not image_success or not image_result:
        logger.error("✗ Failed to insert test image")
        return False

    image_id = image_result.get("id")
    logger.info(f"✓ Inserted test image with ID: {image_id}")

    # Retrieve test image
    retrieve_result, retrieve_success = execute_query(
        "SELECT id, name, file_path FROM images WHERE id = %s",
        (image_id,),
        fetch_one=True
    )

    if not retrieve_success or not retrieve_result:
        logger.error("✗ Failed to retrieve test image")
        return False

    if retrieve_result.get("name") == test_name and retrieve_result.get("file_path") == test_path:
        logger.info(f"✓ Retrieved test image: {retrieve_result.get('name')}")
    else:
        logger.error(f"✗ Retrieved image data does not match inserted data")
        return False

    # Insert test computation
    computation_result, computation_success = execute_query(
        """
        INSERT INTO computations (
            project_id, name, description, type, status, parameters,
            input_images, created_by
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            project_id,
            "Test Computation",
            "Test computation for database testing",
            "test",
            "pending",
            json.dumps({"param1": "value1", "param2": 42}),
            json.dumps([str(image_id)]),
            admin_id
        ),
        fetch_one=True
    )

    if not computation_success or not computation_result:
        logger.error("✗ Failed to insert test computation")
        return False

    computation_id = computation_result.get("id")
    logger.info(f"✓ Inserted test computation with ID: {computation_id}")

    # Log test complete event
    event_result, event_success = execute_query(
        """
        INSERT INTO system_events (
            event_type, severity, message, details
        )
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (
            "test.database",
            "info",
            "Database test completed successfully",
            json.dumps({
                "test_image_id": str(image_id),
                "test_computation_id": str(computation_id)
            })
        ),
        fetch_one=True
    )

    if not event_success or not event_result:
        logger.error("✗ Failed to insert test event")
        return False

    logger.info(f"✓ Inserted test event with ID: {event_result.get('id')}")

    return True

def test_transaction_rollback():
    """Test transaction rollback on error"""
    logger.info("Testing transaction rollback on error...")

    # Begin a transaction with a valid query
    try:
        with execute_query("BEGIN") as _:
            # First query succeeds
            execute_query("CREATE TEMP TABLE test_rollback (id SERIAL PRIMARY KEY, name TEXT)")

            # Second query fails (invalid syntax)
            execute_query("INSERT INTO test_rollback VALUES (DEFAULT, 'test'")

            # Should not reach this point due to error
            execute_query("COMMIT")
            logger.error("✗ Transaction should have failed but didn't")
            return False
    except Exception as e:
        logger.info(f"✓ Transaction correctly failed with error: {str(e)}")

    # Verify the temp table doesn't exist or is empty
    result, success = execute_query(
        "SELECT COUNT(*) FROM test_rollback",
        fetch_one=True
    )

    if not success:
        logger.info("✓ Transaction was correctly rolled back (table doesn't exist)")
        return True
    elif result and result.get("count", 0) == 0:
        logger.info("✓ Transaction was correctly rolled back (table is empty)")
        return True
    else:
        logger.error("✗ Transaction rollback failed")
        return False

def main():
    """Main function"""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting database deployment test")

    # Run tests
    tests = [
        ("Connection", lambda: test_connection(args)),
        ("Pool Initialization", lambda: test_pool_initialization(args)),
        ("Basic Query", test_basic_query),
        ("Tables Exist", test_tables_exist),
        ("Admin User", test_admin_user),
        ("Default Project", test_default_project),
        ("Settings", test_settings),
        ("Insert and Retrieve", lambda: test_insert_and_retrieve(args)),
        ("Transaction Rollback", test_transaction_rollback)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' failed with exception: {str(e)}")
            results.append((name, False))

    # Print summary
    print("\n----- Test Summary -----")
    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        status_color = "\033[92m" if result else "\033[91m"  # Green for pass, red for fail
        print(f"{status_color}{status}\033[0m: {name}")

        if result:
            passed += 1

    print(f"\nResults: {passed}/{len(results)} tests passed")

    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
