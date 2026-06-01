#!/usr/bin/env python3
"""
Database Connection Utility for Negative Space Imaging Project
Provides functions for connecting to and interacting with the database.
"""

import os
import sys
import logging
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("database.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global connection pool
connection_pool = None

def init_db_pool(config_file: Optional[str] = None,
                min_connections: int = 1,
                max_connections: int = 10) -> bool:
    """
    Initialize the database connection pool

    Args:
        config_file: Path to the database configuration file
        min_connections: Minimum number of connections in the pool
        max_connections: Maximum number of connections in the pool

    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global connection_pool

    # If pool already exists, close it
    if connection_pool:
        logger.info("Closing existing connection pool")
        connection_pool.closeall()
        connection_pool = None

    # Load configuration
    db_config = load_db_config(config_file)
    if not db_config:
        return False

    try:
        # Create connection pool
        logger.info(f"Creating connection pool with {min_connections} to {max_connections} connections")
        connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=min_connections,
            maxconn=max_connections,
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 5432),
            dbname=db_config.get("dbname", "negative_space_imaging"),
            user=db_config.get("user", "postgres"),
            password=db_config.get("password", ""),
            connect_timeout=db_config.get("connect_timeout", 10)
        )
        logger.info("Database connection pool initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {str(e)}")
        return False

def load_db_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load database configuration from file or environment variables

    Args:
        config_file: Path to the configuration file (optional)

    Returns:
        Dict[str, Any]: Dictionary with database configuration
    """
    # Default configuration
    db_config = {
        "host": "localhost",
        "port": 5432,
        "dbname": "negative_space_imaging",
        "user": "postgres",
        "password": "",
        "connect_timeout": 10
    }

    # Try to load from file
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)

                if "database" in yaml_config:
                    db_section = yaml_config["database"]
                    for key in db_config.keys():
                        if key in db_section:
                            db_config[key] = db_section[key]

                    logger.info(f"Loaded database configuration from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load database configuration from {config_file}: {str(e)}")

    # Override with environment variables
    env_prefix = "NSIP_DB_"
    for key in db_config.keys():
        env_var = f"{env_prefix}{key.upper()}"
        if env_var in os.environ:
            db_config[key] = os.environ[env_var]

    # Mask password for logging
    log_config = db_config.copy()
    if "password" in log_config and log_config["password"]:
        log_config["password"] = "********"

    logger.info(f"Using database configuration: {log_config}")
    return db_config

def get_db_connection() -> Tuple[Optional[Any], bool]:
    """
    Get a database connection from the pool

    Returns:
        Tuple[Optional[Any], bool]: Tuple with connection object and success flag
    """
    global connection_pool

    if not connection_pool:
        logger.error("Connection pool not initialized")
        return None, False

    try:
        connection = connection_pool.getconn()
        return connection, True
    except Exception as e:
        logger.error(f"Failed to get database connection: {str(e)}")
        return None, False

def release_db_connection(connection: Any) -> bool:
    """
    Release a connection back to the pool

    Args:
        connection: Connection object to release

    Returns:
        bool: True if release was successful, False otherwise
    """
    global connection_pool

    if not connection_pool:
        logger.error("Connection pool not initialized")
        return False

    try:
        connection_pool.putconn(connection)
        return True
    except Exception as e:
        logger.error(f"Failed to release database connection: {str(e)}")
        return False

def execute_query(query: str, params: Optional[Union[Dict[str, Any], List[Any], Tuple[Any]]] = None,
                  fetch_one: bool = False, fetch_all: bool = True,
                  as_dict: bool = True) -> Tuple[Any, bool]:
    """
    Execute a database query and fetch results

    Args:
        query: SQL query to execute
        params: Query parameters
        fetch_one: Whether to fetch one result
        fetch_all: Whether to fetch all results
        as_dict: Whether to return results as dictionaries

    Returns:
        Tuple[Any, bool]: Tuple with query results and success flag
    """
    connection, success = get_db_connection()
    if not success:
        return None, False

    try:
        # Create cursor
        cursor_factory = RealDictCursor if as_dict else None
        cursor = connection.cursor(cursor_factory=cursor_factory)

        # Execute query
        cursor.execute(query, params)

        # Fetch results
        result = None
        if fetch_one:
            result = cursor.fetchone()
        elif fetch_all:
            result = cursor.fetchall()
        else:
            # For INSERT, UPDATE, DELETE operations
            affected_rows = cursor.rowcount
            result = {"affected_rows": affected_rows}

        connection.commit()
        cursor.close()

        return result, True
    except Exception as e:
        logger.error(f"Database query failed: {str(e)}")
        logger.error(f"Query: {query}")
        logger.error(f"Params: {params}")

        try:
            connection.rollback()
        except:
            pass

        return None, False
    finally:
        release_db_connection(connection)

def execute_transaction(queries: List[Dict[str, Any]]) -> Tuple[List[Any], bool]:
    """
    Execute multiple queries as a transaction

    Args:
        queries: List of query dictionaries, each containing:
            - query: SQL query string
            - params: Query parameters
            - fetch_one: Whether to fetch one result
            - fetch_all: Whether to fetch all results

    Returns:
        Tuple[List[Any], bool]: Tuple with list of results and success flag
    """
    connection, success = get_db_connection()
    if not success:
        return [], False

    try:
        # Begin transaction
        connection.autocommit = False

        results = []
        for query_dict in queries:
            query = query_dict.get("query", "")
            params = query_dict.get("params", None)
            fetch_one = query_dict.get("fetch_one", False)
            fetch_all = query_dict.get("fetch_all", True)
            as_dict = query_dict.get("as_dict", True)

            # Create cursor
            cursor_factory = RealDictCursor if as_dict else None
            cursor = connection.cursor(cursor_factory=cursor_factory)

            # Execute query
            cursor.execute(query, params)

            # Fetch results
            result = None
            if fetch_one:
                result = cursor.fetchone()
            elif fetch_all:
                result = cursor.fetchall()
            else:
                # For INSERT, UPDATE, DELETE operations
                affected_rows = cursor.rowcount
                result = {"affected_rows": affected_rows}

            results.append(result)
            cursor.close()

        # Commit transaction
        connection.commit()

        return results, True
    except Exception as e:
        logger.error(f"Transaction failed: {str(e)}")

        try:
            connection.rollback()
        except:
            pass

        return [], False
    finally:
        release_db_connection(connection)

def check_database_connection(config_file: Optional[str] = None) -> bool:
    """
    Check if the database connection is working

    Args:
        config_file: Path to the database configuration file

    Returns:
        bool: True if connection is working, False otherwise
    """
    # Initialize connection pool if needed
    if not connection_pool:
        if not init_db_pool(config_file):
            return False

    # Try to execute a simple query
    result, success = execute_query("SELECT 1 AS connection_test")
    if not success:
        return False

    return True

def create_database(config_file: Optional[str] = None) -> bool:
    """
    Create the database if it doesn't exist

    Args:
        config_file: Path to the database configuration file

    Returns:
        bool: True if database was created or already exists, False otherwise
    """
    # Load configuration
    db_config = load_db_config(config_file)
    if not db_config:
        return False

    # Save the target database name
    target_dbname = db_config["dbname"]

    # Connect to default 'postgres' database first
    db_config["dbname"] = "postgres"

    try:
        # Connect to postgres database
        conn = psycopg2.connect(
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 5432),
            dbname=db_config.get("dbname", "postgres"),
            user=db_config.get("user", "postgres"),
            password=db_config.get("password", ""),
            connect_timeout=db_config.get("connect_timeout", 10)
        )
        conn.autocommit = True

        # Check if database already exists
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_dbname,))
        exists = cursor.fetchone()

        if not exists:
            # Create database
            logger.info(f"Creating database '{target_dbname}'...")
            cursor.execute(f"CREATE DATABASE {target_dbname}")
            logger.info(f"Database '{target_dbname}' created successfully")
        else:
            logger.info(f"Database '{target_dbname}' already exists")

        cursor.close()
        conn.close()

        return True
    except Exception as e:
        logger.error(f"Failed to create database: {str(e)}")
        return False

def initialize_database_schema(schema_file: str) -> bool:
    """
    Initialize the database schema

    Args:
        schema_file: Path to the schema SQL file

    Returns:
        bool: True if schema was initialized successfully, False otherwise
    """
    # Check if schema file exists
    schema_path = Path(schema_file)
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_file}")
        return False

    try:
        # Read schema file
        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        # Execute schema SQL
        connection, success = get_db_connection()
        if not success:
            return False

        try:
            cursor = connection.cursor()
            cursor.execute(schema_sql)
            connection.commit()
            cursor.close()

            logger.info("Database schema initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize schema: {str(e)}")
            connection.rollback()
            return False
        finally:
            release_db_connection(connection)
    except Exception as e:
        logger.error(f"Failed to read schema file: {str(e)}")
        return False

def initialize_database_data(data_file: str) -> bool:
    """
    Initialize the database with initial data

    Args:
        data_file: Path to the data SQL file

    Returns:
        bool: True if data was initialized successfully, False otherwise
    """
    # Check if data file exists
    data_path = Path(data_file)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_file}")
        return False

    try:
        # Read data file
        with open(data_path, 'r') as f:
            data_sql = f.read()

        # Execute data SQL
        connection, success = get_db_connection()
        if not success:
            return False

        try:
            cursor = connection.cursor()
            cursor.execute(data_sql)
            connection.commit()
            cursor.close()

            logger.info("Database data initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize data: {str(e)}")
            connection.rollback()
            return False
        finally:
            release_db_connection(connection)
    except Exception as e:
        logger.error(f"Failed to read data file: {str(e)}")
        return False

def setup_database(config_file: Optional[str] = None,
                 schema_file: Optional[str] = None,
                 data_file: Optional[str] = None) -> bool:
    """
    Complete database setup: create database, initialize schema and data

    Args:
        config_file: Path to the database configuration file
        schema_file: Path to the schema SQL file
        data_file: Path to the data SQL file

    Returns:
        bool: True if setup was successful, False otherwise
    """
    # Create database
    if not create_database(config_file):
        return False

    # Initialize connection pool
    if not init_db_pool(config_file):
        return False

    # Initialize schema
    if schema_file:
        if not initialize_database_schema(schema_file):
            return False

    # Initialize data
    if data_file:
        if not initialize_database_data(data_file):
            return False

    logger.info("Database setup completed successfully")
    return True

if __name__ == "__main__":
    # Command-line interface for database setup
    import argparse

    parser = argparse.ArgumentParser(description="Database Utility for Negative Space Imaging Project")
    parser.add_argument("--config", help="Path to the database configuration file")
    parser.add_argument("--schema", help="Path to the schema SQL file")
    parser.add_argument("--data", help="Path to the data SQL file")
    parser.add_argument("--check", action="store_true", help="Check database connection")
    parser.add_argument("--create", action="store_true", help="Create database if it doesn't exist")
    parser.add_argument("--setup", action="store_true", help="Complete database setup")

    args = parser.parse_args()

    if args.check:
        # Check connection
        if check_database_connection(args.config):
            logger.info("Database connection is working")
            sys.exit(0)
        else:
            logger.error("Database connection failed")
            sys.exit(1)
    elif args.create:
        # Create database
        if create_database(args.config):
            logger.info("Database created or already exists")
            sys.exit(0)
        else:
            logger.error("Failed to create database")
            sys.exit(1)
    elif args.setup:
        # Complete setup
        if setup_database(args.config, args.schema, args.data):
            logger.info("Database setup completed successfully")
            sys.exit(0)
        else:
            logger.error("Database setup failed")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)
