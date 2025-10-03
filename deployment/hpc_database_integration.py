#!/usr/bin/env python3
"""
Database Integration for HPC Environment
Provides utilities to integrate database operations with HPC computations.
"""

import os
import sys
import logging
import argparse
import yaml
import json
import datetime

# Add parent directory to path to import database_connection
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deployment.database_connection import (
    init_db_pool,
    execute_query,
    check_database_connection
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("hpc_database.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="HPC Database Integration")

    # Main actions
    parser.add_argument("--register-node", action="store_true",
                        help="Register this node in the database")
    parser.add_argument("--register-computation", action="store_true",
                        help="Register a new computation")
    parser.add_argument("--update-computation", action="store_true",
                        help="Update computation status")
    parser.add_argument("--log-event", action="store_true",
                        help="Log a system event")

    # Configuration
    parser.add_argument("--config", type=str, default="deployment/config/database.yaml",
                        help="Path to database configuration file")
    parser.add_argument("--node-id", type=str,
                        help="Node identifier (default: hostname)")
    parser.add_argument("--computation-id", type=str,
                        help="Computation ID")
    parser.add_argument("--project-id", type=str,
                        help="Project ID")
    parser.add_argument("--user-id", type=str,
                        help="User ID")

    # Computation details
    parser.add_argument("--name", type=str,
                        help="Name for computation or node")
    parser.add_argument("--description", type=str,
                        help="Description")
    parser.add_argument("--type", type=str,
                        help="Type of computation or event")
    parser.add_argument("--status", type=str,
                        help="Status of computation or node")
    parser.add_argument("--parameters", type=str,
                        help="JSON string of parameters")
    parser.add_argument("--input-images", type=str,
                        help="JSON array of input image IDs")
    parser.add_argument("--output-images", type=str,
                        help="JSON array of output image IDs")
    parser.add_argument("--severity", type=str, default="info",
                        help="Severity of event (info, warning, error, critical)")
    parser.add_argument("--message", type=str,
                        help="Message for event or log")
    parser.add_argument("--details", type=str,
                        help="JSON string of additional details")

    return parser.parse_args()

def get_node_id(args):
    """Get node ID from args or hostname if not provided"""
    if args.node_id:
        return args.node_id

    import socket
    return socket.gethostname()

def register_node(args):
    """Register this node in the database"""
    node_id = get_node_id(args)
    logger.info(f"Registering node: {node_id}")

    # Get node information
    import platform
    import psutil

    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "architecture": platform.machine(),
        "processor": platform.processor()
    }

    memory_info = {
        "total_ram": psutil.virtual_memory().total,
        "available_ram": psutil.virtual_memory().available
    }

    # Try to get GPU information if available
    gpu_info = []
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info.append({
                "id": gpu.id,
                "name": gpu.name,
                "memory_total": gpu.memoryTotal,
                "memory_free": gpu.memoryFree,
                "driver": gpu.driver
            })
    except (ImportError, Exception) as e:
        logger.warning(f"Could not get GPU information: {str(e)}")

    # Check if node already exists in database
    result, success = execute_query(
        "SELECT id FROM system_nodes WHERE node_id = %s",
        (node_id,),
        fetch_one=True
    )

    node_details = {
        "hostname": node_id,
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu": cpu_info,
        "memory": memory_info,
        "gpu": gpu_info,
        "last_seen": datetime.datetime.now().isoformat()
    }

    if success and result:
        # Update existing node
        result, success = execute_query(
            """
            UPDATE system_nodes
            SET status = %s, details = %s, last_seen = NOW(), updated_at = NOW()
            WHERE node_id = %s
            RETURNING id
            """,
            (args.status or "active", json.dumps(node_details), node_id),
            fetch_one=True
        )

        if success and result:
            logger.info(f"Updated node information for: {node_id}")
            return True
        else:
            logger.error(f"Failed to update node information for: {node_id}")
            return False
    else:
        # Create system_nodes table if it doesn't exist
        result, success = execute_query(
            """
            CREATE TABLE IF NOT EXISTS system_nodes (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                node_id VARCHAR(100) NOT NULL UNIQUE,
                name VARCHAR(100),
                status VARCHAR(20) NOT NULL DEFAULT 'active',
                node_type VARCHAR(50) DEFAULT 'compute',
                details JSONB,
                last_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
            )
            """
        )

        if not success:
            logger.error("Failed to create system_nodes table")
            return False

        # Insert new node
        result, success = execute_query(
            """
            INSERT INTO system_nodes (node_id, name, status, node_type, details)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (node_id, args.name or node_id, args.status or "active",
             args.type or "compute", json.dumps(node_details)),
            fetch_one=True
        )

        if success and result:
            logger.info(f"Registered new node: {node_id}")
            return True
        else:
            logger.error(f"Failed to register node: {node_id}")
            return False

def register_computation(args):
    """Register a new computation in the database"""
    if not args.project_id:
        logger.error("Project ID is required to register a computation")
        return False

    if not args.user_id:
        logger.error("User ID is required to register a computation")
        return False

    if not args.name:
        logger.error("Computation name is required")
        return False

    if not args.type:
        logger.error("Computation type is required")
        return False

    logger.info(f"Registering computation: {args.name}")

    # Parse JSON parameters
    parameters = {}
    if args.parameters:
        try:
            parameters = json.loads(args.parameters)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse parameters JSON: {str(e)}")
            return False

    # Parse input/output images
    input_images = []
    if args.input_images:
        try:
            input_images = json.loads(args.input_images)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse input_images JSON: {str(e)}")
            return False

    output_images = []
    if args.output_images:
        try:
            output_images = json.loads(args.output_images)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse output_images JSON: {str(e)}")
            return False

    # Insert computation
    result, success = execute_query(
        """
        INSERT INTO computations (
            project_id, name, description, type, status,
            parameters, input_images, output_images,
            created_by, started_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            args.project_id,
            args.name,
            args.description or "",
            args.type,
            args.status or "pending",
            json.dumps(parameters),
            json.dumps(input_images),
            json.dumps(output_images),
            args.user_id,
            datetime.datetime.now() if args.status == "running" else None
        ),
        fetch_one=True
    )

    if success and result:
        computation_id = result.get("id")
        logger.info(f"Registered computation with ID: {computation_id}")

        # Log initial computation event
        node_id = get_node_id(args)
        execute_query(
            """
            INSERT INTO computation_logs (
                computation_id, log_level, message
            )
            VALUES (%s, %s, %s)
            """,
            (
                computation_id,
                "info",
                f"Computation registered on node {node_id}"
            )
        )

        print(computation_id)
        return True
    else:
        logger.error("Failed to register computation")
        return False

def update_computation(args):
    """Update computation status and details"""
    if not args.computation_id:
        logger.error("Computation ID is required to update a computation")
        return False

    logger.info(f"Updating computation: {args.computation_id}")

    # Build update query dynamically based on provided arguments
    update_fields = []
    update_values = []

    if args.status:
        update_fields.append("status = %s")
        update_values.append(args.status)

        # Set started_at or completed_at based on status
        if args.status == "running":
            update_fields.append("started_at = %s")
            update_values.append(datetime.datetime.now())
        elif args.status in ["completed", "failed", "cancelled"]:
            update_fields.append("completed_at = %s")
            update_values.append(datetime.datetime.now())

    if args.parameters:
        try:
            parameters = json.loads(args.parameters)
            update_fields.append("parameters = %s")
            update_values.append(json.dumps(parameters))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse parameters JSON: {str(e)}")
            return False

    if args.output_images:
        try:
            output_images = json.loads(args.output_images)
            update_fields.append("output_images = %s")
            update_values.append(json.dumps(output_images))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse output_images JSON: {str(e)}")
            return False

    if args.message:
        update_fields.append("error_message = %s")
        update_values.append(args.message)

    if not update_fields:
        logger.error("No update fields provided")
        return False

    # Build and execute update query
    update_query = f"""
    UPDATE computations
    SET {", ".join(update_fields)}
    WHERE id = %s
    RETURNING id
    """

    update_values.append(args.computation_id)

    result, success = execute_query(
        update_query,
        tuple(update_values),
        fetch_one=True
    )

    if success and result:
        logger.info(f"Updated computation: {args.computation_id}")

        # Log computation update
        if args.message:
            log_level = "error" if args.status == "failed" else "info"
            execute_query(
                """
                INSERT INTO computation_logs (
                    computation_id, log_level, message
                )
                VALUES (%s, %s, %s)
                """,
                (
                    args.computation_id,
                    log_level,
                    args.message
                )
            )

        return True
    else:
        logger.error(f"Failed to update computation: {args.computation_id}")
        return False

def log_event(args):
    """Log a system event"""
    if not args.type:
        logger.error("Event type is required")
        return False

    if not args.message:
        logger.error("Event message is required")
        return False

    logger.info(f"Logging system event: {args.type}")

    # Parse details JSON
    details = {}
    if args.details:
        try:
            details = json.loads(args.details)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse details JSON: {str(e)}")
            return False

    # Add node information to details
    node_id = get_node_id(args)
    details["node_id"] = node_id

    # Insert event
    result, success = execute_query(
        """
        INSERT INTO system_events (
            event_type, severity, message, details, node_id
        )
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            args.type,
            args.severity,
            args.message,
            json.dumps(details),
            node_id
        ),
        fetch_one=True
    )

    if success and result:
        logger.info(f"Logged system event with ID: {result.get('id')}")
        return True
    else:
        logger.error("Failed to log system event")
        return False

def main():
    """Main function"""
    args = parse_args()

    # Initialize database connection
    if not check_database_connection(args.config):
        logger.error("Database connection failed")
        return 1

    if not init_db_pool(args.config):
        logger.error("Failed to initialize database connection pool")
        return 1

    # Execute requested action
    success = False

    if args.register_node:
        success = register_node(args)
    elif args.register_computation:
        success = register_computation(args)
    elif args.update_computation:
        success = update_computation(args)
    elif args.log_event:
        success = log_event(args)
    else:
        print("No action specified. Use --help to see available actions.")
        return 0

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
