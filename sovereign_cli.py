#!/usr/bin/env python3
# Sovereign CLI
# © 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

from pathlib import Path
import sys

# Add project root to path to allow importing from sovereign package
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any

from sovereign.master_controller import MasterController, ControlMode


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("SovereignCLI")


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Sovereign Control System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sovereign_cli.py initialize --mode quantum
  sovereign_cli.py execute --directive "ANALYZE_DATA"
  sovereign_cli.py optimize --target all
  sovereign_cli.py status
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Initialize command
    init_parser = subparsers.add_parser("initialize", help="Initialize sovereign system")
    init_parser.add_argument(
        "--mode",
        choices=[m.value for m in ControlMode],
        default=ControlMode.STANDARD.value,
        help="Control mode for operation"
    )

    # Execute command
    exec_parser = subparsers.add_parser("execute", help="Execute sovereign directive")
    exec_parser.add_argument(
        "--directive", "-d",
        required=True,
        help="Directive to execute"
    )

    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Optimize sovereign system")
    opt_parser.add_argument(
        "--target", "-t",
        choices=["all", "quantum", "hypercognition", "acceleration", "control"],
        default="all",
        help="Optimization target"
    )

    # Status command
    subparsers.add_parser("status", help="Get sovereign system status")

    # Save state command
    save_parser = subparsers.add_parser("save", help="Save sovereign system state")
    save_parser.add_argument(
        "--file", "-f",
        help="Filename to save state to"
    )

    # Load state command
    load_parser = subparsers.add_parser("load", help="Load sovereign system state")
    load_parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to state file to load"
    )

    # Execute sovereign operation
    subparsers.add_parser(
        "sovereign",
        help="Execute full sovereign operation with executive authority"
    )

    return parser.parse_args()


def initialize_controller(args) -> MasterController:
    """Initialize the master controller"""
    logger = logging.getLogger("SovereignCLI")
    logger.info(f"Initializing Sovereign Control System in {args.mode} mode")

    # Convert string mode to enum
    mode = ControlMode(args.mode)

    # Initialize controller
    controller = MasterController(
        get_project_root(),
        mode
    )

    logger.info(f"Sovereign Control System initialized with ID: {controller.control_id}")
    return controller


def execute_directive(controller: MasterController, args) -> Dict[str, Any]:
    """Execute a sovereign directive"""
    logger = logging.getLogger("SovereignCLI")
    logger.info(f"Executing directive: {args.directive}")

    result = controller.execute_directive(args.directive)

    if result["status"] == "SUCCESS":
        logger.info(f"Directive executed successfully in {result['execution_time']:.2f} seconds")
    else:
        logger.error(f"Error executing directive: {result.get('error', 'Unknown error')}")

    return result


def optimize_system(controller: MasterController, args) -> Dict[str, Any]:
    """Optimize the sovereign system"""
    logger = logging.getLogger("SovereignCLI")
    logger.info(f"Optimizing system (target: {args.target})")

    result = controller.optimize_system(args.target)

    if result["status"] == "SUCCESS":
        logger.info("System optimization completed successfully")
    else:
        logger.error(f"Error optimizing system: {result.get('error', 'Unknown error')}")

    return result


def get_system_status(controller: MasterController) -> Dict[str, Any]:
    """Get the current system status"""
    logger = logging.getLogger("SovereignCLI")
    logger.info("Retrieving system status")

    status = controller.get_system_status()
    state = controller.get_system_state()

    # Combine status and state
    combined = {
        "status": status,
        "state": {
            "control_id": state["control_id"],
            "mode": state["mode"],
            "uptime_seconds": state["uptime_seconds"],
            "control_state": state["control_state"],
        }
    }

    logger.info(f"System is {status['system_health']} with {status['task_completion']} tasks completed")

    return combined


def save_system_state(controller: MasterController, args) -> str:
    """Save the system state to a file"""
    logger = logging.getLogger("SovereignCLI")

    # Save state
    state_path = controller.save_system_state(args.file)

    logger.info(f"System state saved to: {state_path}")
    return state_path


def load_system_state(controller: MasterController, args) -> Dict[str, Any]:
    """Load system state from a file"""
    logger = logging.getLogger("SovereignCLI")
    logger.info(f"Loading system state from: {args.file}")

    state = controller.load_system_state(args.file)

    logger.info(f"System state loaded successfully, control ID: {state.get('control_id', 'unknown')}")
    return state


def execute_sovereign_operation(controller: MasterController) -> Dict[str, Any]:
    """Execute full sovereign operation with executive authority"""
    logger = logging.getLogger("SovereignCLI")
    logger.info("INITIATING SOVEREIGN OPERATION WITH EXECUTIVE AUTHORITY")

    # Begin sovereign operation
    controller.begin_sovereign_operation()

    # Get status after operation
    status = controller.get_system_status()

    logger.info("Sovereign operation completed successfully")
    return status


def format_output(data: Dict[str, Any]) -> str:
    """Format output data as JSON"""
    return json.dumps(data, indent=2, default=str)


def main():
    """Main entry point"""
    # Setup logging
    logger = setup_logging()

    # Parse arguments
    args = parse_arguments()

    if not args.command:
        logger.error("No command specified. Use --help for usage information.")
        sys.exit(1)

    try:
        # Initialize controller for all commands
        controller = initialize_controller(args)

        # Execute command
        if args.command == "initialize":
            result = {"status": "SUCCESS", "control_id": controller.control_id}
        elif args.command == "execute":
            result = execute_directive(controller, args)
        elif args.command == "optimize":
            result = optimize_system(controller, args)
        elif args.command == "status":
            result = get_system_status(controller)
        elif args.command == "save":
            result = {"state_path": save_system_state(controller, args)}
        elif args.command == "load":
            result = load_system_state(controller, args)
        elif args.command == "sovereign":
            result = execute_sovereign_operation(controller)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)

        # Print result
        print("\nRESULT:")
        print(format_output(result))

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
