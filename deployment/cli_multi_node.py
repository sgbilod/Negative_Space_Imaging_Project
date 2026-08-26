"""
CLI integration for multi-node deployment in the Negative Space Imaging Project
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("cli_multi_node.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the project root directory"""
    return Path.cwd()

def setup_multi_node_parser(subparsers):
    """Set up the multi-node deployment parser"""
    # Multi-node deployment command
    multi_node_parser = subparsers.add_parser(
        'multi-node',
        help='Multi-node deployment operations'
    )
    multi_node_subparsers = multi_node_parser.add_subparsers(
        dest='multi_node_command',
        help='Multi-node deployment commands'
    )

    # Setup command
    setup_parser = multi_node_subparsers.add_parser(
        'setup',
        help='Set up multi-node deployment'
    )
    setup_parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check prerequisites, don\'t set up anything'
    )

    # Deploy command
    deploy_parser = multi_node_subparsers.add_parser(
        'deploy',
        help='Deploy multi-node cluster'
    )
    deploy_parser.add_argument(
        '--config', '-c',
        default='deployment/multi_node_config.yaml',
        help='Path to configuration file'
    )
    deploy_parser.add_argument(
        '--templates', '-t',
        help='Path to template directory'
    )

    # Integrate command
    integrate_parser = multi_node_subparsers.add_parser(
        'integrate',
        help='Integrate multi-node deployment with existing system'
    )
    integrate_parser.add_argument(
        '--config', '-c',
        default='deployment/multi_node_config.yaml',
        help='Path to configuration file'
    )

    # Status command
    status_parser = multi_node_subparsers.add_parser(
        'status',
        help='Check multi-node deployment status'
    )
    status_parser.add_argument(
        '--config', '-c',
        default='deployment/multi_node_config.yaml',
        help='Path to configuration file'
    )

    return multi_node_parser

def handle_multi_node_command(args):
    """Handle multi-node deployment commands"""
    project_root = get_project_root()

    if args.multi_node_command == 'setup':
        logger.info("Setting up multi-node deployment")
        setup_script = project_root / "deployment" / "setup_multi_node.py"

        if not setup_script.exists():
            logger.error(f"Setup script not found: {setup_script}")
            return 1

        cmd = [sys.executable, str(setup_script)]
        if args.check_only:
            cmd.append("--check-only")

        return run_python_script(cmd)

    elif args.multi_node_command == 'deploy':
        logger.info("Deploying multi-node cluster")
        deploy_script = project_root / "deployment" / "multi_node_deploy.py"

        if not deploy_script.exists():
            logger.error(f"Deploy script not found: {deploy_script}")
            return 1

        cmd = [sys.executable, str(deploy_script)]
        if args.config:
            cmd.extend(["--config", args.config])
        if args.templates:
            cmd.extend(["--templates", args.templates])

        return run_python_script(cmd)

    elif args.multi_node_command == 'integrate':
        logger.info("Integrating multi-node deployment with existing system")
        integrate_script = project_root / "deployment" / "multi_node_integration.py"

        if not integrate_script.exists():
            logger.error(f"Integration script not found: {integrate_script}")
            return 1

        # Create environment with config path
        env = os.environ.copy()
        if args.config:
            env["MULTI_NODE_CONFIG"] = args.config

        cmd = [sys.executable, str(integrate_script)]

        return run_python_script(cmd, env=env)

    elif args.multi_node_command == 'status':
        logger.info("Checking multi-node deployment status")

        # Check if output directory exists
        output_dir = project_root / "deployment" / "output"
        status_file = output_dir / "deployment_status.json"

        if not output_dir.exists() or not status_file.exists():
            logger.error("No deployment status found. Has a deployment been run?")
            return 1

        # Read status file
        try:
            import json
            with open(status_file, 'r') as f:
                status = json.load(f)

            # Print status
            print("\nMulti-Node Deployment Status:")
            print(f"State: {status.get('state', 'unknown')}")
            print(f"Progress: {status.get('progress', 0)}%")
            print(f"Last action: {status.get('last_action', 'unknown')}")

            if 'start_time' in status:
                from datetime import datetime
                start_time = datetime.fromtimestamp(status['start_time']).strftime('%Y-%m-%d %H:%M:%S')
                print(f"Start time: {start_time}")

            if 'end_time' in status and status['end_time']:
                end_time = datetime.fromtimestamp(status['end_time']).strftime('%Y-%m-%d %H:%M:%S')
                print(f"End time: {end_time}")

            if 'errors' in status and status['errors']:
                print("\nErrors:")
                for error in status['errors']:
                    error_time = datetime.fromtimestamp(error['time']).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{error_time}] {error['message']}")

            return 0
        except Exception as e:
            logger.error(f"Failed to read status file: {str(e)}")
            return 1

    else:
        logger.error(f"Unknown multi-node command: {args.multi_node_command}")
        return 1

def run_python_script(cmd, env=None):
    """Run a Python script"""
    import subprocess

    try:
        logger.debug(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, env=env)
        return process.returncode
    except Exception as e:
        logger.error(f"Failed to run command: {str(e)}")
        return 1

def integrate_with_main_cli():
    """Function to be called from main CLI to integrate multi-node functionality"""
    # This function is intended to be imported and used in cli.py
    return setup_multi_node_parser, handle_multi_node_command

if __name__ == "__main__":
    # This script is not meant to be run directly
    print("This script is meant to be integrated with the main CLI.")
    print("Please run 'python cli.py multi-node --help' instead.")
    sys.exit(1)
