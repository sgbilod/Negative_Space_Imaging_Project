#!/usr/bin/env python
"""
Security CLI Tool
© 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

Command-line tool for managing security in the Sovereign Control System:
- User management
- Security level configuration
- Key rotation
- File encryption/decryption
- Integrity verification
"""

import os
import sys
import argparse
import logging
import getpass
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sovereign.security_cli')

# Add parent directory to path to import sovereign modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from sovereign.security import SecurityManager, SecurityLevel
except ImportError:
    logger.error("Failed to import SecurityManager. Make sure you're running from the project root.")
    sys.exit(1)


def setup_argparse():
    """Set up command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Sovereign Control System Security Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new user
  python security_cli.py user create --username admin --role administrator

  # Change security level
  python security_cli.py config set-level --level MAXIMUM

  # Rotate encryption keys
  python security_cli.py keys rotate

  # Encrypt a file
  python security_cli.py file encrypt --path /path/to/file.txt

  # Verify file integrity
  python security_cli.py file verify --path /path/to/file.txt
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # User management
    user_parser = subparsers.add_parser('user', help='User management')
    user_subparsers = user_parser.add_subparsers(dest='user_command', help='User command')

    # Create user
    create_user_parser = user_subparsers.add_parser('create', help='Create a new user')
    create_user_parser.add_argument('--username', required=True, help='Username')
    create_user_parser.add_argument('--role', required=True, choices=['administrator', 'manager', 'user', 'guest'], help='User role')

    # Delete user
    delete_user_parser = user_subparsers.add_parser('delete', help='Delete a user')
    delete_user_parser.add_argument('--username', required=True, help='Username')

    # List users
    list_users_parser = user_subparsers.add_parser('list', help='List all users')

    # Reset password
    reset_password_parser = user_subparsers.add_parser('reset-password', help='Reset user password')
    reset_password_parser.add_argument('--username', required=True, help='Username')

    # Configuration management
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_command', help='Configuration command')

    # Set security level
    set_level_parser = config_subparsers.add_parser('set-level', help='Set security level')
    set_level_parser.add_argument('--level', required=True, choices=['STANDARD', 'ENHANCED', 'MAXIMUM', 'QUANTUM'], help='Security level')

    # Show configuration
    show_config_parser = config_subparsers.add_parser('show', help='Show current configuration')

    # Key management
    keys_parser = subparsers.add_parser('keys', help='Encryption key management')
    keys_subparsers = keys_parser.add_subparsers(dest='keys_command', help='Keys command')

    # Rotate keys
    rotate_keys_parser = keys_subparsers.add_parser('rotate', help='Rotate encryption keys')

    # File management
    file_parser = subparsers.add_parser('file', help='File security management')
    file_subparsers = file_parser.add_subparsers(dest='file_command', help='File command')

    # Encrypt file
    encrypt_file_parser = file_subparsers.add_parser('encrypt', help='Encrypt a file')
    encrypt_file_parser.add_argument('--path', required=True, help='Path to file or directory')

    # Decrypt file
    decrypt_file_parser = file_subparsers.add_parser('decrypt', help='Decrypt a file')
    decrypt_file_parser.add_argument('--path', required=True, help='Path to file or directory')

    # Update file checksum
    update_checksum_parser = file_subparsers.add_parser('update-checksum', help='Update file checksum')
    update_checksum_parser.add_argument('--path', required=True, help='Path to file or directory')

    # Verify file integrity
    verify_integrity_parser = file_subparsers.add_parser('verify', help='Verify file integrity')
    verify_integrity_parser.add_argument('--path', required=True, help='Path to file or directory')

    # Audit logs
    logs_parser = subparsers.add_parser('logs', help='Audit log management')
    logs_subparsers = logs_parser.add_subparsers(dest='logs_command', help='Logs command')

    # Show logs
    show_logs_parser = logs_subparsers.add_parser('show', help='Show audit logs')
    show_logs_parser.add_argument('--limit', type=int, default=100, help='Maximum number of logs to show')
    show_logs_parser.add_argument('--event-type', help='Filter by event type')
    show_logs_parser.add_argument('--username', help='Filter by username')
    show_logs_parser.add_argument('--from-date', help='Filter from date (YYYY-MM-DD)')
    show_logs_parser.add_argument('--to-date', help='Filter to date (YYYY-MM-DD)')

    return parser


def initialize_security_manager():
    """Initialize the security manager"""
    # Get project root
    project_root = Path(__file__).parent.parent

    # Check if security level is specified in environment
    security_level = os.environ.get('SOVEREIGN_SECURITY_LEVEL', SecurityLevel.ENHANCED)

    # Initialize security manager
    try:
        security_manager = SecurityManager(project_root, security_level)
        return security_manager
    except Exception as e:
        logger.error(f"Failed to initialize security manager: {str(e)}")
        sys.exit(1)


def handle_user_commands(security_manager, args):
    """Handle user management commands"""
    if args.user_command == 'create':
        # Prompt for password
        password = getpass.getpass("Enter password for new user: ")
        confirm_password = getpass.getpass("Confirm password: ")

        if password != confirm_password:
            logger.error("Passwords do not match")
            return

        success = security_manager.create_user(args.username, password, args.role)

        if success:
            logger.info(f"User {args.username} created successfully with role {args.role}")
        else:
            logger.error(f"Failed to create user {args.username}")

    elif args.user_command == 'delete':
        # Get confirmation
        confirm = input(f"Are you sure you want to delete user {args.username}? (y/n): ")

        if confirm.lower() != 'y':
            logger.info("User deletion cancelled")
            return

        # Load users file
        users_file = security_manager.users_path / "users.json"

        if not users_file.exists():
            logger.error("Users file not found")
            return

        try:
            # Load users data
            users = security_manager._load_encrypted_json(users_file, security_manager.auth_key)

            # Check if user exists
            if args.username not in users:
                logger.error(f"User {args.username} not found")
                return

            # Delete user
            del users[args.username]

            # Save updated users
            security_manager._save_encrypted_json(users_file, users, security_manager.auth_key)

            logger.info(f"User {args.username} deleted successfully")

        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")

    elif args.user_command == 'list':
        # Load users file
        users_file = security_manager.users_path / "users.json"

        if not users_file.exists():
            logger.error("Users file not found")
            return

        try:
            # Load users data
            users = security_manager._load_encrypted_json(users_file, security_manager.auth_key)

            if not users:
                logger.info("No users found")
                return

            # Print users
            print("\nUsers:")
            print(f"{'Username':<20} {'Role':<15} {'Last Login':<25} {'MFA Enabled':<15}")
            print("-" * 80)

            for username, user_data in users.items():
                last_login = user_data.get('last_login', 'Never')
                if last_login and last_login != 'Never':
                    try:
                        last_login = datetime.fromisoformat(last_login).strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass

                mfa_enabled = "Yes" if user_data.get('mfa_enabled') else "No"

                print(f"{username:<20} {user_data.get('role', 'N/A'):<15} {last_login:<25} {mfa_enabled:<15}")

            print()

        except Exception as e:
            logger.error(f"Error listing users: {str(e)}")

    elif args.user_command == 'reset-password':
        # Load users file
        users_file = security_manager.users_path / "users.json"

        if not users_file.exists():
            logger.error("Users file not found")
            return

        try:
            # Load users data
            users = security_manager._load_encrypted_json(users_file, security_manager.auth_key)

            # Check if user exists
            if args.username not in users:
                logger.error(f"User {args.username} not found")
                return

            # Prompt for new password
            new_password = getpass.getpass("Enter new password: ")
            confirm_password = getpass.getpass("Confirm new password: ")

            if new_password != confirm_password:
                logger.error("Passwords do not match")
                return

            # Generate new salt and hash
            salt = security_manager._hash_password_salt()
            password_hash = security_manager._hash_password(new_password, salt)

            # Update user data
            user = users[args.username]
            user['password_hash'] = password_hash
            user['salt'] = salt
            user['require_password_change'] = True
            users[args.username] = user

            # Save updated users
            security_manager._save_encrypted_json(users_file, users, security_manager.auth_key)

            logger.info(f"Password reset for user {args.username}")
            logger.info(f"User will be required to change password on next login")

        except Exception as e:
            logger.error(f"Error resetting password: {str(e)}")

    else:
        logger.error(f"Unknown user command: {args.user_command}")


def handle_config_commands(security_manager, args):
    """Handle configuration management commands"""
    if args.config_command == 'set-level':
        # Update security level
        old_level = security_manager.security_level
        security_manager.security_level = args.level

        # Update settings based on new level
        security_manager.settings = security_manager._load_default_settings()

        # Save settings to file
        settings_file = security_manager.security_path / "settings.json"
        security_manager._save_encrypted_json(settings_file, security_manager.settings, security_manager.config_key)

        logger.info(f"Security level updated from {old_level} to {args.level}")
        security_manager.log_security_event(
            "CLI_SECURITY_LEVEL_CHANGED",
            {"old_level": old_level, "new_level": args.level}
        )

    elif args.config_command == 'show':
        # Show current configuration
        print(f"\nSecurity Level: {security_manager.security_level}")
        print("\nSecurity Settings:")

        for category, settings in security_manager.settings.items():
            print(f"\n{category.capitalize()}:")
            for setting, value in settings.items():
                print(f"  {setting}: {value}")

        print()

    else:
        logger.error(f"Unknown config command: {args.config_command}")


def handle_keys_commands(security_manager, args):
    """Handle encryption key management commands"""
    if args.keys_command == 'rotate':
        # Get confirmation
        confirm = input("Are you sure you want to rotate encryption keys? This operation cannot be undone. (y/n): ")

        if confirm.lower() != 'y':
            logger.info("Key rotation cancelled")
            return

        success = security_manager.rotate_keys()

        if success:
            logger.info("Encryption keys rotated successfully")
            security_manager.log_security_event("CLI_KEYS_ROTATED", {})
        else:
            logger.error("Failed to rotate encryption keys")

    else:
        logger.error(f"Unknown keys command: {args.keys_command}")


def handle_file_commands(security_manager, args):
    """Handle file security management commands"""
    file_path = Path(args.path)

    # Check if path exists
    if not file_path.exists():
        logger.error(f"Path does not exist: {args.path}")
        return

    if args.file_command == 'encrypt':
        if file_path.is_file():
            success = security_manager.encrypt_file(file_path)

            if success:
                logger.info(f"File encrypted successfully: {args.path}")
                security_manager.log_security_event("CLI_FILE_ENCRYPTED", {"path": args.path})
            else:
                logger.error(f"Failed to encrypt file: {args.path}")
        elif file_path.is_dir():
            encrypted_count = 0
            failed_count = 0

            for file in file_path.glob('**/*'):
                if file.is_file() and not str(file).endswith('.enc'):
                    success = security_manager.encrypt_file(file)

                    if success:
                        encrypted_count += 1
                    else:
                        failed_count += 1

            logger.info(f"Directory encryption complete: {encrypted_count} files encrypted, {failed_count} failed")
            security_manager.log_security_event(
                "CLI_DIRECTORY_ENCRYPTED",
                {"path": args.path, "encrypted_count": encrypted_count, "failed_count": failed_count}
            )

    elif args.file_command == 'decrypt':
        if file_path.is_file():
            if not str(file_path).endswith('.enc'):
                logger.error(f"File is not encrypted (missing .enc extension): {args.path}")
                return

            success = security_manager.decrypt_file(file_path)

            if success:
                logger.info(f"File decrypted successfully: {args.path}")
                security_manager.log_security_event("CLI_FILE_DECRYPTED", {"path": args.path})
            else:
                logger.error(f"Failed to decrypt file: {args.path}")
        elif file_path.is_dir():
            decrypted_count = 0
            failed_count = 0

            for file in file_path.glob('**/*.enc'):
                if file.is_file():
                    success = security_manager.decrypt_file(file)

                    if success:
                        decrypted_count += 1
                    else:
                        failed_count += 1

            logger.info(f"Directory decryption complete: {decrypted_count} files decrypted, {failed_count} failed")
            security_manager.log_security_event(
                "CLI_DIRECTORY_DECRYPTED",
                {"path": args.path, "decrypted_count": decrypted_count, "failed_count": failed_count}
            )

    elif args.file_command == 'update-checksum':
        if file_path.is_file():
            success = security_manager.update_file_checksum(file_path)

            if success:
                logger.info(f"File checksum updated successfully: {args.path}")
                security_manager.log_security_event("CLI_CHECKSUM_UPDATED", {"path": args.path})
            else:
                logger.error(f"Failed to update file checksum: {args.path}")
        elif file_path.is_dir():
            updated_count = 0
            failed_count = 0

            for file in file_path.glob('**/*'):
                if file.is_file():
                    success = security_manager.update_file_checksum(file)

                    if success:
                        updated_count += 1
                    else:
                        failed_count += 1

            logger.info(f"Directory checksum update complete: {updated_count} files updated, {failed_count} failed")
            security_manager.log_security_event(
                "CLI_DIRECTORY_CHECKSUMS_UPDATED",
                {"path": args.path, "updated_count": updated_count, "failed_count": failed_count}
            )

    elif args.file_command == 'verify':
        if file_path.is_file():
            integrity = security_manager.verify_file_integrity(file_path)

            if integrity:
                logger.info(f"File integrity verified: {args.path}")
                security_manager.log_security_event("CLI_INTEGRITY_VERIFIED", {"path": args.path})
            else:
                logger.error(f"File integrity check failed: {args.path}")
                security_manager.log_security_event("CLI_INTEGRITY_FAILED", {"path": args.path})
        elif file_path.is_dir():
            verified_count = 0
            failed_count = 0

            for file in file_path.glob('**/*'):
                if file.is_file():
                    integrity = security_manager.verify_file_integrity(file)

                    if integrity:
                        verified_count += 1
                    else:
                        failed_count += 1

            logger.info(f"Directory integrity check complete: {verified_count} files verified, {failed_count} failed")
            security_manager.log_security_event(
                "CLI_DIRECTORY_INTEGRITY_CHECK",
                {"path": args.path, "verified_count": verified_count, "failed_count": failed_count}
            )

    else:
        logger.error(f"Unknown file command: {args.file_command}")


def handle_logs_commands(security_manager, args):
    """Handle audit log management commands"""
    if args.logs_command == 'show':
        log_files = sorted(security_manager.logs_path.glob('audit_*.log'), reverse=True)
        log_entries = []

        # Parse date filters if provided
        from_date = None
        to_date = None

        if args.from_date:
            try:
                from_date = datetime.strptime(args.from_date, "%Y-%m-%d")
            except ValueError:
                logger.error(f"Invalid from date format: {args.from_date}. Use YYYY-MM-DD.")
                return

        if args.to_date:
            try:
                to_date = datetime.strptime(args.to_date, "%Y-%m-%d")
                # Set to end of day
                to_date = to_date.replace(hour=23, minute=59, second=59)
            except ValueError:
                logger.error(f"Invalid to date format: {args.to_date}. Use YYYY-MM-DD.")
                return

        for log_file in log_files:
            if len(log_entries) >= args.limit:
                break

            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if len(log_entries) >= args.limit:
                            break

                        try:
                            import json
                            entry = json.loads(line.strip())

                            # Apply date filters
                            entry_date = datetime.fromisoformat(entry['timestamp'])
                            if from_date and entry_date < from_date:
                                continue
                            if to_date and entry_date > to_date:
                                continue

                            # Filter by event type if specified
                            if args.event_type and entry.get('event_type') != args.event_type:
                                continue

                            # Filter by username if specified
                            if args.username:
                                user_match = False
                                if 'details' in entry:
                                    if entry['details'].get('username') == args.username:
                                        user_match = True
                                    elif entry['details'].get('admin') == args.username:
                                        user_match = True
                                    elif entry['details'].get('new_user') == args.username:
                                        user_match = True
                                    elif entry['details'].get('deleted_user') == args.username:
                                        user_match = True

                                if not user_match:
                                    continue

                            log_entries.append(entry)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {str(e)}")
                continue

        if not log_entries:
            logger.info("No log entries found matching the criteria")
            return

        # Print log entries
        print(f"\nAudit Logs ({len(log_entries)} entries):")
        print(f"{'Timestamp':<25} {'Event Type':<30} {'Details'}")
        print("-" * 100)

        for entry in log_entries:
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            event_type = entry['event_type']
            details = str(entry['details'])

            # Truncate details if too long
            if len(details) > 50:
                details = details[:47] + "..."

            print(f"{timestamp:<25} {event_type:<30} {details}")

        print()

    else:
        logger.error(f"Unknown logs command: {args.logs_command}")


def main():
    """Main function"""
    parser = setup_argparse()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize security manager
    security_manager = initialize_security_manager()

    # Handle commands
    if args.command == 'user' and args.user_command:
        handle_user_commands(security_manager, args)
    elif args.command == 'config' and args.config_command:
        handle_config_commands(security_manager, args)
    elif args.command == 'keys' and args.keys_command:
        handle_keys_commands(security_manager, args)
    elif args.command == 'file' and args.file_command:
        handle_file_commands(security_manager, args)
    elif args.command == 'logs' and args.logs_command:
        handle_logs_commands(security_manager, args)
    else:
        logger.error(f"Missing subcommand for {args.command}")
        if args.command == 'user':
            logger.info("Available user commands: create, delete, list, reset-password")
        elif args.command == 'config':
            logger.info("Available config commands: set-level, show")
        elif args.command == 'keys':
            logger.info("Available keys commands: rotate")
        elif args.command == 'file':
            logger.info("Available file commands: encrypt, decrypt, update-checksum, verify")
        elif args.command == 'logs':
            logger.info("Available logs commands: show")


if __name__ == '__main__':
    main()
