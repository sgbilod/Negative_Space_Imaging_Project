"""
Database Migrations System for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import importlib.util
from typing import List, Dict, Any, Optional

import alembic
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("Migrations")


class MigrationManager:
    """Manages database migrations."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)
        self.config = self._create_alembic_config()

    def _create_alembic_config(self) -> Config:
        """Create Alembic configuration."""
        migrations_dir = Path(__file__).parent / 'migrations'
        migrations_dir.mkdir(exist_ok=True)

        config = Config()
        config.set_main_option('script_location', str(migrations_dir))
        config.set_main_option('sqlalchemy.url', self.connection_string)

        return config

    def create_migration(self, message: str) -> str:
        """Create a new migration."""
        try:
            # Generate migration file
            rev_id = command.revision(
                self.config,
                message=message,
                autogenerate=True
            )

            logger.info(f"Created migration: {rev_id}")
            return str(rev_id)

        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            raise

    def upgrade(self, revision: str = 'head') -> bool:
        """Upgrade database to a later version."""
        try:
            command.upgrade(self.config, revision)
            logger.info(f"Upgraded database to: {revision}")
            return True

        except Exception as e:
            logger.error(f"Upgrade failed: {e}")
            return False

    def downgrade(self, revision: str) -> bool:
        """Downgrade database to a previous version."""
        try:
            command.downgrade(self.config, revision)
            logger.info(f"Downgraded database to: {revision}")
            return True

        except Exception as e:
            logger.error(f"Downgrade failed: {e}")
            return False

    def get_current_revision(self) -> Optional[str]:
        """Get current database revision."""
        try:
            with self.engine.connect() as conn:
                context = alembic.migration.MigrationContext.configure(conn)
                return context.get_current_revision()

        except Exception as e:
            logger.error(f"Failed to get current revision: {e}")
            return None

    def get_history(self) -> List[Dict[str, Any]]:
        """Get migration history."""
        try:
            history = []
            script = self.config.attributes['version_locations'][0]

            for revision in script.walk_revisions():
                history.append({
                    'revision': revision.revision,
                    'down_revision': revision.down_revision,
                    'message': revision.doc,
                    'created': revision.created_date
                })

            return history

        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []

    def verify_database(self) -> bool:
        """Verify database structure matches migrations."""
        try:
            # Get current revision
            current = self.get_current_revision()
            if not current:
                logger.error("No current revision found")
                return False

            # Compare with latest migration
            script = self.config.attributes['version_locations'][0]
            latest = script.get_current_head()

            if current != latest:
                logger.warning(
                    f"Database ({current}) != Latest migration ({latest})"
                )
                return False

            logger.info("Database structure verified")
            return True

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

    def create_backup(self) -> bool:
        """Create database backup before migration."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"backup_{timestamp}.sql"

            # Using pg_dump for backup
            os.system(
                f"pg_dump {self.connection_string} > {backup_file}"
            )

            logger.info(f"Created backup: {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

    def restore_backup(self, backup_file: str) -> bool:
        """Restore database from backup."""
        try:
            # Using psql for restore
            os.system(
                f"psql {self.connection_string} < {backup_file}"
            )

            logger.info(f"Restored from backup: {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False


def main():
    """Main entry point for migrations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Database Migrations Manager"
    )

    parser.add_argument(
        '--connection',
        required=True,
        help='Database connection string'
    )

    parser.add_argument(
        '--action',
        choices=['create', 'upgrade', 'downgrade', 'verify'],
        required=True,
        help='Action to perform'
    )

    parser.add_argument(
        '--message',
        help='Migration message (for create)'
    )

    parser.add_argument(
        '--revision',
        help='Target revision (for upgrade/downgrade)'
    )

    args = parser.parse_args()

    try:
        manager = MigrationManager(args.connection)

        if args.action == 'create':
            if not args.message:
                parser.error("--message required for create action")
            manager.create_migration(args.message)

        elif args.action == 'upgrade':
            revision = args.revision or 'head'
            manager.upgrade(revision)

        elif args.action == 'downgrade':
            if not args.revision:
                parser.error("--revision required for downgrade action")
            manager.downgrade(args.revision)

        elif args.action == 'verify':
            if manager.verify_database():
                print("Database structure verified")
            else:
                print("Database verification failed")
                sys.exit(1)

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
