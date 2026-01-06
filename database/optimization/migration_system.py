"""
Automated Database Migration System with Rollback Capabilities
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import sqlalchemy as sa
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Database migration definition."""
    version: str
    name: str
    description: str
    up_sql: str
    down_sql: Optional[str] = None
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
    status: str = "pending"  # pending, applied, failed, rolled_back


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    success: bool
    migration: Migration
    error_message: Optional[str] = None
    execution_time: float = 0.0
    affected_rows: int = 0


class MigrationManager:
    """Manages database schema migrations with rollback support."""

    def __init__(self, engine, migrations_dir: str = "migrations"):
        self.engine = engine
        self.migrations_dir = Path(migrations_dir)
        self.migrations_dir.mkdir(exist_ok=True)

        # Create migrations table if it doesn't exist
        self._ensure_migrations_table()

        # Load existing migrations
        self.migrations: Dict[str, Migration] = {}
        self._load_migrations()

    def _ensure_migrations_table(self):
        """Ensure the migrations tracking table exists."""
        with self.engine.connect() as conn:
            try:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        checksum VARCHAR(255) NOT NULL,
                        up_sql TEXT NOT NULL,
                        down_sql TEXT,
                        created_at TIMESTAMP NOT NULL,
                        applied_at TIMESTAMP,
                        status VARCHAR(50) NOT NULL DEFAULT 'pending',
                        execution_time FLOAT DEFAULT 0,
                        error_message TEXT
                    )
                """))
                conn.commit()
                logger.info("Migrations table ensured")

            except SQLAlchemyError as e:
                logger.error(f"Failed to create migrations table: {e}")
                raise

    def _load_migrations(self):
        """Load existing migrations from database."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT version, name, description, checksum, up_sql, down_sql,
                           created_at, applied_at, status
                    FROM schema_migrations
                    ORDER BY created_at
                """))

                for row in result:
                    migration = Migration(
                        version=row[0],
                        name=row[1],
                        description=row[2],
                        up_sql=row[3],
                        down_sql=row[4],
                        checksum=row[5],
                        created_at=row[6],
                        applied_at=row[7],
                        status=row[8]
                    )
                    self.migrations[migration.version] = migration

                logger.info(f"Loaded {len(self.migrations)} migrations from database")

        except SQLAlchemyError as e:
            logger.error(f"Failed to load migrations: {e}")

    def create_migration(self, name: str, description: str, up_sql: str,
                        down_sql: Optional[str] = None) -> Migration:
        """Create a new migration file."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        version = f"{timestamp}_{name.lower().replace(' ', '_')}"

        migration = Migration(
            version=version,
            name=name,
            description=description,
            up_sql=up_sql,
            down_sql=down_sql,
            checksum=self._calculate_checksum(up_sql, down_sql)
        )

        # Save to file
        migration_file = self.migrations_dir / f"{version}.json"
        with open(migration_file, 'w') as f:
            json.dump({
                'version': migration.version,
                'name': migration.name,
                'description': migration.description,
                'up_sql': migration.up_sql,
                'down_sql': migration.down_sql,
                'checksum': migration.checksum,
                'created_at': migration.created_at.isoformat()
            }, f, indent=2)

        # Save to database
        self._save_migration_to_db(migration)

        logger.info(f"Created migration: {version}")
        return migration

    def _calculate_checksum(self, up_sql: str, down_sql: Optional[str] = None) -> str:
        """Calculate checksum for migration content."""
        content = up_sql + (down_sql or "")
        return hashlib.sha256(content.encode()).hexdigest()

    def _save_migration_to_db(self, migration: Migration):
        """Save migration to database."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO schema_migrations
                    (version, name, description, checksum, up_sql, down_sql, created_at, status)
                    VALUES (:version, :name, :description, :checksum, :up_sql, :down_sql, :created_at, :status)
                """), {
                    'version': migration.version,
                    'name': migration.name,
                    'description': migration.description,
                    'checksum': migration.checksum,
                    'up_sql': migration.up_sql,
                    'down_sql': migration.down_sql,
                    'created_at': migration.created_at,
                    'status': migration.status
                })
                conn.commit()

        except SQLAlchemyError as e:
            logger.error(f"Failed to save migration to database: {e}")

    def apply_migration(self, version: str) -> MigrationResult:
        """Apply a specific migration."""
        if version not in self.migrations:
            return MigrationResult(
                success=False,
                migration=Migration(version=version, name="", description="", up_sql=""),
                error_message=f"Migration {version} not found"
            )

        migration = self.migrations[version]

        if migration.status == "applied":
            return MigrationResult(
                success=False,
                migration=migration,
                error_message=f"Migration {version} already applied"
            )

        start_time = datetime.now()

        try:
            with self.engine.connect() as conn:
                # Execute migration
                conn.execute(text(migration.up_sql))

                # Update migration status
                conn.execute(text("""
                    UPDATE schema_migrations
                    SET status = 'applied', applied_at = :applied_at, execution_time = :execution_time
                    WHERE version = :version
                """), {
                    'version': version,
                    'applied_at': start_time,
                    'execution_time': (datetime.now() - start_time).total_seconds()
                })

                conn.commit()

                migration.status = "applied"
                migration.applied_at = start_time

                execution_time = (datetime.now() - start_time).total_seconds()

                logger.info(f"Applied migration: {version} ({execution_time:.2f}s)")
                return MigrationResult(
                    success=True,
                    migration=migration,
                    execution_time=execution_time
                )

        except SQLAlchemyError as e:
            error_msg = str(e)
            logger.error(f"Failed to apply migration {version}: {error_msg}")

            # Mark as failed
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        UPDATE schema_migrations
                        SET status = 'failed', error_message = :error
                        WHERE version = :version
                    """), {'version': version, 'error': error_msg})
                    conn.commit()

            except SQLAlchemyError:
                pass  # Ignore errors in error handling

            return MigrationResult(
                success=False,
                migration=migration,
                error_message=error_msg
            )

    def rollback_migration(self, version: str) -> MigrationResult:
        """Rollback a specific migration."""
        if version not in self.migrations:
            return MigrationResult(
                success=False,
                migration=Migration(version=version, name="", description="", up_sql=""),
                error_message=f"Migration {version} not found"
            )

        migration = self.migrations[version]

        if migration.status != "applied":
            return MigrationResult(
                success=False,
                migration=migration,
                error_message=f"Migration {version} is not applied"
            )

        if not migration.down_sql:
            return MigrationResult(
                success=False,
                migration=migration,
                error_message=f"Migration {version} has no rollback SQL"
            )

        start_time = datetime.now()

        try:
            with self.engine.connect() as conn:
                # Execute rollback
                conn.execute(text(migration.down_sql))

                # Update migration status
                conn.execute(text("""
                    UPDATE schema_migrations
                    SET status = 'rolled_back', applied_at = NULL
                    WHERE version = :version
                """), {'version': version})

                conn.commit()

                migration.status = "rolled_back"
                migration.applied_at = None

                execution_time = (datetime.now() - start_time).total_seconds()

                logger.info(f"Rolled back migration: {version} ({execution_time:.2f}s)")
                return MigrationResult(
                    success=True,
                    migration=migration,
                    execution_time=execution_time
                )

        except SQLAlchemyError as e:
            error_msg = str(e)
            logger.error(f"Failed to rollback migration {version}: {error_msg}")

            return MigrationResult(
                success=False,
                migration=migration,
                error_message=error_msg
            )

    def apply_pending_migrations(self) -> List[MigrationResult]:
        """Apply all pending migrations."""
        pending = [m for m in self.migrations.values() if m.status == "pending"]
        results = []

        for migration in sorted(pending, key=lambda m: m.version):
            result = self.apply_migration(migration.version)
            results.append(result)

            if not result.success:
                logger.error(f"Migration chain stopped at {migration.version}")
                break

        return results

    def rollback_to_version(self, target_version: str) -> List[MigrationResult]:
        """Rollback migrations down to a specific version."""
        applied = [m for m in self.migrations.values() if m.status == "applied"]
        applied_sorted = sorted(applied, key=lambda m: m.version, reverse=True)

        results = []
        for migration in applied_sorted:
            if migration.version <= target_version:
                break

            result = self.rollback_migration(migration.version)
            results.append(result)

            if not result.success:
                logger.error(f"Rollback chain stopped at {migration.version}")
                break

        return results

    def get_migration_status(self) -> Dict[str, Any]:
        """Get comprehensive migration status."""
        total = len(self.migrations)
        applied = len([m for m in self.migrations.values() if m.status == "applied"])
        pending = len([m for m in self.migrations.values() if m.status == "pending"])
        failed = len([m for m in self.migrations.values() if m.status == "failed"])

        return {
            'total_migrations': total,
            'applied_migrations': applied,
            'pending_migrations': pending,
            'failed_migrations': failed,
            'current_version': max([m.version for m in self.migrations.values() if m.status == "applied"] or ["none"]),
            'migrations': [
                {
                    'version': m.version,
                    'name': m.name,
                    'status': m.status,
                    'applied_at': m.applied_at.isoformat() if m.applied_at else None
                }
                for m in sorted(self.migrations.values(), key=lambda x: x.version)
            ]
        }


class MigrationGenerator:
    """Generates common migration patterns."""

    @staticmethod
    def create_table(table_name: str, columns: Dict[str, str]) -> str:
        """Generate CREATE TABLE SQL."""
        column_defs = []
        for col_name, col_type in columns.items():
            column_defs.append(f"    {col_name} {col_type}")

        return f"""
CREATE TABLE {table_name} (
{','.join(column_defs)}
);
"""

    @staticmethod
    def add_column(table_name: str, column_name: str, column_type: str) -> str:
        """Generate ADD COLUMN SQL."""
        return f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};"

    @staticmethod
    def drop_column(table_name: str, column_name: str) -> str:
        """Generate DROP COLUMN SQL."""
        return f"ALTER TABLE {table_name} DROP COLUMN {column_name};"

    @staticmethod
    def create_index(table_name: str, index_name: str, columns: List[str]) -> str:
        """Generate CREATE INDEX SQL."""
        cols_str = ', '.join(columns)
        return f"CREATE INDEX {index_name} ON {table_name} ({cols_str});"

    @staticmethod
    def drop_index(index_name: str) -> str:
        """Generate DROP INDEX SQL."""
        return f"DROP INDEX {index_name};"

    @staticmethod
    def add_foreign_key(table_name: str, fk_name: str, column: str,
                       ref_table: str, ref_column: str) -> str:
        """Generate ADD FOREIGN KEY SQL."""
        return f"""
ALTER TABLE {table_name}
ADD CONSTRAINT {fk_name}
FOREIGN KEY ({column}) REFERENCES {ref_table}({ref_column});
"""

    @staticmethod
    def generate_rollback(up_sql: str) -> Optional[str]:
        """Attempt to generate rollback SQL from up SQL."""
        # This is a simplified implementation
        # Production systems would need more sophisticated parsing

        up_lower = up_sql.strip().lower()

        if up_lower.startswith("create table"):
            # Extract table name
            lines = up_sql.strip().split('\n')
            if lines and lines[0].strip().startswith("CREATE TABLE"):
                table_name = lines[0].split()[2].strip('();')
                return f"DROP TABLE {table_name};"

        elif "add column" in up_lower:
            # Extract table and column
            parts = up_sql.replace(";", "").split()
            table_idx = parts.index("table") + 1
            column_idx = parts.index("column") + 1
            if table_idx < len(parts) and column_idx < len(parts):
                table_name = parts[table_idx]
                column_name = parts[column_idx]
                return f"ALTER TABLE {table_name} DROP COLUMN {column_name};"

        elif up_lower.startswith("create index"):
            # Extract index name
            parts = up_sql.split()
            if len(parts) >= 3:
                index_name = parts[2]
                return f"DROP INDEX {index_name};"

        # For complex migrations, return None (manual rollback required)
        return None
