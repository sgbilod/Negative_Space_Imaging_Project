"""
Database Backup and Recovery System with Automated Scheduling
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import gzip
import hashlib
import threading
import schedule

logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """Backup configuration settings."""
    backup_dir: str
    retention_days: int = 30
    compression_level: int = 6  # gzip compression level
    max_parallel_backups: int = 2
    verify_backups: bool = True
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None


@dataclass
class BackupJob:
    """Backup job definition."""
    job_id: str
    name: str
    database_name: str
    backup_type: str  # 'full', 'incremental', 'differential'
    schedule: str  # cron-like or 'daily', 'weekly', 'monthly'
    config: BackupConfig
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: str = "pending"


@dataclass
class BackupResult:
    """Result of a backup operation."""
    job_id: str
    success: bool
    backup_path: Optional[str] = None
    file_size: int = 0
    duration: float = 0.0
    checksum: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RestoreResult:
    """Result of a restore operation."""
    success: bool
    database_name: str
    backup_path: str
    duration: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class BackupManager:
    """Manages database backups with scheduling and retention."""

    def __init__(self, connection_string: str, config: BackupConfig):
        self.connection_string = connection_string
        self.config = config
        self.backup_jobs: Dict[str, BackupJob] = {}
        self.backup_history: List[BackupResult] = []
        self.scheduler_active = False
        self.scheduler_thread: Optional[threading.Thread] = None

        # Ensure backup directory exists
        Path(config.backup_dir).mkdir(parents=True, exist_ok=True)

        # Initialize scheduler
        self.scheduler = schedule.Scheduler()

    def create_backup_job(self, name: str, database_name: str, backup_type: str = "full",
                         schedule: str = "daily") -> str:
        """Create a new backup job."""
        job_id = f"{name}_{int(time.time())}"

        job = BackupJob(
            job_id=job_id,
            name=name,
            database_name=database_name,
            backup_type=backup_type,
            schedule=schedule,
            config=self.config
        )

        self.backup_jobs[job_id] = job
        self._schedule_job(job)

        logger.info(f"Created backup job: {job_id}")
        return job_id

    def _schedule_job(self, job: BackupJob):
        """Schedule a backup job."""
        if job.schedule == "daily":
            self.scheduler.every().day.at("02:00").do(self._run_backup_job, job.job_id)
        elif job.schedule == "weekly":
            self.scheduler.every().week.do(self._run_backup_job, job.job_id)
        elif job.schedule == "monthly":
            self.scheduler.every(30).days.do(self._run_backup_job, job.job_id)
        elif job.schedule.startswith("cron:"):
            # Parse cron-like schedule (simplified)
            # In production, use a proper cron parser
            pass
        else:
            logger.warning(f"Unknown schedule format: {job.schedule}")

    def start_scheduler(self):
        """Start the backup scheduler."""
        if self.scheduler_active:
            return

        self.scheduler_active = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Backup scheduler started")

    def stop_scheduler(self):
        """Stop the backup scheduler."""
        self.scheduler_active = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Backup scheduler stopped")

    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.scheduler_active:
            self.scheduler.run_pending()
            time.sleep(60)  # Check every minute

    def _run_backup_job(self, job_id: str):
        """Execute a scheduled backup job."""
        if job_id not in self.backup_jobs:
            logger.error(f"Backup job not found: {job_id}")
            return

        job = self.backup_jobs[job_id]
        if not job.enabled:
            return

        logger.info(f"Running scheduled backup: {job.name}")
        result = self.perform_backup(job.database_name, job.backup_type, job.name)

        job.last_run = datetime.now()
        self.backup_history.append(result)

        if result.success:
            logger.info(f"Scheduled backup completed: {job.name}")
        else:
            logger.error(f"Scheduled backup failed: {job.name} - {result.error_message}")

    def perform_backup(self, database_name: str, backup_type: str = "full",
                      job_name: str = "manual") -> BackupResult:
        """Perform a database backup."""
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{database_name}_{backup_type}_{timestamp}.sql"

        if self.config.compression_level > 0:
            backup_filename += ".gz"

        backup_path = os.path.join(self.config.backup_dir, backup_filename)

        try:
            # Create pg_dump command
            cmd = ["pg_dump", database_name, "-f", backup_path]

            if backup_type == "schema_only":
                cmd.append("--schema-only")
            elif backup_type == "data_only":
                cmd.append("--data-only")

            # Add compression if enabled
            if self.config.compression_level > 0:
                cmd.extend(["--compress", str(self.config.compression_level)])

            # Execute backup
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env={**os.environ, "PGPASSWORD": self._extract_password()}
            )

            if result.returncode != 0:
                raise Exception(f"pg_dump failed: {result.stderr}")

            # Calculate file size and checksum
            file_size = os.path.getsize(backup_path)
            checksum = self._calculate_checksum(backup_path)

            duration = time.time() - start_time

            # Verify backup if enabled
            if self.config.verify_backups:
                self._verify_backup(backup_path, database_name)

            # Clean up old backups
            self._cleanup_old_backups(database_name)

            backup_result = BackupResult(
                job_id=f"{job_name}_{timestamp}",
                success=True,
                backup_path=backup_path,
                file_size=file_size,
                duration=duration,
                checksum=checksum
            )

            logger.info(f"Backup completed: {backup_path} ({file_size} bytes, {duration:.2f}s)")
            return backup_result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Backup failed: {error_msg}")

            # Clean up failed backup file
            if os.path.exists(backup_path):
                os.remove(backup_path)

            return BackupResult(
                job_id=f"{job_name}_{timestamp}",
                success=False,
                duration=duration,
                error_message=error_msg
            )

    def _extract_password(self) -> str:
        """Extract password from connection string."""
        # Simplified - in production, use proper parsing
        if "password=" in self.connection_string:
            return self.connection_string.split("password=")[1].split()[0]
        return ""

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of backup file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _verify_backup(self, backup_path: str, database_name: str):
        """Verify backup integrity by attempting to restore to a test database."""
        test_db = f"test_restore_{int(time.time())}"

        try:
            # Create test database
            subprocess.run(
                ["createdb", test_db],
                capture_output=True,
                check=True,
                env={**os.environ, "PGPASSWORD": self._extract_password()}
            )

            # Attempt restore
            cmd = ["pg_restore" if backup_path.endswith('.gz') else "psql",
                   "-d", test_db, "-f", backup_path]

            result = subprocess.run(
                cmd,
                capture_output=True,
                env={**os.environ, "PGPASSWORD": self._extract_password()}
            )

            if result.returncode != 0:
                raise Exception(f"Backup verification failed: {result.stderr}")

            logger.info(f"Backup verified successfully: {backup_path}")

        except Exception as e:
            logger.warning(f"Backup verification failed: {e}")
            raise
        finally:
            # Clean up test database
            try:
                subprocess.run(
                    ["dropdb", test_db],
                    capture_output=True,
                    env={**os.environ, "PGPASSWORD": self._extract_password()}
                )
            except:
                pass

    def _cleanup_old_backups(self, database_name: str):
        """Clean up old backup files based on retention policy."""
        try:
            backup_dir = Path(self.config.backup_dir)
            pattern = f"{database_name}_*.sql*"

            backups = []
            for file_path in backup_dir.glob(pattern):
                if file_path.is_file():
                    stat = file_path.stat()
                    backups.append({
                        'path': file_path,
                        'mtime': datetime.fromtimestamp(stat.st_mtime)
                    })

            # Sort by modification time (newest first)
            backups.sort(key=lambda x: x['mtime'], reverse=True)

            # Remove backups older than retention period
            cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)

            for backup in backups[self.config.max_parallel_backups:]:  # Keep at least max_parallel_backups
                if backup['mtime'] < cutoff_date:
                    backup['path'].unlink()
                    logger.info(f"Removed old backup: {backup['path']}")

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

    def perform_restore(self, backup_path: str, target_database: str,
                       drop_existing: bool = False) -> RestoreResult:
        """Restore a database from backup."""
        start_time = time.time()

        try:
            # Drop existing database if requested
            if drop_existing:
                subprocess.run(
                    ["dropdb", target_database],
                    capture_output=True,
                    env={**os.environ, "PGPASSWORD": self._extract_password()}
                )

            # Create target database
            subprocess.run(
                ["createdb", target_database],
                capture_output=True,
                check=True,
                env={**os.environ, "PGPASSWORD": self._extract_password()}
            )

            # Restore from backup
            if backup_path.endswith('.gz'):
                # Decompress and restore
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(backup_path[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                temp_file = backup_path[:-3]
                cmd = ["psql", "-d", target_database, "-f", temp_file]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    env={**os.environ, "PGPASSWORD": self._extract_password()}
                )

                # Clean up temp file
                os.remove(temp_file)
            else:
                cmd = ["psql", "-d", target_database, "-f", backup_path]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    env={**os.environ, "PGPASSWORD": self._extract_password()}
                )

            if result.returncode != 0:
                raise Exception(f"Restore failed: {result.stderr}")

            duration = time.time() - start_time

            logger.info(f"Database restored: {target_database} from {backup_path} ({duration:.2f}s)")

            return RestoreResult(
                success=True,
                database_name=target_database,
                backup_path=backup_path,
                duration=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Restore failed: {error_msg}")

            return RestoreResult(
                success=False,
                database_name=target_database,
                backup_path=backup_path,
                duration=duration,
                error_message=error_msg
            )

    def list_backups(self, database_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available backup files."""
        backup_dir = Path(self.config.backup_dir)
        backups = []

        pattern = f"{database_name}_*.sql*" if database_name else "*.sql*"

        for file_path in backup_dir.glob(pattern):
            if file_path.is_file():
                stat = file_path.stat()
                backups.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_mtime),
                    'database': database_name or file_path.name.split('_')[0]
                })

        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x['created'], reverse=True)
        return backups

    def get_backup_status(self) -> Dict[str, Any]:
        """Get comprehensive backup status."""
        all_backups = self.list_backups()

        # Group by database
        by_database = {}
        for backup in all_backups:
            db = backup['database']
            if db not in by_database:
                by_database[db] = []
            by_database[db].append(backup)

        # Calculate statistics
        total_size = sum(b['size'] for b in all_backups)
        oldest_backup = min((b['created'] for b in all_backups), default=None)
        newest_backup = max((b['created'] for b in all_backups), default=None)

        return {
            'total_backups': len(all_backups),
            'total_size_bytes': total_size,
            'total_size_human': self._format_bytes(total_size),
            'oldest_backup': oldest_backup,
            'newest_backup': newest_backup,
            'backups_by_database': by_database,
            'scheduled_jobs': len([j for j in self.backup_jobs.values() if j.enabled]),
            'recent_history': [
                {
                    'job_id': h.job_id,
                    'success': h.success,
                    'timestamp': h.timestamp,
                    'duration': h.duration,
                    'file_size': h.file_size
                }
                for h in self.backup_history[-10:]  # Last 10 results
            ]
        }

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return ".1f"
            bytes_value /= 1024.0
        return ".1f"

    def export_backup_metadata(self, output_file: str):
        """Export backup metadata to JSON file."""
        metadata = {
            'exported_at': datetime.now().isoformat(),
            'backups': self.list_backups(),
            'jobs': [
                {
                    'job_id': job.job_id,
                    'name': job.name,
                    'database': job.database_name,
                    'type': job.backup_type,
                    'schedule': job.schedule,
                    'enabled': job.enabled,
                    'last_run': job.last_run.isoformat() if job.last_run else None
                }
                for job in self.backup_jobs.values()
            ],
            'history': [
                {
                    'job_id': h.job_id,
                    'success': h.success,
                    'timestamp': h.timestamp.isoformat(),
                    'duration': h.duration,
                    'file_size': h.file_size,
                    'checksum': h.checksum
                }
                for h in self.backup_history
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Backup metadata exported to: {output_file}")


class DisasterRecoveryManager:
    """Manages disaster recovery procedures."""

    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
        self.recovery_procedures: Dict[str, Callable] = {}

    def register_recovery_procedure(self, scenario: str, procedure: Callable):
        """Register a recovery procedure for a specific scenario."""
        self.recovery_procedures[scenario] = procedure
        logger.info(f"Registered recovery procedure: {scenario}")

    def execute_recovery(self, scenario: str, **kwargs) -> Dict[str, Any]:
        """Execute a disaster recovery procedure."""
        if scenario not in self.recovery_procedures:
            raise ValueError(f"Unknown recovery scenario: {scenario}")

        logger.info(f"Executing disaster recovery: {scenario}")

        try:
            result = self.recovery_procedures[scenario](**kwargs)
            logger.info(f"Disaster recovery completed: {scenario}")
            return {'success': True, 'result': result}
        except Exception as e:
            logger.error(f"Disaster recovery failed: {scenario} - {e}")
            return {'success': False, 'error': str(e)}

    def create_point_in_time_recovery(self, target_time: datetime, target_database: str):
        """Create a point-in-time recovery procedure."""
        # This would implement WAL-based point-in-time recovery
        # Simplified implementation for demonstration
        pass

    def validate_recovery_procedures(self) -> Dict[str, bool]:
        """Validate that all recovery procedures are properly configured."""
        results = {}
        for scenario, procedure in self.recovery_procedures.items():
            try:
                # Basic validation - check if callable
                results[scenario] = callable(procedure)
            except Exception:
                results[scenario] = False

        return results
