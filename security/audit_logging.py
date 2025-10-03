"""
Advanced Audit Logging System for Negative Space Imaging
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import json
import time
import os
import hashlib
import hmac
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("AuditSystem")


class AuditLogger:
    """Advanced audit logging system with integrity verification."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        log_dir: Optional[str] = None
    ):
        self.config_path = config_path or 'security_config.json'
        self.log_dir = log_dir or 'audit_logs'

        # Load configuration
        self.config = self._load_config()

        # Initialize logging directory
        self._init_log_directory()

        # Set up integrity tracking
        self.integrity_key = os.urandom(32)
        self.previous_hash = None

        # Thread safety
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=self.config['audit']['max_workers']
        )

        logger.info("Audit logging system initialized")

    def _load_config(self) -> Dict:
        """Load audit system configuration."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                    if 'audit' not in config:
                        config['audit'] = self._default_audit_config()

                        with open(self.config_path, 'w') as f:
                            json.dump(config, f, indent=4)

                    return config

            # Create default config
            config = {'audit': self._default_audit_config()}

            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)

            return config

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _default_audit_config(self) -> Dict:
        """Create default audit configuration."""
        return {
            'max_log_size': 10 * 1024 * 1024,  # 10MB
            'max_log_age': 30 * 24 * 60 * 60,  # 30 days
            'rotation_interval': 24 * 60 * 60,  # 24 hours
            'compression_enabled': True,
            'encryption_enabled': True,
            'integrity_check_interval': 3600,  # 1 hour
            'max_workers': 4,
            'severity_levels': [
                'INFO',
                'WARNING',
                'ERROR',
                'CRITICAL',
                'SECURITY'
            ]
        }

    def _init_log_directory(self):
        """Initialize audit log directory structure."""
        try:
            log_path = Path(self.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # Create subdirectories for different severity levels
            for level in self.config['audit']['severity_levels']:
                (log_path / level.lower()).mkdir(exist_ok=True)

        except Exception as e:
            logger.error(f"Error initializing log directory: {e}")
            raise

    def _get_log_file_path(
        self,
        severity: str,
        timestamp: Optional[float] = None
    ) -> Path:
        """Get appropriate log file path."""
        if timestamp is None:
            timestamp = time.time()

        date_str = datetime.fromtimestamp(
            timestamp,
            tz=timezone.utc
        ).strftime('%Y-%m-%d')

        return Path(self.log_dir) / severity.lower() / f"{date_str}.log"

    def _create_log_entry(
        self,
        event: str,
        severity: str,
        details: Dict,
        source: str,
        timestamp: Optional[float] = None
    ) -> Dict:
        """Create structured log entry with integrity hash."""
        if timestamp is None:
            timestamp = time.time()

        entry = {
            'timestamp': timestamp,
            'event': event,
            'severity': severity.upper(),
            'source': source,
            'details': details,
            'metadata': {
                'hostname': os.uname().nodename,
                'process_id': os.getpid(),
                'thread_id': threading.get_ident()
            }
        }

        # Add integrity hash
        entry_bytes = json.dumps(entry, sort_keys=True).encode()
        if self.previous_hash:
            entry_bytes += self.previous_hash

        hash_obj = hmac.new(
            self.integrity_key,
            entry_bytes,
            hashlib.sha3_512
        )
        entry['integrity_hash'] = hash_obj.hexdigest()

        return entry

    def log_event(
        self,
        event: str,
        severity: str,
        details: Dict,
        source: str,
        sync: bool = False
    ):
        """Log an audit event."""
        try:
            if severity.upper() not in self.config['audit']['severity_levels']:
                raise ValueError(f"Invalid severity level: {severity}")

            timestamp = time.time()
            entry = self._create_log_entry(
                event,
                severity,
                details,
                source,
                timestamp
            )

            if sync:
                self._write_log_entry(entry)
            else:
                self.executor.submit(self._write_log_entry, entry)

        except Exception as e:
            logger.error(f"Error logging event: {e}")
            raise

    def _write_log_entry(self, entry: Dict):
        """Write log entry to appropriate file."""
        try:
            log_file = self._get_log_file_path(
                entry['severity'],
                entry['timestamp']
            )

            with self.lock:
                # Check file size and rotate if needed
                if (
                    log_file.exists() and
                    log_file.stat().st_size >= self.config['audit']['max_log_size']
                ):
                    self._rotate_log(log_file)

                # Write entry
                with open(log_file, 'a') as f:
                    json.dump(entry, f)
                    f.write('\n')

                # Update previous hash
                self.previous_hash = bytes.fromhex(entry['integrity_hash'])

        except Exception as e:
            logger.error(f"Error writing log entry: {e}")
            raise

    def _rotate_log(self, log_file: Path):
        """Rotate log file and compress old logs."""
        try:
            if not log_file.exists():
                return

            # Generate rotated filename
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            rotated_file = log_file.with_suffix(f'.{timestamp}.log')

            # Rotate file
            log_file.rename(rotated_file)

            # Compress if enabled
            if self.config['audit']['compression_enabled']:
                import gzip
                with open(rotated_file, 'rb') as f_in:
                    with gzip.open(
                        str(rotated_file) + '.gz',
                        'wb'
                    ) as f_out:
                        f_out.write(f_in.read())

                # Remove uncompressed file
                rotated_file.unlink()

            logger.info(f"Rotated log file: {log_file}")

        except Exception as e:
            logger.error(f"Error rotating log file: {e}")
            raise

    def verify_integrity(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> bool:
        """Verify integrity of audit logs."""
        try:
            if start_time is None:
                start_time = time.time() - self.config['audit']['max_log_age']
            if end_time is None:
                end_time = time.time()

            # Get all log files in time range
            log_files = []
            for severity in self.config['audit']['severity_levels']:
                severity_dir = Path(self.log_dir) / severity.lower()
                if not severity_dir.exists():
                    continue

                for log_file in severity_dir.glob('*.log'):
                    try:
                        # Extract date from filename
                        date_str = log_file.stem
                        file_time = datetime.strptime(
                            date_str,
                            '%Y-%m-%d'
                        ).timestamp()

                        if start_time <= file_time <= end_time:
                            log_files.append(log_file)
                    except ValueError:
                        continue

            # Verify each file
            for log_file in log_files:
                if not self._verify_file_integrity(log_file):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error verifying integrity: {e}")
            return False

    def _verify_file_integrity(self, log_file: Path) -> bool:
        """Verify integrity of a single log file."""
        try:
            previous_hash = None

            with open(log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)

                    # Verify hash chain
                    entry_copy = entry.copy()
                    stored_hash = entry_copy.pop('integrity_hash')

                    entry_bytes = json.dumps(
                        entry_copy,
                        sort_keys=True
                    ).encode()
                    if previous_hash:
                        entry_bytes += previous_hash

                    hash_obj = hmac.new(
                        self.integrity_key,
                        entry_bytes,
                        hashlib.sha3_512
                    )
                    calculated_hash = hash_obj.hexdigest()

                    if calculated_hash != stored_hash:
                        logger.error(
                            f"Integrity check failed for {log_file}"
                        )
                        return False

                    previous_hash = bytes.fromhex(stored_hash)

            return True

        except Exception as e:
            logger.error(f"Error verifying file integrity: {e}")
            return False

    def cleanup_old_logs(self):
        """Clean up old log files."""
        try:
            cutoff_time = time.time() - self.config['audit']['max_log_age']

            for severity in self.config['audit']['severity_levels']:
                severity_dir = Path(self.log_dir) / severity.lower()
                if not severity_dir.exists():
                    continue

                for log_file in severity_dir.glob('*.log*'):
                    try:
                        # Extract date from filename
                        date_str = log_file.stem.split('.')[0]
                        file_time = datetime.strptime(
                            date_str,
                            '%Y-%m-%d'
                        ).timestamp()

                        if file_time < cutoff_time:
                            log_file.unlink()
                            logger.info(f"Cleaned up old log file: {log_file}")
                    except ValueError:
                        continue

        except Exception as e:
            logger.error(f"Error cleaning up logs: {e}")
            raise

    def search_logs(
        self,
        query: Dict[str, any],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        max_results: Optional[int] = None
    ) -> List[Dict]:
        """Search audit logs with criteria."""
        try:
            if start_time is None:
                start_time = time.time() - self.config['audit']['max_log_age']
            if end_time is None:
                end_time = time.time()

            results = []

            # Search through log files
            for severity in self.config['audit']['severity_levels']:
                severity_dir = Path(self.log_dir) / severity.lower()
                if not severity_dir.exists():
                    continue

                for log_file in severity_dir.glob('*.log'):
                    try:
                        # Check file timeframe
                        date_str = log_file.stem
                        file_time = datetime.strptime(
                            date_str,
                            '%Y-%m-%d'
                        ).timestamp()

                        if not (start_time <= file_time <= end_time):
                            continue

                        # Search file
                        with open(log_file, 'r') as f:
                            for line in f:
                                entry = json.loads(line)

                                # Check if entry matches query
                                if self._matches_query(entry, query):
                                    results.append(entry)

                                    if (
                                        max_results and
                                        len(results) >= max_results
                                    ):
                                        return results
                    except ValueError:
                        continue

            return results

        except Exception as e:
            logger.error(f"Error searching logs: {e}")
            return []

    def _matches_query(self, entry: Dict, query: Dict) -> bool:
        """Check if log entry matches search query."""
        for key, value in query.items():
            if key not in entry:
                return False

            if isinstance(value, dict):
                if not self._matches_query(entry[key], value):
                    return False
            elif entry[key] != value:
                return False

        return True


# Example usage
if __name__ == "__main__":
    # Create audit logger
    audit = AuditLogger()

    # Log some events
    audit.log_event(
        "USER_LOGIN",
        "SECURITY",
        {
            "user_id": "alice",
            "ip_address": "192.168.1.100",
            "success": True
        },
        "auth_service"
    )

    audit.log_event(
        "FILE_ACCESS",
        "INFO",
        {
            "user_id": "bob",
            "file_path": "/data/images/001.jpg",
            "operation": "READ"
        },
        "file_service"
    )

    # Verify integrity
    if audit.verify_integrity():
        print("Log integrity verified")

    # Search logs
    results = audit.search_logs({
        "severity": "SECURITY",
        "details": {"user_id": "alice"}
    })
    print("Search results:", results)
