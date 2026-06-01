"""
Security Event Monitoring System for Negative Space Imaging
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import json
import time
import os
import hashlib
import threading
import queue
from typing import Dict, List, Optional, Union, Callable
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('security.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SecurityMonitor")


class EventSeverity(Enum):
    """Security event severity levels."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class EventCategory(Enum):
    """Security event categories."""
    AUTHENTICATION = auto()
    AUTHORIZATION = auto()
    DATA_ACCESS = auto()
    CONFIGURATION = auto()
    SYSTEM = auto()
    NETWORK = auto()
    MALWARE = auto()
    AUDIT = auto()


@dataclass
class SecurityEvent:
    """Security event definition."""
    timestamp: float
    severity: EventSeverity
    category: EventCategory
    source: str
    event_type: str
    description: str
    details: Dict
    correlation_id: Optional[str] = None
    metadata: Optional[Dict] = None


class SecurityMonitor:
    """Security event monitoring and analysis system."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        event_dir: Optional[str] = None
    ):
        self.config_path = config_path or 'security_config.json'
        self.event_dir = event_dir or 'security_events'

        # Load configuration
        self.config = self._load_config()

        # Initialize event directory
        self._init_event_directory()

        # Set up event processing
        self.event_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(
            max_workers=self.config['monitor']['max_workers']
        )

        # Set up alert handlers
        self.alert_handlers = {}
        self._register_default_handlers()

        # Start monitoring
        self._start_monitoring()

        logger.info("Security monitoring system initialized")

    def _load_config(self) -> Dict:
        """Load monitoring configuration."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                    if 'monitor' not in config:
                        config['monitor'] = self._default_monitor_config()

                        with open(self.config_path, 'w') as f:
                            json.dump(config, f, indent=4)

                    return config

            # Create default config
            config = {'monitor': self._default_monitor_config()}

            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)

            return config

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _default_monitor_config(self) -> Dict:
        """Create default monitoring configuration."""
        return {
            'max_workers': 4,
            'queue_size': 10000,
            'batch_size': 100,
            'processing_interval': 1.0,
            'retention_days': 90,
            'compression_enabled': True,
            'alert_thresholds': {
                'authentication_failures': 5,
                'unauthorized_access': 3,
                'system_errors': 10
            },
            'correlation_window': 300,
            'correlation_enabled': True
        }

    def _init_event_directory(self):
        """Initialize event directory structure."""
        try:
            event_path = Path(self.event_dir)
            event_path.mkdir(parents=True, exist_ok=True)

            # Create subdirectories for different severities
            for severity in EventSeverity:
                (event_path / severity.name.lower()).mkdir(exist_ok=True)

        except Exception as e:
            logger.error(f"Error initializing event directory: {e}")
            raise

    def _register_default_handlers(self):
        """Register default alert handlers."""
        self.register_alert_handler(
            EventCategory.AUTHENTICATION,
            self._handle_authentication_alert
        )
        self.register_alert_handler(
            EventCategory.AUTHORIZATION,
            self._handle_authorization_alert
        )
        self.register_alert_handler(
            EventCategory.SYSTEM,
            self._handle_system_alert
        )

    def register_alert_handler(
        self,
        category: EventCategory,
        handler: Callable[[SecurityEvent], None]
    ):
        """Register custom alert handler for event category."""
        self.alert_handlers[category] = handler
        logger.info(f"Registered alert handler for {category.name}")

    def _start_monitoring(self):
        """Start security monitoring threads."""
        self.running = True

        # Start event processor
        self.processor_thread = threading.Thread(
            target=self._process_events
        )
        self.processor_thread.daemon = True
        self.processor_thread.start()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_old_events
        )
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()

    def stop_monitoring(self):
        """Stop security monitoring."""
        self.running = False
        self.processor_thread.join()
        self.cleanup_thread.join()
        self.executor.shutdown()

    def record_event(self, event: SecurityEvent):
        """Record security event for processing."""
        try:
            # Add correlation ID if needed
            if event.correlation_id is None:
                event.correlation_id = self._generate_correlation_id(event)

            # Queue event for processing
            self.event_queue.put(event)

        except Exception as e:
            logger.error(f"Error recording event: {e}")
            raise

    def _process_events(self):
        """Process security events from queue."""
        while self.running:
            try:
                events = []
                # Collect batch of events
                for _ in range(self.config['monitor']['batch_size']):
                    try:
                        event = self.event_queue.get_nowait()
                        events.append(event)
                    except queue.Empty:
                        break

                if not events:
                    time.sleep(self.config['monitor']['processing_interval'])
                    continue

                # Process events in parallel
                self.executor.map(self._handle_event, events)

            except Exception as e:
                logger.error(f"Error processing events: {e}")

    def _handle_event(self, event: SecurityEvent):
        """Handle individual security event."""
        try:
            # Store event
            self._store_event(event)

            # Check for alerts
            self._check_alerts(event)

            # Perform correlation if enabled
            if self.config['monitor']['correlation_enabled']:
                self._correlate_events(event)

        except Exception as e:
            logger.error(f"Error handling event: {e}")

    def _store_event(self, event: SecurityEvent):
        """Store security event to file."""
        try:
            # Generate filename
            timestamp = datetime.fromtimestamp(
                event.timestamp,
                tz=timezone.utc
            )
            date_str = timestamp.strftime('%Y-%m-%d')
            severity_dir = Path(self.event_dir) / event.severity.name.lower()
            event_file = severity_dir / f"{date_str}.json"

            # Convert event to dict
            event_dict = {
                'timestamp': event.timestamp,
                'severity': event.severity.name,
                'category': event.category.name,
                'source': event.source,
                'event_type': event.event_type,
                'description': event.description,
                'details': event.details,
                'correlation_id': event.correlation_id,
                'metadata': event.metadata or {}
            }

            # Append to file
            with open(event_file, 'a') as f:
                json.dump(event_dict, f)
                f.write('\n')

        except Exception as e:
            logger.error(f"Error storing event: {e}")
            raise

    def _check_alerts(self, event: SecurityEvent):
        """Check if event triggers any alerts."""
        try:
            # Get appropriate handler
            handler = self.alert_handlers.get(event.category)
            if handler:
                handler(event)

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    def _handle_authentication_alert(self, event: SecurityEvent):
        """Handle authentication-related alerts."""
        if event.event_type == "LOGIN_FAILURE":
            # Check for brute force attempts
            user_id = event.details.get('user_id')
            if user_id:
                recent_failures = self._count_recent_events(
                    category=EventCategory.AUTHENTICATION,
                    event_type="LOGIN_FAILURE",
                    details={'user_id': user_id},
                    window=300  # 5 minutes
                )

                if recent_failures >= self.config['monitor']['alert_thresholds']['authentication_failures']:
                    self._raise_alert(
                        severity=EventSeverity.HIGH,
                        category=EventCategory.AUTHENTICATION,
                        description=f"Possible brute force attack on user {user_id}",
                        details={
                            'user_id': user_id,
                            'failure_count': recent_failures,
                            'source_ip': event.details.get('source_ip')
                        }
                    )

    def _handle_authorization_alert(self, event: SecurityEvent):
        """Handle authorization-related alerts."""
        if event.event_type == "ACCESS_DENIED":
            # Check for unauthorized access attempts
            user_id = event.details.get('user_id')
            resource = event.details.get('resource')

            if user_id and resource:
                recent_denials = self._count_recent_events(
                    category=EventCategory.AUTHORIZATION,
                    event_type="ACCESS_DENIED",
                    details={'user_id': user_id},
                    window=300
                )

                if recent_denials >= self.config['monitor']['alert_thresholds']['unauthorized_access']:
                    self._raise_alert(
                        severity=EventSeverity.HIGH,
                        category=EventCategory.AUTHORIZATION,
                        description=f"Multiple unauthorized access attempts by user {user_id}",
                        details={
                            'user_id': user_id,
                            'resource': resource,
                            'denial_count': recent_denials
                        }
                    )

    def _handle_system_alert(self, event: SecurityEvent):
        """Handle system-related alerts."""
        if event.event_type == "SYSTEM_ERROR":
            # Check for system issues
            component = event.details.get('component')
            if component:
                recent_errors = self._count_recent_events(
                    category=EventCategory.SYSTEM,
                    event_type="SYSTEM_ERROR",
                    details={'component': component},
                    window=300
                )

                if recent_errors >= self.config['monitor']['alert_thresholds']['system_errors']:
                    self._raise_alert(
                        severity=EventSeverity.HIGH,
                        category=EventCategory.SYSTEM,
                        description=f"Multiple system errors in component {component}",
                        details={
                            'component': component,
                            'error_count': recent_errors
                        }
                    )

    def _count_recent_events(
        self,
        category: EventCategory,
        event_type: str,
        details: Dict,
        window: int
    ) -> int:
        """Count recent events matching criteria."""
        try:
            count = 0
            now = time.time()
            severity_dir = Path(self.event_dir) / category.name.lower()

            # Get today's file
            date_str = datetime.fromtimestamp(
                now,
                tz=timezone.utc
            ).strftime('%Y-%m-%d')
            event_file = severity_dir / f"{date_str}.json"

            if not event_file.exists():
                return 0

            # Count matching events
            with open(event_file, 'r') as f:
                for line in f:
                    event = json.loads(line)
                    if (
                        event['category'] == category.name and
                        event['event_type'] == event_type and
                        all(
                            event['details'].get(k) == v
                            for k, v in details.items()
                        ) and
                        now - event['timestamp'] <= window
                    ):
                        count += 1

            return count

        except Exception as e:
            logger.error(f"Error counting events: {e}")
            return 0

    def _raise_alert(
        self,
        severity: EventSeverity,
        category: EventCategory,
        description: str,
        details: Dict
    ):
        """Raise security alert."""
        alert = SecurityEvent(
            timestamp=time.time(),
            severity=severity,
            category=category,
            source="SecurityMonitor",
            event_type="SECURITY_ALERT",
            description=description,
            details=details
        )

        self.record_event(alert)
        logger.warning(f"Security Alert: {description}")

    def _correlate_events(self, event: SecurityEvent):
        """Perform event correlation analysis."""
        if not event.correlation_id:
            return

        try:
            # Get related events
            related = self._find_related_events(
                event.correlation_id,
                window=self.config['monitor']['correlation_window']
            )

            if len(related) > 1:
                # Analyze event pattern
                if self._detect_attack_pattern(related):
                    self._raise_alert(
                        severity=EventSeverity.HIGH,
                        category=EventCategory.SYSTEM,
                        description="Detected suspicious event pattern",
                        details={
                            'correlation_id': event.correlation_id,
                            'event_count': len(related),
                            'pattern': self._describe_pattern(related)
                        }
                    )

        except Exception as e:
            logger.error(f"Error correlating events: {e}")

    def _find_related_events(
        self,
        correlation_id: str,
        window: int
    ) -> List[Dict]:
        """Find events with same correlation ID."""
        try:
            related = []
            now = time.time()

            # Search all severity directories
            for severity in EventSeverity:
                severity_dir = Path(self.event_dir) / severity.name.lower()

                # Get today's file
                date_str = datetime.fromtimestamp(
                    now,
                    tz=timezone.utc
                ).strftime('%Y-%m-%d')
                event_file = severity_dir / f"{date_str}.json"

                if not event_file.exists():
                    continue

                # Find related events
                with open(event_file, 'r') as f:
                    for line in f:
                        event = json.loads(line)
                        if (
                            event['correlation_id'] == correlation_id and
                            now - event['timestamp'] <= window
                        ):
                            related.append(event)

            return related

        except Exception as e:
            logger.error(f"Error finding related events: {e}")
            return []

    def _detect_attack_pattern(self, events: List[Dict]) -> bool:
        """Detect suspicious patterns in related events."""
        try:
            # Example pattern detection logic
            auth_failures = sum(
                1 for e in events
                if (
                    e['category'] == 'AUTHENTICATION' and
                    e['event_type'] == 'LOGIN_FAILURE'
                )
            )

            access_denials = sum(
                1 for e in events
                if (
                    e['category'] == 'AUTHORIZATION' and
                    e['event_type'] == 'ACCESS_DENIED'
                )
            )

            return (
                auth_failures >= 3 or
                access_denials >= 2 or
                (auth_failures + access_denials) >= 4
            )

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return False

    def _describe_pattern(self, events: List[Dict]) -> Dict:
        """Create human-readable description of event pattern."""
        try:
            pattern = {
                'sequence': [],
                'timing': {
                    'start': min(e['timestamp'] for e in events),
                    'end': max(e['timestamp'] for e in events),
                    'duration': max(e['timestamp'] for e in events) - min(e['timestamp'] for e in events)
                },
                'categories': {},
                'sources': set()
            }

            # Analyze events
            for event in sorted(events, key=lambda e: e['timestamp']):
                pattern['sequence'].append({
                    'timestamp': event['timestamp'],
                    'category': event['category'],
                    'type': event['event_type']
                })

                pattern['categories'][event['category']] = pattern['categories'].get(event['category'], 0) + 1
                pattern['sources'].add(event['source'])

            pattern['sources'] = list(pattern['sources'])
            return pattern

        except Exception as e:
            logger.error(f"Error describing pattern: {e}")
            return {}

    def _generate_correlation_id(self, event: SecurityEvent) -> str:
        """Generate correlation ID for event."""
        try:
            # Create hash from relevant fields
            hasher = hashlib.sha256()
            hasher.update(event.source.encode())
            hasher.update(event.event_type.encode())

            # Add relevant details
            for key in sorted(event.details.keys()):
                value = str(event.details[key])
                hasher.update(f"{key}:{value}".encode())

            return hasher.hexdigest()[:16]

        except Exception as e:
            logger.error(f"Error generating correlation ID: {e}")
            return os.urandom(8).hex()

    def _cleanup_old_events(self):
        """Clean up old event files."""
        while self.running:
            try:
                cutoff_time = time.time() - (
                    self.config['monitor']['retention_days'] * 24 * 3600
                )

                # Check all severity directories
                for severity in EventSeverity:
                    severity_dir = Path(self.event_dir) / severity.name.lower()
                    if not severity_dir.exists():
                        continue

                    for event_file in severity_dir.glob('*.json*'):
                        try:
                            # Extract date from filename
                            date_str = event_file.stem.split('.')[0]
                            file_time = datetime.strptime(
                                date_str,
                                '%Y-%m-%d'
                            ).replace(tzinfo=timezone.utc).timestamp()

                            if file_time < cutoff_time:
                                event_file.unlink()
                                logger.info(f"Cleaned up old event file: {event_file}")
                        except ValueError:
                            continue

                # Sleep for a day
                time.sleep(24 * 3600)

            except Exception as e:
                logger.error(f"Error cleaning up events: {e}")
                time.sleep(3600)  # Retry in an hour


# Example usage
if __name__ == "__main__":
    # Create security monitor
    monitor = SecurityMonitor()

    # Record some test events
    monitor.record_event(SecurityEvent(
        timestamp=time.time(),
        severity=EventSeverity.HIGH,
        category=EventCategory.AUTHENTICATION,
        source="auth_service",
        event_type="LOGIN_FAILURE",
        description="Failed login attempt",
        details={
            'user_id': "test_user",
            'source_ip': "192.168.1.100",
            'reason': "invalid_password"
        }
    ))

    monitor.record_event(SecurityEvent(
        timestamp=time.time(),
        severity=EventSeverity.MEDIUM,
        category=EventCategory.AUTHORIZATION,
        source="resource_manager",
        event_type="ACCESS_DENIED",
        description="Unauthorized resource access",
        details={
            'user_id': "test_user",
            'resource': "/secure/data",
            'required_permission': "ADMIN"
        }
    ))

    # Let monitor process events
    time.sleep(2)

    # Stop monitoring
    monitor.stop_monitoring()
