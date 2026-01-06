"""
Intrusion Detection System (IDS) Module
========================================

Detects suspicious patterns and anomalies in system behavior:
- Brute force detection
- Port scanning detection
- Data exfiltration detection
- Command injection attempts
- Privilege escalation attempts

Uses signature matching and statistical anomaly detection.

Author: @CIPHER - Advanced Cryptography & Security
Date: December 2025
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import json
from pathlib import Path

logger = logging.getLogger("intrusion_detection")


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 0
    WARNING = 1
    ALERT = 2
    CRITICAL = 3


@dataclass
class SuspiciousActivity:
    """Record of suspicious activity."""
    timestamp: datetime
    activity_type: str
    description: str
    severity: AlertSeverity
    source: str  # IP, user, process, etc.
    context: Dict  # Additional context
    confidence: float  # 0.0 to 1.0


class BruteForceSensor:
    """Detects brute force attacks."""

    def __init__(self, threshold: int = 5, window_seconds: int = 300):
        """
        Initialize brute force sensor.

        Args:
            threshold: Number of failures before alert
            window_seconds: Time window for failures
        """
        self.threshold = threshold
        self.window_seconds = window_seconds
        self._attempts = defaultdict(list)
        self._lock = threading.RLock()

    def record_attempt(self, identifier: str, success: bool) -> bool:
        """
        Record an attempt and check for brute force.

        Args:
            identifier: User/IP identifier
            success: Whether attempt was successful

        Returns:
            True if brute force detected
        """
        with self._lock:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=self.window_seconds)

            # Clean old attempts
            self._attempts[identifier] = [
                t for t in self._attempts[identifier]
                if t > window_start
            ]

            if not success:
                self._attempts[identifier].append(now)

            # Check threshold
            failure_count = len(self._attempts[identifier])
            if failure_count >= self.threshold:
                logger.warning(
                    f"Brute force detected for {identifier}: "
                    f"{failure_count} failures in {self.window_seconds}s"
                )
                return True

        return False


class PortScanSensor:
    """Detects port scanning activity."""

    def __init__(self, threshold: int = 10, window_seconds: int = 60):
        """
        Initialize port scan sensor.

        Args:
            threshold: Number of connection attempts for alert
            window_seconds: Time window
        """
        self.threshold = threshold
        self.window_seconds = window_seconds
        self._connections = defaultdict(list)
        self._lock = threading.RLock()

    def record_connection_attempt(
        self,
        source_ip: str,
        target_port: int
    ) -> bool:
        """
        Record connection attempt and check for port scanning.

        Args:
            source_ip: Source IP address
            target_port: Target port

        Returns:
            True if port scanning detected
        """
        with self._lock:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=self.window_seconds)

            # Clean old connections
            self._connections[source_ip] = [
                (t, p) for t, p in self._connections[source_ip]
                if t > window_start
            ]

            # Record new connection
            self._connections[source_ip].append((now, target_port))

            # Check for port scanning (multiple ports in short time)
            unique_ports = set(p for _, p in self._connections[source_ip])

            if len(unique_ports) >= self.threshold:
                logger.warning(
                    f"Port scanning detected from {source_ip}: "
                    f"{len(unique_ports)} different ports in {self.window_seconds}s"
                )
                return True

        return False


class DataExfiltrationSensor:
    """Detects potential data exfiltration."""

    def __init__(self, threshold_bytes: int = 100 * 1024 * 1024):
        """
        Initialize data exfiltration sensor.

        Args:
            threshold_bytes: Threshold for alert (default 100 MB)
        """
        self.threshold_bytes = threshold_bytes
        self._outbound_traffic = defaultdict(lambda: {"total": 0, "timestamp": None})
        self._lock = threading.RLock()

    def record_outbound_traffic(
        self,
        source_ip: str,
        bytes_transferred: int
    ) -> bool:
        """
        Record outbound traffic and check for data exfiltration.

        Args:
            source_ip: Source IP
            bytes_transferred: Bytes transferred

        Returns:
            True if exfiltration suspected
        """
        with self._lock:
            self._outbound_traffic[source_ip]["total"] += bytes_transferred
            self._outbound_traffic[source_ip]["timestamp"] = datetime.utcnow()

            total = self._outbound_traffic[source_ip]["total"]

            if total > self.threshold_bytes:
                logger.critical(
                    f"Data exfiltration suspected from {source_ip}: "
                    f"{total / 1024 / 1024:.2f} MB transferred"
                )
                return True

        return False


class CommandInjectionSensor:
    """Detects command injection attempts."""

    # Dangerous command patterns
    DANGEROUS_PATTERNS = [
        r";\s*(cat|rm|chmod|dd|wget|curl|nc|bash|sh)",
        r"\|\s*(cat|grep|sed|awk|nc|bash|sh)",
        r"`.*`",  # Command substitution
        r"\$\(.*\)",  # Command substitution
        r"&&\s*(cat|rm|chmod|bash|sh)",
    ]

    def __init__(self):
        """Initialize command injection sensor."""
        import re
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]

    def check_input(self, user_input: str) -> Tuple[bool, Optional[str]]:
        """
        Check user input for command injection patterns.

        Args:
            user_input: User-provided input

        Returns:
            Tuple of (is_suspicious, matched_pattern)
        """
        for pattern in self.patterns:
            if pattern.search(user_input):
                logger.warning(f"Command injection attempt detected: {user_input[:100]}")
                return True, pattern.pattern

        return False, None


class PrivilegeEscalationSensor:
    """Detects privilege escalation attempts."""

    def __init__(self):
        """Initialize privilege escalation sensor."""
        self._suspicious_commands = defaultdict(list)
        self._lock = threading.RLock()

    def check_command(
        self,
        user: str,
        command: str,
        user_privilege_level: int
    ) -> bool:
        """
        Check if command represents privilege escalation attempt.

        Args:
            user: Username
            command: Executed command
            user_privilege_level: Current privilege level (0=root, 1=user, etc.)

        Returns:
            True if escalation attempt suspected
        """
        # Commands that should only be run by root/admin
        privileged_commands = ['sudo', 'su', 'chmod', 'chown', 'iptables', 'visudo']

        is_privileged_command = any(cmd in command for cmd in privileged_commands)

        if is_privileged_command and user_privilege_level > 0:
            with self._lock:
                self._suspicious_commands[user].append({
                    "timestamp": datetime.utcnow(),
                    "command": command
                })

            logger.warning(
                f"Privilege escalation attempt detected for {user}: {command}"
            )
            return True

        return False


class IntrusionDetectionSystem:
    """
    Main IDS system integrating all sensors.
    """

    def __init__(self, alerts_dir: str = "./security/alerts"):
        """Initialize IDS."""
        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sensors
        self.brute_force_sensor = BruteForceSensor()
        self.port_scan_sensor = PortScanSensor()
        self.exfiltration_sensor = DataExfiltrationSensor()
        self.injection_sensor = CommandInjectionSensor()
        self.escalation_sensor = PrivilegeEscalationSensor()

        # Alert storage
        self._alerts: List[SuspiciousActivity] = []
        self._lock = threading.RLock()

        logger.info("Initialized IntrusionDetectionSystem")

    def alert(
        self,
        activity_type: str,
        description: str,
        severity: AlertSeverity,
        source: str,
        context: Optional[Dict] = None,
        confidence: float = 1.0
    ) -> None:
        """
        Generate an alert for suspicious activity.

        Args:
            activity_type: Type of suspicious activity
            description: Description of activity
            severity: Alert severity
            source: Source (IP, user, process, etc.)
            context: Additional context
            confidence: Confidence score (0.0-1.0)
        """
        alert = SuspiciousActivity(
            timestamp=datetime.utcnow(),
            activity_type=activity_type,
            description=description,
            severity=severity,
            source=source,
            context=context or {},
            confidence=confidence
        )

        with self._lock:
            self._alerts.append(alert)

        # Log based on severity
        if severity == AlertSeverity.CRITICAL:
            logger.critical(f"🚨 CRITICAL: {description} from {source}")
        elif severity == AlertSeverity.ALERT:
            logger.error(f"⚠️  ALERT: {description} from {source}")
        else:
            logger.warning(f"⚠️  {severity.name}: {description} from {source}")

        # Write to file
        self._write_alert_to_file(alert)

    def _write_alert_to_file(self, alert: SuspiciousActivity) -> None:
        """Write alert to file for review."""
        try:
            alert_file = self.alerts_dir / f"alerts_{datetime.utcnow().strftime('%Y%m%d')}.jsonl\"\n            alert_entry = {\n                \"timestamp\": alert.timestamp.isoformat(),\n                \"activity_type\": alert.activity_type,\n                \"description\": alert.description,\n                \"severity\": alert.severity.name,\n                \"source\": alert.source,\n                \"confidence\": alert.confidence,\n                \"context\": alert.context\n            }\n            \n            with open(alert_file, 'a') as f:\n                f.write(json.dumps(alert_entry) + '\\n')\n        \n        except Exception as e:\n            logger.error(f\"Error writing alert to file: {e}\")\n    \n    def get_alerts(\n        self,\n        severity_threshold: AlertSeverity = AlertSeverity.WARNING,\n        hours: int = 24\n    ) -> List[SuspiciousActivity]:\n        \"\"\"\n        Get recent alerts above severity threshold.\n        \n        Args:\n            severity_threshold: Minimum severity level\n            hours: Look back how many hours\n            \n        Returns:\n            List of alerts\n        \"\"\"\n        cutoff_time = datetime.utcnow() - timedelta(hours=hours)\n        \n        with self._lock:\n            return [\n                alert for alert in self._alerts\n                if alert.severity.value >= severity_threshold.value\n                and alert.timestamp > cutoff_time\n            ]\n    \n    def get_alert_summary(self) -> Dict:\n        \"\"\"\n        Get summary of recent alerts.\n        \n        Returns:\n            Summary dictionary\n        \"\"\"\n        with self._lock:\n            summary = {\n                \"timestamp\": datetime.utcnow().isoformat(),\n                \"total_alerts\": len(self._alerts),\n                \"by_severity\": {},\n                \"by_type\": {}\n            }\n            \n            for alert in self._alerts:\n                # Count by severity\n                sev = alert.severity.name\n                summary[\"by_severity\"][sev] = summary[\"by_severity\"].get(sev, 0) + 1\n                \n                # Count by type\n                atype = alert.activity_type\n                summary[\"by_type\"][atype] = summary[\"by_type\"].get(atype, 0) + 1\n            \n            return summary\n\n\nif __name__ == \"__main__\":\n    logging.basicConfig(level=logging.INFO)\n    \n    print(\"\\n=== Intrusion Detection System Demo ===\\n\")\n    \n    ids = IntrusionDetectionSystem()\n    \n    # Simulate brute force attack\n    print(\"1. Detecting brute force...\")\n    for i in range(6):\n        is_detected = ids.brute_force_sensor.record_attempt(\"attacker\", False)\n        if is_detected:\n            ids.alert(\n                \"brute_force\",\n                \"Multiple failed login attempts detected\",\n                AlertSeverity.CRITICAL,\n                \"192.168.1.100\"\n            )\n    \n    # Simulate port scanning\n    print(\"2. Detecting port scanning...\")\n    for port in range(22, 32):\n        is_detected = ids.port_scan_sensor.record_connection_attempt(\"192.168.1.50\", port)\n        if is_detected:\n            ids.alert(\n                \"port_scan\",\n                \"Port scanning detected\",\n                AlertSeverity.ALERT,\n                \"192.168.1.50\"\n            )\n    \n    # Check for command injection\n    print(\"3. Detecting command injection...\")\n    is_injection = ids.injection_sensor.check_input(\"filename'; rm -rf /; --\")\n    if is_injection[0]:\n        ids.alert(\n            \"command_injection\",\n            \"Command injection attempt detected\",\n            AlertSeverity.CRITICAL,\n            \"user_input\"\n        )\n    \n    # Summary\n    print(\"\\n4. Alert Summary:\")\n    summary = ids.get_alert_summary()\n    print(f\"   Total Alerts: {summary['total_alerts']}\")\n    print(f\"   By Severity: {summary['by_severity']}\")\n    \n    print(\"\\n=== Demo Complete ===\")\n"
