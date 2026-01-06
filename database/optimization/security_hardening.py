"""
Database Security Hardening System
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import hashlib
import secrets
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import sql
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import hmac
import os

logger = logging.getLogger(__name__)


@dataclass
class SecurityPolicy:
    """Database security policy configuration."""
    name: str
    description: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    user: Optional[str] = None
    ip_address: Optional[str] = None
    database: Optional[str] = None
    query: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncryptionKey:
    """Encryption key metadata."""
    key_id: str
    key_type: str  # 'data', 'backup', 'session'
    algorithm: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    status: str = "active"  # 'active', 'expired', 'revoked'


class KeyManager:
    """Manages encryption keys for database security."""

    def __init__(self, key_store_path: str = "/var/lib/postgresql/keys"):
        self.key_store_path = key_store_path
        self.master_key = self._load_or_generate_master_key()
        self.keys: Dict[str, EncryptionKey] = {}
        self._load_keys()

    def _load_or_generate_master_key(self) -> bytes:
        """Load or generate the master encryption key."""
        master_key_path = os.path.join(self.key_store_path, "master.key")

        if os.path.exists(master_key_path):
            with open(master_key_path, 'rb') as f:
                return f.read()
        else:
            # Generate new master key
            master_key = secrets.token_bytes(32)
            os.makedirs(self.key_store_path, exist_ok=True)

            with open(master_key_path, 'wb') as f:
                f.write(master_key)

            # Set restrictive permissions
            os.chmod(master_key_path, 0o600)

            return master_key

    def _load_keys(self):
        """Load existing keys from storage."""
        # In practice, this would load from encrypted storage
        pass

    def generate_key(self, key_type: str, algorithm: str = "AES256",
                    lifetime_days: int = 365) -> EncryptionKey:
        """Generate a new encryption key."""
        key_id = f"{key_type}_{int(time.time())}_{secrets.token_hex(4)}"

        key = EncryptionKey(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            expires_at=datetime.now() + timedelta(days=lifetime_days)
        )

        self.keys[key_id] = key

        # Store encrypted key
        self._store_key(key)

        logger.info(f"Generated new {key_type} key: {key_id}")
        return key

    def _store_key(self, key: EncryptionKey):
        """Store key securely."""
        # In practice, encrypt and store the actual key material
        pass

    def get_active_key(self, key_type: str) -> Optional[EncryptionKey]:
        """Get the active key for a given type."""
        active_keys = [
            key for key in self.keys.values()
            if key.key_type == key_type and key.status == "active"
            and (not key.expires_at or key.expires_at > datetime.now())
        ]

        return active_keys[0] if active_keys else None

    def rotate_key(self, key_type: str) -> EncryptionKey:
        """Rotate to a new key for the given type."""
        # Mark old key as expired
        old_key = self.get_active_key(key_type)
        if old_key:
            old_key.status = "expired"

        # Generate new key
        new_key = self.generate_key(key_type)

        logger.info(f"Rotated {key_type} key: {old_key.key_id if old_key else 'None'} -> {new_key.key_id}")
        return new_key


class DataEncryption:
    """Handles data encryption/decryption operations."""

    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager

    def encrypt_data(self, data: str, key_type: str = "data") -> str:
        """Encrypt sensitive data."""
        key = self.key_manager.get_active_key(key_type)
        if not key:
            raise ValueError(f"No active {key_type} key available")

        # In practice, use the actual key material
        fernet = Fernet(base64.urlsafe_b64encode(self.key_manager.master_key))

        encrypted = fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_data(self, encrypted_data: str, key_type: str = "data") -> str:
        """Decrypt sensitive data."""
        key = self.key_manager.get_active_key(key_type)
        if not key:
            raise ValueError(f"No active {key_type} key available")

        fernet = Fernet(base64.urlsafe_b64encode(self.key_manager.master_key))

        encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = fernet.decrypt(encrypted)
        return decrypted.decode()


class SQLInjectionProtector:
    """Protects against SQL injection attacks."""

    def __init__(self):
        self.suspicious_patterns = [
            r';\s*--',  # Semicolon followed by comment
            r';\s*/\*',  # Semicolon followed by block comment
            r'union\s+select',  # UNION SELECT
            r';\s*drop\s+table',  # DROP TABLE
            r';\s*delete\s+from',  # DELETE FROM
            r';\s*update.*set',  # UPDATE statements
            r';\s*insert\s+into',  # INSERT statements
            r'--\s*sp_password',  # SQL Server password change
            r';\s*exec\s*\(',  # EXEC statements
            r';\s*xp_',  # Extended stored procedures
        ]

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query for potential SQL injection."""
        import re

        findings = []
        risk_level = "low"

        for pattern in self.suspicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                findings.append({
                    'pattern': pattern,
                    'severity': 'high' if 'drop' in pattern.lower() or 'delete' in pattern.lower() else 'medium'
                })
                if 'high' in [f['severity'] for f in findings]:
                    risk_level = "high"

        return {
            'query': query,
            'risk_level': risk_level,
            'findings': findings,
            'safe': len(findings) == 0
        }

    def sanitize_query(self, query: str, parameters: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Sanitize a query using parameterized statements."""
        # Convert named parameters to positional
        param_values = []
        sanitized_query = query

        for key, value in parameters.items():
            placeholder = f":{key}"
            if placeholder in sanitized_query:
                sanitized_query = sanitized_query.replace(placeholder, "%s", 1)
                param_values.append(value)

        return sanitized_query, param_values


class AccessControlManager:
    """Manages database access control and permissions."""

    def __init__(self):
        self.roles: Dict[str, Dict[str, Any]] = {}
        self.permissions: Dict[str, List[str]] = {}
        self.session_limits: Dict[str, int] = {}

    def create_role(self, role_name: str, permissions: List[str],
                   session_limit: int = 10) -> bool:
        """Create a new database role with permissions."""
        if role_name in self.roles:
            return False

        self.roles[role_name] = {
            'permissions': permissions,
            'created_at': datetime.now(),
            'session_limit': session_limit
        }

        self.permissions[role_name] = permissions
        self.session_limits[role_name] = session_limit

        logger.info(f"Created role: {role_name} with permissions: {permissions}")
        return True

    def check_permission(self, role: str, permission: str) -> bool:
        """Check if a role has a specific permission."""
        return permission in self.permissions.get(role, [])

    def validate_session_limit(self, role: str, active_sessions: int) -> bool:
        """Validate session limit for a role."""
        limit = self.session_limits.get(role, 10)
        return active_sessions < limit

    def audit_access(self, user: str, action: str, resource: str,
                    success: bool) -> SecurityEvent:
        """Create an audit event for access attempts."""
        event = SecurityEvent(
            event_type="access_attempt",
            severity="low" if success else "medium",
            user=user,
            details={
                'action': action,
                'resource': resource,
                'success': success
            }
        )

        if not success:
            logger.warning(f"Access denied: {user} -> {action} on {resource}")

        return event


class AuditLogger:
    """Comprehensive audit logging for database activities."""

    def __init__(self, log_path: str = "/var/log/postgresql/audit.log"):
        self.log_path = log_path
        self.audit_events: List[SecurityEvent] = []

    def log_event(self, event: SecurityEvent):
        """Log a security event."""
        self.audit_events.append(event)

        # Write to file
        log_entry = self._format_log_entry(event)
        with open(self.log_path, 'a') as f:
            f.write(log_entry + '\n')

        # Log to system logger based on severity
        if event.severity == "critical":
            logger.critical(log_entry)
        elif event.severity == "high":
            logger.error(log_entry)
        elif event.severity == "medium":
            logger.warning(log_entry)
        else:
            logger.info(log_entry)

    def _format_log_entry(self, event: SecurityEvent) -> str:
        """Format a security event for logging."""
        return (
            f"{event.timestamp.isoformat()} | "
            f"{event.severity.upper()} | "
            f"{event.event_type} | "
            f"user={event.user or 'unknown'} | "
            f"ip={event.ip_address or 'unknown'} | "
            f"db={event.database or 'unknown'} | "
            f"query={event.query[:100] if event.query else 'none'} | "
            f"details={event.details}"
        )

    def get_events(self, start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  event_type: Optional[str] = None,
                  severity: Optional[str] = None) -> List[SecurityEvent]:
        """Retrieve audit events with filtering."""
        events = self.audit_events

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if severity:
            events = [e for e in events if e.severity == severity]

        return events

    def generate_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate a security audit report."""
        events = self.get_events(start_time, end_time)

        severity_counts = {}
        event_type_counts = {}
        user_activity = {}

        for event in events:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1

            if event.user:
                user_activity[event.user] = user_activity.get(event.user, 0) + 1

        return {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'total_events': len(events),
            'severity_breakdown': severity_counts,
            'event_type_breakdown': event_type_counts,
            'user_activity': user_activity,
            'critical_events': [e for e in events if e.severity == "critical"]
        }


class SecurityMonitor:
    """Monitors database security in real-time."""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.policies: Dict[str, SecurityPolicy] = {}
        self.alerts: List[Dict[str, Any]] = []

    def add_policy(self, policy: SecurityPolicy):
        """Add a security policy."""
        self.policies[policy.name] = policy
        logger.info(f"Added security policy: {policy.name}")

    def check_query(self, query: str, user: str, database: str) -> List[str]:
        """Check a query against security policies."""
        violations = []

        for policy in self.policies.values():
            if not policy.enabled:
                continue

            if policy.name == "sql_injection_protection":
                protector = SQLInjectionProtector()
                analysis = protector.analyze_query(query)
                if not analysis['safe']:
                    violations.append(f"SQL injection risk: {analysis['risk_level']}")

            elif policy.name == "query_complexity_limit":
                # Check query complexity
                if len(query.split()) > policy.parameters.get('max_words', 100):
                    violations.append("Query too complex")

            elif policy.name == "table_access_restriction":
                restricted_tables = policy.parameters.get('restricted_tables', [])
                for table in restricted_tables:
                    if table.lower() in query.lower():
                        violations.append(f"Access to restricted table: {table}")

        # Log violations
        for violation in violations:
            event = SecurityEvent(
                event_type="policy_violation",
                severity="high",
                user=user,
                database=database,
                query=query,
                details={'violation': violation}
            )
            self.audit_logger.log_event(event)

        return violations

    def monitor_connection(self, user: str, ip_address: str, database: str):
        """Monitor database connection."""
        event = SecurityEvent(
            event_type="connection",
            severity="low",
            user=user,
            ip_address=ip_address,
            database=database
        )
        self.audit_logger.log_event(event)

    def detect_anomaly(self, user: str, query_pattern: str, frequency: int):
        """Detect anomalous database activity."""
        # Simple anomaly detection based on frequency
        threshold = 100  # queries per minute

        if frequency > threshold:
            event = SecurityEvent(
                event_type="anomalous_activity",
                severity="medium",
                user=user,
                details={
                    'pattern': query_pattern,
                    'frequency': frequency,
                    'threshold': threshold
                }
            )
            self.audit_logger.log_event(event)
            self.alerts.append({
                'type': 'anomaly',
                'message': f"Anomalous activity detected for user {user}",
                'timestamp': datetime.now().isoformat()
            })


class DatabaseFirewall:
    """Database firewall for connection control."""

    def __init__(self):
        self.whitelist: List[str] = []
        self.blacklist: List[str] = []
        self.rate_limits: Dict[str, Dict[str, Any]] = {}

    def add_to_whitelist(self, ip_address: str):
        """Add IP address to whitelist."""
        if ip_address not in self.whitelist:
            self.whitelist.append(ip_address)
            logger.info(f"Added {ip_address} to whitelist")

    def add_to_blacklist(self, ip_address: str):
        """Add IP address to blacklist."""
        if ip_address not in self.blacklist:
            self.blacklist.append(ip_address)
            logger.info(f"Added {ip_address} to blacklist")

    def set_rate_limit(self, ip_address: str, max_requests: int, window_seconds: int):
        """Set rate limit for an IP address."""
        self.rate_limits[ip_address] = {
            'max_requests': max_requests,
            'window_seconds': window_seconds,
            'requests': [],
            'blocked_until': None
        }

    def check_connection(self, ip_address: str) -> Tuple[bool, str]:
        """Check if connection is allowed."""
        # Check blacklist
        if ip_address in self.blacklist:
            return False, "IP address blacklisted"

        # Check whitelist (if whitelist is enabled)
        if self.whitelist and ip_address not in self.whitelist:
            return False, "IP address not whitelisted"

        # Check rate limit
        if ip_address in self.rate_limits:
            limit = self.rate_limits[ip_address]

            # Check if currently blocked
            if limit['blocked_until'] and datetime.now() < limit['blocked_until']:
                return False, "Rate limit exceeded"

            # Clean old requests
            cutoff = datetime.now() - timedelta(seconds=limit['window_seconds'])
            limit['requests'] = [r for r in limit['requests'] if r > cutoff]

            # Check if under limit
            if len(limit['requests']) >= limit['max_requests']:
                limit['blocked_until'] = datetime.now() + timedelta(minutes=5)
                return False, "Rate limit exceeded"

            # Add current request
            limit['requests'].append(datetime.now())

        return True, "Connection allowed"


class SecurityManager:
    """Central security manager coordinating all security components."""

    def __init__(self):
        self.key_manager = KeyManager()
        self.data_encryption = DataEncryption(self.key_manager)
        self.sql_protector = SQLInjectionProtector()
        self.access_manager = AccessControlManager()
        self.audit_logger = AuditLogger()
        self.security_monitor = SecurityMonitor(self.audit_logger)
        self.firewall = DatabaseFirewall()

        self._setup_default_policies()

    def _setup_default_policies(self):
        """Set up default security policies."""
        policies = [
            SecurityPolicy(
                name="sql_injection_protection",
                description="Protect against SQL injection attacks",
                enabled=True
            ),
            SecurityPolicy(
                name="query_complexity_limit",
                description="Limit query complexity to prevent resource exhaustion",
                enabled=True,
                parameters={'max_words': 100}
            ),
            SecurityPolicy(
                name="table_access_restriction",
                description="Restrict access to sensitive tables",
                enabled=True,
                parameters={'restricted_tables': ['user_credentials', 'audit_logs']}
            )
        ]

        for policy in policies:
            self.security_monitor.add_policy(policy)

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data before storage."""
        return self.data_encryption.encrypt_data(data)

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data after retrieval."""
        return self.data_encryption.decrypt_data(encrypted_data)

    def validate_query(self, query: str, user: str, database: str) -> Tuple[bool, List[str]]:
        """Validate a query for security violations."""
        violations = self.security_monitor.check_query(query, user, database)
        return len(violations) == 0, violations

    def check_connection_access(self, user: str, ip_address: str, database: str) -> bool:
        """Check if connection is allowed."""
        # Check firewall
        allowed, reason = self.firewall.check_connection(ip_address)
        if not allowed:
            event = SecurityEvent(
                event_type="connection_blocked",
                severity="medium",
                user=user,
                ip_address=ip_address,
                database=database,
                details={'reason': reason}
            )
            self.audit_logger.log_event(event)
            return False

        # Check access control
        if not self.access_manager.validate_session_limit(user, 1):  # Simplified
            event = SecurityEvent(
                event_type="session_limit_exceeded",
                severity="medium",
                user=user,
                ip_address=ip_address,
                database=database
            )
            self.audit_logger.log_event(event)
            return False

        # Log successful connection
        self.security_monitor.monitor_connection(user, ip_address, database)
        return True

    def generate_security_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate a comprehensive security report."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        audit_report = self.audit_logger.generate_report(start_time, end_time)

        return {
            'audit_report': audit_report,
            'active_policies': [
                {
                    'name': policy.name,
                    'description': policy.description,
                    'enabled': policy.enabled
                }
                for policy in self.security_monitor.policies.values()
            ],
            'firewall_status': {
                'whitelisted_ips': len(self.firewall.whitelist),
                'blacklisted_ips': len(self.firewall.blacklist),
                'rate_limited_ips': len(self.firewall.rate_limits)
            },
            'encryption_keys': [
                {
                    'key_id': key.key_id,
                    'type': key.key_type,
                    'status': key.status,
                    'expires_at': key.expires_at.isoformat() if key.expires_at else None
                }
                for key in self.key_manager.keys.values()
            ],
            'generated_at': datetime.now().isoformat()
        }
