"""
Exponential Backoff Brute Force Protection
===========================================

Implements exponential lockout duration for failed authentication attempts.
Protects against credential stuffing and password guessing attacks.

Features:
- Exponential backoff: 30min → 60min → 120min → 240min
- Honeypot endpoint to trap attackers
- Complete security audit logging
- Rate limiting dashboard metrics

Philosophy: Attackers should pay an exponentially increasing cost for each failed attempt.

Author: @CIPHER - Advanced Cryptography & Security
Date: December 2025
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import json
from pathlib import Path

logger = logging.getLogger("brute_force_protection")


class LockoutLevel(Enum):
    """Lockout severity levels."""
    NONE = 0              # No lockout
    LIGHT = 1             # 30 minutes
    MODERATE = 2          # 60 minutes
    SEVERE = 3            # 120 minutes
    CRITICAL = 4          # 240 minutes
    PERMANENT = 5         # Permanent lockout


@dataclass
class FailureRecord:
    """Record of failed authentication attempt."""
    timestamp: datetime
    identifier: str        # Username, IP, email, etc.
    attempt_type: str      # "password", "mfa", "api_key", etc.
    failure_reason: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class LockoutState:
    """Current lockout state for an identifier."""
    identifier: str
    level: LockoutLevel
    consecutive_failures: int
    last_failure_time: datetime
    lockout_until: Optional[datetime] = None
    failure_history: list = None  # List of FailureRecord
    honeypot_activated: bool = False

    def __post_init__(self):
        if self.failure_history is None:
            self.failure_history = []


class ExponentialBackoffLockout:
    """
    Implements exponential backoff lockout for failed authentication attempts.

    Lockout Schedule:
    ├─ 1st failure: 30 minutes
    ├─ 2nd failure: 60 minutes
    ├─ 3rd failure: 120 minutes
    ├─ 4th failure: 240 minutes
    └─ 5+ failures: Permanent lockout (admin review required)
    """

    # Exponential backoff durations in minutes
    LOCKOUT_DURATIONS = [
        30,      # 1st failure: 30 minutes
        60,      # 2nd failure: 60 minutes
        120,     # 3rd failure: 120 minutes
        240,     # 4th failure: 240 minutes
        None     # 5+ failures: Permanent
    ]

    def __init__(self, audit_log_dir: str = "./security/audit"):
        """
        Initialize brute force protection.

        Args:
            audit_log_dir: Directory for security audit logs
        """
        self.audit_log_dir = Path(audit_log_dir)
        self.audit_log_dir.mkdir(parents=True, exist_ok=True)

        # In-memory lockout states
        self._lockout_states: Dict[str, LockoutState] = {}
        self._lock = threading.RLock()

        logger.info("Initialized ExponentialBackoffLockout")

    def record_failure(
        self,
        identifier: str,
        attempt_type: str = "password",
        failure_reason: str = "Invalid credentials",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Tuple[bool, str, Optional[datetime]]:
        """
        Record a failed authentication attempt and apply lockout if needed.

        Args:
            identifier: Username, email, or other unique identifier
            attempt_type: Type of authentication attempt
            failure_reason: Reason for failure
            ip_address: Source IP address
            user_agent: User agent string

        Returns:
            Tuple of:
            - is_locked_out (bool): Whether account is now locked
            - lockout_reason (str): Reason for lockout
            - lockout_until (datetime): When lockout expires
        """
        with self._lock:
            # Get or create lockout state
            if identifier not in self._lockout_states:
                self._lockout_states[identifier] = LockoutState(
                    identifier=identifier,
                    level=LockoutLevel.NONE,
                    consecutive_failures=0,
                    last_failure_time=datetime.utcnow()
                )

            state = self._lockout_states[identifier]

            # Record failure
            failure_record = FailureRecord(
                timestamp=datetime.utcnow(),
                identifier=identifier,
                attempt_type=attempt_type,
                failure_reason=failure_reason,
                ip_address=ip_address,
                user_agent=user_agent
            )
            state.failure_history.append(failure_record)
            state.last_failure_time = datetime.utcnow()
            state.consecutive_failures += 1

            # Activate honeypot for excessive failures
            if state.consecutive_failures >= 3:
                state.honeypot_activated = True

            # Determine lockout level and duration
            failure_index = min(state.consecutive_failures - 1, len(self.LOCKOUT_DURATIONS) - 1)
            duration_minutes = self.LOCKOUT_DURATIONS[failure_index]

            if duration_minutes is None:
                # Permanent lockout
                state.level = LockoutLevel.PERMANENT
                state.lockout_until = None
                lockout_reason = f"Permanent lockout after {state.consecutive_failures} failures"
            else:
                # Temporary lockout with exponential backoff
                state.lockout_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
                state.level = LockoutLevel(min(state.consecutive_failures, len(LockoutLevel) - 1))
                lockout_reason = f"Locked for {duration_minutes} minutes (attempt {state.consecutive_failures})"

            # Log to audit trail
            self._log_failure(identifier, state, failure_record, lockout_reason)

            logger.warning(
                f"Failed auth attempt for {identifier}: "
                f"{failure_reason} (consecutive: {state.consecutive_failures}, "
                f"lockout: {lockout_reason})"
            )

            return True, lockout_reason, state.lockout_until

    def is_locked_out(self, identifier: str) -> Tuple[bool, Optional[datetime]]:
        """
        Check if an identifier is currently locked out.

        Args:
            identifier: Username or identifier to check

        Returns:
            Tuple of (is_locked_out, lockout_until)
        """
        with self._lock:
            if identifier not in self._lockout_states:
                return False, None

            state = self._lockout_states[identifier]

            # Check permanent lockout
            if state.level == LockoutLevel.PERMANENT:
                return True, None

            # Check temporary lockout
            if state.lockout_until and datetime.utcnow() < state.lockout_until:
                return True, state.lockout_until

            # Lockout expired
            if state.lockout_until and datetime.utcnow() >= state.lockout_until:
                state.consecutive_failures = 0
                state.level = LockoutLevel.NONE
                state.lockout_until = None
                logger.info(f"Lockout expired for {identifier}")

            return False, None

    def record_success(self, identifier: str) -> None:
        """
        Record a successful authentication to reset failure counter.

        Args:
            identifier: Username or identifier
        """
        with self._lock:
            if identifier in self._lockout_states:
                state = self._lockout_states[identifier]
                if state.consecutive_failures > 0:
                    logger.info(f"Reset failure counter for {identifier}")
                state.consecutive_failures = 0
                state.level = LockoutLevel.NONE
                state.honeypot_activated = False

    def should_activate_honeypot(self, identifier: str) -> bool:
        """
        Determine if honeypot should be activated for this identifier.

        Honeypot endpoints waste attacker's time without revealing real system.

        Args:
            identifier: Username or identifier

        Returns:
            True if honeypot should be activated
        """
        with self._lock:
            if identifier in self._lockout_states:
                state = self._lockout_states[identifier]
                return state.honeypot_activated and state.consecutive_failures >= 3
        return False

    def get_lockout_status(self, identifier: str) -> Dict:
        """
        Get detailed lockout status for an identifier.

        Returns:
            Dictionary with lockout information
        """
        with self._lock:
            if identifier not in self._lockout_states:
                return {
                    "locked_out": False,
                    "consecutive_failures": 0,
                    "level": "NONE"
                }

            state = self._lockout_states[identifier]
            is_locked, lockout_until = self.is_locked_out(identifier)

            return {
                "locked_out": is_locked,
                "consecutive_failures": state.consecutive_failures,
                "level": state.level.name,
                "last_failure_time": state.last_failure_time.isoformat(),
                "lockout_until": lockout_until.isoformat() if lockout_until else None,
                "honeypot_activated": state.honeypot_activated,
                "failure_count": len(state.failure_history)
            }

    def _log_failure(
        self,
        identifier: str,
        state: LockoutState,
        failure_record: FailureRecord,
        lockout_reason: str
    ) -> None:
        """Log failed authentication attempt to audit trail."""
        try:
            # Create audit log entry
            audit_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "authentication_failure",
                "identifier": identifier,
                "attempt_type": failure_record.attempt_type,
                "failure_reason": failure_record.failure_reason,
                "ip_address": failure_record.ip_address,
                "consecutive_failures": state.consecutive_failures,
                "lockout_level": state.level.name,
                "lockout_reason": lockout_reason
            }

            # Write to audit log file
            audit_file = self.audit_log_dir / "authentication_failures.jsonl"
            with open(audit_file, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')

            logger.info(f"Logged failure to {audit_file}")

        except Exception as e:
            logger.error(f"Error writing to audit log: {e}")

    def get_audit_log(self, identifier: Optional[str] = None) -> list:
        """
        Retrieve audit log entries.

        Args:
            identifier: Filter by identifier (optional)

        Returns:
            List of audit log entries
        """
        audit_file = self.audit_log_dir / "authentication_failures.jsonl"
        entries = []

        try:
            if audit_file.exists():
                with open(audit_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            if identifier is None or entry.get('identifier') == identifier:
                                entries.append(entry)
        except Exception as e:
            logger.error(f"Error reading audit log: {e}")

        return entries

    def generate_security_report(self) -> Dict:
        """
        Generate security dashboard report.

        Returns:
            Dictionary with security metrics
        """
        with self._lock:
            locked_out_count = sum(
                1 for identifier in self._lockout_states
                if self.is_locked_out(identifier)[0]
            )

            honeypot_active_count = sum(
                1 for state in self._lockout_states.values()
                if state.honeypot_activated
            )

            total_failures = sum(
                len(state.failure_history)
                for state in self._lockout_states.values()
            )

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_tracked_identifiers": len(self._lockout_states),
                "currently_locked_out": locked_out_count,
                "honeypot_active": honeypot_active_count,
                "total_failures_recorded": total_failures,
                "high_risk_identifiers": [
                    identifier for identifier, state in self._lockout_states.items()
                    if state.consecutive_failures >= 5
                ]
            }


class HoneypotEndpoint:
    """
    Honeypot endpoint that mimics real API but logs all interactions.

    Wastes attacker's time and resources while providing forensic data.
    """

    def __init__(self, brute_force_protection: ExponentialBackoffLockout):
        """
        Initialize honeypot.

        Args:
            brute_force_protection: ExponentialBackoffLockout instance
        """
        self.protection = brute_force_protection

    async def honeypot_login(self, username: str, password: str) -> Dict:
        """
        Fake login endpoint that logs attempts.

        Args:
            username: Provided username
            password: Provided password

        Returns:
            Fake success response (misleading to attacker)
        """
        logger.warning(f"Honeypot activation: Attempted login as {username}")

        # Simulate legitimate processing delay
        await self._random_delay()

        # Return fake token to encourage further attacks
        return {
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "bearer",
            "expires_in": 3600
        }

    async def _random_delay(self) -> None:
        """Add random delay to waste attacker's time."""
        import asyncio
        import random
        delay = random.uniform(2, 5)
        await asyncio.sleep(delay)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*60)
    print("EXPONENTIAL BACKOFF BRUTE FORCE PROTECTION DEMO")
    print("="*60)

    protection = ExponentialBackoffLockout()

    print("\n1. Recording multiple failed attempts...")
    for i in range(6):
        is_locked, reason, lockout_until = protection.record_failure(
            "attacker@example.com",
            attempt_type="password",
            failure_reason="Invalid password",
            ip_address="192.168.1.100"
        )

        status = protection.get_lockout_status("attacker@example.com")
        print(f"\n   Attempt {i+1}:")
        print(f"   - Locked Out: {status['locked_out']}")
        print(f"   - Reason: {reason}")
        print(f"   - Level: {status['level']}")
        if status['lockout_until']:
            print(f"   - Until: {status['lockout_until']}")

    print("\n2. Checking lockout status...")
    is_locked, lockout_until = protection.is_locked_out("attacker@example.com")
    print(f"   Locked Out: {is_locked}")
    print(f"   Until: {lockout_until}")

    print("\n3. Honeypot activation check...")
    honeypot_active = protection.should_activate_honeypot("attacker@example.com")
    print(f"   Honeypot Activated: {honeypot_active}")

    print("\n4. Security Report...")
    report = protection.generate_security_report()
    print(f"   Total Tracked: {report['total_tracked_identifiers']}")
    print(f"   Locked Out: {report['currently_locked_out']}")
    print(f"   Total Failures: {report['total_failures_recorded']}")
    print(f"   High Risk: {report['high_risk_identifiers']}")

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
