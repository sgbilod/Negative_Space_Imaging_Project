#!/usr/bin/env python
"""
Enhanced Security Provider for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import os
import json
import logging
import hashlib
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class SecurityProvider:
    """Enhanced security provider with real-time monitoring."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active = False
        self.shutdown_flag = threading.Event()

        # Configure logging
        self.logger = logging.getLogger("SecurityProvider")

        # Initialize security state
        self.security_state = {
            "encryption_active": False,
            "last_key_rotation": None,
            "auth_tokens": {},
            "active_sessions": {},
            "blocked_ips": set(),
            "security_events": []
        }

        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._security_monitor,
            daemon=True
        )

    def start(self):
        """Start the security provider."""
        try:
            self.logger.info("Starting security provider")

            # Initialize encryption
            self._initialize_encryption()

            # Start monitoring
            self.active = True
            self.monitor_thread.start()

            self.logger.info("Security provider started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start security provider: {e}")
            raise

    def stop(self):
        """Stop the security provider."""
        try:
            self.logger.info("Stopping security provider")

            # Set shutdown flag
            self.shutdown_flag.set()

            # Wait for monitor thread
            self.monitor_thread.join(timeout=10)

            # Cleanup
            self._cleanup()

            self.active = False
            self.logger.info("Security provider stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping security provider: {e}")
            raise

    def _initialize_encryption(self):
        """Initialize encryption system."""
        try:
            from cryptography.fernet import Fernet

            # Generate new key if needed
            if not self.security_state["encryption_active"]:
                key = Fernet.generate_key()
                self.cipher_suite = Fernet(key)
                self.security_state["encryption_active"] = True
                self.security_state["last_key_rotation"] = datetime.now()

            self.logger.info("Encryption system initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            raise

    def _security_monitor(self):
        """Security monitoring thread."""
        while not self.shutdown_flag.is_set():
            try:
                # Check key rotation
                self._check_key_rotation()

                # Clean expired sessions
                self._clean_expired_sessions()

                # Check for security events
                self._process_security_events()

            except Exception as e:
                self.logger.error(f"Security monitor error: {e}")

            finally:
                self.shutdown_flag.wait(60)  # Check every minute

    def _check_key_rotation(self):
        """Check if encryption key rotation is needed."""
        try:
            if not self.security_state["last_key_rotation"]:
                return

            rotation_days = self.config.get("key_rotation_days", 30)
            rotation_delta = timedelta(days=rotation_days)

            if (datetime.now() - self.security_state["last_key_rotation"]
                    > rotation_delta):
                self._rotate_encryption_key()

        except Exception as e:
            self.logger.error(f"Key rotation check failed: {e}")

    def _rotate_encryption_key(self):
        """Rotate encryption key."""
        try:
            from cryptography.fernet import Fernet

            # Generate new key
            new_key = Fernet.generate_key()
            new_cipher = Fernet(new_key)

            # Update state
            self.cipher_suite = new_cipher
            self.security_state["last_key_rotation"] = datetime.now()

            self.logger.info("Encryption key rotated successfully")

        except Exception as e:
            self.logger.error(f"Key rotation failed: {e}")

    def _clean_expired_sessions(self):
        """Clean expired sessions."""
        try:
            current_time = datetime.now()
            timeout = timedelta(
                minutes=self.config.get("session_timeout_minutes", 60)
            )

            expired = []
            for session_id, session in self.security_state["active_sessions"].items():
                if current_time - session["created"] > timeout:
                    expired.append(session_id)

            for session_id in expired:
                self.security_state["active_sessions"].pop(session_id)

            if expired:
                self.logger.info(f"Cleaned {len(expired)} expired sessions")

        except Exception as e:
            self.logger.error(f"Session cleanup failed: {e}")

    def _process_security_events(self):
        """Process security events."""
        try:
            current_time = datetime.now()

            # Clean old events
            self.security_state["security_events"] = [
                event for event in self.security_state["security_events"]
                if current_time - event["timestamp"] < timedelta(hours=24)
            ]

            # Process events
            for event in self.security_state["security_events"]:
                if event["processed"]:
                    continue

                self._handle_security_event(event)
                event["processed"] = True

        except Exception as e:
            self.logger.error(f"Event processing failed: {e}")

    def _handle_security_event(self, event: Dict[str, Any]):
        """Handle a security event."""
        try:
            if event["type"] == "auth_failure":
                self._handle_auth_failure(event)
            elif event["type"] == "invalid_token":
                self._handle_invalid_token(event)
            elif event["type"] == "suspicious_activity":
                self._handle_suspicious_activity(event)

        except Exception as e:
            self.logger.error(f"Event handling failed: {e}")

    def _handle_auth_failure(self, event: Dict[str, Any]):
        """Handle authentication failure event."""
        try:
            ip = event.get("ip")
            if not ip:
                return

            # Check for repeated failures
            failures = sum(
                1 for e in self.security_state["security_events"]
                if e["type"] == "auth_failure"
                and e["ip"] == ip
                and datetime.now() - e["timestamp"] < timedelta(hours=1)
            )

            if failures >= 5:
                self.security_state["blocked_ips"].add(ip)
                self.logger.warning(f"IP blocked due to auth failures: {ip}")

        except Exception as e:
            self.logger.error(f"Auth failure handling failed: {e}")

    def _handle_invalid_token(self, event: Dict[str, Any]):
        """Handle invalid token event."""
        try:
            token = event.get("token")
            if not token:
                return

            # Invalidate token
            if token in self.security_state["auth_tokens"]:
                del self.security_state["auth_tokens"][token]

        except Exception as e:
            self.logger.error(f"Invalid token handling failed: {e}")

    def _handle_suspicious_activity(self, event: Dict[str, Any]):
        """Handle suspicious activity event."""
        try:
            ip = event.get("ip")
            if not ip:
                return

            # Add to blocked IPs
            self.security_state["blocked_ips"].add(ip)
            self.logger.warning(f"IP blocked due to suspicious activity: {ip}")

        except Exception as e:
            self.logger.error(f"Suspicious activity handling failed: {e}")

    def _cleanup(self):
        """Cleanup security state."""
        try:
            # Clear sensitive data
            self.security_state["auth_tokens"].clear()
            self.security_state["active_sessions"].clear()
            self.security_state["security_events"].clear()

            # Reset encryption
            self.security_state["encryption_active"] = False
            self.security_state["last_key_rotation"] = None

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using current key."""
        try:
            if not self.security_state["encryption_active"]:
                raise RuntimeError("Encryption not initialized")

            return self.cipher_suite.encrypt(data)

        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data using current key."""
        try:
            if not self.security_state["encryption_active"]:
                raise RuntimeError("Encryption not initialized")

            return self.cipher_suite.decrypt(data)

        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise

    def create_session(self, user_id: str) -> str:
        """Create a new session."""
        try:
            session_id = hashlib.sha256(os.urandom(32)).hexdigest()

            self.security_state["active_sessions"][session_id] = {
                "user_id": user_id,
                "created": datetime.now()
            }

            return session_id

        except Exception as e:
            self.logger.error(f"Session creation failed: {e}")
            raise

    def validate_session(self, session_id: str) -> bool:
        """Validate a session."""
        try:
            if session_id not in self.security_state["active_sessions"]:
                return False

            session = self.security_state["active_sessions"][session_id]
            timeout = timedelta(
                minutes=self.config.get("session_timeout_minutes", 60)
            )

            return datetime.now() - session["created"] <= timeout

        except Exception as e:
            self.logger.error(f"Session validation failed: {e}")
            return False

    def end_session(self, session_id: str):
        """End a session."""
        try:
            if session_id in self.security_state["active_sessions"]:
                del self.security_state["active_sessions"][session_id]

        except Exception as e:
            self.logger.error(f"Session end failed: {e}")

    def is_ip_blocked(self, ip: str) -> bool:
        """Check if an IP is blocked."""
        return ip in self.security_state["blocked_ips"]

    def add_security_event(self, event_type: str, details: Dict[str, Any]):
        """Add a security event."""
        try:
            event = {
                "type": event_type,
                "details": details,
                "timestamp": datetime.now(),
                "processed": False
            }

            self.security_state["security_events"].append(event)

        except Exception as e:
            self.logger.error(f"Failed to add security event: {e}")
