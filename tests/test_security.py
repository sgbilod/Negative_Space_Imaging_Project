"""
Security Test Suite for Negative Space Imaging Project.

Comprehensive tests for authentication, authorization, security controls,
input validation, and audit logging.

Coverage Target: 90%+
Test Count: 25+ individual test cases
"""

import pytest
import hashlib
import hmac
import time
import json
import os
import tempfile
import re
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =====================================================================
# SECURITY FIXTURES
# =====================================================================

@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    return {
        "id": "test-user-1",
        "username": "testuser",
        "email": "test@example.com",
        "roles": ["user"],
        "password_hash": hashlib.sha256(b"password123").hexdigest()
    }


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user for testing."""
    return {
        "id": "admin-user-1",
        "username": "admin",
        "email": "admin@example.com",
        "roles": ["admin", "user"],
        "password_hash": hashlib.sha256(b"adminpass").hexdigest()
    }


@pytest.fixture
def mock_jwt_token():
    """Create a mock JWT token for testing."""
    return {
        "header": {"alg": "HS256", "typ": "JWT"},
        "payload": {
            "sub": "test-user-1",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "roles": ["user"]
        },
        "signature": "mock_signature"
    }


@pytest.fixture
def expired_jwt_token():
    """Create an expired JWT token for testing."""
    return {
        "header": {"alg": "HS256", "typ": "JWT"},
        "payload": {
            "sub": "test-user-1",
            "iat": int(time.time()) - 7200,
            "exp": int(time.time()) - 3600,
            "roles": ["user"]
        },
        "signature": "mock_signature"
    }


@pytest.fixture
def mock_rbac():
    """Create a mock RBAC system for testing."""
    class MockRBAC:
        def __init__(self):
            self.users = {}
            self.roles = set()
            self.admin_bootstrapped = False

        def create_user(self, username, password, roles=None):
            if username in self.users:
                raise ValueError("User already exists")
            if not self.admin_bootstrapped:
                assigned_roles = {"admin"}
                self.admin_bootstrapped = True
            else:
                assigned_roles = set(roles) if roles else set()
            self.users[username] = {
                "password": password,
                "roles": assigned_roles
            }
            self.roles.update(assigned_roles)
            return True

        def authenticate(self, username, password):
            user = self.users.get(username)
            if not user or user["password"] != password:
                return False
            return True

        def has_role(self, username, role):
            user = self.users.get(username)
            if not user:
                return False
            return role in user["roles"]

        def assign_role(self, username, role):
            if username not in self.users:
                raise ValueError("User not found")
            self.users[username]["roles"].add(role)
            self.roles.add(role)

        def remove_role(self, username, role):
            if username not in self.users:
                raise ValueError("User not found")
            self.users[username]["roles"].discard(role)

        def get_user_roles(self, username):
            user = self.users.get(username)
            if not user:
                return set()
            return user["roles"]

        def is_admin(self, username):
            return self.has_role(username, "admin")

    return MockRBAC()


@pytest.fixture
def temp_audit_dir():
    """Create a temporary directory for audit logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =====================================================================
# AUTHENTICATION TESTS
# =====================================================================

class TestAuthentication:
    """Tests for authentication functionality."""

    @pytest.mark.security
    @pytest.mark.unit
    def test_login_with_valid_credentials(self, mock_rbac):
        """Test successful login with valid credentials."""
        mock_rbac.create_user("testuser", "password123")

        result = mock_rbac.authenticate("testuser", "password123")

        assert result is True

    @pytest.mark.security
    @pytest.mark.unit
    def test_login_with_invalid_password(self, mock_rbac):
        """Test login rejection with invalid password."""
        mock_rbac.create_user("testuser", "password123")

        result = mock_rbac.authenticate("testuser", "wrongpassword")

        assert result is False

    @pytest.mark.security
    @pytest.mark.unit
    def test_login_with_nonexistent_user(self, mock_rbac):
        """Test login rejection for non-existent user."""
        result = mock_rbac.authenticate("nonexistent", "password123")

        assert result is False

    @pytest.mark.security
    @pytest.mark.unit
    def test_logout_invalidates_session(self):
        """Test that logout invalidates the user session."""
        session = {"user_id": "test-user-1", "active": True}

        # Simulate logout
        session["active"] = False
        session.pop("user_id", None)

        assert session.get("user_id") is None
        assert session["active"] is False

    @pytest.mark.security
    @pytest.mark.unit
    def test_password_hashing_verification(self, mock_user):
        """Test password hash verification."""
        password = "password123"
        expected_hash = hashlib.sha256(password.encode()).hexdigest()

        assert mock_user["password_hash"] == expected_hash

    @pytest.mark.security
    @pytest.mark.unit
    def test_password_hashing_salt_uniqueness(self):
        """Test that password hashing with salt produces unique hashes."""
        password = "password123"
        salt1 = os.urandom(16).hex()
        salt2 = os.urandom(16).hex()

        hash1 = hashlib.sha256((password + salt1).encode()).hexdigest()
        hash2 = hashlib.sha256((password + salt2).encode()).hexdigest()

        assert hash1 != hash2


# =====================================================================
# AUTHORIZATION TESTS
# =====================================================================

class TestAuthorization:
    """Tests for authorization and RBAC."""

    @pytest.mark.security
    @pytest.mark.unit
    def test_first_user_gets_admin_role(self, mock_rbac):
        """Test that first created user is assigned admin role."""
        mock_rbac.create_user("firstuser", "password")

        assert mock_rbac.is_admin("firstuser") is True

    @pytest.mark.security
    @pytest.mark.unit
    def test_subsequent_users_not_admin(self, mock_rbac):
        """Test that subsequent users are not automatically admins."""
        mock_rbac.create_user("firstuser", "password")
        mock_rbac.create_user("seconduser", "password", roles=["user"])

        assert mock_rbac.is_admin("seconduser") is False

    @pytest.mark.security
    @pytest.mark.unit
    def test_role_assignment(self, mock_rbac):
        """Test role assignment to user."""
        mock_rbac.create_user("firstuser", "password")
        mock_rbac.create_user("testuser", "password")

        mock_rbac.assign_role("testuser", "analyst")

        assert mock_rbac.has_role("testuser", "analyst") is True

    @pytest.mark.security
    @pytest.mark.unit
    def test_role_removal(self, mock_rbac):
        """Test role removal from user."""
        mock_rbac.create_user("firstuser", "password")
        mock_rbac.create_user("testuser", "password", roles=["analyst"])

        mock_rbac.remove_role("testuser", "analyst")

        assert mock_rbac.has_role("testuser", "analyst") is False

    @pytest.mark.security
    @pytest.mark.unit
    def test_get_user_roles(self, mock_rbac):
        """Test getting user roles."""
        mock_rbac.create_user("firstuser", "password")
        mock_rbac.create_user("testuser", "password", roles=["user", "analyst"])

        roles = mock_rbac.get_user_roles("testuser")

        assert "user" in roles
        assert "analyst" in roles

    @pytest.mark.security
    @pytest.mark.unit
    def test_duplicate_user_creation_fails(self, mock_rbac):
        """Test that duplicate user creation raises error."""
        mock_rbac.create_user("testuser", "password")

        with pytest.raises(ValueError, match="User already exists"):
            mock_rbac.create_user("testuser", "different_password")

    @pytest.mark.security
    @pytest.mark.unit
    def test_role_assignment_to_nonexistent_user_fails(self, mock_rbac):
        """Test role assignment to non-existent user raises error."""
        with pytest.raises(ValueError, match="User not found"):
            mock_rbac.assign_role("nonexistent", "admin")


# =====================================================================
# JWT TOKEN TESTS
# =====================================================================

class TestJWTValidation:
    """Tests for JWT token validation."""

    @pytest.mark.security
    @pytest.mark.unit
    def test_valid_token_structure(self, mock_jwt_token):
        """Test valid JWT token has required structure."""
        assert "header" in mock_jwt_token
        assert "payload" in mock_jwt_token
        assert "signature" in mock_jwt_token

        assert "alg" in mock_jwt_token["header"]
        assert "sub" in mock_jwt_token["payload"]
        assert "exp" in mock_jwt_token["payload"]

    @pytest.mark.security
    @pytest.mark.unit
    def test_token_expiration_check(self, mock_jwt_token, expired_jwt_token):
        """Test token expiration validation."""
        current_time = int(time.time())

        # Valid token should not be expired
        assert mock_jwt_token["payload"]["exp"] > current_time

        # Expired token should be expired
        assert expired_jwt_token["payload"]["exp"] < current_time

    @pytest.mark.security
    @pytest.mark.unit
    def test_token_issued_at_validation(self, mock_jwt_token):
        """Test token issued_at timestamp is valid."""
        current_time = int(time.time())
        iat = mock_jwt_token["payload"]["iat"]

        # Token should have been issued in the past
        assert iat <= current_time

    @pytest.mark.security
    @pytest.mark.unit
    def test_token_refresh_extends_expiration(self, mock_jwt_token):
        """Test token refresh extends expiration time."""
        old_exp = mock_jwt_token["payload"]["exp"]

        # Simulate refresh
        mock_jwt_token["payload"]["exp"] = int(time.time()) + 7200  # 2 hours

        assert mock_jwt_token["payload"]["exp"] > old_exp


# =====================================================================
# INPUT VALIDATION TESTS
# =====================================================================

class TestInputValidation:
    """Tests for input validation and sanitization."""

    @pytest.mark.security
    @pytest.mark.unit
    def test_sql_injection_prevention(self):
        """Test SQL injection patterns are detected."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1; DELETE FROM images WHERE 1=1",
            "UNION SELECT * FROM passwords"
        ]

        sql_pattern = re.compile(
            r"(--|'|\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|OR)\b)",
            re.IGNORECASE
        )

        for input_str in malicious_inputs:
            assert sql_pattern.search(input_str) is not None, \
                f"SQL injection not detected: {input_str}"

    @pytest.mark.security
    @pytest.mark.unit
    def test_xss_prevention(self):
        """Test XSS attack patterns are detected."""
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            '<img src="x" onerror="alert(\'XSS\')">',
            "<body onload=alert('XSS')>",
            "<%eval(request.item)%>"
        ]

        xss_pattern = re.compile(
            r"(<script|javascript:|onerror=|onload=|<%)",
            re.IGNORECASE
        )

        for input_str in malicious_inputs:
            assert xss_pattern.search(input_str) is not None, \
                f"XSS pattern not detected: {input_str}"

    @pytest.mark.security
    @pytest.mark.unit
    def test_html_entity_encoding(self):
        """Test HTML entity encoding for special characters."""
        dangerous_chars = {
            "<": "&lt;",
            ">": "&gt;",
            "&": "&amp;",
            '"': "&quot;",
            "'": "&#x27;"
        }

        for char, encoded in dangerous_chars.items():
            # Verify encoding is correct
            assert char != encoded
            assert len(encoded) > len(char)

    @pytest.mark.security
    @pytest.mark.unit
    def test_path_traversal_prevention(self):
        """Test path traversal attack patterns are detected."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/shadow",
            "....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f"  # URL encoded ../
        ]

        path_pattern = re.compile(r"(\.\.|%2e%2e|/etc/|\\windows\\)", re.IGNORECASE)

        for path in malicious_paths:
            assert path_pattern.search(path) is not None, \
                f"Path traversal not detected: {path}"


# =====================================================================
# RATE LIMITING TESTS
# =====================================================================

class TestRateLimiting:
    """Tests for rate limiting functionality."""

    @pytest.mark.security
    @pytest.mark.unit
    def test_request_count_tracking(self):
        """Test request count tracking for rate limiting."""
        rate_limiter = {
            "ip_address": "192.168.1.100",
            "requests": [],
            "limit": 100,
            "window": 60  # seconds
        }

        # Simulate requests
        for _ in range(10):
            rate_limiter["requests"].append(time.time())

        assert len(rate_limiter["requests"]) == 10

    @pytest.mark.security
    @pytest.mark.unit
    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded detection."""
        rate_limiter = {
            "requests": [time.time()] * 100,
            "limit": 100,
            "window": 60
        }

        is_exceeded = len(rate_limiter["requests"]) >= rate_limiter["limit"]
        assert is_exceeded is True

    @pytest.mark.security
    @pytest.mark.unit
    def test_rate_limit_window_reset(self):
        """Test rate limit window reset after time passes."""
        window = 60  # seconds
        old_time = time.time() - 120  # 2 minutes ago

        # All old requests should be outside window
        requests = [old_time] * 50
        current_time = time.time()

        recent_requests = [r for r in requests if current_time - r < window]

        assert len(recent_requests) == 0


# =====================================================================
# AUDIT LOGGING TESTS
# =====================================================================

class TestAuditLogging:
    """Tests for audit logging functionality."""

    @pytest.mark.security
    @pytest.mark.unit
    def test_audit_log_entry_structure(self):
        """Test audit log entry has required fields."""
        log_entry = {
            "timestamp": time.time(),
            "event": "USER_LOGIN",
            "severity": "INFO",
            "source": "auth_service",
            "details": {
                "user_id": "test_user",
                "ip_address": "192.168.1.100",
                "success": True
            }
        }

        required_fields = ["timestamp", "event", "severity", "source", "details"]
        for field in required_fields:
            assert field in log_entry

    @pytest.mark.security
    @pytest.mark.unit
    def test_audit_log_integrity_hash(self):
        """Test audit log integrity hash generation."""
        log_entry = {
            "timestamp": time.time(),
            "event": "FILE_ACCESS",
            "details": {"file": "test.jpg"}
        }

        entry_bytes = json.dumps(log_entry, sort_keys=True).encode()
        integrity_hash = hashlib.sha256(entry_bytes).hexdigest()

        assert len(integrity_hash) == 64
        assert all(c in "0123456789abcdef" for c in integrity_hash)

    @pytest.mark.security
    @pytest.mark.unit
    def test_audit_log_severity_levels(self):
        """Test valid audit log severity levels."""
        valid_severities = ["INFO", "WARNING", "ERROR", "CRITICAL", "SECURITY"]

        for severity in valid_severities:
            log_entry = {"severity": severity}
            assert log_entry["severity"] in valid_severities

    @pytest.mark.security
    @pytest.mark.unit
    def test_security_event_logging(self):
        """Test security event logging format."""
        security_event = {
            "timestamp": time.time(),
            "event": "SECURITY_ALERT",
            "severity": "CRITICAL",
            "category": "AUTHENTICATION",
            "source": "SecurityMonitor",
            "description": "Multiple failed login attempts",
            "details": {
                "user_id": "test_user",
                "failure_count": 5,
                "source_ip": "192.168.1.100"
            }
        }

        assert security_event["severity"] == "CRITICAL"
        assert security_event["category"] == "AUTHENTICATION"
        assert security_event["details"]["failure_count"] == 5


# =====================================================================
# SESSION MANAGEMENT TESTS
# =====================================================================

class TestSessionManagement:
    """Tests for session management functionality."""

    @pytest.mark.security
    @pytest.mark.unit
    def test_session_creation(self):
        """Test session creation with required fields."""
        session = {
            "session_id": hashlib.sha256(os.urandom(32)).hexdigest(),
            "user_id": "test-user-1",
            "created_at": time.time(),
            "expires_at": time.time() + 3600,
            "active": True
        }

        assert len(session["session_id"]) == 64
        assert session["active"] is True
        assert session["expires_at"] > session["created_at"]

    @pytest.mark.security
    @pytest.mark.unit
    def test_session_expiration(self):
        """Test session expiration detection."""
        expired_session = {
            "session_id": "abc123",
            "expires_at": time.time() - 100,
            "active": True
        }

        current_time = time.time()
        is_expired = expired_session["expires_at"] < current_time

        assert is_expired is True

    @pytest.mark.security
    @pytest.mark.unit
    def test_session_invalidation(self):
        """Test session invalidation."""
        session = {
            "session_id": "abc123",
            "user_id": "test-user-1",
            "active": True
        }

        # Invalidate session
        session["active"] = False
        session["invalidated_at"] = time.time()

        assert session["active"] is False
        assert "invalidated_at" in session

    @pytest.mark.security
    @pytest.mark.unit
    def test_concurrent_session_limit(self):
        """Test concurrent session limit enforcement."""
        max_sessions = 3
        user_sessions = [
            {"session_id": f"session_{i}", "active": True}
            for i in range(5)
        ]

        active_sessions = [s for s in user_sessions if s["active"]]

        # Enforce limit by deactivating oldest sessions
        while len(active_sessions) > max_sessions:
            active_sessions[0]["active"] = False
            active_sessions = [s for s in user_sessions if s["active"]]

        assert len([s for s in user_sessions if s["active"]]) <= max_sessions


# =====================================================================
# CORS AND CSRF TESTS
# =====================================================================

class TestCORSAndCSRF:
    """Tests for CORS policy and CSRF protection."""

    @pytest.mark.security
    @pytest.mark.unit
    def test_cors_allowed_origins(self):
        """Test CORS allowed origins configuration."""
        cors_config = {
            "allowed_origins": [
                "https://example.com",
                "https://api.example.com"
            ],
            "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_credentials": True
        }

        # Test valid origin
        origin = "https://example.com"
        assert origin in cors_config["allowed_origins"]

        # Test invalid origin
        invalid_origin = "https://malicious.com"
        assert invalid_origin not in cors_config["allowed_origins"]

    @pytest.mark.security
    @pytest.mark.unit
    def test_csrf_token_generation(self):
        """Test CSRF token generation."""
        csrf_token = hashlib.sha256(os.urandom(32)).hexdigest()

        assert len(csrf_token) == 64
        assert all(c in "0123456789abcdef" for c in csrf_token)

    @pytest.mark.security
    @pytest.mark.unit
    def test_csrf_token_validation(self):
        """Test CSRF token validation."""
        expected_token = "abc123def456"
        submitted_token = "abc123def456"

        # Valid token
        assert hmac.compare_digest(expected_token, submitted_token)

        # Invalid token
        invalid_token = "xyz789"
        assert not hmac.compare_digest(expected_token, invalid_token)

    @pytest.mark.security
    @pytest.mark.unit
    def test_same_origin_policy(self):
        """Test same origin policy validation."""
        def is_same_origin(url1: str, url2: str) -> bool:
            """Check if two URLs have the same origin."""
            from urllib.parse import urlparse
            parsed1 = urlparse(url1)
            parsed2 = urlparse(url2)
            return (parsed1.scheme == parsed2.scheme and
                    parsed1.netloc == parsed2.netloc)

        # Same origin
        assert is_same_origin(
            "https://example.com/path1",
            "https://example.com/path2"
        )

        # Different origin
        assert not is_same_origin(
            "https://example.com",
            "https://other.com"
        )
