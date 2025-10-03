"""
Security Middleware
© 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

Security middleware for the Sovereign Control System web interface:
- CSRF protection
- XSS prevention
- Content Security Policy
- Rate limiting
- HTTP headers hardening
"""

import time
import hashlib
import logging
from functools import wraps
from flask import request, abort, make_response, g, session
import re
import json
from threading import Lock
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sovereign.security_middleware')

# Rate limiting data stores
ip_request_counts: Dict[str, List[float]] = {}
ip_request_lock = Lock()

# Block list for suspicious IPs
blocked_ips: Dict[str, datetime] = {}
blocked_ips_lock = Lock()

# List of unsafe patterns for input validation
UNSAFE_PATTERNS = [
    # SQL Injection
    r"['\"%;)(\xor\s+or\s+and\s+select\s+union\s+from\s+where\s+group\s+by\s+having\s+order\s+by\s+limit]",
    # XSS
    r"<script.*?>|<.*?javascript:.*?>|<.*?onload=.*?>|<.*?onclick=.*?>",
    # Command Injection
    r";\s*(\w+\s*)*[|&`]",
    # Path Traversal
    r"\.\./|\.\.\x5c",
    # LDAP Injection
    r"\(\s*[|&!]\s*\w+\s*=\s*[\w*]",
    # XML/XXE Injection
    r"<!ENTITY.*?>|<!DOCTYPE.*?SYSTEM",
]


def load_security_middleware(app):
    """
    Load security middleware into the Flask application

    Args:
        app: The Flask application
    """
    # Apply security headers to all responses
    @app.after_request
    def add_security_headers(response):
        """Add security headers to responses"""
        # Content Security Policy
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "  # Unsafe-inline needed for dashboard JS
            "style-src 'self' 'unsafe-inline'; "   # Unsafe-inline needed for styling
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-src 'none'; "
            "object-src 'none'; "
            "base-uri 'self'"
        )

        # Other security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Server'] = 'Sovereign'  # Hide server details

        # Remove information leakage headers
        if 'Server' in response.headers:
            del response.headers['Server']
        if 'X-Powered-By' in response.headers:
            del response.headers['X-Powered-By']

        return response

    # CSRF Protection
    @app.before_request
    def csrf_protect():
        """Protect against CSRF attacks"""
        # Skip for GET, HEAD, OPTIONS, TRACE
        if request.method in ['GET', 'HEAD', 'OPTIONS', 'TRACE']:
            return

        # Check if request has valid CSRF token
        if request.endpoint and not request.endpoint.startswith('static'):
            token = session.get('_csrf_token')
            request_token = request.form.get('_csrf_token') or request.headers.get('X-CSRF-Token')

            if not token or token != request_token:
                logger.warning(f"CSRF validation failed for {request.endpoint}")
                abort(403, "CSRF validation failed")

    # Rate limiting
    @app.before_request
    def rate_limiting():
        """Apply rate limiting"""
        # Skip for static resources
        if request.endpoint and request.endpoint.startswith('static'):
            return

        ip = request.remote_addr

        # Check if IP is blocked
        with blocked_ips_lock:
            if ip in blocked_ips:
                if datetime.now() < blocked_ips[ip]:
                    logger.warning(f"Blocked request from banned IP: {ip}")
                    abort(403, "Your IP address has been temporarily blocked due to suspicious activity")
                else:
                    # Unblock IP if block period has expired
                    del blocked_ips[ip]

        # Apply rate limiting
        with ip_request_lock:
            # Initialize request counts for new IPs
            if ip not in ip_request_counts:
                ip_request_counts[ip] = []

            # Remove requests older than 1 minute
            current_time = time.time()
            ip_request_counts[ip] = [t for t in ip_request_counts[ip] if current_time - t < 60]

            # Add current request
            ip_request_counts[ip].append(current_time)

            # Check rate limit - 60 requests per minute
            if len(ip_request_counts[ip]) > 60:
                logger.warning(f"Rate limit exceeded for IP: {ip}")

                # Block the IP for 5 minutes
                with blocked_ips_lock:
                    blocked_ips[ip] = datetime.now() + timedelta(minutes=5)

                abort(429, "Rate limit exceeded. Please try again later.")

    # Input validation
    @app.before_request
    def validate_input():
        """Validate request input against unsafe patterns"""
        # Skip for static resources
        if request.endpoint and request.endpoint.startswith('static'):
            return

        # Validate query parameters
        for key, value in request.args.items():
            if value and any(re.search(pattern, value, re.IGNORECASE) for pattern in UNSAFE_PATTERNS):
                logger.warning(f"Potentially malicious input detected in query param: {key}={value}")
                abort(400, "Invalid input detected")

        # Validate form data
        for key, value in request.form.items():
            if value and any(re.search(pattern, value, re.IGNORECASE) for pattern in UNSAFE_PATTERNS):
                logger.warning(f"Potentially malicious input detected in form data: {key}={value}")
                abort(400, "Invalid input detected")

        # Validate JSON data
        if request.is_json:
            try:
                validate_json_recursive(request.json)
            except ValueError as e:
                logger.warning(f"Potentially malicious input detected in JSON: {str(e)}")
                abort(400, "Invalid input detected")

    # Generate CSRF token function
    def generate_csrf_token():
        """Generate a CSRF token"""
        if '_csrf_token' not in session:
            session['_csrf_token'] = hashlib.sha256(f"{time.time()}{session.sid}".encode()).hexdigest()
        return session['_csrf_token']

    # Add CSRF token to all templates
    app.jinja_env.globals['csrf_token'] = generate_csrf_token

    logger.info("Security middleware loaded")


def validate_json_recursive(data):
    """
    Recursively validate JSON data against unsafe patterns

    Args:
        data: JSON data to validate

    Raises:
        ValueError: If potentially malicious input is detected
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(key, str) and any(re.search(pattern, key, re.IGNORECASE) for pattern in UNSAFE_PATTERNS):
                raise ValueError(f"Potentially malicious input detected in JSON key: {key}")
            validate_json_recursive(value)
    elif isinstance(data, list):
        for item in data:
            validate_json_recursive(item)
    elif isinstance(data, str) and any(re.search(pattern, data, re.IGNORECASE) for pattern in UNSAFE_PATTERNS):
        raise ValueError(f"Potentially malicious input detected in JSON value: {data}")


def require_content_type(content_type: str):
    """
    Decorator to require a specific Content-Type header

    Args:
        content_type: Required Content-Type
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.headers.get('Content-Type', '').startswith(content_type):
                abort(415, f"Content-Type must be {content_type}")
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def sanitize_output(response_data):
    """
    Sanitize output data to prevent information disclosure

    Args:
        response_data: Data to sanitize

    Returns:
        Sanitized data
    """
    if isinstance(response_data, dict):
        sanitized = {}
        for key, value in response_data.items():
            # Skip sensitive fields
            if key.lower() in ['password', 'token', 'secret', 'key', 'credential', 'auth']:
                continue
            sanitized[key] = sanitize_output(value)
        return sanitized
    elif isinstance(response_data, list):
        return [sanitize_output(item) for item in response_data]
    else:
        return response_data


def set_secure_cookie(response, key, value, max_age=None, expires=None, secure=True, httponly=True, samesite='Lax'):
    """
    Set a secure cookie with appropriate flags

    Args:
        response: Response object
        key: Cookie key
        value: Cookie value
        max_age: Cookie max age in seconds
        expires: Cookie expiration date
        secure: Whether the cookie should only be sent over HTTPS
        httponly: Whether the cookie should be accessible only via HTTP
        samesite: SameSite attribute (Lax, Strict, None)

    Returns:
        Updated response
    """
    response.set_cookie(
        key,
        value,
        max_age=max_age,
        expires=expires,
        path='/',
        secure=secure,
        httponly=httponly,
        samesite=samesite
    )
    return response
