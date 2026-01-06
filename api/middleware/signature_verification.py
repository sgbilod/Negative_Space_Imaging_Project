"""
Request Signature Verification Middleware
=========================================

Implements HMAC-SHA256 request signing and verification to ensure:
- Request authenticity (prevent spoofing)
- Request integrity (detect tampering)
- Replay attack prevention via timestamp validation

Philosophy: Cryptographic integrity is the foundation of secure communication.

Author: @CIPHER - Advanced Cryptography & Security
Date: December 2025
"""

import hmac
import hashlib
import json
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from base64 import b64encode, b64decode
import os

logger = logging.getLogger("signature_verification")


class SignatureConfig:
    """Configuration for HMAC signature verification."""

    # HMAC Algorithm
    ALGORITHM = "SHA256"
    HASH_FUNCTION = hashlib.sha256

    # Timestamp validation window (5 minutes)
    TIMESTAMP_WINDOW_SECONDS = 300

    # Signature header names
    SIGNATURE_HEADER = "X-Signature"
    TIMESTAMP_HEADER = "X-Timestamp"

    # Default signing key (should be environment-based in production)
    DEFAULT_SIGNING_KEY = os.getenv("SIGNING_KEY", "default-insecure-key-change-me")


class RequestSigner:
    """Signs HTTP requests using HMAC-SHA256."""

    def __init__(self, signing_key: Optional[str] = None):
        """
        Initialize request signer.

        Args:
            signing_key: Secret key for HMAC. Uses SignatureConfig.DEFAULT_SIGNING_KEY if None.
        """
        self.signing_key = (signing_key or SignatureConfig.DEFAULT_SIGNING_KEY).encode()
        if len(self.signing_key) < 32:
            logger.warning("Signing key is less than 256 bits. Use stronger keys in production.")

    def sign_request(
        self,
        method: str,
        path: str,
        body: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Tuple[str, str]:
        """
        Sign an HTTP request using HMAC-SHA256.

        Creates a signature over:
        - HTTP method (GET, POST, etc.)
        - Request path
        - Request body (if present)
        - Timestamp (prevents replay attacks)

        Formula:
            signature = HMAC-SHA256(
                signing_key,
                "{method}\n{path}\n{body}\n{timestamp}"
            )

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            body: Request body (optional)
            timestamp: Request timestamp. Uses current time if None.

        Returns:
            Tuple of (signature_hex, timestamp_iso)

        Raises:
            ValueError: If method or path are invalid
        """
        if not method or not isinstance(method, str):
            raise ValueError("method must be a non-empty string")
        if not path or not isinstance(path, str):
            raise ValueError("path must be a non-empty string")

        # Use provided timestamp or current time
        if timestamp is None:
            timestamp = datetime.utcnow()

        timestamp_iso = timestamp.isoformat() + "Z"

        # Create message to sign
        message_parts = [
            method.upper(),
            path,
            body or "",
            timestamp_iso
        ]
        message = "\n".join(message_parts).encode()

        # Compute HMAC-SHA256
        signature = hmac.new(
            self.signing_key,
            message,
            SignatureConfig.HASH_FUNCTION
        ).hexdigest()

        logger.debug(f"Signed request: {method} {path} -> {signature[:16]}...")

        return signature, timestamp_iso

    def get_signature_headers(
        self,
        method: str,
        path: str,
        body: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, str]:
        """
        Get HTTP headers for signed request.

        Returns:
            Dictionary with signature headers ready to add to request
        """
        signature, timestamp_iso = self.sign_request(method, path, body, timestamp)

        return {
            SignatureConfig.SIGNATURE_HEADER: signature,
            SignatureConfig.TIMESTAMP_HEADER: timestamp_iso
        }


class SignatureVerifier:
    """Verifies HMAC-SHA256 signatures on incoming requests."""

    def __init__(self, signing_key: Optional[str] = None):
        """
        Initialize signature verifier.

        Args:
            signing_key: Secret key for HMAC verification. Uses config default if None.
        """
        self.signing_key = (signing_key or SignatureConfig.DEFAULT_SIGNING_KEY).encode()

    def verify_signature(
        self,
        method: str,
        path: str,
        body: Optional[str] = None,
        provided_signature: Optional[str] = None,
        provided_timestamp: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Verify request signature and timestamp.

        Performs two checks:
        1. Signature verification: HMAC-SHA256 matches
        2. Timestamp validation: Request is recent (within 5 minutes)

        Args:
            method: HTTP method
            path: Request path
            body: Request body
            provided_signature: Signature from request headers
            provided_timestamp: Timestamp from request headers

        Returns:
            Tuple of (is_valid, error_message)
            - (True, "") if signature is valid
            - (False, error_reason) if signature is invalid
        """
        # Validate inputs
        if not provided_signature:
            return False, "Missing signature header"

        if not provided_timestamp:
            return False, "Missing timestamp header"

        # Validate timestamp freshness
        is_fresh, timestamp_error = self._validate_timestamp(provided_timestamp)
        if not is_fresh:
            return False, timestamp_error

        # Compute expected signature
        try:
            signer = RequestSigner(self.signing_key.decode())
            expected_signature, _ = signer.sign_request(
                method, path, body,
                datetime.fromisoformat(provided_timestamp.rstrip("Z"))
            )
        except Exception as e:
            logger.error(f"Error computing expected signature: {e}")
            return False, "Signature computation failed"

        # Compare signatures using constant-time comparison
        # Prevents timing attacks that could leak information about valid signatures
        is_valid = hmac.compare_digest(provided_signature, expected_signature)

        if not is_valid:
            logger.warning(f"Signature mismatch for {method} {path}")
            return False, "Signature verification failed"

        return True, ""

    def _validate_timestamp(self, timestamp_str: str) -> Tuple[bool, str]:
        """
        Validate request timestamp freshness.

        Prevents replay attacks by ensuring requests are recent.

        Args:
            timestamp_str: ISO format timestamp string

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Parse timestamp (handle both with/without 'Z' suffix)
            timestamp_str = timestamp_str.rstrip("Z")
            request_time = datetime.fromisoformat(timestamp_str)
        except Exception as e:
            logger.warning(f"Invalid timestamp format: {timestamp_str}")
            return False, "Invalid timestamp format"

        # Check if timestamp is within acceptable window
        current_time = datetime.utcnow()
        time_diff = abs((current_time - request_time).total_seconds())

        if time_diff > SignatureConfig.TIMESTAMP_WINDOW_SECONDS:
            logger.warning(
                f"Timestamp outside acceptable window: {time_diff}s > "
                f"{SignatureConfig.TIMESTAMP_WINDOW_SECONDS}s"
            )
            return False, f"Timestamp too old or in future (diff: {time_diff}s)"

        return True, ""


# Global instances
_signer: Optional[RequestSigner] = None
_verifier: Optional[SignatureVerifier] = None


def get_signer(signing_key: Optional[str] = None) -> RequestSigner:
    """Get or create global request signer instance."""
    global _signer
    if _signer is None:
        _signer = RequestSigner(signing_key)
    return _signer


def get_verifier(signing_key: Optional[str] = None) -> SignatureVerifier:
    """Get or create global signature verifier instance."""
    global _verifier
    if _verifier is None:
        _verifier = SignatureVerifier(signing_key)
    return _verifier


# Middleware function for FastAPI/Starlette
async def signature_verification_middleware(request, call_next):
    """
    FastAPI middleware for signature verification.

    Usage in FastAPI app:
        from api.middleware.signature_verification import signature_verification_middleware
        app.middleware("http")(signature_verification_middleware)

    Example client request with signatures:
        signer = RequestSigner(signing_key="your-secret-key")
        headers = signer.get_signature_headers("POST", "/api/process", json.dumps(data))
        response = requests.post(url, headers=headers, json=data)
    """
    # Extract signature and timestamp from headers
    signature = request.headers.get(SignatureConfig.SIGNATURE_HEADER)
    timestamp = request.headers.get(SignatureConfig.TIMESTAMP_HEADER)

    # Skip verification for public endpoints
    public_endpoints = ["/health", "/status", "/docs", "/openapi.json"]
    if request.url.path not in public_endpoints:
        # Read body (can only be read once)
        body = await request.body()
        body_str = body.decode() if body else None

        # Verify signature
        verifier = get_verifier()
        is_valid, error_msg = verifier.verify_signature(
            request.method,
            request.url.path,
            body_str,
            signature,
            timestamp
        )

        if not is_valid:
            logger.warning(f"Signature verification failed: {error_msg}")
            from fastapi.responses import JSONResponse
            return JSONResponse(
                {"error": "Signature verification failed", "detail": error_msg},
                status_code=401
            )

    response = await call_next(request)
    return response


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)

    # Create signer and verifier with same key
    signing_key = "super-secret-key-at-least-32-chars-long"
    signer = RequestSigner(signing_key)
    verifier = SignatureVerifier(signing_key)

    # Sign a request
    print("\n=== Signing Request ===")
    signature, timestamp = signer.sign_request(
        "POST",
        "/api/process",
        '{"data": "test"}'
    )
    print(f"Signature: {signature}")
    print(f"Timestamp: {timestamp}")

    # Verify it
    print("\n=== Verifying Signature ===")
    is_valid, error = verifier.verify_signature(
        "POST",
        "/api/process",
        '{"data": "test"}',
        signature,
        timestamp
    )
    print(f"Valid: {is_valid}")
    if error:
        print(f"Error: {error}")

    # Test invalid signature
    print("\n=== Testing Invalid Signature ===")
    is_valid, error = verifier.verify_signature(
        "POST",
        "/api/process",
        '{"data": "tampered"}',  # Modified body
        signature,
        timestamp
    )
    print(f"Valid: {is_valid}")
    print(f"Error: {error}")
