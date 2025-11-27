# =============================================================================
# Negative Space Imaging Project - JWT Authentication Manager
# Enterprise-grade JWT authentication with refresh token rotation
# =============================================================================
#
# This module implements a complete JWT authentication system with:
# - RS256 asymmetric signing for enhanced security
# - Short-lived access tokens (default 15 minutes)
# - Refresh token rotation with reuse detection
# - Token revocation tracking
# - Rate limiting for brute force protection
# - HIPAA-adjacent security practices
#
# Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
# =============================================================================

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# Enums
# =============================================================================

class TokenType(str, Enum):
    """Token type enumeration."""
    ACCESS = "access"
    REFRESH = "refresh"


# =============================================================================
# Exception Classes
# =============================================================================

class AuthError(Exception):
    """Base authentication error."""

    def __init__(self, message: str = "Authentication error", code: str = "AUTH_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class TokenExpiredError(AuthError):
    """Raised when a token has expired."""

    def __init__(self, message: str = "Token has expired"):
        super().__init__(message, "TOKEN_EXPIRED")


class TokenInvalidError(AuthError):
    """Raised when a token is invalid."""

    def __init__(self, message: str = "Token is invalid"):
        super().__init__(message, "TOKEN_INVALID")


class TokenRevokedError(AuthError):
    """Raised when a token has been revoked."""

    def __init__(self, message: str = "Token has been revoked"):
        super().__init__(message, "TOKEN_REVOKED")


class RateLimitExceededError(AuthError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        super().__init__(message, "RATE_LIMIT_EXCEEDED")
        self.retry_after = retry_after


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TokenPair:
    """
    Represents a pair of access and refresh tokens.

    Attributes:
        access_token: JWT access token for API authorization
        refresh_token: JWT refresh token for obtaining new access tokens
        access_expires_at: Expiration datetime for access token
        refresh_expires_at: Expiration datetime for refresh token
        token_type: Token type (Bearer)
    """
    access_token: str
    refresh_token: str
    access_expires_at: datetime
    refresh_expires_at: datetime
    token_type: str = "Bearer"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for API responses.

        Returns:
            Dictionary with token info and expires_in seconds
        """
        now = datetime.now(timezone.utc)
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": int((self.access_expires_at - now).total_seconds()),
            "refresh_expires_in": int((self.refresh_expires_at - now).total_seconds()),
        }


@dataclass
class TokenClaims:
    """
    Represents the claims contained in a JWT token.

    Attributes:
        sub: Subject (user ID)
        iat: Issued at timestamp
        exp: Expiration timestamp
        jti: JWT ID (unique identifier)
        iss: Issuer
        aud: Audience
        token_type: Type of token (access/refresh)
        roles: User roles
        permissions: User permissions
        device_id: Device identifier
        session_id: Session identifier
    """
    sub: str  # user_id
    iat: int
    exp: int
    jti: str
    iss: str
    aud: str
    token_type: TokenType
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    device_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for JWT payload.

        Returns:
            Dictionary suitable for JWT encoding
        """
        return {
            "sub": self.sub,
            "iat": self.iat,
            "exp": self.exp,
            "jti": self.jti,
            "iss": self.iss,
            "aud": self.aud,
            "token_type": self.token_type.value,
            "roles": self.roles,
            "permissions": self.permissions,
            "device_id": self.device_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TokenClaims:
        """
        Create TokenClaims from dictionary.

        Args:
            data: Dictionary containing token claims

        Returns:
            TokenClaims instance
        """
        return cls(
            sub=data["sub"],
            iat=data["iat"],
            exp=data["exp"],
            jti=data["jti"],
            iss=data.get("iss", ""),
            aud=data.get("aud", ""),
            token_type=TokenType(data.get("token_type", TokenType.ACCESS.value)),
            roles=data.get("roles", []),
            permissions=data.get("permissions", []),
            device_id=data.get("device_id"),
            session_id=data.get("session_id"),
        )


# =============================================================================
# Token Revocation Store
# =============================================================================

class TokenRevocationStore:
    """
    In-memory store for revoked tokens with automatic cleanup.

    Uses async lock for thread-safe operations. Tokens are stored
    with their expiration time and automatically cleaned up when
    they would have expired anyway.

    Attributes:
        max_entries: Maximum number of entries to store
    """

    def __init__(self, max_entries: int = 100000):
        """
        Initialize the revocation store.

        Args:
            max_entries: Maximum number of revoked tokens to track
        """
        self._store: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        self.max_entries = max_entries
        self._last_cleanup = datetime.now(timezone.utc)
        self._cleanup_interval = timedelta(minutes=5)

    async def revoke(self, jti: str, expires_at: datetime) -> None:
        """
        Revoke a token by its JTI.

        Args:
            jti: JWT ID to revoke
            expires_at: Original expiration time of the token
        """
        async with self._lock:
            # Cleanup if needed
            if datetime.now(timezone.utc) - self._last_cleanup > self._cleanup_interval:
                await self._cleanup_unlocked()

            # Enforce max entries
            if len(self._store) >= self.max_entries:
                await self._cleanup_unlocked()
                if len(self._store) >= self.max_entries:
                    # Remove oldest entries
                    sorted_entries = sorted(self._store.items(), key=lambda x: x[1])
                    entries_to_remove = len(self._store) - self.max_entries + 1000
                    for jti_to_remove, _ in sorted_entries[:entries_to_remove]:
                        del self._store[jti_to_remove]

            self._store[jti] = expires_at
            logger.debug(f"Token revoked: {jti[:8]}...")

    async def is_revoked(self, jti: str) -> bool:
        """
        Check if a token is revoked.

        Args:
            jti: JWT ID to check

        Returns:
            True if token is revoked, False otherwise
        """
        async with self._lock:
            if jti in self._store:
                # Check if still valid (not yet expired)
                if self._store[jti] > datetime.now(timezone.utc):
                    return True
                else:
                    # Token has expired, remove from store
                    del self._store[jti]
            return False

    async def _cleanup(self) -> None:
        """Public cleanup method with lock."""
        async with self._lock:
            await self._cleanup_unlocked()

    async def _cleanup_unlocked(self) -> None:
        """
        Remove expired entries (call within lock context).
        """
        now = datetime.now(timezone.utc)
        expired_jtis = [
            jti for jti, expires_at in self._store.items()
            if expires_at <= now
        ]
        for jti in expired_jtis:
            del self._store[jti]

        self._last_cleanup = now
        logger.debug(f"Cleaned up {len(expired_jtis)} expired revoked tokens")

    @property
    def size(self) -> int:
        """Return the number of stored revocations."""
        return len(self._store)


# =============================================================================
# Refresh Token Store
# =============================================================================

@dataclass
class RefreshTokenEntry:
    """Entry for tracking refresh tokens."""
    jti: str
    family_id: str
    user_id: str
    expires_at: datetime
    used: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RefreshTokenStore:
    """
    Store for tracking refresh token families and detecting reuse.

    Implements refresh token rotation with automatic family revocation
    on reuse detection (to prevent token theft attacks).
    """

    def __init__(self, max_entries: int = 50000):
        """
        Initialize the refresh token store.

        Args:
            max_entries: Maximum number of tokens to track
        """
        self._tokens: Dict[str, RefreshTokenEntry] = {}
        self._families: Dict[str, Set[str]] = {}  # family_id -> set of jtis
        self._lock = asyncio.Lock()
        self.max_entries = max_entries

    async def store(
        self,
        jti: str,
        family_id: str,
        user_id: str,
        expires_at: datetime
    ) -> None:
        """
        Store a refresh token.

        Args:
            jti: JWT ID
            family_id: Token family identifier
            user_id: User ID
            expires_at: Expiration datetime
        """
        async with self._lock:
            # Cleanup if needed
            if len(self._tokens) >= self.max_entries:
                await self._cleanup_unlocked()

            entry = RefreshTokenEntry(
                jti=jti,
                family_id=family_id,
                user_id=user_id,
                expires_at=expires_at
            )
            self._tokens[jti] = entry

            # Track in family
            if family_id not in self._families:
                self._families[family_id] = set()
            self._families[family_id].add(jti)

            logger.debug(f"Stored refresh token {jti[:8]}... in family {family_id[:8]}...")

    async def validate_and_mark_used(
        self,
        jti: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate a refresh token and mark it as used.

        Detects token reuse attacks and revokes the entire family if detected.

        Args:
            jti: JWT ID to validate

        Returns:
            Tuple of (is_valid, user_id, family_id)
        """
        async with self._lock:
            if jti not in self._tokens:
                logger.warning(f"Refresh token not found: {jti[:8]}...")
                return (False, None, None)

            entry = self._tokens[jti]

            # Check expiration
            if entry.expires_at <= datetime.now(timezone.utc):
                logger.debug(f"Refresh token expired: {jti[:8]}...")
                del self._tokens[jti]
                if entry.family_id in self._families:
                    self._families[entry.family_id].discard(jti)
                return (False, None, None)

            # Detect reuse attack
            if entry.used:
                logger.warning(
                    f"Refresh token reuse detected! "
                    f"Token: {jti[:8]}..., Family: {entry.family_id[:8]}..."
                )
                # Revoke entire family
                await self._revoke_family_unlocked(entry.family_id)
                return (False, None, None)

            # Mark as used
            entry.used = True

            return (True, entry.user_id, entry.family_id)

    async def _revoke_family(self, family_id: str) -> None:
        """Revoke all tokens in a family (with lock)."""
        async with self._lock:
            await self._revoke_family_unlocked(family_id)

    async def _revoke_family_unlocked(self, family_id: str) -> None:
        """
        Revoke all tokens in a family (without lock - call within lock context).

        Args:
            family_id: Token family to revoke
        """
        if family_id not in self._families:
            return

        jtis = self._families[family_id]
        for jti in jtis:
            if jti in self._tokens:
                del self._tokens[jti]

        del self._families[family_id]
        logger.info(f"Revoked token family {family_id[:8]}... ({len(jtis)} tokens)")

    async def revoke_user_tokens(self, user_id: str) -> int:
        """
        Revoke all tokens for a user.

        Args:
            user_id: User ID

        Returns:
            Number of tokens revoked
        """
        async with self._lock:
            families_to_revoke = set()
            for jti, entry in self._tokens.items():
                if entry.user_id == user_id:
                    families_to_revoke.add(entry.family_id)

            count = 0
            for family_id in families_to_revoke:
                if family_id in self._families:
                    count += len(self._families[family_id])
                    await self._revoke_family_unlocked(family_id)

            return count

    async def _cleanup_unlocked(self) -> None:
        """Remove expired entries (call within lock context)."""
        now = datetime.now(timezone.utc)
        expired_jtis = [
            jti for jti, entry in self._tokens.items()
            if entry.expires_at <= now
        ]

        for jti in expired_jtis:
            entry = self._tokens[jti]
            if entry.family_id in self._families:
                self._families[entry.family_id].discard(jti)
                if not self._families[entry.family_id]:
                    del self._families[entry.family_id]
            del self._tokens[jti]

        logger.debug(f"Cleaned up {len(expired_jtis)} expired refresh tokens")

    @property
    def size(self) -> int:
        """Return the number of stored tokens."""
        return len(self._tokens)


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for brute force protection.

    Implements the token bucket algorithm with configurable
    requests per minute and burst capacity.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10
    ):
        """
        Initialize the rate limiter.

        Args:
            requests_per_minute: Allowed requests per minute
            burst_size: Maximum burst capacity
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens_per_second = requests_per_minute / 60.0

        self._buckets: Dict[str, Dict[str, float]] = {}
        self._lock = asyncio.Lock()

    async def check_rate_limit(self, identifier: str) -> None:
        """
        Check if the request is within rate limits.

        Args:
            identifier: Client identifier (IP, user ID, etc.)

        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        async with self._lock:
            now = time.time()

            if identifier not in self._buckets:
                self._buckets[identifier] = {
                    "tokens": float(self.burst_size),
                    "last_update": now
                }

            bucket = self._buckets[identifier]

            # Refill tokens based on elapsed time
            elapsed = now - bucket["last_update"]
            bucket["tokens"] = min(
                self.burst_size,
                bucket["tokens"] + elapsed * self.tokens_per_second
            )
            bucket["last_update"] = now

            # Check if we have tokens
            if bucket["tokens"] < 1.0:
                # Calculate retry after
                tokens_needed = 1.0 - bucket["tokens"]
                retry_after = int(tokens_needed / self.tokens_per_second) + 1
                raise RateLimitExceededError(
                    message=f"Rate limit exceeded. Retry after {retry_after}s",
                    retry_after=retry_after
                )

            # Consume a token
            bucket["tokens"] -= 1.0

    async def reset(self, identifier: str) -> None:
        """
        Reset the rate limit for an identifier.

        Args:
            identifier: Client identifier
        """
        async with self._lock:
            if identifier in self._buckets:
                del self._buckets[identifier]

    async def cleanup_old_buckets(self, max_age_seconds: int = 3600) -> None:
        """
        Remove stale rate limit buckets.

        Args:
            max_age_seconds: Maximum age of buckets to keep
        """
        async with self._lock:
            now = time.time()
            stale_identifiers = [
                identifier
                for identifier, bucket in self._buckets.items()
                if now - bucket["last_update"] > max_age_seconds
            ]
            for identifier in stale_identifiers:
                del self._buckets[identifier]


# =============================================================================
# JWT Authentication Manager
# =============================================================================

class JWTAuthenticationManager:
    """
    Complete JWT authentication manager with RS256 signing.

    Features:
    - Asymmetric RS256 signing for enhanced security
    - Short-lived access tokens with refresh rotation
    - Token revocation and family tracking
    - Rate limiting for brute force protection
    - Audit logging for security compliance
    """

    def __init__(
        self,
        private_key: Optional[rsa.RSAPrivateKey] = None,
        public_key: Optional[rsa.RSAPublicKey] = None,
        private_key_path: Optional[Path] = None,
        public_key_path: Optional[Path] = None,
        access_token_lifetime_minutes: int = 15,
        refresh_token_lifetime_days: int = 7,
        issuer: str = "nsi-edge",
        audience: str = "nsi-api",
        rate_limiter: Optional[RateLimiter] = None,
        revocation_store: Optional[TokenRevocationStore] = None,
        refresh_store: Optional[RefreshTokenStore] = None,
    ):
        """
        Initialize the JWT authentication manager.

        Args:
            private_key: RSA private key for signing
            public_key: RSA public key for verification
            private_key_path: Path to PEM-encoded private key file
            public_key_path: Path to PEM-encoded public key file
            access_token_lifetime_minutes: Access token lifetime in minutes
            refresh_token_lifetime_days: Refresh token lifetime in days
            issuer: Token issuer claim
            audience: Token audience claim
            rate_limiter: Rate limiter instance
            revocation_store: Token revocation store
            refresh_store: Refresh token store
        """
        # Load or generate keys
        if private_key and public_key:
            self._private_key = private_key
            self._public_key = public_key
        elif private_key_path and public_key_path:
            self._private_key = self._load_private_key(private_key_path)
            self._public_key = self._load_public_key(public_key_path)
        else:
            # Generate new key pair
            logger.info("Generating new RSA key pair for JWT signing")
            self._private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self._public_key = self._private_key.public_key()

        self.access_token_lifetime = timedelta(minutes=access_token_lifetime_minutes)
        self.refresh_token_lifetime = timedelta(days=refresh_token_lifetime_days)
        self.issuer = issuer
        self.audience = audience

        # Initialize stores
        self.rate_limiter = rate_limiter or RateLimiter()
        self.revocation_store = revocation_store or TokenRevocationStore()
        self.refresh_store = refresh_store or RefreshTokenStore()

        logger.info(
            f"JWT Authentication Manager initialized "
            f"(access: {access_token_lifetime_minutes}min, "
            f"refresh: {refresh_token_lifetime_days}days)"
        )

    def _load_private_key(self, path: Path) -> rsa.RSAPrivateKey:
        """Load RSA private key from PEM file."""
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )

    def _load_public_key(self, path: Path) -> rsa.RSAPublicKey:
        """Load RSA public key from PEM file."""
        with open(path, "rb") as f:
            return serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )

    def _generate_jti(self) -> str:
        """Generate a unique JWT ID."""
        return secrets.token_urlsafe(32)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return secrets.token_urlsafe(16)

    def _generate_family_id(self) -> str:
        """Generate a unique token family ID."""
        return secrets.token_urlsafe(24)

    def _create_token(
        self,
        user_id: str,
        token_type: TokenType,
        roles: List[str],
        permissions: List[str],
        device_id: Optional[str] = None,
        session_id: Optional[str] = None,
        family_id: Optional[str] = None,
        lifetime: Optional[timedelta] = None,
    ) -> Tuple[str, TokenClaims]:
        """
        Create a JWT token with the specified claims.

        Args:
            user_id: User identifier
            token_type: Type of token
            roles: User roles
            permissions: User permissions
            device_id: Device identifier
            session_id: Session identifier
            family_id: Token family ID (for refresh tokens)
            lifetime: Token lifetime override

        Returns:
            Tuple of (encoded token, claims)
        """
        now = datetime.now(timezone.utc)

        if lifetime is None:
            lifetime = (
                self.access_token_lifetime
                if token_type == TokenType.ACCESS
                else self.refresh_token_lifetime
            )

        exp = now + lifetime
        jti = self._generate_jti()

        claims = TokenClaims(
            sub=user_id,
            iat=int(now.timestamp()),
            exp=int(exp.timestamp()),
            jti=jti,
            iss=self.issuer,
            aud=self.audience,
            token_type=token_type,
            roles=roles,
            permissions=permissions,
            device_id=device_id,
            session_id=session_id,
        )

        payload = claims.to_dict()

        # Add family_id for refresh tokens
        if token_type == TokenType.REFRESH and family_id:
            payload["family_id"] = family_id

        # Encode with RS256
        private_key_pem = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        token = jwt.encode(
            payload,
            private_key_pem,
            algorithm="RS256"
        )

        return token, claims

    async def authenticate(
        self,
        user_id: str,
        roles: List[str],
        permissions: List[str],
        device_id: Optional[str] = None,
    ) -> TokenPair:
        """
        Authenticate a user and generate token pair.

        Args:
            user_id: User identifier
            roles: User roles
            permissions: User permissions
            device_id: Device identifier

        Returns:
            TokenPair with access and refresh tokens

        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        # Check rate limit
        await self.rate_limiter.check_rate_limit(f"auth:{user_id}")

        session_id = self._generate_session_id()
        family_id = self._generate_family_id()

        # Create access token
        access_token, access_claims = self._create_token(
            user_id=user_id,
            token_type=TokenType.ACCESS,
            roles=roles,
            permissions=permissions,
            device_id=device_id,
            session_id=session_id,
        )

        # Create refresh token
        refresh_token, refresh_claims = self._create_token(
            user_id=user_id,
            token_type=TokenType.REFRESH,
            roles=roles,
            permissions=permissions,
            device_id=device_id,
            session_id=session_id,
            family_id=family_id,
        )

        # Store refresh token
        await self.refresh_store.store(
            jti=refresh_claims.jti,
            family_id=family_id,
            user_id=user_id,
            expires_at=datetime.fromtimestamp(refresh_claims.exp, tz=timezone.utc),
        )

        access_expires = datetime.fromtimestamp(access_claims.exp, tz=timezone.utc)
        refresh_expires = datetime.fromtimestamp(refresh_claims.exp, tz=timezone.utc)

        logger.info(f"User {user_id} authenticated successfully")

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            access_expires_at=access_expires,
            refresh_expires_at=refresh_expires,
        )

    async def verify_access_token(self, token: str) -> TokenClaims:
        """
        Verify and decode an access token.

        Args:
            token: JWT access token

        Returns:
            TokenClaims if valid

        Raises:
            TokenExpiredError: If token has expired
            TokenInvalidError: If token is invalid
            TokenRevokedError: If token has been revoked
        """
        try:
            public_key_pem = self.get_public_key_pem()

            payload = jwt.decode(
                token,
                public_key_pem,
                algorithms=["RS256"],
                issuer=self.issuer,
                audience=self.audience,
            )

            # Verify token type
            if payload.get("token_type") != TokenType.ACCESS.value:
                raise TokenInvalidError("Not an access token")

            # Check revocation
            jti = payload.get("jti")
            if jti and await self.revocation_store.is_revoked(jti):
                raise TokenRevokedError()

            return TokenClaims.from_dict(payload)

        except jwt.ExpiredSignatureError:
            raise TokenExpiredError()
        except jwt.InvalidTokenError as e:
            raise TokenInvalidError(str(e))

    async def refresh_tokens(self, refresh_token: str) -> TokenPair:
        """
        Refresh tokens using a valid refresh token.

        Implements token rotation - the old refresh token is invalidated
        and a new one is issued in the same family.

        Args:
            refresh_token: JWT refresh token

        Returns:
            New TokenPair

        Raises:
            TokenExpiredError: If refresh token has expired
            TokenInvalidError: If refresh token is invalid
            TokenRevokedError: If token family has been revoked
        """
        try:
            public_key_pem = self.get_public_key_pem()

            payload = jwt.decode(
                refresh_token,
                public_key_pem,
                algorithms=["RS256"],
                issuer=self.issuer,
                audience=self.audience,
            )

            # Verify token type
            if payload.get("token_type") != TokenType.REFRESH.value:
                raise TokenInvalidError("Not a refresh token")

            jti = payload.get("jti")
            family_id = payload.get("family_id")

            if not jti:
                raise TokenInvalidError("Missing token ID")

            # Validate and mark as used (with reuse detection)
            is_valid, user_id, stored_family_id = await self.refresh_store.validate_and_mark_used(jti)

            if not is_valid:
                raise TokenRevokedError("Refresh token invalid or already used")

            # Use stored family_id or generate new one
            family_id = stored_family_id or family_id or self._generate_family_id()

            # Extract claims from old token
            roles = payload.get("roles", [])
            permissions = payload.get("permissions", [])
            device_id = payload.get("device_id")
            session_id = payload.get("session_id") or self._generate_session_id()

            # Create new access token
            access_token, access_claims = self._create_token(
                user_id=user_id,
                token_type=TokenType.ACCESS,
                roles=roles,
                permissions=permissions,
                device_id=device_id,
                session_id=session_id,
            )

            # Create new refresh token (rotation)
            new_refresh_token, refresh_claims = self._create_token(
                user_id=user_id,
                token_type=TokenType.REFRESH,
                roles=roles,
                permissions=permissions,
                device_id=device_id,
                session_id=session_id,
                family_id=family_id,
            )

            # Store new refresh token
            await self.refresh_store.store(
                jti=refresh_claims.jti,
                family_id=family_id,
                user_id=user_id,
                expires_at=datetime.fromtimestamp(refresh_claims.exp, tz=timezone.utc),
            )

            access_expires = datetime.fromtimestamp(access_claims.exp, tz=timezone.utc)
            refresh_expires = datetime.fromtimestamp(refresh_claims.exp, tz=timezone.utc)

            logger.info(f"Tokens refreshed for user {user_id}")

            return TokenPair(
                access_token=access_token,
                refresh_token=new_refresh_token,
                access_expires_at=access_expires,
                refresh_expires_at=refresh_expires,
            )

        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Refresh token has expired")
        except jwt.InvalidTokenError as e:
            raise TokenInvalidError(str(e))

    async def revoke_token(self, token: str) -> None:
        """
        Revoke a token (logout).

        Args:
            token: JWT token to revoke
        """
        try:
            public_key_pem = self.get_public_key_pem()

            # Decode without verification to get claims even if expired
            payload = jwt.decode(
                token,
                public_key_pem,
                algorithms=["RS256"],
                options={"verify_exp": False},
            )

            jti = payload.get("jti")
            exp = payload.get("exp")

            if jti and exp:
                expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
                await self.revocation_store.revoke(jti, expires_at)
                logger.info(f"Token revoked: {jti[:8]}...")

        except jwt.InvalidTokenError:
            # Invalid tokens don't need to be revoked
            pass

    def get_public_key_pem(self) -> bytes:
        """
        Get the public key in PEM format for external verification.

        Returns:
            PEM-encoded public key
        """
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def save_keys(
        self,
        private_key_path: Path,
        public_key_path: Path
    ) -> None:
        """
        Save the key pair to files.

        Args:
            private_key_path: Path for private key
            public_key_path: Path for public key
        """
        # Save private key
        private_pem = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        private_key_path.write_bytes(private_pem)

        # Save public key
        public_pem = self.get_public_key_pem()
        public_key_path.write_bytes(public_pem)

        logger.info(f"Keys saved to {private_key_path} and {public_key_path}")


# =============================================================================
# Decorator for Permission Checking
# =============================================================================

def require_auth(
    permissions: Optional[List[str]] = None,
    roles: Optional[List[str]] = None
) -> Callable[[F], F]:
    """
    Decorator for protecting async functions with permission checks.

    The decorated function must receive 'token_claims' as an argument.

    Args:
        permissions: Required permissions (any match)
        roles: Required roles (any match)

    Returns:
        Decorator function

    Example:
        @require_auth(permissions=["read:images"])
        async def get_image(token_claims: TokenClaims, image_id: str):
            ...
    """
    permissions = permissions or []
    roles = roles or []

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            token_claims = kwargs.get("token_claims")

            if not token_claims:
                raise AuthError("No token claims provided")

            # Check roles
            if roles:
                user_roles = set(token_claims.roles)
                required_roles = set(roles)
                if not user_roles.intersection(required_roles):
                    raise AuthError(
                        f"Insufficient role. Required: {roles}",
                        code="INSUFFICIENT_ROLE"
                    )

            # Check permissions
            if permissions:
                user_permissions = set(token_claims.permissions)
                required_permissions = set(permissions)
                if not user_permissions.intersection(required_permissions):
                    raise AuthError(
                        f"Insufficient permissions. Required: {permissions}",
                        code="INSUFFICIENT_PERMISSIONS"
                    )

            return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Factory Function
# =============================================================================

def create_auth_manager(
    private_key_path: Optional[str] = None,
    public_key_path: Optional[str] = None,
    access_token_lifetime_minutes: int = 15,
    refresh_token_lifetime_days: int = 7,
    issuer: str = "nsi-edge",
    audience: str = "nsi-api",
    requests_per_minute: int = 60,
    burst_size: int = 10,
) -> JWTAuthenticationManager:
    """
    Factory function to create a configured JWTAuthenticationManager.

    Args:
        private_key_path: Path to private key file (optional)
        public_key_path: Path to public key file (optional)
        access_token_lifetime_minutes: Access token lifetime
        refresh_token_lifetime_days: Refresh token lifetime
        issuer: Token issuer
        audience: Token audience
        requests_per_minute: Rate limit requests per minute
        burst_size: Rate limit burst size

    Returns:
        Configured JWTAuthenticationManager
    """
    kwargs: Dict[str, Any] = {
        "access_token_lifetime_minutes": access_token_lifetime_minutes,
        "refresh_token_lifetime_days": refresh_token_lifetime_days,
        "issuer": issuer,
        "audience": audience,
        "rate_limiter": RateLimiter(requests_per_minute, burst_size),
    }

    if private_key_path and public_key_path:
        kwargs["private_key_path"] = Path(private_key_path)
        kwargs["public_key_path"] = Path(public_key_path)

    return JWTAuthenticationManager(**kwargs)
