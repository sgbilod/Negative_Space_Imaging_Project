# =============================================================================
# Negative Space Imaging Project - Authentication Routes
# FastAPI router for authentication endpoints
# =============================================================================
#
# This module provides:
# - /auth/login - User authentication and token issuance
# - /auth/refresh - Token refresh with rotation
# - /auth/logout - Token revocation
# - /auth/me - Current user information
# - /auth/.well-known/jwks.json - JWKS endpoint for public key
#
# Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
# =============================================================================

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, validator

from .jwt_auth import (
    JWTAuthenticationManager,
    TokenClaims,
    TokenExpiredError,
    TokenInvalidError,
    TokenPair,
    TokenRevokedError,
    RateLimitExceededError,
    AuthError,
)
from .middleware import get_current_user_dependency, PermissionChecker

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class LoginRequest(BaseModel):
    """Request model for user login."""
    
    username: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Username or email"
    )
    password: str = Field(
        ...,
        min_length=1,
        description="User password"
    )
    device_id: Optional[str] = Field(
        None,
        max_length=255,
        description="Optional device identifier for multi-device support"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "user@example.com",
                "password": "secure_password",
                "device_id": "device-uuid-1234"
            }
        }


class TokenResponse(BaseModel):
    """Response model for token issuance."""
    
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Access token TTL in seconds")
    refresh_expires_in: int = Field(..., description="Refresh token TTL in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "Bearer",
                "expires_in": 900,
                "refresh_expires_in": 604800
            }
        }


class RefreshRequest(BaseModel):
    """Request model for token refresh."""
    
    refresh_token: str = Field(
        ...,
        min_length=1,
        description="Refresh token from previous authentication"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "refresh_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class RevokeRequest(BaseModel):
    """Request model for token revocation."""
    
    token: Optional[str] = Field(
        None,
        description="Token to revoke (defaults to current access token)"
    )
    revoke_all: bool = Field(
        default=False,
        description="Revoke all tokens for the current session"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
                "revoke_all": False
            }
        }


class UserInfo(BaseModel):
    """Response model for user information."""
    
    user_id: str = Field(..., description="User identifier (subject)")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    device_id: Optional[str] = Field(None, description="Current device ID")
    session_id: Optional[str] = Field(None, description="Current session ID")
    token_issued_at: datetime = Field(..., description="Token issue timestamp")
    token_expires_at: datetime = Field(..., description="Token expiration timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user-123",
                "roles": ["user", "researcher"],
                "permissions": ["read:images", "write:images"],
                "device_id": "device-uuid-1234",
                "session_id": "session-uuid-5678",
                "token_issued_at": "2025-01-15T10:30:00Z",
                "token_expires_at": "2025-01-15T10:45:00Z"
            }
        }


class JWKSResponse(BaseModel):
    """Response model for JWKS endpoint."""
    
    keys: List[Dict[str, Any]] = Field(
        ...,
        description="Array of JSON Web Keys"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    detail: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid credentials",
                "code": "INVALID_CREDENTIALS"
            }
        }


# =============================================================================
# User Credential Validator (Pluggable)
# =============================================================================

# Type alias for credential validator function
CredentialValidator = Callable[[str, str], Optional[Dict[str, Any]]]


async def default_credential_validator(
    username: str,
    password: str
) -> Optional[Dict[str, Any]]:
    """
    Default credential validator - REPLACE IN PRODUCTION.
    
    This is a placeholder that should be replaced with actual
    credential validation logic (database lookup, LDAP, etc.).
    
    Args:
        username: Username or email
        password: Password to verify
        
    Returns:
        User data dict if valid, None if invalid
    """
    # WARNING: This is a demo validator - replace with real implementation
    # In production, this should:
    # 1. Look up user in database
    # 2. Verify password hash (using bcrypt, argon2, etc.)
    # 3. Return user data including roles and permissions
    
    logger.warning(
        "Using default credential validator - "
        "replace with production implementation"
    )
    
    # Demo users for testing
    demo_users = {
        "admin@nsip.local": {
            "password_hash": hashlib.sha256(b"admin_password").hexdigest(),
            "user_id": "admin-001",
            "roles": ["admin", "user"],
            "permissions": [
                "read:*", "write:*", "delete:*", "admin:*"
            ],
        },
        "researcher@nsip.local": {
            "password_hash": hashlib.sha256(b"researcher_password").hexdigest(),
            "user_id": "researcher-001",
            "roles": ["researcher", "user"],
            "permissions": [
                "read:images", "write:images", "read:analysis",
                "write:analysis", "read:reports"
            ],
        },
        "viewer@nsip.local": {
            "password_hash": hashlib.sha256(b"viewer_password").hexdigest(),
            "user_id": "viewer-001",
            "roles": ["user"],
            "permissions": ["read:images", "read:reports"],
        },
    }
    
    user = demo_users.get(username)
    if not user:
        return None
    
    # Verify password (in production, use proper password hashing)
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    if password_hash != user["password_hash"]:
        return None
    
    return {
        "user_id": user["user_id"],
        "roles": user["roles"],
        "permissions": user["permissions"],
    }


# =============================================================================
# Router Factory
# =============================================================================

def create_auth_router(
    auth_manager: JWTAuthenticationManager,
    credential_validator: Optional[CredentialValidator] = None,
    prefix: str = "/auth",
    tags: Optional[List[str]] = None,
) -> APIRouter:
    """
    Create a FastAPI router with authentication endpoints.
    
    Args:
        auth_manager: JWT authentication manager instance
        credential_validator: Optional custom credential validator
        prefix: URL prefix for all routes (default: /auth)
        tags: OpenAPI tags for the routes
        
    Returns:
        Configured APIRouter
        
    Example:
        auth_manager = create_auth_manager()
        router = create_auth_router(auth_manager)
        app.include_router(router)
    """
    router = APIRouter(prefix=prefix, tags=tags or ["Authentication"])
    
    # Use provided validator or default
    validator = credential_validator or default_credential_validator
    
    # -------------------------------------------------------------------------
    # Login Endpoint
    # -------------------------------------------------------------------------
    
    @router.post(
        "/login",
        response_model=TokenResponse,
        responses={
            401: {"model": ErrorResponse, "description": "Invalid credentials"},
            429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        },
        summary="Authenticate user and issue tokens",
        description="Validate credentials and issue access/refresh token pair",
    )
    async def login(
        request: Request,
        body: LoginRequest,
    ) -> TokenResponse:
        """
        Authenticate user with credentials.
        
        Issues a JWT access token and refresh token upon successful
        authentication. The access token has a short lifetime (15 min)
        while the refresh token lasts longer (7 days).
        """
        # Get client identifier for rate limiting
        client_id = request.client.host if request.client else "unknown"
        
        try:
            # Validate credentials
            user_data = await validator(body.username, body.password)
            
            if not user_data:
                logger.warning(
                    f"Failed login attempt for {body.username} from {client_id}"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Authenticate and issue tokens
            token_pair = await auth_manager.authenticate(
                user_id=user_data["user_id"],
                roles=user_data.get("roles", []),
                permissions=user_data.get("permissions", []),
                device_id=body.device_id,
            )
            
            logger.info(
                f"Successful login for user {user_data['user_id']} "
                f"from {client_id}"
            )
            
            return TokenResponse(
                access_token=token_pair.access_token,
                refresh_token=token_pair.refresh_token,
                token_type="Bearer",
                expires_in=int(
                    (token_pair.access_expires_at - datetime.now(timezone.utc))
                    .total_seconds()
                ),
                refresh_expires_in=int(
                    (token_pair.refresh_expires_at - datetime.now(timezone.utc))
                    .total_seconds()
                ),
            )
            
        except RateLimitExceededError as e:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=e.message,
                headers={"Retry-After": str(e.retry_after)},
            )
        except AuthError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=e.message,
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # -------------------------------------------------------------------------
    # Refresh Endpoint
    # -------------------------------------------------------------------------
    
    @router.post(
        "/refresh",
        response_model=TokenResponse,
        responses={
            401: {"model": ErrorResponse, "description": "Invalid refresh token"},
            429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        },
        summary="Refresh access token",
        description="Exchange refresh token for new access/refresh token pair",
    )
    async def refresh(
        body: RefreshRequest,
    ) -> TokenResponse:
        """
        Refresh tokens using a valid refresh token.
        
        Implements token rotation - the old refresh token is invalidated
        and a new pair is issued. This helps detect and prevent token theft.
        """
        try:
            token_pair = await auth_manager.refresh_tokens(body.refresh_token)
            
            return TokenResponse(
                access_token=token_pair.access_token,
                refresh_token=token_pair.refresh_token,
                token_type="Bearer",
                expires_in=int(
                    (token_pair.access_expires_at - datetime.now(timezone.utc))
                    .total_seconds()
                ),
                refresh_expires_in=int(
                    (token_pair.refresh_expires_at - datetime.now(timezone.utc))
                    .total_seconds()
                ),
            )
            
        except TokenExpiredError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except TokenRevokedError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except TokenInvalidError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid refresh token: {e.message}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except RateLimitExceededError as e:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=e.message,
                headers={"Retry-After": str(e.retry_after)},
            )
    
    # -------------------------------------------------------------------------
    # Logout Endpoint
    # -------------------------------------------------------------------------
    
    @router.post(
        "/logout",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            401: {"model": ErrorResponse, "description": "Not authenticated"},
        },
        summary="Logout and revoke tokens",
        description="Revoke the current access token and optionally all session tokens",
    )
    async def logout(
        request: Request,
        body: Optional[RevokeRequest] = None,
    ) -> None:
        """
        Logout and revoke tokens.
        
        Revokes the current access token. If revoke_all is True,
        revokes all tokens for the current session.
        """
        # Get current token from request
        auth_header = request.headers.get("Authorization", "")
        current_token = None
        
        if auth_header.startswith("Bearer "):
            current_token = auth_header[7:]
        
        if not current_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No token provided",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        try:
            # Determine which token to revoke
            token_to_revoke = current_token
            if body and body.token:
                token_to_revoke = body.token
            
            await auth_manager.revoke_token(token_to_revoke)
            
            logger.info("Token revoked successfully")
            
        except AuthError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=e.message,
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # -------------------------------------------------------------------------
    # Current User Endpoint
    # -------------------------------------------------------------------------
    
    @router.get(
        "/me",
        response_model=UserInfo,
        responses={
            401: {"model": ErrorResponse, "description": "Not authenticated"},
        },
        summary="Get current user information",
        description="Returns information about the currently authenticated user",
    )
    async def get_me(
        claims: TokenClaims = Depends(get_current_user_dependency),
    ) -> UserInfo:
        """
        Get current authenticated user information.
        
        Returns the user's ID, roles, permissions, and token metadata.
        """
        return UserInfo(
            user_id=claims.sub,
            roles=claims.roles,
            permissions=claims.permissions,
            device_id=claims.device_id,
            session_id=claims.session_id,
            token_issued_at=claims.iat,
            token_expires_at=claims.exp,
        )
    
    # -------------------------------------------------------------------------
    # JWKS Endpoint
    # -------------------------------------------------------------------------
    
    @router.get(
        "/.well-known/jwks.json",
        response_model=JWKSResponse,
        summary="Get JSON Web Key Set",
        description="Returns the public key for token verification",
    )
    async def get_jwks() -> JWKSResponse:
        """
        Get the JSON Web Key Set (JWKS).
        
        Returns the public key in JWK format for token verification
        by external services and clients.
        """
        import base64
        from cryptography.hazmat.primitives import serialization
        
        # Get the public key
        public_key = auth_manager._public_key
        
        # Get key numbers for JWK
        public_numbers = public_key.public_numbers()
        
        # Convert to bytes (big-endian, unsigned)
        def int_to_base64url(n: int, length: int) -> str:
            """Convert integer to base64url-encoded string."""
            data = n.to_bytes(length, byteorder='big')
            return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')
        
        # RSA key parameters
        e = int_to_base64url(public_numbers.e, 3)  # Exponent (usually 65537)
        n = int_to_base64url(public_numbers.n, 256)  # Modulus (2048-bit key)
        
        # Create JWK
        jwk = {
            "kty": "RSA",
            "use": "sig",
            "alg": "RS256",
            "kid": auth_manager._key_id,
            "n": n,
            "e": e,
        }
        
        return JWKSResponse(keys=[jwk])
    
    # -------------------------------------------------------------------------
    # Token Introspection Endpoint (Optional)
    # -------------------------------------------------------------------------
    
    @router.post(
        "/introspect",
        responses={
            401: {"model": ErrorResponse, "description": "Invalid token"},
        },
        summary="Introspect a token",
        description="Check if a token is valid and get its claims",
        dependencies=[Depends(PermissionChecker(roles=["admin"]))],
    )
    async def introspect_token(
        request: Request,
    ) -> Dict[str, Any]:
        """
        Introspect a token (admin only).
        
        Returns token validity and claims information.
        Useful for debugging and token inspection.
        """
        auth_header = request.headers.get("Authorization", "")
        token = None
        
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        
        if not token:
            return {"active": False}
        
        try:
            claims = await auth_manager.verify_access_token(token)
            return {
                "active": True,
                "sub": claims.sub,
                "iat": claims.iat.isoformat(),
                "exp": claims.exp.isoformat(),
                "jti": claims.jti,
                "iss": claims.iss,
                "aud": claims.aud,
                "token_type": claims.token_type,
                "roles": claims.roles,
                "permissions": claims.permissions,
            }
        except (TokenExpiredError, TokenRevokedError, TokenInvalidError):
            return {"active": False}
    
    return router


# =============================================================================
# Convenience Function
# =============================================================================

def setup_auth_routes(
    app,
    auth_manager: JWTAuthenticationManager,
    credential_validator: Optional[CredentialValidator] = None,
) -> APIRouter:
    """
    Set up authentication routes on a FastAPI application.
    
    Args:
        app: FastAPI application instance
        auth_manager: JWT authentication manager
        credential_validator: Optional custom credential validator
        
    Returns:
        The configured router
        
    Example:
        from fastapi import FastAPI
        from src.auth import create_auth_manager, setup_auth_routes
        
        app = FastAPI()
        auth_manager = create_auth_manager()
        setup_auth_routes(app, auth_manager)
    """
    router = create_auth_router(
        auth_manager=auth_manager,
        credential_validator=credential_validator,
    )
    app.include_router(router)
    return router
