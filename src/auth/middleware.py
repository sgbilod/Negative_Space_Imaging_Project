# =============================================================================
# Negative Space Imaging Project - Authentication Middleware
# FastAPI/Starlette middleware for JWT authentication
# =============================================================================
#
# This module provides:
# - AuthenticationMiddleware for automatic token verification
# - PermissionChecker dependency for route-level authorization
# - Helper functions for accessing current user information
#
# Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
# =============================================================================

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Set

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse, Response

from .jwt_auth import (
    JWTAuthenticationManager,
    TokenClaims,
    TokenExpiredError,
    TokenInvalidError,
    TokenRevokedError,
    RateLimitExceededError,
    AuthError,
)

logger = logging.getLogger(__name__)

# HTTPBearer security scheme for OpenAPI documentation
http_bearer = HTTPBearer(auto_error=False)


# =============================================================================
# Authentication Middleware
# =============================================================================

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware for JWT authentication.
    
    This middleware intercepts all requests and:
    - Skips authentication for excluded paths
    - Extracts Bearer token from Authorization header
    - Verifies the token and populates request.state with user info
    - Returns appropriate error responses for authentication failures
    
    Attributes:
        auth_manager: JWT authentication manager instance
        excluded_paths: Paths that don't require authentication
        excluded_prefixes: Path prefixes that don't require authentication
    """
    
    DEFAULT_EXCLUDED_PATHS: Set[str] = {
        "/health",
        "/healthz",
        "/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/auth/login",
        "/auth/refresh",
        "/auth/.well-known/jwks.json",
    }
    
    DEFAULT_EXCLUDED_PREFIXES: List[str] = [
        "/static/",
        "/public/",
    ]
    
    def __init__(
        self,
        app,
        auth_manager: JWTAuthenticationManager,
        excluded_paths: Optional[Sequence[str]] = None,
        excluded_prefixes: Optional[Sequence[str]] = None,
    ):
        """
        Initialize the authentication middleware.
        
        Args:
            app: ASGI application
            auth_manager: JWT authentication manager
            excluded_paths: Additional paths to exclude from authentication
            excluded_prefixes: Path prefixes to exclude from authentication
        """
        super().__init__(app)
        self.auth_manager = auth_manager
        
        # Combine default and custom excluded paths
        self.excluded_paths = set(self.DEFAULT_EXCLUDED_PATHS)
        if excluded_paths:
            self.excluded_paths.update(excluded_paths)
        
        # Combine default and custom excluded prefixes
        self.excluded_prefixes = list(self.DEFAULT_EXCLUDED_PREFIXES)
        if excluded_prefixes:
            self.excluded_prefixes.extend(excluded_prefixes)
        
        logger.debug(
            f"AuthenticationMiddleware initialized with "
            f"{len(self.excluded_paths)} excluded paths"
        )
    
    def _is_excluded(self, path: str) -> bool:
        """
        Check if a path should be excluded from authentication.
        
        Args:
            path: Request path
            
        Returns:
            True if path is excluded, False otherwise
        """
        # Check exact matches
        if path in self.excluded_paths:
            return True
        
        # Check prefixes
        for prefix in self.excluded_prefixes:
            if path.startswith(prefix):
                return True
        
        return False
    
    def _extract_token(self, request: StarletteRequest) -> Optional[str]:
        """
        Extract Bearer token from Authorization header.
        
        Args:
            request: HTTP request
            
        Returns:
            Token string or None if not found
        """
        auth_header = request.headers.get("Authorization", "")
        
        if not auth_header:
            return None
        
        parts = auth_header.split()
        
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        
        return parts[1]
    
    async def dispatch(
        self,
        request: StarletteRequest,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and verify authentication.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint in chain
            
        Returns:
            Response from downstream or error response
        """
        path = request.url.path
        
        # Skip authentication for excluded paths
        if self._is_excluded(path):
            return await call_next(request)
        
        # Extract token
        token = self._extract_token(request)
        
        if not token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "Missing authentication token",
                    "code": "MISSING_TOKEN"
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        try:
            # Verify token
            claims = await self.auth_manager.verify_access_token(token)
            
            # Populate request state
            request.state.user = claims
            request.state.user_id = claims.sub
            request.state.roles = claims.roles
            request.state.permissions = claims.permissions
            request.state.token = token
            
            # Continue to next handler
            return await call_next(request)
            
        except TokenExpiredError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "Token has expired",
                    "code": "TOKEN_EXPIRED"
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        except TokenRevokedError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "Token has been revoked",
                    "code": "TOKEN_REVOKED"
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        except TokenInvalidError as e:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": f"Invalid token: {e.message}",
                    "code": "TOKEN_INVALID"
                },
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        except RateLimitExceededError as e:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": e.message,
                    "code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": e.retry_after
                },
                headers={"Retry-After": str(e.retry_after)}
            )
            
        except Exception as e:
            logger.error(f"Authentication error: {e}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Internal authentication error",
                    "code": "AUTH_ERROR"
                }
            )


# =============================================================================
# Permission Checker Dependency
# =============================================================================

class PermissionChecker:
    """
    FastAPI dependency for checking permissions and roles.
    
    Use this as a dependency in route handlers to enforce
    authorization requirements.
    
    Example:
        @app.get("/admin")
        async def admin_route(
            _: None = Depends(PermissionChecker(roles=["admin"]))
        ):
            return {"message": "Admin access granted"}
    """
    
    def __init__(
        self,
        required_permissions: Optional[List[str]] = None,
        required_roles: Optional[List[str]] = None,
        require_all_permissions: bool = False,
        require_all_roles: bool = False,
    ):
        """
        Initialize the permission checker.
        
        Args:
            required_permissions: List of required permissions (any match by default)
            required_roles: List of required roles (any match by default)
            require_all_permissions: If True, all permissions must match
            require_all_roles: If True, all roles must match
        """
        self.required_permissions = set(required_permissions or [])
        self.required_roles = set(required_roles or [])
        self.require_all_permissions = require_all_permissions
        self.require_all_roles = require_all_roles
    
    async def __call__(self, request: Request) -> TokenClaims:
        """
        Validate permissions for the current request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            TokenClaims if authorized
            
        Raises:
            HTTPException: 401 if not authenticated, 403 if unauthorized
        """
        # Check if user is authenticated
        if not hasattr(request.state, "user"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        claims: TokenClaims = request.state.user
        user_roles = set(claims.roles)
        user_permissions = set(claims.permissions)
        
        # Check roles
        if self.required_roles:
            if self.require_all_roles:
                if not self.required_roles.issubset(user_roles):
                    missing = self.required_roles - user_roles
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Missing required roles: {list(missing)}"
                    )
            else:
                if not user_roles.intersection(self.required_roles):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Requires one of roles: {list(self.required_roles)}"
                    )
        
        # Check permissions
        if self.required_permissions:
            if self.require_all_permissions:
                if not self.required_permissions.issubset(user_permissions):
                    missing = self.required_permissions - user_permissions
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Missing required permissions: {list(missing)}"
                    )
            else:
                if not user_permissions.intersection(self.required_permissions):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Requires one of permissions: {list(self.required_permissions)}"
                    )
        
        return claims


# =============================================================================
# Helper Functions
# =============================================================================

def require_permissions(*permissions: str) -> PermissionChecker:
    """
    Create a Depends-compatible permission checker.
    
    Args:
        *permissions: Required permissions (any match)
        
    Returns:
        PermissionChecker instance
        
    Example:
        @app.get("/images")
        async def get_images(
            _: None = Depends(require_permissions("read:images"))
        ):
            return {"images": [...]}
    """
    return PermissionChecker(required_permissions=list(permissions))


def require_roles(*roles: str) -> PermissionChecker:
    """
    Create a Depends-compatible role checker.
    
    Args:
        *roles: Required roles (any match)
        
    Returns:
        PermissionChecker instance
        
    Example:
        @app.get("/admin")
        async def admin_route(
            _: None = Depends(require_roles("admin"))
        ):
            return {"message": "Admin access"}
    """
    return PermissionChecker(required_roles=list(roles))


def require_all_permissions(*permissions: str) -> PermissionChecker:
    """
    Create a Depends-compatible permission checker requiring ALL permissions.
    
    Args:
        *permissions: All required permissions
        
    Returns:
        PermissionChecker instance
    """
    return PermissionChecker(
        required_permissions=list(permissions),
        require_all_permissions=True
    )


def require_all_roles(*roles: str) -> PermissionChecker:
    """
    Create a Depends-compatible role checker requiring ALL roles.
    
    Args:
        *roles: All required roles
        
    Returns:
        PermissionChecker instance
    """
    return PermissionChecker(
        required_roles=list(roles),
        require_all_roles=True
    )


def get_current_user(request: Request) -> TokenClaims:
    """
    Get the current authenticated user from request state.
    
    Args:
        request: FastAPI/Starlette request object
        
    Returns:
        TokenClaims for the authenticated user
        
    Raises:
        HTTPException: 401 if not authenticated
        
    Example:
        @app.get("/me")
        async def get_me(request: Request):
            claims = get_current_user(request)
            return {"user_id": claims.sub}
    """
    if not hasattr(request.state, "user"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return request.state.user


async def get_current_user_dependency(request: Request) -> TokenClaims:
    """
    Async dependency version of get_current_user.
    
    Use this with Depends() for dependency injection.
    
    Args:
        request: FastAPI request object
        
    Returns:
        TokenClaims for the authenticated user
        
    Example:
        @app.get("/me")
        async def get_me(user: TokenClaims = Depends(get_current_user_dependency)):
            return {"user_id": user.sub}
    """
    return get_current_user(request)


def get_optional_user(request: Request) -> Optional[TokenClaims]:
    """
    Get the current user if authenticated, None otherwise.
    
    Useful for endpoints that have optional authentication.
    
    Args:
        request: FastAPI/Starlette request object
        
    Returns:
        TokenClaims if authenticated, None otherwise
        
    Example:
        @app.get("/public")
        async def public_route(request: Request):
            user = get_optional_user(request)
            if user:
                return {"message": f"Hello, {user.sub}"}
            return {"message": "Hello, guest"}
    """
    if hasattr(request.state, "user"):
        return request.state.user
    return None


# =============================================================================
# Additional Utilities
# =============================================================================

class RoleBasedAccessControl:
    """
    Utility class for role-based access control with inheritance.
    
    Supports role hierarchies where higher roles inherit permissions
    from lower roles.
    """
    
    def __init__(self, role_hierarchy: Optional[dict] = None):
        """
        Initialize RBAC with optional role hierarchy.
        
        Args:
            role_hierarchy: Dict mapping roles to their parent roles
                           Example: {"admin": ["user"], "superadmin": ["admin"]}
        """
        self.role_hierarchy = role_hierarchy or {
            "superadmin": ["admin"],
            "admin": ["moderator"],
            "moderator": ["user"],
            "user": [],
        }
    
    def get_effective_roles(self, roles: List[str]) -> Set[str]:
        """
        Get all effective roles including inherited ones.
        
        Args:
            roles: User's assigned roles
            
        Returns:
            Set of all effective roles
        """
        effective = set(roles)
        to_process = list(roles)
        
        while to_process:
            role = to_process.pop()
            if role in self.role_hierarchy:
                inherited = self.role_hierarchy[role]
                for parent_role in inherited:
                    if parent_role not in effective:
                        effective.add(parent_role)
                        to_process.append(parent_role)
        
        return effective
    
    def has_role(self, user_roles: List[str], required_role: str) -> bool:
        """
        Check if user has a role (including inherited).
        
        Args:
            user_roles: User's assigned roles
            required_role: Role to check for
            
        Returns:
            True if user has the role
        """
        effective_roles = self.get_effective_roles(user_roles)
        return required_role in effective_roles
