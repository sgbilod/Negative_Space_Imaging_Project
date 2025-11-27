# =============================================================================
# Negative Space Imaging Project - Authentication Module
# =============================================================================
#
# Enterprise-grade JWT authentication for the NSIP platform.
#
# Features:
# - RS256 asymmetric key signing for enhanced security
# - Short-lived access tokens (15 min) with longer refresh tokens (7 days)
# - Refresh token rotation with reuse detection
# - Token revocation store with automatic cleanup
# - Rate limiting to prevent brute force attacks
# - FastAPI middleware for automatic authentication
# - Permission-based access control
#
# Usage:
#     from src.auth import (
#         create_auth_manager,
#         create_auth_router,
#         AuthenticationMiddleware,
#         require_permissions,
#         require_roles,
#     )
#
#     # Create auth manager
#     auth_manager = create_auth_manager(
#         private_key_path="keys/private.pem",
#         public_key_path="keys/public.pem",
#     )
#
#     # Set up FastAPI app
#     app = FastAPI()
#     app.add_middleware(AuthenticationMiddleware, auth_manager=auth_manager)
#     app.include_router(create_auth_router(auth_manager))
#
#     # Protected route with permission check
#     @app.get("/images")
#     async def get_images(
#         _: None = Depends(require_permissions("read:images"))
#     ):
#         return {"images": [...]}
#
# Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
# =============================================================================

from .jwt_auth import (
    # Enums
    TokenType,

    # Exceptions
    AuthError,
    TokenExpiredError,
    TokenInvalidError,
    TokenRevokedError,
    RateLimitExceededError,

    # Data classes
    TokenPair,
    TokenClaims,

    # Stores
    TokenRevocationStore,
    RefreshTokenStore,
    RateLimiter,

    # Main class
    JWTAuthenticationManager,

    # Decorator
    require_auth,

    # Factory
    create_auth_manager,
)

from .middleware import (
    # Middleware
    AuthenticationMiddleware,

    # Dependencies
    PermissionChecker,

    # Convenience functions
    require_permissions,
    require_roles,
    require_all_permissions,
    require_all_roles,
    get_current_user,
    get_current_user_dependency,
    get_optional_user,

    # Utilities
    RoleBasedAccessControl,

    # Security scheme
    http_bearer,
)

from .routes import (
    # Request models
    LoginRequest,
    RefreshRequest,
    RevokeRequest,

    # Response models
    TokenResponse,
    UserInfo,
    JWKSResponse,
    ErrorResponse,

    # Router factory
    create_auth_router,
    setup_auth_routes,

    # Credential validator type
    CredentialValidator,
    default_credential_validator,
)

__all__ = [
    # === jwt_auth ===
    # Enums
    "TokenType",

    # Exceptions
    "AuthError",
    "TokenExpiredError",
    "TokenInvalidError",
    "TokenRevokedError",
    "RateLimitExceededError",

    # Data classes
    "TokenPair",
    "TokenClaims",

    # Stores
    "TokenRevocationStore",
    "RefreshTokenStore",
    "RateLimiter",

    # Main class
    "JWTAuthenticationManager",

    # Decorator
    "require_auth",

    # Factory
    "create_auth_manager",

    # === middleware ===
    # Middleware
    "AuthenticationMiddleware",

    # Dependencies
    "PermissionChecker",

    # Convenience functions
    "require_permissions",
    "require_roles",
    "require_all_permissions",
    "require_all_roles",
    "get_current_user",
    "get_current_user_dependency",
    "get_optional_user",

    # Utilities
    "RoleBasedAccessControl",

    # Security scheme
    "http_bearer",

    # === routes ===
    # Request models
    "LoginRequest",
    "RefreshRequest",
    "RevokeRequest",

    # Response models
    "TokenResponse",
    "UserInfo",
    "JWKSResponse",
    "ErrorResponse",

    # Router factory
    "create_auth_router",
    "setup_auth_routes",

    # Credential validator
    "CredentialValidator",
    "default_credential_validator",
]

__version__ = "1.0.0"
__author__ = "Stephen Bilodeau"
