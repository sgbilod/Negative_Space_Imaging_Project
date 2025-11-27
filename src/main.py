# =============================================================================
# Negative Space Imaging Project - Main Application Entry Point
# =============================================================================
#
# This module serves as the main entry point for the NSIP platform.
# It integrates all components:
# - Agentic Systems (DeepAgent Supervisor)
# - Memory System (Persistent Memory Manager)
# - Authentication (JWT Auth with RS256)
# - API Layer (FastAPI routers)
#
# Usage:
#     python -m src.main --mode dev --port 8080
#     python -m src.main --mode edge --config config/edge-config.yml
#     python -m src.main --mode cloud --reload
#
# Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
# =============================================================================

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
import yaml
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("NSI-Main")


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return {}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge multiple configuration dictionaries.
    
    Later configs override earlier ones.
    """
    result = {}
    for config in configs:
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    return result


# =============================================================================
# Application State
# =============================================================================

class ApplicationState:
    """
    Global application state container.
    
    Manages lifecycle of all major components:
    - DeepAgent Supervisor
    - Persistent Memory Manager
    - JWT Authentication Manager
    """
    
    def __init__(self):
        self.supervisor = None
        self.memory_manager = None
        self.auth_manager = None
        self.config: Dict[str, Any] = {}
        self.mode: str = "dev"
        self._shutdown_event: Optional[asyncio.Event] = None
    
    @property
    def is_dev_mode(self) -> bool:
        """Check if running in development mode."""
        return self.mode == "dev"
    
    @property
    def is_edge_mode(self) -> bool:
        """Check if running in edge deployment mode."""
        return self.mode == "edge"
    
    @property
    def is_cloud_mode(self) -> bool:
        """Check if running in cloud deployment mode."""
        return self.mode == "cloud"


# Global application state
app_state = ApplicationState()


# =============================================================================
# Component Initialization
# =============================================================================

async def initialize_memory_manager(config: Dict[str, Any]) -> None:
    """Initialize the Persistent Memory Manager."""
    try:
        from src.agents import create_memory_manager
        
        data_config = config.get('data', {})
        memory_path = data_config.get('memory_dir', './data/memory')
        
        agents_config = config.get('agents', {}).get('memory', {})
        decay_enabled = agents_config.get('decay_enabled', True)
        decay_interval = agents_config.get('decay_interval_hours', 24)
        min_relevance = agents_config.get('min_relevance_threshold', 0.1)
        
        app_state.memory_manager = create_memory_manager(
            storage_path=memory_path,
            decay_interval_hours=decay_interval,
            min_relevance_threshold=min_relevance,
        )
        
        if decay_enabled:
            await app_state.memory_manager.start_decay_processing()
            logger.info("Memory Manager initialized with decay processing enabled")
        else:
            logger.info("Memory Manager initialized (decay processing disabled)")
            
    except ImportError as e:
        logger.warning(f"Memory manager not available: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize memory manager: {e}")


async def initialize_supervisor(config: Dict[str, Any]) -> None:
    """Initialize the DeepAgent Supervisor."""
    try:
        from src.agents import create_imaging_supervisor
        
        agents_config = config.get('agents', {}).get('supervisor', {})
        max_concurrent = agents_config.get('max_concurrent_tasks', 8)
        
        app_state.supervisor = create_imaging_supervisor(
            max_concurrent_tasks=max_concurrent
        )
        
        agent_count = len(app_state.supervisor.agents)
        logger.info(f"Agent Supervisor initialized with {agent_count} agents")
        
    except ImportError as e:
        logger.warning(f"Agent supervisor not available: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize agent supervisor: {e}")


async def initialize_auth_manager(config: Dict[str, Any]) -> None:
    """Initialize the JWT Authentication Manager."""
    try:
        from src.auth import create_auth_manager
        
        security_config = config.get('security', {}).get('authentication', {})
        
        # Key paths
        keys_dir = Path(config.get('security', {}).get('keys_dir', './keys'))
        private_key_path = keys_dir / 'private.pem'
        public_key_path = keys_dir / 'public.pem'
        
        # Token lifetimes
        access_lifetime = security_config.get('token_lifetime_minutes', 15)
        refresh_lifetime = security_config.get('refresh_token_lifetime_days', 7)
        
        # Issuer/Audience
        issuer = security_config.get('issuer', 'nsi-auth')
        audience = security_config.get('audience', 'nsi-api')
        
        # Create auth manager
        app_state.auth_manager = create_auth_manager(
            private_key_path=str(private_key_path) if private_key_path.exists() else None,
            public_key_path=str(public_key_path) if public_key_path.exists() else None,
            access_token_lifetime_minutes=access_lifetime,
            refresh_token_lifetime_days=refresh_lifetime,
            issuer=issuer,
            audience=audience,
        )
        
        logger.info("Authentication Manager initialized")
        
    except ImportError as e:
        logger.warning(f"Auth manager not available: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize auth manager: {e}")


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown of all components.
    """
    logger.info("=" * 60)
    logger.info("Starting Negative Space Imaging System...")
    logger.info(f"Mode: {app_state.mode}")
    logger.info("=" * 60)
    
    config = app_state.config
    
    # Initialize components in order
    await initialize_memory_manager(config)
    await initialize_supervisor(config)
    await initialize_auth_manager(config)
    
    # Create data directories
    data_config = config.get('data', {})
    for dir_key in ['input_dir', 'output_dir', 'cache_dir', 'memory_dir']:
        dir_path = data_config.get(dir_key)
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("All components initialized successfully")
    logger.info("=" * 60)
    
    yield
    
    # Cleanup
    logger.info("=" * 60)
    logger.info("Shutting down Negative Space Imaging System...")
    
    if app_state.memory_manager:
        try:
            app_state.memory_manager.stop_decay_processing()
            logger.info("Memory Manager shutdown complete")
        except Exception as e:
            logger.error(f"Error during memory manager shutdown: {e}")
    
    if app_state.supervisor:
        try:
            # Supervisor cleanup if needed
            logger.info("Agent Supervisor shutdown complete")
        except Exception as e:
            logger.error(f"Error during supervisor shutdown: {e}")
    
    logger.info("Shutdown complete")
    logger.info("=" * 60)


# =============================================================================
# Application Factory
# =============================================================================

def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured FastAPI application
    """
    app_state.config = config or {}
    
    # Determine API version and info from config
    system_config = app_state.config.get('system', {})
    api_version = system_config.get('api_version', '1.0.0')
    
    app = FastAPI(
        title="Negative Space Imaging API",
        description=(
            "Advanced medical/astronomical imaging platform with AI/ML integration.\n\n"
            "## Features\n"
            "- Agentic imaging pipeline orchestration\n"
            "- Persistent memory with decay and similarity search\n"
            "- JWT authentication with RS256 signing\n"
            "- Edge and cloud deployment modes\n"
        ),
        version=api_version,
        lifespan=lifespan,
        docs_url="/docs" if app_state.is_dev_mode else None,
        redoc_url="/redoc" if app_state.is_dev_mode else None,
    )
    
    # ==========================================================================
    # Middleware
    # ==========================================================================
    
    # CORS middleware
    cors_config = app_state.config.get('security', {}).get('cors', {})
    allowed_origins = cors_config.get('origins', ["*"] if app_state.is_dev_mode else [])
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Authentication middleware (for non-dev modes)
    if not app_state.is_dev_mode and app_state.auth_manager:
        try:
            from src.auth import AuthenticationMiddleware
            
            excluded_paths = [
                "/health", "/healthz", "/ready", "/live", "/metrics",
                "/auth/login", "/auth/refresh", "/auth/.well-known/jwks.json",
                "/docs", "/redoc", "/openapi.json",
            ]
            
            app.add_middleware(
                AuthenticationMiddleware,
                auth_manager=app_state.auth_manager,
                excluded_paths=excluded_paths,
            )
            logger.info("Authentication middleware enabled")
        except ImportError:
            logger.warning("Authentication middleware not available")
    
    # ==========================================================================
    # Routers
    # ==========================================================================
    
    # Health endpoints (always included)
    try:
        from src.api import health_router
        app.include_router(health_router)
        logger.debug("Health router included")
    except ImportError as e:
        logger.warning(f"Health router not available: {e}")
    
    # Auth endpoints
    if app_state.auth_manager:
        try:
            from src.auth import create_auth_router
            auth_router = create_auth_router(app_state.auth_manager)
            app.include_router(auth_router)
            logger.debug("Auth router included")
        except ImportError as e:
            logger.warning(f"Auth router not available: {e}")
    
    # Imaging endpoints
    try:
        from src.api import imaging_router
        app.include_router(imaging_router)
        logger.debug("Imaging router included")
    except ImportError as e:
        logger.warning(f"Imaging router not available: {e}")
    
    # ==========================================================================
    # Exception Handlers
    # ==========================================================================
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        # Don't expose internal errors in production
        if app_state.is_dev_mode:
            detail = str(exc)
        else:
            detail = "An internal error occurred"
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": detail,
                "code": "INTERNAL_ERROR"
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle validation errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Validation error",
                "detail": str(exc),
                "code": "VALIDATION_ERROR"
            }
        )
    
    # ==========================================================================
    # Root Endpoint
    # ==========================================================================
    
    @app.get("/", tags=["Root"])
    async def root():
        """API root endpoint."""
        return {
            "name": "Negative Space Imaging API",
            "version": api_version,
            "mode": app_state.mode,
            "docs": "/docs" if app_state.is_dev_mode else None,
            "health": "/health",
        }
    
    return app


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Negative Space Imaging System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --mode dev --port 8080
  python -m src.main --mode edge --config config/edge-config.yml
  python -m src.main --mode cloud --host 0.0.0.0 --port 80
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["edge", "cloud", "dev"],
        default="dev",
        help="Deployment mode (default: dev)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Path to configuration file (default: config/config.yml)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level (default: info)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line args
    config['mode'] = args.mode
    config.setdefault('system', {})['mode'] = args.mode
    
    # Set application state
    app_state.mode = args.mode
    app_state.config = config
    
    # Create app
    app = create_app(config)
    
    # Log startup info
    logger.info(f"Starting server in {args.mode} mode")
    logger.info(f"Listening on {args.host}:{args.port}")
    
    # Run server
    uvicorn_config = {
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "access_log": args.mode == "dev",
    }
    
    if args.reload and args.mode == "dev":
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = ["src"]
    
    if args.workers > 1 and not args.reload:
        uvicorn_config["workers"] = args.workers
    
    uvicorn.run(app, **uvicorn_config)


# =============================================================================
# ASGI Application (for production servers)
# =============================================================================

def get_app() -> FastAPI:
    """
    Get the ASGI application.
    
    Use this for production ASGI servers like Gunicorn:
        gunicorn src.main:get_app -k uvicorn.workers.UvicornWorker
    """
    config = load_config(os.environ.get("NSI_CONFIG", "config/config.yml"))
    mode = os.environ.get("NSI_MODE", "cloud")
    
    app_state.mode = mode
    app_state.config = config
    
    return create_app(config)


# Create default app instance for imports
app = None


if __name__ == "__main__":
    main()
