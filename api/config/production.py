"""Production configuration for FastAPI application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from .middleware.error_handling import error_handler
import logging
import os
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('production.log'),
        logging.StreamHandler()
    ]
)

class Config:
    # API Settings
    API_VERSION = "1.0.0"
    API_TITLE = "Negative Space Imaging API"
    API_DESCRIPTION = "Advanced imaging system with quantum-enhanced security"

    # Security
    JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")  # Change in production
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30

    # CORS
    CORS_ORIGINS: List[str] = [
        "https://app.negativespace.com"
    ]

    # Rate Limiting
    RATE_LIMIT_BURST = 100
    RATE_LIMIT_PERIOD = 60  # seconds

    # Database (async)
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://user:password@localhost:5432/negativespace"
    )
    # Redis (for caching and rate limiting)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

    # File Storage
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

    # Processing
    MAX_CONCURRENT_JOBS = 10
    JOB_TIMEOUT = 300  # seconds

    # WebSocket
    WS_PING_INTERVAL = 30  # seconds
    WS_PING_TIMEOUT = 10  # seconds

    # Monitoring
    ENABLE_METRICS = True
    METRICS_PREFIX = "negativespace"


def create_production_app() -> FastAPI:
    app = FastAPI(
        title=Config.API_TITLE,
        description=Config.API_DESCRIPTION,
        version=Config.API_VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=Config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # Add Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add error handling middleware
    app.middleware("http")(error_handler)
    # Add profiling middleware
    from api.middleware.profiling import profiling_middleware
    app.middleware("http")(profiling_middleware)

    return app
