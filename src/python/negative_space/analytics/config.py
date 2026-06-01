"""
Analytics Engine Configuration Module

Provides centralized configuration management for the analytics engine,
including database connections, performance tuning, and feature flags.

Design Principles:
- Environment-based configuration (dev/staging/prod)
- Sensible defaults with override capability
- Validation on initialization
- Type-safe configuration with dataclasses
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Supported environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AnomalyDetectionMethod(Enum):
    """Supported anomaly detection methods."""
    ISOLATION_FOREST = "isolation_forest"
    Z_SCORE = "zscore"
    IQR = "iqr"
    AUTOENCODER = "autoencoder"  # Defer to Phase 6.2


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration."""

    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    database: str = field(default_factory=lambda: os.getenv("DB_NAME", "negative_space"))
    username: str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))

    # Connection pooling
    pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "10")))
    max_overflow: int = field(default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "20")))
    pool_timeout: int = field(default_factory=lambda: int(os.getenv("DB_POOL_TIMEOUT", "30")))
    pool_recycle: int = field(default_factory=lambda: int(os.getenv("DB_POOL_RECYCLE", "3600")))

    # Partition configuration
    enable_partitioning: bool = field(default_factory=lambda: os.getenv("DB_ENABLE_PARTITIONING", "true").lower() == "true")
    partition_retention_days: int = field(default_factory=lambda: int(os.getenv("DB_PARTITION_RETENTION_DAYS", "90")))

    def __post_init__(self):
        """Validate database configuration."""
        if not self.host:
            raise ValueError("DB_HOST environment variable not set")
        if not self.password and self.host != "localhost":
            logger.warning("Database password not set - may fail in production")

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return (
            f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        )

    @property
    def psycopg2_dict(self) -> Dict[str, Any]:
        """Generate psycopg2 connection dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.username,
            "password": self.password,
        }


@dataclass
class RedisConfig:
    """Redis caching configuration."""

    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    database: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", None))

    # Cache settings
    default_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("REDIS_DEFAULT_TTL", "3600")))
    metrics_cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("REDIS_METRICS_TTL", "300")))
    aggregation_cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("REDIS_AGG_TTL", "600")))

    # Connection pooling
    max_connections: int = field(default_factory=lambda: int(os.getenv("REDIS_MAX_CONN", "50")))
    socket_timeout: int = field(default_factory=lambda: int(os.getenv("REDIS_TIMEOUT", "5")))

    @property
    def connection_string(self) -> str:
        """Generate Redis connection string."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"redis://{self.host}:{self.port}/{self.database}"


@dataclass
class StreamingConfig:
    """Real-time streaming configuration."""

    # Event ingestion
    event_batch_size: int = field(default_factory=lambda: int(os.getenv("STREAM_BATCH_SIZE", "100")))
    event_batch_timeout_ms: int = field(default_factory=lambda: int(os.getenv("STREAM_BATCH_TIMEOUT_MS", "1000")))

    # Windowing
    window_size_seconds: int = field(default_factory=lambda: int(os.getenv("STREAM_WINDOW_SIZE_SEC", "60")))
    window_slide_seconds: int = field(default_factory=lambda: int(os.getenv("STREAM_WINDOW_SLIDE_SEC", "30")))

    # Processing
    max_queue_size: int = field(default_factory=lambda: int(os.getenv("STREAM_MAX_QUEUE", "10000")))
    processing_threads: int = field(default_factory=lambda: int(os.getenv("STREAM_THREADS", "4")))

    # Buffering and aggregation
    enable_aggregation: bool = field(default_factory=lambda: os.getenv("STREAM_ENABLE_AGG", "true").lower() == "true")
    aggregation_interval_seconds: int = field(default_factory=lambda: int(os.getenv("STREAM_AGG_INTERVAL", "60")))


@dataclass
class AnomalyDetectionConfig:
    """Anomaly detection configuration."""

    # Default method
    default_method: str = field(default_factory=lambda: os.getenv("ANOMALY_METHOD", AnomalyDetectionMethod.ISOLATION_FOREST.value))

    # Isolation Forest settings
    isolation_forest_n_estimators: int = field(default_factory=lambda: int(os.getenv("ANOMALY_IF_ESTIMATORS", "100")))
    isolation_forest_contamination: float = field(default_factory=lambda: float(os.getenv("ANOMALY_IF_CONTAMINATION", "0.1")))
    isolation_forest_random_state: int = field(default_factory=lambda: int(os.getenv("ANOMALY_IF_RANDOM_STATE", "42")))

    # Z-Score settings
    zscore_threshold: float = field(default_factory=lambda: float(os.getenv("ANOMALY_ZSCORE_THRESHOLD", "3.0")))

    # IQR settings
    iqr_multiplier: float = field(default_factory=lambda: float(os.getenv("ANOMALY_IQR_MULTIPLIER", "1.5")))

    # Training data
    min_training_samples: int = field(default_factory=lambda: int(os.getenv("ANOMALY_MIN_SAMPLES", "100")))
    retraining_interval_hours: int = field(default_factory=lambda: int(os.getenv("ANOMALY_RETRAIN_HOURS", "24")))

    # Alert settings
    min_confidence_threshold: float = field(default_factory=lambda: float(os.getenv("ANOMALY_MIN_CONFIDENCE", "0.7")))
    min_severity_threshold: float = field(default_factory=lambda: float(os.getenv("ANOMALY_MIN_SEVERITY", "0.5")))


@dataclass
class MetricsConfig:
    """Metrics collection and aggregation configuration."""

    # Collection
    collection_interval_seconds: int = field(default_factory=lambda: int(os.getenv("METRICS_COLLECTION_INTERVAL", "60")))
    buffer_size: int = field(default_factory=lambda: int(os.getenv("METRICS_BUFFER_SIZE", "1000")))

    # Aggregation levels
    enable_minute_aggregation: bool = field(default_factory=lambda: os.getenv("METRICS_AGG_MINUTE", "true").lower() == "true")
    enable_hour_aggregation: bool = field(default_factory=lambda: os.getenv("METRICS_AGG_HOUR", "true").lower() == "true")
    enable_day_aggregation: bool = field(default_factory=lambda: os.getenv("METRICS_AGG_DAY", "true").lower() == "true")

    # Percentile calculation
    percentiles: list = field(default_factory=lambda: [50, 95, 99])

    # Storage
    retention_days: int = field(default_factory=lambda: int(os.getenv("METRICS_RETENTION_DAYS", "90")))


@dataclass
class AnalyticsConfig:
    """Main Analytics Engine configuration."""

    # Environment
    environment: EnvironmentType = field(
        default_factory=lambda: EnvironmentType(os.getenv("ENVIRONMENT", "development"))
    )
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    anomaly_detection: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))

    # Feature flags
    enable_cache: bool = field(default_factory=lambda: os.getenv("ANALYTICS_ENABLE_CACHE", "true").lower() == "true")
    enable_real_time_processing: bool = field(default_factory=lambda: os.getenv("ANALYTICS_ENABLE_REALTIME", "true").lower() == "true")
    enable_batch_processing: bool = field(default_factory=lambda: os.getenv("ANALYTICS_ENABLE_BATCH", "true").lower() == "true")
    enable_anomaly_detection: bool = field(default_factory=lambda: os.getenv("ANALYTICS_ENABLE_ANOMALY", "true").lower() == "true")

    # Performance monitoring
    enable_profiling: bool = field(default_factory=lambda: os.getenv("ANALYTICS_ENABLE_PROFILING", "false").lower() == "true")
    slow_query_threshold_ms: int = field(default_factory=lambda: int(os.getenv("ANALYTICS_SLOW_QUERY_MS", "1000")))

    def __post_init__(self):
        """Validate configuration after initialization."""
        logger.info(f"Analytics Engine Configuration: environment={self.environment.value}, debug={self.debug}")

        if self.environment == EnvironmentType.PRODUCTION:
            if self.debug:
                logger.warning("Debug mode enabled in production - consider disabling")
            logger.info("Production configuration: extra validation and monitoring enabled")

    @classmethod
    def from_env(cls) -> "AnalyticsConfig":
        """Create configuration from environment variables."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging/debugging."""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
            },
            "streaming": {
                "batch_size": self.streaming.event_batch_size,
                "window_size_seconds": self.streaming.window_size_seconds,
            },
            "anomaly_detection": {
                "method": self.anomaly_detection.default_method,
            },
            "cache_enabled": self.enable_cache,
            "realtime_enabled": self.enable_real_time_processing,
        }


# Global configuration instance
_config: Optional[AnalyticsConfig] = None


def get_config() -> AnalyticsConfig:
    """Get or create global analytics configuration."""
    global _config
    if _config is None:
        _config = AnalyticsConfig.from_env()
    return _config


def set_config(config: AnalyticsConfig) -> None:
    """Override global analytics configuration (for testing)."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration to default (for testing)."""
    global _config
    _config = None
