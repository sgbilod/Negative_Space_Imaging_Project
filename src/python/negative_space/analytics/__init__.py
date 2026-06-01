"""
Analytics Package Initialization

Top-level module for the analytics engine.

Exports:
- Configuration management
- Core event system
- Metrics collection and storage
- Streaming processors
- Statistical algorithms
- Anomaly detection
"""

# Core modules
from . import core
from . import storage
from . import processors
from . import algorithms

# Configuration
from .config import (
    AnalyticsConfig,
    EnvironmentType,
    AnomalyDetectionConfig,
    MetricsConfig,
    DatabaseConfig,
    RedisConfig,
)

# Core classes
from .core import (
    Event,
    MetricEvent,
    AnomalyEvent,
    EventBus,
    EventSubscriber,
    Metric,
    MetricType,
    AggregatedMetric,
    MetricsCollector,
    MetricsAggregator,
)

# Storage
from .storage import (
    TimeSeriesDatabase,
    PostgresMetricsRepository,
    CachedMetricsRepository,
    InMemoryMetricsRepository,
)

# Processors
from .processors import (
    StreamProcessor,
    StreamElement,
    WindowType,
    WindowConfig,
    MetricsStreamAggregator,
)

# Algorithms
from .algorithms import (
    StatisticalAnalyzer,
    CorrelationAnalyzer,
    CorrelationResult,
    TrendResult,
    AnomalyDetector,
    AnomalyScore,
)

__version__ = "0.1.0"

__all__ = [
    # Modules
    "core",
    "storage",
    "processors",
    "algorithms",

    # Configuration
    "AnalyticsConfig",
    "EnvironmentType",
    "AnomalyDetectionConfig",
    "MetricsConfig",
    "DatabaseConfig",
    "RedisConfig",

    # Core
    "Event",
    "MetricEvent",
    "AnomalyEvent",
    "EventBus",
    "EventSubscriber",
    "Metric",
    "MetricType",
    "AggregatedMetric",
    "MetricsCollector",
    "MetricsAggregator",

    # Storage
    "TimeSeriesDatabase",
    "PostgresMetricsRepository",
    "CachedMetricsRepository",
    "InMemoryMetricsRepository",

    # Processors
    "StreamProcessor",
    "StreamElement",
    "WindowType",
    "WindowConfig",
    "MetricsStreamAggregator",

    # Algorithms
    "StatisticalAnalyzer",
    "CorrelationAnalyzer",
    "CorrelationResult",
    "TrendResult",
    "AnomalyDetector",
    "AnomalyScore",
]
