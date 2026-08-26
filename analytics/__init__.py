"""
Advanced Analytics Engine for Negative Space Imaging Project

High-performance analytics with:
- Real-time streaming data processing
- Statistical analysis and hypothesis testing
- Anomaly detection (multi-algorithm)
- Time series data storage and querying
- Interactive dashboards and reporting
"""

__version__ = "1.0.0"
__author__ = "Stephen Bilodeau"

from .core.base import AnalyticsEngine
from .core.events import EventSystem
from .core.metrics import MetricsCollector
from .processors.streaming import StreamingProcessor
from .processors.batch import BatchProcessor
from .algorithms.statistical import StatisticalAnalyzer
from .algorithms.anomaly_detection import AnomalyDetector
from .storage.timeseries_db import TimeSeriesDatabase

__all__ = [
    "AnalyticsEngine",
    "EventSystem",
    "MetricsCollector",
    "StreamingProcessor",
    "BatchProcessor",
    "StatisticalAnalyzer",
    "AnomalyDetector",
    "TimeSeriesDatabase",
]
