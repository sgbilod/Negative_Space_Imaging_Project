"""
Processors for analytics data transformation.

Includes streaming processors, window operations, and real-time
aggregation capabilities for metrics and events.
"""

from .streaming import (
    StreamProcessor,
    StreamElement,
    WindowType,
    WindowConfig,
    MetricsStreamAggregator,
)

__all__ = [
    "StreamProcessor",
    "StreamElement",
    "WindowType",
    "WindowConfig",
    "MetricsStreamAggregator",
]
