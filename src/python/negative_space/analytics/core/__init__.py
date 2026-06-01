"""
Analytics Core Module

Core components for the analytics engine:
- Event system (pub/sub)
- Metrics collection and aggregation
- Data repository interfaces
"""

from .events import (
    Event,
    EventType,
    MetricEvent,
    AnomalyEvent,
    EventSubscriber,
    EventBus,
    get_event_bus,
    set_event_bus,
    reset_event_bus,
)
from .metrics import (
    Metric,
    MetricType,
    AggregatedMetric,
    MetricsRepository,
    MetricsCollector,
    MetricsAggregator,
)

__all__ = [
    # Events
    "Event",
    "EventType",
    "MetricEvent",
    "AnomalyEvent",
    "EventSubscriber",
    "EventBus",
    "get_event_bus",
    "set_event_bus",
    "reset_event_bus",
    # Metrics
    "Metric",
    "MetricType",
    "AggregatedMetric",
    "MetricsRepository",
    "MetricsCollector",
    "MetricsAggregator",
]
