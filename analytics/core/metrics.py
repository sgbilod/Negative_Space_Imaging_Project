"""
Metrics Collection System

Collects, aggregates, and reports metrics from analytics engine.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "tags": self.tags,
        }


@dataclass
class MetricAggregate:
    """Aggregated metric data."""

    metric_name: str
    timestamp: datetime
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    std_dev: float
    p95: float
    p99: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "timestamp": self.timestamp.isoformat(),
            "count": self.count,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "std_dev": self.std_dev,
            "p95": self.p95,
            "p99": self.p99,
        }


class MetricsCollector:
    """
    Metrics collection and aggregation.

    Features:
    - Collect numeric metrics with tags
    - Aggregate metrics over time windows
    - Calculate percentiles and statistics
    - Export metrics in standard formats
    - Time series data retention

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record_metric("latency", 42.5, {"service": "analytics"})
        >>> agg = collector.get_aggregate("processing_latency", "1h")
    """

    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector.

        Args:
            retention_hours: How long to keep metrics (hours)
        """
        self.retention_hours = retention_hours

        # Storage: metric_name -> list of MetricPoint
        self._metrics: Dict[str, List[MetricPoint]] = defaultdict(list)

        # Aggregates cache
        self._aggregates_cache: Dict[str, MetricAggregate] = {}

        # Counters for quick access
        self._counters: Dict[str, int] = defaultdict(int)

        # Gauges for current values
        self._gauges: Dict[str, float] = {}

        logger.info(
            f"Metrics collector initialized (retention: {retention_hours}h)"
        )

    def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Numeric value
            tags: Optional tags for filtering
            timestamp: Optional timestamp (defaults to now)
        """
        try:
            timestamp = timestamp or datetime.utcnow()

            point = MetricPoint(
                timestamp=timestamp,
                value=value,
                tags=tags or {}
            )

            self._metrics[metric_name].append(point)

            # Cleanup old data
            self._cleanup_old_data(metric_name)

        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {e}")

    def increment_counter(
        self,
        counter_name: str,
        increment: int = 1
    ) -> None:
        """
        Increment a counter.

        Args:
            counter_name: Name of the counter
            increment: Amount to increment
        """
        self._counters[counter_name] += increment

    def set_gauge(
        self,
        gauge_name: str,
        value: float
    ) -> None:
        """
        Set gauge value.

        Args:
            gauge_name: Name of the gauge
            value: Value to set
        """
        self._gauges[gauge_name] = value

    def get_aggregate(
        self,
        metric_name: str,
        time_window: str = "1h",
        tags_filter: Optional[Dict[str, str]] = None
    ) -> Optional[MetricAggregate]:
        """
        Get aggregated statistics for a metric.

        Args:
            metric_name: Name of metric
            time_window: Time window ("1h", "24h", "7d", etc.)
            tags_filter: Optional tag filtering

        Returns:
            MetricAggregate or None if no data
        """
        try:
            # Get time window in seconds
            window_seconds = self._parse_time_window(time_window)
            cutoff_time = datetime.utcnow() - timedelta(seconds=window_seconds)

            # Filter metrics
            points = self._metrics.get(metric_name, [])
            filtered_points = [
                p for p in points
                if (
                    p.timestamp >= cutoff_time and
                    self._matches_tags(p, tags_filter)
                )
            ]

            if not filtered_points:
                return None

            # Extract values
            values = [p.value for p in filtered_points]

            # Calculate statistics
            aggregate = MetricAggregate(
                metric_name=metric_name,
                timestamp=datetime.utcnow(),
                count=len(values),
                sum=sum(values),
                min=min(values),
                max=max(values),
                mean=statistics.mean(values),
                median=statistics.median(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
                p95=self._percentile(values, 0.95),
                p99=self._percentile(values, 0.99),
            )

            return aggregate

        except Exception as e:
            logger.error(f"Error calculating aggregate for {metric_name}: {e}")
            return None

    def get_counter(self, counter_name: str) -> int:
        """Get counter value."""
        return self._counters.get(counter_name, 0)

    def get_gauge(self, gauge_name: str) -> Optional[float]:
        """Get gauge value."""
        return self._gauges.get(gauge_name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        return {
            "metrics": {
                name: [p.to_dict() for p in points[-100:]]  # Last 100 points
                for name, points in self._metrics.items()
            },
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Export gauges
        for gauge_name, value in self._gauges.items():
            lines.append(f"{gauge_name} {value}")

        # Export counters
        for counter_name, value in self._counters.items():
            lines.append(f"{counter_name}_total {value}")

        # Export last value of each metric
        for metric_name, points in self._metrics.items():
            if points:
                last_point = points[-1]
                lines.append(f"{metric_name} {last_point.value}")

        return "\n".join(lines) + "\n"

    def reset_metrics(self, metric_name: str) -> None:
        """Reset specific metric."""
        if metric_name in self._metrics:
            self._metrics[metric_name].clear()
            logger.info(f"Reset metric: {metric_name}")

    def reset_all(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._counters.clear()
        self._gauges.clear()
        logger.info("Reset all metrics")

    def _cleanup_old_data(self, metric_name: str) -> None:
        """Remove data older than retention period."""
        if metric_name not in self._metrics:
            return

        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        original_len = len(self._metrics[metric_name])

        self._metrics[metric_name] = [
            p for p in self._metrics[metric_name]
            if p.timestamp >= cutoff_time
        ]

        removed = original_len - len(self._metrics[metric_name])
        if removed > 0:
            logger.debug(
                f"Cleaned up {removed} old data points for {metric_name}"
            )

    def _matches_tags(
        self,
        point: MetricPoint,
        tags_filter: Optional[Dict[str, str]]
    ) -> bool:
        """Check if point matches tag filter."""
        if not tags_filter:
            return True

        for key, value in tags_filter.items():
            if point.tags.get(key) != value:
                return False

        return True

    @staticmethod
    def _parse_time_window(window_str: str) -> int:
        """Parse time window string to seconds."""
        window_str = window_str.strip().lower()

        if window_str.endswith("h"):
            return int(window_str[:-1]) * 3600
        elif window_str.endswith("d"):
            return int(window_str[:-1]) * 86400
        elif window_str.endswith("m"):
            return int(window_str[:-1]) * 60
        elif window_str.endswith("w"):
            return int(window_str[:-1]) * 604800
        else:
            # Default to 1 hour
            return 3600

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
