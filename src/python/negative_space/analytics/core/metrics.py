"""
Metrics Collection and Management

Provides metrics collection, aggregation, and storage operations.
Handles real-time metrics ingestion and pre-aggregated storage.

Design Principles:
- Type-safe metric definitions
- Flexible tagging system for dimensions
- Asynchronous batch writes
- Automatic cleanup of old data
- Redis caching for performance
"""

import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Supported metric types."""
    GAUGE = "gauge"  # Point-in-time value
    COUNTER = "counter"  # Monotonically increasing value
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Percentile aggregation


@dataclass
class Metric:
    """
    Represents a single metric data point.

    Attributes:
        name: Metric name (e.g., 'event_rate', 'error_rate')
        value: Numeric value
        metric_type: Type of metric (gauge, counter, histogram, summary)
        tags: Dictionary of tags for filtering/grouping
        timestamp: When the metric was collected
        created_at: When the metric was stored
    """

    name: str
    value: float
    metric_type: str = MetricType.GAUGE.value
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "metric_type": self.metric_type,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
            "created_at": self.created_at.isoformat(),
        }

    def tag_string(self) -> str:
        """Get tags as sortable string for grouping."""
        if not self.tags:
            return ""
        items = sorted(self.tags.items())
        return ",".join(f"{k}={v}" for k, v in items)


@dataclass
class AggregatedMetric:
    """
    Pre-aggregated metric with statistics.

    Stores computed statistics to avoid recalculating on each query.
    """

    name: str
    aggregation_level: str  # 'minute', 'hour', 'day'
    period_start: datetime
    period_end: datetime
    count: int = 0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    avg_value: Optional[float] = None
    median_value: Optional[float] = None
    stddev_value: Optional[float] = None
    p95_value: Optional[float] = None
    p99_value: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "aggregation_level": self.aggregation_level,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "count": self.count,
            "min": self.min_value,
            "max": self.max_value,
            "avg": self.avg_value,
            "median": self.median_value,
            "stddev": self.stddev_value,
            "p95": self.p95_value,
            "p99": self.p99_value,
            "tags": self.tags,
        }


class MetricsRepository(ABC):
    """Abstract base class for metrics storage."""

    @abstractmethod
    async def save_metric(self, metric: Metric) -> bool:
        """Save a single metric."""
        pass

    @abstractmethod
    async def save_metrics_batch(self, metrics: List[Metric]) -> int:
        """Save multiple metrics. Returns count saved."""
        pass

    @abstractmethod
    async def get_metrics(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Metric]:
        """Retrieve metrics for a name and time range."""
        pass

    @abstractmethod
    async def get_aggregated(
        self,
        name: str,
        aggregation_level: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> List[AggregatedMetric]:
        """Retrieve aggregated metrics."""
        pass

    @abstractmethod
    async def delete_old_metrics(self, older_than: datetime) -> int:
        """Delete metrics older than specified time. Returns count deleted."""
        pass


class MetricsCollector:
    """
    Collects metrics from various sources and batches them for storage.

    Features:
    - Buffering of metrics
    - Automatic batch flush
    - Time-based aggregation
    - Tags support for multi-dimensional analysis
    """

    def __init__(
        self,
        repository: MetricsRepository,
        batch_size: int = 100,
        batch_timeout_seconds: float = 5.0
    ):
        """
        Initialize metrics collector.

        Args:
            repository: Metrics storage repository
            batch_size: Number of metrics to accumulate before flush
            batch_timeout_seconds: Maximum time to wait before flush
        """
        self.repository = repository
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds

        self._buffer: List[Metric] = []
        self._lock = asyncio.Lock()
        self._last_flush = datetime.utcnow()

        self._metrics_collected = 0
        self._batches_flushed = 0
        self._flush_errors = 0

        logger.info(
            f"MetricsCollector initialized: batch_size={batch_size}, "
            f"timeout={batch_timeout_seconds}s"
        )

    async def collect(self, metric: Metric) -> None:
        """
        Collect a single metric.

        Args:
            metric: Metric to collect
        """
        async with self._lock:
            self._buffer.append(metric)
            self._metrics_collected += 1

            # Check if we should flush
            if len(self._buffer) >= self.batch_size:
                await self._flush_buffer()

    async def collect_batch(self, metrics: List[Metric]) -> None:
        """
        Collect multiple metrics.

        Args:
            metrics: List of metrics to collect
        """
        async with self._lock:
            self._buffer.extend(metrics)
            self._metrics_collected += len(metrics)

            # Check if we should flush
            while len(self._buffer) >= self.batch_size:
                await self._flush_buffer()

    async def flush(self) -> int:
        """
        Flush all buffered metrics.

        Returns:
            Number of metrics flushed
        """
        async with self._lock:
            return await self._flush_buffer()

    async def _flush_buffer(self) -> int:
        """Internal method to flush buffer (must be called within lock)."""
        if not self._buffer:
            return 0

        try:
            metrics_to_flush = self._buffer[:self.batch_size]
            count = await self.repository.save_metrics_batch(metrics_to_flush)
            self._buffer = self._buffer[self.batch_size:]
            self._batches_flushed += 1
            self._last_flush = datetime.utcnow()
            logger.debug(f"Flushed {count} metrics to storage")
            return count
        except Exception as e:
            self._flush_errors += 1
            logger.error(f"Error flushing metrics: {e}")
            return 0

    async def run(self) -> None:
        """
        Main collection loop with periodic flushing.
        """
        logger.info("MetricsCollector started")
        try:
            while True:
                await asyncio.sleep(self.batch_timeout_seconds)
                async with self._lock:
                    if self._buffer:
                        await self._flush_buffer()
        except asyncio.CancelledError:
            logger.info("MetricsCollector cancelled")
            # Final flush before shutdown
            await self.flush()
            raise

    def get_stats(self) -> Dict[str, int]:
        """Get collection statistics."""
        return {
            "metrics_collected": self._metrics_collected,
            "batches_flushed": self._batches_flushed,
            "flush_errors": self._flush_errors,
            "buffered_metrics": len(self._buffer),
        }


class MetricsAggregator:
    """
    Aggregates raw metrics into time-bucketed summaries.

    Computes: count, min, max, avg, median, stddev, percentiles
    """

    def __init__(self, repository: MetricsRepository):
        """
        Initialize aggregator.

        Args:
            repository: Metrics storage repository
        """
        self.repository = repository
        self._aggregations_created = 0
        self._aggregation_errors = 0

    async def aggregate_period(
        self,
        name: str,
        aggregation_level: str,
        period_start: datetime,
        period_end: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[AggregatedMetric]:
        """
        Compute aggregation for a time period.

        Args:
            name: Metric name
            aggregation_level: 'minute', 'hour', or 'day'
            period_start: Start of period
            period_end: End of period
            tags: Optional tags to filter metrics

        Returns:
            AggregatedMetric with computed statistics
        """
        try:
            # Get raw metrics for period
            metrics = await self.repository.get_metrics(
                name,
                period_start,
                period_end,
                tags
            )

            if not metrics:
                return None

            # Extract values
            values = [m.value for m in metrics]
            values.sort()

            # Compute statistics
            count = len(values)
            min_val = min(values)
            max_val = max(values)
            avg_val = sum(values) / count
            median_val = self._compute_percentile(values, 50)
            stddev_val = self._compute_stddev(values, avg_val)
            p95_val = self._compute_percentile(values, 95)
            p99_val = self._compute_percentile(values, 99)

            agg = AggregatedMetric(
                name=name,
                aggregation_level=aggregation_level,
                period_start=period_start,
                period_end=period_end,
                count=count,
                min_value=min_val,
                max_value=max_val,
                avg_value=avg_val,
                median_value=median_val,
                stddev_value=stddev_val,
                p95_value=p95_val,
                p99_value=p99_val,
                tags=tags or {}
            )

            self._aggregations_created += 1
            return agg

        except Exception as e:
            self._aggregation_errors += 1
            logger.error(f"Error aggregating metrics: {e}")
            return None

    @staticmethod
    def _compute_percentile(sorted_values: List[float], percentile: int) -> float:
        """Compute percentile of sorted values."""
        index = int((percentile / 100.0) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

    @staticmethod
    def _compute_stddev(values: List[float], mean: float) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def get_stats(self) -> Dict[str, int]:
        """Get aggregation statistics."""
        return {
            "aggregations_created": self._aggregations_created,
            "aggregation_errors": self._aggregation_errors,
        }
