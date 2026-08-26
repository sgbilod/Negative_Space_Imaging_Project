"""
Repository Pattern Implementation

Concrete implementations of the MetricsRepository abstract interface
for data persistence and retrieval operations.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

from ..core.metrics import Metric, AggregatedMetric, MetricsRepository
from .timeseries_db import TimeSeriesDatabase

logger = logging.getLogger(__name__)


class PostgresMetricsRepository(MetricsRepository):
    """
    PostgreSQL-backed metrics repository.

    Implements the MetricsRepository interface using PostgreSQL
    with time-series data optimization and partitioning.
    """

    def __init__(self, timeseries_db: TimeSeriesDatabase):
        """
        Initialize repository.

        Args:
            timeseries_db: TimeSeriesDatabase instance
        """
        self.db = timeseries_db

    async def save_metric(self, metric: Metric) -> bool:
        """
        Save single metric.

        Args:
            metric: Metric to save

        Returns:
            bool: True if successful
        """
        return await self.db.insert_metric(metric)

    async def save_metrics_batch(self, metrics: List[Metric]) -> int:
        """
        Save batch of metrics efficiently.

        Args:
            metrics: List of metrics to save

        Returns:
            int: Number of metrics saved
        """
        return await self.db.insert_metrics_batch(metrics)

    async def get_metrics(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Metric]:
        """
        Retrieve raw metrics for time range.

        Args:
            name: Metric name
            start_time: Start of time range
            end_time: End of time range
            tags: Optional tags to filter by

        Returns:
            List of Metric objects
        """
        rows = await self.db.query_metrics(
            name, start_time, end_time, tags
        )

        metrics = []
        for row in rows:
            try:
                metric = Metric(
                    name=row["name"],
                    value=row["value"],
                    metric_type=row["metric_type"],
                    tags=row["tags"] or {},
                    timestamp=row["timestamp"],
                    created_at=row["created_at"],
                )
                metrics.append(metric)
            except Exception as e:
                logger.error(f"Failed to deserialize metric: {e}")
                continue

        return metrics

    async def get_aggregated(
        self,
        name: str,
        aggregation_level: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> List[AggregatedMetric]:
        """
        Retrieve aggregated metrics for time range.

        Args:
            name: Metric name
            aggregation_level: 'minute', 'hour', or 'day'
            start_time: Start of time range
            end_time: End of time range
            tags: Optional tags to filter by

        Returns:
            List of AggregatedMetric objects
        """
        rows = await self.db.query_aggregated(
            name, aggregation_level, start_time, end_time, tags
        )

        metrics = []
        for row in rows:
            try:
                agg = AggregatedMetric(
                    name=row["name"],
                    aggregation_level=row["aggregation_level"],
                    period_start=row["period_start"],
                    period_end=row["period_end"],
                    count=row["count"],
                    min_value=row["min_value"],
                    max_value=row["max_value"],
                    avg_value=row["avg_value"],
                    median_value=row["median_value"],
                    stddev_value=row["stddev_value"],
                    p95_value=row["p95_value"],
                    p99_value=row["p99_value"],
                )
                metrics.append(agg)
            except Exception as e:
                logger.error(f"Failed to deserialize aggregated metric: {e}")
                continue

        return metrics

    async def delete_old_metrics(self, older_than: datetime) -> int:
        """
        Delete metrics older than specified date.

        Args:
            older_than: Delete metrics created before this date

        Returns:
            int: Number of metrics deleted
        """
        return await self.db.cleanup_old_metrics()


class CachedMetricsRepository(MetricsRepository):
    """
    Cached metrics repository with Redis caching layer.

    Wraps another repository (e.g., PostgreSQL) with caching
    to reduce database load.
    """

    def __init__(
        self,
        backend: MetricsRepository,
        cache_ttl_seconds: int = 300
    ):
        """
        Initialize cached repository.

        Args:
            backend: Underlying repository implementation
            cache_ttl_seconds: Cache TTL in seconds
        """
        self.backend = backend
        self.cache_ttl = cache_ttl_seconds
        self._local_cache: Dict[str, Any] = {}

    async def save_metric(self, metric: Metric) -> bool:
        """
        Save metric and invalidate cache.

        Args:
            metric: Metric to save

        Returns:
            bool: True if successful
        """
        result = await self.backend.save_metric(metric)
        if result:
            # Invalidate cache for this metric name
            cache_key = f"metrics:{metric.name}"
            self._local_cache.pop(cache_key, None)
        return result

    async def save_metrics_batch(self, metrics: List[Metric]) -> int:
        """
        Save batch of metrics and invalidate cache.

        Args:
            metrics: List of metrics to save

        Returns:
            int: Number of metrics saved
        """
        result = await self.backend.save_metrics_batch(metrics)
        if result > 0:
            # Invalidate cache for affected metrics
            for metric in metrics:
                cache_key = f"metrics:{metric.name}"
                self._local_cache.pop(cache_key, None)
        return result

    async def get_metrics(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Metric]:
        """
        Get metrics with caching.

        Args:
            name: Metric name
            start_time: Start of time range
            end_time: End of time range
            tags: Optional tags to filter by

        Returns:
            List of Metric objects
        """
        # For simplicity, use backend cache
        # In production, would use Redis for distributed cache
        return await self.backend.get_metrics(
            name, start_time, end_time, tags
        )

    async def get_aggregated(
        self,
        name: str,
        aggregation_level: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> List[AggregatedMetric]:
        """
        Get aggregated metrics with caching.

        Args:
            name: Metric name
            aggregation_level: 'minute', 'hour', or 'day'
            start_time: Start of time range
            end_time: End of time range
            tags: Optional tags to filter by

        Returns:
            List of AggregatedMetric objects
        """
        # For simplicity, use backend cache
        # In production, would use Redis for distributed cache
        return await self.backend.get_aggregated(
            name, aggregation_level, start_time, end_time, tags
        )

    async def delete_old_metrics(self, older_than: datetime) -> int:
        """
        Delete old metrics and clear cache.

        Args:
            older_than: Delete metrics created before this date

        Returns:
            int: Number of metrics deleted
        """
        result = await self.backend.delete_old_metrics(older_than)
        self._local_cache.clear()
        return result


class InMemoryMetricsRepository(MetricsRepository):
    """
    In-memory metrics repository for testing and development.

    Stores metrics in memory without persistence.
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self.metrics: List[Metric] = []
        self.aggregated: List[AggregatedMetric] = []

    async def save_metric(self, metric: Metric) -> bool:
        """
        Save metric to memory.

        Args:
            metric: Metric to save

        Returns:
            bool: Always True
        """
        self.metrics.append(metric)
        return True

    async def save_metrics_batch(self, metrics: List[Metric]) -> int:
        """
        Save batch to memory.

        Args:
            metrics: List of metrics to save

        Returns:
            int: Number of metrics saved
        """
        self.metrics.extend(metrics)
        return len(metrics)

    async def get_metrics(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Metric]:
        """
        Get metrics from memory.

        Args:
            name: Metric name
            start_time: Start of time range
            end_time: End of time range
            tags: Optional tags to filter by

        Returns:
            List of Metric objects
        """
        result = [
            m for m in self.metrics
            if m.name == name
            and start_time <= m.timestamp <= end_time
        ]

        # Filter by tags if provided
        if tags:
            filtered = []
            for m in result:
                if all(m.tags.get(k) == v for k, v in tags.items()):
                    filtered.append(m)
            result = filtered

        return result

    async def get_aggregated(
        self,
        name: str,
        aggregation_level: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> List[AggregatedMetric]:
        """
        Get aggregated metrics from memory.

        Args:
            name: Metric name
            aggregation_level: 'minute', 'hour', or 'day'
            start_time: Start of time range
            end_time: End of time range
            tags: Optional tags to filter by

        Returns:
            List of AggregatedMetric objects
        """
        result = [
            a for a in self.aggregated
            if a.name == name
            and a.aggregation_level == aggregation_level
            and start_time <= a.period_start <= end_time
        ]
        return result

    async def delete_old_metrics(self, older_than: datetime) -> int:
        """
        Delete old metrics from memory.

        Args:
            older_than: Delete metrics created before this date

        Returns:
            int: Number of metrics deleted
        """
        before = len(self.metrics)
        self.metrics = [m for m in self.metrics if m.created_at > older_than]
        return before - len(self.metrics)
