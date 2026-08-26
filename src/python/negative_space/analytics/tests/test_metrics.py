"""
Unit Tests for Analytics Metrics System

Tests for:
- Metric creation and data types
- Metrics collection and buffering
- Metrics aggregation and statistics
- Repository interface
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from ..core.metrics import (
    Metric,
    MetricType,
    AggregatedMetric,
    MetricsRepository,
    MetricsCollector,
    MetricsAggregator,
)


class TestMetric:
    """Test suite for Metric dataclass."""

    def test_metric_creation(self):
        """Test basic metric creation."""
        metric = Metric(
            name="event_rate",
            value=42.5,
            metric_type=MetricType.GAUGE.value,
            tags={"service": "api"},
        )

        assert metric.name == "event_rate"
        assert metric.value == 42.5
        assert metric.metric_type == MetricType.GAUGE.value
        assert metric.tags == {"service": "api"}
        assert metric.timestamp is not None
        assert metric.created_at is not None

    def test_metric_to_dict(self):
        """Test converting metric to dictionary."""
        metric = Metric(
            name="event_rate",
            value=42.5,
            tags={"service": "api"},
        )

        d = metric.to_dict()
        assert d["name"] == "event_rate"
        assert d["value"] == 42.5
        assert d["tags"] == {"service": "api"}
        assert "timestamp" in d
        assert "created_at" in d

    def test_metric_tag_string(self):
        """Test generating sorted tag string."""
        metric = Metric(
            name="test",
            value=1.0,
            tags={"z_tag": "last", "a_tag": "first", "m_tag": "middle"}
        )

        tag_str = metric.tag_string()
        assert tag_str == "a_tag=first,m_tag=middle,z_tag=last"

    def test_metric_tag_string_empty_tags(self):
        """Test tag string with no tags."""
        metric = Metric(name="test", value=1.0)
        assert metric.tag_string() == ""

    def test_metric_default_metric_type(self):
        """Test default metric type is gauge."""
        metric = Metric(name="test", value=1.0)
        assert metric.metric_type == MetricType.GAUGE.value


class TestAggregatedMetric:
    """Test suite for AggregatedMetric."""

    def test_aggregated_metric_creation(self):
        """Test creating aggregated metric."""
        start = datetime.utcnow()
        end = start + timedelta(hours=1)

        agg = AggregatedMetric(
            name="event_rate",
            aggregation_level="hour",
            period_start=start,
            period_end=end,
            count=100,
            min_value=10.0,
            max_value=50.0,
            avg_value=30.0,
            p95_value=45.0,
            p99_value=49.0,
        )

        assert agg.name == "event_rate"
        assert agg.aggregation_level == "hour"
        assert agg.count == 100
        assert agg.avg_value == 30.0

    def test_aggregated_metric_to_dict(self):
        """Test converting aggregated metric to dictionary."""
        start = datetime.utcnow()
        end = start + timedelta(hours=1)

        agg = AggregatedMetric(
            name="test",
            aggregation_level="hour",
            period_start=start,
            period_end=end,
            count=100,
            avg_value=25.0,
        )

        d = agg.to_dict()
        assert d["name"] == "test"
        assert d["aggregation_level"] == "hour"
        assert d["count"] == 100
        assert d["avg"] == 25.0


class MockMetricsRepository(MetricsRepository):
    """Mock repository for testing."""

    def __init__(self):
        self.metrics: List[Metric] = []
        self.aggregated: List[AggregatedMetric] = []

    async def save_metric(self, metric: Metric) -> bool:
        self.metrics.append(metric)
        return True

    async def save_metrics_batch(self, metrics: List[Metric]) -> int:
        self.metrics.extend(metrics)
        return len(metrics)

    async def get_metrics(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Metric]:
        return [m for m in self.metrics if m.name == name]

    async def get_aggregated(
        self,
        name: str,
        aggregation_level: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> List[AggregatedMetric]:
        return [a for a in self.aggregated if a.name == name]

    async def delete_old_metrics(self, older_than: datetime) -> int:
        before = len(self.metrics)
        self.metrics = [m for m in self.metrics if m.created_at > older_than]
        return before - len(self.metrics)


class TestMetricsCollector:
    """Test suite for MetricsCollector."""

    @pytest.mark.asyncio
    async def test_collector_creation(self):
        """Test MetricsCollector initialization."""
        repo = MockMetricsRepository()
        collector = MetricsCollector(repo, batch_size=10, batch_timeout_seconds=1.0)

        assert collector.batch_size == 10
        assert collector.batch_timeout_seconds == 1.0

    @pytest.mark.asyncio
    async def test_collect_single_metric(self):
        """Test collecting a single metric."""
        repo = MockMetricsRepository()
        collector = MetricsCollector(repo, batch_size=100)

        metric = Metric(name="test", value=1.0)
        await collector.collect(metric)

        assert len(collector._buffer) == 1
        assert collector._metrics_collected == 1

    @pytest.mark.asyncio
    async def test_collect_triggers_flush(self):
        """Test that collecting enough metrics triggers flush."""
        repo = MockMetricsRepository()
        collector = MetricsCollector(repo, batch_size=3)

        # Collect 3 metrics (should trigger flush at 3rd)
        for i in range(3):
            metric = Metric(name=f"test_{i}", value=float(i))
            await collector.collect(metric)

        # Should have flushed on 3rd metric
        assert len(collector._buffer) == 0
        assert len(repo.metrics) == 3
        assert collector._batches_flushed == 1

    @pytest.mark.asyncio
    async def test_collect_batch(self):
        """Test collecting multiple metrics at once."""
        repo = MockMetricsRepository()
        collector = MetricsCollector(repo, batch_size=100)

        metrics = [
            Metric(name=f"test_{i}", value=float(i))
            for i in range(5)
        ]
        await collector.collect_batch(metrics)

        assert len(collector._buffer) == 5
        assert collector._metrics_collected == 5

    @pytest.mark.asyncio
    async def test_flush_buffer(self):
        """Test explicit flush."""
        repo = MockMetricsRepository()
        collector = MetricsCollector(repo, batch_size=100)

        # Add metrics
        for i in range(5):
            metric = Metric(name=f"test_{i}", value=float(i))
            collector._buffer.append(metric)

        # Flush
        count = await collector.flush()

        assert count == 5
        assert len(collector._buffer) == 0
        assert len(repo.metrics) == 5
        assert collector._batches_flushed == 1

    def test_collector_stats(self):
        """Test getting collector statistics."""
        repo = MockMetricsRepository()
        collector = MetricsCollector(repo)

        stats = collector.get_stats()
        assert stats["metrics_collected"] == 0
        assert stats["batches_flushed"] == 0
        assert stats["flush_errors"] == 0
        assert stats["buffered_metrics"] == 0


class TestMetricsAggregator:
    """Test suite for MetricsAggregator."""

    @pytest.mark.asyncio
    async def test_aggregator_creation(self):
        """Test MetricsAggregator initialization."""
        repo = MockMetricsRepository()
        agg = MetricsAggregator(repo)
        assert agg.repository == repo

    @pytest.mark.asyncio
    async def test_aggregate_period_with_metrics(self):
        """Test aggregating metrics for a period."""
        repo = MockMetricsRepository()

        # Add some test metrics
        now = datetime.utcnow()
        repo.metrics = [
            Metric(name="test", value=10.0, timestamp=now - timedelta(minutes=5)),
            Metric(name="test", value=20.0, timestamp=now - timedelta(minutes=3)),
            Metric(name="test", value=30.0, timestamp=now - timedelta(minutes=1)),
        ]

        agg = MetricsAggregator(repo)
        result = await agg.aggregate_period(
            "test",
            "hour",
            now - timedelta(hours=1),
            now,
        )

        assert result is not None
        assert result.count == 3
        assert result.min_value == 10.0
        assert result.max_value == 30.0
        assert result.avg_value == 20.0
        assert result.median_value == 20.0

    @pytest.mark.asyncio
    async def test_aggregate_period_no_metrics(self):
        """Test aggregating when no metrics found."""
        repo = MockMetricsRepository()
        agg = MetricsAggregator(repo)

        result = await agg.aggregate_period(
            "nonexistent",
            "hour",
            datetime.utcnow() - timedelta(hours=1),
            datetime.utcnow(),
        )

        assert result is None

    def test_percentile_calculation(self):
        """Test percentile computation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        p50 = MetricsAggregator._compute_percentile(values, 50)
        assert p50 >= 5  # Median around middle

        p95 = MetricsAggregator._compute_percentile(values, 95)
        assert p95 > p50

        p99 = MetricsAggregator._compute_percentile(values, 99)
        assert p99 >= p95

    def test_stddev_calculation(self):
        """Test standard deviation computation."""
        # Simple case: 0, 1, 2, 3, 4 with mean 2
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        mean = 2.0

        stddev = MetricsAggregator._compute_stddev(values, mean)
        assert stddev > 0
        assert abs(stddev - 1.414) < 0.1  # Sqrt(2) ≈ 1.414

    def test_stddev_single_value(self):
        """Test stddev with single value."""
        stddev = MetricsAggregator._compute_stddev([1.0], 1.0)
        assert stddev == 0.0

    def test_aggregator_stats(self):
        """Test aggregator statistics."""
        repo = MockMetricsRepository()
        agg = MetricsAggregator(repo)

        stats = agg.get_stats()
        assert stats["aggregations_created"] == 0
        assert stats["aggregation_errors"] == 0


class TestMetricsCollectorIntegration:
    """Integration tests for MetricsCollector."""

    @pytest.mark.asyncio
    async def test_collection_and_storage_flow(self):
        """Test full flow: collect -> buffer -> flush -> storage."""
        repo = MockMetricsRepository()
        collector = MetricsCollector(repo, batch_size=2)

        # Collect metrics (should flush at batch_size)
        for i in range(5):
            metric = Metric(name=f"metric_{i}", value=float(i))
            await collector.collect(metric)

        # All metrics should be stored or in buffer
        total = len(repo.metrics) + len(collector._buffer)
        assert total == 5

    @pytest.mark.asyncio
    async def test_metrics_and_aggregation_flow(self):
        """Test metrics collection followed by aggregation."""
        repo = MockMetricsRepository()

        # Add metrics
        now = datetime.utcnow()
        for i in range(10):
            metric = Metric(
                name="response_time",
                value=float(100 + i * 10),
                timestamp=now - timedelta(minutes=5-i)
            )
            await repo.save_metric(metric)

        # Aggregate
        agg = MetricsAggregator(repo)
        result = await agg.aggregate_period(
            "response_time",
            "hour",
            now - timedelta(hours=1),
            now,
        )

        assert result is not None
        assert result.count == 10
        assert result.avg_value > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
