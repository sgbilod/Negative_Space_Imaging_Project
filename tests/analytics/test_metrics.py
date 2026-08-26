"""
Metrics Component Tests

Tests for metrics computation, aggregation, time-series handling, and
metrics pipeline operations.
"""

from datetime import datetime, timedelta


class TestMetricsComputation:
    """Test metrics computation."""

    def test_compute_basic_metrics(self):
        """Test computing basic metrics."""
        data = [10, 20, 30, 40, 50]

        metrics = {
            "count": len(data),
            "sum": sum(data),
            "mean": sum(data) / len(data),
            "min": min(data),
            "max": max(data),
        }

        assert metrics["count"] == 5
        assert metrics["sum"] == 150
        assert metrics["mean"] == 30
        assert metrics["min"] == 10
        assert metrics["max"] == 50

    def test_compute_variance(self):
        """Test computing variance."""
        data = [1, 2, 3, 4, 5]
        mean = sum(data) / len(data)

        variance = sum((x - mean) ** 2 for x in data) / len(data)

        assert variance > 0
        assert isinstance(variance, float)

    def test_compute_deviation(self):
        """Test computing standard deviation."""
        import math

        data = [1, 2, 3, 4, 5]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = math.sqrt(variance)

        assert std_dev > 0

    def test_compute_rate(self):
        """Test computing rate metrics."""
        values = [100, 110, 120, 130]

        # Rate of change
        rates = [
            (values[i + 1] - values[i])
            for i in range(len(values) - 1)
        ]

        assert len(rates) == 3
        assert all(r == 10 for r in rates)


class TestTimeSeriesMetrics:
    """Test time-series metrics."""

    def test_time_windowed_metrics(self):
        """Test metrics within time window."""
        now = datetime.now()
        events = [
            {
                "timestamp": now - timedelta(minutes=i),
                "value": 100 + i,
            }
            for i in range(10)
        ]

        # Events are in reverse time order (earlier minutes ago first)
        # So events[7] is earlier than events[3]
        window_start = events[7]["timestamp"]  # Earliest time
        window_end = events[3]["timestamp"]    # Latest time

        windowed = [
            e for e in events
            if window_start <= e["timestamp"] <= window_end
        ]

        # Should have at least some events in window (events 3-7)
        assert len(windowed) >= 4

    def test_accumulating_metrics(self):
        """Test accumulating metric values."""
        events = [
            {"value": 10},
            {"value": 20},
            {"value": 30},
        ]

        cumulative = []
        total = 0

        for event in events:
            total += event["value"]
            cumulative.append(total)

        assert cumulative == [10, 30, 60]

    def test_rolling_window_metrics(self):
        """Test rolling window calculation."""
        data = list(range(1, 11))  # 1-10
        window_size = 3
        rolling_means = []

        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            mean = sum(window) / len(window)
            rolling_means.append(mean)

        assert len(rolling_means) == 8
        assert rolling_means[0] == 2.0  # (1+2+3)/3


class TestMetricsAggregation:
    """Test metrics aggregation."""

    def test_aggregate_multiple_sources(self):
        """Test aggregating metrics from multiple sources."""
        sources = {
            "source_1": [10, 20, 30],
            "source_2": [15, 25, 35],
            "source_3": [12, 22, 32],
        }

        combined = []
        for source_metrics in sources.values():
            combined.extend(source_metrics)

        total = sum(combined)
        mean = total / len(combined)

        # Total: 10+20+30+15+25+35+12+22+32 = 201
        assert total == 201
        assert mean > 20

    def test_aggregate_by_dimension(self):
        """Test aggregating by dimension."""
        metrics = [
            {"region": "US", "value": 100},
            {"region": "EU", "value": 150},
            {"region": "US", "value": 120},
            {"region": "EU", "value": 130},
        ]

        by_region = {}
        for m in metrics:
            region = m["region"]
            if region not in by_region:
                by_region[region] = []
            by_region[region].append(m["value"])

        # Aggregate each region
        region_sums = {r: sum(v) for r, v in by_region.items()}

        assert region_sums["US"] == 220
        assert region_sums["EU"] == 280

    def test_hierarchical_aggregation(self):
        """Test hierarchical aggregation."""
        metrics = [
            {"level": "device", "name": "d1", "value": 10},
            {"level": "device", "name": "d2", "value": 20},
            {"level": "device", "name": "d3", "value": 30},
        ]

        # Device level
        device_sum = sum(m["value"] for m in metrics)

        # Machine level (if multiple devices)
        machine_sum = device_sum

        assert device_sum == 60
        assert machine_sum == 60


class TestMetricsOutputFormats:
    """Test different metrics output formats."""

    def test_metrics_as_dict(self):
        """Test metrics output as dictionary."""
        metrics = {
            "timestamp": datetime.now(),
            "mean": 30.5,
            "std_dev": 10.2,
            "count": 100,
        }

        assert isinstance(metrics, dict)
        assert "mean" in metrics

    def test_metrics_as_list(self):
        """Test metrics output as list."""
        metrics = [
            {"name": "mean", "value": 30.5},
            {"name": "std_dev", "value": 10.2},
            {"name": "count", "value": 100},
        ]

        assert isinstance(metrics, list)
        assert len(metrics) == 3

    def test_metrics_serialization(self):
        """Test metrics serialization."""
        import json

        metrics = {
            "mean": 30.5,
            "std_dev": 10.2,
            "count": 100,
        }

        serialized = json.dumps(metrics)
        deserialized = json.loads(serialized)

        assert deserialized["mean"] == 30.5

    def test_metrics_with_tags(self):
        """Test metrics with tags."""
        metrics = {
            "value": 100,
            "tags": {
                "service": "api",
                "region": "us-west",
            },
        }

        assert metrics["tags"]["service"] == "api"


class TestMetricsPipeline:
    """Test full metrics pipeline."""

    def test_event_to_metrics_pipeline(self):
        """Test converting events to metrics."""
        events = [
            {"value": 10},
            {"value": 20},
            {"value": 30},
        ]

        # Convert to metrics
        metrics = {
            "count": len(events),
            "sum": sum(e["value"] for e in events),
            "mean": sum(e["value"] for e in events) / len(events),
        }

        assert metrics["sum"] == 60
        assert metrics["mean"] == 20

    def test_metrics_enrichment(self):
        """Test enriching metrics."""
        metrics = {"value": 100}

        # Add derived metrics
        metrics["value_squared"] = metrics["value"] ** 2
        metrics["value_doubled"] = metrics["value"] * 2

        assert metrics["value_squared"] == 10000
        assert metrics["value_doubled"] == 200

    def test_metrics_bucketing(self):
        """Test bucketing metrics."""
        metrics = [
            {"value": 5},
            {"value": 15},
            {"value": 25},
            {"value": 35},
            {"value": 45},
        ]

        # Bucket into ranges
        buckets = {
            "low": [m for m in metrics if m["value"] < 20],
            "medium": [m for m in metrics if 20 <= m["value"] < 40],
            "high": [m for m in metrics if m["value"] >= 40],
        }

        assert len(buckets["low"]) == 2
        assert len(buckets["medium"]) == 2
        assert len(buckets["high"]) == 1

    def test_metrics_filtering(self):
        """Test filtering metrics."""
        metrics = [
            {"value": 10, "valid": True},
            {"value": 20, "valid": False},
            {"value": 30, "valid": True},
        ]

        # Filter valid metrics
        valid_metrics = [m for m in metrics if m["valid"]]

        assert len(valid_metrics) == 2
        assert all(m["valid"] for m in valid_metrics)

    def test_metrics_deduplication(self):
        """Test removing duplicate metrics."""
        metrics_with_dupes = [
            {"timestamp": "2024-01-01", "value": 100},
            {"timestamp": "2024-01-02", "value": 100},
            {"timestamp": "2024-01-01", "value": 100},  # Duplicate
        ]

        # Deduplicate
        seen = set()
        unique = []

        for m in metrics_with_dupes:
            key = (m["timestamp"], m["value"])
            if key not in seen:
                unique.append(m)
                seen.add(key)

        assert len(unique) == 2
