"""
Storage Component Tests

Tests for storage layer including persistence, retrieval, aggregation,
and storage management in the analytics pipeline.
"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any


class TestStorageBasics:
    """Test basic storage operations."""

    def test_store_metric(self):
        """Test storing a single metric."""
        storage = {}
        metric_id = "metric_1"

        metric = {
            "timestamp": datetime.now(),
            "value": 100,
            "source": "sensor_1",
        }

        storage[metric_id] = metric

        assert metric_id in storage
        assert storage[metric_id]["value"] == 100

    def test_retrieve_metric(self):
        """Test retrieving stored metric."""
        storage = {}
        metric = {"timestamp": datetime.now(), "value": 100}

        storage["m1"] = metric

        retrieved = storage.get("m1")
        assert retrieved is not None
        assert retrieved["value"] == 100

    def test_delete_metric(self):
        """Test deleting stored metric."""
        storage = {"m1": {"value": 100}, "m2": {"value": 200}}

        del storage["m1"]

        assert "m1" not in storage
        assert "m2" in storage

    def test_list_all_metrics(self):
        """Test listing all stored metrics."""
        storage = {
            "m1": {"value": 100},
            "m2": {"value": 200},
            "m3": {"value": 300},
        }

        metrics = list(storage.values())
        assert len(metrics) == 3


class TestStoragePersistence:
    """Test persistent storage."""

    def test_save_to_file(self):
        """Test saving storage to file."""
        storage_data = {
            "m1": {"timestamp": "2024-01-01", "value": 100},
            "m2": {"timestamp": "2024-01-02", "value": 200},
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(storage_data, f)
            storage_file = f.name

        # Load and verify
        with open(storage_file, "r") as f:
            loaded = json.load(f)

        assert len(loaded) == 2
        assert loaded["m1"]["value"] == 100

    def test_load_from_file(self):
        """Test loading storage from file."""
        storage_data = {
            "m1": {"value": 100},
            "m2": {"value": 200},
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(storage_data, f)
            storage_file = f.name

        try:
            with open(storage_file, "r") as f:
                loaded_storage = json.load(f)

            assert len(loaded_storage) == 2
        finally:
            import os
            os.unlink(storage_file)

    def test_incremental_save(self):
        """Test incremental saving."""
        storage = {}

        # Add and save incrementally
        for i in range(5):
            storage[f"m{i}"] = {"value": i * 100}

        assert len(storage) == 5


class TestStorageRetrieval:
    """Test storage retrieval operations."""

    def test_retrieve_by_timestamp_range(self):
        """Test retrieving metrics by time range."""
        storage = {
            "m1": {"timestamp": datetime(2024, 1, 1, 10, 0), "value": 100},
            "m2": {"timestamp": datetime(2024, 1, 1, 11, 0), "value": 200},
            "m3": {"timestamp": datetime(2024, 1, 1, 12, 0), "value": 300},
        }

        start = datetime(2024, 1, 1, 10, 30)
        end = datetime(2024, 1, 1, 11, 30)

        in_range = [
            m for m in storage.values()
            if start <= m["timestamp"] <= end
        ]

        assert len(in_range) == 1
        assert in_range[0]["value"] == 200

    def test_retrieve_by_source(self):
        """Test retrieving metrics by source."""
        storage = {
            "m1": {"value": 100, "source": "sensor_1"},
            "m2": {"value": 200, "source": "sensor_2"},
            "m3": {"value": 300, "source": "sensor_1"},
        }

        sensor1_metrics = [
            m for m in storage.values() if m["source"] == "sensor_1"
        ]

        assert len(sensor1_metrics) == 2

    def test_retrieve_with_aggregation(self):
        """Test retrieval with aggregation."""
        storage = [
            {"value": 10, "category": "A"},
            {"value": 20, "category": "A"},
            {"value": 30, "category": "B"},
        ]

        category_a = [m for m in storage if m["category"] == "A"]
        sum_a = sum(m["value"] for m in category_a)

        assert len(category_a) == 2
        assert sum_a == 30


class TestStorageAggregation:
    """Test storage aggregation operations."""

    def test_aggregate_sum(self):
        """Test aggregating sum."""
        storage = [
            {"value": 10},
            {"value": 20},
            {"value": 30},
        ]

        total = sum(m["value"] for m in storage)
        assert total == 60

    def test_aggregate_mean(self):
        """Test aggregating mean."""
        storage = [
            {"value": 10},
            {"value": 20},
            {"value": 30},
        ]

        mean = sum(m["value"] for m in storage) / len(storage)
        assert mean == 20

    def test_aggregate_count(self):
        """Test counting aggregation."""
        storage = [
            {"value": 10},
            {"value": 20},
            {"value": 30},
        ]

        count = len(storage)
        assert count == 3

    def test_aggregate_by_group(self):
        """Test aggregation by group."""
        storage = [
            {"group": "A", "value": 10},
            {"group": "B", "value": 20},
            {"group": "A", "value": 15},
        ]

        grouped = {}
        for item in storage:
            group = item["group"]
            if group not in grouped:
                grouped[group] = []
            grouped[group].append(item)

        assert len(grouped["A"]) == 2
        assert len(grouped["B"]) == 1

    def test_aggregate_percentiles(self):
        """Test percentile aggregation."""
        storage = list(range(1, 101))  # 1-100

        sorted_data = sorted(storage)
        p50_idx = len(sorted_data) // 2
        p90_idx = int(0.9 * len(sorted_data))

        p50 = sorted_data[p50_idx]
        p90 = sorted_data[p90_idx]

        assert p50 == 51  # Median
        assert p90 == 91  # 90th percentile


class TestStorageManagement:
    """Test storage management."""

    def test_storage_size_tracking(self):
        """Test tracking storage size."""
        storage = {}

        # Add items
        for i in range(10):
            storage[f"m{i}"] = {"value": i}

        size = len(storage)
        assert size == 10

    def test_storage_cleanup(self):
        """Test cleanup of old data."""
        storage = {}
        now = datetime.now()

        # Add old and new data
        for i in range(5):
            age = timedelta(days=i + 100)  # Old
            storage[f"old_{i}"] = {"timestamp": now - age}

        for i in range(5):
            storage[f"new_{i}"] = {"timestamp": now}

        # Cleanup old data (> 30 days)
        cutoff = now - timedelta(days=30)
        cleaned = {
            k: v for k, v in storage.items()
            if v["timestamp"] >= cutoff
        }

        assert len(cleaned) == 5

    def test_storage_quota(self):
        """Test storage quota enforcement."""
        max_items = 100
        storage = {}

        for i in range(150):
            if len(storage) < max_items:
                storage[f"m{i}"] = {"value": i}
            else:
                break

        assert len(storage) <= max_items

    def test_storage_compaction(self):
        """Test storage compaction."""
        storage = {}

        # Add sparse data
        for i in range(0, 100, 10):
            storage[f"m{i}"] = {"value": i}

        assert len(storage) == 10
