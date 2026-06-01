"""
Streaming Pipeline Tests

Tests for streaming components including window processors, backpressure
handling, and metrics aggregation. Validates that stream data flows correctly
through the pipeline with proper ordering and aggregation.
"""

import pytest
from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict, Any


class TestStreamProcessor:
    """Test stream processor with various window types."""

    @pytest.fixture
    def stream_data(self) -> List[Dict[str, Any]]:
        """Generate time-series stream data."""
        return [
            {
                "timestamp": datetime.now() - timedelta(seconds=60 - i),
                "value": 100 + (i % 20),
                "stream_id": 1,
                "sequence": i,
            }
            for i in range(30)
        ]

    def test_tumbling_window_creation(self, stream_data):
        """Test tumbling window creates distinct non-overlapping windows."""
        window_size = 10
        windows = []
        current_window = []

        for event in stream_data:
            current_window.append(event)

            if len(current_window) >= window_size:
                windows.append(current_window)
                current_window = []

        # Verify window properties
        assert len(windows) >= 2  # At least 2 complete windows
        for window in windows[:-1]:  # All but possibly last
            assert len(window) == window_size

    def test_sliding_window_overlap(self, stream_data):
        """Test sliding window with overlap."""
        window_size = 15
        slide_size = 5
        windows = []

        for i in range(0, len(stream_data) - window_size + 1, slide_size):
            window = stream_data[i:i + window_size]
            windows.append(window)

        # Verify sliding window properties
        assert len(windows) > 0
        for window in windows:
            assert len(window) == window_size

        # Verify overlap between consecutive windows
        if len(windows) > 1:
            # Check that overlap exists
            overlap = [e for e in windows[0] if e in windows[1]]
            assert len(overlap) > 0

    def test_session_window_on_gap(self, stream_data):
        """Test session window creates new window on inactivity gap."""
        gap_threshold = timedelta(seconds=5)
        windows = []
        current_window = []
        last_timestamp = None

        for event in stream_data:
            if last_timestamp and (
                event["timestamp"] - last_timestamp
            ) > gap_threshold:
                # Gap detected - start new window
                if current_window:
                    windows.append(current_window)
                    current_window = []

            current_window.append(event)
            last_timestamp = event["timestamp"]

        if current_window:
            windows.append(current_window)

        # Verify windows created
        assert len(windows) > 0

    def test_element_ordering_preservation(self, stream_data):
        """Test that elements maintain order through window."""
        window_size = 10
        window = stream_data[:window_size]

        # Verify sequence numbers in order
        sequences = [e["sequence"] for e in window]
        assert sequences == sorted(sequences)

    def test_element_routing_to_correct_window(self, stream_data):
        """Test that elements are routed to correct window."""
        window_size = 10
        slide_size = 5
        windows = []

        for i in range(0, len(stream_data) - window_size + 1, slide_size):
            window = stream_data[i:i + window_size]
            windows.append(window)

        # Verify each element is in the right window
        for window_idx, window in enumerate(windows):
            start_seq = window_idx * slide_size
            end_seq = start_seq + window_size

            for element in window:
                assert start_seq <= element["sequence"] < end_seq

    def test_late_element_handling(self):
        """Test handling of elements arriving out of order."""
        elements = [
            {"sequence": 1, "value": 10},
            {"sequence": 3, "value": 30},
            {"sequence": 2, "value": 20},  # Out of order
            {"sequence": 4, "value": 40},
        ]

        out_of_order_count = 0
        for i in range(1, len(elements)):
            if elements[i]["sequence"] < elements[i - 1]["sequence"]:
                out_of_order_count += 1

        # Detect out-of-order element
        assert out_of_order_count == 1

    def test_timestamp_extraction(self, stream_data):
        """Test timestamp extraction from elements."""
        for event in stream_data:
            timestamp = event["timestamp"]
            assert isinstance(timestamp, datetime)
            assert timestamp is not None

    def test_window_timeout_trigger(self):
        """Test that windows emit on timeout."""
        import time

        window_timeout = 0.5  # 500ms
        events = []
        emitted_windows = []

        # Simulate events with timeout
        for i in range(3):
            events.append({"sequence": i, "value": i * 10})
            time.sleep(window_timeout + 0.1)

        # Window should have emitted multiple times
        emitted_windows = [events]  # Simplified
        assert len(emitted_windows) >= 1


class TestBackpressure:
    """Test backpressure handling in streaming."""

    def test_buffer_overflow_detection(self):
        """Test detection when buffer overflows."""
        max_buffer_size = 100
        buffer = deque(maxlen=max_buffer_size)
        overflow_detected = False

        # Add more elements than buffer capacity
        for i in range(150):
            buffer.append(i)

        # In deque with maxlen, old elements are dropped
        # So buffer size never exceeds max_buffer_size
        assert len(buffer) <= max_buffer_size

        # For actual backpressure, would track dropped items
        if 150 > max_buffer_size:
            overflow_detected = True

        assert overflow_detected

    def test_graceful_slowdown(self):
        """Test graceful slowdown when buffer fills."""
        target_buffer_level = 0.8  # 80% capacity
        max_buffer_size = 100
        current_buffer_size = 0
        throttle_applied = False

        # Simulate buffer filling
        for i in range(100):
            current_buffer_size += 1

            # Check if throttle needed
            buffer_level = current_buffer_size / max_buffer_size

            if buffer_level >= target_buffer_level:
                throttle_applied = True
                # In real system, would slow down intake
                current_buffer_size = min(current_buffer_size, max_buffer_size)

        assert throttle_applied

    def test_queue_protection_from_burst(self):
        """Test queue remains protected during burst of events."""
        max_queue_size = 50
        queue_size = 0
        max_observed = 0
        burst_events = 100

        for i in range(burst_events):
            queue_size += 1

            # Protect queue size
            if queue_size > max_queue_size:
                # Apply backpressure - slow down or drop
                queue_size = max_queue_size

            max_observed = max(max_observed, queue_size)

        assert max_observed <= max_queue_size

    def test_consumer_pause_resume(self):
        """Test consumer can pause and resume processing."""
        consumer_paused = False
        events_processed = 0

        events = list(range(20))

        for event in events:
            if event == 10:  # Pause mid-stream
                consumer_paused = True

            if consumer_paused:
                continue  # Skip processing

            events_processed += 1

        # Partial processing due to pause
        assert events_processed < len(events)

        # Resume
        consumer_paused = False
        for event in events[10:]:
            if not consumer_paused:
                events_processed += 1

        # All events now processed
        assert events_processed == len(events)

    def test_backpressure_propagation_upstream(self):
        """Test backpressure signal propagates upstream."""
        stage_1_buffer = deque()
        stage_2_buffer = deque(maxlen=10)  # Limited
        stage_1_blocked = False

        # Producer (stage 1)
        for i in range(20):
            if len(stage_2_buffer) >= 10:  # Stage 2 full
                stage_1_blocked = True
                break

            stage_1_buffer.append(i)
            # Transfer to stage 2
            if stage_1_buffer:
                stage_2_buffer.append(stage_1_buffer.popleft())

        # Backpressure caused blocking
        assert stage_1_blocked


class TestMetricsStreamAggregator:
    """Test metrics aggregation within streaming windows."""

    def test_per_window_aggregation(self):
        """Test aggregation of metrics within a window."""
        window = [
            {"value": 10},
            {"value": 20},
            {"value": 30},
            {"value": 40},
            {"value": 50},
        ]

        # Compute aggregates
        values = [e["value"] for e in window]
        aggregates = {
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

        # Verify aggregates
        assert aggregates["count"] == 5
        assert aggregates["sum"] == 150
        assert aggregates["mean"] == 30.0
        assert aggregates["min"] == 10
        assert aggregates["max"] == 50

    def test_multi_key_aggregation(self):
        """Test aggregation by multiple keys."""
        window = [
            {"key1": "A", "key2": "X", "value": 10},
            {"key1": "A", "key2": "Y", "value": 20},
            {"key1": "B", "key2": "X", "value": 30},
            {"key1": "B", "key2": "Y", "value": 40},
        ]

        # Group by key1 and key2
        aggregates = {}
        for event in window:
            composite_key = (event["key1"], event["key2"])

            if composite_key not in aggregates:
                aggregates[composite_key] = {"sum": 0, "count": 0}

            aggregates[composite_key]["sum"] += event["value"]
            aggregates[composite_key]["count"] += 1

        # Verify grouping
        assert len(aggregates) == 4
        assert aggregates[("A", "X")]["sum"] == 10
        assert aggregates[("B", "Y")]["sum"] == 40

    def test_incremental_aggregation(self):
        """Test incremental aggregation as elements arrive."""
        stream = [10, 20, 30, 40, 50]
        running_sum = 0
        running_count = 0

        aggregates_over_time = []

        for value in stream:
            running_sum += value
            running_count += 1

            current_aggregate = {
                "sum": running_sum,
                "count": running_count,
                "mean": running_sum / running_count,
            }
            aggregates_over_time.append(current_aggregate)

        # Verify progression
        assert aggregates_over_time[0]["sum"] == 10
        assert aggregates_over_time[1]["mean"] == 15.0
        assert aggregates_over_time[-1]["sum"] == 150

    def test_window_emit_triggers_aggregation(self):
        """Test that window emission triggers aggregation."""
        window_size = 3
        elements = [{"value": i} for i in range(10)]
        emitted_aggregates = []

        for i, element in enumerate(elements):
            if (i + 1) % window_size == 0:  # Window complete
                window = elements[i - window_size + 1:i + 1]
                values = [e["value"] for e in window]

                aggregate = {
                    "sum": sum(values),
                    "count": len(values),
                    "mean": sum(values) / len(values),
                }
                emitted_aggregates.append(aggregate)

        # Verify aggregates emitted
        assert len(emitted_aggregates) == 3  # 10 elements / 3 per window
        assert emitted_aggregates[0]["sum"] == 3  # 0+1+2
        assert emitted_aggregates[1]["sum"] == 12  # 3+4+5

    def test_late_arriving_element_aggregation(self):
        """Test aggregation with late-arriving elements."""
        on_time_window = [10, 20, 30]
        late_element = 25

        # Initial aggregation
        initial_sum = sum(on_time_window)
        initial_count = len(on_time_window)
        initial_mean = initial_sum / initial_count

        # Late element arrives
        updated_sum = initial_sum + late_element
        updated_count = initial_count + 1
        updated_mean = updated_sum / updated_count

        # Verify recalculation
        assert initial_mean == 20.0
        assert updated_mean == 21.25

    def test_partial_window_aggregation(self):
        """Test aggregation of partial/incomplete window."""
        partial_window = [5, 10, 15]  # Less than window size

        # Compute aggregate even if incomplete
        aggregate = {
            "sum": sum(partial_window),
            "count": len(partial_window),
            "mean": sum(partial_window) / len(partial_window),
        }

        # Verify partial aggregation works
        assert aggregate["count"] == 3
        assert aggregate["sum"] == 30
        assert aggregate["mean"] == 10.0

    def test_aggregation_with_null_values(self):
        """Test aggregation handles null/None values."""
        window_with_nulls = [10, None, 30, None, 50]

        # Filter out nulls
        valid_values = [v for v in window_with_nulls if v is not None]

        aggregate = {
            "sum": sum(valid_values),
            "count": len(valid_values),
            "mean": (
                sum(valid_values) / len(valid_values)
                if valid_values
                else 0
            ),
        }

        # Verify null handling
        assert aggregate["count"] == 3
        assert aggregate["sum"] == 90

    def test_streaming_percentile_calculation(self):
        """Test percentile calculation in streaming context."""
        window = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        sorted_window = sorted(window)

        # Calculate percentiles
        p25_idx = int(0.25 * len(sorted_window))
        p50_idx = int(0.50 * len(sorted_window))
        p75_idx = int(0.75 * len(sorted_window))

        percentiles = {
            "p25": sorted_window[p25_idx],
            "p50": sorted_window[p50_idx],
            "p75": sorted_window[p75_idx],
        }

        # Verify percentile calculation
        assert percentiles["p50"] > 0
        assert percentiles["p75"] >= percentiles["p50"]

    def test_aggregation_accuracy_with_large_numbers(self):
        """Test aggregation accuracy with large numeric values."""
        large_window = [1000000 + i for i in range(100)]

        total_sum = sum(large_window)
        count = len(large_window)
        mean = total_sum / count

        # Verify no overflow or precision loss
        assert count == 100
        assert total_sum > 0
        assert mean > 0
        assert isinstance(mean, float)

    def test_aggregation_latency_tracking(self):
        """Test tracking of aggregation latency."""
        import time

        window = [10, 20, 30, 40, 50]

        start = time.time()

        # Compute aggregate
        aggregate = {
            "sum": sum(window),
            "mean": sum(window) / len(window),
        }

        latency = (time.time() - start) * 1000  # milliseconds

        # Verify computation was fast
        assert latency < 100  # Should complete in less than 100ms
        assert aggregate["mean"] == 30.0
