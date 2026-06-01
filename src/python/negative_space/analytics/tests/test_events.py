"""
Unit Tests for Analytics Core Event System

Tests for:
- Event creation and validation
- Event deduplication
- Event bus pub/sub
- Async event dispatch
"""

import pytest
import asyncio
import uuid
from datetime import datetime
from typing import List

from ..core.events import (
    Event,
    EventType,
    MetricEvent,
    AnomalyEvent,
    EventBus,
    EventSubscriber,
)


class TestEvent:
    """Test suite for Event dataclass."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = Event(
            event_type=EventType.METRIC_COLLECTED.value,
            source="test_source",
            payload={"test": "data"},
        )

        assert event.event_id is not None
        assert event.event_type == EventType.METRIC_COLLECTED.value
        assert event.source == "test_source"
        assert event.payload == {"test": "data"}
        assert event.is_valid is True

    def test_event_validation_required_fields(self):
        """Test event validation requires event_type and source."""
        event = Event()
        assert event.validate() is False
        assert "event_type is required" in event.validation_errors
        assert "source is required" in event.validation_errors

    def test_event_validation_payload_type(self):
        """Test event validation checks payload is dict."""
        event = Event(
            event_type="test",
            source="test",
            payload="not a dict"  # Wrong type
        )
        assert event.validate() is False
        assert "payload must be a dictionary" in event.validation_errors

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = Event(
            event_id="test-id",
            event_type="test_type",
            source="test_source",
            payload={"key": "value"},
        )

        d = event.to_dict()
        assert d["event_id"] == "test-id"
        assert d["event_type"] == "test_type"
        assert d["source"] == "test_source"
        assert d["payload"] == {"key": "value"}

    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        data = {
            "event_id": "test-id",
            "event_type": "test_type",
            "source": "test_source",
            "payload": {"key": "value"},
            "created_at": datetime.utcnow().isoformat(),
        }

        event = Event.from_dict(data)
        assert event.event_id == "test-id"
        assert event.event_type == "test_type"
        assert event.source == "test_source"


class TestMetricEvent:
    """Test suite for MetricEvent."""

    def test_metric_event_creation(self):
        """Test metric event creation."""
        event = MetricEvent(
            source="test_service",
            metric_name="event_rate",
            metric_value=42.5,
            metric_type="gauge",
            tags={"service": "api", "region": "us-east"},
        )

        assert event.event_type == EventType.METRIC_COLLECTED.value
        assert event.metric_name == "event_rate"
        assert event.metric_value == 42.5
        assert event.payload["metric_name"] == "event_rate"
        assert event.payload["metric_value"] == 42.5


class TestAnomalyEvent:
    """Test suite for AnomalyEvent."""

    def test_anomaly_event_creation(self):
        """Test anomaly event creation."""
        event = AnomalyEvent(
            source="analytics_engine",
            metric_name="error_rate",
            anomaly_type="outlier",
            severity_score=0.95,
            confidence=0.88,
            observed_value=0.25,
            expected_value=0.05,
        )

        assert event.event_type == EventType.ANOMALY_DETECTED.value
        assert event.severity_score == 0.95
        assert event.payload["anomaly_type"] == "outlier"
        assert event.payload["severity_score"] == 0.95


class MockSubscriber(EventSubscriber):
    """Mock subscriber for testing."""

    def __init__(self, event_types: List[str] = None):
        self.received_events: List[Event] = []
        self.event_types = event_types or [EventType.METRIC_COLLECTED.value]

    async def handle_event(self, event: Event) -> None:
        """Handle event by storing it."""
        self.received_events.append(event)

    def get_event_types(self) -> List[str]:
        """Return event types this subscriber handles."""
        return self.event_types


class TestEventBus:
    """Test suite for EventBus."""

    @pytest.mark.asyncio
    async def test_event_bus_creation(self):
        """Test EventBus initialization."""
        bus = EventBus(max_queue_size=1000)
        assert bus.max_queue_size == 1000
        assert bus.event_queue.maxsize == 1000
        assert bus._published_count == 0

    @pytest.mark.asyncio
    async def test_publish_valid_event(self):
        """Test publishing a valid event."""
        bus = EventBus()
        event = Event(
            event_type=EventType.METRIC_COLLECTED.value,
            source="test"
        )

        result = await bus.publish(event)
        assert result is True
        assert bus._published_count == 1

    @pytest.mark.asyncio
    async def test_publish_invalid_event(self):
        """Test publishing an invalid event returns False."""
        bus = EventBus()
        event = Event()  # Missing required fields

        result = await bus.publish(event)
        assert result is False
        assert bus._failed_count == 1
        assert bus._published_count == 0

    @pytest.mark.asyncio
    async def test_event_deduplication(self):
        """Test duplicate events are rejected."""
        bus = EventBus()
        event_id = "unique-id-" + str(uuid.uuid4())

        event1 = Event(
            event_id=event_id,
            event_type="test",
            source="test"
        )
        event2 = Event(
            event_id=event_id,  # Same ID
            event_type="test",
            source="test"
        )

        result1 = await bus.publish(event1)
        result2 = await bus.publish(event2)

        assert result1 is True
        assert result2 is False
        assert bus._duplicated_count == 1

    @pytest.mark.asyncio
    async def test_subscribe_to_event_type(self):
        """Test subscribing to specific event type."""
        bus = EventBus()
        subscriber = MockSubscriber([EventType.METRIC_COLLECTED.value])

        bus.subscribe(EventType.METRIC_COLLECTED.value, subscriber)

        assert subscriber in bus._subscribers[EventType.METRIC_COLLECTED.value]

    @pytest.mark.asyncio
    async def test_unsubscribe_from_event_type(self):
        """Test unsubscribing from event type."""
        bus = EventBus()
        subscriber = MockSubscriber([EventType.METRIC_COLLECTED.value])

        bus.subscribe(EventType.METRIC_COLLECTED.value, subscriber)
        assert subscriber in bus._subscribers[EventType.METRIC_COLLECTED.value]

        bus.unsubscribe(EventType.METRIC_COLLECTED.value, subscriber)
        assert subscriber not in bus._subscribers[EventType.METRIC_COLLECTED.value]

    @pytest.mark.asyncio
    async def test_dispatch_to_subscriber(self):
        """Test event dispatch to subscriber."""
        bus = EventBus()
        subscriber = MockSubscriber([EventType.METRIC_COLLECTED.value])
        bus.subscribe(EventType.METRIC_COLLECTED.value, subscriber)

        event = Event(
            event_type=EventType.METRIC_COLLECTED.value,
            source="test"
        )
        await bus.dispatch(event)

        assert len(subscriber.received_events) == 1
        assert subscriber.received_events[0] == event

    @pytest.mark.asyncio
    async def test_dispatch_to_all_subscribers(self):
        """Test event dispatch to all subscribers."""
        bus = EventBus()
        subscriber1 = MockSubscriber([])
        subscriber2 = MockSubscriber([])

        bus.subscribe("", subscriber1)  # Subscribe to all events
        bus.subscribe("", subscriber2)

        event = Event(
            event_type=EventType.METRIC_COLLECTED.value,
            source="test"
        )
        await bus.dispatch(event)

        assert len(subscriber1.received_events) == 1
        assert len(subscriber2.received_events) == 1

    @pytest.mark.asyncio
    async def test_dispatch_wrong_event_type(self):
        """Test subscriber only receives matching event types."""
        bus = EventBus()
        subscriber = MockSubscriber([EventType.METRIC_COLLECTED.value])
        bus.subscribe(EventType.METRIC_COLLECTED.value, subscriber)

        event = Event(
            event_type=EventType.ANOMALY_DETECTED.value,  # Different type
            source="test"
        )
        await bus.dispatch(event)

        assert len(subscriber.received_events) == 0

    def test_get_stats(self):
        """Test bus statistics."""
        bus = EventBus()

        stats = bus.get_stats()
        assert stats["published"] == 0
        assert stats["duplicated"] == 0
        assert stats["failed"] == 0

    @pytest.mark.asyncio
    async def test_queue_full_raises_error(self):
        """Test publishing to full queue raises error."""
        bus = EventBus(max_queue_size=1)

        # Fill queue
        event1 = Event(
            event_type="test",
            source="test"
        )
        await bus.publish(event1)

        # Try to overfill
        event2 = Event(
            event_type="test",
            source="test"
        )
        with pytest.raises(asyncio.QueueFull):
            await bus.publish(event2)

    def test_clear_seen_ids(self):
        """Test clearing seen event IDs."""
        bus = EventBus()
        bus._seen_event_ids.add("test-id")

        assert "test-id" in bus._seen_event_ids
        bus.clear_seen_ids()
        assert "test-id" not in bus._seen_event_ids


class TestEventBusIntegration:
    """Integration tests for EventBus."""

    @pytest.mark.asyncio
    async def test_full_publish_dispatch_flow(self):
        """Test full flow: publish -> queue -> dispatch -> subscriber."""
        bus = EventBus()
        subscriber = MockSubscriber([EventType.METRIC_COLLECTED.value])
        bus.subscribe(EventType.METRIC_COLLECTED.value, subscriber)

        # Publish event
        event = Event(
            event_type=EventType.METRIC_COLLECTED.value,
            source="test"
        )
        result = await bus.publish(event)
        assert result is True

        # Get event from queue and dispatch
        queued_event = bus.event_queue.get_nowait()
        await bus.dispatch(queued_event)

        # Verify subscriber received it
        assert len(subscriber.received_events) == 1

    @pytest.mark.asyncio
    async def test_multiple_events_in_sequence(self):
        """Test publishing and dispatching multiple events."""
        bus = EventBus()
        subscriber = MockSubscriber([EventType.METRIC_COLLECTED.value])
        bus.subscribe(EventType.METRIC_COLLECTED.value, subscriber)

        # Publish multiple events
        for i in range(5):
            event = Event(
                event_type=EventType.METRIC_COLLECTED.value,
                source=f"test_{i}"
            )
            await bus.publish(event)

        # Dispatch all
        while not bus.event_queue.empty():
            queued_event = bus.event_queue.get_nowait()
            await bus.dispatch(queued_event)

        assert len(subscriber.received_events) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
