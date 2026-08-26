"""
Analytics Event System

Provides pub/sub event system for analytics pipeline.
Events are collected, validated, and distributed to processors.

Design Principles:
- Type-safe event structures with dataclasses
- Asynchronous event processing
- Event deduplication via UUID
- Flexible JSONB payload for extensibility
- Observer pattern with typed subscribers
"""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Callable, List, Dict, Any, Optional, Coroutine
from abc import ABC, abstractmethod
import asyncio
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types in the analytics system."""

    # Data collection events
    METRIC_COLLECTED = "metric_collected"
    EVENT_INGESTED = "event_ingested"

    # Analytics events
    ANOMALY_DETECTED = "anomaly_detected"
    CORRELATION_COMPUTED = "correlation_computed"
    SUMMARY_GENERATED = "summary_generated"

    # System events
    PROCESSING_STARTED = "processing_started"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_FAILED = "processing_failed"
    CACHE_INVALIDATED = "cache_invalidated"

    # Alert events
    ALERT_TRIGGERED = "alert_triggered"
    ALERT_ACKNOWLEDGED = "alert_acknowledged"
    ALERT_RESOLVED = "alert_resolved"


@dataclass
class Event:
    """
    Base event structure for the analytics system.

    All events include:
    - Unique ID for deduplication
    - Type classification
    - Source identification
    - Flexible payload (JSONB-compatible)
    - Metadata for context
    - Timestamps for tracking
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = field(default="")  # EventType.value
    source: str = field(default="")  # e.g., 'imaging_service', 'user_action'
    payload: Dict[str, Any] = field(default_factory=dict)  # Core event data
    metadata: Dict[str, Any] = field(default_factory=dict)  # Context info
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = field(default=None)

    # Validation and tracking
    is_valid: bool = field(default=True)
    validation_errors: List[str] = field(default_factory=list)
    retry_count: int = field(default=0)

    def validate(self) -> bool:
        """Validate event structure and content."""
        self.validation_errors.clear()

        # Check required fields
        if not self.event_type:
            self.validation_errors.append("event_type is required")
        if not self.source:
            self.validation_errors.append("source is required")

        # Check payload is dict
        if not isinstance(self.payload, dict):
            self.validation_errors.append("payload must be a dictionary")

        # Check metadata is dict
        if not isinstance(self.metadata, dict):
            self.validation_errors.append("metadata must be a dictionary")

        self.is_valid = len(self.validation_errors) == 0
        return self.is_valid

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary (JSONB-compatible)."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "payload": self.payload,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=data.get("event_type", ""),
            source=data.get("source", ""),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
        )


@dataclass
class MetricEvent(Event):
    """Event representing a collected metric."""

    metric_name: str = field(default="")
    metric_value: float = field(default=0.0)
    metric_type: str = field(default="gauge")  # gauge, counter, histogram
    tags: Dict[str, str] = field(default_factory=dict)  # e.g., service, region

    def __post_init__(self):
        """Initialize metric event."""
        self.event_type = EventType.METRIC_COLLECTED.value
        self.payload = {
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_type": self.metric_type,
            "tags": self.tags,
        }


@dataclass
class AnomalyEvent(Event):
    """Event representing a detected anomaly."""

    metric_name: str = field(default="")
    anomaly_type: str = field(default="outlier")
    severity_score: float = field(default=0.0)  # 0.0-1.0
    confidence: float = field(default=0.0)  # 0.0-1.0
    expected_value: Optional[float] = field(default=None)
    observed_value: Optional[float] = field(default=None)
    detection_method: str = field(default="")

    def __post_init__(self):
        """Initialize anomaly event."""
        self.event_type = EventType.ANOMALY_DETECTED.value
        self.payload = {
            "metric_name": self.metric_name,
            "anomaly_type": self.anomaly_type,
            "severity_score": self.severity_score,
            "confidence": self.confidence,
            "expected_value": self.expected_value,
            "observed_value": self.observed_value,
            "detection_method": self.detection_method,
        }


class EventSubscriber(ABC):
    """Abstract base class for event subscribers."""

    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        """Handle an incoming event."""
        pass

    @abstractmethod
    def get_event_types(self) -> List[str]:
        """Return list of event types this subscriber handles."""
        pass


class EventBus:
    """
    Central event bus for the analytics system.

    Responsibilities:
    - Publish events
    - Subscribe to events
    - Route events to subscribers
    - Deduplication
    - Asynchronous processing
    """

    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize event bus.

        Args:
            max_queue_size: Maximum size of event queue
        """
        self.max_queue_size = max_queue_size
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)

        # Subscribers: event_type -> list of subscribers
        self._subscribers: Dict[str, List[EventSubscriber]] = defaultdict(list)
        self._all_subscribers: List[EventSubscriber] = []  # Subscribers for all events

        # Event tracking
        self._seen_event_ids: set = set()  # For deduplication
        self._published_count = 0
        self._duplicated_count = 0
        self._failed_count = 0

        logger.info(f"EventBus initialized with max_queue_size={max_queue_size}")

    async def publish(self, event: Event) -> bool:
        """
        Publish an event.

        Args:
            event: Event to publish

        Returns:
            True if published, False if duplicate or queue full

        Raises:
            asyncio.QueueFull: If event queue is full
        """
        # Validate event
        if not event.validate():
            logger.error(f"Invalid event: {event.validation_errors}")
            self._failed_count += 1
            return False

        # Check for duplicates
        if event.event_id in self._seen_event_ids:
            logger.debug(f"Duplicate event received: {event.event_id}")
            self._duplicated_count += 1
            return False

        self._seen_event_ids.add(event.event_id)

        # Publish to queue
        try:
            self.event_queue.put_nowait(event)
            self._published_count += 1
            logger.debug(f"Published event: {event.event_type} (id={event.event_id})")
            return True
        except asyncio.QueueFull:
            logger.error(f"Event queue full, dropping event: {event.event_id}")
            self._failed_count += 1
            raise

    def subscribe(self, event_type: str, subscriber: EventSubscriber) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Event type to subscribe to (use "" for all events)
            subscriber: Subscriber instance
        """
        if event_type == "":
            self._all_subscribers.append(subscriber)
            logger.info(f"Subscriber registered for all events: {subscriber.__class__.__name__}")
        else:
            self._subscribers[event_type].append(subscriber)
            logger.info(f"Subscriber registered for event type '{event_type}': {subscriber.__class__.__name__}")

    def unsubscribe(self, event_type: str, subscriber: EventSubscriber) -> None:
        """
        Unsubscribe from events.

        Args:
            event_type: Event type to unsubscribe from
            subscriber: Subscriber instance
        """
        if event_type == "":
            if subscriber in self._all_subscribers:
                self._all_subscribers.remove(subscriber)
        else:
            if subscriber in self._subscribers[event_type]:
                self._subscribers[event_type].remove(subscriber)

    async def dispatch(self, event: Event) -> None:
        """
        Dispatch an event to all subscribers.

        Args:
            event: Event to dispatch
        """
        # Get subscribers for this event type and all-event subscribers
        subscribers = (
            self._subscribers.get(event.event_type, []) +
            self._all_subscribers
        )

        if not subscribers:
            logger.debug(f"No subscribers for event type: {event.event_type}")
            return

        # Call all subscribers (wait for all to complete)
        tasks = [
            subscriber.handle_event(event)
            for subscriber in subscribers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Subscriber {subscribers[i].__class__.__name__} failed: {result}",
                    exc_info=result
                )

    async def run(self) -> None:
        """
        Main event processing loop.
        Continuously pulls events from queue and dispatches to subscribers.
        """
        logger.info("EventBus dispatcher started")
        try:
            while True:
                try:
                    # Get event with timeout to allow graceful shutdown
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=1.0
                    )

                    # Mark as processed
                    event.processed_at = datetime.utcnow()

                    # Dispatch to subscribers
                    await self.dispatch(event)

                    # Mark task as done
                    self.event_queue.task_done()

                except asyncio.TimeoutError:
                    # Timeout is normal - just loop again
                    continue

        except asyncio.CancelledError:
            logger.info("EventBus dispatcher cancelled")
            raise

    def get_stats(self) -> Dict[str, int]:
        """Get event statistics."""
        return {
            "published": self._published_count,
            "duplicated": self._duplicated_count,
            "failed": self._failed_count,
            "queue_size": self.event_queue.qsize(),
        }

    def clear_seen_ids(self) -> None:
        """Clear the set of seen event IDs (use with caution)."""
        self._seen_event_ids.clear()


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def set_event_bus(bus: EventBus) -> None:
    """Override global event bus (for testing)."""
    global _event_bus
    _event_bus = bus


def reset_event_bus() -> None:
    """Reset event bus (for testing)."""
    global _event_bus
    _event_bus = None
