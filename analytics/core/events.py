"""
Event System for Analytics Engine

Implements event publishing and subscription patterns.
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """Represents an analytics event."""

    event_type: str
    event_data: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "analytics"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "priority": self.priority.name,
            "metadata": self.metadata,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }


class EventSystem:
    """
    Event publishing and subscription system.

    Features:
    - Event publication with multiple subscribers
    - Priority-based event processing
    - Async event handling
    - Event filtering
    - Metrics and monitoring

    Example:
        >>> event_system = EventSystem()
        >>> event_system.subscribe("data_processed", on_data_processed)
        >>> event = Event("data_processed", {"count": 100})
        >>> await event_system.publish(event)
    """

    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize event system.

        Args:
            max_queue_size: Maximum event queue size
        """
        self.max_queue_size = max_queue_size

        # Subscribers
        self._subscribers: Dict[str, List[Callable]] = {}
        self._wildcard_subscribers: List[Callable] = []

        # Event queue and processing
        self._event_queue: asyncio.PriorityQueue = None
        self._is_running = False
        self._processor_task = None

        # Metrics
        self._published_events = 0
        self._processed_events = 0
        self._failed_events = 0
        self._event_history: List[Event] = []
        self._max_history = 1000

        logger.info("Event system initialized")

    async def initialize(self) -> None:
        """Initialize event system."""
        self._event_queue = asyncio.PriorityQueue(maxsize=self.max_queue_size)
        self._is_running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("Event system started")

    async def shutdown(self) -> None:
        """Shutdown event system."""
        logger.info("Shutting down event system...")
        self._is_running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        logger.info("Event system shutdown complete")

    def subscribe(
        self,
        event_type: str,
        handler: Callable,
        filter_fn: Optional[Callable] = None
    ) -> str:
        """
        Subscribe to events of specific type.

        Args:
            event_type: Type of event to subscribe to (use "*" for wildcard)
            handler: Callback function to handle events
            filter_fn: Optional filter function to receive only matching events

        Returns:
            Subscription ID
        """
        if event_type == "*":
            self._wildcard_subscribers.append((handler, filter_fn))
            logger.debug("Subscribed to wildcard events")
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []

            self._subscribers[event_type].append((handler, filter_fn))
            logger.debug(f"Subscribed to events: {event_type}")

        return str(uuid.uuid4())

    def unsubscribe(self, event_type: str, handler: Callable) -> bool:
        """
        Unsubscribe from events.

        Args:
            event_type: Type of event
            handler: Handler function

        Returns:
            True if unsubscribed, False if not found
        """
        if event_type == "*":
            self._wildcard_subscribers = [
                (h, f) for h, f in self._wildcard_subscribers if h != handler
            ]
            return True
        else:
            if event_type in self._subscribers:
                original_len = len(self._subscribers[event_type])
                self._subscribers[event_type] = [
                    (h, f) for h, f in self._subscribers[event_type] if h != handler
                ]
                return len(self._subscribers[event_type]) < original_len

        return False

    async def publish(self, event: Event) -> None:
        """
        Publish event to subscribers.

        Args:
            event: Event to publish
        """
        if not self._is_running:
            logger.warning("Event system not running, dropping event")
            return

        try:
            # Priority is negative so higher priority events are processed first
            priority = -event.priority.value
            await self._event_queue.put((priority, event))
            self._published_events += 1

            # Keep event history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

        except asyncio.QueueFull:
            logger.error(f"Event queue full, dropping event: {event.event_type}")
            self._failed_events += 1
        except Exception as e:
            logger.error(f"Error publishing event: {e}", exc_info=True)
            self._failed_events += 1

    async def _process_events(self) -> None:
        """Process events from queue."""
        logger.info("Event processor started")

        try:
            while self._is_running:
                try:
                    # Get event with timeout
                    _, event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=1.0
                    )

                    # Dispatch to subscribers
                    await self._dispatch_event(event)
                    self._processed_events += 1

                except asyncio.TimeoutError:
                    # Timeout is normal when queue is empty
                    pass

        except asyncio.CancelledError:
            logger.debug("Event processor cancelled")
        except Exception as e:
            logger.error(f"Event processor error: {e}", exc_info=True)

    async def _dispatch_event(self, event: Event) -> None:
        """Dispatch event to subscribers."""
        handlers = []

        # Get type-specific subscribers
        if event.event_type in self._subscribers:
            handlers.extend(self._subscribers[event.event_type])

        # Add wildcard subscribers
        handlers.extend(self._wildcard_subscribers)

        # Dispatch to all matching handlers
        for handler, filter_fn in handlers:
            try:
                # Apply filter if provided
                if filter_fn and not filter_fn(event):
                    continue

                # Call handler
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)

            except Exception as e:
                logger.error(
                    f"Error in event handler for {event.event_type}: {e}",
                    exc_info=True
                )

    def get_metrics(self) -> Dict[str, Any]:
        """Get event system metrics."""
        return {
            "published_events": self._published_events,
            "processed_events": self._processed_events,
            "failed_events": self._failed_events,
            "queue_size": self._event_queue.qsize() if self._event_queue else 0,
            "subscribers": {
                event_type: len(handlers)
                for event_type, handlers in self._subscribers.items()
            },
            "wildcard_subscribers": len(self._wildcard_subscribers),
        }

    def get_event_history(self, event_type: Optional[str] = None) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Optional filter by event type

        Returns:
            List of events
        """
        if event_type:
            return [e for e in self._event_history if e.event_type == event_type]
        return self._event_history.copy()
