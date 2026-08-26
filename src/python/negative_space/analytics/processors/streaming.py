"""
Real-Time Streaming Processor

Handles continuous data streams with windowing, aggregation,
and real-time analytics operations.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

from ..core.metrics import Metric, MetricsAggregator
from ..core.events import Event, EventBus

logger = logging.getLogger(__name__)


class WindowType(Enum):
    """Types of streaming windows."""
    TUMBLING = "tumbling"  # Non-overlapping
    SLIDING = "sliding"    # Overlapping
    SESSION = "session"    # Activity-based


@dataclass
class WindowConfig:
    """Configuration for streaming window."""
    window_type: WindowType
    window_size_seconds: int = 60
    slide_interval_seconds: int = 10  # For sliding windows
    inactivity_timeout_seconds: int = 300  # For session windows


@dataclass
class StreamElement:
    """Single element in a stream."""
    timestamp: datetime
    data: Any
    window_id: Optional[str] = field(default=None)


class StreamProcessor:
    """
    Real-time stream processor with windowing and aggregation.

    Features:
    - Multiple window types (tumbling, sliding, session)
    - Real-time aggregation
    - Backpressure handling
    - Error recovery
    - Async processing
    """

    def __init__(
        self,
        window_config: WindowConfig,
        max_buffer_size: int = 10000,
        max_concurrent_windows: int = 100,
    ):
        """
        Initialize stream processor.

        Args:
            window_config: Window configuration
            max_buffer_size: Maximum buffered elements
            max_concurrent_windows: Max concurrent windows
        """
        self.config = window_config
        self.max_buffer_size = max_buffer_size
        self.max_concurrent_windows = max_concurrent_windows
        self._buffer: asyncio.Queue[StreamElement] = asyncio.Queue()
        self._windows: Dict[str, List[StreamElement]] = {}
        self._window_timers: Dict[str, asyncio.Task] = {}
        self._running = False
        self._elements_processed = 0
        self._windows_created = 0
        self._errors = 0

    def add_element(self, element: Any) -> bool:
        """
        Add element to stream (non-async).

        Args:
            element: Element to add

        Returns:
            bool: True if queued successfully
        """
        try:
            stream_elem = StreamElement(
                timestamp=datetime.utcnow(),
                data=element,
            )
            self._buffer.put_nowait(stream_elem)
            return True
        except asyncio.QueueFull:
            logger.warning("Stream buffer full, dropping element")
            return False

    async def process_stream(
        self,
        handler: Callable[[List[StreamElement]], Awaitable[None]],
    ) -> None:
        """
        Process stream with handler for window outputs.

        Args:
            handler: Async function to handle window results
        """
        self._running = True

        try:
            while self._running:
                # Get next element
                try:
                    element = await asyncio.wait_for(
                        self._buffer.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Check for expired windows
                    await self._check_expired_windows(handler)
                    continue

                # Route to window
                await self._route_to_window(element, handler)
                self._elements_processed += 1

        except asyncio.CancelledError:
            logger.info("Stream processor cancelled")
        except Exception as e:
            logger.error(f"Stream processor error: {e}")
            self._errors += 1
            raise
        finally:
            self._running = False

    async def _route_to_window(
        self,
        element: StreamElement,
        handler: Callable[[List[StreamElement]], Awaitable[None]],
    ) -> None:
        """
        Route element to appropriate window.

        Args:
            element: Stream element
            handler: Window handler
        """
        try:
            if self.config.window_type == WindowType.TUMBLING:
                await self._route_tumbling(element, handler)
            elif self.config.window_type == WindowType.SLIDING:
                await self._route_sliding(element, handler)
            elif self.config.window_type == WindowType.SESSION:
                await self._route_session(element, handler)
        except Exception as e:
            logger.error(f"Error routing to window: {e}")
            self._errors += 1

    async def _route_tumbling(
        self,
        element: StreamElement,
        handler: Callable[[List[StreamElement]], Awaitable[None]],
    ) -> None:
        """
        Route to tumbling window (non-overlapping).

        Args:
            element: Stream element
            handler: Window handler
        """
        # Window ID based on fixed time boundaries
        window_start = (
            element.timestamp.replace(second=0, microsecond=0)
            - timedelta(seconds=element.timestamp.second %
                        self.config.window_size_seconds)
        )
        window_id = window_start.isoformat()
        element.window_id = window_id

        # Add to window
        if window_id not in self._windows:
            self._windows[window_id] = []
            self._windows_created += 1

        self._windows[window_id].append(element)

        # Check if window is complete
        window_end = window_start + timedelta(
            seconds=self.config.window_size_seconds
        )

        if element.timestamp >= window_end:
            # Emit window
            window_data = self._windows.pop(window_id)
            await handler(window_data)

    async def _route_sliding(
        self,
        element: StreamElement,
        handler: Callable[[List[StreamElement]], Awaitable[None]],
    ) -> None:
        """
        Route to sliding windows (overlapping).

        Args:
            element: Stream element
            handler: Window handler
        """
        # Multiple overlapping windows
        now = element.timestamp
        window_start = now - timedelta(seconds=self.config.window_size_seconds)

        # Find all windows this element belongs to
        windows_to_check = int(
            self.config.window_size_seconds /
            self.config.slide_interval_seconds
        ) + 1

        for i in range(windows_to_check):
            w_start = (
                window_start
                + timedelta(seconds=i * self.config.slide_interval_seconds)
            )
            w_end = w_start + timedelta(seconds=self.config.window_size_seconds)

            if w_start <= element.timestamp < w_end:
                window_id = f"sliding_{w_start.isoformat()}"
                element.window_id = window_id

                if window_id not in self._windows:
                    self._windows[window_id] = []
                    self._windows_created += 1

                self._windows[window_id].append(element)

    async def _route_session(
        self,
        element: StreamElement,
        handler: Callable[[List[StreamElement]], Awaitable[None]],
    ) -> None:
        """
        Route to session windows (activity-based).

        Args:
            element: Stream element
            handler: Window handler
        """
        now = element.timestamp

        # Find active session or create new one
        active_session = None
        for window_id, elements in self._windows.items():
            if not elements:
                continue

            last_element_time = elements[-1].timestamp
            time_diff = (now - last_element_time).total_seconds()

            if time_diff < self.config.inactivity_timeout_seconds:
                active_session = window_id
                break

        if active_session is None:
            # Create new session
            active_session = f"session_{now.isoformat()}"
            self._windows[active_session] = []
            self._windows_created += 1

        element.window_id = active_session
        self._windows[active_session].append(element)

    async def _check_expired_windows(
        self,
        handler: Callable[[List[StreamElement]], Awaitable[None]],
    ) -> None:
        """
        Check for and emit expired windows.

        Args:
            handler: Window handler
        """
        now = datetime.utcnow()
        expired = []

        for window_id, elements in self._windows.items():
            if not elements:
                continue

            last_time = elements[-1].timestamp
            elapsed = (now - last_time).total_seconds()

            # Emit session windows after inactivity
            if (self.config.window_type == WindowType.SESSION
                and elapsed > self.config.inactivity_timeout_seconds):
                expired.append(window_id)

        # Emit expired windows
        for window_id in expired:
            window_data = self._windows.pop(window_id)
            await handler(window_data)

    def stop(self) -> None:
        """Stop stream processor."""
        self._running = False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.

        Returns:
            Dictionary with processing stats
        """
        return {
            "elements_processed": self._elements_processed,
            "windows_created": self._windows_created,
            "active_windows": len(self._windows),
            "buffered_elements": self._buffer.qsize(),
            "errors": self._errors,
        }


class MetricsStreamAggregator:
    """
    Aggregates metrics from a stream.

    Specializes StreamProcessor for metric aggregation.
    """

    def __init__(
        self,
        metrics_aggregator: MetricsAggregator,
        window_config: WindowConfig,
    ):
        """
        Initialize metrics stream aggregator.

        Args:
            metrics_aggregator: Underlying aggregator
            window_config: Window configuration
        """
        self.aggregator = metrics_aggregator
        self.stream = StreamProcessor(window_config)
        self._aggregated_count = 0

    async def process_metric_stream(
        self,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """
        Process metric stream with aggregation.

        Args:
            event_bus: Optional event bus for emitting aggregation events
        """
        async def aggregate_window(elements: List[StreamElement]) -> None:
            """Handle aggregated window of metrics."""
            try:
                if not elements:
                    return

                # Extract metrics from elements
                metrics = [e.data for e in elements if isinstance(e.data, Metric)]

                if not metrics:
                    return

                # Compute aggregation
                min_val = min(m.value for m in metrics)
                max_val = max(m.value for m in metrics)
                avg_val = sum(m.value for m in metrics) / len(metrics)

                logger.info(
                    f"Aggregated {len(metrics)} metrics: "
                    f"min={min_val}, max={max_val}, avg={avg_val:.2f}"
                )

                self._aggregated_count += len(metrics)

                # Emit event if bus provided
                if event_bus:
                    event = Event(
                        source="metric_stream_aggregator",
                        event_type="AGGREGATION_COMPLETE",
                        payload={
                            "metric_count": len(metrics),
                            "min": min_val,
                            "max": max_val,
                            "avg": avg_val,
                            "window_id": elements[0].window_id,
                        }
                    )
                    await event_bus.publish(event)

            except Exception as e:
                logger.error(f"Error aggregating window: {e}")

        await self.stream.process_stream(aggregate_window)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregator statistics.

        Returns:
            Dictionary combining stream and aggregation stats
        """
        stats = self.stream.get_stats()
        stats["metrics_aggregated"] = self._aggregated_count
        return stats
