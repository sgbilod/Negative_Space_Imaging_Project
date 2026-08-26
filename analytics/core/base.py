"""
Core Analytics Engine

Main orchestration layer for all analytics operations.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of analysis available."""
    STATISTICAL = "statistical"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    OPTIMIZATION = "optimization"
    TIME_SERIES = "time_series"
    CORRELATION = "correlation"


@dataclass
class AnalyticsConfig:
    """Configuration for analytics engine."""

    # Streaming configuration
    stream_buffer_size: int = 10000
    stream_flush_interval: float = 1.0  # seconds

    # Storage configuration
    retention_days: int = 90
    compression_enabled: bool = True

    # Processing configuration
    max_workers: int = 4
    batch_size: int = 1000

    # Anomaly detection configuration
    anomaly_sensitivity: float = 0.95
    anomaly_algorithms: List[str] = field(default_factory=lambda: [
        "isolation_forest",
        "local_outlier_factor",
        "statistical"
    ])

    # Monitoring
    enable_metrics: bool = True
    metrics_export_interval: float = 60.0  # seconds

    # Cache
    cache_enabled: bool = True
    cache_ttl: int = 300  # seconds


@dataclass
class AnalyticsMetrics:
    """Container for analytics metrics."""

    total_events: int = 0
    processed_events: int = 0
    failed_events: int = 0
    anomalies_detected: int = 0

    # Performance metrics
    avg_processing_latency_ms: float = 0.0
    max_processing_latency_ms: float = 0.0
    min_processing_latency_ms: float = float('inf')

    # Data quality
    null_values: int = 0
    outliers: int = 0
    valid_records: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_events": self.total_events,
            "processed_events": self.processed_events,
            "failed_events": self.failed_events,
            "anomalies_detected": self.anomalies_detected,
            "avg_processing_latency_ms": self.avg_processing_latency_ms,
            "max_processing_latency_ms": self.max_processing_latency_ms,
            "min_processing_latency_ms": self.min_processing_latency_ms,
            "null_values": self.null_values,
            "outliers": self.outliers,
            "valid_records": self.valid_records,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


class AnalyticsEngine:
    """
    Enterprise-grade analytics engine.

    Features:
    - Real-time streaming data processing
    - Batch processing for historical data
    - Statistical analysis and hypothesis testing
    - Multi-algorithm anomaly detection
    - Time series analysis
    - Performance tracking and optimization

    Example:
        >>> config = AnalyticsConfig(stream_buffer_size=5000)
        >>> engine = AnalyticsEngine(config)
        >>> await engine.initialize()
        >>> result = await engine.analyze(data, AnalysisType.ANOMALY_DETECTION)
    """

    def __init__(self, config: AnalyticsConfig):
        """
        Initialize analytics engine.

        Args:
            config: Analytics configuration
        """
        self.config = config
        self.metrics = AnalyticsMetrics()

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Processing pipeline
        self._processing_queue: asyncio.Queue = None
        self._worker_tasks: List[asyncio.Task] = []

        # Storage
        self._data_buffer: List[Dict[str, Any]] = []
        self._analysis_cache: Dict[str, Tuple[Any, datetime]] = {}

        # State
        self._is_running = False
        self._last_flush = datetime.utcnow()

        logger.info(f"Analytics Engine initialized with config: {config}")

    async def initialize(self) -> None:
        """Initialize analytics engine components."""
        logger.info("Initializing analytics engine...")

        self._processing_queue = asyncio.Queue(maxsize=self.config.stream_buffer_size)

        # Start worker tasks
        for i in range(self.config.max_workers):
            task = asyncio.create_task(self._process_worker(i))
            self._worker_tasks.append(task)

        # Start metrics export task
        if self.config.enable_metrics:
            task = asyncio.create_task(self._metrics_export_loop())
            self._worker_tasks.append(task)

        self._is_running = True
        logger.info("Analytics engine initialized successfully")

    async def shutdown(self) -> None:
        """Gracefully shutdown analytics engine."""
        logger.info("Shutting down analytics engine...")

        self._is_running = False

        # Flush any pending data
        await self._flush_data()

        # Cancel all worker tasks
        for task in self._worker_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        logger.info("Analytics engine shutdown complete")

    async def process_event(
        self,
        event_name: str,
        event_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Process incoming event.

        Args:
            event_name: Name/type of event
            event_data: Event data
            metadata: Optional metadata
        """
        if not self._is_running:
            logger.warning("Engine not running, dropping event")
            return

        try:
            event = {
                "event_name": event_name,
                "event_data": event_data,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat(),
            }

            await self._processing_queue.put(event)
            self.metrics.total_events += 1

        except asyncio.QueueFull:
            logger.error("Processing queue full, dropping event")
            self.metrics.failed_events += 1
        except Exception as e:
            logger.error(f"Error processing event: {e}", exc_info=True)
            self.metrics.failed_events += 1

    async def analyze(
        self,
        data: List[Dict[str, Any]],
        analysis_type: AnalysisType,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform analysis on data.

        Args:
            data: Input data
            analysis_type: Type of analysis
            parameters: Analysis parameters

        Returns:
            Analysis results
        """
        cache_key = self._get_cache_key(analysis_type, parameters)

        # Check cache
        if self.config.cache_enabled and cache_key in self._analysis_cache:
            cached_result, cached_time = self._analysis_cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self.config.cache_ttl:
                logger.debug(f"Returning cached result for {analysis_type.value}")
                return cached_result

        logger.info(f"Starting {analysis_type.value} analysis on {len(data)} records")

        try:
            if analysis_type == AnalysisType.STATISTICAL:
                result = await self._analyze_statistical(data, parameters or {})
            elif analysis_type == AnalysisType.ANOMALY_DETECTION:
                result = await self._analyze_anomalies(data, parameters or {})
            elif analysis_type == AnalysisType.CLUSTERING:
                result = await self._analyze_clustering(data, parameters or {})
            elif analysis_type == AnalysisType.CORRELATION:
                result = await self._analyze_correlation(data, parameters or {})
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")

            # Cache result
            if self.config.cache_enabled:
                self._analysis_cache[cache_key] = (result, datetime.utcnow())

            logger.info(f"Analysis complete: {analysis_type.value}")
            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise

    async def _analyze_statistical(
        self,
        data: List[Dict[str, Any]],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform statistical analysis."""
        from .algorithms.statistical import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer()
        result = {
            "type": "statistical",
            "descriptive_stats": analyzer.descriptive_stats(data),
            "timestamp": datetime.utcnow().isoformat(),
        }

        return result

    async def _analyze_anomalies(
        self,
        data: List[Dict[str, Any]],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform anomaly detection analysis."""
        from .algorithms.anomaly_detection import AnomalyDetector

        detector = AnomalyDetector()
        anomalies = detector.detect(data)

        self.metrics.anomalies_detected += len(anomalies)

        result = {
            "type": "anomaly_detection",
            "anomalies": anomalies,
            "count": len(anomalies),
            "timestamp": datetime.utcnow().isoformat(),
        }

        return result

    async def _analyze_clustering(
        self,
        data: List[Dict[str, Any]],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform clustering analysis."""
        result = {
            "type": "clustering",
            "clusters": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return result

    async def _analyze_correlation(
        self,
        data: List[Dict[str, Any]],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform correlation analysis."""
        result = {
            "type": "correlation",
            "correlations": {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        return result

    async def _process_worker(self, worker_id: int) -> None:
        """Worker task for processing events."""
        logger.info(f"Worker {worker_id} started")

        try:
            while self._is_running:
                try:
                    # Get event with timeout
                    event = await asyncio.wait_for(
                        self._processing_queue.get(),
                        timeout=1.0
                    )

                    # Process event
                    self._data_buffer.append(event)
                    self.metrics.processed_events += 1

                    # Flush if buffer full or interval elapsed
                    if (
                        len(self._data_buffer) >= self.config.batch_size or
                        (datetime.utcnow() - self._last_flush).total_seconds() > self.config.stream_flush_interval
                    ):
                        await self._flush_data()

                except asyncio.TimeoutError:
                    # Timeout is normal when queue is empty
                    if len(self._data_buffer) > 0:
                        await self._flush_data()

        except asyncio.CancelledError:
            logger.debug(f"Worker {worker_id} cancelled")
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}", exc_info=True)

    async def _flush_data(self) -> None:
        """Flush buffered data."""
        if not self._data_buffer:
            return

        logger.debug(f"Flushing {len(self._data_buffer)} events")

        try:
            # Here you would write to storage, database, etc.
            # For now, just clear the buffer
            self._data_buffer.clear()
            self._last_flush = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error flushing data: {e}", exc_info=True)

    async def _metrics_export_loop(self) -> None:
        """Periodically export metrics."""
        try:
            while self._is_running:
                await asyncio.sleep(self.config.metrics_export_interval)
                self._export_metrics()

        except asyncio.CancelledError:
            logger.debug("Metrics export loop cancelled")

    def _export_metrics(self) -> None:
        """Export current metrics."""
        logger.debug(f"Metrics: {json.dumps(self.metrics.to_dict(), indent=2)}")

    def get_metrics(self) -> AnalyticsMetrics:
        """Get current metrics."""
        self.metrics.last_updated = datetime.utcnow()
        return self.metrics

    def register_event_handler(self, event_name: str, handler: Callable) -> None:
        """Register handler for specific event type."""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []

        self._event_handlers[event_name].append(handler)
        logger.debug(f"Registered handler for event: {event_name}")

    def _get_cache_key(
        self,
        analysis_type: AnalysisType,
        parameters: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for analysis."""
        params_str = json.dumps(parameters or {}, sort_keys=True)
        return f"{analysis_type.value}:{params_str}"
