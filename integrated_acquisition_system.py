#!/usr/bin/env python
"""
Integrated Acquisition System
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides a unified acquisition system that coordinates multiple
image acquisition methods including camera, SFTP, and API sources.

Features:
- Unified interface for all acquisition sources
- Queue-based processing with prioritization
- Automatic source selection and failover
- Real-time status monitoring
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class AcquisitionSourceType(Enum):
    """Types of acquisition sources."""
    CAMERA = auto()
    SFTP = auto()
    HTTP_API = auto()
    LOCAL_FILE = auto()
    SIMULATION = auto()


class AcquisitionPriority(Enum):
    """Priority levels for acquisition requests."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class AcquisitionStatus(Enum):
    """Status of an acquisition request."""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AcquisitionRequest:
    """Represents a single acquisition request."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_type: AcquisitionSourceType = AcquisitionSourceType.SIMULATION
    source_config: Dict[str, Any] = field(default_factory=dict)
    priority: AcquisitionPriority = AcquisitionPriority.NORMAL
    status: AcquisitionStatus = AcquisitionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def __lt__(self, other: "AcquisitionRequest") -> bool:
        """Compare by priority for queue ordering."""
        return self.priority.value > other.priority.value


@dataclass
class AcquisitionResult:
    """Result of an acquisition operation."""
    request_id: str
    success: bool
    image_data: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding image data)."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "has_image": self.image_data is not None,
            "image_shape": list(self.image_data.shape) if self.image_data is not None else None,
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
        }


class AcquisitionSource(ABC):
    """Abstract base class for acquisition sources."""

    source_type: AcquisitionSourceType

    @abstractmethod
    async def acquire(self, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Acquire image from source.
        
        Args:
            config: Source-specific configuration
            
        Returns:
            Tuple of (image_data, metadata)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if source is available."""
        pass


class CameraSource(AcquisitionSource):
    """Camera acquisition source."""

    source_type = AcquisitionSourceType.CAMERA

    async def acquire(self, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Acquire image from camera."""
        # Import here to avoid circular imports
        from image_acquisition import ImageAcquisition, ImageFormat, AcquisitionMode

        device_id = config.get("device_id", "0")
        resolution = config.get("resolution", (1920, 1080))
        exposure = config.get("exposure_ms", 100)

        acquisition = ImageAcquisition(
            format=ImageFormat.RAW,
            mode=AcquisitionMode.CAMERA,
        )

        image_data, metadata = acquisition.acquire(
            source=device_id,
            resolution=resolution,
            exposure=exposure,
        )

        # Convert bytes to numpy array if needed
        if isinstance(image_data, bytes):
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image_array = image_array.reshape(resolution[1], resolution[0], -1)
        else:
            image_array = image_data

        return image_array, metadata

    def is_available(self) -> bool:
        """Check if camera is available."""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            available = cap.isOpened()
            cap.release()
            return available
        except Exception:
            return False


class SFTPSource(AcquisitionSource):
    """SFTP acquisition source."""

    source_type = AcquisitionSourceType.SFTP

    async def acquire(self, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Acquire image from SFTP server."""
        from image_acquisition import ImageAcquisition, ImageFormat, AcquisitionMode

        hostname = config.get("hostname")
        username = config.get("username")
        password = config.get("password")
        remote_path = config.get("remote_path")

        acquisition = ImageAcquisition(
            format=ImageFormat.RAW,
            mode=AcquisitionMode.REMOTE_SFTP,
        )

        image_data, metadata = acquisition.acquire(
            source=remote_path,
            hostname=hostname,
            username=username,
            password=password,
        )

        if isinstance(image_data, bytes):
            # Try to decode image
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_data))
            image_array = np.array(img)
        else:
            image_array = image_data

        return image_array, metadata

    def is_available(self) -> bool:
        """Check if SFTP is available (paramiko installed)."""
        try:
            import paramiko
            return True
        except ImportError:
            return False


class HTTPAPISource(AcquisitionSource):
    """HTTP API acquisition source."""

    source_type = AcquisitionSourceType.HTTP_API

    async def acquire(self, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Acquire image from HTTP API."""
        import aiohttp
        from PIL import Image
        import io

        url = config.get("url")
        headers = config.get("headers", {})
        auth = config.get("auth")

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, auth=auth) as response:
                response.raise_for_status()
                image_bytes = await response.read()

        img = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(img)

        metadata = {
            "source": url,
            "content_type": response.content_type,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return image_array, metadata

    def is_available(self) -> bool:
        """Check if HTTP client is available."""
        try:
            import aiohttp
            return True
        except ImportError:
            return False


class LocalFileSource(AcquisitionSource):
    """Local file acquisition source."""

    source_type = AcquisitionSourceType.LOCAL_FILE

    async def acquire(self, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Acquire image from local file."""
        from image_formats import read_image

        file_path = config.get("path")
        if not file_path:
            raise ValueError("File path is required")

        image_array, metadata = read_image(file_path)

        return image_array, metadata.to_dict()

    def is_available(self) -> bool:
        """Local file source is always available."""
        return True


class SimulationSource(AcquisitionSource):
    """Simulation acquisition source for testing."""

    source_type = AcquisitionSourceType.SIMULATION

    async def acquire(self, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate simulated image."""
        width = config.get("width", 512)
        height = config.get("height", 512)
        pattern = config.get("pattern", "random")

        if pattern == "random":
            image_array = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        elif pattern == "gradient":
            x = np.linspace(0, 255, width, dtype=np.uint8)
            y = np.linspace(0, 255, height, dtype=np.uint8)
            xx, yy = np.meshgrid(x, y)
            image_array = ((xx.astype(np.float32) + yy) / 2).astype(np.uint8)
        elif pattern == "negative_space":
            # Create pattern with dark regions (negative space)
            image_array = np.random.randint(120, 200, (height, width), dtype=np.uint8)
            # Add dark circular regions
            for _ in range(3):
                cx, cy = np.random.randint(50, width-50), np.random.randint(50, height-50)
                radius = np.random.randint(20, 80)
                y_grid, x_grid = np.ogrid[:height, :width]
                mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= radius**2
                image_array[mask] = np.random.randint(10, 60)
        else:
            image_array = np.zeros((height, width), dtype=np.uint8)

        metadata = {
            "source": "simulation",
            "pattern": pattern,
            "width": width,
            "height": height,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return image_array, metadata

    def is_available(self) -> bool:
        """Simulation source is always available."""
        return True


class IntegratedAcquisitionSystem:
    """
    Unified acquisition system for coordinating multiple image sources.
    
    Provides:
    - Multi-source acquisition with automatic selection
    - Priority-based request queuing
    - Concurrent processing
    - Status monitoring and callbacks
    """

    def __init__(
        self,
        max_workers: int = 4,
        queue_size: int = 100,
    ):
        """
        Initialize the acquisition system.
        
        Args:
            max_workers: Maximum concurrent acquisition workers
            queue_size: Maximum queue size
        """
        self.max_workers = max_workers
        self._request_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_size)
        self._results: Dict[str, AcquisitionResult] = {}
        self._requests: Dict[str, AcquisitionRequest] = {}
        self._sources: Dict[AcquisitionSourceType, AcquisitionSource] = {}
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._callbacks: List[Callable[[AcquisitionResult], None]] = []
        self._lock = threading.Lock()

        # Register default sources
        self._register_default_sources()

        logger.info(f"Acquisition system initialized with {max_workers} workers")

    def _register_default_sources(self) -> None:
        """Register default acquisition sources."""
        self.register_source(CameraSource())
        self.register_source(SFTPSource())
        self.register_source(HTTPAPISource())
        self.register_source(LocalFileSource())
        self.register_source(SimulationSource())

    def register_source(self, source: AcquisitionSource) -> None:
        """Register an acquisition source."""
        self._sources[source.source_type] = source
        logger.info(f"Registered source: {source.source_type.name}")

    def get_available_sources(self) -> List[AcquisitionSourceType]:
        """Get list of available acquisition sources."""
        return [
            source_type
            for source_type, source in self._sources.items()
            if source.is_available()
        ]

    def submit_request(
        self,
        source_type: AcquisitionSourceType,
        source_config: Dict[str, Any],
        priority: AcquisitionPriority = AcquisitionPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit an acquisition request.
        
        Args:
            source_type: Type of acquisition source
            source_config: Configuration for the source
            priority: Request priority
            metadata: Additional metadata
            
        Returns:
            Request ID
        """
        request = AcquisitionRequest(
            source_type=source_type,
            source_config=source_config,
            priority=priority,
            metadata=metadata or {},
        )

        with self._lock:
            self._requests[request.request_id] = request
            self._request_queue.put(request)

        request.status = AcquisitionStatus.QUEUED
        logger.info(f"Request {request.request_id} queued with priority {priority.name}")

        return request.request_id

    def get_request_status(self, request_id: str) -> Optional[AcquisitionStatus]:
        """Get status of a request."""
        request = self._requests.get(request_id)
        return request.status if request else None

    def get_result(self, request_id: str) -> Optional[AcquisitionResult]:
        """Get result for a completed request."""
        return self._results.get(request_id)

    def register_callback(
        self,
        callback: Callable[[AcquisitionResult], None],
    ) -> None:
        """Register a callback for completed acquisitions."""
        self._callbacks.append(callback)

    async def _process_request(self, request: AcquisitionRequest) -> AcquisitionResult:
        """Process a single acquisition request."""
        request.status = AcquisitionStatus.IN_PROGRESS
        request.started_at = datetime.utcnow()

        source = self._sources.get(request.source_type)
        if source is None:
            return AcquisitionResult(
                request_id=request.request_id,
                success=False,
                error_message=f"Unknown source type: {request.source_type}",
            )

        if not source.is_available():
            return AcquisitionResult(
                request_id=request.request_id,
                success=False,
                error_message=f"Source not available: {request.source_type.name}",
            )

        start_time = time.time()

        try:
            image_data, metadata = await source.acquire(request.source_config)

            processing_time = time.time() - start_time

            result = AcquisitionResult(
                request_id=request.request_id,
                success=True,
                image_data=image_data,
                metadata={**request.metadata, **metadata},
                processing_time=processing_time,
            )

            request.status = AcquisitionStatus.COMPLETED

        except Exception as e:
            logger.error(f"Acquisition error for {request.request_id}: {e}")
            result = AcquisitionResult(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
            )
            request.status = AcquisitionStatus.FAILED
            request.error_message = str(e)

        request.completed_at = datetime.utcnow()
        return result

    async def _worker(self) -> None:
        """Worker coroutine for processing requests."""
        while self._running:
            try:
                request = self._request_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue

            result = await self._process_request(request)

            with self._lock:
                self._results[request.request_id] = result

            # Call callbacks
            for callback in self._callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            self._request_queue.task_done()

    async def start(self) -> None:
        """Start the acquisition system."""
        if self._running:
            return

        self._running = True
        
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker())
            self._workers.append(task)

        logger.info(f"Acquisition system started with {len(self._workers)} workers")

    async def stop(self) -> None:
        """Stop the acquisition system."""
        self._running = False

        for task in self._workers:
            task.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info("Acquisition system stopped")

    async def acquire(
        self,
        source_type: AcquisitionSourceType,
        source_config: Dict[str, Any],
        timeout: float = 60.0,
    ) -> AcquisitionResult:
        """
        Acquire image synchronously.
        
        Args:
            source_type: Type of acquisition source
            source_config: Configuration for the source
            timeout: Timeout in seconds
            
        Returns:
            Acquisition result
        """
        request = AcquisitionRequest(
            source_type=source_type,
            source_config=source_config,
        )

        return await asyncio.wait_for(
            self._process_request(request),
            timeout=timeout,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get acquisition statistics."""
        with self._lock:
            total_requests = len(self._requests)
            completed = sum(
                1 for r in self._requests.values()
                if r.status == AcquisitionStatus.COMPLETED
            )
            failed = sum(
                1 for r in self._requests.values()
                if r.status == AcquisitionStatus.FAILED
            )
            pending = sum(
                1 for r in self._requests.values()
                if r.status in (AcquisitionStatus.PENDING, AcquisitionStatus.QUEUED)
            )

        return {
            "total_requests": total_requests,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "queue_size": self._request_queue.qsize(),
            "workers": len(self._workers),
            "running": self._running,
        }


# Convenience function for simple acquisitions
async def acquire_image(
    source_type: AcquisitionSourceType = AcquisitionSourceType.SIMULATION,
    **config: Any,
) -> AcquisitionResult:
    """
    Simple acquisition function.
    
    Args:
        source_type: Type of acquisition source
        **config: Source configuration
        
    Returns:
        Acquisition result
    """
    system = IntegratedAcquisitionSystem(max_workers=1)
    return await system.acquire(source_type, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def demo():
        print("Integrated Acquisition System Demo")
        print("=" * 40)

        system = IntegratedAcquisitionSystem()
        await system.start()

        try:
            # Test simulation source
            print("\nAcquiring simulated image...")
            result = await system.acquire(
                AcquisitionSourceType.SIMULATION,
                {"width": 256, "height": 256, "pattern": "negative_space"},
            )

            print(f"Success: {result.success}")
            if result.image_data is not None:
                print(f"Image shape: {result.image_data.shape}")
            print(f"Processing time: {result.processing_time:.3f}s")

            # Show statistics
            stats = system.get_statistics()
            print(f"\nStatistics: {stats}")

        finally:
            await system.stop()

    asyncio.run(demo())
