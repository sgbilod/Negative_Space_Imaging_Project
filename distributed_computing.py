#!/usr/bin/env python
"""
Distributed Computing Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

Provides distributed computing capabilities for the Negative Space Imaging system,
including parallel processing, task distribution, and cluster coordination.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ComputeBackend(Enum):
    """Available compute backends."""
    MULTIPROCESSING = "multiprocessing"
    THREADING = "threading"
    RAY = "ray"
    DASK = "dask"
    LOCAL = "local"


@dataclass
class TaskResult:
    """Result of a distributed task."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    worker_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "worker_id": self.worker_id,
            "timestamp": self.timestamp,
        }


@dataclass
class DistributedConfig:
    """Configuration for distributed computing."""
    backend: ComputeBackend = ComputeBackend.MULTIPROCESSING
    num_workers: int = -1  # -1 means auto-detect
    max_tasks_per_worker: int = 100
    timeout_seconds: int = 3600
    retry_count: int = 3

    def __post_init__(self) -> None:
        """Set default number of workers."""
        if self.num_workers == -1:
            self.num_workers = multiprocessing.cpu_count()


class DistributedExecutor:
    """
    Distributed task executor.

    Provides a unified interface for executing tasks across multiple
    workers using various backends.

    Example:
        >>> executor = DistributedExecutor()
        >>> results = executor.map(my_function, data_list)
    """

    def __init__(self, config: Optional[DistributedConfig] = None):
        """
        Initialize the distributed executor.

        Args:
            config: Configuration for the executor
        """
        self.config = config or DistributedConfig()
        self._executor: Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]] = None
        self._initialized = False

        self._setup_executor()

    def _setup_executor(self) -> None:
        """Set up the executor based on backend."""
        if self.config.backend == ComputeBackend.MULTIPROCESSING:
            self._executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
        elif self.config.backend == ComputeBackend.THREADING:
            self._executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        else:
            self._executor = ProcessPoolExecutor(max_workers=self.config.num_workers)

        self._initialized = True
        logger.info(
            f"Distributed executor initialized: {self.config.backend.value} "
            f"with {self.config.num_workers} workers"
        )

    def map(
        self,
        func: Callable[[T], Any],
        items: List[T],
        chunksize: int = 1
    ) -> List[Any]:
        """
        Apply a function to items in parallel.

        Args:
            func: Function to apply
            items: List of items to process
            chunksize: Number of items per task

        Returns:
            List of results
        """
        if not self._initialized or self._executor is None:
            return [func(item) for item in items]

        results = list(self._executor.map(func, items, chunksize=chunksize))
        return results

    def submit(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any
    ) -> "asyncio.Future":
        """
        Submit a task for execution.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future object
        """
        if not self._initialized or self._executor is None:
            raise RuntimeError("Executor not initialized")

        return self._executor.submit(func, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.

        Args:
            wait: Whether to wait for pending tasks
        """
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._initialized = False
            logger.info("Distributed executor shut down")


class DataPartitioner:
    """
    Utilities for partitioning data for distributed processing.
    """

    @staticmethod
    def partition_array(
        data: np.ndarray,
        num_partitions: int
    ) -> List[np.ndarray]:
        """
        Partition a numpy array into chunks.

        Args:
            data: Array to partition
            num_partitions: Number of partitions

        Returns:
            List of array chunks
        """
        return np.array_split(data, num_partitions)

    @staticmethod
    def partition_list(
        items: List[T],
        num_partitions: int
    ) -> List[List[T]]:
        """
        Partition a list into chunks.

        Args:
            items: List to partition
            num_partitions: Number of partitions

        Returns:
            List of list chunks
        """
        chunk_size = max(1, len(items) // num_partitions)
        return [
            items[i:i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]

    @staticmethod
    def partition_by_size(
        data: np.ndarray,
        max_size_mb: float
    ) -> List[np.ndarray]:
        """
        Partition array based on memory size.

        Args:
            data: Array to partition
            max_size_mb: Maximum size per partition in MB

        Returns:
            List of array chunks
        """
        total_size_mb = data.nbytes / (1024 * 1024)
        num_partitions = max(1, int(total_size_mb / max_size_mb))
        return np.array_split(data, num_partitions)


class ParallelImageProcessor:
    """
    Parallel image processing utilities.

    Optimized for processing multiple images or image regions
    in parallel.
    """

    def __init__(
        self,
        num_workers: Optional[int] = None,
        backend: ComputeBackend = ComputeBackend.MULTIPROCESSING
    ):
        """
        Initialize parallel processor.

        Args:
            num_workers: Number of worker processes
            backend: Compute backend to use
        """
        config = DistributedConfig(
            backend=backend,
            num_workers=num_workers or multiprocessing.cpu_count(),
        )
        self.executor = DistributedExecutor(config)

    def process_images(
        self,
        images: List[np.ndarray],
        processor: Callable[[np.ndarray], np.ndarray]
    ) -> List[np.ndarray]:
        """
        Process multiple images in parallel.

        Args:
            images: List of images to process
            processor: Processing function

        Returns:
            List of processed images
        """
        return self.executor.map(processor, images)

    def process_regions(
        self,
        image: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
        processor: Callable[[np.ndarray], Any]
    ) -> List[Any]:
        """
        Process image regions in parallel.

        Args:
            image: Source image
            regions: List of (x, y, width, height) regions
            processor: Processing function

        Returns:
            List of results
        """
        def process_region(region: Tuple[int, int, int, int]) -> Any:
            x, y, w, h = region
            region_data = image[y:y+h, x:x+w]
            return processor(region_data)

        return self.executor.map(process_region, regions)

    def shutdown(self) -> None:
        """Shutdown the processor."""
        self.executor.shutdown()


class ClusterCoordinator:
    """
    Coordinator for distributed cluster operations.

    Manages task distribution, worker health, and result aggregation.
    """

    def __init__(self, config: Optional[DistributedConfig] = None):
        """
        Initialize cluster coordinator.

        Args:
            config: Distributed computing configuration
        """
        self.config = config or DistributedConfig()
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.pending_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}

    def register_worker(
        self,
        worker_id: str,
        capabilities: Dict[str, Any]
    ) -> None:
        """
        Register a worker node.

        Args:
            worker_id: Unique worker identifier
            capabilities: Worker capabilities
        """
        self.workers[worker_id] = {
            "id": worker_id,
            "capabilities": capabilities,
            "status": "idle",
            "tasks_completed": 0,
            "registered_at": datetime.utcnow().isoformat(),
        }
        logger.info(f"Worker registered: {worker_id}")

    def unregister_worker(self, worker_id: str) -> None:
        """
        Unregister a worker node.

        Args:
            worker_id: Worker identifier
        """
        if worker_id in self.workers:
            del self.workers[worker_id]
            logger.info(f"Worker unregistered: {worker_id}")

    def submit_task(
        self,
        task_id: str,
        task_data: Dict[str, Any]
    ) -> None:
        """
        Submit a task for execution.

        Args:
            task_id: Unique task identifier
            task_data: Task data and parameters
        """
        self.pending_tasks[task_id] = {
            "id": task_id,
            "data": task_data,
            "status": "pending",
            "submitted_at": datetime.utcnow().isoformat(),
        }
        logger.debug(f"Task submitted: {task_id}")

    def complete_task(
        self,
        task_id: str,
        result: TaskResult
    ) -> None:
        """
        Mark a task as completed.

        Args:
            task_id: Task identifier
            result: Task result
        """
        if task_id in self.pending_tasks:
            del self.pending_tasks[task_id]

        self.completed_tasks[task_id] = result
        logger.debug(f"Task completed: {task_id}")

    def get_available_workers(self) -> List[str]:
        """Get list of available worker IDs."""
        return [
            w_id for w_id, w_info in self.workers.items()
            if w_info["status"] == "idle"
        ]

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        return {
            "total_workers": len(self.workers),
            "available_workers": len(self.get_available_workers()),
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": len(self.completed_tasks),
            "workers": list(self.workers.values()),
        }


# Convenience function for parallel processing
def parallel_map(
    func: Callable[[T], Any],
    items: List[T],
    num_workers: Optional[int] = None
) -> List[Any]:
    """
    Apply a function to items in parallel.

    Args:
        func: Function to apply
        items: Items to process
        num_workers: Number of worker processes

    Returns:
        List of results
    """
    config = DistributedConfig(num_workers=num_workers or -1)
    executor = DistributedExecutor(config)

    try:
        return executor.map(func, items)
    finally:
        executor.shutdown()


def get_optimal_workers() -> int:
    """Get optimal number of workers for current system."""
    return max(1, multiprocessing.cpu_count() - 1)


# Type alias for regions
Tuple = tuple
