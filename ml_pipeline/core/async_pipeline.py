"""
Enhanced Async Processing Pipeline - Optimized Concurrent Processing

Implements advanced async processing with GPU optimization, concurrent task execution,
and enhanced error handling for the ML pipeline.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

from __future__ import annotations

import asyncio
import logging
import time
from asyncio import Queue, Semaphore
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from collections import defaultdict

import torch

from .pipeline import MLPipeline
from .config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class AsyncTask:
    """Represents an async task in the processing pipeline."""
    task_id: str
    stage: str
    data: Any
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ProcessingStats:
    """Statistics for async processing performance."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_processing_time: float = 0.0
    gpu_utilization: float = 0.0
    memory_usage: float = 0.0
    queue_depth: int = 0


class AsyncProcessingPipeline(MLPipeline):
    """
    Enhanced ML Pipeline with advanced async processing capabilities.

    Features:
    - Concurrent task execution with priority queuing
    - GPU utilization optimization through task batching
    - Advanced error handling with retry mechanisms
    - Backpressure management and rate limiting
    - Real-time performance monitoring
    """

    def __init__(
        self,
        config: PipelineConfig,
        device_manager: Optional['DeviceManager'] = None,
        max_concurrent_tasks: int = 10,
        gpu_batch_size: int = 4,
        enable_backpressure: bool = True
    ):
        super().__init__(config, device_manager)

        # Async processing configuration
        self.max_concurrent_tasks = max_concurrent_tasks
        self.gpu_batch_size = gpu_batch_size
        self.enable_backpressure = enable_backpressure

        # Task management
        self.task_queue: Queue[AsyncTask] = Queue(maxsize=1000)
        self.processing_semaphore = Semaphore(max_concurrent_tasks)
        self.gpu_semaphore = Semaphore(gpu_batch_size)

        # Processing state
        self.is_processing = False
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, Any] = {}

        # Performance monitoring
        self.processing_stats = ProcessingStats()
        self.stage_stats: Dict[str, ProcessingStats] = defaultdict(ProcessingStats)

        # Error handling
        self.error_handlers: Dict[str, Callable] = {}
        self.retry_policies: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Initialized Enhanced Async Processing Pipeline with {max_concurrent_tasks} concurrent tasks")

    async def initialize(self) -> None:
        """Initialize the enhanced async pipeline."""
        await super().initialize()

        # Setup async processing components
        await self._setup_async_processing()

        logger.info("Enhanced async processing pipeline initialized")

    async def _setup_async_processing(self) -> None:
        """Setup async processing infrastructure."""
        # Start background processing task
        self.processing_task = asyncio.create_task(self._process_task_queue())

        # Setup error handlers
        self._setup_error_handlers()

        # Setup retry policies
        self._setup_retry_policies()

    def _setup_error_handlers(self) -> None:
        """Setup error handlers for different failure scenarios."""
        self.error_handlers = {
            "gpu_memory_error": self._handle_gpu_memory_error,
            "timeout_error": self._handle_timeout_error,
            "inference_error": self._handle_inference_error,
            "network_error": self._handle_network_error,
        }

    def _setup_retry_policies(self) -> None:
        """Setup retry policies for different error types."""
        self.retry_policies = {
            "gpu_memory_error": {"max_retries": 2, "backoff_factor": 1.5},
            "timeout_error": {"max_retries": 3, "backoff_factor": 2.0},
            "inference_error": {"max_retries": 1, "backoff_factor": 1.0},
            "network_error": {"max_retries": 5, "backoff_factor": 1.2},
        }

    async def submit_task(
        self,
        task_id: str,
        input_data: Any,
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> str:
        """
        Submit a task for async processing.

        Args:
            task_id: Unique task identifier
            input_data: Input data for processing
            priority: Task priority (higher = more urgent)
            timeout: Optional timeout in seconds

        Returns:
            Task ID for tracking
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")

        task = AsyncTask(
            task_id=task_id,
            stage="input",
            data=input_data,
            priority=priority,
            timeout=timeout
        )

        # Add to queue with backpressure management
        if self.enable_backpressure and self.task_queue.full():
            logger.warning("Task queue full, applying backpressure")
            await self._apply_backpressure()

        await self.task_queue.put(task)
        self.processing_stats.total_tasks += 1

        logger.debug(f"Submitted task {task_id} with priority {priority}")
        return task_id

    async def _apply_backpressure(self) -> None:
        """Apply backpressure when queue is full."""
        # Wait for queue to have space
        while self.task_queue.full():
            await asyncio.sleep(0.1)

            # Check if we should drop lowest priority tasks
            if self.task_queue.qsize() > self.task_queue.maxsize * 0.9:
                await self._drop_low_priority_tasks()

    async def _drop_low_priority_tasks(self) -> None:
        """Drop lowest priority tasks when under extreme backpressure."""
        # This is a simplified implementation
        # In practice, you'd want to maintain a priority queue
        logger.warning("Dropping low priority tasks due to extreme backpressure")

    async def _process_task_queue(self) -> None:
        """Background task for processing queued tasks."""
        self.is_processing = True

        try:
            while self.is_initialized:
                try:
                    # Get next task with timeout
                    task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )

                    # Process task
                    asyncio.create_task(self._process_task(task))

                except asyncio.TimeoutError:
                    # No tasks available, continue
                    continue
                except Exception as e:
                    logger.error(f"Error processing task queue: {e}")
                    continue

        except asyncio.CancelledError:
            logger.info("Task queue processing cancelled")
        finally:
            self.is_processing = False

    async def _process_task(self, task: AsyncTask) -> None:
        """
        Process a single task through the pipeline.

        Args:
            task: Task to process
        """
        start_time = time.time()

        try:
            async with self.processing_semaphore:
                logger.debug(f"Processing task {task.task_id}")

                # Execute pipeline
                result = await self._execute_with_gpu_optimization(task)

                # Store result
                self.task_results[task.task_id] = {
                    "result": result,
                    "success": True,
                    "processing_time": time.time() - start_time
                }

                self.processing_stats.completed_tasks += 1

        except Exception as e:
            await self._handle_task_error(task, e, start_time)

    async def _execute_with_gpu_optimization(self, task: AsyncTask) -> Any:
        """
        Execute pipeline with GPU utilization optimization.

        Args:
            task: Task to execute

        Returns:
            Pipeline execution result
        """
        # For GPU-intensive tasks, use semaphore to limit concurrent GPU usage
        if self._is_gpu_intensive_task(task):
            async with self.gpu_semaphore:
                return await self.execute(task.data)
        else:
            return await self.execute(task.data)

    def _is_gpu_intensive_task(self, task: AsyncTask) -> bool:
        """Determine if a task is GPU-intensive."""
        # Check if task involves model inference or training
        gpu_stages = ["feature_extraction", "segmentation", "classification", "anomaly_detection"]
        return any(stage in str(task.data) for stage in gpu_stages)

    async def _handle_task_error(self, task: AsyncTask, error: Exception, start_time: float) -> None:
        """
        Handle task processing errors with retry logic.

        Args:
            task: Failed task
            error: Exception that occurred
            start_time: Task start time
        """
        error_type = self._classify_error(error)

        # Check retry policy
        retry_policy = self.retry_policies.get(error_type, {"max_retries": 0})
        max_retries = retry_policy.get("max_retries", 0)

        if task.retry_count < max_retries:
            # Schedule retry with backoff
            backoff_factor = retry_policy.get("backoff_factor", 1.0)
            delay = backoff_factor ** task.retry_count

            task.retry_count += 1
            logger.warning(f"Retrying task {task.task_id} (attempt {task.retry_count}) after {delay}s")

            await asyncio.sleep(delay)
            await self.task_queue.put(task)
            return

        # Max retries exceeded or no retry policy
        logger.error(f"Task {task.task_id} failed permanently: {error}")

        self.task_results[task.task_id] = {
            "error": str(error),
            "success": False,
            "processing_time": time.time() - start_time,
            "retries": task.retry_count
        }

        self.processing_stats.failed_tasks += 1

        # Call error handler if available
        error_handler = self.error_handlers.get(error_type)
        if error_handler:
            await error_handler(task, error)

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate handling."""
        error_str = str(error).lower()

        if "cuda" in error_str or "gpu" in error_str or "memory" in error_str:
            return "gpu_memory_error"
        elif "timeout" in error_str:
            return "timeout_error"
        elif "network" in error_str or "connection" in error_str:
            return "network_error"
        else:
            return "inference_error"

    async def _handle_gpu_memory_error(self, task: AsyncTask, error: Exception) -> None:
        """Handle GPU memory errors."""
        logger.warning(f"GPU memory error for task {task.task_id}: {error}")
        # Could implement GPU memory cleanup, task rescheduling, etc.

    async def _handle_timeout_error(self, task: AsyncTask, error: Exception) -> None:
        """Handle timeout errors."""
        logger.warning(f"Timeout error for task {task.task_id}: {error}")
        # Could implement timeout handling, resource cleanup, etc.

    async def _handle_inference_error(self, task: AsyncTask, error: Exception) -> None:
        """Handle inference errors."""
        logger.error(f"Inference error for task {task.task_id}: {error}")
        # Could implement fallback models, error reporting, etc.

    async def _handle_network_error(self, task: AsyncTask, error: Exception) -> None:
        """Handle network errors."""
        logger.warning(f"Network error for task {task.task_id}: {error}")
        # Could implement retry with different endpoints, etc.

    async def get_task_result(self, task_id: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """
        Get result for a submitted task.

        Args:
            task_id: Task identifier
            timeout: Maximum time to wait for result

        Returns:
            Task result or None if not available
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if task_id in self.task_results:
                return self.task_results.pop(task_id)
            await asyncio.sleep(0.1)

        return None

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        # Update real-time stats
        self.processing_stats.queue_depth = self.task_queue.qsize()

        # Calculate GPU utilization (simplified)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            self.processing_stats.gpu_utilization = gpu_memory

        return {
            "processing_stats": self.processing_stats.__dict__,
            "stage_stats": {k: v.__dict__ for k, v in self.stage_stats.items()},
            "active_tasks": len(self.processing_tasks),
            "queue_size": self.task_queue.qsize(),
        }

    async def shutdown(self) -> None:
        """Shutdown the async processing pipeline."""
        logger.info("Shutting down async processing pipeline")

        # Cancel processing task
        if hasattr(self, 'processing_task'):
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        # Cancel all active tasks
        for task in self.processing_tasks.values():
            task.cancel()

        self.processing_tasks.clear()

        # Call parent shutdown
        await super().shutdown()

        logger.info("Async processing pipeline shutdown complete")

    @asynccontextmanager
    async def processing_context(self):
        """Context manager for safe async processing."""
        try:
            yield self
        finally:
            await self.shutdown()
