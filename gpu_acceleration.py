#!/usr/bin/env python
"""
GPU Acceleration Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

Provides GPU acceleration capabilities for the Negative Space Imaging system,
including memory management, batch processing, and parallel computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Check for GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_COUNT = torch.cuda.device_count()
        logger.info(f"GPU acceleration available: {GPU_COUNT} device(s)")
    else:
        GPU_COUNT = 0
        logger.info("GPU not available, using CPU fallback")
except ImportError:
    GPU_AVAILABLE = False
    GPU_COUNT = 0
    torch = None
    logger.warning("PyTorch not installed, GPU acceleration disabled")


@dataclass
class GPUDevice:
    """Information about a GPU device."""
    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory: int  # bytes
    current_memory_allocated: int = 0
    current_memory_cached: int = 0


@dataclass
class GPUStats:
    """Statistics for GPU usage."""
    memory_used: int = 0
    memory_allocated: int = 0
    operations_count: int = 0


class GPUManager:
    """
    GPU resource manager for negative space imaging operations.

    Manages GPU memory, device allocation, and batch processing
    for optimal performance.

    Example:
        >>> manager = GPUManager()
        >>> device = manager.get_next_device()
        >>> tensor = manager.allocate_memory(1024 * 1024)
        >>> manager.cleanup()
    """

    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        Initialize GPU manager.

        Args:
            device_ids: List of GPU device IDs to manage.
                       If None, all available GPUs are used.
        """
        self.device_ids: List[int] = []
        self.current_device: int = 0
        self.usage_stats: Dict[int, GPUStats] = {}
        self._initialized = False

        if not GPU_AVAILABLE:
            logger.warning("No GPU available, operations will use CPU")
            return

        if device_ids is None:
            self.device_ids = list(range(GPU_COUNT))
        else:
            self.device_ids = [d for d in device_ids if d < GPU_COUNT]

        if not self.device_ids:
            logger.warning("No valid GPU devices specified")
            return

        # Initialize stats for each device
        for device_id in self.device_ids:
            self.usage_stats[device_id] = GPUStats()

        self._initialized = True
        logger.info(f"GPU Manager initialized with devices: {self.device_ids}")

    def get_next_device(self) -> "torch.device":
        """
        Get the next available GPU device in round-robin fashion.

        Returns:
            PyTorch device object
        """
        if not self._initialized or not self.device_ids:
            return torch.device("cpu") if torch else None

        device_id = self.device_ids[self.current_device % len(self.device_ids)]
        self.current_device = (self.current_device + 1) % len(self.device_ids)

        return torch.device(f"cuda:{device_id}")

    def allocate_memory(
        self,
        size_bytes: int,
        device_id: Optional[int] = None,
        dtype: Any = None
    ) -> "torch.Tensor":
        """
        Allocate GPU memory.

        Args:
            size_bytes: Size in bytes to allocate
            device_id: Specific device to allocate on
            dtype: Data type for the tensor

        Returns:
            Allocated tensor on GPU
        """
        if not self._initialized:
            raise RuntimeError("GPU Manager not initialized")

        if device_id is None:
            device = self.get_next_device()
            device_id = device.index if device.type == "cuda" else 0
        else:
            device = torch.device(f"cuda:{device_id}")

        if dtype is None:
            dtype = torch.float32

        # Calculate number of elements based on dtype
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = size_bytes // element_size

        tensor = torch.empty(num_elements, dtype=dtype, device=device)

        # Update stats
        if device_id in self.usage_stats:
            self.usage_stats[device_id].memory_used += size_bytes
            self.usage_stats[device_id].memory_allocated += size_bytes
            self.usage_stats[device_id].operations_count += 1

        return tensor

    def free_memory(self, tensor: "torch.Tensor") -> None:
        """
        Free GPU memory associated with a tensor.

        Args:
            tensor: Tensor to free
        """
        if tensor is None:
            return

        device_id = tensor.device.index if tensor.device.type == "cuda" else 0
        size = tensor.numel() * tensor.element_size()

        del tensor

        if torch and GPU_AVAILABLE:
            torch.cuda.empty_cache()

        # Update stats
        if device_id in self.usage_stats:
            self.usage_stats[device_id].memory_used = max(
                0, self.usage_stats[device_id].memory_used - size
            )

    def get_memory_info(self, device_id: int) -> Dict[str, int]:
        """
        Get memory information for a device.

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary with memory statistics
        """
        if not GPU_AVAILABLE or device_id >= GPU_COUNT:
            return {"total": 0, "allocated": 0, "cached": 0, "free": 0}

        torch.cuda.synchronize(device_id)

        return {
            "total": torch.cuda.get_device_properties(device_id).total_memory,
            "allocated": torch.cuda.memory_allocated(device_id),
            "cached": torch.cuda.memory_reserved(device_id),
            "free": torch.cuda.get_device_properties(device_id).total_memory -
                   torch.cuda.memory_allocated(device_id),
        }

    def get_device_stats(self, device_id: int) -> Dict[str, Any]:
        """
        Get statistics for a GPU device.

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary with device statistics
        """
        if not GPU_AVAILABLE or device_id >= GPU_COUNT:
            return {"error": "Device not available"}

        props = torch.cuda.get_device_properties(device_id)
        memory_info = self.get_memory_info(device_id)

        return {
            "name": props.name,
            "compute_capability": (props.major, props.minor),
            "total_memory": props.total_memory,
            "memory_usage": memory_info,
            "usage_stats": self.usage_stats.get(device_id, GPUStats()).__dict__,
        }

    def process_batch(
        self,
        data: Union[np.ndarray, "torch.Tensor"],
        processor: Callable,
        batch_size: int = 32,
        device_id: Optional[int] = None
    ) -> List["torch.Tensor"]:
        """
        Process data in batches on GPU.

        Args:
            data: Input data array
            processor: Function to apply to each batch
            batch_size: Size of each batch
            device_id: GPU device to use

        Returns:
            List of processed batch results
        """
        if device_id is None:
            device = self.get_next_device()
        else:
            device = torch.device(f"cuda:{device_id}" if GPU_AVAILABLE else "cpu")

        # Convert to tensor if necessary
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(device)
        else:
            data = data.to(device)

        results = []
        num_batches = (len(data) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, len(data))
            batch = data[start:end]

            result = processor(batch)
            results.append(result)

        return results

    def parallel_process(
        self,
        data_chunks: List[Union[np.ndarray, "torch.Tensor"]],
        processor: Callable
    ) -> List["torch.Tensor"]:
        """
        Process data chunks in parallel across multiple GPUs.

        Args:
            data_chunks: List of data chunks to process
            processor: Function to apply to each chunk

        Returns:
            List of processed results
        """
        if not self._initialized or len(self.device_ids) == 0:
            # CPU fallback
            return [processor(chunk) for chunk in data_chunks]

        results = []

        for i, chunk in enumerate(data_chunks):
            device_id = self.device_ids[i % len(self.device_ids)]
            device = torch.device(f"cuda:{device_id}")

            if isinstance(chunk, np.ndarray):
                chunk = torch.from_numpy(chunk).to(device)
            else:
                chunk = chunk.to(device)

            result = processor(chunk)
            results.append(result)

        return results

    def optimize_memory(self, device_id: Optional[int] = None) -> None:
        """
        Optimize GPU memory by clearing caches.

        Args:
            device_id: Specific device to optimize, or all if None
        """
        if not GPU_AVAILABLE:
            return

        if device_id is not None:
            torch.cuda.synchronize(device_id)
        else:
            torch.cuda.synchronize()

        torch.cuda.empty_cache()
        logger.debug("GPU memory cache cleared")

    def cleanup(self) -> None:
        """Clean up all GPU resources."""
        if not GPU_AVAILABLE:
            return

        for device_id in self.device_ids:
            torch.cuda.synchronize(device_id)
            self.usage_stats[device_id] = GPUStats()

        torch.cuda.empty_cache()
        logger.info("GPU resources cleaned up")


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get information about all available GPUs.

    Returns:
        List of GPU information dictionaries
    """
    if not GPU_AVAILABLE:
        return []

    gpus = []
    for i in range(GPU_COUNT):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "id": i,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / (1024 ** 3),
            "multi_processor_count": props.multi_processor_count,
        })

    return gpus


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return GPU_AVAILABLE


def get_gpu_count() -> int:
    """Get number of available GPUs."""
    return GPU_COUNT
