"""
GPU Acceleration Module for Negative Space Imaging

High-performance GPU utilities with:
- CUDA kernel wrappers
- Mixed precision training utilities
- Gradient accumulation helpers
- GPU memory profiling
- Compute profiling decorators
- Device management
- Multi-GPU synchronization helpers

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.cuda as cuda
from torch.cuda.amp import GradScaler, autocast

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUMemoryProfiler:
    """Profile GPU memory usage."""

    def __init__(self):
        """Initialize memory profiler."""
        self.memory_snapshots = []

    def capture_snapshot(self, label: str = "") -> Dict[str, float]:
        """Capture current GPU memory state."""
        if not cuda.is_available():
            return {}

        snapshot = {
            "allocated_mb": cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": cuda.memory_reserved() / 1024 / 1024,
            "label": label,
            "timestamp": time.time(),
        }

        self.memory_snapshots.append(snapshot)
        return snapshot

    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        return max((s["allocated_mb"] for s in self.memory_snapshots), default=0.0)

    def get_summary(self) -> Dict[str, float]:
        """Get memory profiling summary."""
        if not self.memory_snapshots:
            return {}

        allocated = [s["allocated_mb"] for s in self.memory_snapshots]
        return {
            "peak_allocated_mb": max(allocated),
            "min_allocated_mb": min(allocated),
            "avg_allocated_mb": np.mean(allocated),
            "num_snapshots": len(self.memory_snapshots),
        }

    def print_summary(self) -> None:
        """Print memory profiling summary."""
        summary = self.get_summary()
        if summary:
            logger.info(f"GPU Memory Profile: {summary}")


class ComputeProfiler:
    """Profile compute performance."""

    def __init__(self):
        """Initialize compute profiler."""
        self.event_timings = {}

    def start_event(self, name: str) -> None:
        """Start timing an event."""
        if not cuda.is_available():
            return

        cuda.synchronize()
        event = cuda.Event(enable_timing=True)
        event.record()
        self.event_timings[name] = {"start": event, "end": None}

    def end_event(self, name: str) -> float:
        """End timing an event."""
        if not cuda.is_available() or name not in self.event_timings:
            return 0.0

        cuda.synchronize()
        event = cuda.Event(enable_timing=True)
        event.record()
        self.event_timings[name]["end"] = event

        start_event = self.event_timings[name]["start"]
        end_event = self.event_timings[name]["end"]

        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        return elapsed_time

    def get_timings(self) -> Dict[str, float]:
        """Get all event timings."""
        timings = {}
        for name, events in self.event_timings.items():
            if events["end"] is not None:
                elapsed = events["start"].elapsed_time(events["end"]) / 1000.0
                timings[name] = elapsed
        return timings


def gpu_profile_decorator(func: Callable) -> Callable:
    """Decorator for profiling GPU computation."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = ComputeProfiler()
        memory_profiler = GPUMemoryProfiler()

        memory_profiler.capture_snapshot("start")
        profiler.start_event(func.__name__)

        try:
            result = func(*args, **kwargs)
        finally:
            elapsed = profiler.end_event(func.__name__)
            memory_profiler.capture_snapshot("end")

            logger.info(f"{func.__name__} elapsed time: {elapsed:.4f}s")
            logger.info(f"Memory stats: {memory_profiler.get_summary()}")

        return result

    return wrapper


class MixedPrecisionTrainer:
    """Mixed precision training utilities."""

    def __init__(self, enabled: bool = True):
        """Initialize mixed precision trainer."""
        self.enabled = enabled and cuda.is_available()
        self.scaler = GradScaler() if self.enabled else None

    def autocast_context(self):
        """Get autocast context manager."""
        if self.enabled:
            return autocast()
        else:
            return NullContext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        if self.enabled and self.scaler:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Step optimizer with loss scaling."""
        if self.enabled and self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def get_scaler(self) -> Optional[GradScaler]:
        """Get GradScaler instance."""
        return self.scaler if self.enabled else None


class GradientAccumulator:
    """Helper for gradient accumulation."""

    def __init__(self, accumulation_steps: int = 1, max_grad_norm: float = 1.0):
        """Initialize gradient accumulator."""
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0

    def should_accumulate(self) -> bool:
        """Check if should accumulate gradients."""
        return (self.step_count + 1) % self.accumulation_steps != 0

    def should_step(self) -> bool:
        """Check if should step optimizer."""
        return (self.step_count + 1) % self.accumulation_steps == 0

    def increment(self) -> None:
        """Increment accumulation step."""
        self.step_count += 1

    def reset(self) -> None:
        """Reset accumulation counter."""
        self.step_count = 0

    def clip_gradients(self, model: torch.nn.Module) -> float:
        """Clip model gradients."""
        if self.max_grad_norm > 0:
            return torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        return 0.0


class DeviceManager:
    """Manage GPU/CPU device allocation."""

    def __init__(self):
        """Initialize device manager."""
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.num_gpus = cuda.device_count()

    def get_device(self) -> torch.device:
        """Get primary device."""
        return self.device

    def get_num_gpus(self) -> int:
        """Get number of GPUs."""
        return self.num_gpus

    def get_device_name(self, device_id: int = 0) -> str:
        """Get GPU device name."""
        if device_id < self.num_gpus:
            return cuda.get_device_name(device_id)
        return "CPU"

    def get_memory_info(self, device_id: int = 0) -> Dict[str, float]:
        """Get GPU memory info."""
        if not cuda.is_available() or device_id >= self.num_gpus:
            return {}

        props = cuda.get_device_properties(device_id)
        allocated = cuda.memory_allocated(device_id) / 1024 / 1024
        reserved = cuda.memory_reserved(device_id) / 1024 / 1024
        total = props.total_memory / 1024 / 1024

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "total_mb": total,
            "free_mb": total - reserved,
        }

    def print_device_info(self) -> None:
        """Print device information."""
        logger.info(f"Primary device: {self.device}")
        logger.info(f"Number of GPUs: {self.num_gpus}")

        for i in range(self.num_gpus):
            name = self.get_device_name(i)
            memory_info = self.get_memory_info(i)
            logger.info(f"GPU {i}: {name}")
            logger.info(f"  Memory: {memory_info['allocated_mb']:.0f}MB / {memory_info['total_mb']:.0f}MB")

    def synchronize(self) -> None:
        """Synchronize all GPUs."""
        if cuda.is_available():
            cuda.synchronize()

    def empty_cache(self) -> None:
        """Empty GPU cache."""
        if cuda.is_available():
            cuda.empty_cache()


class MultiGPUSynchronizer:
    """Handle multi-GPU synchronization."""

    def __init__(self, num_gpus: int = 1):
        """Initialize synchronizer."""
        self.num_gpus = num_gpus or cuda.device_count()
        self.backend = "nccl" if self.num_gpus > 1 and cuda.is_available() else None

    def synchronize_tensors(self, tensors: List[torch.Tensor], operation: str = "mean"):
        """Synchronize tensors across GPUs."""
        if self.num_gpus <= 1 or not self.backend:
            return tensors

        synchronized = []
        for tensor in tensors:
            if operation == "mean":
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                tensor /= self.num_gpus
            elif operation == "sum":
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            synchronized.append(tensor)

        return synchronized

    def broadcast_model(self, model: torch.nn.Module, src_gpu: int = 0) -> None:
        """Broadcast model to all GPUs."""
        if self.num_gpus <= 1 or not self.backend:
            return

        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=src_gpu)


class KernelWrapper:
    """Wrapper for custom CUDA kernels."""

    @staticmethod
    def custom_forward(input_tensor: torch.Tensor, kernel_name: str) -> torch.Tensor:
        """Execute custom CUDA kernel forward pass."""
        # Placeholder for custom kernel execution
        logger.debug(f"Executing custom kernel: {kernel_name}")
        return input_tensor

    @staticmethod
    def custom_backward(grad_output: torch.Tensor, kernel_name: str) -> torch.Tensor:
        """Execute custom CUDA kernel backward pass."""
        logger.debug(f"Executing custom kernel backward: {kernel_name}")
        return grad_output


class NullContext:
    """Null context manager for consistency."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def get_gpu_memory_summary() -> Dict[str, float]:
    """Get GPU memory summary."""
    if not cuda.is_available():
        return {}

    summary = {}
    for i in range(cuda.device_count()):
        props = cuda.get_device_properties(i)
        allocated = cuda.memory_allocated(i) / 1024 / 1024
        reserved = cuda.memory_reserved(i) / 1024 / 1024
        total = props.total_memory / 1024 / 1024

        summary[f"gpu_{i}_allocated_mb"] = allocated
        summary[f"gpu_{i}_reserved_mb"] = reserved
        summary[f"gpu_{i}_total_mb"] = total

    return summary


def optimize_gpu_settings() -> None:
    """Optimize GPU settings for training."""
    if not cuda.is_available():
        return

    cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    logger.info("GPU optimization settings applied")


@gpu_profile_decorator
def benchmark_gpu_throughput(
    batch_size: int = 32,
    input_shape: Tuple[int, ...] = (3, 224, 224),
    num_iterations: int = 100,
) -> Dict[str, float]:
    """Benchmark GPU throughput."""
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    dummy_input = torch.randn(batch_size, *input_shape, device=device)

    if cuda.is_available():
        cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = dummy_input * 2

    if cuda.is_available():
        cuda.synchronize()

    elapsed = time.time() - start_time
    throughput = (batch_size * num_iterations) / elapsed

    return {
        "throughput_samples_per_sec": throughput,
        "total_time": elapsed,
        "batch_size": batch_size,
        "num_iterations": num_iterations,
    }
