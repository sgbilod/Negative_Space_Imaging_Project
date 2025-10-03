"""
GPU Utilities Module for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import torch
import torch.cuda
import numpy as np
from typing import Dict, List, Any, Optional, Union
from contextlib import contextmanager
from time import perf_counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("GPUUtils")


class DeviceContext:
    """Context manager for GPU device selection."""

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.prev_device = None

    def __enter__(self):
        self.prev_device = torch.cuda.current_device()
        torch.cuda.set_device(self.device_id)
        return torch.device(f'cuda:{self.device_id}')

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.set_device(self.prev_device)


@contextmanager
def gpu_timer(name: str = "Operation"):
    """Context manager for timing GPU operations."""
    torch.cuda.synchronize()
    start = perf_counter()

    try:
        yield
    finally:
        torch.cuda.synchronize()
        end = perf_counter()
        logger.info(f"{name} took {end - start:.4f} seconds")


def check_gpu_requirements(
    min_memory_gb: float = 2.0,
    min_compute_capability: tuple = (3, 5)
) -> bool:
    """Check if available GPUs meet requirements."""
    try:
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available")
            return False

        device_count = torch.cuda.device_count()
        if device_count == 0:
            logger.warning("No GPU devices found")
            return False

        for device_id in range(device_count):
            props = torch.cuda.get_device_properties(device_id)

            # Check memory
            memory_gb = props.total_memory / (1024 ** 3)
            if memory_gb < min_memory_gb:
                logger.warning(
                    f"Device {device_id} has insufficient memory: "
                    f"{memory_gb:.1f}GB < {min_memory_gb}GB"
                )
                return False

            # Check compute capability
            capability = (props.major, props.minor)
            if capability < min_compute_capability:
                logger.warning(
                    f"Device {device_id} has insufficient compute "
                    f"capability: {capability} < {min_compute_capability}"
                )
                return False

        return True

    except Exception as e:
        logger.error(f"Error checking GPU requirements: {e}")
        return False


def memory_status(
    device_id: Optional[int] = None
) -> Dict[str, float]:
    """Get detailed memory status for a device."""
    try:
        if device_id is None:
            device_id = torch.cuda.current_device()

        props = torch.cuda.get_device_properties(device_id)
        total = props.total_memory
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        free = total - reserved

        return {
            'total_gb': total / (1024 ** 3),
            'allocated_gb': allocated / (1024 ** 3),
            'reserved_gb': reserved / (1024 ** 3),
            'free_gb': free / (1024 ** 3),
            'utilization': allocated / total * 100
        }

    except Exception as e:
        logger.error(f"Error getting memory status: {e}")
        return {}


def optimize_tensor_memory(
    data: Union[np.ndarray, torch.Tensor],
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Optimize tensor memory usage."""
    try:
        if isinstance(data, np.ndarray):
            # Convert numpy array to torch tensor
            tensor = torch.from_numpy(data)
        else:
            tensor = data

        if dtype is None:
            # Automatically choose optimal dtype
            if tensor.dtype in [torch.float64, torch.double]:
                dtype = torch.float32
            elif tensor.dtype == torch.int64:
                dtype = torch.int32

        if dtype is not None:
            tensor = tensor.to(dtype)

        return tensor

    except Exception as e:
        logger.error(f"Error optimizing tensor memory: {e}")
        return data


def split_for_devices(
    data: Union[np.ndarray, torch.Tensor],
    num_devices: Optional[int] = None
) -> List[torch.Tensor]:
    """Split data for processing across multiple devices."""
    try:
        if num_devices is None:
            num_devices = torch.cuda.device_count()

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        # Calculate split sizes
        total_size = len(data)
        base_size = total_size // num_devices
        remainder = total_size % num_devices

        # Create split sizes list
        split_sizes = [base_size] * num_devices
        for i in range(remainder):
            split_sizes[i] += 1

        # Split data
        chunks = torch.split(data, split_sizes)
        return list(chunks)

    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return [data]


def benchmark_device(
    device_id: int,
    size: int = 1000
) -> Dict[str, float]:
    """Benchmark GPU device performance."""
    try:
        with DeviceContext(device_id):
            device = torch.device(f'cuda:{device_id}')
            results = {}

            # Memory transfer benchmark
            with gpu_timer("Memory transfer"):
                cpu_data = torch.randn(size, size)
                start = perf_counter()
                gpu_data = cpu_data.to(device)
                gpu_data.cpu()
                end = perf_counter()
                results['memory_transfer_ms'] = (end - start) * 1000

            # Compute benchmark
            with gpu_timer("Matrix multiply"):
                start = perf_counter()
                result = torch.matmul(gpu_data, gpu_data)
                torch.cuda.synchronize()
                end = perf_counter()
                results['matrix_multiply_ms'] = (end - start) * 1000

            # Memory bandwidth
            with gpu_timer("Memory bandwidth"):
                start = perf_counter()
                for _ in range(10):
                    result = gpu_data * 2
                torch.cuda.synchronize()
                end = perf_counter()

                bytes_processed = (
                    size * size * 4 * 10  # 4 bytes per float
                )
                bandwidth = bytes_processed / (end - start) / 1e9  # GB/s
                results['memory_bandwidth_gbs'] = bandwidth

            return results

    except Exception as e:
        logger.error(f"Error benchmarking device: {e}")
        return {}


def get_optimal_device() -> torch.device:
    """Get the optimal GPU device based on current usage."""
    try:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("No GPU devices available")

        # Get memory usage for all devices
        memory_usage = []
        for device_id in range(device_count):
            status = memory_status(device_id)
            memory_usage.append(
                (device_id, status.get('utilization', 100))
            )

        # Sort by utilization
        memory_usage.sort(key=lambda x: x[1])
        optimal_device = memory_usage[0][0]

        return torch.device(f'cuda:{optimal_device}')

    except Exception as e:
        logger.error(f"Error getting optimal device: {e}")
        return torch.device('cuda:0')


def async_stream() -> torch.cuda.Stream:
    """Create an asynchronous CUDA stream."""
    try:
        return torch.cuda.Stream(priority=-1)
    except Exception as e:
        logger.error(f"Error creating async stream: {e}")
        return torch.cuda.current_stream()


@contextmanager
def use_stream(stream: Optional[torch.cuda.Stream] = None):
    """Context manager for using a CUDA stream."""
    if stream is None:
        stream = async_stream()

    try:
        with torch.cuda.stream(stream):
            yield stream
    except Exception as e:
        logger.error(f"Error using stream: {e}")
        yield torch.cuda.current_stream()


# Example usage
if __name__ == "__main__":
    # Check GPU requirements
    if check_gpu_requirements(min_memory_gb=4.0):
        print("GPU requirements met")

        # Get optimal device
        device = get_optimal_device()
        print(f"Optimal device: {device}")

        # Benchmark device
        results = benchmark_device(device.index)
        print("Benchmark results:", results)

        # Test async operations
        with use_stream() as stream:
            data = torch.randn(1000, 1000, device=device)
            result = data @ data.T
            print("Async operation complete")
