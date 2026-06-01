"""
Test suite for GPU Utilities
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import pytest
import torch
import numpy as np
from gpu.utils import (
    DeviceContext,
    gpu_timer,
    check_gpu_requirements,
    memory_status,
    optimize_tensor_memory,
    split_for_devices,
    benchmark_device,
    get_optimal_device,
    async_stream,
    use_stream
)


def test_device_context():
    """Test DeviceContext context manager."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    initial_device = torch.cuda.current_device()
    test_device = 0 if initial_device == 1 else 1

    with DeviceContext(test_device):
        assert torch.cuda.current_device() == test_device

    assert torch.cuda.current_device() == initial_device


def test_gpu_timer():
    """Test GPU timer context manager."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    with gpu_timer("Test operation"):
        # Simulate GPU operation
        torch.randn(1000, 1000, device='cuda')


def test_check_gpu_requirements():
    """Test GPU requirements check."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Test with reasonable requirements
    assert check_gpu_requirements(
        min_memory_gb=1.0,
        min_compute_capability=(3, 0)
    )

    # Test with impossible requirements
    assert not check_gpu_requirements(
        min_memory_gb=1000000.0  # Unrealistic memory requirement
    )


def test_memory_status():
    """Test memory status reporting."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    status = memory_status(0)
    assert isinstance(status, dict)
    assert 'total_gb' in status
    assert 'allocated_gb' in status
    assert 'reserved_gb' in status
    assert 'free_gb' in status
    assert 'utilization' in status


def test_optimize_tensor_memory():
    """Test tensor memory optimization."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Test with numpy array
    np_data = np.random.randn(100, 100).astype(np.float64)
    optimized = optimize_tensor_memory(np_data)
    assert isinstance(optimized, torch.Tensor)
    assert optimized.dtype == torch.float32

    # Test with torch tensor
    torch_data = torch.randn(100, 100, dtype=torch.float64)
    optimized = optimize_tensor_memory(torch_data)
    assert optimized.dtype == torch.float32


def test_split_for_devices():
    """Test data splitting for multiple devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    data = torch.randn(100, 100)
    num_devices = torch.cuda.device_count()

    chunks = split_for_devices(data, num_devices)
    assert len(chunks) == num_devices

    # Check total elements preserved
    total_elements = sum(chunk.numel() for chunk in chunks)
    assert total_elements == data.numel()


def test_benchmark_device():
    """Test device benchmarking."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device_id = 0
    results = benchmark_device(device_id, size=100)

    assert isinstance(results, dict)
    assert 'memory_transfer_ms' in results
    assert 'matrix_multiply_ms' in results
    assert 'memory_bandwidth_gbs' in results


def test_get_optimal_device():
    """Test optimal device selection."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = get_optimal_device()
    assert isinstance(device, torch.device)
    assert device.type == 'cuda'


def test_async_stream():
    """Test async stream creation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stream = async_stream()
    assert isinstance(stream, torch.cuda.Stream)


def test_use_stream():
    """Test stream context manager."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    with use_stream() as stream:
        assert isinstance(stream, torch.cuda.Stream)
        assert torch.cuda.current_stream().cuda_stream == (
            stream.cuda_stream
        )


@pytest.mark.stress
def test_stress_memory_optimization():
    """Stress test memory optimization."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    initial_memory = memory_status(0)

    # Create and optimize multiple large tensors
    tensors = []
    for _ in range(10):
        data = np.random.randn(1000, 1000).astype(np.float64)
        tensor = optimize_tensor_memory(data)
        tensors.append(tensor)

    # Clear tensors
    tensors.clear()
    torch.cuda.empty_cache()

    final_memory = memory_status(0)
    assert final_memory['allocated_gb'] <= initial_memory['allocated_gb']


@pytest.mark.stress
def test_stress_parallel_processing():
    """Stress test parallel processing with multiple streams."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    num_streams = 4
    streams = [async_stream() for _ in range(num_streams)]
    results = []

    # Process data in parallel streams
    for stream in streams:
        with use_stream(stream):
            data = torch.randn(1000, 1000, device='cuda')
            result = torch.matmul(data, data.T)
            results.append(result)

    # Synchronize all streams
    torch.cuda.synchronize()

    assert len(results) == num_streams
    for result in results:
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()
