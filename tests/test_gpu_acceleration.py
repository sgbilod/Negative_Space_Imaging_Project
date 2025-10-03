"""
Test suite for GPU Acceleration Framework
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import pytest
import torch
import numpy as np
from gpu.acceleration import GPUManager


@pytest.fixture
def gpu_manager():
    """Fixture to create GPU manager instance."""
    manager = GPUManager()
    yield manager
    manager.cleanup()


def test_gpu_initialization(gpu_manager):
    """Test GPU manager initialization."""
    assert len(gpu_manager.device_ids) > 0, "No GPU devices found"
    assert gpu_manager.current_device == 0


def test_device_cycling(gpu_manager):
    """Test cycling through available devices."""
    initial_device = gpu_manager.current_device
    device = gpu_manager.get_next_device()

    assert isinstance(device, torch.device)
    assert device.type == 'cuda'
    assert gpu_manager.current_device != initial_device


def test_memory_allocation(gpu_manager):
    """Test GPU memory allocation."""
    size_bytes = 1024 * 1024  # 1MB
    tensor = gpu_manager.allocate_memory(size_bytes)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type == 'cuda'

    # Check usage stats
    device_id = tensor.device.index
    assert gpu_manager.usage_stats[device_id]['memory_used'] > 0

    # Cleanup
    gpu_manager.free_memory(tensor)


def test_memory_info(gpu_manager):
    """Test getting GPU memory information."""
    device_id = gpu_manager.device_ids[0]
    info = gpu_manager.get_memory_info(device_id)

    assert isinstance(info, dict)
    assert 'total' in info
    assert 'allocated' in info
    assert 'cached' in info


def test_batch_processing(gpu_manager):
    """Test processing data in batches."""
    data = np.random.randn(100, 100)

    def processor(batch):
        return batch * 2

    results = gpu_manager.process_batch(
        data,
        processor,
        batch_size=10
    )

    assert len(results) > 0
    assert isinstance(results[0], torch.Tensor)


def test_parallel_processing(gpu_manager):
    """Test parallel processing across GPUs."""
    data_chunks = [
        np.random.randn(50, 50) for _ in range(4)
    ]

    def processor(chunk):
        return chunk * 2

    results = gpu_manager.parallel_process(data_chunks, processor)

    assert len(results) == len(data_chunks)
    assert isinstance(results[0], torch.Tensor)


def test_device_stats(gpu_manager):
    """Test getting device statistics."""
    device_id = gpu_manager.device_ids[0]
    stats = gpu_manager.get_device_stats(device_id)

    assert isinstance(stats, dict)
    assert 'name' in stats
    assert 'compute_capability' in stats
    assert 'total_memory' in stats
    assert 'memory_usage' in stats
    assert 'usage_stats' in stats


def test_memory_optimization(gpu_manager):
    """Test memory optimization."""
    device_id = gpu_manager.device_ids[0]

    # Allocate some memory
    tensor = gpu_manager.allocate_memory(1024 * 1024)

    # Get initial memory state
    initial_memory = gpu_manager.get_memory_info(device_id)

    # Optimize memory
    gpu_manager.optimize_memory(device_id)

    # Get final memory state
    final_memory = gpu_manager.get_memory_info(device_id)

    # Cleanup
    gpu_manager.free_memory(tensor)

    assert final_memory['allocated'] <= initial_memory['allocated']


def test_cleanup(gpu_manager):
    """Test resource cleanup."""
    # Allocate some resources
    tensor = gpu_manager.allocate_memory(1024 * 1024)
    device_id = tensor.device.index

    # Record initial stats
    initial_stats = gpu_manager.usage_stats[device_id].copy()

    # Cleanup
    gpu_manager.cleanup()

    # Check final stats
    final_stats = gpu_manager.usage_stats[device_id]
    assert final_stats['memory_used'] == 0
    assert final_stats != initial_stats


@pytest.mark.stress
def test_stress_memory_management(gpu_manager):
    """Stress test memory management."""
    device_id = gpu_manager.device_ids[0]
    tensors = []

    try:
        # Repeatedly allocate and free memory
        for _ in range(100):
            size = np.random.randint(1024, 1024 * 1024)
            tensor = gpu_manager.allocate_memory(size, device_id)
            tensors.append(tensor)

            if len(tensors) > 10:
                gpu_manager.free_memory(tensors.pop(0))

    finally:
        # Cleanup
        for tensor in tensors:
            gpu_manager.free_memory(tensor)

    # Verify cleanup successful
    final_memory = gpu_manager.get_memory_info(device_id)
    assert final_memory['allocated'] == 0


@pytest.mark.stress
def test_stress_parallel_processing(gpu_manager):
    """Stress test parallel processing."""
    num_chunks = len(gpu_manager.device_ids) * 4
    data_chunks = [
        np.random.randn(1000, 1000) for _ in range(num_chunks)
    ]

    def processor(chunk):
        # Simulate intensive computation
        for _ in range(10):
            chunk = chunk @ chunk.T
        return chunk

    results = gpu_manager.parallel_process(data_chunks, processor)

    assert len(results) == num_chunks
    for result in results:
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()
