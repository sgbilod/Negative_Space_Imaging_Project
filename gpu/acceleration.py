"""
GPU Acceleration Framework for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import numpy as np
import torch
import torch.cuda
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("GPUAcceleration")


class GPUManager:
    """Manages GPU resources and acceleration."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device_ids: Optional[List[int]] = None
    ):
        self.config = config or {}
        self.device_ids = device_ids or self._get_available_devices()
        self.current_device = 0

        # Initialize GPU state
        self._initialize_gpu()

        # Track GPU usage
        self.usage_stats = {
            device_id: {
                'memory_used': 0,
                'compute_used': 0,
                'jobs_processed': 0
            }
            for device_id in self.device_ids
        }

    def _get_available_devices(self) -> List[int]:
        """Get available GPU devices."""
        try:
            device_count = torch.cuda.device_count()
            return list(range(device_count))

        except Exception as e:
            logger.error(f"Failed to get GPU devices: {e}")
            return []

    def _initialize_gpu(self):
        """Initialize GPU system."""
        try:
            if not self.device_ids:
                raise RuntimeError("No GPU devices available")

            # Set up CUDA devices
            for device_id in self.device_ids:
                device = torch.device(f'cuda:{device_id}')
                torch.cuda.set_device(device)

                # Clear cache
                torch.cuda.empty_cache()

                # Test device
                test_tensor = torch.zeros(1).to(device)
                del test_tensor

            logger.info(f"Initialized {len(self.device_ids)} GPU devices")

        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            raise

    def get_next_device(self) -> torch.device:
        """Get next available GPU device."""
        try:
            device_id = self.device_ids[self.current_device]
            self.current_device = (
                self.current_device + 1
            ) % len(self.device_ids)

            return torch.device(f'cuda:{device_id}')

        except Exception as e:
            logger.error(f"Failed to get next device: {e}")
            raise

    def allocate_memory(
        self,
        size_bytes: int,
        device_id: Optional[int] = None
    ) -> torch.Tensor:
        """Allocate GPU memory."""
        try:
            if device_id is None:
                device_id = self.device_ids[self.current_device]

            device = torch.device(f'cuda:{device_id}')
            tensor = torch.empty(size_bytes, device=device)

            # Update usage stats
            self.usage_stats[device_id]['memory_used'] += size_bytes

            return tensor

        except Exception as e:
            logger.error(f"Memory allocation failed: {e}")
            raise

    def free_memory(
        self,
        tensor: torch.Tensor,
        device_id: Optional[int] = None
    ):
        """Free GPU memory."""
        try:
            if device_id is None:
                device_id = tensor.device.index

            size_bytes = tensor.element_size() * tensor.nelement()

            # Update usage stats
            self.usage_stats[device_id]['memory_used'] -= size_bytes

            # Free memory
            del tensor
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Memory free failed: {e}")

    def get_memory_info(
        self,
        device_id: Optional[int] = None
    ) -> Dict[str, int]:
        """Get GPU memory information."""
        try:
            if device_id is None:
                device_id = self.device_ids[self.current_device]

            return {
                'total': torch.cuda.get_device_properties(
                    device_id
                ).total_memory,
                'allocated': torch.cuda.memory_allocated(device_id),
                'cached': torch.cuda.memory_reserved(device_id)
            }

        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {}

    def optimize_memory(self, device_id: Optional[int] = None):
        """Optimize GPU memory usage."""
        try:
            if device_id is None:
                device_id = self.device_ids[self.current_device]

            # Clear cache
            torch.cuda.empty_cache()

            # Force garbage collection
            import gc
            gc.collect()

            logger.info(f"Optimized memory for device {device_id}")

        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")

    def process_batch(
        self,
        data: Union[np.ndarray, torch.Tensor],
        processor: callable,
        batch_size: int = 32,
        device_id: Optional[int] = None
    ) -> List[Any]:
        """Process data batch on GPU."""
        try:
            if device_id is None:
                device_id = self.device_ids[self.current_device]

            device = torch.device(f'cuda:{device_id}')

            # Convert to tensor if needed
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)

            results = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size].to(device)

                # Process batch
                result = processor(batch)

                # Move result to CPU
                if isinstance(result, torch.Tensor):
                    result = result.cpu()

                results.append(result)

                # Update stats
                self.usage_stats[device_id]['jobs_processed'] += 1

            return results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise

    def parallel_process(
        self,
        data_chunks: List[Union[np.ndarray, torch.Tensor]],
        processor: callable
    ) -> List[Any]:
        """Process data chunks in parallel across GPUs."""
        try:
            results = []
            for i, chunk in enumerate(data_chunks):
                device_id = self.device_ids[i % len(self.device_ids)]
                device = torch.device(f'cuda:{device_id}')

                # Convert to tensor if needed
                if isinstance(chunk, np.ndarray):
                    chunk = torch.from_numpy(chunk)

                # Process on GPU
                chunk = chunk.to(device)
                result = processor(chunk)

                # Move result to CPU
                if isinstance(result, torch.Tensor):
                    result = result.cpu()

                results.append(result)

                # Update stats
                self.usage_stats[device_id]['jobs_processed'] += 1

            return results

        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            raise

    def get_device_stats(
        self,
        device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get device statistics."""
        try:
            if device_id is None:
                device_id = self.device_ids[self.current_device]

            properties = torch.cuda.get_device_properties(device_id)
            memory_info = self.get_memory_info(device_id)

            return {
                'name': properties.name,
                'compute_capability': (
                    properties.major,
                    properties.minor
                ),
                'total_memory': properties.total_memory,
                'memory_usage': memory_info,
                'usage_stats': self.usage_stats[device_id]
            }

        except Exception as e:
            logger.error(f"Failed to get device stats: {e}")
            return {}

    def cleanup(self):
        """Clean up GPU resources."""
        try:
            # Clear memory on all devices
            for device_id in self.device_ids:
                torch.cuda.set_device(device_id)
                torch.cuda.empty_cache()

            # Reset usage stats
            for device_id in self.device_ids:
                self.usage_stats[device_id] = {
                    'memory_used': 0,
                    'compute_used': 0,
                    'jobs_processed': 0
                }

            logger.info("GPU resources cleaned up")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Example usage
if __name__ == "__main__":
    import time

    # Create GPU manager
    manager = GPUManager()

    # Example data processing
    data = np.random.randn(1000, 1000)

    def process_func(batch):
        # Simulate processing
        time.sleep(0.1)
        return batch * 2

    # Process in batches
    results = manager.process_batch(
        data,
        process_func,
        batch_size=100
    )

    # Get stats
    for device_id in manager.device_ids:
        stats = manager.get_device_stats(device_id)
        print(f"Device {device_id} stats:", stats)

    # Cleanup
    manager.cleanup()
