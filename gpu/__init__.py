"""
GPU Acceleration Module for Negative Space Imaging Project
Provides GPU-accelerated processing for imaging and machine learning tasks
"""

from gpu.acceleration import GPUAccelerator
from gpu.image_processing import GPUImageProcessor
from gpu.utils import get_gpu_info, setup_gpu_memory

__version__ = "0.1.0"
__all__ = [
    "GPUAccelerator",
    "GPUImageProcessor",
    "get_gpu_info",
    "setup_gpu_memory",
]
