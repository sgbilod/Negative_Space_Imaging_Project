#!/usr/bin/env python
"""
Pro-Plus Feature Integration
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module integrates Pro-Plus features into the core system.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

import numpy as np

from pro_plus_activator import (
    ProPlusActivator,
    get_activator,
    is_pro_plus_active,
    require_pro_plus,
    LicenseType,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ProPlusFeatures:
    """
    Pro-Plus feature integration.
    
    Provides access to advanced features when Pro-Plus is active,
    with graceful fallback to basic features otherwise.
    """

    def __init__(self):
        self.activator = get_activator()

    @property
    def is_active(self) -> bool:
        """Check if Pro-Plus is active."""
        return self.activator.is_pro_plus()

    def get_feature_status(self) -> Dict[str, bool]:
        """Get status of all Pro-Plus features."""
        return {
            feature: self.activator.has_feature(feature)
            for feature in ProPlusActivator.PRO_PLUS_FEATURES
        }

    # Advanced Detection
    @require_pro_plus
    def advanced_detection(
        self,
        image: np.ndarray,
        sensitivity: float = 0.5,
        multi_scale: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform advanced negative space detection.
        
        Args:
            image: Input image array
            sensitivity: Detection sensitivity (0-1)
            multi_scale: Enable multi-scale analysis
            
        Returns:
            Detection results dictionary
        """
        logger.info("Running advanced detection (Pro-Plus)")
        
        # Multi-scale analysis
        scales = [1.0, 0.5, 0.25] if multi_scale else [1.0]
        all_regions = []
        
        for scale in scales:
            if scale != 1.0:
                from scipy.ndimage import zoom
                scaled = zoom(image, scale)
            else:
                scaled = image
            
            # Advanced thresholding
            threshold = np.percentile(scaled, sensitivity * 100)
            mask = scaled < threshold
            
            # Connected component analysis
            from scipy import ndimage
            labeled, num_features = ndimage.label(mask)
            
            for i in range(1, num_features + 1):
                region_mask = labeled == i
                area = np.sum(region_mask) / (scale ** 2)
                if area > 50:  # Minimum area threshold
                    all_regions.append({
                        "scale": scale,
                        "area": float(area),
                        "mean_intensity": float(np.mean(scaled[region_mask])),
                    })
        
        return {
            "regions": all_regions,
            "total_regions": len(all_regions),
            "method": "advanced_multi_scale",
        }

    # GPU Acceleration
    @require_pro_plus
    def gpu_accelerated_processing(
        self,
        image: np.ndarray,
        operation: str = "denoise",
    ) -> np.ndarray:
        """
        GPU-accelerated image processing.
        
        Args:
            image: Input image
            operation: Processing operation
            
        Returns:
            Processed image
        """
        logger.info(f"Running GPU-accelerated {operation}")
        
        try:
            import cupy as cp
            
            # Transfer to GPU
            gpu_image = cp.asarray(image)
            
            if operation == "denoise":
                from cupyx.scipy.ndimage import gaussian_filter
                result = gaussian_filter(gpu_image, sigma=1.0)
            elif operation == "edge_detect":
                from cupyx.scipy.ndimage import sobel
                result = sobel(gpu_image)
            else:
                result = gpu_image
            
            # Transfer back to CPU
            return cp.asnumpy(result)
            
        except ImportError:
            logger.warning("CuPy not available, falling back to CPU")
            return self._cpu_fallback(image, operation)

    def _cpu_fallback(self, image: np.ndarray, operation: str) -> np.ndarray:
        """CPU fallback for GPU operations."""
        from scipy.ndimage import gaussian_filter, sobel
        
        if operation == "denoise":
            return gaussian_filter(image, sigma=1.0)
        elif operation == "edge_detect":
            return sobel(image)
        return image

    # Batch Processing
    @require_pro_plus
    def unlimited_batch_process(
        self,
        images: List[np.ndarray],
        operation: Callable[[np.ndarray], np.ndarray],
        parallel: bool = True,
    ) -> List[np.ndarray]:
        """
        Process unlimited batch of images.
        
        Args:
            images: List of images to process
            operation: Processing function
            parallel: Enable parallel processing
            
        Returns:
            List of processed images
        """
        logger.info(f"Processing batch of {len(images)} images")
        
        if parallel:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(operation, images))
        else:
            results = [operation(img) for img in images]
        
        return results

    # Custom Model Support
    @require_pro_plus
    def load_custom_model(self, model_path: str) -> Any:
        """
        Load a custom trained model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading custom model from {model_path}")
        
        # Support various model formats
        if model_path.endswith(".pt") or model_path.endswith(".pth"):
            import torch
            return torch.load(model_path)
        elif model_path.endswith(".h5"):
            from tensorflow import keras
            return keras.models.load_model(model_path)
        elif model_path.endswith(".onnx"):
            import onnxruntime
            return onnxruntime.InferenceSession(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")


# Feature-specific decorators
def with_gpu_fallback(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator that provides CPU fallback for GPU functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            if is_pro_plus_active():
                return func(*args, **kwargs)
            else:
                logger.info("Pro-Plus not active, using basic implementation")
                return func(*args, use_gpu=False, **kwargs)
        except Exception as e:
            logger.warning(f"GPU processing failed, falling back to CPU: {e}")
            return func(*args, use_gpu=False, **kwargs)
    return wrapper


def feature_gate(feature_name: str, fallback: Optional[Callable] = None):
    """
    Decorator to gate features based on Pro-Plus status.
    
    Args:
        feature_name: Name of the Pro-Plus feature
        fallback: Optional fallback function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            activator = get_activator()
            if activator.has_feature(feature_name):
                return func(*args, **kwargs)
            elif fallback:
                logger.info(f"Feature {feature_name} not available, using fallback")
                return fallback(*args, **kwargs)
            else:
                raise PermissionError(
                    f"Feature '{feature_name}' requires Pro-Plus license"
                )
        return wrapper
    return decorator


# Convenience function to get Pro-Plus features
def get_pro_plus_features() -> ProPlusFeatures:
    """Get Pro-Plus features instance."""
    return ProPlusFeatures()


# Integration with core modules
def integrate_with_pipeline(pipeline: Any) -> None:
    """
    Integrate Pro-Plus features with the processing pipeline.
    
    Args:
        pipeline: Processing pipeline instance
    """
    if is_pro_plus_active():
        logger.info("Integrating Pro-Plus features with pipeline")
        features = get_pro_plus_features()
        
        # Enable advanced features
        if hasattr(pipeline, 'enable_gpu'):
            pipeline.enable_gpu = True
        if hasattr(pipeline, 'enable_advanced_detection'):
            pipeline.enable_advanced_detection = True
    else:
        logger.info("Pro-Plus not active, using basic pipeline features")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Pro-Plus Integration Module")
    print("=" * 40)

    features = get_pro_plus_features()
    
    print(f"\nPro-Plus Active: {features.is_active}")
    print("\nFeature Status:")
    for feature, available in features.get_feature_status().items():
        status = "✓" if available else "✗"
        print(f"  {status} {feature}")
