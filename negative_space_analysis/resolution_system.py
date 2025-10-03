from typing import List
#!/usr/bin/env python
"""
Dynamic Resolution System
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements adaptive resolution handling for negative space analysis,
enabling robust analysis across multiple scales and detail levels.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import cv2
from scipy.ndimage import zoom
from skimage.transform import pyramid_gaussian

@dataclass
class ScaleParameters:
    """Parameters for multi-scale analysis."""
    scale_factor: float
    base_resolution: Tuple[int, int]
    detail_level: float  # Higher values indicate more detail preservation
    quality_score: float


class DynamicResolutionAnalyzer:
    """Handles adaptive resolution and scale analysis."""

    def __init__(
        self,
        min_resolution: Tuple[int, int] = (32, 32),
        max_resolution: Tuple[int, int] = (1024, 1024),
        num_scales: int = 4,
        detail_threshold: float = 0.1,
        device: Optional[torch.device] = None
    ):
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.num_scales = num_scales
        self.detail_threshold = detail_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize scale-space network
        self.scale_net = ScaleSpaceNetwork(
            base_channels=64,
            num_scales=num_scales
        ).to(self.device)

    def analyze_optimal_scale(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> ScaleParameters:
        """
        Determine optimal scale parameters for the given image region.

        Args:
            image: Input image
            mask: Optional region mask

        Returns:
            ScaleParameters object with optimal parameters
        """
        # Apply mask if provided
        if mask is not None:
            image = image * mask

        # Generate image pyramid
        pyramid = list(pyramid_gaussian(image, max_layer=self.num_scales - 1))

        # Convert pyramid to tensor
        pyramid_tensors = []
        for level in pyramid:
            tensor = torch.from_numpy(level).float()
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            pyramid_tensors.append(tensor.to(self.device))

        # Analyze scale space
        with torch.no_grad():
            scale_features = self.scale_net(pyramid_tensors)
            quality_scores = self._compute_quality_scores(scale_features)

        # Find optimal scale
        optimal_idx = torch.argmax(quality_scores).item()
        optimal_scale = 1.0 / (2 ** optimal_idx)

        # Compute detail level
        detail_level = self._compute_detail_level(
            image,
            pyramid[optimal_idx]
        )

        # Calculate base resolution
        h, w = image.shape[:2]
        scaled_h = int(h * optimal_scale)
        scaled_w = int(w * optimal_scale)

        # Ensure resolution bounds
        scaled_h = max(self.min_resolution[0], min(scaled_h, self.max_resolution[0]))
        scaled_w = max(self.min_resolution[1], min(scaled_w, self.max_resolution[1]))

        return ScaleParameters(
            scale_factor=float(optimal_scale),
            base_resolution=(scaled_h, scaled_w),
            detail_level=float(detail_level),
            quality_score=float(quality_scores[optimal_idx].item())
        )

    def rescale_region(
        self,
        image: np.ndarray,
        params: ScaleParameters,
        mask: Optional[np.ndarray] = None,
        preserve_details: bool = True
    ) -> np.ndarray:
        """
        Rescale image region according to scale parameters.

        Args:
            image: Input image
            params: Scale parameters from analyze_optimal_scale
            mask: Optional region mask
            preserve_details: Whether to use detail-preserving upscaling

        Returns:
            Rescaled image
        """
        if mask is not None:
            image = image * mask

        if preserve_details and params.scale_factor < 1.0:
            # Use detail-preserving downscaling
            return self._detail_aware_downscale(
                image,
                params.base_resolution,
                params.detail_level
            )
        else:
            # Use standard scaling
            return cv2.resize(
                image,
                params.base_resolution[::-1],  # OpenCV uses (width, height)
                interpolation=cv2.INTER_AREA if params.scale_factor < 1.0
                else cv2.INTER_CUBIC
            )

    def _compute_quality_scores(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quality scores for each scale level.

        Args:
            features: Features from scale space network

        Returns:
            Quality scores tensor
        """
        # Compute feature statistics
        mean_features = torch.mean(features, dim=(2, 3))
        var_features = torch.var(features, dim=(2, 3))

        # Calculate quality metrics
        detail_preservation = torch.mean(var_features, dim=1)
        feature_consistency = -torch.mean(
            torch.abs(mean_features[1:] - mean_features[:-1]),
            dim=1
        )

        # Combine metrics
        scores = detail_preservation + 0.5 * feature_consistency
        return torch.softmax(scores, dim=0)

    def _compute_detail_level(
        self,
        original: np.ndarray,
        scaled: np.ndarray
    ) -> float:
        """
        Compute detail preservation level between original and scaled images.

        Args:
            original: Original image
            scaled: Scaled image

        Returns:
            Detail preservation score in [0, 1]
        """
        # Resize scaled image to original size for comparison
        rescaled = cv2.resize(
            scaled,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

        # Compute gradients
        grad_orig = np.gradient(original)
        grad_scaled = np.gradient(rescaled)

        # Calculate gradient correlation
        correlation = np.mean([
            np.corrcoef(go.ravel(), gs.ravel())[0, 1]
            for go, gs in zip(grad_orig, grad_scaled)
        ])

        return float(max(0, correlation))

    def _detail_aware_downscale(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        detail_level: float
    ) -> np.ndarray:
        """
        Downscale image while preserving important details.

        Args:
            image: Input image
            target_size: Desired output size (height, width)
            detail_level: Detail preservation level

        Returns:
            Downscaled image
        """
        # Compute edge map for detail preservation
        edges = cv2.Canny(
            (image * 255).astype(np.uint8),
            50,
            150
        ).astype(np.float32) / 255.0

        # Create detail mask
        detail_mask = cv2.dilate(
            edges,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )

        # Standard downscaling
        standard = cv2.resize(
            image,
            target_size[::-1],
            interpolation=cv2.INTER_AREA
        )

        # High-quality downscaling
        high_quality = cv2.resize(
            image,
            target_size[::-1],
            interpolation=cv2.INTER_CUBIC
        )

        # Blend based on detail level and detail mask
        detail_mask = cv2.resize(
            detail_mask,
            target_size[::-1],
            interpolation=cv2.INTER_LINEAR
        )

        blend_weights = detail_mask * detail_level
        return standard * (1 - blend_weights) + high_quality * blend_weights


class ScaleSpaceNetwork(nn.Module):
    """Neural network for analyzing scale space features."""

    def __init__(self, base_channels: int = 64, num_scales: int = 4):
        super().__init__()

        # Feature extractors for each scale
        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, base_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.ReLU()
            )
            for _ in range(num_scales)
        ])

        # Scale attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels, base_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        pyramid: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Process image pyramid.

        Args:
            pyramid: List of image tensors at different scales

        Returns:
            Multi-scale features
        """
        # Extract features at each scale
        features = []
        for level, extractor in zip(pyramid, self.extractors):
            feat = extractor(level)
            # Compute attention weights
            weights = self.attention(feat)
            # Apply attention
            features.append(feat * weights)

        return torch.stack(features)
