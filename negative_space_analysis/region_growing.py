#!/usr/bin/env python
"""
Adaptive Region Growing Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements an intelligent region growing system that uses deep
learning to guide the growth process and adapt to image characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from queue import PriorityQueue
import cv2


@dataclass
class RegionGrowthResult:
    """Results from adaptive region growing."""
    mask: np.ndarray
    confidence_map: np.ndarray
    boundary_points: np.ndarray
    feature_map: np.ndarray
    growth_history: List[np.ndarray]
    region_features: Dict[str, float]


class AdaptiveFeatureExtractor(nn.Module):
    """Neural network for extracting growth-guiding features."""
    
    def __init__(
        self,
        in_channels: int = 1,
        feature_channels: int = 32,
        num_scales: int = 3
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale feature extraction
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    feature_channels,
                    feature_channels * (2 ** i),
                    3,
                    stride=2 ** i,
                    padding=1
                ),
                nn.BatchNorm2d(feature_channels * (2 ** i)),
                nn.ReLU(inplace=True)
            )
            for i in range(num_scales)
        ])
        
        # Feature refinement
        total_channels = feature_channels * sum(2 ** i for i in range(num_scales))
        self.refine = nn.Sequential(
            nn.Conv2d(total_channels, feature_channels * 2, 1),
            nn.BatchNorm2d(feature_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels * 2, feature_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial features
        x = self.conv1(x)
        
        # Multi-scale features
        features = []
        for scale in self.scales:
            scale_features = scale(x)
            # Upscale to original size
            if scale_features.shape[-2:] != x.shape[-2:]:
                scale_features = F.interpolate(
                    scale_features,
                    size=x.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            features.append(scale_features)
        
        # Combine and refine features
        combined = torch.cat(features, dim=1)
        refined = self.refine(combined)
        
        return refined


class AdaptiveRegionGrower:
    """Intelligent region growing system using learned features."""
    
    def __init__(
        self,
        feature_channels: int = 32,
        confidence_threshold: float = 0.85,
        min_region_size: int = 50,
        max_iterations: int = 1000,
        device: Optional[torch.device] = None
    ):
        self.feature_channels = feature_channels
        self.confidence_threshold = confidence_threshold
        self.min_region_size = min_region_size
        self.max_iterations = max_iterations
        self.device = device or torch.device('cpu')
        
        # Initialize feature extractor
        self.feature_extractor = AdaptiveFeatureExtractor(
            in_channels=1,
            feature_channels=feature_channels
        ).to(self.device)
    
    def grow_region(
        self,
        image: np.ndarray,
        seed_points: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> RegionGrowthResult:
        """
        Grow region from seed points using learned features.
        
        Args:
            image: Input image
            seed_points: Initial points for region growing
            mask: Optional mask for region of interest
            
        Returns:
            RegionGrowthResult object
        """
        # Convert image to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
        
        # Convert features to numpy
        feature_map = features.cpu().numpy()[0]
        
        # Initialize region
        height, width = image.shape
        region_mask = np.zeros((height, width), dtype=np.uint8)
        visited = set()
        confidence_map = np.zeros_like(image, dtype=np.float32)
        growth_history = []
        
        # Priority queue for region growing
        queue = PriorityQueue()
        
        # Add seed points
        for seed in seed_points:
            x, y = int(seed[0]), int(seed[1])
            if 0 <= x < width and 0 <= y < height:
                queue.put((-1.0, (y, x)))  # -1.0 for highest priority
                region_mask[y, x] = 1
                visited.add((y, x))
        
        # Grow region
        iteration = 0
        while not queue.empty() and iteration < self.max_iterations:
            _, (y, x) = queue.get()
            
            # Check neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, nx = y + dy, x + dx
                    
                    # Check bounds and mask
                    if (
                        ny < 0 or ny >= height or
                        nx < 0 or nx >= width or
                        (ny, nx) in visited or
                        (mask is not None and not mask[ny, nx])
                    ):
                        continue
                    
                    # Compute growth confidence
                    confidence = self._compute_growth_confidence(
                        feature_map,
                        region_mask,
                        ny,
                        nx
                    )
                    
                    if confidence > self.confidence_threshold:
                        queue.put((-confidence, (ny, nx)))
                        region_mask[ny, nx] = 1
                        visited.add((ny, nx))
                        confidence_map[ny, nx] = confidence
            
            # Save growth history periodically
            if iteration % 10 == 0:
                growth_history.append(region_mask.copy())
            
            iteration += 1
        
        # Extract boundary points
        boundary = self._extract_boundary_points(region_mask)
        
        # Compute region features
        region_features = self._compute_region_features(
            image,
            region_mask,
            feature_map
        )
        
        return RegionGrowthResult(
            mask=region_mask,
            confidence_map=confidence_map,
            boundary_points=boundary,
            feature_map=feature_map,
            growth_history=growth_history,
            region_features=region_features
        )
    
    def _compute_growth_confidence(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        y: int,
        x: int
    ) -> float:
        """Compute confidence score for growing to a new pixel."""
        # Extract local feature patch
        patch_size = 3
        half_size = patch_size // 2
        
        h, w = mask.shape
        y1 = max(0, y - half_size)
        y2 = min(h, y + half_size + 1)
        x1 = max(0, x - half_size)
        x2 = min(w, x + half_size + 1)
        
        # Get feature vectors
        center_features = features[:, y, x]
        mask_patch = mask[y1:y2, x1:x2]
        
        # Compute feature similarity with existing region
        if np.sum(mask_patch) == 0:
            return 0.0
        
        neighbor_features = features[:, y1:y2, x1:x2]
        region_features = neighbor_features[:, mask_patch == 1]
        
        # Compute cosine similarity
        similarity = np.mean([
            self._cosine_similarity(center_features, ref_features)
            for ref_features in region_features.T
        ])
        
        return float(similarity)
    
    def _cosine_similarity(
        self,
        v1: np.ndarray,
        v2: np.ndarray
    ) -> float:
        """Compute cosine similarity between feature vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def _extract_boundary_points(
        self,
        mask: np.ndarray
    ) -> np.ndarray:
        """Extract boundary points from region mask."""
        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return np.array([])
        
        # Combine all contour points
        boundary = np.vstack([cont.squeeze() for cont in contours])
        return boundary
    
    def _compute_region_features(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        features: np.ndarray
    ) -> Dict[str, float]:
        """Compute statistical features of the grown region."""
        # Basic shape features
        area = float(np.sum(mask))
        perimeter = float(len(self._extract_boundary_points(mask)))
        
        # Intensity statistics
        region_intensities = image[mask > 0]
        mean_intensity = float(np.mean(region_intensities))
        std_intensity = float(np.std(region_intensities))
        
        # Feature statistics
        region_features = features[:, mask > 0]
        mean_features = np.mean(region_features, axis=1)
        std_features = np.std(region_features, axis=1)
        
        return {
            "area": area,
            "perimeter": perimeter,
            "compactness": 4 * np.pi * area / (perimeter ** 2),
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "mean_feature_magnitude": float(np.mean(mean_features)),
            "std_feature_magnitude": float(np.mean(std_features))
        }
