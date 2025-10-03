#!/usr/bin/env python
"""
Negative Space Analysis Algorithm
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements the core negative space analysis algorithm for medical and
astronomical imaging. It uses advanced image processing techniques to detect and
analyze patterns in the negative space between visible objects.

Key Features:
- Multi-scale negative space detection
- Topological pattern analysis
- Region connectivity analysis
- Feature extraction from negative regions
- Machine learning-based pattern recognition
"""

import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional
import cv2
import torch
import torch.nn as nn
from dataclasses import dataclass

from .uncertainty_management import (
    EnsembleUncertaintyEstimator,
    UncertaintyMetrics
)
from .contour_analysis import ContourMorphologyAnalyzer
from .semantic_segmentation import SemanticNegativeSpaceSegmenter
from .pattern_recognition import NegativeSpacePatternNet
from .region_growing import AdaptiveRegionGrower
from .graph_analysis import NegativeSpaceGraphAnalyzer, GraphFeatures
from .topology_analysis import TopologicalAnalyzer, TopologicalFeatures
from .resolution_system import (
    DynamicResolutionAnalyzer,
    ScaleParameters
)


@dataclass
class NegativeSpaceFeatures:
    """Features extracted from negative space regions."""
    area: float
    perimeter: float
    centroid: Tuple[float, float]
    topology_index: float
    connectivity: float
    pattern_score: float
    confidence: float
    topological_features: Optional[TopologicalFeatures] = None
    uncertainty_metrics: Optional[UncertaintyMetrics] = None
    scale_params: Optional[ScaleParameters] = None


class NegativeSpaceAnalyzer:
    """Core class for negative space analysis."""
    
    def __init__(
        self,
        use_gpu: bool = True,
        model_path: Optional[str] = None,
        detection_threshold: float = 0.85,
        min_region_size: int = 100,
        base_channels: int = 64
    ):
        use_cuda = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.detection_threshold = detection_threshold
        self.min_region_size = min_region_size
        
        # Initialize components
        self.uncertainty_estimator = EnsembleUncertaintyEstimator()
        self.contour_analyzer = ContourMorphologyAnalyzer()
        self.segmenter = SemanticNegativeSpaceSegmenter(
            in_channels=1,
            base_channels=base_channels
        )
        self.region_grower = AdaptiveRegionGrower(
            feature_channels=base_channels,
            device=self.device
        )
        self.graph_analyzer = NegativeSpaceGraphAnalyzer(
            feature_dim=base_channels,
            device=self.device
        )
        self.topology_analyzer = TopologicalAnalyzer(
            max_dimension=2,
            device=self.device
        )
        self.resolution_analyzer = DynamicResolutionAnalyzer(
            min_resolution=(32, 32),
            max_resolution=(512, 512),
            device=self.device
        )
        
        self._load_model(model_path)
        
    def _load_model(self, model_path: Optional[str] = None):
        """Load or initialize the pattern recognition model."""
        self.model = NegativeSpacePatternNet().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def _detect_negative_spaces(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Detect negative spaces using semantic segmentation and adaptive region
        growing.
        
        Args:
            image: Preprocessed image
            mask: Optional mask of regions to exclude
            
        Returns:
            Dictionary mapping region IDs to binary masks
        """
        # Prepare image tensor
        img_tensor = torch.from_numpy(image).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Get initial segmentation
        seg_result = self.segmenter(img_tensor)
        
        # Convert probabilities to numpy
        prob_maps = seg_result.probabilities.cpu().numpy()[0]
        
        # Find potential seed points for negative spaces
        negative_spaces = {}
        
        # Process each detected region
        for region_idx in range(prob_maps.shape[0]):
            prob_map = prob_maps[region_idx]
            
            # Threshold probability map
            threshold = self.detection_threshold
            thresh_mask = (prob_map > threshold).astype(np.uint8)
            
            if mask is not None:
                thresh_mask *= mask
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(thresh_mask)
            
            # Process each component
            for label in range(1, num_labels):
                component_mask = (labels == label).astype(np.uint8)
                area = np.sum(component_mask)
                
                if area < self.min_region_size:
                    continue
                
                # Find seed points for region growing
                y_indices, x_indices = np.where(component_mask > 0)
                
                if len(y_indices) == 0:
                    continue
                
                # Select seed points using high confidence areas
                confidences = prob_map[y_indices, x_indices]
                top_k = min(5, len(confidences))
                top_indices = np.argsort(confidences)[-top_k:]
                
                seed_points = np.column_stack([
                    x_indices[top_indices],
                    y_indices[top_indices]
                ])
                
                # Grow region adaptively
                growth_result = self.region_grower.grow_region(
                    image,
                    seed_points,
                    mask
                )
                
                if np.sum(growth_result.mask) >= self.min_region_size:
                    region_id = f"region_{region_idx}_{label}"
                    negative_spaces[region_id] = growth_result.mask
        
        return negative_spaces
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for analysis."""
        # Ensure image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Convert to floating point
        processed = image.astype(np.float32) / 255.0
        
        # Apply contrast enhancement
        processed = cv2.equalizeHist((processed * 255).astype(np.uint8))
        processed = processed.astype(np.float32) / 255.0
        
        # Denoise
        processed = cv2.fastNlMeansDenoising(
            (processed * 255).astype(np.uint8),
            None,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        processed = processed.astype(np.float32) / 255.0
        
        return processed
        
    def _analyze_graph_patterns(
        self,
        regions: Dict[str, np.ndarray],
        features: Dict[str, NegativeSpaceFeatures]
    ) -> GraphFeatures:
        """
        Analyze patterns in negative spaces using graph-based analysis.
        
        Args:
            regions: Dictionary of region masks
            features: Dictionary of region features
            
        Returns:
            GraphFeatures object with pattern analysis results
        """
        # Convert features to feature vectors
        feature_vectors = {
            region_id: np.array([
                features[region_id].area,
                features[region_id].perimeter,
                features[region_id].topology_index,
                features[region_id].connectivity,
                features[region_id].pattern_score
            ])
            for region_id in regions.keys()
        }
        
        # Analyze patterns using graph analyzer
        graph_features = self.graph_analyzer.analyze_pattern(
            regions,
            feature_vectors
        )
        
        return graph_features
    
    def _extract_features(
        self,
        image: np.ndarray,
        region_mask: np.ndarray
    ) -> NegativeSpaceFeatures:
        """
        Extract features from a negative space region.
        
        Args:
            image: Original preprocessed image
            region_mask: Binary mask of the region
            
        Returns:
            NegativeSpaceFeatures object
        """
        # 1. Determine optimal resolution
        scale_params = self.resolution_analyzer.analyze_optimal_scale(
            image,
            region_mask
        )
        
        # Scale region for analysis
        scaled_image = self.resolution_analyzer.rescale_region(
            image,
            scale_params,
            region_mask,
            preserve_details=True
        )
        scaled_mask = self.resolution_analyzer.rescale_region(
            region_mask.astype(np.float32),
            scale_params,
            preserve_details=False
        )
        
        # 2. Basic shape features from scaled region
        area = np.sum(region_mask)  # Use original for accurate area
        contours, _ = cv2.findContours(
            (scaled_mask > 0.5).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = cv2.arcLength(contours[0], True) / scale_params.scale_factor
        
        # 3. Calculate centroid
        M = cv2.moments(scaled_mask)
        if M["m00"] != 0:
            cx = (M["m10"] / M["m00"]) / scale_params.scale_factor
            cy = (M["m01"] / M["m00"]) / scale_params.scale_factor
        else:
            cx, cy = 0, 0
        
        # 3. Topology analysis
        # Calculate Euler number (number of objects - number of holes)
        holes = len(contours) - 1
        topology_index = 1 - holes
        
        # 4. Connectivity analysis using skeletonization
        skeleton = cv2.ximgproc.thinning(region_mask.astype(np.uint8))
        endpoints = self._find_endpoints(skeleton)
        skeleton_sum = np.sum(skeleton > 0)
        connectivity = len(endpoints) / skeleton_sum if skeleton_sum > 0 else 0
        
        # 5. Pattern analysis and uncertainty estimation
        pattern_tensor = self._prepare_pattern_tensor(
            scaled_image * scaled_mask
        )
        pattern_features = self.model.extract_features(pattern_tensor)
        
        with torch.no_grad():
            pattern_score = self.model(pattern_tensor).max().item()
            uncertainty_metrics, _ = (
                self.uncertainty_estimator
                .estimate_uncertainty(pattern_features)
            )
        
        # 6. Perform advanced topological analysis
        y_coords, x_coords = np.where(region_mask)
        points = np.column_stack([x_coords, y_coords])
        features = np.column_stack([
            pattern_tensor.cpu().numpy().reshape(len(points), -1),
            connectivity * np.ones((len(points), 1)),
            topology_index * np.ones((len(points), 1))
        ])
        topological_features = self.topology_analyzer.analyze_topology(
            points, features
        )

        # Calculate overall confidence incorporating topological features
        confidence = self._calculate_confidence(
            area=area,
            perimeter=perimeter,
            topology_index=topology_index,
            connectivity=connectivity,
            pattern_score=pattern_score,
            persistence_entropy=np.mean(
                list(topological_features.persistence_entropy.values())
            )
        )
        
        return NegativeSpaceFeatures(
            area=float(area),
            perimeter=float(perimeter),
            scale_params=scale_params,
            centroid=(float(cx), float(cy)),
            topology_index=float(topology_index),
            connectivity=float(connectivity),
            pattern_score=float(pattern_score),
            confidence=float(confidence),
            topological_features=topological_features
        )
        
    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find endpoints in a skeletonized image."""
        endpoints = []
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                if skeleton[i, j] == 0:
                    continue
                neighbors = np.sum(skeleton[i-1:i+2, j-1:j+2]) - skeleton[i, j]
                if neighbors == 1:
                    endpoints.append((i, j))
        return endpoints
        
    def _prepare_pattern_tensor(
        self,
        region_image: np.ndarray
    ) -> torch.Tensor:
        """Prepare image region for CNN analysis."""
        # Resize to expected input size
        resized = cv2.resize(region_image, (64, 64))
        # Convert to tensor and add batch and channel dimensions
        tensor = torch.from_numpy(resized).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
        
    def _calculate_confidence(
        self,
        area: float,
        perimeter: float,
        topology_index: float,
        connectivity: float,
        pattern_score: float,
        persistence_entropy: Optional[float] = None,
        uncertainty: Optional[UncertaintyMetrics] = None
    ) -> float:
        """Calculate overall confidence score."""
        # Weighted combination of features with topological metrics
        weights = {
            'area': 0.1,
            'perimeter': 0.1,
            'topology': 0.15,
            'connectivity': 0.1,
            'pattern': 0.15,
            'persistence': 0.15,
            'uncertainty': 0.25  # Give high weight to uncertainty
        }
        
        # Normalize features
        norm_area = min(area / self.min_region_size, 1.0)
        norm_perimeter = min(perimeter / (4 * np.sqrt(area)), 1.0)
        
        # Base confidence from geometric and topological features
        base_confidence = (
            weights['area'] * norm_area +
            weights['perimeter'] * norm_perimeter +
            weights['topology'] * (topology_index + 1) / 2 +  # Scale to [0,1]
            weights['connectivity'] * connectivity +
            weights['pattern'] * pattern_score +
            weights['persistence'] * (persistence_entropy or 0.0)
        )
        
        # Incorporate uncertainty if available
        if uncertainty is not None:
            # Calculate uncertainty penalty factor
            epistemic_penalty = uncertainty.epistemic
            aleatoric_penalty = 0.5 * uncertainty.aleatoric  # Partial penalty
            entropy_penalty = 0.3 * uncertainty.entropy  # Small penalty
            
            # Combine penalties and clip to valid range
            uncertainty_factor = max(0, min(1, 1 - (
                epistemic_penalty +
                aleatoric_penalty +
                entropy_penalty
            )))
            confidence = (
                (1 - weights['uncertainty']) * base_confidence +
                weights['uncertainty'] * uncertainty_factor
            )
        else:
            confidence = base_confidence
        
        return confidence
