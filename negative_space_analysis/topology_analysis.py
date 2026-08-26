#!/usr/bin/env python
"""
Topological Data Analysis Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements advanced topological data analysis for negative space
patterns using persistent homology and related techniques.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass
import ripser
from scipy.spatial.distance import pdist, squareform
from persim import PersistenceImager


@dataclass
class TopologicalFeatures:
    """Features extracted from topological analysis."""
    persistence_diagrams: Dict[int, np.ndarray]  # dimension -> diagram
    betti_curves: Dict[int, np.ndarray]  # dimension -> betti curve
    persistence_images: Dict[int, np.ndarray]  # dimension -> persistence image
    persistence_entropy: Dict[int, float]  # dimension -> entropy
    homology_features: Dict[str, float]  # Named topological features
    persistence_landscape: Dict[int, np.ndarray]  # dimension -> landscape


class TopologicalEncoder(nn.Module):
    """Neural network for encoding topological features."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Persistence diagram encoding
        self.diagram_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Betti curve encoding
        self.betti_encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Persistence image encoding
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        diagrams: torch.Tensor,
        betti_curves: torch.Tensor,
        persistence_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode topological features.
        
        Args:
            diagrams: Persistence diagrams [batch, points, 2]
            betti_curves: Betti curves [batch, length]
            persistence_images: Persistence images [batch, height, width]
            
        Returns:
            Encoded features [batch, hidden_dim]
        """
        # Encode persistence diagrams
        diagram_features = self.diagram_encoder(diagrams.flatten(1))
        
        # Encode Betti curves
        betti_features = self.betti_encoder(
            betti_curves.unsqueeze(1)
        ).squeeze(-1)
        
        # Encode persistence images
        image_features = self.image_encoder(
            persistence_images.unsqueeze(1)
        ).squeeze(-1).squeeze(-1)
        
        # Fuse features
        combined = torch.cat([
            diagram_features,
            betti_features,
            image_features
        ], dim=-1)
        
        return self.fusion(combined)


class TopologicalAnalyzer:
    """Analyzes negative space patterns using topological methods."""
    
    def __init__(
        self,
        max_dimension: int = 2,
        num_landscape_layers: int = 5,
        resolution: int = 100,
        persistence_threshold: float = 0.1,
        device: Optional[torch.device] = None
    ):
        self.max_dimension = max_dimension
        self.num_landscape_layers = num_landscape_layers
        self.resolution = resolution
        self.persistence_threshold = persistence_threshold
        self.device = device or torch.device('cpu')
        
        # Initialize persistence image transformer
        self.persistence_imager = PersistenceImager(
            pixel_size=0.1,
            birth_range=(-0.1, 1.1),
            pers_range=(-0.1, 1.1)
        )
    
    def analyze_topology(
        self,
        points: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> TopologicalFeatures:
        """
        Analyze topology of point cloud data.
        
        Args:
            points: Point cloud data [num_points, dimensions]
            mask: Optional mask for filtering points
            
        Returns:
            TopologicalFeatures object
        """
        if mask is not None:
            points = points[mask]
        
        # Compute distance matrix
        distances = squareform(pdist(points))
        
        # Compute persistent homology
        diagrams = self._compute_persistence_diagrams(distances)
        
        # Compute derived features
        betti_curves = self._compute_betti_curves(diagrams)
        persistence_images = self._compute_persistence_images(diagrams)
        entropy = self._compute_persistence_entropy(diagrams)
        homology = self._compute_homology_features(diagrams)
        landscapes = self._compute_persistence_landscapes(diagrams)
        
        return TopologicalFeatures(
            persistence_diagrams=diagrams,
            betti_curves=betti_curves,
            persistence_images=persistence_images,
            persistence_entropy=entropy,
            homology_features=homology,
            persistence_landscape=landscapes
        )
    
    def _compute_persistence_diagrams(
        self,
        distances: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """Compute persistence diagrams using Ripser."""
        # Compute persistent homology
        diagrams = ripser.ripser(
            distances,
            maxdim=self.max_dimension,
            distance_matrix=True
        )["dgms"]
        
        return {i: diag for i, diag in enumerate(diagrams)}
    
    def _compute_betti_curves(
        self,
        diagrams: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Compute Betti curves from persistence diagrams."""
        curves = {}
        
        for dim, diagram in diagrams.items():
            # Create evaluation points
            t = np.linspace(0, 1, self.resolution)
            curve = np.zeros_like(t)
            
            # Count persistent features at each time
            for birth, death in diagram:
                if death > birth:  # Exclude diagonal points
                    curve += (
                        (t >= birth) & (t < death)
                    ).astype(np.float32)
            
            curves[dim] = curve
        
        return curves
    
    def _compute_persistence_images(
        self,
        diagrams: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Transform persistence diagrams to images."""
        images = {}
        
        for dim, diagram in diagrams.items():
            # Filter points by persistence
            pers = diagram[:, 1] - diagram[:, 0]
            significant = diagram[pers > self.persistence_threshold]
            
            if len(significant) > 0:
                images[dim] = self.persistence_imager.transform(
                    significant
                )
            else:
                images[dim] = np.zeros(
                    (self.resolution, self.resolution)
                )
        
        return images
    
    def _compute_persistence_entropy(
        self,
        diagrams: Dict[int, np.ndarray]
    ) -> Dict[int, float]:
        """Compute persistence entropy for each dimension."""
        entropy = {}
        
        for dim, diagram in diagrams.items():
            # Compute persistence values
            pers = diagram[:, 1] - diagram[:, 0]
            
            # Filter significant features
            significant = pers[pers > self.persistence_threshold]
            
            if len(significant) > 0:
                # Normalize persistences
                total_pers = np.sum(significant)
                if total_pers > 0:
                    probs = significant / total_pers
                    entropy[dim] = float(-np.sum(
                        probs * np.log2(probs + 1e-10)
                    ))
                else:
                    entropy[dim] = 0.0
            else:
                entropy[dim] = 0.0
        
        return entropy
    
    def _compute_homology_features(
        self,
        diagrams: Dict[int, np.ndarray]
    ) -> Dict[str, float]:
        """Compute various homological features."""
        features = {}
        
        for dim, diagram in diagrams.items():
            # Compute persistence values
            pers = diagram[:, 1] - diagram[:, 0]
            significant = pers[pers > self.persistence_threshold]
            
            features[f"total_persistence_{dim}"] = float(np.sum(significant))
            features[f"max_persistence_{dim}"] = float(
                np.max(significant) if len(significant) > 0 else 0
            )
            features[f"num_features_{dim}"] = float(len(significant))
            
            if len(significant) > 0:
                features[f"mean_persistence_{dim}"] = float(np.mean(significant))
                features[f"std_persistence_{dim}"] = float(np.std(significant))
            else:
                features[f"mean_persistence_{dim}"] = 0.0
                features[f"std_persistence_{dim}"] = 0.0
        
        return features
    
    def _compute_persistence_landscapes(
        self,
        diagrams: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Compute persistence landscapes."""
        landscapes = {}
        
        for dim, diagram in diagrams.items():
            # Filter significant points
            pers = diagram[:, 1] - diagram[:, 0]
            significant = diagram[pers > self.persistence_threshold]
            
            if len(significant) > 0:
                # Create landscape functions
                t = np.linspace(0, 1, self.resolution)
                landscape = np.zeros(
                    (self.num_landscape_layers, len(t))
                )
                
                for birth, death in significant:
                    # Compute triangle function
                    peak = (birth + death) / 2
                    height = (death - birth) / 2
                    
                    triangle = np.maximum(
                        0,
                        np.minimum(
                            height - np.abs(t - peak),
                            height
                        )
                    )
                    
                    # Add to landscape layers
                    for k in range(self.num_landscape_layers):
                        landscape[k] = np.maximum(
                            landscape[k],
                            triangle
                        )
                        triangle = np.minimum(
                            landscape[k],
                            triangle
                        )
                
                landscapes[dim] = landscape
            else:
                landscapes[dim] = np.zeros(
                    (self.num_landscape_layers, self.resolution)
                )
        
        return landscapes
    
    def analyze_negative_spaces(
        self,
        masks: Dict[str, np.ndarray],
        features: Dict[str, np.ndarray]
    ) -> Dict[str, TopologicalFeatures]:
        """
        Analyze topology of negative space regions.
        
        Args:
            masks: Dictionary of region masks
            features: Dictionary of region features
            
        Returns:
            Dictionary mapping region IDs to TopologicalFeatures
        """
        topology = {}
        
        for region_id, mask in masks.items():
            # Get points from mask
            y_coords, x_coords = np.where(mask > 0)
            points = np.column_stack([x_coords, y_coords])
            
            # Add feature dimensions if available
            if region_id in features:
                feature_points = features[region_id]
                if len(feature_points.shape) == 1:
                    feature_points = feature_points[:, np.newaxis]
                points = np.hstack([points, feature_points])
            
            # Analyze topology
            topology[region_id] = self.analyze_topology(points)
        
        return topology
