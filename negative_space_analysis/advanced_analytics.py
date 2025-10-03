#!/usr/bin/env python
"""
Advanced Analytics for Negative Space Analysis
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides advanced analytics capabilities for negative space
analysis, including:
- Pattern recognition using deep learning
- Statistical analysis of negative space features
- Temporal pattern tracking
- Anomaly detection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from scipy.stats import norm

from .negative_space_algorithm import NegativeSpaceFeatures


@dataclass
class AnalyticsResult:
    """Results from negative space analytics."""
    pattern_type: str
    confidence: float
    anomaly_score: float
    temporal_stability: float
    feature_importance: Dict[str, float]
    related_patterns: List[str]


class NegativeSpacePatternNet(nn.Module):
    """Neural network for negative space pattern recognition."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 pattern types
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class NegativeSpaceAnalytics:
    """Advanced analytics for negative space patterns."""
    
    def __init__(
        self,
        use_gpu: bool = True,
        model_path: Optional[str] = None,
        anomaly_threshold: float = 0.95
    ):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.anomaly_threshold = anomaly_threshold
        self.pattern_net = NegativeSpacePatternNet().to(self.device)
        if model_path:
            self.pattern_net.load_state_dict(torch.load(model_path))
        self.pattern_net.eval()
        
        # Initialize anomaly detector
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def analyze_patterns(
        self,
        features: List[NegativeSpaceFeatures],
        temporal_data: Optional[List[List[NegativeSpaceFeatures]]] = None
    ) -> List[AnalyticsResult]:
        """
        Perform comprehensive pattern analysis.
        
        Args:
            features: List of features from current analysis
            temporal_data: Optional list of historical feature lists
            
        Returns:
            List of analytics results for each pattern
        """
        # Convert features to tensor for pattern recognition
        feature_tensor = self._features_to_tensor(features)
        
        # Get pattern predictions
        with torch.no_grad():
            pattern_probs = self.pattern_net(feature_tensor)
        
        # Detect anomalies
        anomaly_scores = self._detect_anomalies(features)
        
        # Analyze temporal stability if temporal data provided
        temporal_stability = self._analyze_temporal_stability(features, temporal_data)
        
        # Generate analytics results
        results = []
        for i, probs in enumerate(pattern_probs):
            pattern_type = self._get_pattern_type(probs)
            result = AnalyticsResult(
                pattern_type=pattern_type,
                confidence=float(torch.max(probs)),
                anomaly_score=anomaly_scores[i],
                temporal_stability=temporal_stability[i],
                feature_importance=self._analyze_feature_importance(features[i]),
                related_patterns=self._find_related_patterns(pattern_type)
            )
            results.append(result)
        
        return results
        
    def _features_to_tensor(self, features: List[NegativeSpaceFeatures]) -> torch.Tensor:
        """Convert features to tensor for pattern recognition."""
        # Extract numerical features into a matrix
        feature_matrix = np.array([
            [
                f.area,
                f.perimeter,
                f.centroid[0],
                f.centroid[1],
                f.topology_index,
                f.connectivity,
                f.pattern_score,
                f.confidence
            ]
            for f in features
        ])
        
        # Normalize features
        feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / (
            np.std(feature_matrix, axis=0) + 1e-10)
        
        # Convert to tensor
        return torch.from_numpy(feature_matrix).float().to(self.device)
    
    def _detect_anomalies(self, features: List[NegativeSpaceFeatures]) -> np.ndarray:
        """Detect anomalies in feature patterns."""
        feature_matrix = np.array([
            [f.area, f.perimeter, f.topology_index, f.connectivity, f.pattern_score]
            for f in features
        ])
        
        # Fit and predict anomalies
        self.anomaly_detector.fit(feature_matrix)
        scores = self.anomaly_detector.score_samples(feature_matrix)
        
        # Convert to anomaly scores (higher means more anomalous)
        normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return 1 - normalized_scores
    
    def _analyze_temporal_stability(
        self,
        features: List[NegativeSpaceFeatures],
        temporal_data: Optional[List[List[NegativeSpaceFeatures]]] = None
    ) -> np.ndarray:
        """Analyze temporal stability of patterns."""
        if temporal_data is None:
            return np.ones(len(features))  # No temporal data available
            
        # Calculate feature vectors for current and historical data
        current_features = self._features_to_tensor(features).cpu().numpy()
        
        stability_scores = []
        for feat in current_features:
            # Compare with historical features
            historical_scores = []
            for hist_features in temporal_data:
                hist_tensor = self._features_to_tensor(hist_features).cpu().numpy()
                # Calculate similarity with each historical instance
                similarities = 1 / (1 + np.linalg.norm(hist_tensor - feat, axis=1))
                historical_scores.append(np.max(similarities))
            
            # Calculate stability as the consistency of historical matches
            stability = np.mean(historical_scores) if historical_scores else 1.0
            stability_scores.append(stability)
            
        return np.array(stability_scores)
    
    def _analyze_feature_importance(
        self,
        features: NegativeSpaceFeatures
    ) -> Dict[str, float]:
        """Analyze the importance of different features."""
        feature_dict = {
            'area': features.area,
            'perimeter': features.perimeter,
            'centroid_x': features.centroid[0],
            'centroid_y': features.centroid[1],
            'topology': features.topology_index,
            'connectivity': features.connectivity,
            'pattern_score': features.pattern_score
        }
        
        # Calculate importance based on deviation from mean
        feature_values = np.array(list(feature_dict.values()))
        mean_values = np.mean(feature_values)
        std_values = np.std(feature_values)
        
        # Calculate z-scores and convert to importance scores
        importance_scores = np.abs(feature_values - mean_values) / (std_values + 1e-10)
        importance_scores = importance_scores / np.sum(importance_scores)
        
        return {
            name: float(score)
            for name, score in zip(feature_dict.keys(), importance_scores)
        }
    
    def _get_pattern_type(self, probabilities: torch.Tensor) -> str:
        """Convert network output to pattern type."""
        pattern_types = [
            "circular",
            "linear",
            "branching",
            "spiral",
            "mesh",
            "clustered",
            "scattered",
            "concentric",
            "radial",
            "irregular"
        ]
        pattern_idx = torch.argmax(probabilities).item()
        return pattern_types[pattern_idx]
    
    def _find_related_patterns(self, pattern_type: str) -> List[str]:
        """Find patterns related to the given type."""
        pattern_relationships = {
            "circular": ["concentric", "radial"],
            "linear": ["branching", "mesh"],
            "branching": ["linear", "mesh", "radial"],
            "spiral": ["circular", "concentric"],
            "mesh": ["linear", "branching", "scattered"],
            "clustered": ["scattered", "irregular"],
            "scattered": ["clustered", "mesh", "irregular"],
            "concentric": ["circular", "spiral"],
            "radial": ["circular", "branching"],
            "irregular": ["scattered", "clustered"]
        }
        return pattern_relationships.get(pattern_type, [])
