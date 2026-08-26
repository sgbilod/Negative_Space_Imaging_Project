#!/usr/bin/env python
"""
Uncertainty Management System
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements sophisticated uncertainty quantification and management
for negative space analysis, handling ambiguous cases and providing robust
confidence estimates.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
import torch.distributions as dist


@dataclass
class UncertaintyMetrics:
    """Container for various uncertainty metrics."""
    epistemic: float  # Model uncertainty
    aleatoric: float  # Data uncertainty
    entropy: float    # Prediction entropy
    variance: float   # Feature variance
    confidence: float           # Overall confidence score
    ensemble_disagreement: float  # Ensemble model disagreement
    mutual_information: float     # Information between predictions


class EnsembleUncertaintyEstimator:
    """Estimates uncertainty using ensemble methods."""
    
    def __init__(
        self,
        num_models: int = 5,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
        device: Optional[torch.device] = None
    ):
        self.num_models = num_models
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize ensemble members
        self.ensemble = nn.ModuleList([
            UncertaintyNet(hidden_dim, dropout_rate)
            for _ in range(num_models)
        ]).to(self.device)
        
        # Random forest for feature-based uncertainty
        self.forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5
        )
        
    def estimate_uncertainty(
        self,
        features: torch.Tensor,
        return_predictions: bool = False
    ) -> Tuple[UncertaintyMetrics, Optional[torch.Tensor]]:
        """
        Estimate uncertainty metrics from input features.
        
        Args:
            features: Input feature tensor [batch_size, feature_dim]
            return_predictions: Whether to return model predictions
            
        Returns:
            UncertaintyMetrics object and optionally predictions
        """
        predictions = []
        feature_maps = []
        
        # Get predictions from each ensemble member
        for model in self.ensemble:
            with torch.no_grad():
                pred, feat = model(features)
                predictions.append(pred)
                feature_maps.append(feat)
        
        # Stack predictions and features
        # Stack ensemble outputs:
        # [num_models, batch_size, num_classes]
        predictions = torch.stack(predictions)
        # [num_models, batch_size, hidden_dim]
        feature_maps = torch.stack(feature_maps)
        
        # Calculate various uncertainty metrics
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = torch.mean(var_pred).item()
        
        # Aleatoric uncertainty (data uncertainty)
        aleatoric = torch.mean(
            mean_pred * (1 - mean_pred)  # Binary case
        ).item()
        
        # Prediction entropy
        ent = entropy(mean_pred.cpu().numpy(), axis=1)
        mean_entropy = float(np.mean(ent))
        
        # Feature variance
        feat_var = torch.var(feature_maps, dim=0)
        mean_feat_var = torch.mean(feat_var).item()
        
        # Ensemble disagreement
        disagreement = torch.mean(
            torch.std(predictions, dim=0)
        ).item()
        
        # Mutual information between predictions
        mutual_info = self._compute_mutual_information(predictions)
        
        # Combine metrics into confidence score
        confidence = self._compute_confidence(
            epistemic=epistemic,
            aleatoric=aleatoric,
            entropy=mean_entropy,
            variance=mean_feat_var,
            disagreement=disagreement,
            mutual_info=mutual_info
        )
        
        metrics = UncertaintyMetrics(
            epistemic=epistemic,
            aleatoric=aleatoric,
            entropy=mean_entropy,
            variance=mean_feat_var,
            confidence=confidence,
            ensemble_disagreement=disagreement,
            mutual_information=mutual_info
        )
        
        if return_predictions:
            return metrics, mean_pred
        return metrics, None
    
    def _compute_mutual_information(
        self,
        predictions: torch.Tensor
    ) -> float:
        """
        Compute mutual information between ensemble predictions.
        
        Args:
            predictions: Prediction tensor [num_models, batch_size, num_classes]
            
        Returns:
            Mutual information score
        """
        # Convert to probability distributions
        pred_dist = dist.Categorical(logits=predictions)
        
        # Compute entropy of mean prediction
        mean_pred = torch.mean(predictions, dim=0)
        mean_entropy = dist.Categorical(logits=mean_pred).entropy().mean()
        
        # Compute mean of individual entropies
        individual_entropies = pred_dist.entropy().mean(dim=0)
        mean_individual_entropy = individual_entropies.mean()
        
        # Mutual information is difference between these entropies
        return float(mean_entropy - mean_individual_entropy)
    
    def _compute_confidence(
        self,
        epistemic: float,
        aleatoric: float,
        entropy: float,
        variance: float,
        disagreement: float,
        mutual_info: float
    ) -> float:
        """
        Compute overall confidence score from uncertainty metrics.
        
        Args:
            epistemic: Model uncertainty
            aleatoric: Data uncertainty
            entropy: Prediction entropy
            variance: Feature variance
            disagreement: Ensemble disagreement
            mutual_info: Mutual information
            
        Returns:
            Confidence score in [0, 1]
        """
        # Normalize metrics to [0, 1]
        norm_epistemic = 1 - min(epistemic, 1)
        norm_aleatoric = 1 - min(aleatoric, 1)
        norm_entropy = 1 - min(entropy / np.log(2), 1)  # Log2 for binary case
        norm_variance = 1 - min(variance, 1)
        norm_disagreement = 1 - min(disagreement, 1)
        norm_mutual_info = 1 - min(mutual_info / np.log(2), 1)
        
        # Weighted combination of metrics
        weights = {
            'epistemic': 0.25,
            'aleatoric': 0.2,
            'entropy': 0.15,
            'variance': 0.15,
            'disagreement': 0.15,
            'mutual_info': 0.1
        }
        
        confidence = (
            weights['epistemic'] * norm_epistemic +
            weights['aleatoric'] * norm_aleatoric +
            weights['entropy'] * norm_entropy +
            weights['variance'] * norm_variance +
            weights['disagreement'] * norm_disagreement +
            weights['mutual_info'] * norm_mutual_info
        )
        
        return float(confidence)
    
    def update_forest(self, features: np.ndarray, labels: np.ndarray):
        """Update random forest with new data."""
        self.forest.fit(features, labels)
    
    def feature_uncertainty(self, features: np.ndarray) -> np.ndarray:
        """Get uncertainty estimates from random forest."""
        # Get probabilities from all trees
        probas = np.array([
            tree.predict_proba(features)
            for tree in self.forest.estimators_
        ])
        
        # Calculate variance of predictions across trees
        return np.var(probas, axis=0)


class UncertaintyNet(nn.Module):
    """Neural network for uncertainty estimation."""
    
    def __init__(self, hidden_dim: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both predictions and feature maps.
        
        Args:
            x: Input tensor [batch_size, feature_dim]
            
        Returns:
            predictions: Class probabilities [batch_size, num_classes]
            features: Encoded features [batch_size, hidden_dim]
        """
        features = self.encoder(x)
        predictions = self.classifier(features)
        return predictions, features
