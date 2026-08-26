#!/usr/bin/env python
"""
Uncertainty Quantification Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements Bayesian uncertainty quantification for negative space analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class UncertaintyMetrics:
    """Stores uncertainty metrics for negative space analysis."""
    aleatoric_uncertainty: torch.Tensor  # Data uncertainty
    epistemic_uncertainty: torch.Tensor  # Model uncertainty
    combined_uncertainty: torch.Tensor   # Total uncertainty
    confidence_score: float              # Overall confidence
    region_uncertainties: Dict[str, float]  # Per-region uncertainties


class BayesianConvBlock(nn.Module):
    """Bayesian convolutional block with dropout-based uncertainty."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size//2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        return self.activation(x)


class UncertaintyEstimator(nn.Module):
    """Estimates uncertainty in negative space detection and analysis."""
    
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 64,
        n_monte_carlo: int = 20,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.n_monte_carlo = n_monte_carlo
        
        # Uncertainty estimation network
        self.encoder = nn.Sequential(
            BayesianConvBlock(input_channels, base_channels, dropout_rate=dropout_rate),
            BayesianConvBlock(base_channels, base_channels * 2, dropout_rate=dropout_rate),
            BayesianConvBlock(base_channels * 2, base_channels * 4, dropout_rate=dropout_rate)
        )
        
        # Aleatoric uncertainty head
        self.aleatoric_head = nn.Sequential(
            BayesianConvBlock(base_channels * 4, base_channels * 2),
            nn.Conv2d(base_channels * 2, 2, 1)  # Mean and variance
        )
        
        # Epistemic uncertainty will be estimated through MC dropout
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with uncertainty estimation."""
        features = self.encoder(x)
        aleatoric_params = self.aleatoric_head(features)
        
        if return_features:
            return aleatoric_params, features
        return aleatoric_params
    
    def estimate_uncertainty(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> UncertaintyMetrics:
        """
        Estimate uncertainty metrics for the input image.
        
        Args:
            image: Input image tensor
            mask: Optional mask for regions of interest
            
        Returns:
            UncertaintyMetrics object containing various uncertainty measures
        """
        self.train()  # Enable dropout for MC sampling
        
        # Monte Carlo sampling
        predictions = []
        for _ in range(self.n_monte_carlo):
            with torch.no_grad():
                pred, _ = self.forward(image, return_features=True)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Calculate uncertainties
        mean_pred = predictions.mean(dim=0)
        aleatoric_uncertainty = torch.exp(mean_pred[:, 1:])  # Log variance -> variance
        epistemic_uncertainty = predictions.var(dim=0)
        combined_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        # Calculate confidence score
        if mask is not None:
            confidence_score = self._compute_masked_confidence(
                combined_uncertainty,
                mask
            )
        else:
            confidence_score = self._compute_confidence(combined_uncertainty)
        
        # Calculate per-region uncertainties
        region_uncertainties = self._compute_region_uncertainties(
            combined_uncertainty,
            mask
        )
        
        return UncertaintyMetrics(
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            combined_uncertainty=combined_uncertainty,
            confidence_score=confidence_score,
            region_uncertainties=region_uncertainties
        )
    
    def _compute_confidence(self, uncertainty: torch.Tensor) -> float:
        """Compute overall confidence score from uncertainty."""
        # Transform uncertainty to confidence using soft mapping
        confidence = torch.exp(-uncertainty)
        return float(confidence.mean())
    
    def _compute_masked_confidence(
        self,
        uncertainty: torch.Tensor,
        mask: torch.Tensor
    ) -> float:
        """Compute confidence score for masked regions."""
        masked_uncertainty = uncertainty * mask
        masked_conf = torch.exp(-masked_uncertainty)
        return float((masked_conf * mask).sum() / (mask.sum() + 1e-6))
    
    def _compute_region_uncertainties(
        self,
        uncertainty: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute per-region uncertainty scores."""
        if mask is None:
            return {"global": float(uncertainty.mean())}
        
        # Connected component analysis
        labels = torch.zeros_like(mask)
        n_regions = labels.max() + 1
        
        uncertainties = {}
        for i in range(n_regions):
            region_mask = (labels == i)
            region_uncertainty = float(
                (uncertainty * region_mask).sum() / (region_mask.sum() + 1e-6)
            )
            uncertainties[f"region_{i}"] = region_uncertainty
        
        return uncertainties


class EnsembleUncertaintyEstimator:
    """Ensemble-based uncertainty estimation."""
    
    def __init__(
        self,
        n_estimators: int = 5,
        input_channels: int = 1,
        base_channels: int = 64
    ):
        self.estimators = nn.ModuleList([
            UncertaintyEstimator(input_channels, base_channels)
            for _ in range(n_estimators)
        ])
    
    def estimate_uncertainty(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> UncertaintyMetrics:
        """Estimate uncertainty using ensemble."""
        ensemble_predictions = []
        for estimator in self.estimators:
            metrics = estimator.estimate_uncertainty(image, mask)
            ensemble_predictions.append(metrics)
        
        # Aggregate ensemble predictions
        aleatoric = torch.stack([m.aleatoric_uncertainty for m in ensemble_predictions])
        epistemic = torch.stack([m.epistemic_uncertainty for m in ensemble_predictions])
        combined = torch.stack([m.combined_uncertainty for m in ensemble_predictions])
        
        mean_aleatoric = aleatoric.mean(dim=0)
        mean_epistemic = epistemic.mean(dim=0)
        mean_combined = combined.mean(dim=0)
        
        # Additional ensemble uncertainty
        ensemble_uncertainty = combined.var(dim=0)
        
        return UncertaintyMetrics(
            aleatoric_uncertainty=mean_aleatoric,
            epistemic_uncertainty=mean_epistemic + ensemble_uncertainty,
            combined_uncertainty=mean_combined + ensemble_uncertainty,
            confidence_score=float(torch.exp(-mean_combined).mean()),
            region_uncertainties=self._aggregate_region_uncertainties(
                ensemble_predictions
            )
        )
    
    def _aggregate_region_uncertainties(
        self,
        predictions: List[UncertaintyMetrics]
    ) -> Dict[str, float]:
        """Aggregate region uncertainties from ensemble."""
        aggregated = {}
        for region in predictions[0].region_uncertainties.keys():
            values = [p.region_uncertainties[region] for p in predictions]
            aggregated[region] = float(np.mean(values))
        return aggregated
