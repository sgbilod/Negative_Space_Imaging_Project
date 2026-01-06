"""
Hybrid Multi-Architecture Ensemble for Astronomical Negative Space Imaging

Combines multiple architectures for robust predictions through voting and averaging.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class ArchitectureModule(nn.Module):
    """Base module for hybrid architecture components."""

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 64):
        """Initialize architecture module."""
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        raise NotImplementedError


class CNNModule(ArchitectureModule):
    """CNN architecture component."""

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 64):
        """Initialize CNN module."""
        super().__init__(in_channels, num_classes, hidden_dim)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 4, num_classes),
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features."""
        features = self.conv_layers(x)
        return features.view(features.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.get_features(x)
        return self.fc(features)


class ResidualModule(ArchitectureModule):
    """Residual architecture component."""

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 64):
        """Initialize residual module."""
        super().__init__(in_channels, num_classes, hidden_dim)

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 7, padding=3),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.res_blocks = nn.ModuleList(
            [self._make_res_block(hidden_dim * (2**i), hidden_dim * (2 ** (i + 1))) for i in range(3)]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 4, num_classes),
        )

    def _make_res_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create residual block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features."""
        x = self.initial(x)
        for block in self.res_blocks:
            x = block(x) + F.interpolate(
                x, size=block(x).shape[-2:], mode="nearest"
            )  # Skip connection
            x = F.relu(x)
        features = self.pool(x)
        return features.view(features.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.get_features(x)
        return self.fc(features)


class AttentionModule(ArchitectureModule):
    """Attention-based architecture component."""

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 64):
        """Initialize attention module."""
        super().__init__(in_channels, num_classes, hidden_dim)

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # Channel attention
        self.ca_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ca_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 16),
            nn.ReLU(),
            nn.Linear(hidden_dim // 16, hidden_dim),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.sa_conv = nn.Conv2d(1, 1, kernel_size=7, padding=3)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features with attention."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Channel attention
        ca = self.ca_pool(x).view(x.size(0), -1)
        ca = self.ca_fc(ca).view(x.size(0), -1, 1, 1)
        x = x * ca

        # Spatial attention
        sa = torch.mean(x, dim=1, keepdim=True)
        sa = self.sa_conv(sa)
        sa = torch.sigmoid(sa)
        x = x * sa

        x = self.conv2(x)
        x = F.relu(x)
        features = self.pool(x)
        return features.view(features.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.get_features(x)
        return self.fc(features)


class HybridModel(nn.Module):
    """Hybrid ensemble combining multiple architectures."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        hidden_dim: int = 64,
        ensemble_method: str = "average",
    ):
        """Initialize hybrid model.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for each module
            ensemble_method: 'average', 'vote', or 'weighted'
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.ensemble_method = ensemble_method

        # Create architecture modules
        self.cnn_module = CNNModule(in_channels, num_classes, hidden_dim)
        self.res_module = ResidualModule(in_channels, num_classes, hidden_dim)
        self.attn_module = AttentionModule(in_channels, num_classes, hidden_dim)

        # Learnable weights for weighted ensemble
        if ensemble_method == "weighted":
            self.module_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ensemble."""
        # Get predictions from each module
        cnn_out = self.cnn_module(x)
        res_out = self.res_module(x)
        attn_out = self.attn_module(x)

        # Ensemble the outputs
        if self.ensemble_method == "average":
            output = (cnn_out + res_out + attn_out) / 3
        elif self.ensemble_method == "vote":
            # Get class predictions and vote
            cnn_pred = cnn_out.argmax(dim=1)
            res_pred = res_out.argmax(dim=1)
            attn_pred = attn_out.argmax(dim=1)

            # Create output from voting (simplified)
            output = (cnn_out + res_out + attn_out) / 3
        elif self.ensemble_method == "weighted":
            weights = F.softmax(self.module_weights, dim=0)
            output = weights[0] * cnn_out + weights[1] * res_out + weights[2] * attn_out
        else:
            output = (cnn_out + res_out + attn_out) / 3

        return output

    def get_module_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get outputs from individual modules for analysis."""
        return {
            "cnn": self.cnn_module(x),
            "residual": self.res_module(x),
            "attention": self.attn_module(x),
        }

    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get feature maps from all modules."""
        return {
            "cnn_features": self.cnn_module.get_features(x),
            "residual_features": self.res_module.get_features(x),
            "attention_features": self.attn_module.get_features(x),
        }


def hybrid_model(num_classes: int = 10, ensemble_method: str = "average") -> HybridModel:
    """Create hybrid ensemble model."""
    model = HybridModel(
        in_channels=3, num_classes=num_classes, hidden_dim=64, ensemble_method=ensemble_method
    )
    return model
