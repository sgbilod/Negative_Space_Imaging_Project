"""
Deep Feature Extraction Model for Astronomical Negative Space Imaging

Advanced architecture with multi-scale feature extraction and dense connections.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """Dense block with multiple concatenated convolutions."""

    def __init__(self, in_channels: int, growth_rate: int = 32, num_layers: int = 4):
        """Initialize dense block."""
        super().__init__()

        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = in_channels + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(layer_in),
                    nn.ReLU(),
                    nn.Conv2d(layer_in, growth_rate, kernel_size=3, padding=1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with feature concatenation."""
        features = [x]
        for layer in self.layers:
            x = layer(torch.cat(features, 1))
            features.append(x)
        return torch.cat(features, 1)


class FeatureExtractor(nn.Module):
    """Multi-scale feature extraction module."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize feature extractor."""
        super().__init__()

        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale features."""
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        return torch.cat([s1, s2, s3], dim=1)


class DeepModel(nn.Module):
    """Deep feature extraction model for negative space imaging."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        growth_rate: int = 32,
        num_dense_layers: int = 4,
    ):
        """Initialize deep model."""
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.num_dense_layers = num_dense_layers

        base_channels = 64

        # Initial convolution
        self.conv_init = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn_init = nn.BatchNorm2d(base_channels)
        self.pool_init = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense block 1 with feature extraction
        self.feature_extractor1 = FeatureExtractor(base_channels, base_channels)
        self.dense1_in = base_channels * 3
        self.dense1 = DenseBlock(self.dense1_in, growth_rate, num_dense_layers)
        self.dense1_out = self.dense1_in + growth_rate * num_dense_layers

        # Transition layer
        self.transition1 = nn.Sequential(
            nn.Conv2d(self.dense1_out, self.dense1_out // 2, kernel_size=1),
            nn.BatchNorm2d(self.dense1_out // 2),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        transition1_out = self.dense1_out // 2

        # Dense block 2
        self.feature_extractor2 = FeatureExtractor(transition1_out, transition1_out)
        self.dense2_in = transition1_out * 3
        self.dense2 = DenseBlock(self.dense2_in, growth_rate, num_dense_layers)
        self.dense2_out = self.dense2_in + growth_rate * num_dense_layers

        # Transition layer 2
        self.transition2 = nn.Sequential(
            nn.Conv2d(self.dense2_out, self.dense2_out // 2, kernel_size=1),
            nn.BatchNorm2d(self.dense2_out // 2),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        transition2_out = self.dense2_out // 2

        # Global average pooling + FC
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(transition2_out, transition2_out)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(transition2_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Initial conv
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = F.relu(x)
        x = self.pool_init(x)

        # Dense block 1
        x = self.feature_extractor1(x)
        x = self.dense1(x)
        x = self.transition1(x)

        # Dense block 2
        x = self.feature_extractor2(x)
        x = self.dense2(x)
        x = self.transition2(x)

        # Global pooling + FC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_feature_extractor(self):
        """Get feature extractor (without classification head)."""
        return nn.Sequential(
            self.conv_init,
            self.bn_init,
            nn.ReLU(),
            self.pool_init,
            self.feature_extractor1,
            self.dense1,
            self.transition1,
            self.feature_extractor2,
            self.dense2,
            self.transition2,
            self.global_pool,
        )


def deep_model(num_classes: int = 10, growth_rate: int = 32) -> DeepModel:
    """Create deep feature extraction model."""
    model = DeepModel(
        in_channels=3, num_classes=num_classes, growth_rate=growth_rate, num_dense_layers=4
    )
    return model
