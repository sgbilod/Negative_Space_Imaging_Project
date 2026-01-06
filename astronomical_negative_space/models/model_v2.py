"""
ResNet-Based Model for Astronomical Negative Space Imaging

Advanced architecture using residual connections for better feature learning.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """Initialize residual block."""
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        residual = self.skip(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + residual
        x = F.relu(x)

        return x


class ResNetModel(nn.Module):
    """ResNet-based model for negative space imaging."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        depth: int = 3,
        width_multiplier: int = 1,
    ):
        """Initialize ResNet model."""
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.width_multiplier = width_multiplier

        base_channels = 64 * width_multiplier

        # Initial conv layer
        self.conv_init = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn_init = nn.BatchNorm2d(base_channels)
        self.pool_init = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(base_channels, base_channels, depth, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, depth, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, depth, stride=2)

        # Global average pooling + FC
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(base_channels * 4, base_channels * 4)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(base_channels * 4, num_classes)

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Initial conv
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = F.relu(x)
        x = self.pool_init(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

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
            self.layer1,
            self.layer2,
            self.layer3,
            self.global_pool,
        )


def model_v2(num_classes: int = 10, depth: int = 3, width: int = 1) -> ResNetModel:
    """Create ResNet-based model."""
    model = ResNetModel(
        in_channels=3, num_classes=num_classes, depth=depth, width_multiplier=width
    )
    return model
