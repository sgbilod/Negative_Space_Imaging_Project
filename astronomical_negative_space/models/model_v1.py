"""
Baseline CNN Model for Astronomical Negative Space Imaging

Simple convolutional architecture for baseline comparison.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """Baseline CNN model for negative space imaging."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        hidden_dim: int = 64,
    ):
        """Initialize CNN model."""
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Convolutional blocks
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Global average pooling + FC
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim * 4)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

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
            self.conv1,
            self.bn1,
            nn.ReLU(),
            self.pool1,
            self.conv2,
            self.bn2,
            nn.ReLU(),
            self.pool2,
            self.conv3,
            self.bn3,
            nn.ReLU(),
            self.pool3,
            self.global_pool,
        )


def model_v1(num_classes: int = 10, pretrained: bool = False) -> CNNModel:
    """Create baseline CNN model."""
    model = CNNModel(in_channels=3, num_classes=num_classes, hidden_dim=64)
    return model
