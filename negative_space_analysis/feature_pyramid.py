#!/usr/bin/env python
"""
Feature Pyramid Network for Negative Space Analysis
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements a Feature Pyramid Network (FPN) for multi-scale feature
detection in negative space analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class FPNFeatures:
    """Features extracted at different pyramid levels."""
    level_p2: torch.Tensor  # 1/4 scale
    level_p3: torch.Tensor  # 1/8 scale
    level_p4: torch.Tensor  # 1/16 scale
    level_p5: torch.Tensor  # 1/32 scale
    

class ConvBlock(nn.Module):
    """Basic convolutional block with batch norm and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size//2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature detection."""
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        # Backbone network (ResNet-like)
        self.layer1 = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            ConvBlock(256, 512, stride=2),
            ConvBlock(512, 512)
        )
        
        # Lateral connections
        self.lateral4 = nn.Conv2d(512, 256, 1)
        self.lateral3 = nn.Conv2d(256, 256, 1)
        self.lateral2 = nn.Conv2d(128, 256, 1)
        self.lateral1 = nn.Conv2d(64, 256, 1)
        
        # Smooth layers
        self.smooth4 = ConvBlock(256, 256)
        self.smooth3 = ConvBlock(256, 256)
        self.smooth2 = ConvBlock(256, 256)
        self.smooth1 = ConvBlock(256, 256)
        
    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Upsample x and add to y."""
        _, _, H, W = y.shape
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y
    
    def forward(self, x: torch.Tensor) -> FPNFeatures:
        # Bottom-up pathway
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        # Top-down pathway
        p4 = self.lateral4(c4)
        p3 = self._upsample_add(p4, self.lateral3(c3))
        p2 = self._upsample_add(p3, self.lateral2(c2))
        p1 = self._upsample_add(p2, self.lateral1(c1))
        
        # Smooth
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)
        
        return FPNFeatures(
            level_p2=p1,
            level_p3=p2,
            level_p4=p3,
            level_p5=p4
        )
