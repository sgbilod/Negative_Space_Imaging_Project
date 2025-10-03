#!/usr/bin/env python
"""
Pattern Recognition Network
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements the pattern recognition network for negative space analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PatternRecognitionResult:
    """Results from pattern recognition."""
    pattern_score: float
    feature_vector: torch.Tensor
    attention_weights: torch.Tensor
    class_probabilities: torch.Tensor


class SelfAttention(nn.Module):
    """Self-attention mechanism for pattern recognition."""
    
    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        num_heads: int = 4
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1,
            bias=False
        )
        
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels * 3,
            bias=False
        )
        
        self.project_out = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Scaled dot-product attention
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        # Apply attention to V
        out = (attn @ v)
        
        out = out.reshape(B, C, H, W)
        out = self.project_out(out)
        
        return out, attn


class NegativeSpacePatternNet(nn.Module):
    """Neural network for pattern recognition in negative spaces."""
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_classes: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Initial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks with attention
        self.res1 = self._make_layer(base_channels, base_channels, 2)
        self.attn1 = SelfAttention(base_channels)
        
        self.res2 = self._make_layer(base_channels, base_channels * 2, 2)
        self.attn2 = SelfAttention(base_channels * 2)
        
        self.res3 = self._make_layer(base_channels * 2, base_channels * 4, 2)
        self.attn3 = SelfAttention(base_channels * 4)
        
        # Global feature aggregation
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, num_classes)
        )
        
        # Pattern score head
        self.pattern_score = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels, 1),
            nn.Sigmoid()
        )
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int
    ) -> nn.Sequential:
        """Create a layer of residual blocks."""
        layers = []
        
        # Handle potential dimension change
        if in_channels != out_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Add residual blocks
        for _ in range(num_blocks):
            layers.append(
                ResidualBlock(out_channels, out_channels)
            )
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> PatternRecognitionResult:
        # Initial convolution
        x = self.conv1(x)
        
        # First residual block and attention
        x = self.res1(x)
        x, attn1 = self.attn1(x)
        
        # Second residual block and attention
        x = self.res2(x)
        x, attn2 = self.attn2(x)
        
        # Third residual block and attention
        x = self.res3(x)
        x, attn3 = self.attn3(x)
        
        # Global pooling
        features = self.gap(x)
        features = self.flatten(features)
        
        # Generate outputs
        class_logits = self.classifier(features)
        pattern_score = self.pattern_score(features)
        
        # Combine attention maps
        attention_weights = torch.cat([
            F.interpolate(
                attn.mean(1)[:, None],
                size=x.shape[2:],
                mode='bilinear'
            )
            for attn in [attn1, attn2, attn3]
        ], dim=1)
        
        return PatternRecognitionResult(
            pattern_score=pattern_score.squeeze(),
            feature_vector=features,
            attention_weights=attention_weights,
            class_probabilities=F.softmax(class_logits, dim=1)
        )


class ResidualBlock(nn.Module):
    """Basic residual block with pre-activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-activation
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add residual connection
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out
