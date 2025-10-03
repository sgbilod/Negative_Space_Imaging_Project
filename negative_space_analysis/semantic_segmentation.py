#!/usr/bin/env python
"""
Semantic Negative Space Segmentation
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements advanced semantic segmentation for negative spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class SegmentationResult:
    """Results from semantic segmentation."""
    masks: torch.Tensor  # Segmentation masks
    probabilities: torch.Tensor  # Class probabilities
    features: torch.Tensor  # Extracted features
    attention_maps: torch.Tensor  # Attention visualization
    uncertainty: torch.Tensor  # Segmentation uncertainty


class ConvBlock(nn.Module):
    """Convolutional block with residual connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size//2
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=kernel_size//2
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection if dimensions differ
        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        x = self.relu(x)
        
        return self.dropout(x)


class AttentionModule(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.scale = (channels // num_heads) ** -0.5
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        x = (attn @ v).reshape(B, C, H, W)
        x = self.proj(x)
        
        return x, attn.mean(1)  # Return features and attention map


class DecoderBlock(nn.Module):
    """Decoder block with skip connections and attention."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = AttentionModule(in_channels)
        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Upsample
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear')
        
        # Apply attention
        x, attn = self.attention(x)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolution
        x = self.conv(x)
        
        return x, attn


class SemanticNegativeSpaceSegmenter(nn.Module):
    """Advanced semantic segmentation for negative spaces."""
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels, dropout=dropout)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, dropout=dropout)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, dropout=dropout)
        
        # Bridge with attention
        self.bridge = nn.Sequential(
            AttentionModule(base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 16, dropout=dropout)
        )
        
        # Decoder
        self.dec4 = DecoderBlock(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
            dropout=dropout
        )
        self.dec3 = DecoderBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            dropout=dropout
        )
        self.dec2 = DecoderBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            dropout=dropout
        )
        self.dec1 = DecoderBlock(
            base_channels * 2,
            base_channels,
            base_channels,
            dropout=dropout
        )
        
        # Output heads
        self.segmentation_head = nn.Conv2d(base_channels, num_classes, 1)
        self.uncertainty_head = nn.Conv2d(base_channels, 1, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True
    ) -> SegmentationResult:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bridge
        bridge, bridge_attn = self.bridge(F.max_pool2d(e4, 2))
        
        # Decoder with attention maps
        d4, attn4 = self.dec4(bridge, e4)
        d3, attn3 = self.dec3(d4, e3)
        d2, attn2 = self.dec2(d3, e2)
        d1, attn1 = self.dec1(d2, e1)
        
        # Generate outputs
        segmentation = self.segmentation_head(d1)
        probabilities = F.softmax(segmentation, dim=1)
        uncertainty = self.uncertainty_head(d1).sigmoid()
        
        # Combine attention maps
        attention_maps = torch.stack([
            F.interpolate(attn, size=x.shape[-2:], mode='bilinear')
            for attn in [attn1, attn2, attn3, attn4, bridge_attn]
        ])
        
        return SegmentationResult(
            masks=segmentation,
            probabilities=probabilities,
            features=d1 if return_features else None,
            attention_maps=attention_maps,
            uncertainty=uncertainty
        )
    
    def compute_loss(
        self,
        result: SegmentationResult,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute comprehensive loss metrics."""
        # Segmentation loss (Dice + Cross Entropy)
        dice_loss = self._dice_loss(result.probabilities, target)
        ce_loss = F.cross_entropy(
            result.masks,
            target,
            reduction='none'
        )
        
        if mask is not None:
            ce_loss = ce_loss * mask
        
        ce_loss = ce_loss.mean()
        
        # Uncertainty loss
        uncertainty_loss = self._uncertainty_loss(
            result.uncertainty,
            result.probabilities,
            target
        )
        
        return {
            "dice_loss": dice_loss,
            "cross_entropy_loss": ce_loss,
            "uncertainty_loss": uncertainty_loss,
            "total_loss": dice_loss + ce_loss + 0.1 * uncertainty_loss
        }
    
    def _dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-5
    ) -> torch.Tensor:
        """Compute Dice loss."""
        pred = pred.float()
        target = F.one_hot(
            target,
            num_classes=pred.size(1)
        ).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def _uncertainty_loss(
        self,
        uncertainty: torch.Tensor,
        prob: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute uncertainty loss."""
        # Get prediction error
        target_one_hot = F.one_hot(
            target,
            num_classes=prob.size(1)
        ).permute(0, 3, 1, 2).float()
        error = torch.abs(prob - target_one_hot).mean(dim=1, keepdim=True)
        
        # Uncertainty should correlate with error
        uncertainty_loss = F.mse_loss(uncertainty, error)
        
        return uncertainty_loss
