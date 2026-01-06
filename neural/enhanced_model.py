"""
Enhanced Vision Transformer for Astronomical Negative Space Imaging

Advanced transformer architecture with multi-head attention and multi-scale features.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0):
        """Initialize multi-head attention."""
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-head attention."""
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0):
        """Initialize transformer block."""
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, attn_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        """Initialize patch embedding."""
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patches and embed."""
        x = self.proj(x)  # (B, embed_dim, num_patches**0.5, num_patches**0.5)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiScaleFeatureExtractor(nn.Module):
    """Extract multi-scale features from different transformer layers."""

    def __init__(self, embed_dim: int, num_scales: int = 3):
        """Initialize multi-scale feature extractor."""
        super().__init__()

        self.num_scales = num_scales
        self.scale_projections = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(num_scales)]
        )

    def forward(self, features_list: list) -> torch.Tensor:
        """Combine multi-scale features.

        Args:
            features_list: List of feature tensors from different layers

        Returns:
            Combined multi-scale features
        """
        combined = []
        for i, (features, proj) in enumerate(zip(features_list[-self.num_scales :], self.scale_projections)):
            projected = proj(features)
            combined.append(projected)

        # Concatenate along feature dimension
        combined = torch.cat(combined, dim=-1)
        return combined


class EnhancedVisionTransformer(nn.Module):
    """Enhanced Vision Transformer with multi-head attention and multi-scale features."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        """Initialize Enhanced Vision Transformer."""
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate
                )
                for _ in range(num_layers)
            ]
        )

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Multi-scale feature extraction
        self.multi_scale = MultiScaleFeatureExtractor(embed_dim, num_scales=min(3, num_layers))

        # Classification head
        num_scale_features = embed_dim * min(3, num_layers)
        self.head = nn.Sequential(
            nn.Linear(num_scale_features, embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | Tuple:
        """Forward pass.

        Args:
            x: Input images (B, C, H, W)
            return_features: Whether to return intermediate features

        Returns:
            Classification logits or (logits, features) if return_features=True
        """
        B = x.shape[0]

        # Embed patches
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Collect features from all layers for multi-scale extraction
        features_list = [x]

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            features_list.append(x)

        # Apply layer norm
        x = self.norm(x)

        # Extract multi-scale features
        multi_scale_features = self.multi_scale(features_list)

        # Classification: use pooled multi-scale features
        pooled_features = multi_scale_features.mean(dim=1)  # Global average pooling

        # Classification head
        logits = self.head(pooled_features)

        if return_features:
            return logits, {
                "embeddings": x,
                "cls_token": x[:, 0],
                "multi_scale": multi_scale_features,
            }

        return logits

    def get_attention_maps(self, x: torch.Tensor) -> dict:
        """Get attention maps from all heads (for visualization)."""
        B = x.shape[0]

        # Embed patches
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attention_maps = {}

        # Track attention through blocks
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            # Store intermediate activations
            attention_maps[f"layer_{i}"] = x.detach()

        return attention_maps


def enhanced_model(
    num_classes: int = 10,
    img_size: int = 224,
    patch_size: int = 16,
    num_layers: int = 12,
) -> EnhancedVisionTransformer:
    """Create enhanced Vision Transformer model.

    Args:
        num_classes: Number of output classes
        img_size: Input image size
        patch_size: Patch size for tokenization
        num_layers: Number of transformer layers

    Returns:
        Enhanced Vision Transformer model
    """
    model = EnhancedVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=768,
        num_heads=12,
        num_layers=num_layers,
        mlp_ratio=4.0,
        drop_rate=0.1,
    )
    return model
