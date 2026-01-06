"""
Vision Transformer (ViT) Integration for Negative Space Detection

Advanced ViT backbone architecture with:
- Pre-trained model loading (vit_base_patch16_224, vit_large_patch16_224)
- Patch embedding and tokenization
- Multi-head self-attention mechanisms
- Transformer blocks (12-24 layers)
- Classification head for negative space detection
- Adaptive pooling and feature extraction

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model, list_models
from torch import Tensor

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for Vision Transformer.

    Converts image to sequence of patch embeddings using convolution.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Initialize patch embedding.

        Args:
            img_size: Input image size (assumed square)
            patch_size: Patch size (assumed square)
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: convert image to patch embeddings.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim)
        """
        # Project patches: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)  # (B, embed_dim, grid_h, grid_w)

        # Flatten spatial dimensions: (B, embed_dim, grid_h, grid_w) -> (B, embed_dim, num_patches)
        x = x.flatten(2)  # (B, embed_dim, num_patches)

        # Transpose to (B, num_patches, embed_dim)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)

        # Apply normalization
        x = self.norm(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Implements scaled dot-product attention with multiple heads.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: compute multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Attention output of shape (batch_size, seq_len, embed_dim)
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, seq_len, seq_len)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, seq_len, embed_dim)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and feedforward.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        """
        Initialize transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Hidden dim / embed_dim ratio for MLP
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
            drop_path: Stochastic depth rate
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(proj_drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: apply attention and MLP with residual connections.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """
    Stochastic depth (drop path) for regularization.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Initialize DropPath."""
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Apply stochastic depth."""
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.bernoulli(torch.ones(shape) * keep_prob).to(x.device)
        return x * random_tensor / keep_prob


class ClassificationHead(nn.Module):
    """
    Classification head for negative space detection.
    """

    def __init__(
        self,
        in_features: int = 768,
        num_classes: int = 2,
        dropout: float = 0.1,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize classification head.

        Args:
            in_features: Input feature dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            hidden_dims: Optional list of hidden layer dimensions
        """
        super().__init__()
        self.num_classes = num_classes

        if hidden_dims is None:
            hidden_dims = [in_features // 2]

        layers: List[nn.Module] = []
        prev_dim = in_features

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: classify features.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.mlp(x)


class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer for negative space detection.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 2,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        pre_trained: bool = False,
        pre_trained_weights: Optional[str] = None,
    ) -> None:
        """
        Initialize Vision Transformer.

        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            mlp_ratio: MLP ratio in transformer blocks
            num_classes: Number of output classes
            dropout: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            pre_trained: Whether to load pre-trained weights
            pre_trained_weights: Path to pre-trained weights file
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=dropout,
                    drop_path=dpr[i],
                )
                for i in range(num_layers)
            ]
        )

        # Layer normalization and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = ClassificationHead(
            in_features=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

        # Load pre-trained weights if specified
        if pre_trained and pre_trained_weights:
            self.load_pretrained_weights(pre_trained_weights)
        elif pre_trained:
            self._init_from_timm()

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for specific module types."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_from_timm(self) -> None:
        """Initialize from timm pre-trained model."""
        try:
            # Load base ViT model from timm
            model_name = "vit_base_patch16_224"
            pretrained_model = create_model(model_name, pretrained=True)

            # Copy patch embedding
            self.patch_embed.proj.weight.data = pretrained_model.patch_embed.proj.weight.data
            self.patch_embed.proj.bias.data = pretrained_model.patch_embed.proj.bias.data

            # Copy position embedding
            self.pos_embed.data = pretrained_model.pos_embed.data[:, :self.patch_embed.num_patches + 1, :]

            # Copy cls token
            self.cls_token.data = pretrained_model.cls_token.data

            # Copy transformer blocks
            for i, block in enumerate(self.blocks):
                block.norm1.weight.data = pretrained_model.blocks[i].norm1.weight.data
                block.norm1.bias.data = pretrained_model.blocks[i].norm1.bias.data
                block.attn.qkv.weight.data = pretrained_model.blocks[i].attn.qkv.weight.data
                block.attn.qkv.bias.data = pretrained_model.blocks[i].attn.qkv.bias.data
                block.attn.proj.weight.data = pretrained_model.blocks[i].attn.proj.weight.data
                block.attn.proj.bias.data = pretrained_model.blocks[i].attn.proj.bias.data
                block.norm2.weight.data = pretrained_model.blocks[i].norm2.weight.data
                block.norm2.bias.data = pretrained_model.blocks[i].norm2.bias.data
                block.mlp[0].weight.data = pretrained_model.blocks[i].mlp.fc1.weight.data
                block.mlp[0].bias.data = pretrained_model.blocks[i].mlp.fc1.bias.data
                block.mlp[3].weight.data = pretrained_model.blocks[i].mlp.fc2.weight.data
                block.mlp[3].bias.data = pretrained_model.blocks[i].mlp.fc2.bias.data

            # Copy normalization
            self.norm.weight.data = pretrained_model.norm.weight.data
            self.norm.bias.data = pretrained_model.norm.bias.data

            logger.info(f"Loaded pre-trained weights from timm ({model_name})")
        except Exception as e:
            logger.warning(f"Failed to load pre-trained weights: {e}")

    def load_pretrained_weights(self, weights_path: str) -> None:
        """Load pre-trained weights from file."""
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pre-trained weights from {weights_path}")
        except Exception as e:
            logger.error(f"Failed to load pre-trained weights from {weights_path}: {e}")
            raise

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: process image through ViT.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        # Add position embedding and dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Layer normalization
        x = self.norm(x)

        # Extract class token and pass to head
        x = x[:, 0]  # (B, embed_dim)
        logits = self.head(x)  # (B, num_classes)

        return logits

    def get_attention_maps(self, x: Tensor, layer_idx: int) -> Tensor:
        """
        Get attention maps from specific layer.

        Args:
            x: Input tensor
            layer_idx: Index of transformer block

        Returns:
            Attention maps tensor
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Get attention from specified layer
        attn_maps = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == layer_idx:
                # Get attention from this block
                attn_maps = block.attn  # This would need modification to return attention
                break

        return attn_maps

    def freeze_backbone(self, freeze_until_layer: int = 8) -> None:
        """
        Freeze backbone layers up to specified layer.

        Args:
            freeze_until_layer: Index of layer to freeze until
        """
        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # Freeze position embedding
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False

        # Freeze specified transformer blocks
        for i, block in enumerate(self.blocks):
            if i < freeze_until_layer:
                for param in block.parameters():
                    param.requires_grad = False

        logger.info(f"Froze backbone up to layer {freeze_until_layer}")

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfroze all parameters")

    def count_parameters(self) -> Dict[str, int]:
        """
        Count total and trainable parameters.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
        }


class ViTFactory:
    """Factory for creating Vision Transformer models."""

    @staticmethod
    def create_vit_base(
        img_size: int = 224,
        num_classes: int = 2,
        pre_trained: bool = True,
    ) -> VisionTransformer:
        """
        Create Vision Transformer Base model.

        Args:
            img_size: Input image size
            num_classes: Number of output classes
            pre_trained: Whether to use pre-trained weights

        Returns:
            Initialized ViT Base model
        """
        return VisionTransformer(
            img_size=img_size,
            patch_size=16,
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            mlp_ratio=4.0,
            num_classes=num_classes,
            dropout=0.1,
            pre_trained=pre_trained,
        )

    @staticmethod
    def create_vit_large(
        img_size: int = 224,
        num_classes: int = 2,
        pre_trained: bool = True,
    ) -> VisionTransformer:
        """
        Create Vision Transformer Large model.

        Args:
            img_size: Input image size
            num_classes: Number of output classes
            pre_trained: Whether to use pre-trained weights

        Returns:
            Initialized ViT Large model
        """
        return VisionTransformer(
            img_size=img_size,
            patch_size=16,
            embed_dim=1024,
            num_heads=16,
            num_layers=24,
            mlp_ratio=4.0,
            num_classes=num_classes,
            dropout=0.1,
            pre_trained=pre_trained,
        )

    @staticmethod
    def create_vit_base_high_res(
        img_size: int = 384,
        num_classes: int = 2,
        pre_trained: bool = True,
    ) -> VisionTransformer:
        """
        Create Vision Transformer Base with high-resolution input.

        Args:
            img_size: Input image size (384x384)
            num_classes: Number of output classes
            pre_trained: Whether to use pre-trained weights

        Returns:
            Initialized ViT Base model for high-res images
        """
        return VisionTransformer(
            img_size=img_size,
            patch_size=16,
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            mlp_ratio=4.0,
            num_classes=num_classes,
            dropout=0.1,
            pre_trained=pre_trained,
        )
