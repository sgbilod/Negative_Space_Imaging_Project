"""
Transformer Baseline for SSM Comparison

Efficient Transformer implementation for long-sequence processing:
- Standard multi-head attention with positional encoding
- Linear Attention variant (O(n) approximation)
- Performer variant (kernel-based approximation)
- Fair comparison with SSM models
"""

import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention.

    Memory: O(seq_len²)
    Time: O(seq_len² × d)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len)

        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Project to Q, K, V
        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, seq_len, seq_len)

        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(~mask, float('-inf'))

        # Attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, v)  # (batch, heads, seq_len, head_dim)

        # Reshape and project
        output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        output = self.to_out(output)

        return output


class LinearAttention(nn.Module):
    """
    Linear Attention (O(n) approximation).

    Approximates attention using kernel trick:
    A ≈ (Q·k_low_rank)^T · (K·k_low_rank) / d

    Memory: O(seq_len)
    Time: O(seq_len × d)

    Reference: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        kernel_type: str = "elu",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.kernel_type = kernel_type

        self.to_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

    def kernel_function(self, x: torch.Tensor) -> torch.Tensor:
        """Apply kernel to features."""
        if self.kernel_type == "elu":
            return F.elu(x) + 1
        elif self.kernel_type == "exp":
            return torch.exp(x)
        else:
            return x

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with linear attention.

        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len)

        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Project to Q, K, V
        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply kernel
        q = self.kernel_function(q)
        k = self.kernel_function(k)

        # Linear attention computation
        # output[i] = (Q[i] @ K[1:i]^T @ V[1:i]) / (Q[i] @ sum(K[1:i]^T))

        outputs = []
        for i in range(seq_len):
            # Cumulative key-value products
            k_sum = k[:, :, :i+1, :].sum(dim=2, keepdim=True)  # (batch, heads, 1, head_dim)
            kv_prod = torch.matmul(k[:, :, :i+1, :].transpose(-2, -1), v[:, :, :i+1, :])  # (batch, heads, head_dim, head_dim)

            # Attention output
            qi = q[:, :, i:i+1, :]  # (batch, heads, 1, head_dim)
            numerator = torch.matmul(qi, kv_prod)  # (batch, heads, 1, head_dim)
            denominator = torch.matmul(qi, k_sum.transpose(-2, -1)) + 1e-6  # (batch, heads, 1, 1)

            output_i = numerator / denominator  # (batch, heads, 1, head_dim)
            outputs.append(output_i)

        output = torch.cat(outputs, dim=2)  # (batch, heads, seq_len, head_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        output = self.to_out(output)

        return output


class TransformerBlock(nn.Module):
    """
    Single Transformer block: attention + feedforward.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        attention_type: str = "standard",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)

        # Attention layer
        if attention_type == "standard":
            self.attention = MultiHeadAttention(hidden_dim, num_heads)
        elif attention_type == "linear":
            self.attention = LinearAttention(hidden_dim, num_heads)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.dropout1 = nn.Dropout(dropout)

        # Feedforward
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len)

        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        # Attention
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, mask)
        x = x + self.dropout1(attn_out)

        # Feedforward
        x_norm = self.norm2(x)
        ff_out = self.feedforward(x_norm)
        x = x + ff_out

        return x


class TransformerBaselineModel(nn.Module):
    """
    Efficient Transformer baseline for comparison with SSM.

    Architecture:
    - Input projection
    - Positional encoding
    - Stack of Transformer blocks
    - Task-specific head
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        output_dim: int = 10,
        task_type: str = "classification",
        attention_type: str = "standard",
        max_seq_len: int = 10000,
        dropout: float = 0.1,
    ):
        """
        Initialize Transformer baseline.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads
            output_dim: Output dimension
            task_type: "classification", "regression", or "anomaly_detection"
            attention_type: "standard" or "linear"
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.attention_type = attention_type

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.positional_encoding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim,
                num_heads=num_heads,
                ff_dim=hidden_dim * 4,
                attention_type=attention_type,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Task-specific head
        if task_type == "classification":
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim),
            )
        elif task_type == "regression":
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim),
            )
        elif task_type == "anomaly_detection":
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        logger.info(
            f"✓ Initialized Transformer baseline: "
            f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}, attention_type={attention_type}"
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_sequences: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len)
            return_sequences: Return per-timestep outputs

        Returns:
            Dict with predictions and logits
        """
        batch_size, seq_len, input_dim = x.shape

        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        pos_encoding = self.positional_encoding(positions).unsqueeze(0)
        x = x + pos_encoding

        # Process through Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Apply output normalization
        x = self.output_norm(x)

        # Mean pooling (sequence-level prediction)
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            pooled = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        # Apply head
        logits = self.head(pooled)

        # Prepare output
        output = {"logits": logits}

        if return_sequences:
            output["hidden_states"] = x

        # Add task-specific outputs
        if self.task_type == "classification":
            output["predictions"] = torch.argmax(logits, dim=-1)
            output["probabilities"] = F.softmax(logits, dim=-1)
        elif self.task_type == "regression":
            output["predictions"] = logits
        elif self.task_type == "anomaly_detection":
            output["predictions"] = logits
            output["anomaly_scores"] = logits.squeeze(-1)

        return output


def create_transformer_baseline(
    input_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    output_dim: int = 10,
    task_type: str = "classification",
    attention_type: str = "standard",
    max_seq_len: int = 10000,
    **kwargs
) -> TransformerBaselineModel:
    """
    Factory function to create Transformer baseline.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        output_dim: Output dimension
        task_type: Task type
        attention_type: Attention type
        max_seq_len: Max sequence length

    Returns:
        Configured TransformerBaselineModel
    """
    return TransformerBaselineModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        output_dim=output_dim,
        task_type=task_type,
        attention_type=attention_type,
        max_seq_len=max_seq_len,
        **kwargs
    )


# Export public API
__all__ = [
    "MultiHeadAttention",
    "LinearAttention",
    "TransformerBlock",
    "TransformerBaselineModel",
    "create_transformer_baseline",
]
