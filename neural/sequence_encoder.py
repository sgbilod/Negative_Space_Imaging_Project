"""
Sequence Encoder for Astronomical Time Series

Encodes raw astronomical time-series data into SSM-compatible format:
- Positional encoding (time, frequency domains)
- Feature normalization and scaling
- Variable-length sequence handling
- Batch processing with padding/masking
- Efficient data loading integration
"""

import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for time-series data.

    Supports multiple encoding schemes:
    - Sinusoidal (Vaswani et al.): sin/cos at different frequencies
    - Learned: trainable positional embeddings
    - Relative: encodes time differences between samples
    """

    def __init__(
        self,
        max_seq_len: int,
        feature_dim: int,
        encoding_type: str = "sinusoidal",
        dropout: float = 0.1,
    ):
        """
        Initialize positional encoding.

        Args:
            max_seq_len: Maximum sequence length
            feature_dim: Feature/embedding dimension
            encoding_type: "sinusoidal" or "learned"
            dropout: Dropout rate
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.encoding_type = encoding_type
        self.max_seq_len = max_seq_len

        if encoding_type == "sinusoidal":
            self._create_sinusoidal_encoding(max_seq_len, feature_dim)
        elif encoding_type == "learned":
            self.positional_embedding = nn.Embedding(max_seq_len, feature_dim)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

        self.dropout = nn.Dropout(dropout)

    def _create_sinusoidal_encoding(self, max_seq_len: int, feature_dim: int):
        """Create sinusoidal positional encoding matrix."""
        position = torch.arange(max_seq_len).unsqueeze(1)  # (seq_len, 1)

        # Calculate dimension scaling
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2) * -(np.log(10000.0) / feature_dim)
        )  # (feature_dim // 2,)

        # Create encoding
        pe = torch.zeros(max_seq_len, feature_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        if feature_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # Odd dimensions
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("positional_encoding", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: (batch, seq_len, feature_dim)

        Returns:
            x + positional_encoding: (batch, seq_len, feature_dim)
        """
        seq_len = x.size(1)

        if self.encoding_type == "sinusoidal":
            pos_enc = self.positional_encoding[:seq_len, :]  # (seq_len, feature_dim)
            x = x + pos_enc.unsqueeze(0)  # (batch, seq_len, feature_dim)
        else:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            pos_enc = self.positional_embedding(positions)  # (seq_len, feature_dim)
            x = x + pos_enc.unsqueeze(0)

        return self.dropout(x)


class FrequencyEncoding(nn.Module):
    """
    Frequency-domain encoding for spectral data.

    Applies FFT to time-domain features and encodes frequency components.
    Useful for capturing periodic patterns in astronomical observations.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_fft: int = 64,
        include_phase: bool = True,
    ):
        """
        Initialize frequency encoding.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            n_fft: FFT size for frequency analysis
            include_phase: Include phase information in encoding
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_fft = n_fft
        self.include_phase = include_phase

        # Projection for frequency features
        freq_dim = n_fft // 2 + 1
        if include_phase:
            freq_features = freq_dim * 2
        else:
            freq_features = freq_dim

        # Combine time-domain and frequency-domain features
        total_features = input_dim + freq_features
        self.freq_projection = nn.Linear(total_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode frequency components.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            encoded: (batch, seq_len, output_dim)
        """
        batch_size, seq_len, input_dim = x.shape

        # Compute FFT for each feature dimension
        freq_components = []

        for i in range(input_dim):
            # FFT of i-th feature: (batch, seq_len) -> (batch, freq_bins)
            fft_out = torch.fft.rfft(x[:, :, i], n=self.n_fft, dim=-1)

            # Extract magnitude (normalized by seq_len)
            magnitude = torch.abs(fft_out) / seq_len

            if self.include_phase:
                phase = torch.angle(fft_out)
                freq_feat = torch.cat([magnitude, phase], dim=-1)
            else:
                freq_feat = magnitude

            freq_components.append(freq_feat.unsqueeze(-1))  # (batch, freq_bins, 1)

        # Stack frequency components
        freq_features = torch.cat(freq_components, dim=-1)  # (batch, freq_bins, input_dim)

        # Average over frequency bins to match sequence length
        freq_features = F.adaptive_avg_pool1d(
            freq_features.permute(0, 2, 1),  # (batch, input_dim, freq_bins)
            output_size=seq_len
        ).permute(0, 2, 1)  # (batch, seq_len, input_dim)

        # Concatenate time and frequency features
        combined = torch.cat([x, freq_features], dim=-1)  # (batch, seq_len, 2*input_dim)

        # Project to output dimension
        encoded = self.freq_projection(combined)

        return encoded


class FeatureNormalizer(nn.Module):
    """
    Adaptive feature normalization for astronomical data.

    Handles:
    - Per-channel normalization (different scales across feature dimensions)
    - Robust statistics (handles outliers/NaNs)
    - Optional learnable scaling/shifting
    """

    def __init__(
        self,
        input_dim: int,
        normalization_type: str = "standardization",
        learnable_affine: bool = True,
    ):
        """
        Initialize feature normalizer.

        Args:
            input_dim: Input feature dimension
            normalization_type: "standardization", "minmax", or "robust"
            learnable_affine: Learn scale and shift parameters
        """
        super().__init__()

        self.input_dim = input_dim
        self.normalization_type = normalization_type

        if learnable_affine:
            self.scale = nn.Parameter(torch.ones(input_dim))
            self.shift = nn.Parameter(torch.zeros(input_dim))
        else:
            self.register_buffer("scale", torch.ones(input_dim))
            self.register_buffer("shift", torch.zeros(input_dim))

        # Running statistics for batch normalization fallback
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Normalize input features.

        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
            mask: (batch, seq_len) or (batch,) to mask padding

        Returns:
            normalized: Same shape as x
        """
        # Handle 2D input
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        batch_size, seq_len, input_dim = x.shape

        if self.normalization_type == "standardization":
            # Compute mean and std
            if mask is not None:
                mask = rearrange(mask, "b s -> b s 1")
                mean = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                centered = (x - mean.unsqueeze(1)) * mask
                std = torch.sqrt((centered ** 2).sum(dim=1) / mask.sum(dim=1).clamp(min=1) + 1e-6)
            else:
                mean = x.mean(dim=1, keepdim=True)
                std = x.std(dim=1, keepdim=True) + 1e-6

            normalized = (x - mean) / std

        elif self.normalization_type == "minmax":
            # Min-max normalization to [0, 1]
            if mask is not None:
                mask = rearrange(mask, "b s -> b s 1")
                x_masked = x.clone()
                x_masked[mask == 0] = float('inf')
                x_min = x_masked.min(dim=1, keepdim=True)[0]
                x_max = x.max(dim=1, keepdim=True)[0]
            else:
                x_min = x.min(dim=1, keepdim=True)[0]
                x_max = x.max(dim=1, keepdim=True)[0]

            normalized = (x - x_min) / (x_max - x_min + 1e-6)

        elif self.normalization_type == "robust":
            # Robust normalization using median and IQR
            if mask is not None:
                x_masked = x.clone()
                x_masked[rearrange(mask, "b s -> b s 1") == 0] = float('nan')
            else:
                x_masked = x

            median = torch.nanmedian(rearrange(x_masked, "b s d -> b d s"), dim=-1)[0]
            q1 = torch.nanquantile(rearrange(x_masked, "b s d -> b d s"), 0.25, dim=-1)
            q3 = torch.nanquantile(rearrange(x_masked, "b s d -> b d s"), 0.75, dim=-1)
            iqr = (q3 - q1).clamp(min=1e-6)

            normalized = (x - median.unsqueeze(1)) / iqr.unsqueeze(1)

        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")

        # Apply learnable scale and shift
        normalized = normalized * self.scale.unsqueeze(0).unsqueeze(0) + self.shift.unsqueeze(0).unsqueeze(0)

        if squeeze:
            normalized = normalized.squeeze(1)

        return normalized


class AstronomicalSequenceEncoder(nn.Module):
    """
    Complete sequence encoder for astronomical time-series data.

    Pipeline:
    1. Feature normalization
    2. Optional frequency encoding
    3. Positional encoding
    4. Dimension projection
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        max_seq_len: int = 10000,
        use_frequency_encoding: bool = True,
        positional_encoding_type: str = "sinusoidal",
        normalization_type: str = "standardization",
        dropout: float = 0.1,
    ):
        """
        Initialize astronomical sequence encoder.

        Args:
            input_dim: Raw feature dimension (e.g., 256 for spectra)
            output_dim: Output embedding dimension for SSM
            max_seq_len: Maximum sequence length to support
            use_frequency_encoding: Include frequency domain features
            positional_encoding_type: Type of positional encoding
            normalization_type: Feature normalization strategy
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.use_frequency_encoding = use_frequency_encoding

        # Feature normalization
        self.normalizer = FeatureNormalizer(
            input_dim,
            normalization_type=normalization_type,
            learnable_affine=True,
        )

        # Frequency encoding (optional)
        if use_frequency_encoding:
            self.freq_encoder = FrequencyEncoding(
                input_dim,
                output_dim // 2,
                n_fft=min(64, max_seq_len // 8),
                include_phase=False,
            )
            encoding_output_dim = output_dim // 2
        else:
            self.freq_encoder = None
            encoding_output_dim = 0

        # Time-domain projection
        self.time_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim - encoding_output_dim),
            nn.LayerNorm(output_dim - encoding_output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            max_seq_len,
            output_dim,
            encoding_type=positional_encoding_type,
            dropout=dropout,
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode sequence.

        Args:
            x: (batch, seq_len, input_dim) raw astronomical data
            lengths: (batch,) actual sequence lengths (excluding padding)
            mask: (batch, seq_len) binary mask (1 for valid, 0 for padding)

        Returns:
            encoded: (batch, seq_len, output_dim) encoded features
            mask: (batch, seq_len) output mask
        """
        batch_size, seq_len, input_dim = x.shape

        # Validate input dimension
        if input_dim != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {input_dim}"
            )

        # Generate mask if not provided
        if mask is None:
            if lengths is not None:
                mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            else:
                mask = torch.ones(batch_size, seq_len, device=x.device, dtype=torch.bool)

        # Step 1: Normalize features
        x_norm = self.normalizer(x, mask.float() if mask.dtype == torch.bool else mask)

        # Step 2: Project to output dimension
        x_time = self.time_projection(x_norm)  # (batch, seq_len, output_dim - freq_dim)

        # Step 3: Optional frequency encoding
        if self.use_frequency_encoding:
            x_freq = self.freq_encoder(x_norm)  # (batch, seq_len, output_dim // 2)
            x_combined = torch.cat([x_time, x_freq], dim=-1)  # (batch, seq_len, output_dim)
        else:
            x_combined = x_time

        # Step 4: Add positional encoding
        x_encoded = self.positional_encoding(x_combined)  # (batch, seq_len, output_dim)

        # Step 5: Apply mask
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            x_encoded = x_encoded * mask_float

        # Step 6: Final normalization
        x_encoded = self.output_norm(x_encoded)

        return x_encoded, mask


def create_sequence_encoder(
    input_dim: int,
    output_dim: int = 256,
    max_seq_len: int = 10000,
    **kwargs
) -> AstronomicalSequenceEncoder:
    """
    Factory function to create sequence encoder.

    Args:
        input_dim: Raw feature dimension
        output_dim: Output embedding dimension
        max_seq_len: Maximum sequence length
        **kwargs: Additional arguments for encoder

    Returns:
        Configured AstronomicalSequenceEncoder
    """
    return AstronomicalSequenceEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        max_seq_len=max_seq_len,
        **kwargs
    )


# Export public API
__all__ = [
    "PositionalEncoding",
    "FrequencyEncoding",
    "FeatureNormalizer",
    "AstronomicalSequenceEncoder",
    "create_sequence_encoder",
]
