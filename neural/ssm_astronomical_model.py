"""
SSM-based Astronomical Model

Complete end-to-end model for astronomical data processing:
- Stacked SSM layers with skip connections
- Task-specific output heads (classification/regression)
- Mixed precision support (FP16/FP32)
- Dual inference modes: batch and streaming
- ONNX export for production deployment
"""

import logging
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural.state_space_models import SSMStack, StreamingSSMInference, MambaBlock
from neural.sequence_encoder import AstronomicalSequenceEncoder

logger = logging.getLogger(__name__)


class SSMAstronomicalModel(nn.Module):
    """
    Complete SSM-based model for astronomical time-series.

    Architecture:
    - Input encoder (normalization, projection)
    - Positional encoding (temporal)
    - SSM stack (4-8 layers)
    - Task-specific head
    - Optional attention mechanisms
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        output_dim: int = 10,
        task_type: str = "classification",
        max_seq_len: int = 10000,
        dropout: float = 0.1,
        use_mamba: bool = True,
        mixed_precision: bool = False,
    ):
        """
        Initialize SSM astronomical model.

        Args:
            input_dim: Input feature dimension (e.g., 256 for spectra)
            hidden_dim: Hidden state dimension for SSM
            num_layers: Number of stacked SSM layers
            output_dim: Output dimension (num_classes or 1 for regression)
            task_type: "classification", "regression", or "anomaly_detection"
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            use_mamba: Use Mamba (True) or Structured State Space (False)
            mixed_precision: Enable mixed precision training (FP16/FP32)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.task_type = task_type
        self.mixed_precision = mixed_precision

        # Input encoding
        self.encoder = AstronomicalSequenceEncoder(
            input_dim=input_dim,
            output_dim=hidden_dim,
            max_seq_len=max_seq_len,
            use_frequency_encoding=True,
            positional_encoding_type="sinusoidal",
            normalization_type="standardization",
            dropout=dropout,
        )

        # SSM stack
        self.ssm_stack = SSMStack(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_mamba=use_mamba,
        )

        # Task-specific head
        if task_type == "classification":
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim),
            )
        elif task_type == "regression":
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim),
            )
        elif task_type == "anomaly_detection":
            # Outputs anomaly scores [0, 1]
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Optional attention pooling (mean, max, or attention-weighted)
        self.pooling_type = "mean"
        if output_dim == 1 and task_type == "anomaly_detection":
            self.pooling_type = "max"  # Take max anomaly score

        # Attention pooling (optional)
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        logger.info(
            f"✓ Initialized SSM model: "
            f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}, task_type={task_type}"
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
            x: (batch, seq_len, input_dim) input sequences
            mask: (batch, seq_len) attention mask
            return_sequences: If True, return per-timestep outputs

        Returns:
            Dict with keys:
                - "logits" or "predictions": output predictions
                - "hidden_states": (batch, seq_len, hidden_dim) if return_sequences
                - "attention_weights": attention pooling weights if applicable
        """
        batch_size, seq_len, input_dim = x.shape

        # Encode input
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                x_encoded, mask_out = self.encoder(x, mask=mask)
        else:
            x_encoded, mask_out = self.encoder(x, mask=mask)

        # SSM processing
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                hidden = self.ssm_stack(x_encoded, mask=mask_out)
        else:
            hidden = self.ssm_stack(x_encoded, mask=mask_out)

        # Pooling strategy
        if self.pooling_type == "mean":
            if mask_out is not None:
                # Mean pooling excluding padded positions
                mask_float = mask_out.float().unsqueeze(-1)  # (batch, seq_len, 1)
                pooled = (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden.mean(dim=1)

        elif self.pooling_type == "max":
            if mask_out is not None:
                mask_float = mask_out.float().unsqueeze(-1)
                hidden_masked = hidden.clone()
                hidden_masked[mask_float == 0] = float('-inf')
                pooled = hidden_masked.max(dim=1)[0]
            else:
                pooled = hidden.max(dim=1)[0]

        elif self.pooling_type == "attention":
            # Attention-weighted pooling
            attention_weights = self.attention_pool(hidden)  # (batch, seq_len, 1)
            if mask_out is not None:
                mask_float = mask_out.float().unsqueeze(-1)
                attention_weights = attention_weights * mask_float
                attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            pooled = (hidden * attention_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Apply task-specific head
        logits = self.head(pooled)

        # Prepare output dict
        output = {"logits": logits}

        if return_sequences:
            output["hidden_states"] = hidden

        # Add predictions based on task type
        if self.task_type == "classification":
            output["predictions"] = torch.argmax(logits, dim=-1)
            output["probabilities"] = F.softmax(logits, dim=-1)
        elif self.task_type == "regression":
            output["predictions"] = logits
        elif self.task_type == "anomaly_detection":
            output["predictions"] = logits
            output["anomaly_scores"] = logits.squeeze(-1)

        return output

    def get_inference_model(self) -> "StreamingSSMAstronomicalModel":
        """Get streaming inference wrapper."""
        return StreamingSSMAstronomicalModel(self)


class StreamingSSMAstronomicalModel(nn.Module):
    """
    Streaming inference wrapper for low-latency production deployment.

    Enables:
    - Single-step inference with state carryover
    - O(1) memory per step
    - Real-time anomaly detection
    """

    def __init__(self, model: SSMAstronomicalModel):
        super().__init__()
        self.model = model
        self.ssm_inference = StreamingSSMInference(model.ssm_stack)

    def reset_states(self, batch_size: int = 1, device: torch.device = None):
        """Reset internal states."""
        if device is None:
            device = next(self.model.parameters()).device
        self.ssm_inference.reset_states(batch_size, device)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Process single timestep or small batch.

        Args:
            x: (batch, input_dim) or (batch, seq_len, input_dim)

        Returns:
            predictions: (batch,) for classification/anomaly, (batch, output_dim) for regression
        """
        # Handle dimensionality
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        # Encode
        x_encoded, _ = self.model.encoder(x)

        # SSM inference (streaming mode)
        hidden = self.ssm_inference(x_encoded)  # (batch, hidden_dim)

        # Apply head
        logits = self.model.head(hidden)

        if squeeze:
            logits = logits.squeeze(1)

        if self.model.task_type == "classification":
            predictions = torch.argmax(logits, dim=-1)
            return {
                "logits": logits,
                "predictions": predictions,
                "probabilities": F.softmax(logits, dim=-1),
            }
        else:
            return {
                "logits": logits,
                "predictions": logits,
            }


class SSMEnsembleModel(nn.Module):
    """
    Ensemble of SSM models for improved robustness.

    Combines predictions from multiple SSM models with different:
    - Initializations
    - Hyperparameters
    - Training data augmentations
    """

    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        ensemble_method: str = "voting",
    ):
        """
        Initialize ensemble.

        Args:
            model_configs: List of model configuration dicts
            ensemble_method: "voting" (classification) or "averaging" (regression)
        """
        super().__init__()

        self.models = nn.ModuleList([
            SSMAstronomicalModel(**config)
            for config in model_configs
        ])
        self.ensemble_method = ensemble_method

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all models.

        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len)

        Returns:
            Ensemble predictions
        """
        outputs = [model(x, mask) for model in self.models]

        if self.ensemble_method == "voting":
            # Voting for classification
            predictions = torch.stack([o["predictions"] for o in outputs], dim=1)
            ensemble_pred = torch.mode(predictions, dim=1)[0]

            # Average probabilities
            probs = torch.stack([o["probabilities"] for o in outputs], dim=0)
            ensemble_prob = probs.mean(dim=0)

            return {
                "predictions": ensemble_pred,
                "probabilities": ensemble_prob,
                "individual_predictions": predictions,
            }

        elif self.ensemble_method == "averaging":
            # Averaging for regression
            logits = torch.stack([o["logits"] for o in outputs], dim=0)
            ensemble_logits = logits.mean(dim=0)

            return {
                "predictions": ensemble_logits,
                "logits": ensemble_logits,
                "individual_logits": logits,
            }

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")


def create_ssm_model(
    input_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 4,
    output_dim: int = 10,
    task_type: str = "classification",
    max_seq_len: int = 10000,
    **kwargs
) -> SSMAstronomicalModel:
    """
    Factory function to create SSM model.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden state dimension
        num_layers: Number of SSM layers
        output_dim: Output dimension
        task_type: Task type (classification/regression/anomaly_detection)
        max_seq_len: Maximum sequence length
        **kwargs: Additional arguments

    Returns:
        Configured SSMAstronomicalModel
    """
    return SSMAstronomicalModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        task_type=task_type,
        max_seq_len=max_seq_len,
        **kwargs
    )


# Export public API
__all__ = [
    "SSMAstronomicalModel",
    "StreamingSSMAstronomicalModel",
    "SSMEnsembleModel",
    "create_ssm_model",
]
