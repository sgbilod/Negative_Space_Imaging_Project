"""
SSM Inference Service

Production-ready inference service with:
- Model loading from checkpoints
- Batch and streaming inference modes
- REST API integration
- Performance monitoring
- Error handling and validation
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

from neural.ssm_astronomical_model import SSMAstronomicalModel, StreamingSSMAstronomicalModel

logger = logging.getLogger(__name__)


class SSMInferenceService:
    """
    Production inference service for SSM models.

    Supports:
    - Loading trained models
    - Batch inference
    - Streaming inference (low latency)
    - Model ensembles
    - Performance metrics
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        enable_streaming: bool = True,
    ):
        """
        Initialize inference service.

        Args:
            model_path: Path to model checkpoint
            device: Device for inference
            enable_streaming: Enable streaming inference mode
        """
        self.device = device
        self.model = None
        self.streaming_model = None
        self.model_config = None
        self.inference_stats = {
            "total_samples": 0,
            "total_time": 0,
            "batches_processed": 0,
        }

        if model_path:
            self.load_model(model_path)

        if enable_streaming:
            self.enable_streaming()

        logger.info(f"✓ Initialized SSM inference service on {device}")

    def load_model(self, model_path: str):
        """
        Load trained model from checkpoint.

        Args:
            model_path: Path to checkpoint file
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract model config if available
        if "config" in checkpoint:
            self.model_config = checkpoint["config"]
        else:
            logger.warning("Model config not found in checkpoint")
            self.model_config = {}

        # Create model
        self.model = SSMAstronomicalModel(**self.model_config)

        # Load weights
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"✓ Loaded model from {model_path}")

    def enable_streaming(self):
        """Enable streaming inference mode."""
        if self.model is None:
            logger.warning("Cannot enable streaming without loaded model")
            return

        self.streaming_model = StreamingSSMAstronomicalModel(self.model)
        logger.info("✓ Enabled streaming inference mode")

    def predict_batch(
        self,
        x: np.ndarray,
        return_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """
        Batch inference on multiple sequences.

        Args:
            x: (num_samples, seq_len, input_dim) numpy array
            return_embeddings: Return hidden state embeddings

        Returns:
            Dict with predictions and metadata
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        # Convert to tensor
        x_tensor = torch.from_numpy(x).float().to(self.device)

        self.model.eval()

        with torch.no_grad():
            start_time.record()

            output = self.model(
                x_tensor,
                return_sequences=return_embeddings,
            )

            end_time.record()
            torch.cuda.synchronize()

        # Extract predictions
        predictions = output["predictions"].cpu().numpy()
        logits = output["logits"].cpu().numpy()

        result = {
            "predictions": predictions,
            "logits": logits,
            "num_samples": x.shape[0],
            "seq_len": x.shape[1],
        }

        # Add probabilities if classification
        if "probabilities" in output:
            result["probabilities"] = output["probabilities"].cpu().numpy()

        # Add embeddings if requested
        if return_embeddings and "hidden_states" in output:
            result["embeddings"] = output["hidden_states"].cpu().numpy()

        # Compute inference time
        inference_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        result["inference_time_sec"] = inference_time
        result["latency_ms_per_sample"] = (inference_time / x.shape[0]) * 1000

        # Update stats
        self.inference_stats["total_samples"] += x.shape[0]
        self.inference_stats["total_time"] += inference_time
        self.inference_stats["batches_processed"] += 1

        return result

    def predict_streaming(
        self,
        x: np.ndarray,
        reset_state: bool = True,
    ) -> Dict[str, Any]:
        """
        Streaming inference on single or few samples.

        Maintains state across calls for sequential processing.

        Args:
            x: (num_samples, input_dim) or (num_samples, seq_len, input_dim) numpy array
            reset_state: Reset internal state before inference

        Returns:
            Dict with predictions
        """
        if self.streaming_model is None:
            raise RuntimeError("Streaming mode not enabled")

        # Reset state if requested
        if reset_state:
            batch_size = x.shape[0]
            self.streaming_model.reset_states(batch_size, self.device)

        # Convert to tensor
        x_tensor = torch.from_numpy(x).float().to(self.device)

        self.streaming_model.model.eval()

        with torch.no_grad():
            output = self.streaming_model(x_tensor)

        # Extract predictions
        predictions = output["predictions"].cpu().numpy()
        logits = output["logits"].cpu().numpy()

        result = {
            "predictions": predictions,
            "logits": logits,
        }

        if "probabilities" in output:
            result["probabilities"] = output["probabilities"].cpu().numpy()

        return result

    def predict_single(
        self,
        x: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Single sample inference.

        Args:
            x: (seq_len, input_dim) numpy array

        Returns:
            Dict with prediction
        """
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)

        return self.predict_batch(x)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {}

        info = {
            "task_type": self.model.task_type,
            "input_dim": self.model.input_dim,
            "hidden_dim": self.model.hidden_dim,
            "num_layers": self.model.num_layers,
            "output_dim": self.model.output_dim,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }

        return info

    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = self.inference_stats.copy()

        if stats["batches_processed"] > 0:
            stats["avg_time_per_batch"] = stats["total_time"] / stats["batches_processed"]
            stats["avg_time_per_sample"] = stats["total_time"] / stats["total_samples"]

        return stats


class SSMInferenceAPI:
    """
    REST API interface for SSM inference.

    Endpoints:
    - POST /api/ssm/predict: Batch inference
    - POST /api/ssm/predict_streaming: Streaming inference
    - GET /api/ssm/info: Model information
    - GET /api/ssm/stats: Inference statistics
    """

    def __init__(self, service: SSMInferenceService):
        self.service = service

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle batch prediction request.

        Request format:
        {
            "data": [[seq_1], [seq_2], ...],  # List of sequences
            "return_embeddings": bool (optional)
        }
        """
        try:
            data = request.get("data")
            if data is None:
                return {"error": "No data provided"}

            x = np.array(data, dtype=np.float32)

            return_embeddings = request.get("return_embeddings", False)

            result = self.service.predict_batch(x, return_embeddings)

            # Convert numpy arrays to lists for JSON serialization
            result = self._numpy_to_list(result)

            return {"status": "success", "result": result}

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"status": "error", "error": str(e)}

    def predict_streaming(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle streaming prediction request.

        Request format:
        {
            "data": [batch_1, batch_2, ...],
            "reset_state": bool (optional, default: True)
        }
        """
        try:
            data = request.get("data")
            if data is None:
                return {"error": "No data provided"}

            x = np.array(data, dtype=np.float32)
            reset_state = request.get("reset_state", True)

            result = self.service.predict_streaming(x, reset_state)
            result = self._numpy_to_list(result)

            return {"status": "success", "result": result}

        except Exception as e:
            logger.error(f"Streaming prediction error: {e}")
            return {"status": "error", "error": str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = self.service.get_model_info()
        return {"status": "success", "info": info}

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = self.service.get_inference_stats()
        return {"status": "success", "stats": stats}

    @staticmethod
    def _numpy_to_list(obj):
        """Recursively convert numpy arrays to lists."""
        if isinstance(obj, dict):
            return {k: SSMInferenceAPI._numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [SSMInferenceAPI._numpy_to_list(item) for item in obj]
        else:
            return obj


# Export public API
__all__ = [
    "SSMInferenceService",
    "SSMInferenceAPI",
]
