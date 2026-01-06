"""
ML Pipeline Models - Model Registry and Base Classes

Manages ML model loading, inference, and lifecycle for the pipeline.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ..core.config import ModelConfig, PipelineConfig
from ..core.pipeline import DeviceManager

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing ML models with versioning and caching.

    Features:
    - Model versioning and rollback
    - Automatic model loading and caching
    - Memory management for GPU models
    - Model health monitoring
    """

    def __init__(self, config: PipelineConfig, device_manager: DeviceManager):
        self.config = config
        self.device_manager = device_manager

        # Model storage
        self.models: Dict[str, BaseModel] = {}
        self.model_versions: Dict[str, str] = {}
        self.model_cache: Dict[str, Any] = {}

        # Model directory
        self.model_dir = Path(config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized model registry at {self.model_dir}")

    async def load_model(self, model_name: str, config: ModelConfig) -> BaseModel:
        """
        Load a model by name with configuration.

        Args:
            model_name: Name of the model to load
            config: Model configuration

        Returns:
            Loaded model instance
        """
        cache_key = self._get_cache_key(model_name, config)

        # Check cache first
        if cache_key in self.model_cache:
            logger.debug(f"Loading {model_name} from cache")
            return self.model_cache[cache_key]

        logger.info(f"Loading model: {model_name}")

        try:
            # Create model instance
            model = await self._create_model_instance(model_name, config)

            # Load model weights/parameters
            await model.load()

            # Move to device
            device = self.device_manager.get_device(config.device)
            model.to_device(device)

            # Warm up model
            await model.warmup()

            # Cache model
            self.models[model_name] = model
            self.model_versions[model_name] = config.version
            self.model_cache[cache_key] = model

            logger.info(f"Successfully loaded model: {model_name}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    async def _create_model_instance(self, model_name: str, config: ModelConfig) -> BaseModel:
        """Create a model instance based on configuration."""
        model_classes = {
            "feature_extractor": FeatureExtractorModel,
            "segmentation": SegmentationModel,
            "classification": ClassificationModel,
            "anomaly_detector": AnomalyDetectionModel,
        }

        if model_name not in model_classes:
            raise ValueError(f"Unknown model type: {model_name}")

        model_class = model_classes[model_name]

        return model_class(
            name=model_name,
            config=config,
            device_manager=self.device_manager
        )

    def _get_cache_key(self, model_name: str, config: ModelConfig) -> str:
        """Generate cache key for model."""
        config_str = f"{model_name}:{config.version}:{config.device}:{config.precision}"
        return hashlib.md5(config_str.encode()).hexdigest()

    async def unload_model(self, model_name: str) -> None:
        """Unload a model and free resources."""
        if model_name in self.models:
            model = self.models[model_name]
            await model.cleanup()

            # Remove from cache
            cache_key = self._get_cache_key(model_name, self.config.get_model_config(model_name))
            if cache_key in self.model_cache:
                del self.model_cache[cache_key]

            del self.models[model_name]
            del self.model_versions[model_name]

            logger.info(f"Unloaded model: {model_name}")

    async def get_model(self, model_name: str) -> Optional[BaseModel]:
        """Get a loaded model by name."""
        return self.models.get(model_name)

    async def list_models(self) -> Dict[str, str]:
        """List all loaded models with versions."""
        return self.model_versions.copy()

    async def register_model(
        self,
        name: str,
        model_class: type,
        config: Dict[str, Any],
        model_instance: Optional[Any] = None
    ) -> None:
        """
        Register a model instance for testing purposes.

        Args:
            name: Model name
            model_class: Model class
            config: Model configuration
            model_instance: Optional pre-loaded model instance
        """
        model_config = ModelConfig(name, **config)
        model = model_class(name=name, config=model_config, device_manager=self.device_manager)

        if model_instance:
            model.model = model_instance
            model.is_loaded = True

        self.models[name] = model
        self.model_versions[name] = model_config.version

        logger.debug(f"Registered model: {name}")


class BaseModel(ABC):
    """
    Base class for all ML models in the pipeline.

    Provides common functionality for model loading, inference, and lifecycle management.
    """

    def __init__(
        self,
        name: str,
        config: ModelConfig,
        device_manager: DeviceManager
    ):
        self.name = name
        self.config = config
        self.device_manager = device_manager

        self.model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None
        self.is_loaded = False
        self.is_warm = False

        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.last_inference_time = 0.0

        logger.debug(f"Initialized model: {name}")

    @abstractmethod
    async def load(self) -> None:
        """Load model weights and initialize."""
        pass

    @abstractmethod
    async def warmup(self) -> None:
        """Warm up the model with dummy data."""
        pass

    @abstractmethod
    async def inference(self, input_data: Any) -> Any:
        """Run inference on input data."""
        pass

    def to_device(self, device: torch.device) -> None:
        """Move model to specified device."""
        if self.model is not None:
            self.model.to(device)
        self.device = device
        logger.debug(f"Moved model {self.name} to device: {device}")

    async def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None

        self.is_loaded = False
        self.is_warm = False

        # Clear CUDA cache if needed
        if self.device and self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.debug(f"Cleaned up model: {self.name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get model performance statistics."""
        return {
            "name": self.name,
            "device": str(self.device) if self.device else None,
            "is_loaded": self.is_loaded,
            "is_warm": self.is_warm,
            "inference_count": self.inference_count,
            "total_inference_time": self.total_inference_time,
            "average_inference_time": (
                self.total_inference_time / max(1, self.inference_count)
            ),
            "last_inference_time": self.last_inference_time,
        }


class FeatureExtractorModel(BaseModel):
    """Feature extraction model using pre-trained CNN."""

    async def load(self) -> None:
        """Load ResNet-based feature extractor."""
        try:
            # Load pre-trained ResNet
            architecture = self.config.model_params.get("architecture", "resnet50")

            if architecture == "resnet50":
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
                # Remove classification head
                self.model = nn.Sequential(*list(self.model.children())[:-1])
            elif architecture == "resnet34":
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
                self.model = nn.Sequential(*list(self.model.children())[:-1])
            else:
                raise ValueError(f"Unsupported architecture: {architecture}")

            self.model.eval()
            self.is_loaded = True

            logger.info(f"Loaded feature extractor: {architecture}")

        except Exception as e:
            logger.error(f"Failed to load feature extractor: {e}")
            raise

    async def warmup(self) -> None:
        """Warm up with dummy data."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)

        if self.device:
            dummy_input = dummy_input.to(self.device)

        # Run warmup inference
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = self.model(dummy_input)

        self.is_warm = True
        logger.debug(f"Warmed up feature extractor: {self.name}")

    async def inference(self, input_data: Any) -> Any:
        """Extract features from input images."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        try:
            # Convert input to tensor
            if isinstance(input_data, dict) and "image" in input_data:
                image_tensor = input_data["image"]
            else:
                # Assume input_data is already a tensor
                image_tensor = input_data

            # Ensure tensor is on correct device
            if self.device:
                image_tensor = image_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                features = self.model(image_tensor)

            # Flatten features
            features = features.view(features.size(0), -1)

            # Convert to numpy for output
            features_np = features.cpu().numpy()

            inference_time = time.time() - start_time
            self._update_stats(inference_time)

            return {
                "features": features_np,
                "feature_dim": features_np.shape[1],
                "batch_size": features_np.shape[0],
                "inference_time": inference_time,
            }

        except Exception as e:
            logger.error(f"Feature extraction inference failed: {e}")
            raise


class SegmentationModel(BaseModel):
    """Medical image segmentation model."""

    async def load(self) -> None:
        """Load U-Net segmentation model."""
        try:
            # For now, create a simple U-Net architecture
            # In production, this would load pre-trained weights
            from torchvision.models.segmentation import fcn_resnet50

            self.model = fcn_resnet50(pretrained=False, num_classes=2)
            self.model.eval()
            self.is_loaded = True

            logger.info("Loaded segmentation model")

        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            raise

    async def warmup(self) -> None:
        """Warm up with dummy data."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        dummy_input = torch.randn(1, 3, 224, 224)

        if self.device:
            dummy_input = dummy_input.to(self.device)

        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = self.model(dummy_input)

        self.is_warm = True
        logger.debug(f"Warmed up segmentation model: {self.name}")

    async def inference(self, input_data: Any) -> Any:
        """Run segmentation on input images."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        try:
            # Convert input to tensor
            if isinstance(input_data, dict) and "image" in input_data:
                image_tensor = input_data["image"]
            else:
                image_tensor = input_data

            if self.device:
                image_tensor = image_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                output = self.model(image_tensor)

            # Get segmentation mask
            mask = output['out'].argmax(dim=1)

            # Convert to numpy
            mask_np = mask.cpu().numpy()

            inference_time = time.time() - start_time
            self._update_stats(inference_time)

            return {
                "segmentation_mask": mask_np,
                "num_classes": 2,
                "inference_time": inference_time,
            }

        except Exception as e:
            logger.error(f"Segmentation inference failed: {e}")
            raise


class ClassificationModel(BaseModel):
    """Image classification model."""

    async def load(self) -> None:
        """Load classification model."""
        try:
            # Load pre-trained EfficientNet
            architecture = self.config.model_params.get("architecture", "efficientnet_b0")
            num_classes = self.config.model_params.get("num_classes", 5)

            if architecture == "efficientnet_b0":
                self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
                # Modify final layer for our number of classes
                self.model.classifier.fc = nn.Linear(self.model.classifier.fc.in_features, num_classes)
            else:
                raise ValueError(f"Unsupported architecture: {architecture}")

            self.model.eval()
            self.is_loaded = True

            logger.info(f"Loaded classification model: {architecture}")

        except Exception as e:
            logger.error(f"Failed to load classification model: {e}")
            raise

    async def warmup(self) -> None:
        """Warm up with dummy data."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        dummy_input = torch.randn(1, 3, 224, 224)

        if self.device:
            dummy_input = dummy_input.to(self.device)

        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = self.model(dummy_input)

        self.is_warm = True
        logger.debug(f"Warmed up classification model: {self.name}")

    async def inference(self, input_data: Any) -> Any:
        """Classify input images."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        try:
            # Convert input to tensor
            if isinstance(input_data, dict) and "image" in input_data:
                image_tensor = input_data["image"]
            else:
                image_tensor = input_data

            if self.device:
                image_tensor = image_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

            # Convert to numpy
            probs_np = probabilities.cpu().numpy()
            preds_np = predictions.cpu().numpy()

            inference_time = time.time() - start_time
            self._update_stats(inference_time)

            return {
                "predictions": preds_np,
                "probabilities": probs_np,
                "inference_time": inference_time,
            }

        except Exception as e:
            logger.error(f"Classification inference failed: {e}")
            raise


class AnomalyDetectionModel(BaseModel):
    """Anomaly detection using autoencoder."""

    async def load(self) -> None:
        """Load autoencoder model."""
        try:
            # Create simple autoencoder architecture
            # In production, this would load pre-trained weights
            latent_dim = self.config.model_params.get("latent_dim", 128)

            class Autoencoder(nn.Module):
                def __init__(self, latent_dim):
                    super().__init__()
                    # Encoder
                    self.encoder = nn.Sequential(
                        nn.Conv2d(3, 32, 3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(64 * 56 * 56, latent_dim),
                    )
                    # Decoder
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, 64 * 56 * 56),
                        nn.ReLU(),
                        nn.Unflatten(1, (64, 56, 56)),
                        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
                        nn.Sigmoid(),
                    )

                def forward(self, x):
                    latent = self.encoder(x)
                    reconstructed = self.decoder(latent)
                    return reconstructed, latent

            self.model = Autoencoder(latent_dim)
            self.model.eval()
            self.is_loaded = True

            logger.info("Loaded anomaly detection model")

        except Exception as e:
            logger.error(f"Failed to load anomaly detection model: {e}")
            raise

    async def warmup(self) -> None:
        """Warm up with dummy data."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        dummy_input = torch.randn(1, 3, 224, 224)

        if self.device:
            dummy_input = dummy_input.to(self.device)

        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = self.model(dummy_input)

        self.is_warm = True
        logger.debug(f"Warmed up anomaly detection model: {self.name}")

    async def inference(self, input_data: Any) -> Any:
        """Detect anomalies in input images."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        try:
            # Convert input to tensor
            if isinstance(input_data, dict) and "image" in input_data:
                image_tensor = input_data["image"]
            else:
                image_tensor = input_data

            if self.device:
                image_tensor = image_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                reconstructed, latent = self.model(image_tensor)

            # Calculate reconstruction error
            mse_loss = nn.MSELoss(reduction='none')
            reconstruction_error = mse_loss(reconstructed, image_tensor).mean(dim=[1, 2, 3])

            # Convert to numpy
            error_np = reconstruction_error.cpu().numpy()
            latent_np = latent.cpu().numpy()

            # Determine anomalies based on threshold
            threshold = self.config.model_params.get("reconstruction_threshold", 0.1)
            anomalies = error_np > threshold

            inference_time = time.time() - start_time
            self._update_stats(inference_time)

            return {
                "anomalies": anomalies,
                "reconstruction_errors": error_np,
                "latent_features": latent_np,
                "threshold": threshold,
                "inference_time": inference_time,
            }

        except Exception as e:
            logger.error(f"Anomaly detection inference failed: {e}")
            raise

    def _update_stats(self, inference_time: float) -> None:
        """Update performance statistics."""
        self.inference_count += 1
        self.total_inference_time += inference_time
        self.last_inference_time = inference_time
