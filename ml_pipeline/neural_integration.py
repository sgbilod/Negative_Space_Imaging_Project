"""
Neural Architecture Integration Module

Unified interface for Vision Transformer and Diffusion Model integration
into the main ml_pipeline system.

Provides:
- Model factory with configuration management
- Training pipeline orchestration
- Seamless trainer.py integration
- Experiment tracking and logging
- Model export and deployment preparation

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from pathlib import Path

logger = logging.getLogger(__name__)


class NeuralArchitectureConfig:
    """Configuration for neural architecture selection and training."""

    def __init__(
        self,
        architecture_type: str = "vit",
        model_size: str = "base",
        pretrained: bool = True,
        device: str = "cuda",
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Initialize architecture config.

        Args:
            architecture_type: "vit" or "diffusion"
            model_size: "base", "large", "small" etc
            pretrained: Use pretrained weights
            device: Device to use
            seed: Random seed
            **kwargs: Additional configuration
        """
        self.architecture_type = architecture_type
        self.model_size = model_size
        self.pretrained = pretrained
        self.device = device
        self.seed = seed
        self.extra_config = kwargs

        # Set seeds for reproducibility
        self._set_seeds()

    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        import numpy as np

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


class NeuralArchitectureFactory:
    """Factory for creating neural architecture models."""

    @staticmethod
    def create_model(
        config: NeuralArchitectureConfig,
    ) -> nn.Module:
        """
        Create neural model based on configuration.

        Args:
            config: Architecture configuration

        Returns:
            PyTorch model
        """
        logger.info(
            f"Creating {config.architecture_type} model "
            f"(size: {config.model_size})"
        )

        if config.architecture_type == "vit":
            return NeuralArchitectureFactory._create_vit(config)
        elif config.architecture_type == "diffusion":
            return NeuralArchitectureFactory._create_diffusion(config)
        else:
            raise ValueError(
                f"Unknown architecture: {config.architecture_type}"
            )

    @staticmethod
    def _create_vit(config: NeuralArchitectureConfig) -> nn.Module:
        """Create Vision Transformer model."""
        try:
            from neural.vision_transformer_integration import ViTFactory
        except ImportError as e:
            logger.error(f"Failed to import ViT: {e}")
            raise

        if config.model_size == "base":
            model = ViTFactory.create_vit_base(pretrained=config.pretrained)
        elif config.model_size == "large":
            model = ViTFactory.create_vit_large(pretrained=config.pretrained)
        elif config.model_size == "base_high_res":
            model = ViTFactory.create_vit_base_high_res(
                pretrained=config.pretrained
            )
        else:
            raise ValueError(f"Unknown ViT size: {config.model_size}")

        model.to(config.device)
        logger.info(f"Created ViT {config.model_size} model")

        return model

    @staticmethod
    def _create_diffusion(config: NeuralArchitectureConfig) -> nn.Module:
        """Create Diffusion model."""
        try:
            from neural.diffusion_model_prototype import DiffusionFactory
        except ImportError as e:
            logger.error(f"Failed to import Diffusion: {e}")
            raise

        if config.model_size == "fast":
            model = DiffusionFactory.create_model_fast()
        elif config.model_size == "high_quality":
            model = DiffusionFactory.create_model_high_quality()
        else:
            model = DiffusionFactory.create_model()

        model.to(config.device)
        logger.info(f"Created Diffusion {config.model_size} model")

        return model


class UnifiedTrainingPipeline:
    """Unified training pipeline for both ViT and Diffusion models."""

    def __init__(
        self,
        model: nn.Module,
        config: NeuralArchitectureConfig,
        output_dir: str = "./checkpoints",
    ) -> None:
        """
        Initialize unified training pipeline.

        Args:
            model: Neural model to train
            config: Architecture configuration
            output_dir: Directory for checkpoints
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.training_logs: Dict[str, Any] = {}

    def get_trainer(
        self,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        **trainer_kwargs: Any,
    ) -> Any:
        """
        Get appropriate trainer for model type.

        Args:
            learning_rate: Learning rate
            num_epochs: Number of epochs
            **trainer_kwargs: Additional trainer arguments

        Returns:
            Trainer instance
        """
        if self.config.architecture_type == "vit":
            return self._get_vit_trainer(
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                **trainer_kwargs,
            )
        elif self.config.architecture_type == "diffusion":
            return self._get_diffusion_trainer(
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                **trainer_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown architecture: {self.config.architecture_type}"
            )

    def _get_vit_trainer(
        self,
        learning_rate: float,
        num_epochs: int,
        **kwargs: Any,
    ) -> Any:
        """Get ViT trainer."""
        try:
            from ml_pipeline.training.vit_finetuner import ViTFineTuner
        except ImportError as e:
            logger.error(f"Failed to import ViTFineTuner: {e}")
            raise

        trainer = ViTFineTuner(
            model=self.model,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=self.device,
            **kwargs,
        )

        logger.info("Created ViT trainer")
        return trainer

    def _get_diffusion_trainer(
        self,
        learning_rate: float,
        num_epochs: int,
        **kwargs: Any,
    ) -> Any:
        """Get Diffusion trainer."""
        try:
            from ml_pipeline.training.diffusion_trainer import (
                DiffusionTrainer,
                DiffusionTrainingConfig,
            )
        except ImportError as e:
            logger.error(f"Failed to import DiffusionTrainer: {e}")
            raise

        config = DiffusionTrainingConfig(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=self.device,
            **kwargs,
        )

        trainer = DiffusionTrainer(
            model=self.model,
            config=config,
        )

        logger.info("Created Diffusion trainer")
        return trainer

    def save_checkpoint(self, name: str = "model.pt") -> Path:
        """
        Save model checkpoint.

        Args:
            name: Checkpoint name

        Returns:
            Path to checkpoint
        """
        path = self.output_dir / name
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint from {path}")

    def export_to_onnx(
        self,
        output_path: str = "model.onnx",
        sample_input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Path:
        """
        Export model to ONNX format.

        Args:
            output_path: Output ONNX file path
            sample_input_shape: Shape of sample input

        Returns:
            Path to ONNX file
        """
        try:
            sample_input = torch.randn(sample_input_shape).to(self.device)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if self.config.architecture_type == "vit":
                torch.onnx.export(
                    self.model,
                    sample_input,
                    str(output_path),
                    opset_version=14,
                    input_names=["image"],
                    output_names=["logits"],
                    dynamic_axes={
                        "image": {0: "batch_size"},
                        "logits": {0: "batch_size"},
                    },
                )
            else:
                # Diffusion export with timestep
                sample_timestep = torch.tensor([50]).to(self.device)
                torch.onnx.export(
                    self.model,
                    (sample_input, sample_timestep),
                    str(output_path),
                    opset_version=14,
                    input_names=["x", "t"],
                    output_names=["noise_pred"],
                )

            logger.info(f"Exported model to ONNX: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Model info dictionary
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "architecture": self.config.architecture_type,
            "model_size": self.config.model_size,
            "total_parameters": num_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "model_class": self.model.__class__.__name__,
        }


class ModelRegistry:
    """Registry for trained models and configurations."""

    def __init__(self, registry_path: str = "./model_registry.json") -> None:
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry file
        """
        self.registry_path = Path(registry_path)
        self.registry: Dict[str, Dict[str, Any]] = {}

        if self.registry_path.exists():
            import json
            with open(self.registry_path) as f:
                self.registry = json.load(f)

    def register_model(
        self,
        name: str,
        config: NeuralArchitectureConfig,
        checkpoint_path: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        Register a trained model.

        Args:
            name: Model name
            config: Architecture configuration
            checkpoint_path: Path to checkpoint
            metrics: Training metrics
        """
        self.registry[name] = {
            "architecture": config.architecture_type,
            "model_size": config.model_size,
            "checkpoint": checkpoint_path,
            "metrics": metrics,
        }

        self._save_registry()
        logger.info(f"Registered model: {name}")

    def _save_registry(self) -> None:
        """Save registry to file."""
        import json

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get registered model info.

        Args:
            name: Model name

        Returns:
            Model info or None
        """
        return self.registry.get(name)

    def list_models(self) -> list:
        """
        List all registered models.

        Returns:
            List of model names
        """
        return list(self.registry.keys())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Neural architecture integration module loaded successfully")
