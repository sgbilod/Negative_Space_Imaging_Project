"""
ML Pipeline Configuration - Pipeline Configuration Management

Handles configuration for ML pipeline components, models, and execution parameters.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single ML model."""

    name: str
    path: Optional[str] = None
    version: str = "latest"
    device: str = "auto"  # auto, cpu, cuda, mps
    batch_size: int = 1
    precision: str = "fp32"  # fp32, fp16, int8
    max_memory_gb: Optional[float] = None
    warmup_iterations: int = 3
    timeout_seconds: float = 30.0

    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.device == "auto":
            # Auto-detect best available device
            self.device = self._detect_best_device()

        self._validate_config()

    def _detect_best_device(self) -> str:
        """Detect the best available device."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, 'mps') and torch.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        valid_devices = ["cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            raise ValueError(f"Invalid device: {self.device}. Must be one of {valid_devices}")

        valid_precisions = ["fp32", "fp16", "int8"]
        if self.precision not in valid_precisions:
            raise ValueError(f"Invalid precision: {self.precision}. Must be one of {valid_precisions}")

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")


@dataclass
class PipelineStageConfig:
    """Configuration for a pipeline stage."""

    name: str
    component_type: str
    enabled: bool = True
    timeout_seconds: float = 60.0
    retry_count: int = 1
    retry_delay_seconds: float = 1.0

    # Stage-specific parameters
    stage_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

        if self.retry_count < 0:
            raise ValueError("retry_count must be >= 0")

        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be >= 0")


@dataclass
class PipelineConfig:
    """Main configuration for the ML pipeline."""

    # Pipeline metadata
    name: str = "negative_space_ml_pipeline"
    version: str = "1.0.0"
    description: str = "ML Pipeline for Negative Space Imaging"

    # Execution settings
    device: str = "auto"
    max_concurrent_tasks: int = 4
    enable_gpu_acceleration: bool = True
    enable_mixed_precision: bool = True
    memory_limit_gb: Optional[float] = None

    # Model configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    # Pipeline stages
    stages: List[PipelineStageConfig] = field(default_factory=list)

    # Performance settings
    batch_size: int = 8
    prefetch_factor: int = 2
    num_workers: int = 2
    pin_memory: bool = True

    # Monitoring and logging
    enable_monitoring: bool = True
    log_level: str = "INFO"
    metrics_interval_seconds: float = 10.0

    # Paths
    model_dir: str = "./models"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"

    # Advanced settings
    enable_profiling: bool = False
    enable_model_optimization: bool = True
    enable_quantization: bool = False

    def __post_init__(self) -> None:
        """Validate and initialize configuration after creation."""
        self._validate_config()
        self._initialize_defaults()
        self._setup_paths()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks must be >= 1")

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}. Must be one of {valid_log_levels}")

        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be > 0")

    def _initialize_defaults(self) -> None:
        """Initialize default configurations."""
        if not self.models:
            self._setup_default_models()

        if not self.stages:
            self._setup_default_stages()

    def _setup_default_models(self) -> None:
        """Setup default model configurations."""
        self.models = {
            "feature_extractor": ModelConfig(
                name="feature_extractor",
                device=self.device,
                batch_size=self.batch_size,
                model_params={
                    "architecture": "resnet50",
                    "pretrained": True,
                    "feature_dim": 2048
                }
            ),
            "segmentation": ModelConfig(
                name="segmentation",
                device=self.device,
                batch_size=self.batch_size,
                model_params={
                    "architecture": "unet",
                    "num_classes": 2,
                    "encoder": "resnet34"
                }
            ),
            "classification": ModelConfig(
                name="classification",
                device=self.device,
                batch_size=self.batch_size,
                model_params={
                    "architecture": "efficientnet_b0",
                    "num_classes": 5,
                    "pretrained": True
                }
            ),
            "anomaly_detector": ModelConfig(
                name="anomaly_detector",
                device=self.device,
                batch_size=self.batch_size,
                model_params={
                    "architecture": "autoencoder",
                    "latent_dim": 128,
                    "reconstruction_threshold": 0.1
                }
            )
        }

    def _setup_default_stages(self) -> None:
        """Setup default pipeline stages."""
        self.stages = [
            PipelineStageConfig(
                name="data_loading",
                component_type="DataLoaderComponent",
                stage_params={"supported_formats": ["jpg", "png", "dicom"]}
            ),
            PipelineStageConfig(
                name="preprocessing",
                component_type="PreprocessingComponent",
                stage_params={"normalization": "standard"}
            ),
            PipelineStageConfig(
                name="feature_extraction",
                component_type="FeatureExtractionModel",
                stage_params={"model_name": "feature_extractor"}
            ),
            PipelineStageConfig(
                name="segmentation",
                component_type="SegmentationModel",
                stage_params={"model_name": "segmentation"}
            ),
            PipelineStageConfig(
                name="classification",
                component_type="ClassificationModel",
                stage_params={"model_name": "classification"}
            ),
            PipelineStageConfig(
                name="anomaly_detection",
                component_type="AnomalyDetectionModel",
                stage_params={"model_name": "anomaly_detector"}
            ),
            PipelineStageConfig(
                name="postprocessing",
                component_type="PostprocessingComponent",
                stage_params={"output_format": "structured"}
            ),
            PipelineStageConfig(
                name="validation",
                component_type="ValidationComponent",
                stage_params={"validation_rules": ["confidence_threshold", "spatial_consistency"]}
            )
        ]

    def _setup_paths(self) -> None:
        """Setup and validate paths."""
        paths = [self.model_dir, self.cache_dir, self.log_dir]
        for path_str in paths:
            path = Path(path_str)
            path.mkdir(parents=True, exist_ok=True)

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        return self.models[model_name]

    def get_stage_config(self, stage_name: str) -> PipelineStageConfig:
        """Get configuration for a specific stage."""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        raise ValueError(f"Stage '{stage_name}' not found in configuration")

    def is_stage_enabled(self, stage_name: str) -> bool:
        """Check if a stage is enabled."""
        try:
            stage = self.get_stage_config(stage_name)
            return stage.enabled
        except ValueError:
            return False

    def get_enabled_stages(self) -> List[PipelineStageConfig]:
        """Get all enabled stages."""
        return [stage for stage in self.stages if stage.enabled]

    def update_model_config(self, model_name: str, updates: Dict[str, Any]) -> None:
        """Update configuration for a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        current_config = self.models[model_name]
        for key, value in updates.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)
            else:
                current_config.model_params[key] = value

        # Re-validate
        current_config._validate_config()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "device": self.device,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "enable_gpu_acceleration": self.enable_gpu_acceleration,
            "enable_mixed_precision": self.enable_mixed_precision,
            "memory_limit_gb": self.memory_limit_gb,
            "models": {name: self._model_config_to_dict(config) for name, config in self.models.items()},
            "stages": [self._stage_config_to_dict(stage) for stage in self.stages],
            "batch_size": self.batch_size,
            "prefetch_factor": self.prefetch_factor,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "enable_monitoring": self.enable_monitoring,
            "log_level": self.log_level,
            "metrics_interval_seconds": self.metrics_interval_seconds,
            "model_dir": self.model_dir,
            "cache_dir": self.cache_dir,
            "log_dir": self.log_dir,
            "enable_profiling": self.enable_profiling,
            "enable_model_optimization": self.enable_model_optimization,
            "enable_quantization": self.enable_quantization,
        }

    def _model_config_to_dict(self, config: ModelConfig) -> Dict[str, Any]:
        """Convert ModelConfig to dictionary."""
        return {
            "name": config.name,
            "path": config.path,
            "version": config.version,
            "device": config.device,
            "batch_size": config.batch_size,
            "precision": config.precision,
            "max_memory_gb": config.max_memory_gb,
            "warmup_iterations": config.warmup_iterations,
            "timeout_seconds": config.timeout_seconds,
            "model_params": config.model_params,
        }

    def _stage_config_to_dict(self, config: PipelineStageConfig) -> Dict[str, Any]:
        """Convert PipelineStageConfig to dictionary."""
        return {
            "name": config.name,
            "component_type": config.component_type,
            "enabled": config.enabled,
            "timeout_seconds": config.timeout_seconds,
            "retry_count": config.retry_count,
            "retry_delay_seconds": config.retry_delay_seconds,
            "stage_params": config.stage_params,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> PipelineConfig:
        """Create configuration from dictionary."""
        # Convert model configs
        models = {}
        for name, model_dict in config_dict.get("models", {}).items():
            models[name] = ModelConfig(**model_dict)

        # Convert stage configs
        stages = []
        for stage_dict in config_dict.get("stages", []):
            stages.append(PipelineStageConfig(**stage_dict))

        # Create main config
        config_dict_copy = config_dict.copy()
        config_dict_copy["models"] = models
        config_dict_copy["stages"] = stages

        return cls(**config_dict_copy)

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        import yaml

        config_dict = self.to_dict()
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {file_path}")

    @classmethod
    def load_from_file(cls, file_path: str) -> PipelineConfig:
        """Load configuration from YAML file."""
        import yaml

        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)
