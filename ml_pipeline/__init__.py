"""
Negative Space Imaging ML Pipeline

Comprehensive machine learning pipeline for negative space imaging with GPU acceleration,
batch processing, model management, and monitoring.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

from .core import MLPipeline, PipelineConfig, DeviceManager
from .inference import InferenceEngine
from .models import (
    AnomalyDetectionModel,
    BaseModel,
    ClassificationModel,
    FeatureExtractorModel,
    ModelRegistry,
    SegmentationModel,
)
from .monitoring import ModelMonitor
from .training import TrainingEngine
from .deployment import ModelDeploymentPipeline

__version__ = "1.0.0"

__all__ = [
    # Core components
    "MLPipeline",
    "PipelineConfig",
    "DeviceManager",

    # Model components
    "BaseModel",
    "ModelRegistry",
    "FeatureExtractorModel",
    "SegmentationModel",
    "ClassificationModel",
    "AnomalyDetectionModel",

    # Inference components
    "InferenceEngine",

    # Training components
    "TrainingEngine",

    # Monitoring components
    "ModelMonitor",

    # Deployment components
    "ModelDeploymentPipeline",
]
