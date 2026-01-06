"""
ML Pipeline Models Module

ML model implementations, registry, and management for the pipeline.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

from .registry import (
    ModelRegistry,
    BaseModel,
    FeatureExtractorModel,
    SegmentationModel,
    ClassificationModel,
    AnomalyDetectionModel,
)

__all__ = [
    # Model registry
    "ModelRegistry",

    # Base model class
    "BaseModel",

    # Specific model implementations
    "FeatureExtractorModel",
    "SegmentationModel",
    "ClassificationModel",
    "AnomalyDetectionModel",
]
