"""
Machine Learning Pipeline for Negative Space Imaging Project

Integrated ML pipeline with:
- Feature extraction and engineering
- Multiple model types (classification, regression, clustering)
- Training and hyperparameter optimization
- Real-time inference serving
- Continuous model monitoring
- Model registry and versioning
"""

__version__ = "1.0.0"
__author__ = "Stephen Bilodeau"

from .core.pipeline import MLPipeline, PipelineStage, PipelineConfig
from .models.feature_extraction import FeatureExtractor
from .training.trainer import ModelTrainer
from .inference.realtime import RealtimePredictor
from .monitoring.model_monitoring import ModelMonitor
from .registry.model_registry import ModelRegistry

__all__ = [
    "MLPipeline",
    "PipelineStage",
    "PipelineConfig",
    "FeatureExtractor",
    "ModelTrainer",
    "RealtimePredictor",
    "ModelMonitor",
    "ModelRegistry",
]
