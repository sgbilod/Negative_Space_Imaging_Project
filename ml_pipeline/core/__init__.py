"""
ML Pipeline Core Module

Core components for ML pipeline orchestration, configuration, and execution.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

from .config import (
    ModelConfig,
    PipelineConfig,
    PipelineStageConfig,
)

from .pipeline import (
    MLPipeline,
    DeviceManager,
    PipelineExecutionStats,
    PipelineMonitor,
)

from .async_pipeline import (
    AsyncProcessingPipeline,
    AsyncTask,
    ProcessingStats,
)

__all__ = [
    # Configuration
    "ModelConfig",
    "PipelineConfig",
    "PipelineStageConfig",

    # Pipeline orchestration
    "MLPipeline",
    "AsyncProcessingPipeline",
    "DeviceManager",
    "PipelineExecutionStats",
    "PipelineMonitor",

    # Async processing
    "AsyncTask",
    "ProcessingStats",

    # Pipeline components
    "PipelineComponent",
    "PipelineStage",
    "DataLoaderComponent",
    "PreprocessingComponent",
    "ModelComponent",
    "PostprocessingComponent",
    "ValidationComponent",
]
