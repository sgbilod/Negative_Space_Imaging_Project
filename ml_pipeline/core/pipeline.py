"""
ML Pipeline Core - Main Pipeline Orchestration

Orchestrates ML pipeline execution with GPU acceleration, async processing, and agent integration.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from .config import PipelineConfig, PipelineStageConfig
from .stages import (
    PipelineComponent,
    PipelineStage,
    DataLoaderComponent,
    PreprocessingComponent,
    ModelComponent,
    PostprocessingComponent,
    ValidationComponent
)

logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Main ML Pipeline orchestrator for negative space imaging.

    Features:
    - GPU-accelerated ML inference
    - Async processing with concurrent task execution
    - Component-based architecture with dependency management
    - Comprehensive monitoring and error handling
    - Integration with agent supervisor framework
    """

    def __init__(
        self,
        config: PipelineConfig,
        device_manager: Optional['DeviceManager'] = None
    ):
        """
        Initialize the ML pipeline.

        Args:
            config: Pipeline configuration
            device_manager: Optional device manager for GPU allocation
        """
        self.config = config
        self.device_manager = device_manager or DeviceManager()

        # Component registry
        self.components: Dict[str, PipelineComponent] = {}
        self.stage_components: Dict[PipelineStage, PipelineComponent] = {}

        # Execution state
        self.is_initialized = False
        self.is_running = False
        self.execution_stats = PipelineExecutionStats()

        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)

        # Monitoring
        self.monitor = PipelineMonitor(config) if config.enable_monitoring else None

        logger.info(f"Initialized ML Pipeline: {config.name} v{config.version}")

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        if self.is_initialized:
            logger.warning("Pipeline already initialized")
            return

        logger.info("Initializing ML Pipeline components...")

        try:
            # Initialize device manager
            await self.device_manager.initialize()

            # Create and initialize components
            await self._create_components()
            await self._initialize_components()

            # Setup monitoring
            if self.monitor:
                await self.monitor.start()

            self.is_initialized = True
            logger.info("ML Pipeline initialization complete")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            await self.cleanup()
            raise

    async def _create_components(self) -> None:
        """Create pipeline components based on configuration."""
        enabled_stages = self.config.get_enabled_stages()

        for stage_config in enabled_stages:
            try:
                component = await self._create_component(stage_config)
                self.components[stage_config.name] = component

                # Map stage to component
                stage = PipelineStage[stage_config.name.upper()]
                self.stage_components[stage] = component

                logger.debug(f"Created component: {stage_config.name}")

            except Exception as e:
                logger.error(f"Failed to create component {stage_config.name}: {e}")
                raise

    async def _create_component(self, stage_config: PipelineStageConfig) -> PipelineComponent:
        """Create a component instance based on configuration."""
        component_type = stage_config.component_type

        # Map component types to classes
        component_classes = {
            "DataLoaderComponent": DataLoaderComponent,
            "PreprocessingComponent": PreprocessingComponent,
            "FeatureExtractionModel": ModelComponent,
            "SegmentationModel": ModelComponent,
            "ClassificationModel": ModelComponent,
            "AnomalyDetectionModel": ModelComponent,
            "PostprocessingComponent": PostprocessingComponent,
            "ValidationComponent": ValidationComponent,
        }

        if component_type not in component_classes:
            raise ValueError(f"Unknown component type: {component_type}")

        component_class = component_classes[component_type]

        # Create component with configuration
        component = component_class(
            name=stage_config.name,
            config=stage_config,
            pipeline_config=self.config,
            device_manager=self.device_manager
        )

        return component

    async def _initialize_components(self) -> None:
        """Initialize all created components."""
        init_tasks = []

        for component in self.components.values():
            task = asyncio.create_task(component.initialize())
            init_tasks.append(task)

        # Wait for all components to initialize
        await asyncio.gather(*init_tasks, return_exceptions=False)

        logger.info(f"Initialized {len(self.components)} pipeline components")

    async def execute(
        self,
        input_data: Any,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the ML pipeline on input data.

        Args:
            input_data: Input data for processing
            execution_context: Optional execution context

        Returns:
            Pipeline execution results
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        if self.is_running:
            raise RuntimeError("Pipeline is already running")

        self.is_running = True
        start_time = time.time()

        try:
            logger.info("Starting pipeline execution")

            # Setup execution context
            context = execution_context or {}
            context.update({
                "pipeline_start_time": start_time,
                "input_data": input_data,
                "execution_stats": self.execution_stats,
            })

            # Execute pipeline stages
            result = await self._execute_pipeline(input_data, context)

            # Update statistics
            execution_time = time.time() - start_time
            self.execution_stats.record_execution(execution_time, success=True)

            logger.info(".2f")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.execution_stats.record_execution(execution_time, success=False)

            logger.error(f"Pipeline execution failed: {e}")
            raise

        finally:
            self.is_running = False

    async def _execute_pipeline(
        self,
        input_data: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute pipeline stages in order."""
        current_data = input_data
        stage_results = {}

        # Execute stages in order
        execution_order = [
            PipelineStage.DATA_LOADING,
            PipelineStage.PREPROCESSING,
            PipelineStage.FEATURE_EXTRACTION,
            PipelineStage.SEGMENTATION,
            PipelineStage.CLASSIFICATION,
            PipelineStage.ANOMALY_DETECTION,
            PipelineStage.POSTPROCESSING,
            PipelineStage.VALIDATION,
        ]

        for stage in execution_order:
            if stage not in self.stage_components:
                continue

            component = self.stage_components[stage]
            stage_name = stage.name.lower()

            try:
                logger.debug(f"Executing stage: {stage_name}")

                # Execute component with timeout
                stage_config = self.config.get_stage_config(stage_name)
                timeout = stage_config.timeout_seconds

                result = await asyncio.wait_for(
                    component.execute(current_data, context),
                    timeout=timeout
                )

                # Store result and update data for next stage
                stage_results[stage_name] = result
                current_data = result.get("output_data", result)

                # Update monitoring
                if self.monitor:
                    await self.monitor.record_stage_execution(
                        stage_name, time.time(), success=True
                    )

            except asyncio.TimeoutError:
                logger.error(f"Stage {stage_name} timed out after {timeout}s")
                raise
            except Exception as e:
                logger.error(f"Stage {stage_name} failed: {e}")

                if self.monitor:
                    await self.monitor.record_stage_execution(
                        stage_name, time.time(), success=False, error=str(e)
                    )

                raise

        return {
            "final_result": current_data,
            "stage_results": stage_results,
            "execution_context": context,
        }

    async def execute_batch(
        self,
        batch_data: List[Any],
        execution_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute pipeline on a batch of data with concurrent processing.

        Args:
            batch_data: List of input data items
            execution_context: Optional execution context

        Returns:
            List of pipeline execution results
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        logger.info(f"Executing pipeline on batch of {len(batch_data)} items")

        # Create tasks for concurrent execution
        tasks = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        async def execute_with_semaphore(data_item: Any) -> Dict[str, Any]:
            async with semaphore:
                return await self.execute(data_item, execution_context)

        for data_item in batch_data:
            task = asyncio.create_task(execute_with_semaphore(data_item))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed: {result}")
                processed_results.append({
                    "error": str(result),
                    "batch_index": i,
                    "success": False
                })
            else:
                processed_results.append(result)

        return processed_results

    async def warmup(self) -> None:
        """Warm up the pipeline by running inference on dummy data."""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        logger.info("Warming up ML Pipeline...")

        # Create dummy data for warmup
        dummy_data = self._create_dummy_data()

        try:
            # Run warmup iterations
            for i in range(self.config.models.get("feature_extractor", ModelConfig("dummy")).warmup_iterations):
                await self.execute(dummy_data, {"warmup": True, "iteration": i})

            logger.info("Pipeline warmup complete")

        except Exception as e:
            logger.warning(f"Pipeline warmup failed: {e}")

    def _create_dummy_data(self) -> Any:
        """Create dummy data for pipeline warmup."""
        # This should be implemented based on expected input format
        # For now, return a simple dict
        return {"dummy": True, "shape": (224, 224, 3)}

    async def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "initialized": self.is_initialized,
            "running": self.is_running,
            "config": self.config.to_dict(),
            "execution_stats": self.execution_stats.to_dict(),
            "components": {
                name: await component.get_status()
                for name, component in self.components.items()
            },
            "device_info": await self.device_manager.get_status(),
        }

    async def cleanup(self) -> None:
        """Clean up pipeline resources."""
        logger.info("Cleaning up ML Pipeline...")

        # Stop monitoring
        if self.monitor:
            await self.monitor.stop()

        # Clean up components
        cleanup_tasks = []
        for component in self.components.values():
            task = asyncio.create_task(component.cleanup())
            cleanup_tasks.append(task)

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Clean up device manager
        await self.device_manager.cleanup()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        self.is_initialized = False
        logger.info("ML Pipeline cleanup complete")

    @asynccontextmanager
    async def pipeline_context(self):
        """Context manager for pipeline execution."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()


class DeviceManager:
    """Manages GPU and CPU device allocation for the pipeline."""

    def __init__(self):
        self.devices: Dict[str, torch.device] = {}
        self.memory_limits: Dict[str, float] = {}
        self.current_memory_usage: Dict[str, float] = {}

    async def initialize(self) -> None:
        """Initialize device management."""
        # Detect available devices
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            for i in range(cuda_count):
                device_name = f"cuda:{i}"
                self.devices[device_name] = torch.device(device_name)
                self.memory_limits[device_name] = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                self.current_memory_usage[device_name] = 0.0

        if hasattr(torch, 'mps') and torch.mps.is_available():
            self.devices["mps"] = torch.device("mps")
            self.memory_limits["mps"] = 0.0  # MPS doesn't report memory limits
            self.current_memory_usage["mps"] = 0.0

        self.devices["cpu"] = torch.device("cpu")
        self.memory_limits["cpu"] = 0.0  # No memory limit for CPU
        self.current_memory_usage["cpu"] = 0.0

        logger.info(f"Initialized device manager with devices: {list(self.devices.keys())}")

    def get_device(self, device_name: str = "auto") -> torch.device:
        """Get a device by name."""
        if device_name == "auto":
            # Return best available device
            if "cuda:0" in self.devices:
                return self.devices["cuda:0"]
            elif "mps" in self.devices:
                return self.devices["mps"]
            else:
                return self.devices["cpu"]

        if device_name not in self.devices:
            raise ValueError(f"Device {device_name} not available")

        return self.devices[device_name]

    async def allocate_memory(self, device_name: str, memory_gb: float) -> bool:
        """Attempt to allocate memory on a device."""
        if device_name not in self.memory_limits:
            return True  # No limit for this device

        current_usage = self.current_memory_usage[device_name]
        limit = self.memory_limits[device_name]

        if limit > 0 and current_usage + memory_gb > limit:
            return False  # Not enough memory

        self.current_memory_usage[device_name] += memory_gb
        return True

    def free_memory(self, device_name: str, memory_gb: float) -> None:
        """Free allocated memory on a device."""
        if device_name in self.current_memory_usage:
            self.current_memory_usage[device_name] = max(
                0, self.current_memory_usage[device_name] - memory_gb
            )

    async def get_status(self) -> Dict[str, Any]:
        """Get device status information."""
        return {
            "available_devices": list(self.devices.keys()),
            "memory_limits": self.memory_limits.copy(),
            "current_memory_usage": self.current_memory_usage.copy(),
        }

    async def cleanup(self) -> None:
        """Clean up device resources."""
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.devices.clear()
        self.memory_limits.clear()
        self.current_memory_usage.clear()


class PipelineExecutionStats:
    """Tracks pipeline execution statistics."""

    def __init__(self):
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time = 0.0
        self.average_execution_time = 0.0
        self.min_execution_time = float('inf')
        self.max_execution_time = 0.0

    def record_execution(self, execution_time: float, success: bool) -> None:
        """Record a pipeline execution."""
        self.total_executions += 1
        self.total_execution_time += execution_time

        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        self.average_execution_time = self.total_execution_time / self.total_executions
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": self.successful_executions / max(1, self.total_executions),
            "total_execution_time": self.total_execution_time,
            "average_execution_time": self.average_execution_time,
            "min_execution_time": self.min_execution_time if self.min_execution_time != float('inf') else 0.0,
            "max_execution_time": self.max_execution_time,
        }


class PipelineMonitor:
    """Monitors pipeline execution and performance."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.is_running = False
        self.metrics: Dict[str, Any] = {}

    async def start(self) -> None:
        """Start monitoring."""
        self.is_running = True
        logger.info("Pipeline monitoring started")

    async def stop(self) -> None:
        """Stop monitoring."""
        self.is_running = False
        logger.info("Pipeline monitoring stopped")

    async def record_stage_execution(
        self,
        stage_name: str,
        timestamp: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Record stage execution metrics."""
        if not self.is_running:
            return

        metric = {
            "stage": stage_name,
            "timestamp": timestamp,
            "success": success,
            "error": error,
        }

        if stage_name not in self.metrics:
            self.metrics[stage_name] = []

        self.metrics[stage_name].append(metric)

        # Keep only recent metrics
        max_metrics = 1000
        if len(self.metrics[stage_name]) > max_metrics:
            self.metrics[stage_name] = self.metrics[stage_name][-max_metrics:]
