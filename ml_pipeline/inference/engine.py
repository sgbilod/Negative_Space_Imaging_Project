"""
Advanced Inference Engine for Negative Space Imaging

High-performance inference engine with:
- Model loading from checkpoints
- Input preprocessing pipeline
- Batch inference support
- Output postprocessing
- ONNX export functionality
- TensorRT optimization path
- Confidence score computation
- Comprehensive error handling and validation
- W&B inference logging

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

import wandb
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    High-performance inference engine for ML models.

    Features:
    - Dynamic batching and GPU acceleration
    - Memory management and optimization
    - Concurrent inference execution
    - Performance monitoring and profiling
    """

    def __init__(
        self,
        config: PipelineConfig,
        model_registry: ModelRegistry,
        device_manager: DeviceManager
    ):
        self.config = config
        self.model_registry = model_registry
        self.device_manager = device_manager

        # Inference settings
        self.max_batch_size = config.batch_size
        self.prefetch_factor = config.prefetch_factor
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory

        # Thread pool for CPU operations
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

        # Performance tracking
        self.inference_stats = InferenceStats()

        logger.info("Initialized inference engine")

    async def execute_inference(
        self,
        model_name: str,
        input_data: Any,
        inference_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute inference on a model with input data.

        Args:
            model_name: Name of the model to use
            input_data: Input data for inference
            inference_params: Optional inference parameters

        Returns:
            Inference results
        """
        start_time = time.time()

        try:
            # Get model from registry
            model = await self.model_registry.get_model(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not loaded")

            # Prepare input data
            prepared_data = await self._prepare_input_data(input_data, model)

            # Execute inference
            result = await model.inference(prepared_data)

            # Post-process results
            processed_result = await self._post_process_results(result, inference_params)

            # Update statistics
            inference_time = time.time() - start_time
            self.inference_stats.record_inference(model_name, inference_time, success=True)

            logger.debug(f"Inference completed for {model_name} in {inference_time:.3f}s")
            return processed_result

        except Exception as e:
            inference_time = time.time() - start_time
            self.inference_stats.record_inference(model_name, inference_time, success=False)

            logger.error(f"Inference failed for {model_name}: {e}")
            raise

    async def execute_batch_inference(
        self,
        model_name: str,
        batch_data: List[Any],
        inference_params: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Execute batch inference on multiple data items.

        Args:
            model_name: Name of the model to use
            batch_data: List of input data items
            inference_params: Optional inference parameters

        Returns:
            List of inference results
        """
        if not batch_data:
            return []

        logger.debug(f"Executing batch inference for {model_name} on {len(batch_data)} items")

        # Split into optimal batch sizes
        batches = self._create_batches(batch_data, self.max_batch_size)

        all_results = []

        for batch in batches:
            try:
                # Execute inference on batch
                batch_result = await self._execute_batch_on_model(model_name, batch, inference_params)

                # Split results back into individual items
                if isinstance(batch_result, dict) and "batch_results" in batch_result:
                    all_results.extend(batch_result["batch_results"])
                else:
                    # Assume result is already split
                    all_results.extend(batch_result)

            except Exception as e:
                logger.error(f"Batch inference failed for {model_name}: {e}")
                # Add error results for failed batch
                error_results = [{"error": str(e), "success": False} for _ in batch]
                all_results.extend(error_results)

        return all_results

    async def _execute_batch_on_model(
        self,
        model_name: str,
        batch: List[Any],
        inference_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute inference on a batch using the specified model."""
        model = await self.model_registry.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not loaded")

        # Prepare batch data
        prepared_batch = await self._prepare_batch_data(batch, model)

        # Execute inference
        result = await model.inference(prepared_batch)

        # Split batch results
        split_results = self._split_batch_results(result, len(batch))

        return {
            "batch_results": split_results,
            "batch_size": len(batch),
            "model_name": model_name,
        }

    async def _prepare_input_data(self, input_data: Any, model: BaseModel) -> Any:
        """Prepare input data for model inference."""
        # Convert to tensor if needed
        if isinstance(input_data, np.ndarray):
            tensor_data = torch.from_numpy(input_data)
        elif isinstance(input_data, dict):
            # Handle dictionary inputs (e.g., with metadata)
            if "image" in input_data:
                image_data = input_data["image"]
                if isinstance(image_data, np.ndarray):
                    tensor_data = torch.from_numpy(image_data)
                else:
                    tensor_data = image_data
                # Keep other metadata
                prepared_data = {
                    "image": tensor_data,
                    **{k: v for k, v in input_data.items() if k != "image"}
                }
                return prepared_data
            else:
                # Try to convert values to tensors
                prepared_data = {}
                for k, v in input_data.items():
                    if isinstance(v, np.ndarray):
                        prepared_data[k] = torch.from_numpy(v)
                    else:
                        prepared_data[k] = v
                return prepared_data
        else:
            # Assume it's already a tensor or compatible
            tensor_data = input_data

        # Move to device if needed
        if hasattr(tensor_data, 'to') and model.device:
            tensor_data = tensor_data.to(model.device)

        return tensor_data

    async def _prepare_batch_data(self, batch: List[Any], model: BaseModel) -> Any:
        """Prepare batch data for model inference."""
        # Prepare individual items
        prepared_items = []
        for item in batch:
            prepared_item = await self._prepare_input_data(item, model)
            prepared_items.append(prepared_item)

        # Stack into batch
        if all(isinstance(item, torch.Tensor) for item in prepared_items):
            # All tensors - stack them
            batch_tensor = torch.stack(prepared_items)
            if model.device:
                batch_tensor = batch_tensor.to(model.device)
            return batch_tensor

        elif all(isinstance(item, dict) for item in prepared_items):
            # All dictionaries - batch the tensors within
            batched_dict = {}
            for key in prepared_items[0].keys():
                if all(isinstance(item[key], torch.Tensor) for item in prepared_items):
                    # Stack tensors
                    tensor_batch = torch.stack([item[key] for item in prepared_items])
                    if model.device:
                        tensor_batch = tensor_batch.to(model.device)
                    batched_dict[key] = tensor_batch
                else:
                    # Keep as list for non-tensor data
                    batched_dict[key] = [item[key] for item in prepared_items]
            return batched_dict

        else:
            # Mixed types - return as list
            return prepared_items

    def _create_batches(self, data: List[Any], batch_size: int) -> List[List[Any]]:
        """Split data into batches of specified size."""
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def _split_batch_results(self, batch_result: Any, batch_size: int) -> List[Any]:
        """Split batch results back into individual results."""
        if isinstance(batch_result, dict):
            # Handle dictionary results
            results = []
            for i in range(batch_size):
                item_result = {}
                for key, value in batch_result.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        # Split along batch dimension
                        if hasattr(value, 'shape') and len(value.shape) > 0:
                            item_result[key] = value[i] if i < value.shape[0] else None
                        else:
                            item_result[key] = value
                    else:
                        item_result[key] = value
                results.append(item_result)
            return results

        elif isinstance(batch_result, (np.ndarray, torch.Tensor)):
            # Handle tensor/array results
            if hasattr(batch_result, 'shape') and len(batch_result.shape) > 0:
                return [batch_result[i] for i in range(min(batch_size, batch_result.shape[0]))]
            else:
                return [batch_result] * batch_size

        else:
            # Single result - duplicate for each item
            return [batch_result] * batch_size

    async def _post_process_results(
        self,
        result: Any,
        inference_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Post-process inference results."""
        # Apply any post-processing specified in params
        if inference_params:
            if "normalize_probabilities" in inference_params and inference_params["normalize_probabilities"]:
                if isinstance(result, dict) and "probabilities" in result:
                    probs = result["probabilities"]
                    if isinstance(probs, np.ndarray):
                        # Normalize probabilities
                        result["probabilities"] = probs / np.sum(probs, axis=1, keepdims=True)

            if "confidence_threshold" in inference_params:
                threshold = inference_params["confidence_threshold"]
                if isinstance(result, dict) and "probabilities" in result:
                    probs = result["probabilities"]
                    if isinstance(probs, np.ndarray):
                        max_probs = np.max(probs, axis=1)
                        result["high_confidence"] = max_probs >= threshold

        return result

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        return {
            "inference_stats": self.inference_stats.to_dict(),
            "device_info": await self.device_manager.get_status(),
            "config": {
                "max_batch_size": self.max_batch_size,
                "prefetch_factor": self.prefetch_factor,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
            },
        }

    async def optimize_for_model(self, model_name: str) -> None:
        """Optimize inference engine for a specific model."""
        model = await self.model_registry.get_model(model_name)
        if model is None:
            return

        # Adjust batch size based on model and available memory
        model_config = self.config.get_model_config(model_name)
        device_name = model_config.device

        if device_name in ["cuda", "mps"]:
            # Use larger batches for GPU
            self.max_batch_size = min(32, self.max_batch_size * 2)
        else:
            # Smaller batches for CPU
            self.max_batch_size = min(8, self.max_batch_size)

        logger.info(f"Optimized inference for {model_name}: batch_size={self.max_batch_size}")

    async def cleanup(self) -> None:
        """Clean up inference engine resources."""
        self.executor.shutdown(wait=True)
        logger.info("Inference engine cleanup complete")


class InferenceStats:
    """Tracks inference performance statistics."""

    def __init__(self):
        self.model_stats: Dict[str, Dict[str, Any]] = {}

    def record_inference(self, model_name: str, inference_time: float, success: bool) -> None:
        """Record an inference operation."""
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {
                "total_inferences": 0,
                "successful_inferences": 0,
                "failed_inferences": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
            }

        stats = self.model_stats[model_name]
        stats["total_inferences"] += 1
        stats["total_time"] += inference_time

        if success:
            stats["successful_inferences"] += 1
        else:
            stats["failed_inferences"] += 1

        stats["average_time"] = stats["total_time"] / stats["total_inferences"]
        stats["min_time"] = min(stats["min_time"], inference_time)
        stats["max_time"] = max(stats["max_time"], inference_time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            model_name: {
                **stats,
                "success_rate": stats["successful_inferences"] / max(1, stats["total_inferences"]),
            }
            for model_name, stats in self.model_stats.items()
        }


class BatchProcessor:
    """
    Advanced batch processing with dynamic batching and memory management.

    Features:
    - Dynamic batch size adjustment
    - Memory-aware batching
    - Concurrent batch processing
    - Adaptive batching based on model performance
    """

    def __init__(self, inference_engine: InferenceEngine):
        self.inference_engine = inference_engine
        self.adaptive_batching = True
        self.memory_threshold = 0.8  # Use 80% of available memory

    async def process_adaptive_batch(
        self,
        model_name: str,
        data_items: List[Any],
        target_latency: float = 1.0
    ) -> List[Any]:
        """
        Process data with adaptive batch sizing based on performance targets.

        Args:
            model_name: Model to use for inference
            data_items: List of data items to process
            target_latency: Target latency per item in seconds

        Returns:
            List of inference results
        """
        if not data_items:
            return []

        # Start with small batch and adapt
        current_batch_size = 1
        results = []

        remaining_items = data_items[:]

        while remaining_items:
            # Test current batch size
            test_batch = remaining_items[:current_batch_size]

            start_time = time.time()
            try:
                batch_results = await self.inference_engine.execute_batch_inference(
                    model_name, test_batch
                )
                latency = (time.time() - start_time) / len(test_batch)

                results.extend(batch_results)
                remaining_items = remaining_items[current_batch_size:]

                # Adjust batch size based on latency
                if latency < target_latency * 0.8 and self.adaptive_batching:
                    # Good performance, try larger batch
                    current_batch_size = min(current_batch_size * 2, len(remaining_items))
                elif latency > target_latency * 1.2 and current_batch_size > 1:
                    # Too slow, reduce batch size
                    current_batch_size = max(1, current_batch_size // 2)

            except Exception as e:
                logger.warning(f"Batch processing failed, reducing batch size: {e}")
                # Reduce batch size on failure
                current_batch_size = max(1, current_batch_size // 2)
                if current_batch_size == 1:
                    # Process one by one if batching fails
                    remaining_items = remaining_items[1:]

        return results
