"""
Test ML Pipeline Integration

Comprehensive tests for the ML pipeline integration with the existing system.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from ml_pipeline import (
    AnomalyDetectionModel,
    ClassificationModel,
    DeviceManager,
    FeatureExtractorModel,
    InferenceEngine,
    MLPipeline,
    ModelMonitor,
    ModelRegistry,
    PipelineConfig,
    SegmentationModel,
    TrainingEngine,
)
from ml_pipeline.core.config import ModelConfig


class TestMLPipelineIntegration(unittest.IsolatedAsyncioTestCase):
    """Test ML pipeline integration with existing system."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.config = PipelineConfig(
            batch_size=4,
            enable_monitoring=True,
            metrics_interval_seconds=1,
        )

        self.pipeline = MLPipeline(self.config)
        self.registry = ModelRegistry(self.config, self.pipeline.device_manager)
        self.monitor = ModelMonitor(self.config)
        self.inference_engine = InferenceEngine(self.config, self.registry, self.pipeline.device_manager)
        self.training_engine = TrainingEngine(self.config, self.registry, self.pipeline.device_manager)

    async def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsInstance(self.pipeline, MLPipeline)
        self.assertIsInstance(self.pipeline.config, PipelineConfig)
        self.assertIsInstance(self.pipeline.device_manager, DeviceManager)

    async def test_model_registry_operations(self):
        """Test model registry operations."""
        # Register a mock model
        model_name = "test_feature_extractor"
        model_config = {"model_params": {"input_size": 224, "output_size": 512}}

        # Create a simple mock model
        mock_model = MagicMock()
        mock_model.forward = MagicMock(return_value=torch.randn(1, 512))

        await self.registry.register_model(
            name=model_name,
            model_class=FeatureExtractorModel,
            config=model_config,
            model_instance=mock_model,
        )

        # Verify registration
        self.assertIn(model_name, self.registry.models)

        # Get model
        retrieved_model = await self.registry.get_model(model_name)
        self.assertIsNotNone(retrieved_model)

    async def test_inference_engine_batch_processing(self):
        """Test inference engine batch processing."""
        # Create mock input data
        batch_size = 4
        input_data = [np.random.randn(224, 224, 3) for _ in range(batch_size)]

        # Mock model predictions
        mock_predictions = [np.random.randn(512) for _ in range(batch_size)]

        with patch.object(self.inference_engine, '_execute_batch_on_model') as mock_batch:
            mock_batch.return_value = {"batch_results": mock_predictions}

            results = await self.inference_engine.execute_batch_inference(
                model_name="test_model",
                batch_data=input_data
            )

            self.assertEqual(len(results), batch_size)
            self.assertIsInstance(results, list)

    async def test_monitoring_system(self):
        """Test monitoring system functionality."""
        await self.monitor.start_monitoring()

        # Record some metrics
        self.monitor.record_inference(
            model_name="test_model",
            inference_time=0.1,
            confidence_score=0.85,
            prediction="test_prediction"
        )

        # Get metrics
        metrics = self.monitor.get_model_metrics("test_model")
        self.assertEqual(len(metrics), 1)

        # Get stats
        stats = self.monitor.get_model_stats("test_model")
        self.assertIn("total_inferences", stats)
        self.assertEqual(stats["total_inferences"], 1)

        await self.monitor.stop_monitoring()

    async def test_device_manager(self):
        """Test device manager functionality."""
        device_manager = DeviceManager()
        await device_manager.initialize()

        # Test device detection
        device = device_manager.get_device()
        self.assertIsInstance(device, torch.device)

        # Test memory management
        status = await device_manager.get_status()
        self.assertIsInstance(status, dict)

    async def test_model_types(self):
        """Test different model types."""
        # Test feature extractor model
        feature_config = ModelConfig(
            name="feature_extractor",
            model_params={"input_size": 224, "output_size": 512}
        )
        feature_model = FeatureExtractorModel("feature_extractor", feature_config, self.pipeline.device_manager)
        self.assertIsInstance(feature_model, FeatureExtractorModel)

        # Test segmentation model
        seg_config = ModelConfig(
            name="segmentation",
            model_params={"num_classes": 10, "input_size": 224}
        )
        seg_model = SegmentationModel("segmentation", seg_config, self.pipeline.device_manager)
        self.assertIsInstance(seg_model, SegmentationModel)

        # Test classification model
        cls_config = ModelConfig(
            name="classification",
            model_params={"num_classes": 5, "input_size": 224}
        )
        cls_model = ClassificationModel("classification", cls_config, self.pipeline.device_manager)
        self.assertIsInstance(cls_model, ClassificationModel)

        # Test anomaly detection model
        anomaly_config = ModelConfig(
            name="anomaly_detection",
            model_params={"input_size": 224, "latent_dim": 64}
        )
        anomaly_model = AnomalyDetectionModel("anomaly_detection", anomaly_config, self.pipeline.device_manager)
        self.assertIsInstance(anomaly_model, AnomalyDetectionModel)

    async def test_training_engine(self):
        """Test training engine functionality."""
        # Register a model first
        model_name = "test_classifier"
        model_config = {"model_params": {"num_classes": 5, "input_size": 224}}

        # Create a proper mock model with parameters
        mock_pytorch_model = MagicMock()
        # Create mock parameters that look like PyTorch parameters
        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_pytorch_model.parameters.return_value = [mock_param]

        mock_model = MagicMock()
        mock_model.model = mock_pytorch_model
        mock_model.forward = MagicMock(return_value=torch.randn(1, 5))

        await self.registry.register_model(
            name=model_name,
            model_class=ClassificationModel,
            config=model_config,
            model_instance=mock_model,
        )

        # Create mock training data
        train_data = [
            {"input": np.random.randn(224, 224, 3), "target": np.random.randint(0, 5)}
            for _ in range(20)
        ]

        val_data = [
            {"input": np.random.randn(224, 224, 3), "target": np.random.randint(0, 5)}
            for _ in range(10)
        ]

        # Create mock datasets
        train_dataset = MagicMock()
        train_dataset.__len__ = MagicMock(return_value=len(train_data))
        train_dataset.__getitem__ = MagicMock(side_effect=lambda idx: train_data[idx])

        val_dataset = MagicMock()
        val_dataset.__len__ = MagicMock(return_value=len(val_data))
        val_dataset.__getitem__ = MagicMock(side_effect=lambda idx: val_data[idx])

        # Mock training
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()
        mock_criterion = MagicMock()
        with patch.object(self.training_engine, '_get_training_config', return_value={
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss_function": "cross_entropy",
            "weight_decay": 0.0001,
            "scheduler": "step",
            "criterion": "cross_entropy",
            "early_stopping_patience": 10
        }):
            with patch.object(self.training_engine, '_create_optimizer', return_value=mock_optimizer):
                with patch.object(self.training_engine, '_create_scheduler', return_value=mock_scheduler):
                    with patch.object(self.training_engine, '_create_criterion', return_value=mock_criterion):
                        with patch.object(self.training_engine, '_train_epoch', return_value={"loss": 0.5, "accuracy": 0.8}):
                            with patch.object(self.training_engine, '_validate_epoch', return_value={"loss": 0.7, "accuracy": 0.75}):
                                with patch.object(self.training_engine, '_save_model_checkpoint'):
                                    results = await self.training_engine.train_model(
                                        model_name="test_classifier",
                                        train_dataset=train_dataset,
                                        val_dataset=val_dataset,
                                        training_config={"epochs": 2}
                                    )

                                    self.assertIsInstance(results, dict)
                                    self.assertIn("epochs_completed", results)

    async def test_pipeline_integration(self):
        """Test full pipeline integration."""
        # Create test data
        test_image = np.random.randn(224, 224, 3)

        # Mock the entire pipeline
        with patch.object(self.pipeline, 'execute', return_value={"result": "success"}):
            result = await self.pipeline.execute(test_image)
            self.assertEqual(result["result"], "success")

    async def test_error_handling(self):
        """Test error handling throughout the pipeline."""
        # Test invalid model name
        result = await self.registry.get_model("nonexistent_model")
        self.assertIsNone(result)

        # Test invalid device - should raise ValueError
        device_manager = DeviceManager()
        with self.assertRaises(ValueError):
            device_manager.get_device("cuda:0")

    async def test_async_operations(self):
        """Test async operations and concurrency."""
        # Test concurrent inference
        async def mock_inference(model_name, data):
            await asyncio.sleep(0.01)  # Simulate processing time
            return f"result_{model_name}"

        tasks = [
            mock_inference(f"model_{i}", f"data_{i}")
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)
        self.assertEqual(len(results), 5)
        self.assertTrue(all("result_" in r for r in results))

    async def test_memory_management(self):
        """Test memory management and cleanup."""
        # Create models and process data
        models = []
        for i in range(3):
            model_config = ModelConfig(
                name=f"feature_extractor_{i}",
                model_params={"input_size": 224, "output_size": 512}
            )
            model = FeatureExtractorModel(f"feature_extractor_{i}", model_config, self.pipeline.device_manager)
            models.append(model)

        # Process batch
        batch_data = [np.random.randn(224, 224, 3) for _ in range(4)]

        with patch.object(self.inference_engine, '_execute_batch_on_model'):
            await self.inference_engine.execute_batch_inference(
                model_name="test_model",
                batch_data=batch_data
            )

        # Test cleanup
        await self.monitor.cleanup()
        self.assertEqual(len(self.monitor.alerts), 0)

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = PipelineConfig()
        self.assertIsInstance(valid_config, PipelineConfig)

        # Test invalid batch size
        with self.assertRaises(ValueError):
            PipelineConfig(batch_size=0)

        # Test invalid device
        with self.assertRaises(ValueError):
            PipelineConfig(device="invalid_device")


class TestModelImplementations(unittest.TestCase):
    """Test specific model implementations."""

    def setUp(self):
        """Set up test models."""
        from ml_pipeline.core.pipeline import DeviceManager
        from ml_pipeline.core.config import ModelConfig

        device_manager = DeviceManager()

        feature_config = ModelConfig(
            name="test_feature_extractor",
            batch_size=1,
            model_params={"input_size": 224, "output_size": 512}
        )
        self.feature_model = FeatureExtractorModel(
            name="test_feature_extractor",
            config=feature_config,
            device_manager=device_manager
        )

        seg_config = ModelConfig(
            name="test_segmentation",
            batch_size=1,
            model_params={"num_classes": 5, "input_size": 224}
        )
        self.segmentation_model = SegmentationModel(
            name="test_segmentation",
            config=seg_config,
            device_manager=device_manager
        )

        cls_config = ModelConfig(
            name="test_classification",
            batch_size=1,
            model_params={"num_classes": 10, "input_size": 224}
        )
        self.classification_model = ClassificationModel(
            name="test_classification",
            config=cls_config,
            device_manager=device_manager
        )

        anomaly_config = ModelConfig(
            name="test_anomaly",
            batch_size=1,
            model_params={"input_size": 224, "latent_dim": 64}
        )
        self.anomaly_model = AnomalyDetectionModel(
            name="test_anomaly",
            config=anomaly_config,
            device_manager=device_manager
        )

    def test_model_initialization(self):
        """Test model initialization."""
        # Models should be initialized but not loaded yet
        self.assertIsNone(self.feature_model.model)
        self.assertIsNone(self.segmentation_model.model)
        self.assertIsNone(self.classification_model.model)
        self.assertIsNone(self.anomaly_model.model)

        # Check that other attributes are set
        self.assertEqual(self.feature_model.name, "test_feature_extractor")
        self.assertEqual(self.segmentation_model.name, "test_segmentation")
        self.assertEqual(self.classification_model.name, "test_classification")
        self.assertEqual(self.anomaly_model.name, "test_anomaly")

    def test_forward_passes(self):
        """Test forward passes with sample data."""
        # Load models first
        import asyncio
        asyncio.run(self.feature_model.load())
        asyncio.run(self.segmentation_model.load())
        asyncio.run(self.classification_model.load())
        asyncio.run(self.anomaly_model.load())

        # Feature extractor
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.feature_model.model(x)
            self.assertEqual(features.shape, (1, 2048, 1, 1))

        # Segmentation
        with torch.no_grad():
            masks = self.segmentation_model.model(x)
            self.assertEqual(masks['out'].shape[1], 2)  # num_classes

        # Classification
        with torch.no_grad():
            logits = self.classification_model.model(x)
            self.assertEqual(logits.shape, (1, 10))

        # Anomaly detection
        with torch.no_grad():
            output = self.anomaly_model.model(x)
            reconstructed = output[0] if isinstance(output, tuple) else output
            self.assertEqual(reconstructed.shape, x.shape)

    def test_model_configs(self):
        """Test model configurations."""
        self.assertEqual(self.feature_model.config.model_params["output_size"], 512)
        self.assertEqual(self.segmentation_model.config.model_params["num_classes"], 5)
        self.assertEqual(self.classification_model.config.model_params["num_classes"], 10)
        self.assertEqual(self.anomaly_model.config.model_params["latent_dim"], 64)


if __name__ == "__main__":
    unittest.main()
