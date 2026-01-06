"""
Integration Tests for Vision Transformer and Diffusion Models

Comprehensive test suite validating:
- Model creation and initialization
- Training pipeline functionality
- Inference and generation
- API endpoints
- End-to-end workflows

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TestVisionTransformerIntegration(unittest.TestCase):
    """Test Vision Transformer integration."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.input_size = 224
        cls.batch_size = 2

    def test_vit_import(self) -> None:
        """Test ViT module import."""
        try:
            from neural.vision_transformer_integration import (
                VisionTransformer,
                ViTFactory,
            )
            self.assertIsNotNone(VisionTransformer)
            self.assertIsNotNone(ViTFactory)
        except ImportError as e:
            self.fail(f"Failed to import ViT modules: {e}")

    def test_vit_creation_base(self) -> None:
        """Test ViT base model creation."""
        try:
            from neural.vision_transformer_integration import ViTFactory

            model = ViTFactory.create_vit_base()
            self.assertIsNotNone(model)
            model.to(self.device)
            logger.info("✓ ViT Base model created successfully")
        except Exception as e:
            self.fail(f"ViT base creation failed: {e}")

    def test_vit_creation_large(self) -> None:
        """Test ViT large model creation."""
        try:
            from neural.vision_transformer_integration import ViTFactory

            model = ViTFactory.create_vit_large()
            self.assertIsNotNone(model)
            model.to(self.device)
            logger.info("✓ ViT Large model created successfully")
        except Exception as e:
            self.fail(f"ViT large creation failed: {e}")

    def test_vit_forward_pass(self) -> None:
        """Test ViT forward pass."""
        try:
            from neural.vision_transformer_integration import ViTFactory

            model = ViTFactory.create_vit_base()
            model.to(self.device)
            model.eval()

            # Create dummy input
            x = torch.randn(
                self.batch_size,
                3,
                self.input_size,
                self.input_size,
                device=self.device
            )

            with torch.no_grad():
                output = model(x)

            self.assertEqual(output.shape[0], self.batch_size)
            logger.info(
                f"✓ ViT forward pass successful, "
                f"output shape: {output.shape}"
            )
        except Exception as e:
            self.fail(f"ViT forward pass failed: {e}")

    def test_vit_finetuner_creation(self) -> None:
        """Test ViT fine-tuner creation."""
        try:
            from neural.vision_transformer_integration import ViTFactory
            from ml_pipeline.training.vit_finetuner import ViTFineTuner

            model = ViTFactory.create_vit_base()
            finetuner = ViTFineTuner(
                model=model,
                learning_rate=1e-4,
                num_epochs=2,
                device=self.device,
            )
            self.assertIsNotNone(finetuner)
            logger.info("✓ ViT fine-tuner created successfully")
        except Exception as e:
            self.fail(f"ViT fine-tuner creation failed: {e}")


class TestDiffusionIntegration(unittest.TestCase):
    """Test Diffusion model integration."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.image_size = 64
        cls.batch_size = 2

    def test_diffusion_import(self) -> None:
        """Test Diffusion module import."""
        try:
            from neural.diffusion_model_prototype import (
                DiffusionModel,
                DiffusionFactory,
            )
            self.assertIsNotNone(DiffusionModel)
            self.assertIsNotNone(DiffusionFactory)
        except ImportError as e:
            self.fail(f"Failed to import Diffusion modules: {e}")

    def test_diffusion_creation(self) -> None:
        """Test Diffusion model creation."""
        try:
            from neural.diffusion_model_prototype import DiffusionFactory

            model = DiffusionFactory.create_model()
            self.assertIsNotNone(model)
            model.to(self.device)
            logger.info("✓ Diffusion model created successfully")
        except Exception as e:
            self.fail(f"Diffusion creation failed: {e}")

    def test_diffusion_forward_pass(self) -> None:
        """Test Diffusion forward pass."""
        try:
            from neural.diffusion_model_prototype import DiffusionFactory

            model = DiffusionFactory.create_model()
            model.to(self.device)
            model.eval()

            # Create dummy input
            x = torch.randn(
                self.batch_size,
                3,
                self.image_size,
                self.image_size,
                device=self.device
            )
            t = torch.randint(0, 1000, (self.batch_size,), device=self.device)

            with torch.no_grad():
                noise_pred = model(x, t)

            self.assertEqual(noise_pred.shape, x.shape)
            logger.info(
                f"✓ Diffusion forward pass successful, "
                f"output shape: {noise_pred.shape}"
            )
        except Exception as e:
            self.fail(f"Diffusion forward pass failed: {e}")

    def test_diffusion_sampling(self) -> None:
        """Test Diffusion sampling."""
        try:
            from neural.diffusion_model_prototype import DiffusionFactory

            model = DiffusionFactory.create_model_fast()  # Use fast for testing
            model.to(self.device)
            model.eval()

            with torch.no_grad():
                samples = model.sample(num_samples=2, num_steps=10)

            self.assertEqual(samples.shape[0], 2)
            logger.info(
                f"✓ Diffusion sampling successful, "
                f"sample shape: {samples.shape}"
            )
        except Exception as e:
            self.fail(f"Diffusion sampling failed: {e}")

    def test_diffusion_trainer_creation(self) -> None:
        """Test Diffusion trainer creation."""
        try:
            from neural.diffusion_model_prototype import DiffusionFactory
            from ml_pipeline.training.diffusion_trainer import (
                DiffusionTrainer,
                DiffusionTrainingConfig,
            )

            model = DiffusionFactory.create_model()
            config = DiffusionTrainingConfig(
                learning_rate=1e-4,
                num_epochs=2,
                device=self.device,
            )
            trainer = DiffusionTrainer(model=model, config=config)
            self.assertIsNotNone(trainer)
            logger.info("✓ Diffusion trainer created successfully")
        except Exception as e:
            self.fail(f"Diffusion trainer creation failed: {e}")


class TestSyntheticDataGeneration(unittest.TestCase):
    """Test synthetic data generation."""

    def test_data_generator_import(self) -> None:
        """Test data generator import."""
        try:
            from scripts.generate_synthetic_data import (
                AstronomicalImageGenerator,
                SyntheticAstronomicalDataset,
            )
            self.assertIsNotNone(AstronomicalImageGenerator)
            self.assertIsNotNone(SyntheticAstronomicalDataset)
        except ImportError as e:
            self.fail(f"Failed to import data modules: {e}")

    def test_image_generation(self) -> None:
        """Test synthetic image generation."""
        try:
            from scripts.generate_synthetic_data import (
                AstronomicalImageGenerator,
            )

            generator = AstronomicalImageGenerator(image_size=64)
            image = generator.generate_synthetic_image()

            self.assertEqual(image.shape, (64, 64, 3))
            self.assertTrue(np.all(image >= 0))
            self.assertTrue(np.all(image <= 1))
            logger.info("✓ Synthetic image generation successful")
        except Exception as e:
            self.fail(f"Image generation failed: {e}")

    def test_dataset_creation(self) -> None:
        """Test synthetic dataset creation."""
        try:
            from scripts.generate_synthetic_data import (
                SyntheticAstronomicalDataset,
            )

            dataset = SyntheticAstronomicalDataset(
                num_samples=10,
                image_size=64,
            )

            self.assertEqual(len(dataset), 10)

            # Test single sample
            sample = dataset[0]
            self.assertIsInstance(sample, torch.Tensor)
            self.assertEqual(sample.shape, (3, 64, 64))

            logger.info("✓ Synthetic dataset creation successful")
        except Exception as e:
            self.fail(f"Dataset creation failed: {e}")


class TestBenchmarking(unittest.TestCase):
    """Test benchmarking suite."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_benchmark_import(self) -> None:
        """Test benchmark module import."""
        try:
            from scripts.benchmark_vit import (
                ModelBenchmark,
                SimpleCNN,
                SimpleResNet,
            )
            self.assertIsNotNone(ModelBenchmark)
            self.assertIsNotNone(SimpleCNN)
            self.assertIsNotNone(SimpleResNet)
        except ImportError as e:
            self.fail(f"Failed to import benchmark modules: {e}")

    def test_baseline_models(self) -> None:
        """Test baseline model creation."""
        try:
            from scripts.benchmark_vit import SimpleCNN, SimpleResNet

            cnn = SimpleCNN(num_classes=10)
            resnet = SimpleResNet(num_classes=10)

            self.assertIsNotNone(cnn)
            self.assertIsNotNone(resnet)

            # Test forward pass
            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                cnn_out = cnn(x)
                resnet_out = resnet(x)

            self.assertEqual(cnn_out.shape[0], 2)
            self.assertEqual(resnet_out.shape[0], 2)

            logger.info("✓ Baseline models work correctly")
        except Exception as e:
            self.fail(f"Baseline model test failed: {e}")


class TestNeuralIntegration(unittest.TestCase):
    """Test neural architecture integration."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_integration_module_import(self) -> None:
        """Test integration module import."""
        try:
            from ml_pipeline.neural_integration import (
                NeuralArchitectureConfig,
                NeuralArchitectureFactory,
                UnifiedTrainingPipeline,
            )
            self.assertIsNotNone(NeuralArchitectureConfig)
            self.assertIsNotNone(NeuralArchitectureFactory)
            self.assertIsNotNone(UnifiedTrainingPipeline)
        except ImportError as e:
            self.fail(f"Failed to import integration modules: {e}")

    def test_vit_factory_integration(self) -> None:
        """Test ViT factory integration."""
        try:
            from ml_pipeline.neural_integration import (
                NeuralArchitectureConfig,
                NeuralArchitectureFactory,
            )

            config = NeuralArchitectureConfig(
                architecture_type="vit",
                model_size="base",
                device=self.device,
            )

            model = NeuralArchitectureFactory.create_model(config)
            self.assertIsNotNone(model)
            logger.info("✓ ViT factory integration works")
        except Exception as e:
            self.fail(f"ViT factory integration failed: {e}")

    def test_diffusion_factory_integration(self) -> None:
        """Test Diffusion factory integration."""
        try:
            from ml_pipeline.neural_integration import (
                NeuralArchitectureConfig,
                NeuralArchitectureFactory,
            )

            config = NeuralArchitectureConfig(
                architecture_type="diffusion",
                model_size="fast",
                device=self.device,
            )

            model = NeuralArchitectureFactory.create_model(config)
            self.assertIsNotNone(model)
            logger.info("✓ Diffusion factory integration works")
        except Exception as e:
            self.fail(f"Diffusion factory integration failed: {e}")

    def test_unified_training_pipeline(self) -> None:
        """Test unified training pipeline."""
        try:
            from ml_pipeline.neural_integration import (
                NeuralArchitectureConfig,
                NeuralArchitectureFactory,
                UnifiedTrainingPipeline,
            )

            config = NeuralArchitectureConfig(
                architecture_type="vit",
                model_size="base",
                device=self.device,
            )

            model = NeuralArchitectureFactory.create_model(config)
            pipeline = UnifiedTrainingPipeline(
                model=model,
                config=config,
                output_dir="./test_checkpoints",
            )

            self.assertIsNotNone(pipeline)
            info = pipeline.get_model_info()
            self.assertIn("total_parameters", info)
            logger.info("✓ Unified training pipeline works")
        except Exception as e:
            self.fail(f"Unified pipeline test failed: {e}")


class TestDiffusionService(unittest.TestCase):
    """Test Diffusion API service."""

    def test_service_import(self) -> None:
        """Test service module import."""
        try:
            from api.services.diffusion_service import (
                DiffusionService,
                DiffusionServiceConfig,
            )
            self.assertIsNotNone(DiffusionService)
            self.assertIsNotNone(DiffusionServiceConfig)
        except ImportError as e:
            self.fail(f"Failed to import service modules: {e}")

    def test_service_config(self) -> None:
        """Test service configuration."""
        try:
            from api.services.diffusion_service import DiffusionServiceConfig

            config = DiffusionServiceConfig(
                device="cpu",
                max_batch_size=32,
                default_num_steps=50,
            )

            self.assertEqual(config.max_batch_size, 32)
            self.assertEqual(config.default_num_steps, 50)
            logger.info("✓ Service config works")
        except Exception as e:
            self.fail(f"Service config test failed: {e}")

    def test_image_processor(self) -> None:
        """Test image processor."""
        try:
            from api.services.diffusion_service import ImageProcessor

            processor = ImageProcessor()

            # Create test image
            test_img = np.random.rand(64, 64, 3)

            # Test encoding
            encoded = processor.image_to_base64(test_img)
            self.assertIsInstance(encoded, str)

            # Test decoding
            decoded = processor.base64_to_image(encoded)
            self.assertEqual(decoded.shape[:2], (64, 64))

            logger.info("✓ Image processor works")
        except Exception as e:
            self.fail(f"Image processor test failed: {e}")


def run_tests(verbosity: int = 2) -> None:
    """
    Run all integration tests.

    Args:
        verbosity: Test output verbosity
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting integration tests...")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVisionTransformerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDiffusionIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSyntheticDataGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarking))
    suite.addTests(loader.loadTestsFromTestCase(TestNeuralIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDiffusionService))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info("="*80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests(verbosity=2)
    exit(0 if success else 1)
