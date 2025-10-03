#!/usr/bin/env python
"""
Unit tests for dynamic resolution system.
"""

import unittest
import numpy as np
import torch

from negative_space_analysis.resolution_system import (
    ScaleParameters,
    DynamicResolutionAnalyzer,
    ScaleSpaceNetwork
)


class TestDynamicResolution(unittest.TestCase):
    """Test suite for dynamic resolution system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.min_res = (32, 32)
        self.max_res = (256, 256)
        self.num_scales = 3
        
        self.analyzer = DynamicResolutionAnalyzer(
            min_resolution=self.min_res,
            max_resolution=self.max_res,
            num_scales=self.num_scales
        )
        
        # Create test image
        self.test_image = np.random.randn(128, 128).astype(np.float32)
        self.test_mask = np.ones_like(self.test_image, dtype=bool)
        self.test_mask[64:, 64:] = False
    
    def test_optimal_scale_analysis(self):
        """Test optimal scale parameter calculation."""
        params = self.analyzer.analyze_optimal_scale(self.test_image)
        
        # Check scale parameters
        self.assertIsInstance(params, ScaleParameters)
        self.assertTrue(0 < params.scale_factor <= 1)
        self.assertTrue(0 <= params.detail_level <= 1)
        self.assertTrue(0 <= params.quality_score <= 1)
        
        # Check resolution bounds
        h, w = params.base_resolution
        self.assertTrue(self.min_res[0] <= h <= self.max_res[0])
        self.assertTrue(self.min_res[1] <= w <= self.max_res[1])
    
    def test_scale_analysis_with_mask(self):
        """Test scale analysis with region mask."""
        params = self.analyzer.analyze_optimal_scale(
            self.test_image,
            self.test_mask
        )
        
        # Basic validation
        self.assertIsInstance(params, ScaleParameters)
        self.assertTrue(0 < params.scale_factor <= 1)
    
    def test_region_rescaling(self):
        """Test image region rescaling."""
        # Get scale parameters
        params = self.analyzer.analyze_optimal_scale(self.test_image)
        
        # Test rescaling
        rescaled = self.analyzer.rescale_region(
            self.test_image,
            params
        )
        
        # Check output
        self.assertEqual(
            rescaled.shape[:2],
            params.base_resolution
        )
        self.assertTrue(np.all(np.isfinite(rescaled)))
    
    def test_detail_preservation(self):
        """Test detail-preserving rescaling."""
        # Create image with strong edges
        edge_image = np.zeros((128, 128), dtype=np.float32)
        edge_image[32:96, 32:96] = 1.0
        
        params = self.analyzer.analyze_optimal_scale(edge_image)
        
        # Test with and without detail preservation
        standard = self.analyzer.rescale_region(
            edge_image,
            params,
            preserve_details=False
        )
        preserved = self.analyzer.rescale_region(
            edge_image,
            params,
            preserve_details=True
        )
        
        # Detail-preserved version should have sharper edges
        standard_grad = np.mean(np.abs(np.gradient(standard)))
        preserved_grad = np.mean(np.abs(np.gradient(preserved)))
        self.assertGreater(preserved_grad, standard_grad)
    
    def test_extreme_cases(self):
        """Test handling of extreme cases."""
        # Very small image
        tiny_image = np.random.randn(16, 16).astype(np.float32)
        params = self.analyzer.analyze_optimal_scale(tiny_image)
        self.assertGreaterEqual(
            params.base_resolution[0],
            self.min_res[0]
        )
        
        # Very large image
        large_image = np.random.randn(512, 512).astype(np.float32)
        params = self.analyzer.analyze_optimal_scale(large_image)
        self.assertLessEqual(
            params.base_resolution[0],
            self.max_res[0]
        )
    
    def test_quality_score_consistency(self):
        """Test consistency of quality scores."""
        # Multiple runs should give similar results
        params1 = self.analyzer.analyze_optimal_scale(self.test_image)
        params2 = self.analyzer.analyze_optimal_scale(self.test_image)
        
        self.assertAlmostEqual(
            params1.quality_score,
            params2.quality_score,
            places=5
        )


class TestScaleSpaceNetwork(unittest.TestCase):
    """Test suite for ScaleSpaceNetwork."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_channels = 32
        self.num_scales = 3
        self.batch_size = 2
        
        self.network = ScaleSpaceNetwork(
            base_channels=self.base_channels,
            num_scales=self.num_scales
        )
        
        # Create test pyramid
        self.pyramid = [
            torch.randn(
                self.batch_size,
                1,  # channels
                32 // (2 ** i),  # height
                32 // (2 ** i)   # width
            )
            for i in range(self.num_scales)
        ]
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        features = self.network(self.pyramid)
        
        # Check output shape
        self.assertEqual(
            features.shape,
            (self.num_scales, self.batch_size, self.base_channels, 32, 32)
        )
        
        # Check feature properties
        self.assertTrue(torch.all(torch.isfinite(features)))
        self.assertTrue(torch.any(features != 0))  # Non-zero output
    
    def test_attention_mechanism(self):
        """Test scale attention behavior."""
        features = self.network(self.pyramid)
        
        # Check that attention weights are applied
        # (features should vary across scales)
        scale_means = torch.mean(features, dim=(1, 2, 3, 4))
        self.assertTrue(torch.any(torch.diff(scale_means) != 0))
    
    def test_gradient_flow(self):
        """Test gradient flow through network."""
        self.network.train()
        features = self.network(self.pyramid)
        loss = features.mean()
        loss.backward()
        
        # Check gradients
        for param in self.network.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.all(torch.isfinite(param.grad)))


if __name__ == '__main__':
    unittest.main()
