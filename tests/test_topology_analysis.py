#!/usr/bin/env python
"""
Unit tests for topological analysis module.
"""

import unittest
import numpy as np
import torch

from negative_space_analysis.topology_analysis import (
    TopologicalAnalyzer,
    TopologicalFeatures,
    TopologicalEncoder
)


class TestTopologicalAnalyzer(unittest.TestCase):
    """Test suite for TopologicalAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TopologicalAnalyzer(
            max_dimension=2,
            num_landscape_layers=3,
            resolution=50
        )
        
        # Create simple test data
        self.points = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],  # Square
            [0.5, 0.5]  # Center point
        ])
        
        self.features = np.random.randn(len(self.points), 10)
        
        # Create test mask
        self.mask = np.zeros((2, 2), dtype=bool)
        self.mask[0, 0] = True
        self.mask[1, 1] = True
    
    def test_analyze_topology(self):
        """Test basic topology analysis."""
        features = self.analyzer.analyze_topology(self.points)
        
        # Check persistence diagrams
        self.assertIn(0, features.persistence_diagrams)
        self.assertIn(1, features.persistence_diagrams)
        
        # Check Betti curves
        self.assertIn(0, features.betti_curves)
        curve = features.betti_curves[0]
        self.assertEqual(len(curve), self.analyzer.resolution)
        
        # Check persistence images
        self.assertIn(0, features.persistence_images)
        image = features.persistence_images[0]
        self.assertEqual(
            image.shape,
            (self.analyzer.resolution, self.analyzer.resolution)
        )
        
        # Check persistence entropy
        self.assertIn(0, features.persistence_entropy)
        self.assertIsInstance(features.persistence_entropy[0], float)
        
        # Check homology features
        self.assertIn("total_persistence_0", features.homology_features)
        self.assertIsInstance(
            features.homology_features["total_persistence_0"],
            float
        )
        
        # Check persistence landscape
        self.assertIn(0, features.persistence_landscape)
        landscape = features.persistence_landscape[0]
        self.assertEqual(
            landscape.shape,
            (self.analyzer.num_landscape_layers, self.analyzer.resolution)
        )
    
    def test_analyze_topology_with_mask(self):
        """Test topology analysis with mask."""
        features = self.analyzer.analyze_topology(self.points, self.mask)
        
        # Basic validation
        self.assertIsInstance(features, TopologicalFeatures)
        self.assertTrue(all(
            isinstance(d, np.ndarray)
            for d in features.persistence_diagrams.values()
        ))
    
    def test_analyze_negative_spaces(self):
        """Test analysis of multiple negative space regions."""
        # Create test data
        masks = {
            "region1": self.mask,
            "region2": ~self.mask
        }
        features = {
            "region1": self.features[:2],
            "region2": self.features[2:]
        }
        
        results = self.analyzer.analyze_negative_spaces(masks, features)
        
        # Validate results
        self.assertEqual(set(results.keys()), {"region1", "region2"})
        for region_features in results.values():
            self.assertIsInstance(region_features, TopologicalFeatures)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Empty point cloud
        with self.assertRaises(ValueError):
            self.analyzer.analyze_topology(np.array([]))
        
        # Invalid mask shape
        invalid_mask = np.zeros((3, 3), dtype=bool)
        with self.assertRaises(ValueError):
            self.analyzer.analyze_topology(self.points, invalid_mask)
        
        # Mismatched features
        invalid_features = np.random.randn(len(self.points) + 1, 10)
        with self.assertRaises(ValueError):
            self.analyzer.analyze_topology(
                self.points,
                features=invalid_features
            )


class TestTopologicalEncoder(unittest.TestCase):
    """Test suite for TopologicalEncoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 10
        self.hidden_dim = 64
        self.batch_size = 4
        
        self.encoder = TopologicalEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Create test inputs
        self.diagrams = torch.randn(
            self.batch_size,
            5,  # num points
            2   # birth-death coords
        )
        self.betti_curves = torch.randn(self.batch_size, 50)
        self.persistence_images = torch.randn(self.batch_size, 50, 50)
    
    def test_forward_pass(self):
        """Test forward pass through encoder."""
        output = self.encoder(
            self.diagrams,
            self.betti_curves,
            self.persistence_images
        )
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        
        # Check output values
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_gradient_flow(self):
        """Test gradient flow through the encoder."""
        self.encoder.train()
        output = self.encoder(
            self.diagrams,
            self.betti_curves,
            self.persistence_images
        )
        
        # Compute loss and backprop
        loss = output.mean()
        loss.backward()
        
        # Check gradients
        for param in self.encoder.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.all(torch.isfinite(param.grad)))


if __name__ == '__main__':
    unittest.main()
