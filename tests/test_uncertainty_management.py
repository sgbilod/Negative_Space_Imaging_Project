#!/usr/bin/env python
"""
Unit tests for uncertainty management system.
"""

import unittest
import numpy as np
import torch

from negative_space_analysis.uncertainty_management import (
    UncertaintyMetrics,
    EnsembleUncertaintyEstimator,
    UncertaintyNet
)


class TestUncertaintyManagement(unittest.TestCase):
    """Test suite for uncertainty management system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_dim = 64
        self.batch_size = 8
        self.num_models = 3
        
        self.estimator = EnsembleUncertaintyEstimator(
            num_models=self.num_models,
            hidden_dim=self.hidden_dim
        )
        
        # Create test data
        self.features = torch.randn(
            self.batch_size,
            self.hidden_dim,
            device=self.estimator.device
        )
    
    def test_uncertainty_estimation(self):
        """Test basic uncertainty estimation."""
        metrics, predictions = self.estimator.estimate_uncertainty(
            self.features,
            return_predictions=True
        )
        
        # Check metrics object
        self.assertIsInstance(metrics, UncertaintyMetrics)
        self.assertTrue(0 <= metrics.confidence <= 1)
        self.assertTrue(0 <= metrics.epistemic <= 1)
        self.assertTrue(0 <= metrics.aleatoric <= 1)
        
        # Check predictions
        self.assertEqual(
            predictions.shape,
            (self.batch_size, 1)
        )
        self.assertTrue(torch.all(predictions >= 0))
        self.assertTrue(torch.all(predictions <= 1))
    
    def test_ensemble_consistency(self):
        """Test consistency of ensemble predictions."""
        # Multiple forward passes should give similar results
        metrics1, pred1 = self.estimator.estimate_uncertainty(
            self.features,
            return_predictions=True
        )
        metrics2, pred2 = self.estimator.estimate_uncertainty(
            self.features,
            return_predictions=True
        )
        
        # Check prediction consistency
        torch.testing.assert_close(pred1, pred2)
        
        # Check metric consistency
        self.assertAlmostEqual(
            metrics1.confidence,
            metrics2.confidence,
            places=5
        )
        self.assertAlmostEqual(
            metrics1.epistemic,
            metrics2.epistemic,
            places=5
        )
    
    def test_feature_uncertainty(self):
        """Test feature-based uncertainty estimation."""
        # Create test data for random forest
        X = np.random.randn(100, self.hidden_dim)
        y = np.random.randint(0, 2, 100)
        
        # Update forest and get uncertainties
        self.estimator.update_forest(X, y)
        uncertainties = self.estimator.feature_uncertainty(X)
        
        # Check shape and values
        self.assertEqual(uncertainties.shape, (100, 2))
        self.assertTrue(np.all(uncertainties >= 0))
        self.assertTrue(np.all(uncertainties <= 1))
    
    def test_mutual_information(self):
        """Test mutual information computation."""
        # Create test predictions
        test_preds = torch.randn(
            self.num_models,
            self.batch_size,
            2,  # binary classification
            device=self.estimator.device
        )
        test_preds = torch.softmax(test_preds, dim=-1)
        
        # Compute mutual information
        mi = self.estimator._compute_mutual_information(test_preds)
        
        # Check bounds and type
        self.assertIsInstance(mi, float)
        self.assertTrue(0 <= mi <= np.log(2))  # Binary case
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        confidence = self.estimator._compute_confidence(
            epistemic=0.3,
            aleatoric=0.2,
            entropy=0.5,
            variance=0.4,
            disagreement=0.3,
            mutual_info=0.2
        )
        
        # Check bounds and type
        self.assertIsInstance(confidence, float)
        self.assertTrue(0 <= confidence <= 1)
    
    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Empty features
        empty_features = torch.empty(
            0, self.hidden_dim,
            device=self.estimator.device
        )
        with self.assertRaises(ValueError):
            self.estimator.estimate_uncertainty(empty_features)
        
        # Invalid feature dimensions
        invalid_features = torch.randn(
            self.batch_size,
            self.hidden_dim + 1,
            device=self.estimator.device
        )
        with self.assertRaises(RuntimeError):
            self.estimator.estimate_uncertainty(invalid_features)


class TestUncertaintyNet(unittest.TestCase):
    """Test suite for UncertaintyNet."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_dim = 64
        self.batch_size = 8
        self.net = UncertaintyNet(self.hidden_dim)
        
        self.features = torch.randn(
            self.batch_size,
            self.hidden_dim
        )
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        predictions, features = self.net(self.features)
        
        # Check shapes
        self.assertEqual(
            predictions.shape,
            (self.batch_size, 1)
        )
        self.assertEqual(
            features.shape,
            (self.batch_size, self.hidden_dim)
        )
        
        # Check value ranges
        self.assertTrue(torch.all(predictions >= 0))
        self.assertTrue(torch.all(predictions <= 1))
        self.assertTrue(torch.all(torch.isfinite(features)))
    
    def test_dropout_effect(self):
        """Test that dropout creates variation in training."""
        self.net.train()
        
        # Multiple forward passes should give different results
        pred1, _ = self.net(self.features)
        pred2, _ = self.net(self.features)
        
        # Check predictions differ
        self.assertTrue(torch.any(pred1 != pred2))
    
    def test_eval_mode(self):
        """Test consistent predictions in eval mode."""
        self.net.eval()
        
        # Multiple forward passes should give same results
        with torch.no_grad():
            pred1, _ = self.net(self.features)
            pred2, _ = self.net(self.features)
        
        # Check predictions match
        torch.testing.assert_close(pred1, pred2)


if __name__ == '__main__':
    unittest.main()
