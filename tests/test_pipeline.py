#!/usr/bin/env python
"""
Negative Space Analysis Tests
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements comprehensive tests for the negative space analysis pipeline
and all its subsystems.
"""

import unittest
import numpy as np
import torch
import cv2
from pathlib import Path
import logging
import tempfile
import json
from typing import Dict, List, Optional

from negative_space_analysis.pipeline import (
    NegativeSpaceAnalyzer,
    AnalysisConfig,
    AnalysisResult
)
from negative_space_analysis.multimodal_system import (
    MultiModalAnalyzer,
    ModalityType,
    MultiModalFeatures
)
from negative_space_analysis.semantic_system import (
    SemanticContextAnalyzer,
    SemanticContext,
    RelationType
)
from negative_space_analysis.temporal_system import (
    TemporalAnalyzer,
    TemporalChange,
    ChangeType
)
from negative_space_analysis.interactive_system import (
    InteractiveRefinement,
    UserFeedback,
    FeedbackType
)


class TestMultiModalSystem(unittest.TestCase):
    """Test cases for multi-modal analysis system."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device('cpu')
        self.analyzer = MultiModalAnalyzer(
            feature_dim=128,
            device=self.device
        )
        
        # Create sample data
        self.image = np.random.randint(
            0, 255,
            (100, 100, 3),
            dtype=np.uint8
        )
        self.depth = np.random.rand(100, 100).astype(np.float32)
        self.audio = np.random.rand(1000).astype(np.float32)
        self.text = "Sample text description"
    
    def test_modality_encoding(self):
        """Test encoding of different modalities."""
        inputs = {
            ModalityType.VISUAL: self.image,
            ModalityType.DEPTH: self.depth,
            ModalityType.AUDIO: self.audio,
            ModalityType.TEXT: self.text
        }
        
        result = self.analyzer.analyze(inputs)
        
        self.assertIsNotNone(result)
        self.assertTrue(len(result.region_ids) > 0)
        self.assertEqual(
            len(result.region_masks),
            len(result.region_ids)
        )
    
    def test_feature_extraction(self):
        """Test feature extraction from regions."""
        inputs = {ModalityType.VISUAL: self.image}
        result = self.analyzer.analyze(inputs)
        
        for rid, features in result.features.items():
            self.assertIsInstance(features, MultiModalFeatures)
            self.assertEqual(features.visual.shape[-1], 128)
    
    def test_empty_input(self):
        """Test handling of empty inputs."""
        with self.assertRaises(ValueError):
            self.analyzer.analyze({})


class TestSemanticSystem(unittest.TestCase):
    """Test cases for semantic context system."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device('cpu')
        self.analyzer = SemanticContextAnalyzer(
            feature_dim=128,
            device=self.device
        )
        
        # Create sample regions and features
        self.regions = {
            "r1": np.ones((50, 50), dtype=bool),
            "r2": np.ones((50, 50), dtype=bool),
        }
        self.features = {
            "r1": torch.randn(128),
            "r2": torch.randn(128)
        }
    
    def test_context_analysis(self):
        """Test semantic context analysis."""
        contexts = self.analyzer.analyze_context(
            self.regions,
            self.features
        )
        
        self.assertEqual(len(contexts), len(self.regions))
        for rid, context in contexts.items():
            self.assertIsInstance(context, SemanticContext)
            self.assertEqual(context.region_id, rid)
    
    def test_relation_types(self):
        """Test semantic relationship classification."""
        contexts = self.analyzer.analyze_context(
            self.regions,
            self.features
        )
        
        for context in contexts.values():
            for relation in context.relations:
                self.assertIsInstance(
                    relation.relation_type,
                    RelationType
                )


class TestTemporalSystem(unittest.TestCase):
    """Test cases for temporal analysis system."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device('cpu')
        self.analyzer = TemporalAnalyzer(
            feature_dim=128,
            device=self.device
        )
        
        # Create sequence of regions and features
        self.regions_seq = [
            {
                "r1": np.ones((50, 50), dtype=bool),
                "r2": np.ones((50, 50), dtype=bool)
            },
            {
                "r1": np.ones((50, 50), dtype=bool),
                "r3": np.ones((50, 50), dtype=bool)
            }
        ]
        self.features_seq = [
            {
                "r1": torch.randn(128),
                "r2": torch.randn(128)
            },
            {
                "r1": torch.randn(128),
                "r3": torch.randn(128)
            }
        ]
    
    def test_temporal_tracking(self):
        """Test temporal region tracking."""
        changes1 = self.analyzer.update(
            self.regions_seq[0],
            self.features_seq[0]
        )
        changes2 = self.analyzer.update(
            self.regions_seq[1],
            self.features_seq[1]
        )
        
        self.assertTrue(any(
            c.change_type == ChangeType.DISAPPEAR
            for c in changes2
        ))
        self.assertTrue(any(
            c.change_type == ChangeType.APPEAR
            for c in changes2
        ))
    
    def test_trajectory_retrieval(self):
        """Test trajectory retrieval."""
        self.analyzer.update(
            self.regions_seq[0],
            self.features_seq[0]
        )
        
        trajectory = self.analyzer.get_trajectory("r1")
        self.assertIsNotNone(trajectory)
        self.assertEqual(len(trajectory.points), 1)


class TestInteractiveSystem(unittest.TestCase):
    """Test cases for interactive refinement system."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device('cpu')
        self.analyzer = InteractiveRefinement(
            feature_dim=128,
            device=self.device
        )
        
        # Create sample regions and features
        self.regions = {
            "r1": np.ones((50, 50), dtype=bool),
            "r2": np.ones((50, 50), dtype=bool)
        }
        self.features = {
            "r1": torch.randn(128),
            "r2": torch.randn(128)
        }
    
    def test_feedback_application(self):
        """Test applying user feedback."""
        feedback = UserFeedback(
            feedback_type=FeedbackType.REGION_MERGE,
            region_ids=["r1", "r2"],
            parameters={},
            timestamp=0.0,
            confidence=1.0
        )
        
        updated_regions = self.analyzer.add_feedback(
            feedback,
            self.regions,
            self.features
        )
        
        self.assertIn("r1", updated_regions)
        self.assertNotIn("r2", updated_regions)
    
    def test_suggestion_generation(self):
        """Test refinement suggestion generation."""
        suggestions = self.analyzer.get_suggestions(
            self.regions,
            self.features
        )
        
        self.assertIsInstance(suggestions, list)
        for suggestion in suggestions:
            self.assertGreaterEqual(suggestion.confidence, 0.0)
            self.assertLessEqual(suggestion.confidence, 1.0)


class TestMainPipeline(unittest.TestCase):
    """Test cases for main analysis pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = AnalysisConfig(
            feature_dim=128,
            batch_size=32,
            device=torch.device('cpu'),
            enable_temporal=True,
            enable_refinement=True
        )
        self.analyzer = NegativeSpaceAnalyzer(self.config)
        
        # Create sample frame data
        self.image = np.random.randint(
            0, 255,
            (100, 100, 3),
            dtype=np.uint8
        )
        self.depth = np.random.rand(100, 100).astype(np.float32)
    
    def test_full_pipeline(self):
        """Test complete analysis pipeline."""
        result = self.analyzer.analyze_frame(
            image=self.image,
            depth=self.depth
        )
        
        self.assertIsInstance(result, AnalysisResult)
        self.assertTrue(len(result.region_ids) > 0)
        self.assertEqual(
            len(result.semantic_contexts),
            len(result.region_ids)
        )
    
    def test_feedback_integration(self):
        """Test feedback integration in pipeline."""
        # Initial analysis
        result = self.analyzer.analyze_frame(
            image=self.image
        )
        
        if result.refinement_suggestions:
            suggestion = result.refinement_suggestions[0]
            feedback = UserFeedback(
                feedback_type=suggestion.suggestion_type,
                region_ids=suggestion.region_ids,
                parameters=suggestion.parameters,
                timestamp=0.0,
                confidence=1.0
            )
            
            updated_result = self.analyzer.apply_feedback(feedback)
            self.assertIsNotNone(updated_result)
    
    def test_error_handling(self):
        """Test error handling in pipeline."""
        with self.assertRaises(ValueError):
            self.analyzer.analyze_frame(
                image=np.zeros((0, 0, 3))  # Invalid image
            )


def run_tests():
    """Run all test cases."""
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main()


if __name__ == "__main__":
    run_tests()
