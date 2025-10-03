"""
Test suite for negative space analysis components.
"""

import os
import numpy as np
import pytest
from typing import Dict, List

from negative_space_analysis.negative_space_algorithm import (
    NegativeSpaceAnalyzer,
    NegativeSpaceFeatures
)
from negative_space_analysis.visualization import (
    NegativeSpaceVisualizer,
    VisualizationConfig
)
from negative_space_analysis.advanced_analytics import (
    NegativeSpaceAnalytics,
    AnalyticsResult
)


@pytest.fixture
def test_image():
    """Create a simple test image with known negative spaces."""
    image = np.ones((100, 100))
    # Add a circle
    y, x = np.ogrid[-50:50, -50:50]
    mask = x*x + y*y <= 30*30
    image[mask] = 0
    return image


@pytest.fixture
def analyzer():
    """Create a negative space analyzer instance."""
    return NegativeSpaceAnalyzer(use_gpu=False)


@pytest.fixture
def visualizer():
    """Create a visualizer instance."""
    return NegativeSpaceVisualizer(
        config=VisualizationConfig(
            show_boundaries=True,
            show_features=True,
            interactive=False
        )
    )


@pytest.fixture
def analytics():
    """Create an analytics instance."""
    return NegativeSpaceAnalytics(use_gpu=False)


def test_negative_space_detection(test_image, analyzer):
    """Test basic negative space detection."""
    features = analyzer.analyze_image(test_image)
    
    assert isinstance(features, dict)
    assert len(features) > 0
    
    # Check first region's features
    first_region = list(features.values())[0][0]
    assert isinstance(first_region, NegativeSpaceFeatures)
    assert first_region.area > 0
    assert first_region.confidence > 0
    assert 0 <= first_region.pattern_score <= 1


def test_visualization(test_image, analyzer, visualizer, tmp_path):
    """Test visualization generation."""
    features = analyzer.analyze_image(test_image)
    output_path = os.path.join(tmp_path, "test_vis.png")
    
    # Test basic visualization
    visualizer.visualize_analysis(
        test_image,
        features,
        output_path=output_path,
        show=False
    )
    assert os.path.exists(output_path)
    
    # Test interactive visualization
    fig = visualizer.create_interactive_view(test_image, features)
    assert fig is not None


def test_analytics(test_image, analyzer, analytics):
    """Test analytics functionality."""
    features = analyzer.analyze_image(test_image)
    results = analytics.analyze_patterns(list(features.values())[0])
    
    assert isinstance(results, list)
    assert len(results) > 0
    
    first_result = results[0]
    assert isinstance(first_result, AnalyticsResult)
    assert isinstance(first_result.pattern_type, str)
    assert 0 <= first_result.confidence <= 1
    assert 0 <= first_result.anomaly_score <= 1
    assert isinstance(first_result.feature_importance, dict)
    assert isinstance(first_result.related_patterns, list)


def test_end_to_end_pipeline(test_image, analyzer, visualizer, analytics, tmp_path):
    """Test the complete analysis pipeline."""
    # 1. Analyze image
    features = analyzer.analyze_image(test_image)
    assert isinstance(features, dict)
    
    # 2. Generate visualizations
    output_path = os.path.join(tmp_path, "pipeline_test.png")
    visualizer.visualize_analysis(
        test_image,
        features,
        output_path=output_path,
        show=False
    )
    assert os.path.exists(output_path)
    
    # 3. Run analytics
    results = analytics.analyze_patterns(list(features.values())[0])
    assert isinstance(results, list)
    assert len(results) > 0
    
    # 4. Verify results
    assert all(0 <= r.confidence <= 1 for r in results)
    assert all(0 <= r.anomaly_score <= 1 for r in results)
    assert all(isinstance(r.pattern_type, str) for r in results)


if __name__ == "__main__":
    pytest.main([__file__])
