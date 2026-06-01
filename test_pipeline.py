#!/usr/bin/env python
"""
Test script for the negative space analysis pipeline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from negative_space_analysis.negative_space_algorithm import NegativeSpaceAnalyzer
from negative_space_analysis.visualization import (
    NegativeSpaceVisualizer,
    VisualizationConfig
)
from negative_space_analysis.advanced_analytics import NegativeSpaceAnalytics


def create_test_image(size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Create a test image with known negative spaces."""
    image = np.ones(size)
    
    # Create some objects to generate negative spaces
    for i in range(5):
        # Random circles
        center = np.random.randint(50, size[0]-50, 2)
        radius = np.random.randint(20, 50)
        y, x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
        mask = x*x + y*y <= radius*radius
        image[mask] = 0
    
    # Add some lines
    for i in range(3):
        start = np.random.randint(0, size[0], 2)
        end = np.random.randint(0, size[0], 2)
        rr, cc = line(start[0], start[1], end[0], end[1])
        image[rr, cc] = 0
    
    return image


def line(x0: int, y0: int, x1: int, y1: int) -> Tuple[np.ndarray, np.ndarray]:
    """Draw a line using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            yield x, y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            yield x, y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    yield x, y


def run_pipeline_test():
    """Run the complete pipeline on a test image."""
    # Create output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create test image
    print("Creating test image...")
    image = create_test_image()
    plt.imsave(os.path.join(output_dir, "test_image.png"), image, cmap='gray')
    
    # 2. Initialize components
    print("Initializing analysis pipeline...")
    analyzer = NegativeSpaceAnalyzer(use_gpu=True)
    visualizer = NegativeSpaceVisualizer(
        config=VisualizationConfig(
            show_boundaries=True,
            show_features=True,
            interactive=True
        )
    )
    analytics = NegativeSpaceAnalytics(use_gpu=True)
    
    # 3. Run analysis
    print("Analyzing negative spaces...")
    features = analyzer.analyze_image(image)
    
    # 4. Generate visualizations
    print("Generating visualizations...")
    # Basic visualization
    visualizer.visualize_analysis(
        image,
        features,
        output_path=os.path.join(output_dir, "analysis.png")
    )
    
    # Interactive visualization
    fig = visualizer.create_interactive_view(image, features)
    fig.write_html(os.path.join(output_dir, "interactive.html"))
    
    # 5. Run analytics
    print("Running advanced analytics...")
    analytics_results = analytics.analyze_patterns(
        list(features.values())[0]  # First region's features
    )
    
    # 6. Print results
    print("\nAnalysis Results:")
    print("-" * 50)
    for i, result in enumerate(analytics_results, 1):
        print(f"\nPattern {i}:")
        print(f"Type: {result.pattern_type}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Anomaly Score: {result.anomaly_score:.3f}")
        print(f"Temporal Stability: {result.temporal_stability:.2f}")
        print("\nFeature Importance:")
        for feature, importance in result.feature_importance.items():
            print(f"  {feature}: {importance:.3f}")
        print("\nRelated Patterns:", ", ".join(result.related_patterns))
    
    print("\nTest complete! Results saved to:", output_dir)


if __name__ == "__main__":
    run_pipeline_test()
