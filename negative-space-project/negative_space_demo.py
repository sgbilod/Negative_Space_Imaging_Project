#!/usr/bin/env python
"""
Negative Space Analysis Demo
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This script demonstrates the complete negative space analysis pipeline,
integrating the core algorithm, advanced analytics, and visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

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


def run_pipeline_demo(
    image_path: str,
    output_dir: str,
    use_gpu: bool = True,
    model_path: Optional[str] = None
) -> None:
    """
    Run the complete negative space analysis pipeline.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for output files
        use_gpu: Whether to use GPU acceleration
        model_path: Optional path to pre-trained model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load and preprocess image
    image = plt.imread(image_path)
    if image.ndim == 3:
        # Convert to grayscale if needed
        image = np.mean(image, axis=2)
    
    # 2. Initialize components
    analyzer = NegativeSpaceAnalyzer(
        use_gpu=use_gpu,
        model_path=model_path
    )
    
    visualizer = NegativeSpaceVisualizer(
        config=VisualizationConfig(
            show_boundaries=True,
            show_features=True,
            interactive=True
        )
    )
    
    analytics = NegativeSpaceAnalytics(
        use_gpu=use_gpu,
        model_path=model_path
    )
    
    # 3. Run negative space analysis
    print("Analyzing negative spaces...")
    features = analyzer.analyze_image(image)
    
    # 4. Perform advanced analytics
    print("Performing pattern analysis...")
    analytics_results = analytics.analyze_patterns(
        list(features.values())[0]  # First region's features
    )
    
    # 5. Generate visualizations
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
    
    # 6. Print analytics results
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
    
    print("\nAnalysis complete! Results saved to:", output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Negative Space Analysis Demo")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("output_dir", help="Directory for output files")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--model", help="Path to pre-trained model")
    
    args = parser.parse_args()
    run_pipeline_demo(args.image_path, args.output_dir, args.gpu, args.model)
