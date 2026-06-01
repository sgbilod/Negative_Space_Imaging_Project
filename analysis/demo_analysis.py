#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis System Demo Script
Author: Stephen Bilodeau
Date: August 13, 2025

This script demonstrates the use of the analysis system.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import analysis modules
from analysis.models.neural_network import NeuralNetworkAnalyzer
from analysis.visualizations.visualization import DataVisualizer
from analysis.reporting.report_generator import ReportGenerator


def generate_test_data(output_dir, num_samples=1000, num_features=20):
    """Generate test data for demonstration."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate random data with patterns
    X = np.random.randn(num_samples, num_features)

    # Add some correlations between features
    for i in range(5):
        X[:, i+5] = 0.7 * X[:, i] + 0.3 * np.random.randn(num_samples)

    # Add a non-linear pattern
    X[:, 15] = np.sin(X[:, 0]) + 0.1 * np.random.randn(num_samples)

    # Save data
    np.save(os.path.join(output_dir, 'processed_data.npy'), X)

    # Create metadata
    metadata = {
        'features': [f'feature_{i}' for i in range(num_features)],
        'samples': num_samples,
        'description': 'Synthetic test data for analysis demo',
        'has_patterns': True,
        'data_type': 'numpy'
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    return X, metadata


def run_full_analysis(data_dir, output_dir):
    """Run a full analysis pipeline on the data."""
    # Set up directories
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Load or generate data
    data_path = os.path.join(data_dir, 'processed_data.npy')
    metadata_path = os.path.join(data_dir, 'metadata.json')

    if os.path.exists(data_path) and os.path.exists(metadata_path):
        # Load existing data
        data = np.load(data_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logging.info(f"Loaded existing data with shape {data.shape}")
    else:
        # Generate test data
        logging.info("Generating test data...")
        data, metadata = generate_test_data(data_dir)
        logging.info(f"Generated test data with shape {data.shape}")

    # 1. Run neural network analysis
    logging.info("Running neural network analysis...")
    analyzer = NeuralNetworkAnalyzer()
    nn_results = analyzer.analyze(
        input_data=data,
        output_dir=model_dir,
        model_type='dense',
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )

    # Save analysis results
    with open(os.path.join(output_dir, 'nn_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for key, value in nn_results.items():
            if isinstance(value, np.ndarray):
                results_json[key] = value.tolist()
            else:
                results_json[key] = value
        json.dump(results_json, f, indent=4)

    # 2. Generate visualizations
    logging.info("Generating visualizations...")
    visualizer = DataVisualizer()

    # Generate different visualization types
    viz_types = ['pca', 'tsne', 'correlation']
    for viz_type in viz_types:
        viz_path = visualizer.generate_visualization(
            input_dir=data_dir,
            output_dir=viz_dir,
            viz_type=viz_type
        )
        logging.info(f"Generated {viz_type} visualization: {viz_path}")

    # 3. Generate report
    logging.info("Generating analysis report...")
    report_generator = ReportGenerator()
    report_path = report_generator.generate_report(
        input_dir=output_dir,
        output_file=os.path.join(output_dir, 'analysis_report.pdf'),
        template='standard'
    )
    logging.info(f"Generated report: {report_path}")

    return {
        'nn_results': nn_results,
        'visualizations': viz_types,
        'report': report_path
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analysis System Demo')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/demo',
        help='Directory for input data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_results/demo',
        help='Directory for analysis results'
    )
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Force generation of new test data'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of samples for test data generation'
    )
    parser.add_argument(
        '--features',
        type=int,
        default=20,
        help='Number of features for test data generation'
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate data if requested or if it doesn't exist
    if args.generate_data or not os.path.exists(os.path.join(args.data_dir, 'processed_data.npy')):
        logging.info("Generating test data...")
        generate_test_data(
            args.data_dir,
            num_samples=args.samples,
            num_features=args.features
        )

    # Run analysis
    logging.info("Starting analysis pipeline...")
    results = run_full_analysis(args.data_dir, args.output_dir)

    logging.info("Analysis complete!")
    logging.info(f"Results saved to: {os.path.abspath(args.output_dir)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
