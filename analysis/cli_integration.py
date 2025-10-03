#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI Integration for Data Analysis System
Author: Stephen Bilodeau
Date: August 13, 2025

This module integrates the Data Analysis System with the main CLI.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, Optional

# Import the data analysis system
try:
    from data_analysis_system import DataAnalysisSystem
    DATA_ANALYSIS_AVAILABLE = True
except ImportError:
    DATA_ANALYSIS_AVAILABLE = False


def setup_data_analysis_parser(subparsers):
    """Set up the argument parser for data analysis commands.

    Args:
        subparsers: Subparser object from the main CLI

    Returns:
        The data analysis subparser
    """
    # Create the parser for the "analyze" command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze imaging data using the Data Analysis System'
    )

    # Add arguments
    analyze_parser.add_argument(
        '--data',
        required=True,
        help='Path to data file (CSV, JSON, or numpy array)'
    )
    analyze_parser.add_argument(
        '--config',
        default='data_analysis_config.json',
        help='Path to configuration file'
    )
    analyze_parser.add_argument(
        '--output',
        help='Output prefix for results'
    )
    analyze_parser.add_argument(
        '--type',
        dest='analysis_types',
        nargs='+',
        choices=[
            'statistical', 'clustering', 'dimensionality',
            'anomaly', 'pattern', 'trend', 'correlation', 'all'
        ],
        help='Analysis types to perform'
    )
    analyze_parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Disable visualization generation'
    )
    analyze_parser.add_argument(
        '--generate-test-data',
        action='store_true',
        help='Generate test data before analysis'
    )
    analyze_parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of samples for test data generation'
    )

    return analyze_parser


def run_data_analysis(args) -> int:
    """Run the data analysis system with the provided arguments.

    Args:
        args: Command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not DATA_ANALYSIS_AVAILABLE:
        print("Error: Data Analysis System not available.")
        print("Make sure data_analysis_system.py is in your Python path.")
        return 1

    # Generate test data if requested
    if args.generate_test_data:
        try:
            from test_data_generator import generate_test_dataset
            data_path, _ = generate_test_dataset(
                output_path='test_data',
                n_samples=args.samples
            )
            args.data = data_path
            print(f"Generated test data at: {data_path}")
        except ImportError:
            print("Error: Test data generator not available.")
            print("Make sure test_data_generator.py is in your Python path.")
            return 1

    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        return 1

    # Initialize the data analysis system
    try:
        analyzer = DataAnalysisSystem(args.config)
    except Exception as e:
        print(f"Error initializing Data Analysis System: {e}")
        return 1

    # Prepare analysis types
    if args.analysis_types:
        if 'all' in args.analysis_types:
            analysis_types = list(analyzer.analysis_modules.keys())
        else:
            analysis_types = args.analysis_types
    else:
        analysis_types = None  # Use defaults from config

    # Run analysis
    try:
        results = analyzer.analyze_data(
            data_path=args.data,
            analysis_types=analysis_types,
            output_prefix=args.output,
            visualization=not args.no_visualizations
        )

        print(f"Analysis completed. Results saved in {analyzer.results_dir}")

        # Print summary
        if 'statistical' in results:
            print("\nSummary Statistics:")
            if 'summary' in results['statistical']:
                for feature, stats in results['statistical']['summary'].items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        print(f"  {feature}: mean={stats['mean']:.4f}, "
                              f"std={stats.get('std', 'N/A')}")

        if 'anomaly' in results:
            print("\nAnomaly Detection:")
            for method, anomaly_results in results['anomaly'].items():
                if 'anomaly_count' in anomaly_results:
                    print(f"  {method}: detected {anomaly_results['anomaly_count']} anomalies "
                          f"({anomaly_results.get('anomaly_percentage', 0):.2f}%)")

        # Check for visualizations directory
        vis_dir = os.path.join(analyzer.results_dir,
                              f"{args.output or 'analysis'}_visualizations")
        vis_index = os.path.join(vis_dir, "visualizations.html")
        if os.path.exists(vis_index):
            print(f"\nVisualizations available at: {vis_index}")

        return 0
    except Exception as e:
        print(f"Error during data analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


def integrate_with_main_cli(main_subparsers):
    """Integrate data analysis with the main CLI.

    Args:
        main_subparsers: Subparsers object from the main CLI
    """
    # Add data analysis commands
    analyze_parser = setup_data_analysis_parser(main_subparsers)
    analyze_parser.set_defaults(func=run_data_analysis)
