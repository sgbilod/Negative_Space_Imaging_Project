#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Visualization Module for Negative Space Imaging Project
Author: Stephen Bilodeau
Date: August 13, 2025

This module provides visualization tools for the analysis results.
"""

import os
import json
import logging
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Configure logging
logger = logging.getLogger(__name__)


class DataVisualizer:
    """Class for generating visualizations from imaging data."""

    def __init__(self):
        """Initialize the DataVisualizer."""
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.visualization_types = {
            'pca': self._generate_pca,
            'tsne': self._generate_tsne,
            'umap': self._generate_umap,
            'correlation': self._generate_correlation_matrix,
            'heatmap': self._generate_heatmap
        }

        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.2)

    def generate_visualization(self, input_dir, output_dir, viz_type='pca'):
        """
        Generate visualizations for the given data.

        Args:
            input_dir (str): Directory containing processed data
            output_dir (str): Directory to save visualizations
            viz_type (str): Type of visualization to generate
                Options: 'pca', 'tsne', 'umap', 'correlation', 'heatmap'

        Returns:
            str: Path to saved visualization
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        try:
            data, metadata = self._load_data(input_dir)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

        # Generate visualization
        if viz_type in self.visualization_types:
            logger.info(f"Generating {viz_type} visualization")
            viz_path = self.visualization_types[viz_type](data, metadata, output_dir)
            return viz_path
        else:
            available_types = ", ".join(self.visualization_types.keys())
            logger.error(f"Unsupported visualization type: {viz_type}. "
                         f"Available types: {available_types}")
            raise ValueError(f"Unsupported visualization type: {viz_type}")

    def _load_data(self, input_dir):
        """
        Load data from the input directory.

        Args:
            input_dir (str): Directory containing processed data

        Returns:
            tuple: (data, metadata)
        """
        input_path = Path(input_dir)

        # First try to load from numpy
        data_file = input_path / 'processed_data.npy'
        if data_file.exists():
            data = np.load(data_file)
            metadata_file = input_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                # Generate basic metadata
                metadata = {
                    'features': [f'feature_{i}' for i in range(data.shape[1])],
                    'data_type': 'numpy',
                    'shape': data.shape
                }
            return data, metadata

        # Try loading from CSV
        data_file = input_path / 'processed_data.csv'
        if data_file.exists():
            data = np.loadtxt(data_file, delimiter=',')
            metadata_file = input_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                # Generate basic metadata
                metadata = {
                    'features': [f'feature_{i}' for i in range(data.shape[1])],
                    'data_type': 'csv',
                    'shape': data.shape
                }
            return data, metadata

        # If we get here, no suitable data file was found
        raise FileNotFoundError(f"No suitable data file found in {input_dir}")

    def _generate_pca(self, data, metadata, output_dir):
        """Generate PCA visualization."""
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data)

        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        plt.title('PCA Visualization of Imaging Data')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.grid(True)

        # Add loading vectors
        feature_names = metadata.get('features', [f'Feature {i}' for i in range(data.shape[1])])
        if len(feature_names) <= 10:  # Only show loadings if not too crowded
            scale = 3
            for i, (name, x, y) in enumerate(zip(
                    feature_names,
                    pca.components_[0],
                    pca.components_[1]
            )):
                plt.arrow(0, 0, x * scale, y * scale, color='r', alpha=0.5)
                plt.text(x * scale * 1.2, y * scale * 1.2, name, color='g')

        # Save figure
        viz_path = os.path.join(output_dir, f'pca_visualization_{self.timestamp}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        return viz_path

    def _generate_tsne(self, data, metadata, output_dir):
        """Generate t-SNE visualization."""
        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        tsne_result = tsne.fit_transform(data)

        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7)
        plt.title('t-SNE Visualization of Imaging Data')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True)

        # Save figure
        viz_path = os.path.join(output_dir, f'tsne_visualization_{self.timestamp}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        return viz_path

    def _generate_umap(self, data, metadata, output_dir):
        """Generate UMAP visualization."""
        # Perform UMAP
        reducer = umap.UMAP(random_state=42)
        umap_result = reducer.fit_transform(data)

        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(umap_result[:, 0], umap_result[:, 1], alpha=0.7)
        plt.title('UMAP Visualization of Imaging Data')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.grid(True)

        # Save figure
        viz_path = os.path.join(output_dir, f'umap_visualization_{self.timestamp}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        return viz_path

    def _generate_correlation_matrix(self, data, metadata, output_dir):
        """Generate correlation matrix visualization."""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data, rowvar=False)

        # Create plot
        plt.figure(figsize=(12, 10))
        feature_names = metadata.get('features', [f'F{i}' for i in range(data.shape[1])])
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .5},
            xticklabels=feature_names,
            yticklabels=feature_names
        )
        plt.title('Feature Correlation Matrix')

        # Save figure
        viz_path = os.path.join(output_dir, f'correlation_matrix_{self.timestamp}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        return viz_path

    def _generate_heatmap(self, data, metadata, output_dir):
        """Generate heatmap visualization."""
        # Create plot
        plt.figure(figsize=(14, 10))
        feature_names = metadata.get('features', [f'F{i}' for i in range(data.shape[1])])

        # Use a subset of the data if it's too large
        sample_size = min(1000, data.shape[0])
        sampled_data = data[:sample_size]

        ax = sns.heatmap(
            sampled_data,
            cmap='viridis',
            xticklabels=feature_names,
            yticklabels=False
        )
        plt.title('Data Heatmap (Sample)')
        plt.xlabel('Features')
        plt.ylabel('Samples')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Save figure
        viz_path = os.path.join(output_dir, f'heatmap_{self.timestamp}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        return viz_path


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Generate data visualizations')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing processed data')
    parser.add_argument('--output-dir', type=str, default='analysis/visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--type', type=str, default='pca',
                        choices=['pca', 'tsne', 'umap', 'correlation', 'heatmap'],
                        help='Visualization type')

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    visualizer = DataVisualizer()
    viz_path = visualizer.generate_visualization(
        args.input_dir,
        args.output_dir,
        args.type
    )

    print(f"Visualization saved to: {viz_path}")
