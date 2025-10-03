#!/usr/bin/env python
"""
Negative Space Visualization
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides advanced visualization capabilities for negative space
analysis results, including:
- Interactive 3D visualization of negative spaces
- Pattern highlighting and annotation
- Real-time analysis visualization
- Comparative visualization tools
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
import torch
from dataclasses import dataclass

from .negative_space_algorithm import NegativeSpaceFeatures


@dataclass
class VisualizationConfig:
    """Configuration for visualization options."""
    show_boundaries: bool = True
    show_features: bool = True
    show_confidence: bool = True
    interactive: bool = True
    colormap: str = 'viridis'
    alpha: float = 0.7
    annotation_size: int = 10


class NegativeSpaceVisualizer:
    """Handles visualization of negative space analysis results."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualizer with optional configuration."""
        self.config = config or VisualizationConfig()
        self._setup_plotting_backend()
    
    def _setup_plotting_backend(self):
        """Configure the plotting backend based on environment."""
        plt.style.use('dark_background')
        # Enable interactive mode if requested
        if self.config.interactive:
            plt.ion()
    
    def visualize_analysis(
        self,
        image: np.ndarray,
        features: Dict[str, List[NegativeSpaceFeatures]],
        output_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Create comprehensive visualization of negative space analysis.
        
        Args:
            image: Original image
            features: Dictionary of negative space features
            output_path: Optional path to save visualization
            show: Whether to display the visualization
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Original image with highlighted negative spaces
        ax1 = plt.subplot(221)
        highlights = self._plot_original_with_highlights(ax1, image, features)
        
        # Add interactive tooltips for regions
        if self.config.interactive:
            for region_id, region_features in features.items():
                for feature in region_features:
                    tooltip = self._create_region_tooltip(region_id, feature)
                    plt.gca().annotate(
                        tooltip,
                        xy=feature.centroid,
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->'),
                        visible=False
                    )
        
        # 2. 3D topology visualization with interactive rotation
        ax2 = plt.subplot(222, projection='3d')
        topology = self._plot_3d_topology(ax2, features)
        if self.config.interactive:
            ax2.view_init(elev=30, azim=45)
            
        # 3. Feature distribution plot with highlighting
        ax3 = plt.subplot(223)
        dist_plot = self._plot_feature_distribution(ax3, features)
        if self.config.interactive:
            self._add_distribution_interactivity(dist_plot)
        
        # 4. Pattern confidence heatmap with hover info
        ax4 = plt.subplot(224)
        heatmap = self._plot_confidence_heatmap(ax4, features)
        if self.config.interactive:
            self._add_heatmap_interactivity(heatmap)
        
        # Add color bar for confidence scores
        if self.config.show_confidence:
            plt.colorbar(heatmap, ax=ax4, label='Confidence Score')
        
        plt.tight_layout()
        
        if self.config.interactive:
            self._setup_interactivity(fig, [highlights, topology, dist_plot, heatmap])
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
            
    def _create_region_tooltip(
        self,
        region_id: str,
        feature: NegativeSpaceFeatures
    ) -> str:
        """Create tooltip text for a region."""
        return (
            f"Region: {region_id}\n"
            f"Area: {feature.area:.1f}\n"
            f"Pattern Score: {feature.pattern_score:.2f}\n"
            f"Confidence: {feature.confidence:.2%}"
        )
    
    def _add_distribution_interactivity(self, dist_plot):
        """Add interactive features to distribution plot."""
        for bar in dist_plot:
            bar.set_picker(True)
            
    def _add_heatmap_interactivity(self, heatmap):
        """Add interactive features to heatmap."""
        heatmap.set_picker(True)
        
    def _setup_interactivity(self, fig, plots):
        """Setup interactive features for the visualization."""
        def on_pick(event):
            if event.mouseevent.inaxes == plt.gca():
                artist = event.artist
                if isinstance(artist, plt.Rectangle):  # Bar plot
                    value = artist.get_height()
                    plt.gca().set_title(f'Selected value: {value:.2f}')
                elif isinstance(artist, plt.QuadMesh):  # Heatmap
                    val = artist.get_array()[event.ind[0]]
                    plt.gca().set_title(f'Confidence: {val:.2%}')
                
        def on_mouse_move(event):
            if event.inaxes:
                for annotation in event.inaxes.texts:
                    # Show annotation if mouse is close to its position
                    if isinstance(annotation, plt.Annotation):
                        annotation.set_visible(
                            np.linalg.norm(
                                np.array([event.xdata, event.ydata])
                                - annotation.xy
                            ) < 10
                        )
                plt.draw()
                
        fig.canvas.mpl_connect('pick_event', on_pick)
        fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    
    def _plot_original_with_highlights(
        self,
        ax,
        image: np.ndarray,
        features: Dict[str, List[NegativeSpaceFeatures]]
    ):
        """Plot original image with highlighted negative spaces."""
        # Display original image
        ax.imshow(image, cmap='gray')
        
        # Create overlay for negative spaces
        overlay = np.zeros_like(image)
        
        # Plot each region with different colors
        colors = plt.cm.get_cmap(self.config.colormap)(
            np.linspace(0, 1, len(features))
        )
        
        highlights = []
        for (region_id, region_features), color in zip(features.items(), colors):
            for feature in region_features:
                # Create circular highlight around centroid
                circle = plt.Circle(
                    feature.centroid,
                    np.sqrt(feature.area / np.pi),
                    color=color,
                    alpha=self.config.alpha,
                    fill=False,
                    linewidth=2
                )
                ax.add_patch(circle)
                highlights.append(circle)
                
                if self.config.show_features:
                    # Add feature labels
                    ax.text(
                        feature.centroid[0],
                        feature.centroid[1],
                        f'{region_id}\n{feature.pattern_score:.2f}',
                        color='white',
                        fontsize=self.config.annotation_size,
                        ha='center',
                        va='center',
                        bbox=dict(
                            facecolor='black',
                            alpha=0.7,
                            pad=1
                        )
                    )
        
        ax.set_title('Negative Space Analysis')
        return highlights
    
    def _plot_3d_topology(
        self,
        ax,
        features: Dict[str, List[NegativeSpaceFeatures]]
    ):
        """Create 3D visualization of negative space topology."""
        # Extract feature coordinates for 3D plotting
        xs, ys, zs = [], [], []
        colors = []
        sizes = []
        
        for region_features in features.values():
            for feature in region_features:
                xs.append(feature.centroid[0])
                ys.append(feature.centroid[1])
                zs.append(feature.topology_index)
                colors.append(feature.confidence)
                sizes.append(feature.area / 100)
        
        # Create 3D scatter plot
        scatter = ax.scatter(
            xs, ys, zs,
            c=colors,
            s=sizes,
            cmap=self.config.colormap,
            alpha=self.config.alpha
        )
        
        # Add connecting lines between related points
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            for j in range(i + 1, len(xs)):
                # Connect points if they're close in feature space
                distance = np.sqrt(
                    (x - xs[j])**2 + (y - ys[j])**2 + (z - zs[j])**2
                )
                if distance < np.mean([sizes[i], sizes[j]]):
                    ax.plot(
                        [x, xs[j]],
                        [y, ys[j]],
                        [z, zs[j]],
                        'gray',
                        alpha=0.3
                    )
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Topology Index')
        ax.set_title('3D Topology Visualization')
        
        return scatter
    
    def _plot_feature_distribution(
        self,
        ax,
        features: Dict[str, List[NegativeSpaceFeatures]]
    ):
        """Plot distribution of feature values."""
        # Collect all feature values
        feature_values = {
            'area': [],
            'perimeter': [],
            'topology': [],
            'connectivity': [],
            'confidence': []
        }
        
        for region_features in features.values():
            for feature in region_features:
                feature_values['area'].append(feature.area)
                feature_values['perimeter'].append(feature.perimeter)
                feature_values['topology'].append(feature.topology_index)
                feature_values['connectivity'].append(feature.connectivity)
                feature_values['confidence'].append(feature.confidence)
        
        # Create grouped bar plot
        x = np.arange(len(feature_values))
        width = 0.35
        
        means = [np.mean(vals) for vals in feature_values.values()]
        stds = [np.std(vals) for vals in feature_values.values()]
        
        bars = ax.bar(
            x,
            means,
            width,
            yerr=stds,
            alpha=self.config.alpha,
            color=plt.cm.get_cmap(self.config.colormap)(np.linspace(0, 1, 5))
        )
        
        ax.set_ylabel('Value')
        ax.set_title('Feature Distributions')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_values.keys(), rotation=45)
        
        return bars
    
    def _plot_confidence_heatmap(
        self,
        ax,
        features: Dict[str, List[NegativeSpaceFeatures]]
    ):
        """Create heatmap of confidence scores."""
        from scipy.ndimage import gaussian_filter
        
        # Create 2D grid of confidence scores
        grid_size = 50
        x = np.linspace(
            min(f.centroid[0] for fs in features.values() for f in fs),
            max(f.centroid[0] for fs in features.values() for f in fs),
            grid_size
        )
        y = np.linspace(
            min(f.centroid[1] for fs in features.values() for f in fs),
            max(f.centroid[1] for fs in features.values() for f in fs),
            grid_size
        )
        
        confidence_grid = np.zeros((grid_size, grid_size))
        
        # Interpolate confidence values
        for region_features in features.values():
            for feature in region_features:
                i = np.argmin(np.abs(x - feature.centroid[0]))
                j = np.argmin(np.abs(y - feature.centroid[1]))
                confidence_grid[j, i] = feature.confidence
        
        # Apply Gaussian smoothing
        confidence_grid = gaussian_filter(confidence_grid, sigma=1)
        
        # Create heatmap
        heatmap = ax.imshow(
            confidence_grid,
            extent=[x[0], x[-1], y[0], y[-1]],
            origin='lower',
            cmap=self.config.colormap,
            alpha=self.config.alpha
        )
        
        ax.set_title('Confidence Heatmap')
        
        return heatmap

    def create_interactive_view(
        self,
        image: np.ndarray,
        features: Dict[str, List[NegativeSpaceFeatures]]
    ) -> go.Figure:
        """Create an interactive Plotly visualization."""
        fig = go.Figure()
        
        # Add the original image as a background
        fig.add_trace(go.Image(z=image))
        
        # Add negative space overlays
        for region_id, region_features in features.items():
            for feature in region_features:
                fig.add_trace(go.Scatter(
                    x=[feature.centroid[0]],
                    y=[feature.centroid[1]],
                    mode='markers+text',
                    marker=dict(
                        size=feature.area / 100,
                        color=feature.confidence,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=f"Region {region_id}",
                    hoverinfo='text+x+y'
                ))
        
        fig.update_layout(
            title="Interactive Negative Space Analysis",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            hovermode='closest',
            showlegend=False,
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [{
                    "label": "Play Animation",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 500, "redraw": True},
                        "fromcurrent": True
                    }]
                }]
            }]
        )
        return fig
            title="Interactive Negative Space Analysis",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            hovermode='closest'
        )
        
        return fig
