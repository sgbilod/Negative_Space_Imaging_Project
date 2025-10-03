#!/usr/bin/env python
"""
Interactive Visualization Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides interactive visualization capabilities for negative space
analysis results, with a focus on real-time interaction and exploration.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .negative_space_algorithm import NegativeSpaceFeatures


@dataclass
class InteractiveConfig:
    """Configuration for interactive visualization."""
    animation_duration: int = 500
    marker_scale: float = 100.0
    colorscale: str = 'Viridis'
    show_scale: bool = True
    hover_template: str = (
        "Region: %{text}<br>"
        "Position: (%{x:.1f}, %{y:.1f})<br>"
        "Confidence: %{marker.color:.2%}"
    )


class InteractiveVisualizer:
    """Handles interactive visualization of negative space analysis."""
    
    def __init__(self, config: Optional[InteractiveConfig] = None):
        """Initialize interactive visualizer with config."""
        self.config = config or InteractiveConfig()
    
    def create_view(
        self,
        image: np.ndarray,
        features: Dict[str, List[NegativeSpaceFeatures]]
    ) -> go.Figure:
        """Create interactive visualization."""
        fig = go.Figure()
        
        # Base image layer
        fig.add_trace(go.Image(z=image))
        
        # Feature overlays
        self._add_feature_overlays(fig, features)
        
        # Layout configuration
        self._configure_layout(fig)
        
        return fig
    
    def _add_feature_overlays(
        self,
        fig: go.Figure,
        features: Dict[str, List[NegativeSpaceFeatures]]
    ):
        """Add feature overlays to the figure."""
        for region_id, region_features in features.items():
            for feature in region_features:
                # Main scatter point for region
                fig.add_trace(go.Scatter(
                    x=[feature.centroid[0]],
                    y=[feature.centroid[1]],
                    mode='markers+text',
                    marker=dict(
                        size=feature.area / self.config.marker_scale,
                        color=feature.confidence,
                        colorscale=self.config.colorscale,
                        showscale=self.config.show_scale,
                        line=dict(width=1, color='white')
                    ),
                    text=f"Region {region_id}",
                    hovertemplate=self.config.hover_template,
                    name=region_id
                ))
                
                # Circular boundary
                self._add_region_boundary(fig, feature)
    
    def _add_region_boundary(
        self,
        fig: go.Figure,
        feature: NegativeSpaceFeatures
    ):
        """Add circular boundary for a region."""
        radius = np.sqrt(feature.area / np.pi)
        theta = np.linspace(0, 2*np.pi, 100)
        x = feature.centroid[0] + radius * np.cos(theta)
        y = feature.centroid[1] + radius * np.sin(theta)
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(
                color='white',
                width=1,
                dash='dash'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    def _configure_layout(self, fig: go.Figure):
        """Configure figure layout and interactivity."""
        fig.update_layout(
            title=dict(
                text="Interactive Negative Space Analysis",
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="X Position",
            yaxis_title="Y Position",
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(
                    label="Reset View",
                    method="relayout",
                    args=[{"xaxis.range": None, "yaxis.range": None}]
                )]
            )],
            dragmode='zoom',
            paper_bgcolor='rgba(0,0,0,0.8)',
            plot_bgcolor='rgba(0,0,0,0.8)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False
            )
        )
