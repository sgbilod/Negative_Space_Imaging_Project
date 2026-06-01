"""
Negative Space Tracker

This module provides functionality for tracking changes in negative spaces over time.
It enables temporal analysis of how voids and interstitial spaces evolve.

Classes:
    NegativeSpaceTracker: Tracks changes in negative spaces over time
    TemporalSignature: Represents a signature that captures temporal patterns
    ChangeMetrics: Metrics for quantifying changes in negative spaces
"""

import os
import sys
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import Open3D with fallback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.open3d_support import o3d, np, OPEN3D_AVAILABLE

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """Types of changes that can occur in negative spaces"""
    EXPANSION = 1      # Void space has increased
    CONTRACTION = 2    # Void space has decreased
    EMERGENCE = 3      # New void space has appeared
    DISSOLUTION = 4    # Existing void space has disappeared
    DEFORMATION = 5    # Shape of void space has changed
    STABLE = 6         # No significant change


class ChangeMetrics:
    """Metrics for quantifying changes in negative spaces"""
    
    def __init__(self):
        """Initialize change metrics"""
        self.volume_delta: float = 0.0
        self.surface_area_delta: float = 0.0
        self.centroid_displacement: float = 0.0
        self.shape_deformation: float = 0.0
        self.signature_difference: float = 0.0
        self.void_count_delta: int = 0
        self.timestamp: datetime = datetime.now()
    
    def to_array(self) -> np.ndarray:
        """Convert metrics to a numpy array"""
        return np.array([
            self.volume_delta,
            self.surface_area_delta,
            self.centroid_displacement,
            self.shape_deformation,
            self.signature_difference,
            float(self.void_count_delta)
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'ChangeMetrics':
        """Create ChangeMetrics from a numpy array"""
        metrics = cls()
        metrics.volume_delta = array[0]
        metrics.surface_area_delta = array[1]
        metrics.centroid_displacement = array[2]
        metrics.shape_deformation = array[3]
        metrics.signature_difference = array[4]
        metrics.void_count_delta = int(array[5])
        return metrics
    
    def __str__(self) -> str:
        """String representation of change metrics"""
        return (
            f"ChangeMetrics(volume_delta={self.volume_delta:.4f}, "
            f"surface_area_delta={self.surface_area_delta:.4f}, "
            f"centroid_displacement={self.centroid_displacement:.4f}, "
            f"shape_deformation={self.shape_deformation:.4f}, "
            f"signature_difference={self.signature_difference:.4f}, "
            f"void_count_delta={self.void_count_delta})"
        )


class TemporalSignature:
    """Represents a signature that captures temporal patterns in negative space"""
    
    def __init__(self, window_size: int = 10):
        """
        Initialize a temporal signature
        
        Args:
            window_size: Number of time steps to include in the signature
        """
        self.window_size = window_size
        self.metrics_history: List[ChangeMetrics] = []
        self.signature: Optional[np.ndarray] = None
    
    def add_metrics(self, metrics: ChangeMetrics) -> None:
        """
        Add change metrics to the history
        
        Args:
            metrics: The change metrics to add
        """
        self.metrics_history.append(metrics)
        
        # Keep only the most recent window_size metrics
        if len(self.metrics_history) > self.window_size:
            self.metrics_history = self.metrics_history[-self.window_size:]
        
        # Update the signature
        self._compute_signature()
    
    def _compute_signature(self) -> None:
        """Compute the temporal signature from the metrics history"""
        if not self.metrics_history:
            self.signature = None
            return
        
        # Convert metrics to arrays
        metrics_arrays = [m.to_array() for m in self.metrics_history]
        
        # Pad with zeros if we don't have enough history
        while len(metrics_arrays) < self.window_size:
            metrics_arrays.append(np.zeros_like(metrics_arrays[0]))
        
        # Stack arrays
        metrics_matrix = np.vstack(metrics_arrays)
        
        # Compute statistical features over the time dimension
        means = np.mean(metrics_matrix, axis=0)
        stds = np.std(metrics_matrix, axis=0)
        maxs = np.max(metrics_matrix, axis=0)
        mins = np.min(metrics_matrix, axis=0)
        
        # Compute trends (simple linear regression coefficients)
        n = len(metrics_arrays)
        x = np.arange(n).reshape(-1, 1)
        trends = np.array([np.polyfit(x.flatten(), metrics_matrix[:, i], 1)[0] for i in range(metrics_matrix.shape[1])])
        
        # Combine features into signature
        self.signature = np.concatenate([means, stds, maxs, mins, trends])
    
    def visualize(self, output_path: Optional[str] = None) -> None:
        """
        Visualize the temporal signature
        
        Args:
            output_path: Optional path to save the visualization. If None, the
                       visualization is displayed but not saved.
        """
        if not self.metrics_history:
            logger.warning("No metrics history to visualize")
            return
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Temporal Negative Space Analysis', fontsize=16)
        
        # Extract time series for each metric
        times = list(range(len(self.metrics_history)))
        volume_deltas = [m.volume_delta for m in self.metrics_history]
        surface_area_deltas = [m.surface_area_delta for m in self.metrics_history]
        centroid_displacements = [m.centroid_displacement for m in self.metrics_history]
        shape_deformations = [m.shape_deformation for m in self.metrics_history]
        signature_differences = [m.signature_difference for m in self.metrics_history]
        void_count_deltas = [m.void_count_delta for m in self.metrics_history]
        
        # Plot each metric
        axs[0, 0].plot(times, volume_deltas, 'b-', marker='o')
        axs[0, 0].set_title('Volume Change')
        axs[0, 0].set_ylabel('Volume Delta')
        axs[0, 0].grid(True, alpha=0.3)
        
        axs[0, 1].plot(times, surface_area_deltas, 'g-', marker='o')
        axs[0, 1].set_title('Surface Area Change')
        axs[0, 1].set_ylabel('Surface Area Delta')
        axs[0, 1].grid(True, alpha=0.3)
        
        axs[1, 0].plot(times, centroid_displacements, 'r-', marker='o')
        axs[1, 0].set_title('Centroid Movement')
        axs[1, 0].set_ylabel('Displacement')
        axs[1, 0].grid(True, alpha=0.3)
        
        axs[1, 1].plot(times, shape_deformations, 'm-', marker='o')
        axs[1, 1].set_title('Shape Deformation')
        axs[1, 1].set_ylabel('Deformation Factor')
        axs[1, 1].grid(True, alpha=0.3)
        
        axs[2, 0].plot(times, signature_differences, 'c-', marker='o')
        axs[2, 0].set_title('Signature Difference')
        axs[2, 0].set_ylabel('Difference')
        axs[2, 0].grid(True, alpha=0.3)
        
        axs[2, 1].plot(times, void_count_deltas, 'y-', marker='o')
        axs[2, 1].set_title('Void Count Change')
        axs[2, 1].set_ylabel('Count Delta')
        axs[2, 1].grid(True, alpha=0.3)
        
        # Add timestamp to all x-axes
        for i in range(3):
            for j in range(2):
                axs[i, j].set_xlabel('Time Step')
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Temporal signature visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save(self, filepath: str) -> None:
        """
        Save the temporal signature to a file
        
        Args:
            filepath: Path to save the signature
        """
        if self.signature is None:
            logger.warning("No signature to save")
            return
        
        np.save(filepath, self.signature)
        logger.info(f"Temporal signature saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TemporalSignature':
        """
        Load a temporal signature from a file
        
        Args:
            filepath: Path to the saved signature
            
        Returns:
            TemporalSignature: The loaded signature
        """
        signature_obj = cls()
        signature_obj.signature = np.load(filepath)
        logger.info(f"Temporal signature loaded from {filepath}")
        return signature_obj


class NegativeSpaceTracker:
    """Tracks changes in negative spaces over time"""
    
    def __init__(self, memory_frames: int = 5):
        """
        Initialize a negative space tracker
        
        Args:
            memory_frames: Number of previous frames to store
        """
        self.memory_frames = memory_frames
        self.point_cloud_history = []
        self.change_history = []
        self.temporal_signature = TemporalSignature()
    
    def add_point_cloud(self, point_cloud: Any) -> ChangeMetrics:
        """
        Add a new point cloud to the tracker and compute changes
        
        Args:
            point_cloud: The point cloud to add (either Open3D or simplified)
            
        Returns:
            ChangeMetrics: Metrics describing the changes
        """
        # Add point cloud to history
        self.point_cloud_history.append(point_cloud)
        
        # Keep only the most recent memory_frames
        if len(self.point_cloud_history) > self.memory_frames:
            self.point_cloud_history = self.point_cloud_history[-self.memory_frames:]
        
        # If we don't have at least two frames, we can't compute changes
        if len(self.point_cloud_history) < 2:
            metrics = ChangeMetrics()
            self.change_history.append(metrics)
            self.temporal_signature.add_metrics(metrics)
            return metrics
        
        # Compute changes between the two most recent frames
        prev_cloud = self.point_cloud_history[-2]
        curr_cloud = self.point_cloud_history[-1]
        
        metrics = self._compute_changes(prev_cloud, curr_cloud)
        self.change_history.append(metrics)
        
        # Update temporal signature
        self.temporal_signature.add_metrics(metrics)
        
        return metrics
    
    def _compute_changes(self, prev_cloud: Any, curr_cloud: Any) -> ChangeMetrics:
        """
        Compute changes between two point clouds
        
        Args:
            prev_cloud: Previous point cloud
            curr_cloud: Current point cloud
            
        Returns:
            ChangeMetrics: Metrics describing the changes
        """
        metrics = ChangeMetrics()
        
        # Simplified metric calculations that work with or without Open3D
        
        # 1. Change in void points count
        if hasattr(prev_cloud, 'void_points') and hasattr(curr_cloud, 'void_points'):
            prev_void_count = len(prev_cloud.void_points)
            curr_void_count = len(curr_cloud.void_points)
            metrics.void_count_delta = curr_void_count - prev_void_count
        
        # 2. Rough estimate of volume change based on point counts
        # (actual volume calculation would need mesh creation)
        if hasattr(prev_cloud, 'void_points') and hasattr(curr_cloud, 'void_points'):
            prev_void_volume = len(prev_cloud.void_points)
            curr_void_volume = len(curr_cloud.void_points)
            metrics.volume_delta = (curr_void_volume - prev_void_volume) / max(1, prev_void_volume)
        
        # 3. Centroid displacement (if we have void points)
        if (hasattr(prev_cloud, 'void_points') and hasattr(curr_cloud, 'void_points') and
            len(prev_cloud.void_points) > 0 and len(curr_cloud.void_points) > 0):
            prev_centroid = np.mean(prev_cloud.void_points, axis=0)
            curr_centroid = np.mean(curr_cloud.void_points, axis=0)
            metrics.centroid_displacement = np.linalg.norm(curr_centroid - prev_centroid)
        
        # 4. Simple signature difference if available
        if hasattr(prev_cloud, 'compute_spatial_signature') and hasattr(curr_cloud, 'compute_spatial_signature'):
            prev_sig = prev_cloud.compute_spatial_signature()
            curr_sig = curr_cloud.compute_spatial_signature()
            if len(prev_sig) == len(curr_sig):
                metrics.signature_difference = np.linalg.norm(curr_sig - prev_sig)
        
        # More complex metrics that require Open3D for full implementation
        if OPEN3D_AVAILABLE:
            # These would use actual mesh creation and analysis
            # but we'll skip them if Open3D isn't available
            pass
        
        return metrics
    
    def get_change_type(self, metrics: ChangeMetrics) -> ChangeType:
        """
        Determine the type of change based on metrics
        
        Args:
            metrics: Change metrics to analyze
            
        Returns:
            ChangeType: The type of change that occurred
        """
        # Simple rules to determine change type
        if metrics.void_count_delta > 0 and metrics.volume_delta > 0.2:
            return ChangeType.EMERGENCE
        elif metrics.void_count_delta < 0 and metrics.volume_delta < -0.2:
            return ChangeType.DISSOLUTION
        elif metrics.volume_delta > 0.1:
            return ChangeType.EXPANSION
        elif metrics.volume_delta < -0.1:
            return ChangeType.CONTRACTION
        elif metrics.centroid_displacement > 0.1 or metrics.shape_deformation > 0.1:
            return ChangeType.DEFORMATION
        else:
            return ChangeType.STABLE
    
    def visualize_changes(self, output_path: Optional[str] = None) -> None:
        """
        Visualize the changes over time
        
        Args:
            output_path: Optional path to save the visualization. If None, the
                       visualization is displayed but not saved.
        """
        if not self.change_history:
            logger.warning("No change history to visualize")
            return
        
        # Use the temporal signature visualization
        self.temporal_signature.visualize(output_path)
    
    def get_temporal_signature(self) -> TemporalSignature:
        """
        Get the current temporal signature
        
        Returns:
            TemporalSignature: The current temporal signature
        """
        return self.temporal_signature
