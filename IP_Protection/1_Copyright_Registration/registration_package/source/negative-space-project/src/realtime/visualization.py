"""
Advanced Visualization Module for Real-time Negative Space Analysis

This module provides enhanced visualization capabilities for real-time negative space
analysis, including 3D rendering, interactive displays, and augmented reality overlays.

Classes:
    RealTimeVisualizer: Core visualization framework
    AROverlay: Augmented reality visualization overlays
    PerformanceDisplay: Real-time performance metrics display
    InteractiveControls: Interactive control elements for visualization
"""

import os
import sys
import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from enum import Enum
from pathlib import Path
import math

# Import from centralized fallbacks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.fallbacks import (
    np, cv2, plt, o3d, NUMPY_AVAILABLE, OPENCV_AVAILABLE, 
    MATPLOTLIB_AVAILABLE, OPEN3D_AVAILABLE
)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisualizationMode(Enum):
    """Visualization modes for the real-time visualizer"""
    BASIC = 1        # Simple 2D visualization
    ADVANCED_2D = 2  # Enhanced 2D visualization with overlays
    BASIC_3D = 3     # Basic 3D visualization
    ADVANCED_3D = 4  # Advanced 3D visualization with effects
    AR = 5           # Augmented reality visualization


class OverlayType(Enum):
    """Types of overlays for visualization"""
    BOUNDING_BOX = 1       # Bounding boxes around objects
    NEGATIVE_SPACE = 2     # Highlight negative spaces
    HEATMAP = 3            # Heatmap of interesting areas
    METRICS = 4            # Display performance metrics
    TRAJECTORY = 5         # Show object trajectories
    SIGNATURE = 6          # Display spatial signatures
    CONFIDENCE = 7         # Confidence levels for detections
    BLOCKCHAIN_STATUS = 8  # Blockchain verification status


class ColorPalette:
    """Color palette for visualization"""
    
    # Standard RGB colors (values between 0 and 1)
    WHITE = (1.0, 1.0, 1.0)
    BLACK = (0.0, 0.0, 0.0)
    RED = (1.0, 0.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)
    YELLOW = (1.0, 1.0, 0.0)
    CYAN = (0.0, 1.0, 1.0)
    MAGENTA = (1.0, 0.0, 1.0)
    
    # Advanced colors
    DEEP_BLUE = (0.0, 0.2, 0.8)
    LIGHT_BLUE = (0.6, 0.8, 1.0)
    ORANGE = (1.0, 0.6, 0.0)
    PURPLE = (0.5, 0.0, 0.5)
    LIME = (0.6, 1.0, 0.2)
    TEAL = (0.0, 0.8, 0.8)
    PINK = (1.0, 0.4, 0.6)
    GOLD = (1.0, 0.84, 0.0)
    
    # Specialized colors for negative space visualization
    VOID_COLOR = (0.2, 0.0, 0.4)  # Dark purple for voids
    BOUNDARY_COLOR = (0.0, 0.7, 0.7)  # Teal for boundaries
    OBJECT_COLOR = (0.1, 0.5, 0.9)  # Blue for objects
    HIGHLIGHT_COLOR = (1.0, 0.5, 0.0)  # Orange for highlights
    
    @staticmethod
    def get_sequential_color(index: int, total: int) -> Tuple[float, float, float]:
        """
        Get a color from a sequential palette
        
        Args:
            index: Index of the color
            total: Total number of colors needed
            
        Returns:
            RGB color tuple
        """
        hue = index / max(1, total)
        saturation = 0.8
        value = 0.9
        
        if not NUMPY_AVAILABLE:
            # Fallback: use a few basic colors
            colors = [
                ColorPalette.RED, ColorPalette.GREEN, ColorPalette.BLUE,
                ColorPalette.YELLOW, ColorPalette.CYAN, ColorPalette.MAGENTA,
                ColorPalette.ORANGE, ColorPalette.PURPLE, ColorPalette.LIME
            ]
            return colors[index % len(colors)]
        
        # Convert HSV to RGB
        c = value * saturation
        x = c * (1 - abs((hue * 6) % 2 - 1))
        m = value - c
        
        if hue < 1/6:
            r, g, b = c, x, 0
        elif hue < 2/6:
            r, g, b = x, c, 0
        elif hue < 3/6:
            r, g, b = 0, c, x
        elif hue < 4/6:
            r, g, b = 0, x, c
        elif hue < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (r + m, g + m, b + m)
    
    @staticmethod
    def get_diverging_color(value: float, min_val: float = 0.0, max_val: float = 1.0) -> Tuple[float, float, float]:
        """
        Get a color from a diverging palette (blue to red)
        
        Args:
            value: Value to map to color
            min_val: Minimum value in range
            max_val: Maximum value in range
            
        Returns:
            RGB color tuple
        """
        if not NUMPY_AVAILABLE:
            # Fallback: use a few basic colors
            if value < (min_val + max_val) / 2:
                return ColorPalette.BLUE
            else:
                return ColorPalette.RED
        
        # Normalize value to [0, 1]
        norm_value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        
        # Clamp to [0, 1]
        norm_value = max(0.0, min(1.0, norm_value))
        
        # Blue to red diverging palette
        if norm_value < 0.5:
            # Blue to white
            t = norm_value * 2
            r = t
            g = t
            b = 1.0
        else:
            # White to red
            t = (norm_value - 0.5) * 2
            r = 1.0
            g = 1.0 - t
            b = 1.0 - t
        
        return (r, g, b)
    
    @staticmethod
    def to_uint8(color: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """
        Convert a color from float [0, 1] to uint8 [0, 255]
        
        Args:
            color: RGB color tuple with values between 0 and 1
            
        Returns:
            RGB color tuple with values between 0 and 255
        """
        return (
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255)
        )
    
    @staticmethod
    def to_bgr(color: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """
        Convert an RGB color to BGR (for OpenCV)
        
        Args:
            color: RGB color tuple with values between 0 and 1
            
        Returns:
            BGR color tuple with values between 0 and 255
        """
        r, g, b = ColorPalette.to_uint8(color)
        return (b, g, r)


class PerformanceDisplay:
    """Display for real-time performance metrics"""
    
    def __init__(self, position: Tuple[int, int] = (10, 30), 
                 font_scale: float = 0.7, thickness: int = 2):
        """
        Initialize a performance display
        
        Args:
            position: Position of the first line of text (x, y)
            font_scale: Font scale for text
            thickness: Thickness of text
        """
        self.position = position
        self.font_scale = font_scale
        self.thickness = thickness
        self.line_height = int(30 * font_scale)
        self.metrics = {}
        self.background_opacity = 0.5
        self.history = {}
        self.history_length = 100
    
    def update_metric(self, name: str, value: float):
        """
        Update a performance metric
        
        Args:
            name: Name of the metric
            value: Current value of the metric
        """
        self.metrics[name] = value
        
        # Update history
        if name not in self.history:
            self.history[name] = []
        
        self.history[name].append(value)
        
        # Keep history at fixed length
        if len(self.history[name]) > self.history_length:
            self.history[name] = self.history[name][-self.history_length:]
    
    def draw(self, frame: Any) -> Any:
        """
        Draw performance metrics on a frame
        
        Args:
            frame: Frame to draw on
            
        Returns:
            Frame with metrics drawn on it
        """
        if not OPENCV_AVAILABLE or frame is None:
            return frame
        
        # Create a copy of the frame
        display = frame.copy()
        
        # Draw semi-transparent background for better readability
        metrics_height = len(self.metrics) * self.line_height
        metrics_width = 300
        bg_rect = (
            self.position[0] - 5,
            self.position[1] - 25,
            metrics_width,
            metrics_height + 30
        )
        
        overlay = display.copy()
        cv2.rectangle(
            overlay,
            (bg_rect[0], bg_rect[1]),
            (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]),
            (0, 0, 0),
            -1
        )
        
        # Apply overlay with transparency
        cv2.addWeighted(
            overlay, self.background_opacity,
            display, 1 - self.background_opacity,
            0, display
        )
        
        # Draw header
        cv2.putText(
            display,
            "Performance Metrics",
            (self.position[0], self.position[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            ColorPalette.to_bgr(ColorPalette.GOLD),
            self.thickness
        )
        
        # Draw metrics
        for i, (name, value) in enumerate(self.metrics.items()):
            y = self.position[1] + i * self.line_height
            
            # Determine color based on history
            color = ColorPalette.WHITE
            if name in self.history and len(self.history[name]) > 1:
                prev_value = self.history[name][-2] if len(self.history[name]) > 1 else value
                if value > prev_value:
                    color = ColorPalette.GREEN  # Increasing
                elif value < prev_value:
                    color = ColorPalette.RED    # Decreasing
            
            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            
            # Draw text
            cv2.putText(
                display,
                f"{name}: {formatted_value}",
                (self.position[0], y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                ColorPalette.to_bgr(color),
                self.thickness
            )
        
        return display
    
    def create_history_plot(self, metric_name: str, width: int = 300, height: int = 100) -> Optional[Any]:
        """
        Create a plot of a metric's history
        
        Args:
            metric_name: Name of the metric to plot
            width: Width of the plot
            height: Height of the plot
            
        Returns:
            Image of the plot, or None if matplotlib is not available
        """
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return None
        
        if metric_name not in self.history:
            return None
        
        # Create a figure and axis
        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.plot(self.history[metric_name], color='green')
        plt.title(metric_name)
        plt.grid(True)
        
        # Save to a buffer
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert to OpenCV image
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_arr, 1)
        
        return img


class AROverlay:
    """Augmented reality overlay for real-time visualization"""
    
    def __init__(self):
        """Initialize an AR overlay"""
        self.overlays = {}
        self.enabled = True
        self.opacity = 0.7
    
    def add_overlay(self, name: str, overlay_type: OverlayType):
        """
        Add an overlay
        
        Args:
            name: Name of the overlay
            overlay_type: Type of overlay
        """
        self.overlays[name] = {
            'type': overlay_type,
            'enabled': True,
            'data': None
        }
    
    def update_overlay_data(self, name: str, data: Any):
        """
        Update overlay data
        
        Args:
            name: Name of the overlay
            data: Data for the overlay
        """
        if name in self.overlays:
            self.overlays[name]['data'] = data
    
    def enable_overlay(self, name: str):
        """
        Enable an overlay
        
        Args:
            name: Name of the overlay
        """
        if name in self.overlays:
            self.overlays[name]['enabled'] = True
    
    def disable_overlay(self, name: str):
        """
        Disable an overlay
        
        Args:
            name: Name of the overlay
        """
        if name in self.overlays:
            self.overlays[name]['enabled'] = False
    
    def draw(self, frame: Any, point_cloud: Any) -> Any:
        """
        Draw AR overlays on a frame
        
        Args:
            frame: Frame to draw on
            point_cloud: Point cloud data
            
        Returns:
            Frame with overlays drawn on it
        """
        if not self.enabled or not OPENCV_AVAILABLE or frame is None:
            return frame
        
        # Create a copy of the frame
        display = frame.copy()
        
        # Draw each enabled overlay
        for name, overlay in self.overlays.items():
            if overlay['enabled'] and overlay['data'] is not None:
                display = self._draw_overlay(
                    display,
                    overlay['type'],
                    overlay['data'],
                    point_cloud
                )
        
        return display
    
    def _draw_overlay(self, frame: Any, overlay_type: OverlayType, data: Any, point_cloud: Any) -> Any:
        """
        Draw a specific overlay
        
        Args:
            frame: Frame to draw on
            overlay_type: Type of overlay
            data: Data for the overlay
            point_cloud: Point cloud data
            
        Returns:
            Frame with overlay drawn on it
        """
        if overlay_type == OverlayType.BOUNDING_BOX:
            return self._draw_bounding_boxes(frame, data)
        
        elif overlay_type == OverlayType.NEGATIVE_SPACE:
            return self._draw_negative_spaces(frame, data, point_cloud)
        
        elif overlay_type == OverlayType.HEATMAP:
            return self._draw_heatmap(frame, data)
        
        elif overlay_type == OverlayType.TRAJECTORY:
            return self._draw_trajectories(frame, data)
        
        elif overlay_type == OverlayType.SIGNATURE:
            return self._draw_signature(frame, data)
        
        elif overlay_type == OverlayType.CONFIDENCE:
            return self._draw_confidence(frame, data)
        
        elif overlay_type == OverlayType.BLOCKCHAIN_STATUS:
            return self._draw_blockchain_status(frame, data)
        
        return frame
    
    def _draw_bounding_boxes(self, frame: Any, boxes: List[Dict]) -> Any:
        """
        Draw bounding boxes on a frame
        
        Args:
            frame: Frame to draw on
            boxes: List of bounding boxes
            
        Returns:
            Frame with bounding boxes drawn on it
        """
        display = frame.copy()
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.get('coords', (0, 0, 0, 0))
            label = box.get('label', '')
            confidence = box.get('confidence', 1.0)
            
            # Determine color based on confidence
            color = ColorPalette.to_bgr(
                ColorPalette.get_diverging_color(confidence)
            )
            
            # Draw box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(
                display,
                f"{label} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return display
    
    def _draw_negative_spaces(self, frame: Any, spaces: List[Dict], point_cloud: Any) -> Any:
        """
        Draw negative spaces on a frame
        
        Args:
            frame: Frame to draw on
            spaces: List of negative spaces
            point_cloud: Point cloud data
            
        Returns:
            Frame with negative spaces drawn on it
        """
        display = frame.copy()
        
        # Create an overlay for the negative spaces
        overlay = display.copy()
        
        for i, space in enumerate(spaces):
            # Get space center and radius (simplified)
            center = space.get('center', (0, 0))
            radius = space.get('radius', 50)
            importance = space.get('importance', 0.5)
            
            # Convert 3D center to 2D coordinates (simplified)
            # In a real implementation, this would use proper 3D to 2D projection
            center_x = int(center[0] * frame.shape[1] + frame.shape[1] / 2)
            center_y = int(center[1] * frame.shape[0] + frame.shape[0] / 2)
            
            # Determine color based on importance
            color = ColorPalette.to_bgr(
                ColorPalette.get_diverging_color(importance)
            )
            
            # Draw filled circle
            cv2.circle(overlay, (center_x, center_y), radius, color, -1)
            
            # Draw label
            cv2.putText(
                overlay,
                f"Void {i}",
                (center_x - 20, center_y - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Apply overlay with transparency
        cv2.addWeighted(
            overlay, self.opacity,
            display, 1 - self.opacity,
            0, display
        )
        
        return display
    
    def _draw_heatmap(self, frame: Any, heatmap_data: Dict) -> Any:
        """
        Draw a heatmap on a frame
        
        Args:
            frame: Frame to draw on
            heatmap_data: Heatmap data
            
        Returns:
            Frame with heatmap drawn on it
        """
        if not NUMPY_AVAILABLE or not OPENCV_AVAILABLE:
            return frame
        
        display = frame.copy()
        
        # Get heatmap matrix and scale
        heatmap_matrix = heatmap_data.get('matrix', None)
        min_val = heatmap_data.get('min_val', 0.0)
        max_val = heatmap_data.get('max_val', 1.0)
        
        if heatmap_matrix is None:
            return display
        
        # Resize heatmap to match frame
        heatmap_resized = cv2.resize(
            heatmap_matrix,
            (display.shape[1], display.shape[0])
        )
        
        # Normalize heatmap
        heatmap_normalized = (heatmap_resized - min_val) / (max_val - min_val)
        heatmap_normalized = np.clip(heatmap_normalized, 0, 1)
        
        # Convert to color map
        heatmap_color = cv2.applyColorMap(
            (heatmap_normalized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Apply overlay with transparency
        cv2.addWeighted(
            heatmap_color, self.opacity,
            display, 1 - self.opacity,
            0, display
        )
        
        return display
    
    def _draw_trajectories(self, frame: Any, trajectories: List[Dict]) -> Any:
        """
        Draw object trajectories on a frame
        
        Args:
            frame: Frame to draw on
            trajectories: List of trajectories
            
        Returns:
            Frame with trajectories drawn on it
        """
        display = frame.copy()
        
        for traj in trajectories:
            # Get trajectory points
            points = traj.get('points', [])
            label = traj.get('label', '')
            
            if not points:
                continue
            
            # Determine color
            color = ColorPalette.to_bgr(
                ColorPalette.get_sequential_color(
                    traj.get('id', 0) % 10, 10
                )
            )
            
            # Draw trajectory line
            for i in range(1, len(points)):
                cv2.line(
                    display,
                    (int(points[i-1][0]), int(points[i-1][1])),
                    (int(points[i][0]), int(points[i][1])),
                    color,
                    2
                )
            
            # Draw current position
            cv2.circle(
                display,
                (int(points[-1][0]), int(points[-1][1])),
                5,
                color,
                -1
            )
            
            # Draw label
            cv2.putText(
                display,
                label,
                (int(points[-1][0]) + 10, int(points[-1][1]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return display
    
    def _draw_signature(self, frame: Any, signature_data: Dict) -> Any:
        """
        Draw a spatial signature on a frame
        
        Args:
            frame: Frame to draw on
            signature_data: Signature data
            
        Returns:
            Frame with signature drawn on it
        """
        if not NUMPY_AVAILABLE or not OPENCV_AVAILABLE:
            return frame
        
        display = frame.copy()
        
        # Get signature values
        values = signature_data.get('values', [])
        label = signature_data.get('label', 'Spatial Signature')
        
        if not values:
            return display
        
        # Create signature visualization
        h, w = 100, 400
        margin = 50
        
        # Create visualization area
        vis_area = np.zeros((h + 2 * margin, w + 2 * margin, 3), dtype=np.uint8)
        
        # Draw signature bars
        n_values = len(values)
        bar_width = w // n_values
        
        for i, value in enumerate(values):
            # Determine bar height
            bar_height = int(value * h)
            
            # Determine color
            color = ColorPalette.to_bgr(
                ColorPalette.get_sequential_color(i, n_values)
            )
            
            # Draw bar
            cv2.rectangle(
                vis_area,
                (margin + i * bar_width, margin + h - bar_height),
                (margin + (i + 1) * bar_width - 1, margin + h),
                color,
                -1
            )
        
        # Draw border and label
        cv2.rectangle(
            vis_area,
            (margin, margin),
            (margin + w, margin + h),
            (255, 255, 255),
            1
        )
        
        cv2.putText(
            vis_area,
            label,
            (margin, margin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Place signature visualization in bottom-right corner
        sig_h, sig_w = vis_area.shape[:2]
        x_offset = display.shape[1] - sig_w
        y_offset = display.shape[0] - sig_h
        
        if x_offset >= 0 and y_offset >= 0:
            # Create ROI
            roi = display[y_offset:y_offset+sig_h, x_offset:x_offset+sig_w]
            
            # Create mask
            mask = np.where(vis_area > 0, 1, 0).astype(np.uint8)
            mask_inv = 1 - mask
            
            # Apply mask
            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            fg = cv2.bitwise_and(vis_area, vis_area, mask=mask)
            
            # Combine
            result = cv2.add(bg, fg)
            display[y_offset:y_offset+sig_h, x_offset:x_offset+sig_w] = result
        
        return display
    
    def _draw_confidence(self, frame: Any, confidence_data: Dict) -> Any:
        """
        Draw confidence indicators on a frame
        
        Args:
            frame: Frame to draw on
            confidence_data: Confidence data
            
        Returns:
            Frame with confidence indicators drawn on it
        """
        display = frame.copy()
        
        # Get confidence values
        overall = confidence_data.get('overall', 0.0)
        details = confidence_data.get('details', {})
        
        # Draw overall confidence bar
        bar_width = 200
        bar_height = 20
        margin = 10
        
        # Background
        cv2.rectangle(
            display,
            (margin, margin),
            (margin + bar_width, margin + bar_height),
            (100, 100, 100),
            -1
        )
        
        # Filled portion
        filled_width = int(overall * bar_width)
        color = ColorPalette.to_bgr(
            ColorPalette.get_diverging_color(overall)
        )
        
        cv2.rectangle(
            display,
            (margin, margin),
            (margin + filled_width, margin + bar_height),
            color,
            -1
        )
        
        # Label
        cv2.putText(
            display,
            f"Confidence: {overall:.2f}",
            (margin, margin + bar_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Draw detailed confidence values
        y_offset = margin + bar_height + 40
        for label, value in details.items():
            # Filled portion
            filled_width = int(value * bar_width)
            color = ColorPalette.to_bgr(
                ColorPalette.get_diverging_color(value)
            )
            
            # Background
            cv2.rectangle(
                display,
                (margin, y_offset),
                (margin + bar_width, y_offset + bar_height),
                (100, 100, 100),
                -1
            )
            
            # Filled portion
            cv2.rectangle(
                display,
                (margin, y_offset),
                (margin + filled_width, y_offset + bar_height),
                color,
                -1
            )
            
            # Label
            cv2.putText(
                display,
                f"{label}: {value:.2f}",
                (margin, y_offset + bar_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            y_offset += bar_height + 30
        
        return display
    
    def _draw_blockchain_status(self, frame: Any, status_data: Dict) -> Any:
        """
        Draw blockchain verification status on a frame
        
        Args:
            frame: Frame to draw on
            status_data: Blockchain status data
            
        Returns:
            Frame with blockchain status drawn on it
        """
        display = frame.copy()
        
        # Get status values
        verified = status_data.get('verified', False)
        hash_value = status_data.get('hash', '')
        timestamp = status_data.get('timestamp', '')
        
        # Determine color based on verification status
        color = ColorPalette.to_bgr(ColorPalette.GREEN if verified else ColorPalette.RED)
        
        # Draw status box
        margin = 10
        box_width = 350
        box_height = 120
        
        # Position in top-right corner
        x = display.shape[1] - box_width - margin
        y = margin
        
        # Draw semi-transparent background
        overlay = display.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + box_width, y + box_height),
            (0, 0, 0),
            -1
        )
        
        # Apply overlay with transparency
        cv2.addWeighted(
            overlay, 0.7,
            display, 0.3,
            0, display
        )
        
        # Draw header
        cv2.putText(
            display,
            "Blockchain Verification",
            (x + 10, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Draw status
        status_text = "VERIFIED" if verified else "NOT VERIFIED"
        cv2.putText(
            display,
            status_text,
            (x + 10, y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        # Draw hash (truncated)
        hash_short = hash_value[:20] + "..." if len(hash_value) > 20 else hash_value
        cv2.putText(
            display,
            f"Hash: {hash_short}",
            (x + 10, y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Draw timestamp
        cv2.putText(
            display,
            f"Time: {timestamp}",
            (x + 10, y + 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return display


class InteractiveControls:
    """Interactive control elements for visualization"""
    
    def __init__(self):
        """Initialize interactive controls"""
        self.controls = {}
        self.enabled = True
        self.hover_control = None
        self.active_control = None
    
    def add_button(self, name: str, position: Tuple[int, int], 
                  size: Tuple[int, int], label: str, 
                  callback: Callable[[], None]):
        """
        Add a button control
        
        Args:
            name: Name of the button
            position: Position of the button (x, y)
            size: Size of the button (width, height)
            label: Label text for the button
            callback: Function to call when the button is clicked
        """
        self.controls[name] = {
            'type': 'button',
            'position': position,
            'size': size,
            'label': label,
            'callback': callback,
            'state': 'normal'  # normal, hover, pressed
        }
    
    def add_slider(self, name: str, position: Tuple[int, int], 
                  size: Tuple[int, int], label: str, 
                  min_value: float, max_value: float, 
                  current_value: float, 
                  callback: Callable[[float], None]):
        """
        Add a slider control
        
        Args:
            name: Name of the slider
            position: Position of the slider (x, y)
            size: Size of the slider (width, height)
            label: Label text for the slider
            min_value: Minimum value
            max_value: Maximum value
            current_value: Current value
            callback: Function to call when the slider value changes
        """
        self.controls[name] = {
            'type': 'slider',
            'position': position,
            'size': size,
            'label': label,
            'min_value': min_value,
            'max_value': max_value,
            'current_value': current_value,
            'callback': callback,
            'state': 'normal'  # normal, hover, active
        }
    
    def add_checkbox(self, name: str, position: Tuple[int, int], 
                    size: Tuple[int, int], label: str, 
                    checked: bool, 
                    callback: Callable[[bool], None]):
        """
        Add a checkbox control
        
        Args:
            name: Name of the checkbox
            position: Position of the checkbox (x, y)
            size: Size of the checkbox (width, height)
            label: Label text for the checkbox
            checked: Whether the checkbox is checked
            callback: Function to call when the checkbox state changes
        """
        self.controls[name] = {
            'type': 'checkbox',
            'position': position,
            'size': size,
            'label': label,
            'checked': checked,
            'callback': callback,
            'state': 'normal'  # normal, hover
        }
    
    def handle_mouse_move(self, x: int, y: int):
        """
        Handle mouse movement
        
        Args:
            x: Mouse x coordinate
            y: Mouse y coordinate
        """
        if not self.enabled:
            return
        
        # Reset hover state
        if self.hover_control:
            self.controls[self.hover_control]['state'] = 'normal'
            self.hover_control = None
        
        # Check if mouse is over a control
        for name, control in self.controls.items():
            if self._is_point_in_rect(x, y, control['position'], control['size']):
                if name != self.active_control:
                    control['state'] = 'hover'
                    self.hover_control = name
                break
    
    def handle_mouse_down(self, x: int, y: int):
        """
        Handle mouse button press
        
        Args:
            x: Mouse x coordinate
            y: Mouse y coordinate
        """
        if not self.enabled:
            return
        
        # Check if mouse is over a control
        for name, control in self.controls.items():
            if self._is_point_in_rect(x, y, control['position'], control['size']):
                if control['type'] == 'button':
                    control['state'] = 'pressed'
                    self.active_control = name
                
                elif control['type'] == 'slider':
                    control['state'] = 'active'
                    self.active_control = name
                    self._update_slider_value(name, x)
                
                elif control['type'] == 'checkbox':
                    control['checked'] = not control['checked']
                    if control['callback']:
                        control['callback'](control['checked'])
                
                break
    
    def handle_mouse_up(self, x: int, y: int):
        """
        Handle mouse button release
        
        Args:
            x: Mouse x coordinate
            y: Mouse y coordinate
        """
        if not self.enabled or not self.active_control:
            return
        
        control = self.controls[self.active_control]
        
        if control['type'] == 'button':
            # Check if mouse is still over the button
            if self._is_point_in_rect(x, y, control['position'], control['size']):
                if control['callback']:
                    control['callback']()
                control['state'] = 'hover'
                self.hover_control = self.active_control
            else:
                control['state'] = 'normal'
        
        elif control['type'] == 'slider':
            control['state'] = 'normal'
            if self._is_point_in_rect(x, y, control['position'], control['size']):
                control['state'] = 'hover'
                self.hover_control = self.active_control
        
        self.active_control = None
    
    def handle_mouse_drag(self, x: int, y: int):
        """
        Handle mouse drag
        
        Args:
            x: Mouse x coordinate
            y: Mouse y coordinate
        """
        if not self.enabled or not self.active_control:
            return
        
        control = self.controls[self.active_control]
        
        if control['type'] == 'slider':
            self._update_slider_value(self.active_control, x)
    
    def _update_slider_value(self, name: str, x: int):
        """
        Update a slider value based on mouse position
        
        Args:
            name: Name of the slider
            x: Mouse x coordinate
        """
        control = self.controls[name]
        pos_x, _ = control['position']
        width, _ = control['size']
        min_val = control['min_value']
        max_val = control['max_value']
        
        # Calculate new value based on x position
        x_rel = max(0, min(x - pos_x, width))
        value = min_val + (max_val - min_val) * (x_rel / width)
        
        # Update value
        control['current_value'] = value
        
        # Call callback
        if control['callback']:
            control['callback'](value)
    
    def _is_point_in_rect(self, x: int, y: int, 
                         position: Tuple[int, int], 
                         size: Tuple[int, int]) -> bool:
        """
        Check if a point is inside a rectangle
        
        Args:
            x: Point x coordinate
            y: Point y coordinate
            position: Rectangle position (x, y)
            size: Rectangle size (width, height)
            
        Returns:
            True if the point is inside the rectangle, False otherwise
        """
        rect_x, rect_y = position
        rect_w, rect_h = size
        
        return (rect_x <= x <= rect_x + rect_w and 
                rect_y <= y <= rect_y + rect_h)
    
    def draw(self, frame: Any) -> Any:
        """
        Draw interactive controls on a frame
        
        Args:
            frame: Frame to draw on
            
        Returns:
            Frame with controls drawn on it
        """
        if not self.enabled or not OPENCV_AVAILABLE or frame is None:
            return frame
        
        # Create a copy of the frame
        display = frame.copy()
        
        # Draw each control
        for name, control in self.controls.items():
            if control['type'] == 'button':
                display = self._draw_button(display, control)
            elif control['type'] == 'slider':
                display = self._draw_slider(display, control)
            elif control['type'] == 'checkbox':
                display = self._draw_checkbox(display, control)
        
        return display
    
    def _draw_button(self, frame: Any, control: Dict) -> Any:
        """
        Draw a button control
        
        Args:
            frame: Frame to draw on
            control: Button control data
            
        Returns:
            Frame with button drawn on it
        """
        display = frame
        
        # Get button properties
        pos_x, pos_y = control['position']
        width, height = control['size']
        label = control['label']
        state = control['state']
        
        # Determine color based on state
        if state == 'normal':
            bg_color = (100, 100, 100)
            fg_color = (255, 255, 255)
        elif state == 'hover':
            bg_color = (150, 150, 150)
            fg_color = (255, 255, 255)
        else:  # pressed
            bg_color = (50, 50, 50)
            fg_color = (200, 200, 200)
        
        # Draw button background
        cv2.rectangle(
            display,
            (pos_x, pos_y),
            (pos_x + width, pos_y + height),
            bg_color,
            -1
        )
        
        # Draw button border
        cv2.rectangle(
            display,
            (pos_x, pos_y),
            (pos_x + width, pos_y + height),
            (0, 0, 0),
            1
        )
        
        # Draw button label
        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            1
        )[0]
        
        text_x = pos_x + (width - text_size[0]) // 2
        text_y = pos_y + (height + text_size[1]) // 2
        
        cv2.putText(
            display,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            fg_color,
            1
        )
        
        return display
    
    def _draw_slider(self, frame: Any, control: Dict) -> Any:
        """
        Draw a slider control
        
        Args:
            frame: Frame to draw on
            control: Slider control data
            
        Returns:
            Frame with slider drawn on it
        """
        display = frame
        
        # Get slider properties
        pos_x, pos_y = control['position']
        width, height = control['size']
        label = control['label']
        min_val = control['min_value']
        max_val = control['max_value']
        value = control['current_value']
        state = control['state']
        
        # Determine colors
        track_color = (100, 100, 100)
        thumb_color = (150, 150, 150) if state == 'normal' else (200, 200, 200)
        
        # Draw slider track
        cv2.rectangle(
            display,
            (pos_x, pos_y + height // 2 - 2),
            (pos_x + width, pos_y + height // 2 + 2),
            track_color,
            -1
        )
        
        # Calculate thumb position
        value_ratio = (value - min_val) / (max_val - min_val)
        thumb_x = int(pos_x + value_ratio * width)
        
        # Draw slider thumb
        cv2.circle(
            display,
            (thumb_x, pos_y + height // 2),
            height // 2,
            thumb_color,
            -1
        )
        
        # Draw slider border
        cv2.circle(
            display,
            (thumb_x, pos_y + height // 2),
            height // 2,
            (0, 0, 0),
            1
        )
        
        # Draw slider label
        cv2.putText(
            display,
            f"{label}: {value:.2f}",
            (pos_x, pos_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        return display
    
    def _draw_checkbox(self, frame: Any, control: Dict) -> Any:
        """
        Draw a checkbox control
        
        Args:
            frame: Frame to draw on
            control: Checkbox control data
            
        Returns:
            Frame with checkbox drawn on it
        """
        display = frame
        
        # Get checkbox properties
        pos_x, pos_y = control['position']
        width, height = control['size']
        label = control['label']
        checked = control['checked']
        state = control['state']
        
        # Determine color based on state
        if state == 'normal':
            border_color = (100, 100, 100)
        else:  # hover
            border_color = (150, 150, 150)
        
        # Draw checkbox border
        cv2.rectangle(
            display,
            (pos_x, pos_y),
            (pos_x + width, pos_y + height),
            border_color,
            1
        )
        
        # Draw checkbox fill if checked
        if checked:
            cv2.rectangle(
                display,
                (pos_x + 2, pos_y + 2),
                (pos_x + width - 2, pos_y + height - 2),
                (0, 200, 0),
                -1
            )
        
        # Draw checkbox label
        cv2.putText(
            display,
            label,
            (pos_x + width + 10, pos_y + height - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        return display


class RealTimeVisualizer:
    """Core visualization framework for real-time negative space analysis"""
    
    def __init__(self, mode: VisualizationMode = VisualizationMode.BASIC):
        """
        Initialize a real-time visualizer
        
        Args:
            mode: Visualization mode
        """
        self.mode = mode
        self.window_name = "Real-Time Negative Space Analysis"
        self.width = 1280
        self.height = 720
        
        # Initialize components
        self.performance_display = PerformanceDisplay()
        self.ar_overlay = AROverlay()
        self.controls = InteractiveControls()
        
        # Initialize visualization window if OpenCV is available
        if OPENCV_AVAILABLE:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)
            
            # Register mouse callback
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # For 3D visualization
        self.o3d_window = None
        self.o3d_vis = None
        self.o3d_thread = None
        self.o3d_running = False
        
        # Initialize 3D visualization if Open3D is available
        if mode in [VisualizationMode.BASIC_3D, VisualizationMode.ADVANCED_3D] and OPEN3D_AVAILABLE:
            self._init_3d_visualization()
        
        # Frame count and FPS calculation
        self.frame_count = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        # Add standard overlays
        self._setup_standard_overlays()
        
        # Add standard controls
        self._setup_standard_controls()
    
    def _mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback for interactive controls
        
        Args:
            event: Mouse event type
            x: Mouse x coordinate
            y: Mouse y coordinate
            flags: Event flags
            param: User data
        """
        if event == cv2.EVENT_MOUSEMOVE:
            self.controls.handle_mouse_move(x, y)
            
            # Check for dragging
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self.controls.handle_mouse_drag(x, y)
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.controls.handle_mouse_down(x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.controls.handle_mouse_up(x, y)
    
    def _init_3d_visualization(self):
        """Initialize 3D visualization with Open3D"""
        if not OPEN3D_AVAILABLE:
            logger.warning("Open3D is not available, cannot initialize 3D visualization")
            return
        
        try:
            # Initialize Open3D visualizer
            self.o3d_vis = o3d.visualization.Visualizer()
            self.o3d_vis.create_window(
                window_name="3D Negative Space Visualization",
                width=800,
                height=600
            )
            
            # Set rendering options
            render_option = self.o3d_vis.get_render_option()
            render_option.background_color = [0.1, 0.1, 0.1]
            render_option.point_size = 3.0
            
            # Start visualization thread
            self.o3d_running = True
            self.o3d_thread = threading.Thread(target=self._o3d_thread_func)
            self.o3d_thread.daemon = True
            self.o3d_thread.start()
            
        except Exception as e:
            logger.error(f"Error initializing 3D visualization: {e}")
            self.o3d_vis = None
    
    def _o3d_thread_func(self):
        """Thread function for Open3D visualization"""
        if not self.o3d_vis:
            return
        
        try:
            while self.o3d_running:
                # Update visualization
                self.o3d_vis.poll_events()
                self.o3d_vis.update_renderer()
                
                # Sleep to avoid high CPU usage
                time.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Error in Open3D visualization thread: {e}")
        
        finally:
            # Clean up
            if self.o3d_vis:
                self.o3d_vis.destroy_window()
            self.o3d_vis = None
    
    def _setup_standard_overlays(self):
        """Set up standard AR overlays"""
        self.ar_overlay.add_overlay("bounding_boxes", OverlayType.BOUNDING_BOX)
        self.ar_overlay.add_overlay("negative_spaces", OverlayType.NEGATIVE_SPACE)
        self.ar_overlay.add_overlay("confidence", OverlayType.CONFIDENCE)
        self.ar_overlay.add_overlay("signature", OverlayType.SIGNATURE)
    
    def _setup_standard_controls(self):
        """Set up standard interactive controls"""
        # Add mode selection buttons
        button_width = 150
        button_height = 30
        button_spacing = 10
        start_y = 20
        
        self.controls.add_button(
            "mode_basic",
            (self.width - button_width - 20, start_y),
            (button_width, button_height),
            "Basic Mode",
            lambda: self.set_mode(VisualizationMode.BASIC)
        )
        
        self.controls.add_button(
            "mode_advanced_2d",
            (self.width - button_width - 20, start_y + button_height + button_spacing),
            (button_width, button_height),
            "Advanced 2D",
            lambda: self.set_mode(VisualizationMode.ADVANCED_2D)
        )
        
        self.controls.add_button(
            "mode_basic_3d",
            (self.width - button_width - 20, start_y + 2 * (button_height + button_spacing)),
            (button_width, button_height),
            "Basic 3D",
            lambda: self.set_mode(VisualizationMode.BASIC_3D)
        )
        
        self.controls.add_button(
            "mode_advanced_3d",
            (self.width - button_width - 20, start_y + 3 * (button_height + button_spacing)),
            (button_width, button_height),
            "Advanced 3D",
            lambda: self.set_mode(VisualizationMode.ADVANCED_3D)
        )
        
        self.controls.add_button(
            "mode_ar",
            (self.width - button_width - 20, start_y + 4 * (button_height + button_spacing)),
            (button_width, button_height),
            "AR Mode",
            lambda: self.set_mode(VisualizationMode.AR)
        )
        
        # Add overlay toggles
        checkbox_size = 20
        checkbox_spacing = 30
        start_y = 200
        
        self.controls.add_checkbox(
            "overlay_bounding_boxes",
            (self.width - 200, start_y),
            (checkbox_size, checkbox_size),
            "Bounding Boxes",
            True,
            lambda checked: self._toggle_overlay("bounding_boxes", checked)
        )
        
        self.controls.add_checkbox(
            "overlay_negative_spaces",
            (self.width - 200, start_y + checkbox_spacing),
            (checkbox_size, checkbox_size),
            "Negative Spaces",
            True,
            lambda checked: self._toggle_overlay("negative_spaces", checked)
        )
        
        self.controls.add_checkbox(
            "overlay_confidence",
            (self.width - 200, start_y + 2 * checkbox_spacing),
            (checkbox_size, checkbox_size),
            "Confidence",
            True,
            lambda checked: self._toggle_overlay("confidence", checked)
        )
        
        self.controls.add_checkbox(
            "overlay_signature",
            (self.width - 200, start_y + 3 * checkbox_spacing),
            (checkbox_size, checkbox_size),
            "Signature",
            True,
            lambda checked: self._toggle_overlay("signature", checked)
        )
        
        # Add slider controls
        slider_width = 150
        slider_height = 20
        slider_spacing = 40
        start_y = 350
        
        self.controls.add_slider(
            "opacity",
            (self.width - slider_width - 20, start_y),
            (slider_width, slider_height),
            "Opacity",
            0.0,
            1.0,
            0.7,
            lambda value: self._set_overlay_opacity(value)
        )
    
    def _toggle_overlay(self, name: str, enabled: bool):
        """
        Toggle an overlay
        
        Args:
            name: Name of the overlay
            enabled: Whether the overlay should be enabled
        """
        if enabled:
            self.ar_overlay.enable_overlay(name)
        else:
            self.ar_overlay.disable_overlay(name)
    
    def _set_overlay_opacity(self, opacity: float):
        """
        Set overlay opacity
        
        Args:
            opacity: Opacity value (0.0 - 1.0)
        """
        self.ar_overlay.opacity = opacity
    
    def set_mode(self, mode: VisualizationMode):
        """
        Set the visualization mode
        
        Args:
            mode: New visualization mode
        """
        # Check if 3D visualization is available for 3D modes
        if mode in [VisualizationMode.BASIC_3D, VisualizationMode.ADVANCED_3D] and not OPEN3D_AVAILABLE:
            logger.warning("Open3D is not available, cannot switch to 3D mode")
            return
        
        self.mode = mode
        
        # Initialize 3D visualization if needed
        if mode in [VisualizationMode.BASIC_3D, VisualizationMode.ADVANCED_3D] and not self.o3d_vis:
            self._init_3d_visualization()
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """
        Update performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
        """
        for name, value in metrics.items():
            self.performance_display.update_metric(name, value)
    
    def update_overlay_data(self, name: str, data: Any):
        """
        Update overlay data
        
        Args:
            name: Name of the overlay
            data: New data for the overlay
        """
        self.ar_overlay.update_overlay_data(name, data)
    
    def update_3d_visualization(self, point_cloud: Any):
        """
        Update 3D visualization with a new point cloud
        
        Args:
            point_cloud: Point cloud to visualize
        """
        if not self.o3d_vis or not OPEN3D_AVAILABLE or self.mode not in [
            VisualizationMode.BASIC_3D, VisualizationMode.ADVANCED_3D
        ]:
            return
        
        try:
            # Convert to Open3D point cloud if needed
            from src.realtime.webcam_integration import convert_to_open3d
            o3d_cloud = convert_to_open3d(point_cloud)
            
            if o3d_cloud:
                # Clear existing geometry
                self.o3d_vis.clear_geometries()
                
                # Add new point cloud
                self.o3d_vis.add_geometry(o3d_cloud)
                
                # Reset view if needed
                self.o3d_vis.reset_view_point(True)
        
        except Exception as e:
            logger.error(f"Error updating 3D visualization: {e}")
    
    def visualize(self, frame: Any, point_cloud: Any) -> Any:
        """
        Visualize a frame and point cloud
        
        Args:
            frame: Video frame to visualize
            point_cloud: Point cloud to visualize
            
        Returns:
            Visualization frame
        """
        if not OPENCV_AVAILABLE or frame is None:
            return None
        
        # Create a copy of the frame
        display = frame.copy()
        
        # Update frame count and calculate FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
            self.performance_display.update_metric("FPS", fps)
        
        # Update performance metrics every second
        if current_time - self.last_update_time >= 1.0:
            self.last_update_time = current_time
            
            # Add more metrics here if needed
            if point_cloud and hasattr(point_cloud, 'void_count'):
                self.performance_display.update_metric("Void Count", point_cloud.void_count)
        
        # Apply AR overlays
        if self.mode in [VisualizationMode.ADVANCED_2D, VisualizationMode.AR]:
            display = self.ar_overlay.draw(display, point_cloud)
        
        # Draw performance metrics
        display = self.performance_display.draw(display)
        
        # Draw interactive controls
        display = self.controls.draw(display)
        
        # Update 3D visualization if in 3D mode
        if self.mode in [VisualizationMode.BASIC_3D, VisualizationMode.ADVANCED_3D]:
            self.update_3d_visualization(point_cloud)
        
        # Show result
        cv2.imshow(self.window_name, display)
        
        return display
    
    def close(self):
        """Close the visualizer and release resources"""
        # Close 3D visualization
        if self.o3d_vis:
            self.o3d_running = False
            if self.o3d_thread:
                self.o3d_thread.join(timeout=2.0)
            self.o3d_vis = None
        
        # Close OpenCV windows
        if OPENCV_AVAILABLE:
            cv2.destroyAllWindows()
    
    def save_screenshot(self, filename: str) -> bool:
        """
        Save a screenshot of the current visualization
        
        Args:
            filename: Filename to save the screenshot to
            
        Returns:
            True if successful, False otherwise
        """
        if not OPENCV_AVAILABLE:
            return False
        
        try:
            # Get the current window content
            screenshot = None
            for window in range(10):  # Try different windows
                screenshot = cv2.getWindowImage(self.window_name)
                if screenshot is not None:
                    break
            
            if screenshot is None:
                logger.error("Could not capture screenshot")
                return False
            
            # Save to file
            cv2.imwrite(filename, screenshot)
            logger.info(f"Screenshot saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            return False
