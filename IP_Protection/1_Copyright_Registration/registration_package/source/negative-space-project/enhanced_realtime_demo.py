"""
Enhanced Real-Time Negative Space Analysis Demo

This script demonstrates the advanced visualization capabilities of the
real-time negative space analysis system, including AR overlays,
interactive controls, and 3D visualization.

Usage:
    python enhanced_realtime_demo.py [--mode {webcam|video|synthetic}] 
                                    [--viz {basic|advanced_2d|basic_3d|advanced_3d|ar}]
                                    [--video PATH_TO_VIDEO]

Key Features:
    - Advanced 2D and 3D visualizations
    - Interactive controls for mode selection
    - AR overlays for negative space visualization
    - Real-time performance metrics
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from enum import Enum

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir))
sys.path.append(project_root)

# Import from centralized fallbacks
from src.utils.fallbacks import (
    np, cv2, plt, NUMPY_AVAILABLE, OPENCV_AVAILABLE, 
    MATPLOTLIB_AVAILABLE, OPEN3D_AVAILABLE
)

# Import real-time tracking modules
from src.realtime.real_time_tracker import RealTimeTracker, AnalysisMode
from src.realtime.webcam_integration import (
    CameraSource, PointCloudGenerator, DepthEstimator, CameraResolution
)
from src.realtime.visualization import (
    RealTimeVisualizer, VisualizationMode, OverlayType, ColorPalette
)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for required dependencies
DEPENDENCIES_MET = NUMPY_AVAILABLE and OPENCV_AVAILABLE
if not DEPENDENCIES_MET:
    logger.warning("Some required dependencies are missing. Functionality may be limited.")
    missing = []
    if not NUMPY_AVAILABLE:
        missing.append("NumPy")
    if not OPENCV_AVAILABLE:
        missing.append("OpenCV")
    logger.warning(f"Missing dependencies: {', '.join(missing)}")


class DemoMode(Enum):
    """Demo modes for enhanced real-time tracking"""
    WEBCAM = 1       # Use webcam input
    VIDEO_FILE = 2   # Use pre-recorded video
    SYNTHETIC = 3    # Generate synthetic data


class EnhancedRealTimeDemo:
    """Enhanced real-time negative space analysis demo"""
    
    def __init__(self, mode: DemoMode = DemoMode.WEBCAM, 
                 viz_mode: VisualizationMode = VisualizationMode.ADVANCED_2D,
                 analysis_mode: AnalysisMode = AnalysisMode.CONTINUOUS,
                 video_path: str = None,
                 camera_id: int = 0,
                 resolution: CameraResolution = CameraResolution.MEDIUM):
        """
        Initialize the enhanced real-time demo
        
        Args:
            mode: Demo mode (webcam, video file, or synthetic)
            viz_mode: Visualization mode
            analysis_mode: Analysis mode for the tracker
            video_path: Path to video file (for VIDEO_FILE mode)
            camera_id: Camera ID (for WEBCAM mode)
            resolution: Camera resolution (for WEBCAM mode)
        """
        self.mode = mode
        self.viz_mode = viz_mode
        self.analysis_mode = analysis_mode
        self.video_path = video_path
        self.camera_id = camera_id
        self.resolution = resolution
        
        # Check dependencies
        if not DEPENDENCIES_MET:
            logger.warning("Missing dependencies. Some features may not work.")
        
        # Initialize components
        self.init_components()
    
    def init_components(self):
        """Initialize demo components"""
        # Create tracker
        self.tracker = RealTimeTracker(target_fps=30.0, mode=self.analysis_mode)
        
        # Create visualizer
        self.visualizer = RealTimeVisualizer(mode=self.viz_mode)
        
        # Create camera source if needed
        self.camera = None
        if self.mode == DemoMode.WEBCAM:
            try:
                self.camera = CameraSource(camera_id=self.camera_id, 
                                          resolution=self.resolution)
            except ImportError:
                logger.error("Could not initialize camera. OpenCV is required.")
                self.mode = DemoMode.SYNTHETIC
        
        # Create video capture if needed
        self.video_capture = None
        if self.mode == DemoMode.VIDEO_FILE:
            if not os.path.exists(self.video_path):
                logger.error(f"Video file not found: {self.video_path}")
                self.mode = DemoMode.SYNTHETIC
            else:
                try:
                    self.video_capture = cv2.VideoCapture(self.video_path)
                    if not self.video_capture.isOpened():
                        logger.error(f"Could not open video file: {self.video_path}")
                        self.mode = DemoMode.SYNTHETIC
                except:
                    logger.error("Could not initialize video capture")
                    self.mode = DemoMode.SYNTHETIC
        
        # Create point cloud generator
        self.point_cloud_generator = PointCloudGenerator(
            depth_estimator=DepthEstimator("mono"), 
            downsample_factor=0.1
        )
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        self.last_metrics_time = self.start_time
        
        # Initialize dummy overlay data
        self._setup_initial_overlays()
    
    def _setup_initial_overlays(self):
        """Set up initial overlay data"""
        # Bounding boxes
        self.visualizer.update_overlay_data("bounding_boxes", [
            {
                'coords': (100, 100, 300, 300),
                'label': 'Object 1',
                'confidence': 0.95
            },
            {
                'coords': (400, 150, 550, 350),
                'label': 'Object 2',
                'confidence': 0.87
            }
        ])
        
        # Negative spaces
        self.visualizer.update_overlay_data("negative_spaces", [
            {
                'center': (0.3, 0.3),
                'radius': 40,
                'importance': 0.8
            },
            {
                'center': (0.6, 0.5),
                'radius': 30,
                'importance': 0.6
            }
        ])
        
        # Confidence
        self.visualizer.update_overlay_data("confidence", {
            'overall': 0.89,
            'details': {
                'Object Recognition': 0.95,
                'Negative Space': 0.85,
                'Spatial Signature': 0.88
            }
        })
        
        # Signature
        self.visualizer.update_overlay_data("signature", {
            'values': [0.2, 0.4, 0.6, 0.3, 0.8, 0.5, 0.7, 0.1, 0.9, 0.3, 0.5, 0.7],
            'label': 'Spatial Signature'
        })
    
    def start(self):
        """Start the demo"""
        logger.info(f"Starting enhanced demo in {self.mode.name} mode with {self.viz_mode.name} visualization")
        
        # Start tracker
        self.tracker.start()
        
        # Start camera if needed
        if self.mode == DemoMode.WEBCAM and self.camera:
            self.camera.start()
        
        # Run the main loop
        try:
            self.run_loop()
        except KeyboardInterrupt:
            logger.info("Demo stopped by user")
        finally:
            self.cleanup()
    
    def run_loop(self):
        """Main processing loop"""
        self.running = True
        
        while self.running:
            # Get frame
            frame = self.get_frame()
            if frame is None:
                logger.info("No more frames available")
                break
            
            # Generate point cloud
            point_cloud = self.point_cloud_generator.generate_point_cloud(frame)
            
            # Process with tracker
            if point_cloud:
                self.tracker.process_frame(point_cloud)
            
            # Update metrics
            self._update_metrics()
            
            # Update overlay data
            self._update_overlays(point_cloud)
            
            # Visualize
            self.visualizer.visualize(frame, point_cloud)
            
            # Check for key press to exit
            if OPENCV_AVAILABLE and cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Demo stopped by user (key press)")
                break
            
            # Limit frame rate
            time.sleep(1.0 / 30.0)
    
    def _update_metrics(self):
        """Update performance metrics"""
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
            
            # Update visualizer metrics
            metrics = {
                "FPS": fps,
                "Frame Count": self.frame_count
            }
            
            # Add tracker metrics
            tracker_metrics = self.tracker.get_performance_metrics()
            for key, value in tracker_metrics.items():
                metrics[key] = value
            
            self.visualizer.update_performance_metrics(metrics)
            
            # Print metrics every 5 seconds
            if current_time - self.last_metrics_time > 5.0:
                logger.info(f"FPS: {fps:.1f}, Frames: {self.frame_count}")
                self.last_metrics_time = current_time
    
    def _update_overlays(self, point_cloud):
        """
        Update overlay data based on point cloud
        
        Args:
            point_cloud: Current point cloud
        """
        if not point_cloud:
            return
        
        # Update negative spaces overlay
        if hasattr(point_cloud, 'void_points') and NUMPY_AVAILABLE:
            # Calculate simplified negative spaces from void points
            void_points = getattr(point_cloud, 'void_points', [])
            if len(void_points) > 0:
                # Simple clustering (in a real implementation, this would be more sophisticated)
                max_clusters = 5
                centers = []
                radii = []
                
                if isinstance(void_points, np.ndarray) and void_points.shape[0] > 0:
                    # Take a sample of void points for efficiency
                    sample_size = min(100, void_points.shape[0])
                    indices = np.random.choice(void_points.shape[0], sample_size, replace=False)
                    samples = void_points[indices]
                    
                    # Simple clustering (just for visualization)
                    for i in range(min(max_clusters, sample_size)):
                        center = samples[i, :2]  # Use only x,y for simplicity
                        
                        # Find closest points
                        dists = np.sqrt(np.sum((samples[:, :2] - center) ** 2, axis=1))
                        closest_idx = np.argsort(dists)[:10]
                        radius = np.mean(dists[closest_idx]) * 50  # Scale for visualization
                        
                        centers.append(center)
                        radii.append(radius)
                
                # Create negative spaces data
                spaces = []
                for i, (center, radius) in enumerate(zip(centers, radii)):
                    spaces.append({
                        'center': center,
                        'radius': max(20, min(100, radius)),  # Clamp radius
                        'importance': 0.5 + 0.5 * (radius / 100)  # Importance based on radius
                    })
                
                self.visualizer.update_overlay_data("negative_spaces", spaces)
        
        # Update signature overlay
        if hasattr(point_cloud, 'compute_spatial_signature'):
            try:
                signature = point_cloud.compute_spatial_signature()
                if signature is not None:
                    self.visualizer.update_overlay_data("signature", {
                        'values': signature,
                        'label': 'Spatial Signature'
                    })
            except:
                pass
    
    def get_frame(self):
        """
        Get a frame from the current source
        
        Returns:
            Frame from current source, or None if no frames are available
        """
        if self.mode == DemoMode.WEBCAM and self.camera:
            return self.camera.get_frame()
            
        elif self.mode == DemoMode.VIDEO_FILE and self.video_capture:
            ret, frame = self.video_capture.read()
            if not ret:
                return None
            return frame
            
        elif self.mode == DemoMode.SYNTHETIC:
            # Generate a synthetic frame
            if NUMPY_AVAILABLE and OPENCV_AVAILABLE:
                # Create a blank frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Draw a rotating cube
                t = time.time()
                center_x, center_y = 320, 240
                size = 100
                
                # Cube vertices in 3D
                vertices = np.array([
                    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
                ])
                
                # Rotate vertices
                angle_x = t * 0.5
                angle_y = t * 0.7
                angle_z = t * 0.3
                
                # Rotation matrices
                rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)]
                ])
                
                ry = np.array([
                    [np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]
                ])
                
                rz = np.array([
                    [np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1]
                ])
                
                # Apply rotations
                vertices = vertices @ rx @ ry @ rz
                
                # Project to 2D
                vertices_2d = []
                for v in vertices:
                    x = int(v[0] * size + center_x)
                    y = int(v[1] * size + center_y)
                    vertices_2d.append((x, y))
                
                # Draw edges
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)
                ]
                
                for i, j in edges:
                    cv2.line(frame, vertices_2d[i], vertices_2d[j], (255, 255, 255), 2)
                
                # Add second cube for interstitial space demonstration
                center_x2, center_y2 = 420, 340
                size2 = 80
                
                vertices2_2d = []
                for v in vertices:
                    x = int(v[0] * size2 + center_x2)
                    y = int(v[1] * size2 + center_y2)
                    vertices2_2d.append((x, y))
                
                for i, j in edges:
                    cv2.line(frame, vertices2_2d[i], vertices2_2d[j], (200, 200, 0), 2)
                
                return frame
            
            else:
                logger.error("NumPy and OpenCV are required for synthetic frames")
                self.running = False
                return None
        
        return None
    
    def cleanup(self):
        """Clean up resources"""
        # Stop tracker
        if self.tracker:
            self.tracker.stop()
        
        # Stop camera
        if self.mode == DemoMode.WEBCAM and self.camera:
            self.camera.stop()
        
        # Release video capture
        if self.mode == DemoMode.VIDEO_FILE and self.video_capture:
            self.video_capture.release()
        
        # Close visualizer
        if self.visualizer:
            self.visualizer.close()


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Real-Time Negative Space Analysis Demo')
    
    parser.add_argument('--mode', choices=['webcam', 'video', 'synthetic'], 
                        default='synthetic', help='Demo mode')
    
    parser.add_argument('--viz', choices=['basic', 'advanced_2d', 'basic_3d', 'advanced_3d', 'ar'], 
                        default='advanced_2d', help='Visualization mode')
    
    parser.add_argument('--analysis', choices=['continuous', 'interval', 'adaptive', 'trigger'], 
                        default='continuous', help='Analysis mode')
    
    parser.add_argument('--camera', type=int, default=0, 
                        help='Camera ID for webcam mode')
    
    parser.add_argument('--video', type=str, default=None, 
                        help='Path to video file for video mode')
    
    parser.add_argument('--resolution', choices=['low', 'medium', 'high', 'fhd', 'uhd'], 
                        default='medium', help='Camera resolution for webcam mode')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set demo mode
    if args.mode == 'webcam':
        demo_mode = DemoMode.WEBCAM
    elif args.mode == 'video':
        demo_mode = DemoMode.VIDEO_FILE
    else:
        demo_mode = DemoMode.SYNTHETIC
    
    # Set visualization mode
    if args.viz == 'basic':
        viz_mode = VisualizationMode.BASIC
    elif args.viz == 'advanced_2d':
        viz_mode = VisualizationMode.ADVANCED_2D
    elif args.viz == 'basic_3d':
        viz_mode = VisualizationMode.BASIC_3D
    elif args.viz == 'advanced_3d':
        viz_mode = VisualizationMode.ADVANCED_3D
    else:
        viz_mode = VisualizationMode.AR
    
    # Set analysis mode
    if args.analysis == 'continuous':
        analysis_mode = AnalysisMode.CONTINUOUS
    elif args.analysis == 'interval':
        analysis_mode = AnalysisMode.INTERVAL
    elif args.analysis == 'adaptive':
        analysis_mode = AnalysisMode.ADAPTIVE
    else:
        analysis_mode = AnalysisMode.TRIGGER
    
    # Set resolution
    if args.resolution == 'low':
        resolution = CameraResolution.LOW
    elif args.resolution == 'medium':
        resolution = CameraResolution.MEDIUM
    elif args.resolution == 'high':
        resolution = CameraResolution.HIGH
    elif args.resolution == 'fhd':
        resolution = CameraResolution.FULL_HD
    else:
        resolution = CameraResolution.UHD
    
    # Create and start demo
    demo = EnhancedRealTimeDemo(
        mode=demo_mode,
        viz_mode=viz_mode,
        analysis_mode=analysis_mode,
        video_path=args.video,
        camera_id=args.camera,
        resolution=resolution
    )
    
    demo.start()


if __name__ == '__main__':
    main()
