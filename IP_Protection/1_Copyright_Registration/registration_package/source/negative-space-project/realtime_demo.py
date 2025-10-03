"""
Real-Time Negative Space Analysis Demo

This script demonstrates the real-time negative space analysis capabilities
using webcam input or pre-recorded video.

Usage:
    python realtime_demo.py [--video PATH_TO_VIDEO] [--mode {continuous|interval|adaptive|trigger}]

Key Features:
    - Real-time capture from webcam
    - Point cloud generation from video frames
    - Negative space detection and tracking
    - Performance metrics display
    - Multiple analysis modes
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, List, Optional
from pathlib import Path
from enum import Enum

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir))
sys.path.append(project_root)

# Import from centralized fallbacks
from src.utils.fallbacks import (
    np, cv2, plt, NUMPY_AVAILABLE, OPENCV_AVAILABLE, MATPLOTLIB_AVAILABLE
)

# Import real-time tracking modules
from src.realtime.real_time_tracker import RealTimeTracker, AnalysisMode
from src.realtime.webcam_integration import (
    CameraSource, PointCloudGenerator, DepthEstimator, CameraResolution
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
    """Demo modes for real-time tracking"""
    WEBCAM = 1       # Use webcam input
    VIDEO_FILE = 2   # Use pre-recorded video
    SYNTHETIC = 3    # Generate synthetic data


class RealTimeDemo:
    """Real-time negative space analysis demo"""
    
    def __init__(self, mode: DemoMode = DemoMode.WEBCAM, 
                 analysis_mode: AnalysisMode = AnalysisMode.CONTINUOUS,
                 video_path: str = None,
                 camera_id: int = 0,
                 resolution: CameraResolution = CameraResolution.MEDIUM,
                 display_results: bool = True):
        """
        Initialize the real-time demo
        
        Args:
            mode: Demo mode (webcam, video file, or synthetic)
            analysis_mode: Analysis mode for the tracker
            video_path: Path to video file (for VIDEO_FILE mode)
            camera_id: Camera ID (for WEBCAM mode)
            resolution: Camera resolution (for WEBCAM mode)
            display_results: Whether to display results in a window
        """
        self.mode = mode
        self.analysis_mode = analysis_mode
        self.video_path = video_path
        self.camera_id = camera_id
        self.resolution = resolution
        self.display_results = display_results
        
        # Check dependencies
        if not DEPENDENCIES_MET:
            logger.warning("Missing dependencies. Some features may not work.")
        
        # Initialize components
        self.init_components()
    
    def init_components(self):
        """Initialize demo components"""
        # Create tracker
        self.tracker = RealTimeTracker(target_fps=30.0, mode=self.analysis_mode)
        
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
        
        # Initialize visualization window if needed
        self.viz_window = None
        if self.display_results and OPENCV_AVAILABLE:
            cv2.namedWindow('Real-Time Negative Space Analysis', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Real-Time Negative Space Analysis', 800, 600)
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        self.last_metrics_time = self.start_time
    
    def start(self):
        """Start the demo"""
        logger.info(f"Starting demo in {self.mode.name} mode")
        
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
            
            # Display results
            if self.display_results:
                self.display_frame(frame, point_cloud)
            
            # Update metrics
            self.frame_count += 1
            
            # Print metrics every 5 seconds
            if time.time() - self.last_metrics_time > 5.0:
                self.print_metrics()
                self.last_metrics_time = time.time()
            
            # Check for key press to exit
            if OPENCV_AVAILABLE and cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Demo stopped by user (key press)")
                break
            
            # Limit frame rate
            time.sleep(1.0 / 30.0)
    
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
                
                return frame
            
            else:
                logger.error("NumPy and OpenCV are required for synthetic frames")
                self.running = False
                return None
        
        return None
    
    def display_frame(self, frame, point_cloud):
        """
        Display the current frame and analysis results
        
        Args:
            frame: Current video frame
            point_cloud: Generated point cloud
        """
        if not self.display_results or not OPENCV_AVAILABLE:
            return
        
        # Create display frame
        display = frame.copy()
        
        # Add performance metrics
        fps = self.frame_count / max(1, time.time() - self.start_time)
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if point_cloud and hasattr(point_cloud, 'void_count'):
            cv2.putText(display, f"Void Count: {point_cloud.void_count}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Get performance metrics from tracker
        metrics = self.tracker.get_performance_metrics()
        if 'average_fps' in metrics:
            cv2.putText(display, f"Tracker FPS: {metrics['average_fps']:.1f}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show demo mode
        cv2.putText(display, f"Mode: {self.mode.name}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show analysis mode
        cv2.putText(display, f"Analysis: {self.analysis_mode.name}", (10, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Real-Time Negative Space Analysis', display)
    
    def print_metrics(self):
        """Print performance metrics"""
        # Calculate FPS
        fps = self.frame_count / max(1, time.time() - self.start_time)
        
        # Get tracker metrics
        metrics = self.tracker.get_performance_metrics()
        
        logger.info(f"FPS: {fps:.1f}")
        logger.info(f"Frames processed: {self.frame_count}")
        logger.info(f"Tracker metrics: {metrics}")
    
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
        
        # Close windows
        if self.display_results and OPENCV_AVAILABLE:
            cv2.destroyAllWindows()


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Real-Time Negative Space Analysis Demo')
    
    parser.add_argument('--mode', choices=['webcam', 'video', 'synthetic'], 
                        default='webcam', help='Demo mode')
    
    parser.add_argument('--analysis', choices=['continuous', 'interval', 'adaptive', 'trigger'], 
                        default='continuous', help='Analysis mode')
    
    parser.add_argument('--camera', type=int, default=0, 
                        help='Camera ID for webcam mode')
    
    parser.add_argument('--video', type=str, default=None, 
                        help='Path to video file for video mode')
    
    parser.add_argument('--resolution', choices=['low', 'medium', 'high', 'fhd', 'uhd'], 
                        default='medium', help='Camera resolution for webcam mode')
    
    parser.add_argument('--no-display', action='store_true', 
                        help='Disable result display')
    
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
    demo = RealTimeDemo(
        mode=demo_mode,
        analysis_mode=analysis_mode,
        video_path=args.video,
        camera_id=args.camera,
        resolution=resolution,
        display_results=not args.no_display
    )
    
    demo.start()


if __name__ == '__main__':
    main()
