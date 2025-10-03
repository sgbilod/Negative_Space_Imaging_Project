"""
Webcam Integration Module

This module provides functionality for capturing and processing
video streams from webcams or other camera sources for real-time
negative space analysis.

Classes:
    CameraSource: Interface to camera hardware
    DepthEstimator: Estimates depth from 2D images
    PointCloudGenerator: Converts video frames to point clouds
"""

import os
import sys
import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from enum import Enum
from queue import Queue

# Import from centralized fallbacks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.fallbacks import (
    np, cv2, NUMPY_AVAILABLE, OPENCV_AVAILABLE, OPEN3D_AVAILABLE
)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CameraResolution(Enum):
    """Standard camera resolutions"""
    LOW = (320, 240)
    MEDIUM = (640, 480)
    HIGH = (1280, 720)
    FULL_HD = (1920, 1080)
    UHD = (3840, 2160)


class CameraSource:
    """Interface to camera hardware for capturing video frames"""
    
    def __init__(self, camera_id: int = 0, 
                 resolution: CameraResolution = CameraResolution.MEDIUM):
        """
        Initialize a camera source
        
        Args:
            camera_id: ID of the camera to use
            resolution: Resolution to capture at
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.camera = None
        self.is_running = False
        self.frame_buffer = Queue(maxsize=30)
        self.capture_thread = None
        
        # Check if OpenCV is available
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV is required for camera capture")
            raise ImportError("OpenCV is required for camera capture")
    
    def start(self):
        """Start capturing frames from the camera"""
        if self.is_running:
            logger.warning("Camera is already running")
            return False
        
        # Initialize camera
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            logger.error(f"Could not open camera {self.camera_id}")
            return False
        
        # Set resolution
        width, height = self.resolution.value
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Start capture thread
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        logger.info(f"Camera {self.camera_id} started at resolution {width}x{height}")
        return True
    
    def stop(self):
        """Stop capturing frames from the camera"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
            self.capture_thread = None
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        logger.info(f"Camera {self.camera_id} stopped")
    
    def _capture_loop(self):
        """Capture loop for the camera thread"""
        while self.is_running and self.camera and self.camera.isOpened():
            try:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to capture frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Add to buffer, removing oldest if full
                if self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()
                    except:
                        pass
                
                self.frame_buffer.put(frame)
                
            except Exception as e:
                logger.error(f"Error in camera capture loop: {e}")
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[Any]:
        """
        Get the latest frame from the camera
        
        Returns:
            The latest frame, or None if no frames are available
        """
        if not self.is_running or self.frame_buffer.empty():
            return None
        
        return self.frame_buffer.get()
    
    def get_camera_properties(self) -> Dict:
        """
        Get properties of the camera
        
        Returns:
            Dict of camera properties
        """
        properties = {}
        
        if self.camera and self.camera.isOpened():
            properties['width'] = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            properties['height'] = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            properties['fps'] = self.camera.get(cv2.CAP_PROP_FPS)
            properties['brightness'] = self.camera.get(cv2.CAP_PROP_BRIGHTNESS)
            properties['contrast'] = self.camera.get(cv2.CAP_PROP_CONTRAST)
            properties['saturation'] = self.camera.get(cv2.CAP_PROP_SATURATION)
            properties['hue'] = self.camera.get(cv2.CAP_PROP_HUE)
            properties['gain'] = self.camera.get(cv2.CAP_PROP_GAIN)
            properties['exposure'] = self.camera.get(cv2.CAP_PROP_EXPOSURE)
        
        return properties
    
    def set_camera_property(self, property_id: int, value: float) -> bool:
        """
        Set a camera property
        
        Args:
            property_id: OpenCV property ID
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        if not self.camera or not self.camera.isOpened():
            return False
        
        return self.camera.set(property_id, value)
    
    def get_available_cameras(self) -> List[int]:
        """
        Get a list of available cameras
        
        Returns:
            List of camera IDs
        """
        available_cameras = []
        
        # Try to open cameras with IDs 0-9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        return available_cameras


class DepthEstimator:
    """Estimates depth from 2D images"""
    
    def __init__(self, method: str = "stereo"):
        """
        Initialize a depth estimator
        
        Args:
            method: Method to use for depth estimation ("stereo", "mono", or "structured_light")
        """
        self.method = method
        self.initialized = False
        
        # Check if OpenCV is available
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV is required for depth estimation")
            return
        
        if method == "stereo":
            # Initialize stereo matching
            self.stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
            self.initialized = True
            
        elif method == "mono":
            # In a real implementation, this would initialize a neural network
            # for monocular depth estimation
            logger.warning("Monocular depth estimation is not fully implemented")
            self.initialized = False
            
        elif method == "structured_light":
            # In a real implementation, this would initialize structured light
            # depth estimation
            logger.warning("Structured light depth estimation is not fully implemented")
            self.initialized = False
        
        else:
            logger.error(f"Unknown depth estimation method: {method}")
            return
    
    def estimate_depth(self, left_frame, right_frame=None) -> Optional[Any]:
        """
        Estimate depth from images
        
        Args:
            left_frame: Left camera frame (or single frame for mono)
            right_frame: Right camera frame (only for stereo)
            
        Returns:
            Depth map, or None if depth estimation failed
        """
        if not self.initialized or not OPENCV_AVAILABLE:
            return None
        
        if self.method == "stereo":
            if right_frame is None:
                logger.error("Right frame is required for stereo depth estimation")
                return None
            
            # Convert to grayscale
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            
            # Compute disparity
            disparity = self.stereo.compute(left_gray, right_gray)
            
            # Convert disparity to depth (simplified)
            # In a real implementation, this would use camera calibration parameters
            depth = np.float32(disparity) / 16.0
            
            return depth
            
        elif self.method == "mono":
            # In a real implementation, this would use a neural network
            # for monocular depth estimation
            
            # For now, return a dummy depth map
            if NUMPY_AVAILABLE:
                return np.zeros(left_frame.shape[:2], dtype=np.float32)
            
            return None
            
        elif self.method == "structured_light":
            # In a real implementation, this would use structured light
            # depth estimation
            
            # For now, return a dummy depth map
            if NUMPY_AVAILABLE:
                return np.zeros(left_frame.shape[:2], dtype=np.float32)
            
            return None
        
        return None
    
    def visualize_depth(self, depth_map) -> Optional[Any]:
        """
        Visualize a depth map
        
        Args:
            depth_map: Depth map to visualize
            
        Returns:
            Visualization of the depth map, or None if visualization failed
        """
        if not OPENCV_AVAILABLE or depth_map is None:
            return None
        
        # Normalize depth map for visualization
        min_val = np.min(depth_map)
        max_val = np.max(depth_map)
        if max_val > min_val:
            normalized = (depth_map - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(depth_map)
        
        # Convert to color map
        visualization = cv2.applyColorMap((normalized * 255).astype(np.uint8), 
                                          cv2.COLORMAP_JET)
        
        return visualization


class PointCloudGenerator:
    """Converts video frames to point clouds"""
    
    def __init__(self, depth_estimator: DepthEstimator = None, 
                 downsample_factor: float = 0.1):
        """
        Initialize a point cloud generator
        
        Args:
            depth_estimator: Depth estimator to use
            downsample_factor: Factor to downsample frames by
        """
        self.depth_estimator = depth_estimator
        self.downsample_factor = downsample_factor
        
        # Check if NumPy is available
        if not NUMPY_AVAILABLE:
            logger.error("NumPy is required for point cloud generation")
            return
    
    def generate_point_cloud(self, frame, depth_map=None) -> Optional[Any]:
        """
        Generate a point cloud from a frame and optional depth map
        
        Args:
            frame: Frame to generate point cloud from
            depth_map: Optional depth map (will be estimated if not provided)
            
        Returns:
            Point cloud object, or None if generation failed
        """
        # Import simplified demo classes
        sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
        try:
            from simplified_demo import SimplePointCloud
        except ImportError:
            logger.error("Could not import SimplePointCloud from simplified_demo")
            return None
        
        # Create a point cloud
        point_cloud = SimplePointCloud()
        
        # Check if we have the required dependencies
        if not NUMPY_AVAILABLE or not OPENCV_AVAILABLE:
            return point_cloud
        
        # If no depth map is provided and we have a depth estimator, estimate depth
        if depth_map is None and self.depth_estimator:
            depth_map = self.depth_estimator.estimate_depth(frame)
        
        # If we have a depth map, use it to generate a point cloud
        if depth_map is not None:
            # Downsample for performance
            h, w = frame.shape[:2]
            ds_h, ds_w = int(h * self.downsample_factor), int(w * self.downsample_factor)
            
            # Resize frame and depth map
            resized_frame = cv2.resize(frame, (ds_w, ds_h))
            resized_depth = cv2.resize(depth_map, (ds_w, ds_h))
            
            # Generate points from depth
            points = []
            colors = []
            
            for y in range(ds_h):
                for x in range(ds_w):
                    # Get depth value
                    z = resized_depth[y, x]
                    
                    # Skip invalid depth values
                    if z <= 0 or np.isinf(z) or np.isnan(z):
                        continue
                    
                    # Normalized coordinates
                    nx = (x / ds_w) - 0.5
                    ny = (y / ds_h) - 0.5
                    nz = z / 255.0  # Normalize depth
                    
                    points.append([nx, ny, nz])
                    
                    # Get color
                    if len(resized_frame.shape) == 3:
                        b = resized_frame[y, x, 0] / 255.0
                        g = resized_frame[y, x, 1] / 255.0
                        r = resized_frame[y, x, 2] / 255.0
                        colors.append([r, g, b])
                    else:
                        v = resized_frame[y, x] / 255.0
                        colors.append([v, v, v])
            
            # Add points to point cloud
            if points:
                point_cloud.add_points(np.array(points), np.array(colors))
        
        else:
            # No depth map available, use intensity-based pseudo-depth
            h, w = frame.shape[:2]
            
            # Convert to grayscale if needed
            gray = frame
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Downsample for performance
            ds_h, ds_w = int(h * self.downsample_factor), int(w * self.downsample_factor)
            downsampled = cv2.resize(gray, (ds_w, ds_h))
            
            # Create points based on intensity
            points = []
            colors = []
            
            for y in range(ds_h):
                for x in range(ds_w):
                    if downsampled[y, x] > 50:  # Only use non-dark pixels
                        # Project 2D to 3D (placeholder using intensity as depth)
                        nx = (x / ds_w) - 0.5
                        ny = (y / ds_h) - 0.5
                        nz = (downsampled[y, x] / 255.0) - 0.5
                        
                        points.append([nx, ny, nz])
                        
                        # Use RGB from original frame if available
                        if len(frame.shape) == 3:
                            orig_y, orig_x = int(y / self.downsample_factor), int(x / self.downsample_factor)
                            b = frame[orig_y, orig_x, 0] / 255.0
                            g = frame[orig_y, orig_x, 1] / 255.0
                            r = frame[orig_y, orig_x, 2] / 255.0
                            colors.append([r, g, b])
                        else:
                            colors.append([0.5, 0.5, 0.5])
            
            # Add points to point cloud
            if points:
                point_cloud.add_points(np.array(points), np.array(colors))
        
        # Analyze the point cloud
        point_cloud.classify_points()
        point_cloud.generate_void_points()
        
        return point_cloud


# If Open3D is available, provide a converter to Open3D point clouds
if OPEN3D_AVAILABLE:
    def convert_to_open3d(point_cloud) -> Any:
        """
        Convert a SimplePointCloud to an Open3D point cloud
        
        Args:
            point_cloud: SimplePointCloud object
            
        Returns:
            Open3D point cloud
        """
        try:
            # Only attempt to import if Open3D is available
            from utils.fallbacks import o3d
            
            # Create Open3D point cloud
            o3d_cloud = o3d.geometry.PointCloud()
            
            # Add points
            if hasattr(point_cloud, 'points') and len(point_cloud.points) > 0:
                o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud.points)
            
            # Add colors
            if hasattr(point_cloud, 'colors') and len(point_cloud.colors) > 0:
                o3d_cloud.colors = o3d.utility.Vector3dVector(point_cloud.colors)
            
            return o3d_cloud
            
        except Exception as e:
            logger.error(f"Error converting to Open3D point cloud: {e}")
            return None
else:
    def convert_to_open3d(point_cloud) -> None:
        """
        Dummy converter when Open3D is not available
        
        Args:
            point_cloud: SimplePointCloud object
            
        Returns:
            None
        """
        logger.warning("Open3D is not available, cannot convert point cloud")
        return None
