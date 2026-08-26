"""
Camera Interface Module for Negative Space Imaging Project

This module provides interfaces for connecting to and controlling various camera types,
optimized for capturing images suitable for negative space analysis.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json

class CameraInterface:
    """
    Handles connection and communication with various camera types.
    Supports standard webcams, DSLR cameras via gphoto2, and specialized
    depth cameras like Intel RealSense.
    """
    
    def __init__(self, camera_type: str = "webcam", camera_id: int = 0, 
                 config: Optional[Dict] = None):
        """
        Initialize camera interface with specific camera type and settings.
        
        Args:
            camera_type: Type of camera ("webcam", "dslr", "realsense", etc.)
            camera_id: ID or index of the camera
            config: Additional configuration parameters
        """
        self.camera_type = camera_type
        self.camera_id = camera_id
        self.config = config or {}
        self.camera = None
        self.connected = False
        self.calibration_data = None
        
        # Extended properties for advanced camera control
        self.supported_resolutions = []
        self.current_resolution = None
        self.supported_features = {}
        
    def connect(self) -> bool:
        """
        Establish connection to the camera.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.camera_type == "webcam":
                self.camera = cv2.VideoCapture(self.camera_id)
                
                # Apply any configuration settings
                if "resolution" in self.config:
                    width, height = self.config["resolution"]
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Verify camera is opened
                if not self.camera.isOpened():
                    print(f"Failed to open camera with ID {self.camera_id}")
                    return False
                
                # Get actual resolution (may differ from requested)
                actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                self.current_resolution = (int(actual_width), int(actual_height))
                
                # Query supported features
                self._query_camera_capabilities()
                
                self.connected = True
                return True
                
            elif self.camera_type == "dslr":
                # Placeholder for DSLR connection via gphoto2
                # Would use subprocess to call gphoto2 commands
                print("DSLR camera support requires additional setup")
                return False
                
            elif self.camera_type == "realsense":
                try:
                    import pyrealsense2 as rs
                    self.camera = rs.pipeline()
                    config = rs.config()
                    config.enable_stream(rs.stream.depth)
                    config.enable_stream(rs.stream.color)
                    self.camera.start(config)
                    self.connected = True
                    return True
                except ImportError:
                    print("pyrealsense2 library not found. Install it for RealSense support.")
                    return False
            else:
                print(f"Unsupported camera type: {self.camera_type}")
                return False
                
        except Exception as e:
            print(f"Error connecting to camera: {str(e)}")
            return False
    
    def _query_camera_capabilities(self):
        """Query and store camera capabilities and supported features"""
        if self.camera_type == "webcam" and self.camera is not None:
            # Get supported resolutions by testing common formats
            test_resolutions = [
                (640, 480), (800, 600), (1280, 720), 
                (1920, 1080), (2560, 1440), (3840, 2160)
            ]
            
            self.supported_resolutions = []
            current_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            current_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            for width, height in test_resolutions:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                # If the actual resolution matches the requested one, it's supported
                if abs(actual_width - width) < 10 and abs(actual_height - height) < 10:
                    self.supported_resolutions.append((int(actual_width), int(actual_height)))
            
            # Restore original resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, current_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, current_height)
            
            # Check for other supported features
            self.supported_features = {
                "auto_exposure": self.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                "auto_wb": self.camera.get(cv2.CAP_PROP_AUTO_WB),
                "brightness": self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
                "contrast": self.camera.get(cv2.CAP_PROP_CONTRAST),
                "saturation": self.camera.get(cv2.CAP_PROP_SATURATION),
                "gain": self.camera.get(cv2.CAP_PROP_GAIN),
                "fps": self.camera.get(cv2.CAP_PROP_FPS)
            }
    
    def capture_image(self) -> Tuple[bool, np.ndarray]:
        """
        Capture a single image from the camera.
        
        Returns:
            Tuple[bool, numpy.ndarray]: Success flag and the captured image
        """
        if not self.connected or self.camera is None:
            print("Camera not connected")
            return False, None
        
        try:
            if self.camera_type == "webcam":
                # For webcams, we might need to capture a few frames to "warm up" the camera
                for _ in range(5):  # Discard first few frames
                    ret, _  = self.camera.read()
                    if not ret:
                        return False, None
                    time.sleep(0.1)
                
                # Now capture the actual frame we want
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture image")
                    return False, None
                
                # Apply any immediate post-processing
                if "flip_horizontal" in self.config and self.config["flip_horizontal"]:
                    frame = cv2.flip(frame, 1)
                
                return True, frame
                
            elif self.camera_type == "realsense":
                try:
                    import pyrealsense2 as rs
                    frames = self.camera.wait_for_frames()
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    
                    if not depth_frame or not color_frame:
                        return False, None
                    
                    # Convert to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Return combined data as dict-like numpy array
                    result = {
                        "color": color_image,
                        "depth": depth_image
                    }
                    return True, result
                except Exception as e:
                    print(f"Error capturing RealSense image: {str(e)}")
                    return False, None
                    
            else:
                print(f"Image capture not implemented for {self.camera_type}")
                return False, None
                
        except Exception as e:
            print(f"Error during image capture: {str(e)}")
            return False, None
    
    def capture_sequence(self, num_frames: int, delay: float = 0.5) -> List[np.ndarray]:
        """
        Capture a sequence of images, useful for multi-view reconstruction.
        
        Args:
            num_frames: Number of frames to capture
            delay: Delay between captures (seconds)
            
        Returns:
            List of captured images
        """
        if not self.connected or self.camera is None:
            print("Camera not connected")
            return []
        
        frames = []
        for i in range(num_frames):
            success, frame = self.capture_image()
            if success:
                frames.append(frame)
                print(f"Captured frame {i+1}/{num_frames}")
            else:
                print(f"Failed to capture frame {i+1}")
            
            if i < num_frames - 1:  # Don't delay after the last frame
                time.sleep(delay)
                
        return frames
    
    def calibrate(self, checkerboard_size: Tuple[int, int] = (9, 6), 
                 num_images: int = 10) -> bool:
        """
        Perform camera calibration using a checkerboard pattern.
        
        Args:
            checkerboard_size: Number of internal corners in the checkerboard pattern
            num_images: Number of calibration images to capture
            
        Returns:
            bool: True if calibration successful
        """
        if not self.connected or self.camera is None:
            print("Camera not connected")
            return False
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        print(f"Starting calibration. Please show checkerboard in different orientations.")
        print(f"Will capture {num_images} images...")
        
        for i in range(num_images):
            input(f"Press Enter to capture calibration image {i+1}/{num_images}...")
            success, img = self.capture_image()
            
            if not success:
                print("Failed to capture calibration image")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            
            if ret:
                objpoints.append(objp)
                
                # Refine corner locations
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                # Draw and display the corners
                cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
                cv2.imshow('Calibration Image', img)
                cv2.waitKey(500)
            else:
                print("Checkerboard not found in image. Try again.")
                i -= 1  # Try again for this image
        
        cv2.destroyAllWindows()
        
        if len(objpoints) < 5:
            print("Not enough calibration images with detected checkerboard")
            return False
            
        # Perform the calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        if ret:
            self.calibration_data = {
                "camera_matrix": mtx.tolist(),
                "distortion_coefficients": dist.tolist(),
                "image_size": gray.shape[::-1]
            }
            
            # Calculate reprojection error
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            
            print(f"Calibration complete. Mean reprojection error: {mean_error/len(objpoints)}")
            return True
        else:
            print("Calibration failed")
            return False
    
    def save_calibration(self, file_path: str) -> bool:
        """
        Save camera calibration data to a JSON file.
        
        Args:
            file_path: Path to save the calibration data
            
        Returns:
            bool: True if saved successfully
        """
        if self.calibration_data is None:
            print("No calibration data available")
            return False
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.calibration_data, f, indent=4)
            print(f"Calibration data saved to {file_path}")
            return True
        except Exception as e:
            print(f"Failed to save calibration data: {str(e)}")
            return False
    
    def load_calibration(self, file_path: str) -> bool:
        """
        Load camera calibration data from a JSON file.
        
        Args:
            file_path: Path to the calibration data file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            with open(file_path, 'r') as f:
                self.calibration_data = json.load(f)
            print(f"Calibration data loaded from {file_path}")
            return True
        except Exception as e:
            print(f"Failed to load calibration data: {str(e)}")
            return False
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set camera resolution.
        
        Args:
            width: Desired width in pixels
            height: Desired height in pixels
            
        Returns:
            bool: True if resolution was set successfully
        """
        if not self.connected or self.camera is None:
            print("Camera not connected")
            return False
        
        if self.camera_type == "webcam":
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify the resolution was set
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if abs(actual_width - width) > 10 or abs(actual_height - height) > 10:
                print(f"Warning: Requested resolution ({width}x{height}) not set exactly. " 
                      f"Actual resolution: {actual_width}x{actual_height}")
            
            self.current_resolution = (int(actual_width), int(actual_height))
            return True
        else:
            print(f"Setting resolution not implemented for {self.camera_type}")
            return False
    
    def disconnect(self):
        """Release camera resources and disconnect."""
        if self.connected and self.camera is not None:
            if self.camera_type == "webcam":
                self.camera.release()
            elif self.camera_type == "realsense":
                self.camera.stop()
            
            self.camera = None
            self.connected = False
            print("Camera disconnected")
    
    def __del__(self):
        """Ensure camera resources are released when object is destroyed."""
        self.disconnect()
