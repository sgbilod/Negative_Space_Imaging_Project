"""
Real-time Negative Space Analysis Module

This module provides functionality for real-time tracking and analysis
of negative spaces, enabling continuous monitoring of spatial changes.

Classes:
    RealTimeTracker: Tracks negative spaces in real-time
    FrameBuffer: Buffer for frame-by-frame comparison
    StreamProcessor: Processes streams of point clouds
"""

import os
import sys
import time
import threading
import logging
import queue
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from enum import Enum
from pathlib import Path
from datetime import datetime

# Import from centralized fallbacks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.fallbacks import np, NUMPY_AVAILABLE

# Import project modules
from temporal_variants.negative_space_tracker import (
    NegativeSpaceTracker, ChangeMetrics, ChangeType
)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisMode(Enum):
    """Analysis modes for real-time tracking"""
    CONTINUOUS = 1  # Process every frame
    INTERVAL = 2    # Process frames at fixed intervals
    ADAPTIVE = 3    # Adaptively adjust processing based on complexity
    TRIGGER = 4     # Process only when triggered by events


class PerformanceMetrics:
    """Performance metrics for real-time processing"""
    
    def __init__(self):
        """Initialize performance metrics"""
        self.processing_times = []
        self.frame_rates = []
        self.memory_usage = []
        self.start_time = time.time()
        self.processed_frames = 0
        self.skipped_frames = 0
        self.last_frame_time = self.start_time
    
    def record_frame_processed(self, processing_time: float):
        """
        Record metrics for a processed frame
        
        Args:
            processing_time: Time taken to process the frame in seconds
        """
        self.processing_times.append(processing_time)
        
        # Compute instantaneous frame rate
        current_time = time.time()
        if self.last_frame_time != self.start_time:
            frame_rate = 1.0 / (current_time - self.last_frame_time)
            self.frame_rates.append(frame_rate)
        
        self.last_frame_time = current_time
        self.processed_frames += 1
        
        # Keep metrics lists from growing too large
        max_history = 1000
        if len(self.processing_times) > max_history:
            self.processing_times = self.processing_times[-max_history:]
        if len(self.frame_rates) > max_history:
            self.frame_rates = self.frame_rates[-max_history:]
    
    def record_frame_skipped(self):
        """Record a skipped frame"""
        self.skipped_frames += 1
    
    def get_average_frame_rate(self) -> float:
        """
        Get the average frame rate
        
        Returns:
            float: Average frame rate in frames per second
        """
        if not self.frame_rates:
            return 0.0
        return sum(self.frame_rates) / len(self.frame_rates)
    
    def get_average_processing_time(self) -> float:
        """
        Get the average processing time
        
        Returns:
            float: Average processing time in seconds
        """
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    def get_total_runtime(self) -> float:
        """
        Get the total runtime
        
        Returns:
            float: Total runtime in seconds
        """
        return time.time() - self.start_time
    
    def summarize(self) -> Dict[str, float]:
        """
        Summarize performance metrics
        
        Returns:
            Dict[str, float]: Summary of performance metrics
        """
        avg_fps = self.get_average_frame_rate()
        avg_time = self.get_average_processing_time()
        total_time = self.get_total_runtime()
        
        return {
            "average_fps": avg_fps,
            "average_processing_time": avg_time,
            "total_runtime": total_time,
            "processed_frames": self.processed_frames,
            "skipped_frames": self.skipped_frames,
            "efficiency": self.processed_frames / (self.processed_frames + self.skipped_frames) 
                         if (self.processed_frames + self.skipped_frames) > 0 else 1.0
        }


class FrameBuffer:
    """Buffer for storing and accessing frames for analysis"""
    
    def __init__(self, max_size: int = 30):
        """
        Initialize a frame buffer
        
        Args:
            max_size: Maximum number of frames to store
        """
        self.max_size = max_size
        self.frames = []
        self.timestamps = []
        self.lock = threading.Lock()
    
    def add_frame(self, frame: Any):
        """
        Add a frame to the buffer
        
        Args:
            frame: The frame to add
        """
        with self.lock:
            self.frames.append(frame)
            self.timestamps.append(time.time())
            
            # Remove oldest frames if buffer is full
            if len(self.frames) > self.max_size:
                self.frames = self.frames[-self.max_size:]
                self.timestamps = self.timestamps[-self.max_size:]
    
    def get_frame(self, index: int = -1) -> Optional[Any]:
        """
        Get a frame from the buffer
        
        Args:
            index: Index of the frame to get (-1 for latest)
            
        Returns:
            The frame at the specified index, or None if index is invalid
        """
        with self.lock:
            if not self.frames or index >= len(self.frames) or abs(index) > len(self.frames):
                return None
            return self.frames[index]
    
    def get_frames(self, start_index: int = 0, end_index: int = None) -> List[Any]:
        """
        Get a range of frames from the buffer
        
        Args:
            start_index: Start index (inclusive)
            end_index: End index (exclusive), or None for all frames from start_index
            
        Returns:
            List of frames in the specified range
        """
        with self.lock:
            if not self.frames:
                return []
            if end_index is None:
                end_index = len(self.frames)
            return self.frames[start_index:end_index]
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.frames = []
            self.timestamps = []
    
    def get_buffer_size(self) -> int:
        """
        Get the current size of the buffer
        
        Returns:
            int: Number of frames in the buffer
        """
        with self.lock:
            return len(self.frames)


class StreamProcessor:
    """Processes streams of point clouds in real-time"""
    
    def __init__(self, buffer_size: int = 30, 
                 mode: AnalysisMode = AnalysisMode.CONTINUOUS):
        """
        Initialize a stream processor
        
        Args:
            buffer_size: Size of the frame buffer
            mode: Analysis mode
        """
        self.buffer = FrameBuffer(buffer_size)
        self.mode = mode
        self.tracker = NegativeSpaceTracker()
        self.performance = PerformanceMetrics()
        
        self.processing_thread = None
        self.running = False
        self.frame_queue = queue.Queue()
        
        self.interval = 1  # For INTERVAL mode, process every nth frame
        self.adaptive_threshold = 0.1  # For ADAPTIVE mode
        self.triggers = []  # For TRIGGER mode
        
        self.alert_callbacks = []
    
    def start(self):
        """Start the stream processor"""
        if self.processing_thread and self.running:
            logger.warning("Stream processor is already running")
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info(f"Stream processor started in {self.mode.name} mode")
    
    def stop(self):
        """Stop the stream processor"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            self.processing_thread = None
        
        logger.info("Stream processor stopped")
    
    def add_frame(self, frame: Any):
        """
        Add a frame to the processor
        
        Args:
            frame: The frame to add
        """
        self.buffer.add_frame(frame)
        self.frame_queue.put(frame)
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get the next frame
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Decide whether to process this frame
                process_frame = self._should_process_frame(frame)
                
                if process_frame:
                    # Process the frame
                    start_time = time.time()
                    self._process_frame(frame)
                    processing_time = time.time() - start_time
                    
                    # Record performance metrics
                    self.performance.record_frame_processed(processing_time)
                else:
                    # Skip this frame
                    self.performance.record_frame_skipped()
                
                # Mark the frame as processed
                self.frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                continue
    
    def _should_process_frame(self, frame: Any) -> bool:
        """
        Determine whether to process a frame based on the analysis mode
        
        Args:
            frame: The frame to check
            
        Returns:
            bool: True if the frame should be processed, False otherwise
        """
        if self.mode == AnalysisMode.CONTINUOUS:
            return True
        
        elif self.mode == AnalysisMode.INTERVAL:
            # Process every nth frame
            frame_index = self.buffer.get_buffer_size() - 1
            return frame_index % self.interval == 0
        
        elif self.mode == AnalysisMode.ADAPTIVE:
            # Check if the frame is significantly different from the previous frame
            prev_frame = self.buffer.get_frame(-2)
            if prev_frame is None:
                return True
            
            # Simple metric: check difference in number of points
            # In a real implementation, we would use a more sophisticated metric
            if hasattr(frame, 'points') and hasattr(prev_frame, 'points'):
                point_count_diff = abs(len(frame.points) - len(prev_frame.points))
                point_count_ratio = point_count_diff / max(1, len(prev_frame.points))
                return point_count_ratio > self.adaptive_threshold
            
            return True
        
        elif self.mode == AnalysisMode.TRIGGER:
            # Check if any triggers are active
            for trigger_func in self.triggers:
                if trigger_func(frame):
                    return True
            
            return False
        
        return True
    
    def _process_frame(self, frame: Any):
        """
        Process a frame
        
        Args:
            frame: The frame to process
        """
        # Add to tracker and get change metrics
        metrics = self.tracker.add_point_cloud(frame)
        
        # Determine type of change
        change_type = self.tracker.get_change_type(metrics)
        
        # Check for significant changes that require alerts
        if change_type in [ChangeType.EMERGENCE, ChangeType.DISSOLUTION]:
            self._trigger_alert(frame, metrics, change_type)
    
    def _trigger_alert(self, frame: Any, metrics: ChangeMetrics, change_type: ChangeType):
        """
        Trigger alerts for significant changes
        
        Args:
            frame: The frame that triggered the alert
            metrics: Change metrics
            change_type: Type of change
        """
        alert_data = {
            'frame': frame,
            'metrics': metrics,
            'change_type': change_type,
            'timestamp': datetime.now(),
            'message': f"Significant change detected: {change_type.name}"
        }
        
        # Call all registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable[[Dict], None]):
        """
        Add a callback function to be called when an alert is triggered
        
        Args:
            callback: Function to call when an alert is triggered
        """
        self.alert_callbacks.append(callback)
    
    def set_mode(self, mode: AnalysisMode, **kwargs):
        """
        Set the analysis mode
        
        Args:
            mode: The analysis mode to set
            **kwargs: Additional parameters for the mode
        """
        self.mode = mode
        
        # Set mode-specific parameters
        if mode == AnalysisMode.INTERVAL and 'interval' in kwargs:
            self.interval = kwargs['interval']
        
        elif mode == AnalysisMode.ADAPTIVE and 'threshold' in kwargs:
            self.adaptive_threshold = kwargs['threshold']
        
        elif mode == AnalysisMode.TRIGGER and 'triggers' in kwargs:
            self.triggers = kwargs['triggers']
        
        logger.info(f"Analysis mode set to {mode.name}")
    
    def add_trigger(self, trigger_func: Callable[[Any], bool]):
        """
        Add a trigger function for TRIGGER mode
        
        Args:
            trigger_func: Function that takes a frame and returns True if processing should be triggered
        """
        self.triggers.append(trigger_func)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get a summary of performance metrics
        
        Returns:
            Dict[str, float]: Summary of performance metrics
        """
        return self.performance.summarize()


class RealTimeTracker:
    """Tracks negative spaces in real-time"""
    
    def __init__(self, target_fps: float = 30.0, 
                 mode: AnalysisMode = AnalysisMode.CONTINUOUS):
        """
        Initialize a real-time tracker
        
        Args:
            target_fps: Target frames per second
            mode: Analysis mode
        """
        self.target_fps = target_fps
        self.processor = StreamProcessor(buffer_size=int(target_fps * 3), mode=mode)
        self.last_results = None
        self.is_running = False
        
        # For camera integration (if available)
        self.camera = None
        self.camera_thread = None
    
    def start(self):
        """Start the real-time tracker"""
        self.processor.start()
        self.is_running = True
        
        logger.info(f"Real-time tracker started with target FPS: {self.target_fps}")
    
    def stop(self):
        """Stop the real-time tracker"""
        self.processor.stop()
        self.is_running = False
        
        if self.camera_thread:
            self.camera_thread = None
        
        logger.info("Real-time tracker stopped")
    
    def process_frame(self, frame: Any):
        """
        Process a single frame
        
        Args:
            frame: The frame to process
        """
        if not self.is_running:
            logger.warning("Tracker is not running. Call start() first.")
            return
        
        self.processor.add_frame(frame)
    
    def start_camera_capture(self, camera_id: int = 0):
        """
        Start capturing frames from a camera
        
        Args:
            camera_id: Camera ID to capture from
        """
        # Import OpenCV through fallbacks
        from utils.fallbacks import cv2, OPENCV_AVAILABLE
        
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV is required for camera capture")
            return False
        
        if self.camera_thread:
            logger.warning("Camera capture is already running")
            return False
        
        # Start the processor if not already running
        if not self.is_running:
            self.start()
        
        # Initialize camera
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return False
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self._camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        logger.info(f"Camera capture started on camera {camera_id}")
        return True
    
    def _camera_loop(self):
        """Camera capture loop"""
        while self.is_running and self.camera and self.camera.isOpened():
            try:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to capture frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Convert frame to point cloud
                point_cloud = self._frame_to_point_cloud(frame)
                
                # Process the point cloud
                self.process_frame(point_cloud)
                
                # Maintain target FPS
                time.sleep(1.0 / self.target_fps)
                
            except Exception as e:
                logger.error(f"Error in camera loop: {e}")
                time.sleep(0.1)
    
    def _frame_to_point_cloud(self, frame):
        """
        Convert a camera frame to a point cloud
        
        Args:
            frame: The camera frame to convert
            
        Returns:
            A point cloud object
        """
        # In a real implementation, this would perform depth estimation
        # and convert the image to a point cloud
        # For this example, we'll create a simple dummy point cloud
        
        # Import simplified demo classes
        sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
        try:
            from simplified_demo import SimplePointCloud
        except ImportError:
            logger.error("Could not import SimplePointCloud from simplified_demo")
            return None
        
        # Create a dummy point cloud with some random points
        point_cloud = SimplePointCloud()
        
        # Add random points based on the frame (this is just a placeholder)
        if NUMPY_AVAILABLE:
            h, w = frame.shape[:2]
            
            # Create points based on image intensity
            gray = frame
            if len(frame.shape) == 3:
                from utils.fallbacks import cv2
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Downsample to reduce point count
            scale = 0.1
            downsampled = cv2.resize(gray, (int(w * scale), int(h * scale)))
            
            # Create points based on intensity
            points = []
            colors = []
            for y in range(downsampled.shape[0]):
                for x in range(downsampled.shape[1]):
                    if downsampled[y, x] > 128:  # Only use bright pixels
                        # Project 2D to 3D (placeholder)
                        points.append([
                            x / downsampled.shape[1] - 0.5,
                            y / downsampled.shape[0] - 0.5,
                            downsampled[y, x] / 255.0 - 0.5
                        ])
                        
                        # Use RGB from original frame if available
                        if len(frame.shape) == 3:
                            b = frame[int(y/scale), int(x/scale), 0] / 255.0
                            g = frame[int(y/scale), int(x/scale), 1] / 255.0
                            r = frame[int(y/scale), int(x/scale), 2] / 255.0
                            colors.append([r, g, b])
                        else:
                            colors.append([0.5, 0.5, 0.5])
            
            # Add points to point cloud
            if points:
                point_cloud.add_points(np.array(points), np.array(colors))
                point_cloud.classify_points()
                point_cloud.generate_void_points()
        
        return point_cloud
    
    def set_alert_callback(self, callback: Callable[[Dict], None]):
        """
        Set a callback function for alerts
        
        Args:
            callback: Function to call when an alert is triggered
        """
        self.processor.add_alert_callback(callback)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        return self.processor.get_performance_summary()
    
    def get_latest_results(self) -> Dict:
        """
        Get the latest results
        
        Returns:
            Dict: Latest results
        """
        return self.last_results
    
    def set_mode(self, mode: AnalysisMode, **kwargs):
        """
        Set the analysis mode
        
        Args:
            mode: The analysis mode to set
            **kwargs: Additional parameters for the mode
        """
        self.processor.set_mode(mode, **kwargs)
