"""
Feature Detector Module for Negative Space Imaging Project

This module implements specialized feature detection algorithms that are
optimized for negative space boundaries, focusing on the transitions between
objects and empty space.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum
import time

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Types of features to detect"""
    SIFT = "sift"
    ORB = "orb"
    AKAZE = "akaze"
    BRISK = "brisk"
    BOUNDARY = "boundary"  # Custom boundary detector
    VOID_EDGE = "void_edge"  # Specialized for edges of negative space
    OBJECT_SILHOUETTE = "object_silhouette"  # Outlines of objects
    MULTI_SCALE = "multi_scale"  # Multi-scale feature detection

@dataclass
class FeatureDetectionParams:
    """Parameters for feature detection"""
    # General parameters
    max_features: int = 5000
    feature_quality: float = 0.01
    min_distance: int = 10
    
    # SIFT specific
    n_octave_layers: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: float = 10
    sigma: float = 1.6
    
    # ORB specific
    scale_factor: float = 1.2
    n_levels: int = 8
    edge_threshold_orb: int = 31
    first_level: int = 0
    WTA_K: int = 2
    
    # Boundary detection specific
    boundary_threshold: int = 30
    boundary_kernel_size: int = 3
    boundary_iterations: int = 2
    
    # Void edge specific
    void_edge_sensitivity: float = 0.5
    void_area_threshold: float = 100
    
    # Depth-aware parameters
    use_depth: bool = False
    depth_weight: float = 0.5
    depth_threshold: int = 20

class FeatureDetector:
    """
    Specialized feature detector optimized for negative space analysis.
    
    This class implements various feature detection algorithms that are
    specifically designed to identify features along the boundaries between
    objects and negative space.
    """
    
    def __init__(self, feature_type: Union[str, FeatureType] = FeatureType.SIFT,
                 params: Optional[FeatureDetectionParams] = None):
        """
        Initialize the feature detector with specific feature type and parameters.
        
        Args:
            feature_type: Type of features to detect
            params: Parameters for feature detection
        """
        if isinstance(feature_type, str):
            try:
                self.feature_type = FeatureType(feature_type)
            except ValueError:
                logger.warning(f"Invalid feature type '{feature_type}'. Using SIFT instead.")
                self.feature_type = FeatureType.SIFT
        else:
            self.feature_type = feature_type
        
        self.params = params or FeatureDetectionParams()
        
        # Initialize the appropriate detector
        self._init_detector()
        
        # Storage for detected features
        self.keypoints = []
        self.descriptors = None
        
        # Metadata from detection process
        self.metadata = {}
    
    def _init_detector(self):
        """Initialize the feature detector based on the feature type"""
        try:
            if self.feature_type == FeatureType.SIFT:
                self.detector = cv2.SIFT_create(
                    nfeatures=self.params.max_features,
                    nOctaveLayers=self.params.n_octave_layers,
                    contrastThreshold=self.params.contrast_threshold,
                    edgeThreshold=self.params.edge_threshold,
                    sigma=self.params.sigma
                )
            elif self.feature_type == FeatureType.ORB:
                self.detector = cv2.ORB_create(
                    nfeatures=self.params.max_features,
                    scaleFactor=self.params.scale_factor,
                    nlevels=self.params.n_levels,
                    edgeThreshold=self.params.edge_threshold_orb,
                    firstLevel=self.params.first_level,
                    WTA_K=self.params.WTA_K
                )
            elif self.feature_type == FeatureType.AKAZE:
                self.detector = cv2.AKAZE_create()
            elif self.feature_type == FeatureType.BRISK:
                self.detector = cv2.BRISK_create()
            elif self.feature_type in [FeatureType.BOUNDARY, FeatureType.VOID_EDGE, 
                                      FeatureType.OBJECT_SILHOUETTE, FeatureType.MULTI_SCALE]:
                # These are custom implementations, no OpenCV detector to initialize
                self.detector = None
            else:
                logger.warning(f"Unsupported feature type: {self.feature_type}. Using SIFT instead.")
                self.feature_type = FeatureType.SIFT
                self.detector = cv2.SIFT_create(nfeatures=self.params.max_features)
        except Exception as e:
            logger.error(f"Error initializing detector: {str(e)}")
            self.detector = None
    
    def detect(self, image: np.ndarray, 
              depth_map: Optional[np.ndarray] = None,
              mask: Optional[np.ndarray] = None) -> Tuple[List, np.ndarray]:
        """
        Detect features in an image.
        
        Args:
            image: Input image
            depth_map: Optional depth map
            mask: Optional mask for feature detection
            
        Returns:
            Tuple containing:
                - List of keypoints
                - Feature descriptors
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
        
        start_time = time.time()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Store metadata
        self.metadata = {
            "image_shape": image.shape,
            "feature_type": self.feature_type.value,
            "timestamp": start_time,
            "has_depth": depth_map is not None,
            "has_mask": mask is not None
        }
        
        # Detect features based on the feature type
        if self.feature_type in [FeatureType.SIFT, FeatureType.ORB, 
                                FeatureType.AKAZE, FeatureType.BRISK]:
            if self.detector is None:
                raise RuntimeError("Feature detector not properly initialized")
            
            self.keypoints, self.descriptors = self.detector.detectAndCompute(gray, mask)
            
        elif self.feature_type == FeatureType.BOUNDARY:
            self.keypoints, self.descriptors = self._detect_boundary_features(gray, depth_map, mask)
            
        elif self.feature_type == FeatureType.VOID_EDGE:
            self.keypoints, self.descriptors = self._detect_void_edge_features(gray, depth_map, mask)
            
        elif self.feature_type == FeatureType.OBJECT_SILHOUETTE:
            self.keypoints, self.descriptors = self._detect_silhouette_features(gray, depth_map, mask)
            
        elif self.feature_type == FeatureType.MULTI_SCALE:
            self.keypoints, self.descriptors = self._detect_multi_scale_features(gray, depth_map, mask)
        
        # Update metadata
        processing_time = time.time() - start_time
        self.metadata["processing_time"] = processing_time
        self.metadata["num_features"] = len(self.keypoints)
        
        return self.keypoints, self.descriptors
    
    def _detect_boundary_features(self, gray: np.ndarray,
                                 depth_map: Optional[np.ndarray] = None,
                                 mask: Optional[np.ndarray] = None) -> Tuple[List, np.ndarray]:
        """
        Custom detector for boundary features.
        
        This detector focuses on finding features along the boundaries between
        objects and negative space, emphasizing the transition regions.
        
        Args:
            gray: Grayscale image
            depth_map: Optional depth map
            mask: Optional mask
            
        Returns:
            Tuple of keypoints and descriptors
        """
        # Detect edges using Canny
        edges = cv2.Canny(gray, self.params.boundary_threshold, 
                         self.params.boundary_threshold * 2)
        
        # If depth map is available, incorporate depth edges
        if depth_map is not None and self.params.use_depth:
            # Normalize depth map
            if depth_map.dtype != np.uint8:
                normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                normalized_depth = depth_map
            
            # Detect edges in depth map
            depth_edges = cv2.Canny(normalized_depth, 
                                   self.params.depth_threshold, 
                                   self.params.depth_threshold * 2)
            
            # Combine RGB and depth edges
            combined_edges = cv2.addWeighted(
                edges, 1.0 - self.params.depth_weight,
                depth_edges, self.params.depth_weight, 0
            )
        else:
            combined_edges = edges
        
        # Apply mask if provided
        if mask is not None:
            combined_edges = cv2.bitwise_and(combined_edges, combined_edges, mask=mask)
        
        # Dilate edges to create boundary regions
        kernel = np.ones((self.params.boundary_kernel_size, 
                          self.params.boundary_kernel_size), np.uint8)
        dilated_edges = cv2.dilate(combined_edges, kernel, 
                                  iterations=self.params.boundary_iterations)
        
        # Find contours in the edge image
        contours, _ = cv2.findContours(dilated_edges, 
                                      cv2.RETR_LIST, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to keypoints
        keypoints = []
        for contour in contours:
            # Sample points along the contour
            step = max(1, len(contour) // 100)  # Limit to ~100 points per contour
            for i in range(0, len(contour), step):
                x, y = contour[i][0]
                keypoints.append(cv2.KeyPoint(float(x), float(y), 
                                            size=self.params.boundary_kernel_size))
        
        # Limit the number of keypoints
        if len(keypoints) > self.params.max_features:
            # Sort by response (strength) and take the strongest ones
            keypoints = sorted(keypoints, key=lambda x: -x.response)[:self.params.max_features]
        
        # Compute descriptors using SIFT
        sift = cv2.SIFT_create()
        _, descriptors = sift.compute(gray, keypoints)
        
        return keypoints, descriptors
    
    def _detect_void_edge_features(self, gray: np.ndarray,
                                  depth_map: Optional[np.ndarray] = None,
                                  mask: Optional[np.ndarray] = None) -> Tuple[List, np.ndarray]:
        """
        Custom detector for void edge features.
        
        This detector specifically looks for features along the edges of
        negative space regions, prioritizing the void boundaries.
        
        Args:
            gray: Grayscale image
            depth_map: Optional depth map
            mask: Optional mask
            
        Returns:
            Tuple of keypoints and descriptors
        """
        # Start with adaptive thresholding to find potential void regions
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours of potential void regions
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.params.void_area_threshold]
        
        # Create a mask of large void regions
        void_mask = np.zeros_like(gray)
        cv2.drawContours(void_mask, large_contours, -1, 255, -1)
        
        # Find the boundary between void and non-void
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(void_mask, kernel, iterations=1)
        edge_mask = cv2.subtract(dilated, void_mask)
        
        # If depth map is available, refine void regions
        if depth_map is not None and self.params.use_depth:
            # Normalize depth map
            if depth_map.dtype != np.uint8:
                normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                normalized_depth = depth_map
            
            # Find regions with larger depth (likely void)
            depth_threshold = int(255 * self.params.void_edge_sensitivity)
            far_regions = normalized_depth > depth_threshold
            far_regions = far_regions.astype(np.uint8) * 255
            
            # Combine with image-based void detection
            combined_void = cv2.bitwise_and(void_mask, far_regions)
            
            # Find edge of combined void
            dilated_combined = cv2.dilate(combined_void, kernel, iterations=1)
            refined_edge_mask = cv2.subtract(dilated_combined, combined_void)
            
            # Use refined edge mask
            edge_mask = refined_edge_mask
        
        # Apply user mask if provided
        if mask is not None:
            edge_mask = cv2.bitwise_and(edge_mask, mask)
        
        # Extract points along the void edges
        edge_points = np.where(edge_mask > 0)
        y_points, x_points = edge_points[0], edge_points[1]
        
        # Convert points to keypoints
        keypoints = []
        step = max(1, len(x_points) // self.params.max_features)
        for i in range(0, len(x_points), step):
            x, y = x_points[i], y_points[i]
            # Calculate feature response based on local gradient
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx[y, x]**2 + gy[y, x]**2)
            
            kp = cv2.KeyPoint(float(x), float(y), 
                             size=5, 
                             response=float(magnitude))
            keypoints.append(kp)
        
        # Sort by response and limit
        keypoints = sorted(keypoints, key=lambda x: -x.response)[:self.params.max_features]
        
        # Compute descriptors using SIFT
        sift = cv2.SIFT_create()
        _, descriptors = sift.compute(gray, keypoints)
        
        return keypoints, descriptors
    
    def _detect_silhouette_features(self, gray: np.ndarray,
                                   depth_map: Optional[np.ndarray] = None,
                                   mask: Optional[np.ndarray] = None) -> Tuple[List, np.ndarray]:
        """
        Custom detector for object silhouette features.
        
        This detector focuses on the complete outlines of objects,
        defining the boundaries of the negative space.
        
        Args:
            gray: Grayscale image
            depth_map: Optional depth map
            mask: Optional mask
            
        Returns:
            Tuple of keypoints and descriptors
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to separate objects from background
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # If depth map is available, incorporate it
        if depth_map is not None and self.params.use_depth:
            # Normalize depth map
            if depth_map.dtype != np.uint8:
                normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                normalized_depth = depth_map
            
            # Threshold depth to get objects vs background
            _, depth_thresh = cv2.threshold(
                normalized_depth, 127, 255, cv2.THRESH_BINARY
            )
            
            # Combine with image threshold
            combined_thresh = cv2.bitwise_and(thresh, depth_thresh)
        else:
            combined_thresh = thresh
        
        # Apply mask if provided
        if mask is not None:
            combined_thresh = cv2.bitwise_and(combined_thresh, mask)
        
        # Find contours of objects
        contours, _ = cv2.findContours(
            combined_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        # Extract silhouette points
        keypoints = []
        for contour in contours:
            # Skip very small contours
            if len(contour) < 10:
                continue
                
            # Simplify the contour to reduce redundant points
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Extract points from the simplified contour
            for point in approx:
                x, y = point[0]
                kp = cv2.KeyPoint(float(x), float(y), size=7)
                keypoints.append(kp)
        
        # Limit the number of keypoints
        if len(keypoints) > self.params.max_features:
            keypoints = keypoints[:self.params.max_features]
        
        # Compute descriptors using SIFT
        sift = cv2.SIFT_create()
        _, descriptors = sift.compute(gray, keypoints)
        
        return keypoints, descriptors
    
    def _detect_multi_scale_features(self, gray: np.ndarray,
                                    depth_map: Optional[np.ndarray] = None,
                                    mask: Optional[np.ndarray] = None) -> Tuple[List, np.ndarray]:
        """
        Custom multi-scale feature detector.
        
        This detector combines features from multiple scales and methods,
        providing a comprehensive set of features for negative space analysis.
        
        Args:
            gray: Grayscale image
            depth_map: Optional depth map
            mask: Optional mask
            
        Returns:
            Tuple of keypoints and descriptors
        """
        all_keypoints = []
        
        # 1. Detect SIFT features
        sift = cv2.SIFT_create(nfeatures=self.params.max_features // 3)
        sift_kp, _ = sift.detectAndCompute(gray, mask)
        all_keypoints.extend(sift_kp)
        
        # 2. Detect boundary features
        boundary_detector = FeatureDetector(FeatureType.BOUNDARY, self.params)
        boundary_kp, _ = boundary_detector._detect_boundary_features(gray, depth_map, mask)
        all_keypoints.extend(boundary_kp)
        
        # 3. Detect void edge features
        void_detector = FeatureDetector(FeatureType.VOID_EDGE, self.params)
        void_kp, _ = void_detector._detect_void_edge_features(gray, depth_map, mask)
        all_keypoints.extend(void_kp)
        
        # Remove duplicate keypoints
        filtered_keypoints = []
        used_positions = set()
        
        for kp in all_keypoints:
            pos = (int(kp.pt[0]), int(kp.pt[1]))
            if pos not in used_positions:
                filtered_keypoints.append(kp)
                used_positions.add(pos)
        
        # Limit the number of keypoints
        if len(filtered_keypoints) > self.params.max_features:
            filtered_keypoints = sorted(filtered_keypoints, 
                                      key=lambda x: -x.response)[:self.params.max_features]
        
        # Compute descriptors for all keypoints
        sift = cv2.SIFT_create()
        _, descriptors = sift.compute(gray, filtered_keypoints)
        
        return filtered_keypoints, descriptors
    
    def match_features(self, descriptors1: np.ndarray, 
                      descriptors2: np.ndarray,
                      ratio_threshold: float = 0.7) -> List[cv2.DMatch]:
        """
        Match features between two sets of descriptors.
        
        Args:
            descriptors1: First set of descriptors
            descriptors2: Second set of descriptors
            ratio_threshold: Threshold for Lowe's ratio test
            
        Returns:
            List of matches that pass the ratio test
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        
        if descriptors1.dtype != np.float32:
            descriptors1 = np.float32(descriptors1)
        if descriptors2.dtype != np.float32:
            descriptors2 = np.float32(descriptors2)
        
        # Create FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Match descriptors
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def draw_features(self, image: np.ndarray, 
                     keypoints: List = None) -> np.ndarray:
        """
        Draw detected features on an image.
        
        Args:
            image: Input image
            keypoints: Optional list of keypoints to draw (uses stored keypoints if None)
            
        Returns:
            Image with features drawn on it
        """
        if keypoints is None:
            keypoints = self.keypoints
        
        img_with_features = cv2.drawKeypoints(
            image, keypoints, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return img_with_features
    
    def draw_matches(self, img1: np.ndarray, keypoints1: List,
                    img2: np.ndarray, keypoints2: List,
                    matches: List[cv2.DMatch]) -> np.ndarray:
        """
        Draw feature matches between two images.
        
        Args:
            img1: First image
            keypoints1: Keypoints from first image
            img2: Second image
            keypoints2: Keypoints from second image
            matches: List of matches
            
        Returns:
            Image showing the matches
        """
        img_matches = cv2.drawMatches(
            img1, keypoints1, img2, keypoints2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return img_matches
    
    def filter_matches_by_geometry(self, keypoints1: List, keypoints2: List, 
                                 matches: List[cv2.DMatch],
                                 threshold: float = 3.0) -> List[cv2.DMatch]:
        """
        Filter matches using geometric constraints (RANSAC).
        
        Args:
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image
            matches: Initial matches
            threshold: Distance threshold for RANSAC
            
        Returns:
            Filtered matches that satisfy geometric constraints
        """
        if len(matches) < 4:
            return matches
        
        # Extract coordinates from keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
        
        # Filter matches based on mask
        filtered_matches = [matches[i] for i in range(len(matches)) if mask[i][0] > 0]
        
        return filtered_matches
    
    def set_feature_type(self, feature_type: Union[str, FeatureType]):
        """
        Change the feature detection method.
        
        Args:
            feature_type: New feature type to use
        """
        if isinstance(feature_type, str):
            try:
                self.feature_type = FeatureType(feature_type)
            except ValueError:
                logger.warning(f"Invalid feature type '{feature_type}'. Keeping current type.")
                return
        else:
            self.feature_type = feature_type
        
        # Reinitialize the detector
        self._init_detector()
        logger.info(f"Feature type set to {self.feature_type.value}")
    
    def set_params(self, params: FeatureDetectionParams):
        """
        Update feature detection parameters.
        
        Args:
            params: New parameters
        """
        self.params = params
        
        # Reinitialize the detector with new parameters
        self._init_detector()
        logger.info("Feature detection parameters updated")
    
    def detect_and_describe(self, images: List[np.ndarray],
                           depth_maps: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Detect and describe features in multiple images.
        
        Args:
            images: List of input images
            depth_maps: Optional list of corresponding depth maps
            
        Returns:
            Dictionary with keypoints and descriptors for each image
        """
        results = {}
        
        for i, img in enumerate(images):
            depth = None if depth_maps is None else depth_maps[i]
            
            # Detect features
            kp, desc = self.detect(img, depth)
            
            # Store results
            results[f"image_{i}"] = {
                "keypoints": kp,
                "descriptors": desc
            }
        
        return results
