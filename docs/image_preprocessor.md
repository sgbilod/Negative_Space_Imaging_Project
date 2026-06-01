# Documentation for image_preprocessor.py

```python
"""
Image Preprocessor Module for Negative Space Imaging Project

This module handles specialized preprocessing of images specifically designed
to enhance features important for negative space analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum
import time
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreprocessingMode(Enum):
    """Different preprocessing modes optimized for different scenarios"""
    STANDARD = "standard"  # Basic preprocessing
    NEGATIVE_SPACE_FOCUS = "negative_space_focus"  # Enhances boundaries between objects and void
    FEATURE_ENHANCEMENT = "feature_enhancement"  # Enhances features for better detection
    LOW_LIGHT = "low_light"  # Optimized for low light conditions
    HIGH_CONTRAST = "high_contrast"  # Maximizes contrast between objects
    DEPTH_AWARE = "depth_aware"  # Uses depth information when available
    CUSTOM = "custom"  # Custom pipeline defined by user

@dataclass
class ProcessingParams:
    """Parameters for image preprocessing"""
    # Core parameters
    blur_size: int = 5
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    threshold_type: int = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    morphology_kernel_size: int = 5
    edge_detection_low_threshold: int = 50
    edge_detection_high_threshold: int = 150
    
    # Advanced parameters
    denoise_h: int = 10  # For fastNlMeansDenoisingColored
    bilateral_d: int = 9  # For bilateralFilter
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75
    shadow_correction: bool = False
    gamma: float = 1.0  # Gamma correction value
    
    # Negative space specific
    enhance_boundaries: bool = True
    boundary_thickness: int = 2
    boundary_color_boost: float = 1.5
    void_darkness: float = 0.8  # How dark to make the void areas
    
    # Feature detection parameters
    feature_threshold: float = 0.01
    max_features: int = 5000
    feature_quality_level: float = 0.01
    min_feature_distance: int = 10

class ImagePreprocessor:
    """
    Handles specialized preprocessing of images for negative space analysis.
    
    This class implements various image enhancement and preprocessing techniques
    specifically designed to highlight the boundaries between objects and
    empty space, which is crucial for negative space mapping.
    """
    
    def __init__(self, mode: Union[str, PreprocessingMode] = PreprocessingMode.STANDARD,
                 params: Optional[ProcessingParams] = None):
        """
        Initialize the image preprocessor with specific mode and parameters.
        
        Args:
            mode: Preprocessing mode to use
            params: Optional custom parameters, if None, default parameters are used
        """
        if isinstance(mode, str):
            try:
                self.mode = PreprocessingMode(mode)
            except ValueError:
                logger.warning(f"Invalid mode '{mode}'. Using STANDARD instead.")
                self.mode = PreprocessingMode.STANDARD
        else:
            self.mode = mode
        
        self.params = params or ProcessingParams()
        self.calibration_data = None
        
        # Metadata from preprocessing that might be useful later
        self.metadata = {}
        
        # For parallel processing
        self.num_threads = max(1, cv2.getNumThreads())
        logger.info(f"ImagePreprocessor initialized with {self.num_threads} threads available")
        
        # For batch processing
        self.batch_results = {}
        self._batch_lock = threading.Lock()
    
    def preprocess(self, image: np.ndarray, 
                   depth_map: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess an image based on the current mode and parameters.
        
        Args:
            image: Input image as numpy array
            depth_map: Optional depth map corresponding to the image
            
        Returns:
            Tuple containing:
                - Preprocessed image
                - Metadata dict with information about the preprocessing
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
        
        start_time = time.time()
        metadata = {
            "original_shape": image.shape,
            "mode": self.mode.value,
            "timestamp": start_time
        }
        
        # Make a copy to avoid modifying the original
        processed = image.copy()
        
        # Apply preprocessing based on the selected mode
        if self.mode == PreprocessingMode.STANDARD:
            processed = self._standard_preprocessing(processed)
        elif self.mode == PreprocessingMode.NEGATIVE_SPACE_FOCUS:
            processed = self._negative_space_preprocessing(processed, depth_map)
        elif self.mode == PreprocessingMode.FEATURE_ENHANCEMENT:
            processed = self._feature_enhancement_preprocessing(processed)
        elif self.mode == PreprocessingMode.LOW_LIGHT:
            processed = self._low_light_preprocessing(processed)
        elif self.mode == PreprocessingMode.HIGH_CONTRAST:
            processed = self._high_contrast_preprocessing(processed)
        elif self.mode == PreprocessingMode.DEPTH_AWARE and depth_map is not None:
            processed = self._depth_aware_preprocessing(processed, depth_map)
        elif self.mode == PreprocessingMode.CUSTOM:
            processed = self._custom_preprocessing(processed, depth_map)
        else:
            # Fallback to standard if mode not implemented or depth map missing
            logger.warning(f"Mode {self.mode.value} not implemented or missing depth map. Using standard preprocessing.")
            processed = self._standard_preprocessing(processed)
        
        # Update metadata with processing time
        processing_time = time.time() - start_time
        metadata["processing_time"] = processing_time
        metadata["processed_shape"] = processed.shape
        
        # Store metadata for future reference
        self.metadata = metadata
        
        return processed, metadata
    
    def _standard_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Standard image preprocessing pipeline:
        1. Convert to RGB if needed
        2. Apply noise reduction
        3. Enhance contrast using CLAHE
        4. Apply moderate sharpening
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # Ensure RGB format
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.params.clahe_clip_limit, 
            tileGridSize=self.params.clahe_grid_size
        )
        l = clahe.apply(l)
        
        # Merge channels back
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(
            enhanced, 
            d=self.params.bilateral_d,
            sigmaColor=self.params.bilateral_sigma_color,
            sigmaSpace=self.params.bilateral_sigma_space
        )
        
        # Apply unsharp mask for sharpening
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 3)
        sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        
        return sharpened
    
    def _negative_space_preprocessing(self, image: np.ndarray, 
                                     depth_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Preprocessing optimized for negative space analysis:
        1. Enhance edges/boundaries between objects
        2. Reduce texture within objects
        3. Normalize void areas
        
        Args:
            image: Input image
            depth_map: Optional depth information
            
        Returns:
            Processed image optimized for negative space analysis
        """
        # Start with standard preprocessing
        processed = self._standard_preprocessing(image)
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(
            gray, 
            self.params.edge_detection_low_threshold, 
            self.params.edge_detection_high_threshold
        )
        
        # Dilate edges to make them more prominent
        kernel = np.ones((self.params.boundary_thickness, self.params.boundary_thickness), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Create a mask from edges
        edge_mask = dilated_edges > 0
        
        # Create edge-enhanced image
        edge_enhanced = processed.copy()
        
        # Boost color along edges
        if self.params.enhance_boundaries:
            # Create an HSV version for color manipulation
            hsv = cv2.cvtColor(edge_enhanced, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Increase saturation along edges
            s_boost = s.copy()
            s_boost[edge_mask] = np.clip(
                s_boost[edge_mask] * self.params.boundary_color_boost, 
                0, 255
            ).astype(np.uint8)
            
            # Increase value (brightness) along edges
            v_boost = v.copy()
            v_boost[edge_mask] = np.clip(
                v_boost[edge_mask] * self.params.boundary_color_boost, 
                0, 255
            ).astype(np.uint8)
            
            # Merge channels
            hsv_boosted = cv2.merge((h, s_boost, v_boost))
            edge_enhanced = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
        
        # Incorporate depth information if available
        if depth_map is not None and self.params.enhance_boundaries:
            # Normalize depth map
            if depth_map.dtype != np.uint8:
                normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                normalized_depth = depth_map
            
            # Find depth discontinuities
            depth_edges = cv2.Canny(normalized_depth, 50, 150)
            depth_edge_mask = depth_edges > 0
            
            # Combine with RGB edges
            combined_edge_mask = np.logical_or(edge_mask, depth_edge_mask)
            
            # Enhance edges further based on depth
            edge_enhanced[combined_edge_mask] = np.clip(
                edge_enhanced[combined_edge_mask] * 1.2, 
                0, 255
            ).astype(np.uint8)
            
            # Darken areas that are likely void (far in depth)
            if normalized_depth.size > 0:
                void_threshold = np.percentile(normalized_depth, 75)  # Assume far areas are void
                void_mask = normalized_depth > void_threshold
                
                # Apply darkening to likely void areas
                darkening_factor = self.params.void_darkness
                edge_enhanced[void_mask] = (edge_enhanced[void_mask] * darkening_factor).astype(np.uint8)
        
        return edge_enhanced
    
    def _feature_enhancement_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing optimized for feature detection:
        1. Enhance corners and distinctive features
        2. Normalize lighting
        3. Increase local contrast
        
        Args:
            image: Input image
            
        Returns:
            Processed image optimized for feature detection
        """
        # Start with standard preprocessing
        processed = self._standard_preprocessing(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Detect corners
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.params.max_features,
            qualityLevel=self.params.feature_quality_level,
            minDistance=self.params.min_feature_distance
        )
        
        # Enhance detected corners
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(processed, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Apply adaptive histogram equalization for better local contrast
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE with stronger parameters
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        
        # Merge channels back
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply sharpening to enhance features
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def _low_light_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing optimized for low light conditions:
        1. Boost brightness and contrast
        2. Apply noise reduction
        3. Enhance details in shadows
        
        Args:
            image: Input image
            
        Returns:
            Processed image optimized for low light
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel with stronger parameters
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Gamma correction to boost shadows
        gamma = 0.7  # Value less than 1 brightens dark regions
        l_gamma = np.array(255 * (l / 255) ** gamma, dtype=np.uint8)
        
        # Merge channels back
        lab = cv2.merge((l_gamma, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Denoise more aggressively for low light images
        denoised = cv2.fastNlMeansDenoisingColored(
            enhanced, 
            None, 
            h=15,  # Higher h for stronger noise reduction
            hColor=15,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Apply contrast stretching
        for i in range(3):  # For each color channel
            channel = denoised[:, :, i]
            min_val = np.percentile(channel, 5)
            max_val = np.percentile(channel, 95)
            
            # Clip to the range to avoid extreme outliers
            channel = np.clip(channel, min_val, max_val)
            
            # Stretch to full range
            channel = 255 * (channel - min_val) / (max_val - min_val)
            channel = np.clip(channel, 0, 255).astype(np.uint8)
            denoised[:, :, i] = channel
        
        return denoised
    
    def _high_contrast_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing for maximizing contrast:
        1. Apply aggressive contrast enhancement
        2. Normalize lighting
        3. Enhance edges
        
        Args:
            image: Input image
            
        Returns:
            High contrast processed image
        """
        # Apply standard preprocessing first
        processed = self._standard_preprocessing(image)
        
        # Convert to LAB for better color processing
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE with high clip limit
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        
        # Apply contrast stretching to L channel
        p5 = np.percentile(l, 5)
        p95 = np.percentile(l, 95)
        l = np.clip(255 * (l - p5) / (p95 - p5), 0, 255).astype(np.uint8)
        
        # Merge channels back
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply unsharp mask for edge enhancement
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)
        
        return sharpened
    
    def _depth_aware_preprocessing(self, image: np.ndarray, 
                                  depth_map: np.ndarray) -> np.ndarray:
        """
        Preprocessing that incorporates depth information:
        1. Enhance edges based on depth discontinuities
        2. Adjust processing based on distance
        3. Highlight negative space boundaries
        
        Args:
            image: Input color image
            depth_map: Corresponding depth map
            
        Returns:
            Depth-enhanced processed image
        """
        if depth_map is None or depth_map.size == 0:
            logger.warning("Depth map not provided for depth-aware preprocessing")
            return self._standard_preprocessing(image)
        
        # Normalize depth map for easier processing
        if depth_map.dtype != np.uint8:
            normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            normalized_depth = depth_map
        
        # Find depth discontinuities (edges in depth map)
        depth_edges = cv2.Canny(normalized_depth, 30, 100)
        
        # Dilate edges to make them more prominent
        kernel = np.ones((3, 3), np.uint8)
        dilated_depth_edges = cv2.dilate(depth_edges, kernel, iterations=1)
        
        # Apply standard preprocessing to color image
        processed = self._standard_preprocessing(image)
        
        # Create a colorized depth map for visualization
        depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        
        # Highlight depth edges in the processed image
        processed[dilated_depth_edges > 0] = [0, 255, 255]  # Yellow for depth edges
        
        # Create a composite visualization with depth information
        alpha = 0.7  # Weight for color image
        beta = 0.3   # Weight for depth colormap
        composite = cv2.addWeighted(processed, alpha, depth_colormap, beta, 0)
        
        # Further enhance areas based on depth
        # Assume closer objects are of more interest
        close_threshold = np.percentile(normalized_depth, 25)  # Closest 25%
        far_threshold = np.percentile(normalized_depth, 75)    # Furthest 25%
        
        close_mask = normalized_depth < close_threshold
        far_mask = normalized_depth > far_threshold
        
        # Enhance closer objects
        hsv = cv2.cvtColor(composite, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Increase saturation for close objects
        s[close_mask] = np.clip(s[close_mask] * 1.3, 0, 255).astype(np.uint8)
        
        # Decrease saturation for far objects (likely negative space)
        s[far_mask] = np.clip(s[far_mask] * 0.7, 0, 255).astype(np.uint8)
        
        hsv_enhanced = cv2.merge((h, s, v))
        enhanced_composite = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        return enhanced_composite
    
    def _custom_preprocessing(self, image: np.ndarray, 
                             depth_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Custom preprocessing pipeline defined by user parameters.
        This allows for flexible combinations of various techniques.
        
        Args:
            image: Input image
            depth_map: Optional depth map
            
        Returns:
            Custom processed image
        """
        # This is a placeholder that can be extended based on specific needs
        processed = image.copy()
        
        # Apply gamma correction if specified
        if self.params.gamma != 1.0:
            inv_gamma = 1.0 / self.params.gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
            ]).astype(np.uint8)
            processed = cv2.LUT(processed, table)
        
        # Apply denoise if specified
        if self.params.denoise_h > 0:
            processed = cv2.fastNlMeansDenoisingColored(
                processed, None, 
                h=self.params.denoise_h, 
                hColor=self.params.denoise_h, 
                templateWindowSize=7, 
                searchWindowSize=21
            )
        
        # Apply other customizations based on parameters
        # This can be extended based on specific project needs
        
        return processed
    
    def preprocess_batch(self, images: List[np.ndarray], 
                         depth_maps: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Process a batch of images in parallel.
        
        Args:
            images: List of input images
            depth_maps: Optional list of corresponding depth maps
            
        Returns:
            List of processed images
        """
        if not images:
            return []
        
        # Initialize results container
        results = [None] * len(images)
        
        # Process images in parallel
        threads = []
        
        def process_image(idx, img, depth=None):
            processed, _ = self.preprocess(img, depth)
            with self._batch_lock:
                results[idx] = processed
        
        # Create and start threads
        for i, img in enumerate(images):
            depth = None if depth_maps is None else depth_maps[i]
            thread = threading.Thread(
                target=process_image,
                args=(i, img, depth)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return results
    
    def extract_metadata(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract metadata from an image that might be useful for spatial referencing.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "dimensions": image.shape,
            "channels": image.shape[2] if len(image.shape) > 2 else 1,
            "mean_color": [float(np.mean(image[:, :, i])) for i in range(min(3, image.shape[2]))] 
                           if len(image.shape) > 2 else float(np.mean(image)),
            "std_dev": [float(np.std(image[:, :, i])) for i in range(min(3, image.shape[2]))]
                        if len(image.shape) > 2 else float(np.std(image)),
            "histogram": [cv2.calcHist([image], [i], None, [256], [0, 256]).flatten().tolist() 
                          for i in range(min(3, image.shape[2]))]
                          if len(image.shape) > 2 else cv2.calcHist([image], [0], None, [256], [0, 256]).flatten().tolist(),
        }
        
        # Extract basic image features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Detect corners
        corners = cv2.goodFeaturesToTrack(
            gray, 
            maxCorners=100, 
            qualityLevel=0.01, 
            minDistance=10
        )
        
        if corners is not None:
            metadata["num_corners"] = len(corners)
            metadata["corner_locations"] = corners.reshape(-1, 2).tolist()
        else:
            metadata["num_corners"] = 0
            metadata["corner_locations"] = []
        
        # Basic edge information
        edges = cv2.Canny(gray, 100, 200)
        metadata["edge_pixels"] = int(np.sum(edges > 0))
        metadata["edge_ratio"] = float(metadata["edge_pixels"] / (gray.shape[0] * gray.shape[1]))
        
        return metadata
    
    def set_mode(self, mode: Union[str, PreprocessingMode]):
        """
        Change the preprocessing mode.
        
        Args:
            mode: New preprocessing mode
        """
        if isinstance(mode, str):
            try:
                self.mode = PreprocessingMode(mode)
            except ValueError:
                logger.warning(f"Invalid mode '{mode}'. Keeping current mode.")
        else:
            self.mode = mode
        
        logger.info(f"Preprocessing mode set to {self.mode.value}")
    
    def set_params(self, params: ProcessingParams):
        """
        Update preprocessing parameters.
        
        Args:
            params: New parameters
        """
        self.params = params
        logger.info("Preprocessing parameters updated")
    
    def compare_methods(self, image: np.ndarray, 
                       depth_map: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Apply all preprocessing methods to an image for comparison.
        
        Args:
            image: Input image
            depth_map: Optional depth map
            
        Returns:
            Dictionary mapping mode names to processed images
        """
        results = {}
        original_mode = self.mode
        
        for mode in PreprocessingMode:
            self.mode = mode
            try:
                processed, _ = self.preprocess(image, depth_map)
                results[mode.value] = processed
            except Exception as e:
                logger.error(f"Error processing with mode {mode.value}: {str(e)}")
                results[mode.value] = image.copy()  # Fallback to original
        
        # Restore original mode
        self.mode = original_mode
        
        return results

```