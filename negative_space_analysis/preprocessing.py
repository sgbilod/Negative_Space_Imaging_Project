#!/usr/bin/env python
"""
Data Preprocessing Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides data preprocessing utilities for negative space analysis.
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing options."""
    normalize: bool = True
    denoise: bool = True
    contrast_enhance: bool = True
    edge_enhance: bool = False
    resize_target: Optional[Tuple[int, int]] = None
    denoise_strength: float = 10.0
    contrast_clip_limit: float = 3.0
    edge_kernel_size: int = 3


class ImagePreprocessor:
    """Handles image preprocessing for negative space analysis."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize preprocessor with configuration."""
        self.config = config or PreprocessingConfig()
    
    def preprocess(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Preprocess image for analysis.
        
        Args:
            image: Input image
            mask: Optional mask for region of interest
            
        Returns:
            Preprocessed image
        """
        # Convert to float32
        processed = image.astype(np.float32)
        
        # Apply mask if provided
        if mask is not None:
            processed *= mask
        
        # Normalize
        if self.config.normalize:
            processed = self._normalize(processed)
        
        # Resize if needed
        if self.config.resize_target:
            processed = self._resize(processed)
        
        # Denoise
        if self.config.denoise:
            processed = self._denoise(processed)
        
        # Enhance contrast
        if self.config.contrast_enhance:
            processed = self._enhance_contrast(processed)
        
        # Edge enhancement
        if self.config.edge_enhance:
            processed = self._enhance_edges(processed)
        
        return processed
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0,1] range."""
        image_min = np.min(image)
        image_max = np.max(image)
        
        if image_max > image_min:
            return (image - image_min) / (image_max - image_min)
        return image
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        if self.config.resize_target:
            return cv2.resize(
                image,
                self.config.resize_target,
                interpolation=cv2.INTER_AREA
            )
        return image
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising."""
        uint8_image = (image * 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(
            uint8_image,
            None,
            h=self.config.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised.astype(np.float32) / 255
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        uint8_image = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.contrast_clip_limit,
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(uint8_image)
        return enhanced.astype(np.float32) / 255
    
    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges using unsharp masking."""
        kernel_size = self.config.edge_kernel_size
        blurred = cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            0
        )
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


class BatchPreprocessor:
    """Handles batch preprocessing of images."""
    
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        batch_size: int = 32
    ):
        """Initialize batch preprocessor."""
        self.preprocessor = ImagePreprocessor(config)
        self.batch_size = batch_size
    
    def preprocess_batch(
        self,
        images: Union[List[np.ndarray], np.ndarray],
        masks: Optional[Union[List[np.ndarray], np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images or 4D array (batch, height, width, channels)
            masks: Optional list of masks or 4D array
            
        Returns:
            List of preprocessed images
        """
        if isinstance(images, np.ndarray) and images.ndim == 4:
            images = list(images)
        
        if masks is not None:
            if isinstance(masks, np.ndarray) and masks.ndim == 4:
                masks = list(masks)
        else:
            masks = [None] * len(images)
        
        processed_images = []
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i+self.batch_size]
            batch_masks = masks[i:i+self.batch_size]
            
            # Process batch
            batch_processed = [
                self.preprocessor.preprocess(img, mask)
                for img, mask in zip(batch_images, batch_masks)
            ]
            processed_images.extend(batch_processed)
        
        return processed_images
