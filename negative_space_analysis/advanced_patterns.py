#!/usr/bin/env python
"""
Advanced Pattern Analysis Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides advanced pattern analysis capabilities including:
- Symmetry detection
- Fractal dimension analysis
- Adaptive thresholding
"""

import numpy as np
from skimage.feature import graycomatrix
from skimage.util import view_as_blocks
from typing import List, Tuple, Optional
from dataclasses import dataclass
import cv2


@dataclass
class SymmetryFeatures:
    """Features related to symmetry analysis."""
    bilateral_score: float
    radial_score: float
    axis_angle: float
    center_point: Tuple[int, int]
    confidence: float


@dataclass
class FractalFeatures:
    """Features related to fractal analysis."""
    box_dimension: float
    lacunarity: float
    hurst_exponent: float
    multifractal_spectrum: np.ndarray


class SymmetryDetector:
    """Detects various types of symmetry in images."""
    
    def __init__(
        self,
        angular_resolution: float = 1.0,
        min_score: float = 0.7
    ):
        self.angular_resolution = angular_resolution
        self.min_score = min_score
    
    def detect_bilateral_symmetry(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[SymmetryFeatures]:
        """
        Detect bilateral symmetry axes in the image.
        
        Args:
            image: Input image
            mask: Optional mask for region of interest
            
        Returns:
            List of detected symmetry features
        """
        features = []
        angles = np.arange(0, 180, self.angular_resolution)
        
        # Convert to edge image for symmetry detection
        edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
        
        if mask is not None:
            edges *= mask.astype(np.uint8)
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        for angle in angles:
            # Create rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(edges, M, (w, h))
            
            # Compare left and right halves
            left = rotated[:, :w//2]
            right = cv2.flip(rotated[:, w//2:], 1)
            
            # Calculate symmetry score
            score = self._calculate_symmetry_score(left, right)
            
            if score > self.min_score:
                features.append(SymmetryFeatures(
                    bilateral_score=score,
                    radial_score=0.0,
                    axis_angle=angle,
                    center_point=center,
                    confidence=score
                ))
        
        return features
    
    def detect_radial_symmetry(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[SymmetryFeatures]:
        """
        Detect radial symmetry in the image.
        
        Args:
            image: Input image
            mask: Optional mask for region of interest
            
        Returns:
            List of detected symmetry features
        """
        features = []
        
        # Convert to edge image
        edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
        
        if mask is not None:
            edges *= mask.astype(np.uint8)
        
        h, w = image.shape[:2]
        
        # Test different center points
        for y in range(h//4, 3*h//4, h//8):
            for x in range(w//4, 3*w//4, w//8):
                center = (x, y)
                score = self._calculate_radial_symmetry(edges, center)
                
                if score > self.min_score:
                    features.append(SymmetryFeatures(
                        bilateral_score=0.0,
                        radial_score=score,
                        axis_angle=0.0,
                        center_point=center,
                        confidence=score
                    ))
        
        return features
    
    def _calculate_symmetry_score(
        self,
        region1: np.ndarray,
        region2: np.ndarray
    ) -> float:
        """Calculate symmetry score between two regions."""
        if region1.shape != region2.shape:
            min_width = min(region1.shape[1], region2.shape[1])
            region1 = region1[:, :min_width]
            region2 = region2[:, :min_width]
        
        # Using normalized cross-correlation
        return np.corrcoef(region1.ravel(), region2.ravel())[0, 1]
    
    def _calculate_radial_symmetry(
        self,
        edges: np.ndarray,
        center: Tuple[int, int]
    ) -> float:
        """Calculate radial symmetry score for a given center point."""
        y, x = np.ogrid[:edges.shape[0], :edges.shape[1]]
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        # Calculate gradient direction
        dx = cv2.Sobel(edges, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(edges, cv2.CV_32F, 0, 1)
        
        # Calculate angle between gradient and radius vector
        gradient_angle = np.arctan2(dy, dx)
        radius_angle = np.arctan2(y - center[1], x - center[0])
        
        # Score based on alignment of gradients with radius vectors
        alignment = np.abs(np.cos(gradient_angle - radius_angle))
        
        return float(np.mean(alignment * (edges > 0)))


class FractalAnalyzer:
    """Analyzes fractal properties of negative spaces."""
    
    def __init__(self, box_sizes: Optional[List[int]] = None):
        self.box_sizes = box_sizes or [2, 4, 8, 16, 32, 64]
    
    def compute_fractal_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> FractalFeatures:
        """
        Compute fractal features of the image.
        
        Args:
            image: Input image
            mask: Optional mask for region of interest
            
        Returns:
            FractalFeatures object
        """
        if mask is not None:
            image = image * mask
        
        # Calculate box-counting dimension
        box_dim = self._box_counting_dimension(image)
        
        # Calculate lacunarity
        lacunarity = self._calculate_lacunarity(image)
        
        # Calculate Hurst exponent
        hurst = self._hurst_exponent(image)
        
        # Calculate multifractal spectrum
        spectrum = self._multifractal_spectrum(image)
        
        return FractalFeatures(
            box_dimension=box_dim,
            lacunarity=lacunarity,
            hurst_exponent=hurst,
            multifractal_spectrum=spectrum
        )
    
    def _box_counting_dimension(self, image: np.ndarray) -> float:
        """Calculate box-counting dimension."""
        counts = []
        for size in self.box_sizes:
            count = self._count_boxes(image, size)
            counts.append(count)
        
        # Fit line to log-log plot
        x = np.log(self.box_sizes)
        y = np.log(counts)
        coeffs = np.polyfit(x, y, 1)
        
        return -coeffs[0]
    
    def _count_boxes(self, image: np.ndarray, box_size: int) -> int:
        """Count number of boxes needed to cover the image."""
        boxes = view_as_blocks(image, (box_size, box_size))
        return np.sum(np.any(boxes > 0, axis=(2, 3)))
    
    def _calculate_lacunarity(self, image: np.ndarray) -> float:
        """Calculate lacunarity (measure of texture)."""
        glcm = graycomatrix(
            (image * 255).astype(np.uint8),
            [1],
            [0, 45, 90, 135],
            symmetric=True,
            normed=True
        )
        return float(np.var(glcm) / np.mean(glcm))
    
    def _hurst_exponent(self, image: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        pixels = image.ravel()
        cumsum = np.cumsum(pixels - np.mean(pixels))
        R = np.max(cumsum) - np.min(cumsum)
        S = np.std(pixels)
        if S == 0:
            return 0.5
        return np.log(R/S) / np.log(len(pixels))
    
    def _multifractal_spectrum(self, image: np.ndarray) -> np.ndarray:
        """Calculate multifractal spectrum."""
        q_values = np.linspace(-5, 5, 11)
        spectrum = []
        
        for q in q_values:
            dim = self._generalized_dimension(image, q)
            spectrum.append(dim)
        
        return np.array(spectrum)
    
    def _generalized_dimension(
        self,
        image: np.ndarray,
        q: float
    ) -> float:
        """Calculate generalized dimension for given q."""
        if q == 1:
            return self._box_counting_dimension(image)
        
        counts = []
        for size in self.box_sizes:
            measure = self._box_measure(image, size)
            if q == 0:
                count = np.sum(measure > 0)
            else:
                count = np.sum(measure**q)
            counts.append(count)
        
        x = np.log(self.box_sizes)
        y = np.log(counts)
        coeffs = np.polyfit(x, y, 1)
        
        return coeffs[0] / (q - 1)
    
    def _box_measure(
        self,
        image: np.ndarray,
        box_size: int
    ) -> np.ndarray:
        """Calculate box measures for multifractal analysis."""
        boxes = view_as_blocks(image, (box_size, box_size))
        return np.sum(boxes, axis=(2, 3)) / (box_size * box_size)
