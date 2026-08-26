#!/usr/bin/env python
"""
Advanced Contour Analysis Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements sophisticated contour analysis using differential geometry
and topological features.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import splprep, splev
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import cv2


@dataclass
class ContourFeatures:
    """Features extracted from contour analysis."""
    curvature_spectrum: np.ndarray
    convexity_defects: List[np.ndarray]
    shape_descriptors: Dict[str, float]
    topological_features: Dict[str, float]
    differential_invariants: np.ndarray
    symmetry_measures: Dict[str, float]


class ContourMorphologyAnalyzer:
    """Analyzes contour shapes using differential geometry."""
    
    def __init__(
        self,
        spline_smoothing: float = 0.1,
        curvature_window: int = 5,
        n_sample_points: int = 100
    ):
        self.spline_smoothing = spline_smoothing
        self.curvature_window = curvature_window
        self.n_sample_points = n_sample_points
    
    def analyze_contour(
        self,
        contour: np.ndarray,
        compute_all: bool = True
    ) -> ContourFeatures:
        """
        Perform comprehensive contour analysis.
        
        Args:
            contour: Input contour points
            compute_all: Whether to compute all features
            
        Returns:
            ContourFeatures object
        """
        # Ensure contour has enough points
        if len(contour) < 3:
            raise ValueError("Contour must have at least 3 points")
        
        # Normalize and smooth contour
        contour_smooth = self._smooth_contour(contour)
        
        # Extract features
        curvature = self._compute_curvature(contour_smooth)
        convexity = self._analyze_convexity(contour) if compute_all else []
        shape = self._compute_shape_descriptors(contour_smooth)
        topology = self._compute_topological_features(contour_smooth)
        invariants = self._compute_differential_invariants(contour_smooth)
        symmetry = self._analyze_symmetry(contour_smooth) if compute_all else {}
        
        return ContourFeatures(
            curvature_spectrum=curvature,
            convexity_defects=convexity,
            shape_descriptors=shape,
            topological_features=topology,
            differential_invariants=invariants,
            symmetry_measures=symmetry
        )
    
    def _smooth_contour(self, contour: np.ndarray) -> np.ndarray:
        """Smooth contour using spline interpolation."""
        # Convert to periodic spline
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]
        
        # Fit spline
        tck, _ = splprep([x, y], s=self.spline_smoothing, per=1)
        
        # Generate smooth points
        u = np.linspace(0, 1, self.n_sample_points)
        smooth_x, smooth_y = splev(u, tck)
        
        return np.column_stack((smooth_x, smooth_y))
    
    def _compute_curvature(self, points: np.ndarray) -> np.ndarray:
        """Compute curvature along the contour."""
        # Compute derivatives
        dx = np.gradient(points[:, 0])
        dy = np.gradient(points[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Compute curvature
        curvature = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
        
        return self._smooth_signal(curvature)
    
    def _analyze_convexity(self, contour: np.ndarray) -> List[np.ndarray]:
        """Analyze convexity defects in the contour."""
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        if defects is None:
            return []
        
        return [defect for defect in defects if defect[0, 3] > 1000]
    
    def _compute_shape_descriptors(
        self,
        points: np.ndarray
    ) -> Dict[str, float]:
        """Compute various shape descriptors."""
        # Fit ellipse
        if len(points) < 5:
            return {}
            
        ellipse = cv2.fitEllipse(points.astype(np.float32))
        
        # Calculate metrics
        area = cv2.contourArea(points.astype(np.float32))
        perimeter = cv2.arcLength(points.astype(np.float32), True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "circularity": float(circularity),
            "aspect_ratio": float(ellipse[1][1] / ellipse[1][0]) if ellipse[1][0] > 0 else 0,
            "orientation": float(ellipse[2])
        }
    
    def _compute_topological_features(
        self,
        points: np.ndarray
    ) -> Dict[str, float]:
        """Compute topological features of the contour."""
        # Create binary image from contour
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        width = int(x_max - x_min + 10)
        height = int(y_max - y_min + 10)
        
        if width <= 0 or height <= 0:
            return {}
        
        # Draw contour on binary image
        binary = np.zeros((height, width), dtype=np.uint8)
        shifted_points = points - [x_min - 5, y_min - 5]
        cv2.drawContours(binary, [shifted_points.astype(np.int32)], 0, 1, -1)
        
        # Compute Euler characteristic
        euler = cv2.countNonZero(binary) - cv2.countNonZero(cv2.Canny(binary, 100, 200))
        
        return {
            "euler_characteristic": float(euler),
            "genus": float(1 - euler/2)
        }
    
    def _compute_differential_invariants(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """Compute differential invariants of the curve."""
        # Compute first and second derivatives
        dx = np.gradient(points[:, 0])
        dy = np.gradient(points[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Compute speed and acceleration
        speed = np.sqrt(dx * dx + dy * dy)
        acc = np.sqrt(ddx * ddx + ddy * ddy)
        
        # Compute torsion (third derivative)
        dddx = np.gradient(ddx)
        dddy = np.gradient(ddy)
        
        return np.column_stack([speed, acc, dddx, dddy])
    
    def _analyze_symmetry(self, points: np.ndarray) -> Dict[str, float]:
        """Analyze various symmetry measures of the contour."""
        # Center the points
        centroid = points.mean(axis=0)
        centered = points - centroid
        
        # Compute principal components
        _, s, vh = np.linalg.svd(centered)
        
        # Compute symmetry scores
        reflection_score = self._compute_reflection_symmetry(centered, vh[0])
        rotation_score = self._compute_rotation_symmetry(centered)
        
        return {
            "reflection_symmetry": float(reflection_score),
            "rotation_symmetry": float(rotation_score),
            "aspect_symmetry": float(s[0] / (s[1] + 1e-6))
        }
    
    def _compute_reflection_symmetry(
        self,
        points: np.ndarray,
        axis: np.ndarray
    ) -> float:
        """Compute reflection symmetry score."""
        # Project points onto symmetry axis
        projected = np.outer(points.dot(axis), axis)
        reflected = 2 * projected - points
        
        # Compute symmetric pairs
        distances = np.linalg.norm(points[:, None] - reflected, axis=2)
        min_distances = distances.min(axis=1)
        
        return float(1 / (1 + np.mean(min_distances)))
    
    def _compute_rotation_symmetry(self, points: np.ndarray) -> float:
        """Compute rotational symmetry score."""
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        scores = []
        
        for angle in angles:
            # Rotate points
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s], [s, c]])
            rotated = points.dot(rotation)
            
            # Compute matching score
            distances = np.linalg.norm(points[:, None] - rotated, axis=2)
            score = 1 / (1 + distances.min(axis=1).mean())
            scores.append(score)
        
        return float(np.max(scores))
    
    def _smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Smooth 1D signal using moving average."""
        window = np.ones(self.curvature_window) / self.curvature_window
        return np.convolve(signal, window, mode='same')
