"""
Spatial Signature Generator

This module extracts unique spatial signatures from negative space between objects.
It serves as the foundation for various revenue-generating services in the project.
"""

import hashlib
import numpy as np
from typing import List, Tuple, Dict, Union, Optional


class SpatialSignatureGenerator:
    """
    Generates cryptographically secure signatures from spatial coordinate data
    representing negative space between objects.
    """
    
    def __init__(self, 
                 hash_algorithm: str = 'sha256', 
                 normalization: bool = True,
                 use_relative_positions: bool = True):
        """
        Initialize the signature generator with configurable parameters.
        
        Args:
            hash_algorithm: Hashing algorithm to use ('sha256', 'sha512', etc.)
            normalization: Whether to normalize coordinate values
            use_relative_positions: Use relative positions instead of absolute
        """
        self.hash_algorithm = hash_algorithm
        self.normalization = normalization
        self.use_relative_positions = use_relative_positions
    
    def generate(self, coordinates: List[List[float]]) -> str:
        """
        Generate a spatial signature from a list of 3D coordinates.
        
        Args:
            coordinates: List of [x, y, z] coordinates representing points in 3D space
            
        Returns:
            A unique string signature derived from the spatial relationships
        """
        if not coordinates or len(coordinates) < 2:
            raise ValueError("At least two coordinates are required to generate a signature")
        
        processed_coords = self._preprocess_coordinates(coordinates)
        
        # Calculate relational features
        features = self._extract_spatial_features(processed_coords)
        
        # Create a deterministic string representation
        feature_str = self._serialize_features(features)
        
        # Generate the hash
        if self.hash_algorithm == 'sha256':
            signature = hashlib.sha256(feature_str.encode()).hexdigest()
        elif self.hash_algorithm == 'sha512':
            signature = hashlib.sha512(feature_str.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")
            
        return signature
    
    def _preprocess_coordinates(self, coordinates: List[List[float]]) -> np.ndarray:
        """
        Preprocess the input coordinates (normalize, filter, etc.).
        """
        coords_array = np.array(coordinates, dtype=np.float64)
        
        if self.normalization:
            # Normalize to range [0, 1]
            mins = coords_array.min(axis=0)
            maxs = coords_array.max(axis=0)
            ranges = maxs - mins
            # Avoid division by zero
            ranges[ranges == 0] = 1.0
            coords_array = (coords_array - mins) / ranges
            
        return coords_array
    
    def _extract_spatial_features(self, coordinates: np.ndarray) -> Dict[str, float]:
        """
        Extract various spatial features from the coordinates.
        """
        features = {}
        
        # Basic statistical features
        features['centroid'] = coordinates.mean(axis=0).tolist()
        
        # Calculate pairwise distances
        n_points = coordinates.shape[0]
        distances = []
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                distances.append(dist)
        
        features['mean_distance'] = np.mean(distances)
        features['max_distance'] = np.max(distances)
        features['min_distance'] = np.min(distances)
        features['std_distance'] = np.std(distances)
        
        # Calculate volume and area of the bounding box
        min_coords = coordinates.min(axis=0)
        max_coords = coordinates.max(axis=0)
        dimensions = max_coords - min_coords
        features['bounding_box_volume'] = np.prod(dimensions)
        
        # Calculate more complex features
        if n_points >= 3:
            # Calculate angles between triplets of points
            angles = []
            for i in range(n_points):
                for j in range(n_points):
                    if i == j:
                        continue
                    for k in range(n_points):
                        if k == i or k == j:
                            continue
                        vec1 = coordinates[j] - coordinates[i]
                        vec2 = coordinates[k] - coordinates[i]
                        # Normalize vectors
                        vec1_norm = np.linalg.norm(vec1)
                        vec2_norm = np.linalg.norm(vec2)
                        if vec1_norm > 0 and vec2_norm > 0:
                            cosine = np.dot(vec1, vec2) / (vec1_norm * vec2_norm)
                            # Clip to handle floating point errors
                            cosine = np.clip(cosine, -1.0, 1.0)
                            angle = np.arccos(cosine)
                            angles.append(angle)
            
            if angles:
                features['mean_angle'] = np.mean(angles)
                features['std_angle'] = np.std(angles)
        
        return features
    
    def _serialize_features(self, features: Dict[str, Union[float, List[float]]]) -> str:
        """
        Create a deterministic string representation of the features.
        """
        # Sort keys to ensure deterministic output
        sorted_keys = sorted(features.keys())
        
        # Create a list of "key:value" strings
        serialized = []
        for key in sorted_keys:
            value = features[key]
            if isinstance(value, list):
                # Convert lists to strings with fixed precision
                value_str = ','.join([f"{v:.10f}" for v in value])
                serialized.append(f"{key}:[{value_str}]")
            else:
                # Format floats with fixed precision
                serialized.append(f"{key}:{value:.10f}")
        
        # Join with a deterministic separator
        return "|".join(serialized)
