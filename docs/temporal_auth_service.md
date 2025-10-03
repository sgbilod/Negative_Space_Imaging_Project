# Documentation for temporal_auth_service.py

```python
"""
Temporal Authentication Service - Phase 1 Implementation

This module implements the "3-Factor AR Gesture Authentication" concept for the 
Temporal Authentication as a Service (TAaaS). It provides authentication tokens
that combine spatial signatures, AR gestures, and temporal elements.
"""

import hashlib
import time
import json
import jwt
import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

from ...negative_mapping.spatial_signature_generator import SpatialSignatureGenerator


class GestureRecognizer:
    """
    Recognizes and validates AR gestures in 3D space.
    """
    
    def __init__(self, 
                similarity_threshold: float = 0.75,
                time_sensitivity: float = 1.0):
        """
        Initialize the gesture recognizer.
        
        Args:
            similarity_threshold: Threshold for gesture similarity (0.0-1.0)
            time_sensitivity: How sensitive authentication is to timing variations
        """
        self.similarity_threshold = similarity_threshold
        self.time_sensitivity = time_sensitivity
        
    def register_gesture(self, gesture_path: List[List[float]]) -> Dict[str, Any]:
        """
        Register a new gesture pattern.
        
        Args:
            gesture_path: List of 3D coordinates representing the gesture path
            
        Returns:
            Gesture registration data
        """
        # Validate input
        if len(gesture_path) < 3:
            raise ValueError("Gesture must have at least 3 points")
            
        # Normalize the gesture path
        normalized_path = self._normalize_path(gesture_path)
        
        # Calculate gesture features
        features = self._extract_gesture_features(normalized_path)
        
        # Generate a unique ID for this gesture
        gesture_id = str(uuid.uuid4())
        
        # Create the gesture pattern
        pattern = {
            "gesture_id": gesture_id,
            "created_at": datetime.now().isoformat(),
            "normalized_path": normalized_path,
            "features": features,
            "point_count": len(gesture_path)
        }
        
        return pattern
        
    def verify_gesture(self, 
                      registered_pattern: Dict[str, Any], 
                      input_path: List[List[float]]) -> Dict[str, Any]:
        """
        Verify an input gesture against a registered pattern.
        
        Args:
            registered_pattern: The registered gesture pattern
            input_path: The input gesture path to verify
            
        Returns:
            Verification result with similarity score
        """
        # Validate input
        if len(input_path) < 3:
            return {
                "verified": False,
                "reason": "insufficient_points",
                "details": "Gesture must have at least 3 points"
            }
            
        # Normalize the input path
        normalized_input = self._normalize_path(input_path)
        
        # Extract features from the input
        input_features = self._extract_gesture_features(normalized_input)
        
        # Compare the gestures
        similarity_score = self._calculate_similarity(
            registered_pattern["features"],
            input_features,
            registered_pattern["normalized_path"],
            normalized_input
        )
        
        # Determine if the gesture is verified
        is_verified = similarity_score >= self.similarity_threshold
        
        # Return verification result
        return {
            "verified": is_verified,
            "similarity_score": similarity_score,
            "threshold": self.similarity_threshold,
            "registered_points": registered_pattern["point_count"],
            "input_points": len(input_path)
        }
        
    def _normalize_path(self, path: List[List[float]]) -> List[List[float]]:
        """
        Normalize a gesture path to make it invariant to scale and position.
        """
        # Convert to numpy array
        path_array = np.array(path, dtype=np.float64)
        
        # Center the path around the origin
        centroid = np.mean(path_array, axis=0)
        centered = path_array - centroid
        
        # Scale to unit size
        max_distance = np.max(np.linalg.norm(centered, axis=1))
        if max_distance > 0:
            normalized = centered / max_distance
        else:
            normalized = centered
            
        # Convert back to list format
        return normalized.tolist()
        
    def _extract_gesture_features(self, path: List[List[float]]) -> Dict[str, Any]:
        """
        Extract features from a gesture path for comparison.
        """
        # Convert to numpy array
        path_array = np.array(path, dtype=np.float64)
        
        # Calculate distances between consecutive points
        diffs = np.diff(path_array, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        
        # Calculate angles between consecutive segments
        angles = []
        for i in range(len(diffs) - 1):
            vec1 = diffs[i]
            vec2 = diffs[i + 1]
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                dot_product = np.dot(vec1, vec2) / (norm1 * norm2)
                # Clip to handle numerical errors
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = np.arccos(dot_product)
                angles.append(float(angle))
                
        # Extract features
        features = {
            "point_count": len(path),
            "path_length": float(np.sum(distances)),
            "bounding_box": {
                "min": np.min(path_array, axis=0).tolist(),
                "max": np.max(path_array, axis=0).tolist()
            }
        }
        
        # Add angle statistics if available
        if angles:
            features["mean_angle"] = float(np.mean(angles))
            features["std_angle"] = float(np.std(angles))
            
        # Add curvature
        if len(path) > 2:
            try:
                # Calculate curvature at each point (excluding endpoints)
                curvature = []
                for i in range(1, len(path) - 1):
                    p0 = np.array(path[i-1])
                    p1 = np.array(path[i])
                    p2 = np.array(path[i+1])
                    
                    # Vectors between points
                    v1 = p1 - p0
                    v2 = p2 - p1
                    
                    # Cross product magnitude divided by product of lengths
                    cross = np.linalg.norm(np.cross(v1, v2))
                    product = np.linalg.norm(v1) * np.linalg.norm(v2)
                    
                    if product > 0:
                        curvature.append(cross / product)
                
                if curvature:
                    features["mean_curvature"] = float(np.mean(curvature))
                    features["max_curvature"] = float(np.max(curvature))
            except:
                # Skip curvature calculation if it fails
                pass
                
        return features
        
    def _calculate_similarity(self, 
                             pattern_features: Dict[str, Any], 
                             input_features: Dict[str, Any],
                             pattern_path: List[List[float]],
                             input_path: List[List[float]]) -> float:
        """
        Calculate similarity score between a registered pattern and input gesture.
        """
        # Different features have different weights in the similarity calculation
        weights = {
            "path_length": 0.2,
            "mean_angle": 0.3,
            "std_angle": 0.1,
            "mean_curvature": 0.2,
            "dtw_distance": 0.4  # Dynamic Time Warping for path similarity
        }
        
        scores = {}
        
        # Compare path length
        if "path_length" in pattern_features and "path_length" in input_features:
            max_length = max(pattern_features["path_length"], input_features["path_length"])
            if max_length > 0:
                length_diff = abs(pattern_features["path_length"] - input_features["path_length"])
                scores["path_length"] = 1.0 - min(1.0, length_diff / max_length)
            else:
                scores["path_length"] = 1.0
                
        # Compare angles
        if "mean_angle" in pattern_features and "mean_angle" in input_features:
            angle_diff = abs(pattern_features["mean_angle"] - input_features["mean_angle"])
            # Normalize by pi (maximum possible angle difference)
            scores["mean_angle"] = 1.0 - min(1.0, angle_diff / np.pi)
            
        if "std_angle" in pattern_features and "std_angle" in input_features:
            std_diff = abs(pattern_features["std_angle"] - input_features["std_angle"])
            # Normalize by pi/2 (reasonable maximum for angle standard deviation)
            scores["std_angle"] = 1.0 - min(1.0, std_diff / (np.pi / 2))
            
        # Compare curvature
        if "mean_curvature" in pattern_features and "mean_curvature" in input_features:
            curv_diff = abs(pattern_features["mean_curvature"] - input_features["mean_curvature"])
            # Normalize by 2.0 (reasonable maximum for curvature difference)
            scores["mean_curvature"] = 1.0 - min(1.0, curv_diff / 2.0)
            
        # Calculate Dynamic Time Warping distance
        try:
            from scipy.spatial.distance import directed_hausdorff
            
            # Convert to numpy arrays
            pattern_array = np.array(pattern_path)
            input_array = np.array(input_path)
            
            # Use Hausdorff distance as a simple approximation of DTW
            forward_dist, _, _ = directed_hausdorff(pattern_array, input_array)
            backward_dist, _, _ = directed_hausdorff(input_array, pattern_array)
            hausdorff_dist = max(forward_dist, backward_dist)
            
            # Normalize by a reasonable maximum distance (1.0 after normalization)
            scores["dtw_distance"] = 1.0 - min(1.0, hausdorff_dist)
        except ImportError:
            # Fall back to a simpler comparison if scipy is not available
            scores["dtw_distance"] = 0.5  # Neutral score
            
        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        
        for feature, score in scores.items():
            if feature in weights:
                weight = weights[feature]
                weighted_sum += score * weight
                total_weight += weight
                
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0  # No valid features to compare


class TemporalAuthService:
    """
    Service for temporal-based authentication using AR gestures and spatial signatures.
    """
    
    def __init__(self,
                token_expiry_minutes: int = 5,
                jwt_secret: str = None):
        """
        Initialize the authentication service.
        
        Args:
            token_expiry_minutes: How long tokens are valid, in minutes
            jwt_secret: Secret for JWT token signing (generates random if None)
        """
        self.token_expiry_minutes = token_expiry_minutes
        self.jwt_secret = jwt_secret or secrets.token_hex(32)
        self.signature_generator = SpatialSignatureGenerator()
        self.gesture_recognizer = GestureRecognizer()
        
        # In-memory storage for demo purposes
        self.registered_users = {}
        self.registered_gestures = {}
        
    def register_user(self, 
                     user_id: str, 
                     user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new user.
        
        Args:
            user_id: Unique identifier for the user
            user_data: User metadata
            
        Returns:
            User registration data
        """
        # Create user record
        user = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "data": user_data,
            "gestures": []
        }
        
        # Store user
        self.registered_users[user_id] = user
        
        return user
        
    def register_gesture(self, 
                        user_id: str, 
                        gesture_path: List[List[float]],
                        gesture_name: str = "default") -> Dict[str, Any]:
        """
        Register a gesture for a user.
        
        Args:
            user_id: User ID
            gesture_path: List of 3D coordinates representing the gesture
            gesture_name: Name for this gesture
            
        Returns:
            Gesture registration data
        """
        # Check if user exists
        if user_id not in self.registered_users:
            raise ValueError(f"User not found: {user_id}")
            
        # Register the gesture
        gesture = self.gesture_recognizer.register_gesture(gesture_path)
        
        # Add metadata
        gesture["user_id"] = user_id
        gesture["name"] = gesture_name
        
        # Store the gesture
        gesture_id = gesture["gesture_id"]
        self.registered_gestures[gesture_id] = gesture
        
        # Add to user's gestures
        self.registered_users[user_id]["gestures"].append({
            "gesture_id": gesture_id,
            "name": gesture_name
        })
        
        return gesture
        
    def authenticate(self, 
                    user_id: str, 
                    spatial_signature: Union[str, List[List[float]]],
                    gesture_path: List[List[float]],
                    gesture_name: str = "default") -> Dict[str, Any]:
        """
        Authenticate a user with spatial signature and gesture.
        
        Args:
            user_id: User ID
            spatial_signature: Spatial signature or raw coordinates
            gesture_path: Gesture path for verification
            gesture_name: Name of the gesture to verify against
            
        Returns:
            Authentication result with token if successful
        """
        # Check if user exists
        user = self.registered_users.get(user_id)
        if not user:
            return {
                "authenticated": False,
                "reason": "user_not_found",
                "details": f"User not found: {user_id}"
            }
            
        # Find the specified gesture
        gesture_id = None
        for g in user["gestures"]:
            if g["name"] == gesture_name:
                gesture_id = g["gesture_id"]
                break
                
        if not gesture_id:
            return {
                "authenticated": False,
                "reason": "gesture_not_found",
                "details": f"Gesture '{gesture_name}' not registered for user"
            }
            
        # Get the registered gesture pattern
        pattern = self.registered_gestures.get(gesture_id)
        if not pattern:
            return {
                "authenticated": False,
                "reason": "gesture_not_found",
                "details": "Registered gesture pattern not found"
            }
            
        # Verify the gesture
        gesture_verification = self.gesture_recognizer.verify_gesture(pattern, gesture_path)
        
        if not gesture_verification["verified"]:
            return {
                "authenticated": False,
                "reason": "gesture_mismatch",
                "details": "Gesture verification failed",
                "verification_details": gesture_verification
            }
            
        # If we got here, the gesture is verified
        # Generate a signature if raw coordinates were provided
        if isinstance(spatial_signature, list):
            signature = self.signature_generator.generate(spatial_signature)
        else:
            signature = spatial_signature
            
        # Get current timestamp
        now = datetime.now()
        expiry = now + timedelta(minutes=self.token_expiry_minutes)
        
        # Create token payload
        payload = {
            "sub": user_id,
            "iat": int(now.timestamp()),
            "exp": int(expiry.timestamp()),
            "signature_hash": hashlib.sha256(signature.encode()).hexdigest(),
            "auth_type": "spatial_gesture",
            "gesture_id": gesture_id,
            "gesture_score": gesture_verification["similarity_score"]
        }
        
        # Generate JWT token
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        
        # Return authentication result
        return {
            "authenticated": True,
            "token": token,
            "expires_at": expiry.isoformat(),
            "user_id": user_id,
            "verification_details": gesture_verification
        }
        
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify an authentication token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Token verification result
        """
        try:
            # Decode and verify the token
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if user exists
            user_id = payload["sub"]
            if user_id not in self.registered_users:
                return {
                    "valid": False,
                    "reason": "user_not_found",
                    "details": f"User not found: {user_id}"
                }
                
            # Return verification result
            return {
                "valid": True,
                "user_id": user_id,
                "expires_at": datetime.fromtimestamp(payload["exp"]).isoformat(),
                "auth_type": payload.get("auth_type"),
                "payload": payload
            }
        except jwt.ExpiredSignatureError:
            return {
                "valid": False,
                "reason": "expired",
                "details": "Token has expired"
            }
        except jwt.InvalidTokenError:
            return {
                "valid": False,
                "reason": "invalid_token",
                "details": "Token is invalid"
            }


class TemporalAuthAPI:
    """
    API for the Temporal Authentication service.
    This is a skeleton implementation for Phase 1.
    """
    
    def __init__(self):
        """Initialize the Temporal Auth API."""
        self.auth_service = TemporalAuthService()
        
    def register_user(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new user.
        
        Args:
            request_data: Dictionary containing user_id and user_data
            
        Returns:
            User registration data
        """
        # Extract request parameters
        user_id = request_data.get("user_id")
        user_data = request_data.get("user_data", {})
        
        # Validate inputs
        if not user_id:
            return {"error": "Missing required parameter: user_id"}
            
        # Register user
        try:
            user = self.auth_service.register_user(
                user_id=user_id,
                user_data=user_data
            )
            
            return user
        except Exception as e:
            return {"error": f"User registration failed: {str(e)}"}
            
    def register_gesture(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a gesture for a user.
        
        Args:
            request_data: Dictionary containing user_id, gesture_path, and gesture_name
            
        Returns:
            Gesture registration data
        """
        # Extract request parameters
        user_id = request_data.get("user_id")
        gesture_path = request_data.get("gesture_path")
        gesture_name = request_data.get("gesture_name", "default")
        
        # Validate inputs
        if not user_id:
            return {"error": "Missing required parameter: user_id"}
        if not gesture_path:
            return {"error": "Missing required parameter: gesture_path"}
            
        # Register gesture
        try:
            gesture = self.auth_service.register_gesture(
                user_id=user_id,
                gesture_path=gesture_path,
                gesture_name=gesture_name
            )
            
            return gesture
        except Exception as e:
            return {"error": f"Gesture registration failed: {str(e)}"}
            
    def authenticate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Authenticate a user.
        
        Args:
            request_data: Dictionary containing authentication parameters
            
        Returns:
            Authentication result
        """
        # Extract request parameters
        user_id = request_data.get("user_id")
        spatial_signature = request_data.get("spatial_signature")
        gesture_path = request_data.get("gesture_path")
        gesture_name = request_data.get("gesture_name", "default")
        
        # Validate inputs
        if not user_id:
            return {"error": "Missing required parameter: user_id"}
        if not spatial_signature:
            return {"error": "Missing required parameter: spatial_signature"}
        if not gesture_path:
            return {"error": "Missing required parameter: gesture_path"}
            
        # Authenticate
        try:
            result = self.auth_service.authenticate(
                user_id=user_id,
                spatial_signature=spatial_signature,
                gesture_path=gesture_path,
                gesture_name=gesture_name
            )
            
            return result
        except Exception as e:
            return {"error": f"Authentication failed: {str(e)}"}
            
    def verify_token(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify an authentication token.
        
        Args:
            request_data: Dictionary containing token
            
        Returns:
            Token verification result
        """
        # Extract request parameters
        token = request_data.get("token")
        
        # Validate inputs
        if not token:
            return {"error": "Missing required parameter: token"}
            
        # Verify token
        try:
            result = self.auth_service.verify_token(token)
            
            return result
        except Exception as e:
            return {"error": f"Token verification failed: {str(e)}"}

```