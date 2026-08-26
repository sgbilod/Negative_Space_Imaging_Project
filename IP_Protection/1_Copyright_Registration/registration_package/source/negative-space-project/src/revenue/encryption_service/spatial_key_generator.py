"""
Spatial Key Generator - Phase 1 Implementation

This module implements the "Living Keys" concept for the Spatial Encryption Key Generation Service.
It generates quantum-resistant encryption keys from negative space signatures that are
dynamically validated against the physical environment.
"""

import hashlib
import base64
import time
import secrets
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

from ...negative_mapping.spatial_signature_generator import SpatialSignatureGenerator


class SpatialKeyGenerator:
    """
    Generates dynamic, geofenced encryption keys based on negative space signatures.
    These "Living Keys" are only valid when the physical environment matches the initial AR capture.
    """
    
    def __init__(self, 
                 key_length: int = 256, 
                 hash_algorithm: str = 'sha256',
                 time_sensitivity: float = 1.0,
                 geo_precision: float = 10.0):
        """
        Initialize the spatial key generator.
        
        Args:
            key_length: Length of the generated key in bits (256, 384, 512)
            hash_algorithm: Hash algorithm to use for key derivation
            time_sensitivity: How sensitive the key is to time changes (1.0 = normal)
            geo_precision: Precision of geofencing in meters
        """
        self.key_length = key_length
        self.hash_algorithm = hash_algorithm
        self.time_sensitivity = time_sensitivity
        self.geo_precision = geo_precision
        self.signature_generator = SpatialSignatureGenerator(hash_algorithm=hash_algorithm)
    
    def generate_key(self, 
                     spatial_signature: Union[str, List[List[float]]],
                     gps_coordinates: Optional[Tuple[float, float]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a spatial encryption key from a negative space signature.
        
        Args:
            spatial_signature: Either a pre-computed signature string or raw coordinates
            gps_coordinates: Optional GPS coordinates (latitude, longitude) for geofencing
            metadata: Optional metadata to include in the key
            
        Returns:
            A dictionary containing the key and its metadata
        """
        # Generate signature if raw coordinates were provided
        if isinstance(spatial_signature, list):
            signature = self.signature_generator.generate(spatial_signature)
        else:
            signature = spatial_signature
            
        # Get current timestamp
        timestamp = time.time()
        timestamp_formatted = datetime.fromtimestamp(timestamp).isoformat()
        
        # Generate a salt for additional entropy
        salt = secrets.token_hex(16)
        
        # Combine all components to create the key material
        key_components = [
            signature,
            str(timestamp),
            salt
        ]
        
        # Add GPS coordinates if provided
        if gps_coordinates:
            key_components.append(f"{gps_coordinates[0]:.6f},{gps_coordinates[1]:.6f}")
        
        # Create a deterministic string by joining components
        key_material = "|".join(key_components)
        
        # Generate the final key using the appropriate hash function
        if self.hash_algorithm == 'sha256':
            hash_func = hashlib.sha256
        elif self.hash_algorithm == 'sha512':
            hash_func = hashlib.sha512
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")
        
        key_hash = hash_func(key_material.encode()).digest()
        
        # Truncate or extend to desired key length
        key_bytes = key_hash[:self.key_length // 8]
        
        # Encode as base64 for easier handling
        key_base64 = base64.b64encode(key_bytes).decode('utf-8')
        
        # Prepare the result dictionary
        result = {
            "key": key_base64,
            "created_at": timestamp_formatted,
            "expires_at": (datetime.fromtimestamp(timestamp) + timedelta(hours=24)).isoformat(),
            "signature_hash": hashlib.sha256(signature.encode()).hexdigest(),
            "algorithm": self.hash_algorithm,
            "key_length": self.key_length,
            "version": "1.0"
        }
        
        # Add GPS data if provided
        if gps_coordinates:
            result["geo_fence"] = {
                "latitude": gps_coordinates[0],
                "longitude": gps_coordinates[1],
                "radius_meters": self.geo_precision
            }
            
        # Add any additional metadata
        if metadata:
            result["metadata"] = metadata
            
        return result
        
    def validate_key(self, 
                     key_data: Dict[str, Any], 
                     current_signature: Union[str, List[List[float]]],
                     current_gps: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Validate a spatial key against the current environment.
        
        Args:
            key_data: The key data returned from generate_key
            current_signature: Current spatial signature or coordinates
            current_gps: Current GPS coordinates
            
        Returns:
            Validation result with status and details
        """
        # Generate signature if raw coordinates were provided
        if isinstance(current_signature, list):
            current_sig = self.signature_generator.generate(current_signature)
        else:
            current_sig = current_signature
            
        # Get current timestamp
        current_time = time.time()
        
        # Check if the key is expired
        key_created_at = datetime.fromisoformat(key_data["created_at"])
        key_expires_at = datetime.fromisoformat(key_data["expires_at"])
        current_datetime = datetime.fromtimestamp(current_time)
        
        if current_datetime > key_expires_at:
            return {
                "valid": False,
                "reason": "expired",
                "details": f"Key expired at {key_expires_at.isoformat()}"
            }
            
        # Check the spatial signature similarity
        original_sig_hash = key_data["signature_hash"]
        current_sig_hash = hashlib.sha256(current_sig.encode()).hexdigest()
        
        if original_sig_hash != current_sig_hash:
            return {
                "valid": False,
                "reason": "signature_mismatch",
                "details": "The spatial signature has changed"
            }
            
        # Check geofence if applicable
        if "geo_fence" in key_data and current_gps:
            geo_fence = key_data["geo_fence"]
            original_lat = geo_fence["latitude"]
            original_lon = geo_fence["longitude"]
            radius = geo_fence["radius_meters"]
            
            # Calculate approximate distance using Haversine formula
            from math import sin, cos, sqrt, atan2, radians
            
            R = 6371000  # Earth radius in meters
            lat1, lon1 = radians(original_lat), radians(original_lon)
            lat2, lon2 = radians(current_gps[0]), radians(current_gps[1])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = R * c
            
            if distance > radius:
                return {
                    "valid": False,
                    "reason": "geofence_violation",
                    "details": f"Location is {distance:.1f}m outside the allowed radius of {radius}m"
                }
                
        # If we got here, the key is valid
        return {
            "valid": True,
            "details": "Key is valid and active",
            "validation_time": current_datetime.isoformat()
        }


class KeyManagementAPI:
    """
    RESTful API for managing spatial encryption keys.
    This is a skeleton implementation for Phase 1.
    """
    
    def __init__(self):
        """Initialize the Key Management API."""
        self.key_generator = SpatialKeyGenerator()
        self.keys_db = {}  # In-memory storage for demo purposes
        
    def generate_key(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a new spatial encryption key.
        
        Args:
            request_data: Dictionary containing spatial signature, GPS, and metadata
            
        Returns:
            Generated key and metadata
        """
        # Extract request parameters
        spatial_signature = request_data.get("spatial_signature")
        gps_coordinates = request_data.get("gps_coordinates")
        metadata = request_data.get("metadata", {})
        
        # Validate inputs
        if not spatial_signature:
            return {"error": "Missing required parameter: spatial_signature"}
            
        # Generate the key
        try:
            key_data = self.key_generator.generate_key(
                spatial_signature=spatial_signature,
                gps_coordinates=gps_coordinates,
                metadata=metadata
            )
            
            # Store the key with a unique ID
            key_id = secrets.token_hex(8)
            self.keys_db[key_id] = key_data
            
            # Add the ID to the returned data
            result = key_data.copy()
            result["key_id"] = key_id
            
            return result
        except Exception as e:
            return {"error": f"Key generation failed: {str(e)}"}
            
    def validate_key(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a spatial encryption key against the current environment.
        
        Args:
            request_data: Dictionary containing key_id, current signature, and GPS
            
        Returns:
            Validation result
        """
        # Extract request parameters
        key_id = request_data.get("key_id")
        current_signature = request_data.get("current_signature")
        current_gps = request_data.get("current_gps")
        
        # Validate inputs
        if not key_id:
            return {"error": "Missing required parameter: key_id"}
        if not current_signature:
            return {"error": "Missing required parameter: current_signature"}
            
        # Get the stored key data
        key_data = self.keys_db.get(key_id)
        if not key_data:
            return {"error": f"Key not found: {key_id}"}
            
        # Validate the key
        try:
            result = self.key_generator.validate_key(
                key_data=key_data,
                current_signature=current_signature,
                current_gps=current_gps
            )
            
            return result
        except Exception as e:
            return {"error": f"Key validation failed: {str(e)}"}
