"""
Digital Twin Asset Verification - Phase 1 Implementation

This module implements the "Digital Twin Lifecycle with AR Auditing" concept for the 
Spatial-Temporal Proof of Existence Protocol. It creates a digital twin of an asset's 
negative space signature and provides AR-based auditing capabilities.
"""

import hashlib
import time
import json
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

from ...negative_mapping.spatial_signature_generator import SpatialSignatureGenerator


class DigitalTwinManager:
    """
    Manages digital twins of assets based on their negative space signatures.
    """
    
    def __init__(self):
        """
        Initialize the Digital Twin Manager.
        """
        self.signature_generator = SpatialSignatureGenerator(hash_algorithm='sha512')
        self.twins_db = {}  # In-memory storage for demo purposes
    
    def create_digital_twin(self, 
                          asset_data: Dict[str, Any],
                          spatial_signature: Union[str, List[List[float]]],
                          gps_coordinates: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Create a new digital twin for an asset.
        
        Args:
            asset_data: Metadata about the asset (name, type, owner, etc.)
            spatial_signature: Either a pre-computed signature string or raw coordinates
            gps_coordinates: Optional GPS coordinates where the asset was registered
            
        Returns:
            Digital twin data including the unique asset ID
        """
        # Generate asset ID
        asset_id = str(uuid.uuid4())
        
        # Get current timestamp
        timestamp = time.time()
        timestamp_formatted = datetime.fromtimestamp(timestamp).isoformat()
        
        # Generate signature if raw coordinates were provided
        if isinstance(spatial_signature, list):
            signature = self.signature_generator.generate(spatial_signature)
            coordinates = spatial_signature
        else:
            signature = spatial_signature
            coordinates = None
            
        # Create the digital twin record
        twin = {
            "asset_id": asset_id,
            "asset_data": asset_data,
            "created_at": timestamp_formatted,
            "signature": signature,
            "signature_hash": hashlib.sha256(signature.encode()).hexdigest(),
            "lifecycle": [
                {
                    "event_type": "creation",
                    "timestamp": timestamp_formatted,
                    "details": "Digital twin created"
                }
            ],
            "verification_status": "verified"
        }
        
        # Add coordinates if available
        if coordinates:
            twin["coordinates"] = coordinates
            
        # Add GPS data if available
        if gps_coordinates:
            twin["location"] = {
                "latitude": gps_coordinates[0],
                "longitude": gps_coordinates[1],
                "recorded_at": timestamp_formatted
            }
            
        # Store the twin
        self.twins_db[asset_id] = twin
        
        return twin
    
    def verify_asset(self,
                    asset_id: str,
                    current_signature: Union[str, List[List[float]]],
                    current_gps: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Verify an asset against its digital twin.
        
        Args:
            asset_id: The ID of the asset to verify
            current_signature: Current spatial signature or coordinates
            current_gps: Current GPS coordinates
            
        Returns:
            Verification result with match details
        """
        # Get the digital twin
        twin = self.twins_db.get(asset_id)
        if not twin:
            return {
                "verified": False,
                "reason": "unknown_asset",
                "details": f"No digital twin found for asset ID: {asset_id}"
            }
            
        # Generate signature if raw coordinates were provided
        if isinstance(current_signature, list):
            current_sig = self.signature_generator.generate(current_signature)
            current_coords = current_signature
        else:
            current_sig = current_signature
            current_coords = None
            
        # Get current timestamp
        timestamp = time.time()
        timestamp_formatted = datetime.fromtimestamp(timestamp).isoformat()
        
        # Compare signatures
        original_sig = twin["signature"]
        
        # For simplicity in Phase 1, we just check if signatures match exactly
        # In a real implementation, we would compute a similarity score
        is_match = (original_sig == current_sig)
        
        # Calculate similarity metrics if we have coordinates
        similarity_metrics = None
        if "coordinates" in twin and current_coords:
            similarity_metrics = self._calculate_similarity(twin["coordinates"], current_coords)
            
        # Create verification record
        verification = {
            "verified": is_match,
            "timestamp": timestamp_formatted,
            "original_signature_hash": twin["signature_hash"],
            "current_signature_hash": hashlib.sha256(current_sig.encode()).hexdigest()
        }
        
        # Add similarity metrics if available
        if similarity_metrics:
            verification["similarity_metrics"] = similarity_metrics
            
        # Add GPS data if available
        if current_gps:
            verification["current_location"] = {
                "latitude": current_gps[0],
                "longitude": current_gps[1]
            }
            
        # Update the twin's lifecycle
        twin["lifecycle"].append({
            "event_type": "verification",
            "timestamp": timestamp_formatted,
            "details": "Asset verified" if is_match else "Verification failed",
            "verification_data": verification
        })
        
        # Update verification status
        twin["verification_status"] = "verified" if is_match else "failed_verification"
        twin["last_verified_at"] = timestamp_formatted
        
        # Return verification result
        result = verification.copy()
        result["asset_id"] = asset_id
        result["asset_data"] = twin["asset_data"]
        
        return result
    
    def _calculate_similarity(self, original_coords: List[List[float]], current_coords: List[List[float]]) -> Dict[str, float]:
        """
        Calculate similarity metrics between original and current coordinates.
        """
        import numpy as np
        
        # Convert to numpy arrays
        orig = np.array(original_coords)
        curr = np.array(current_coords)
        
        # Basic checks
        if len(orig) != len(curr):
            return {
                "point_count_match": False,
                "original_points": len(orig),
                "current_points": len(curr)
            }
            
        # Calculate Euclidean distances between corresponding points
        distances = np.sqrt(np.sum((orig - curr) ** 2, axis=1))
        
        # Calculate metrics
        metrics = {
            "point_count_match": True,
            "mean_distance": float(np.mean(distances)),
            "max_distance": float(np.max(distances)),
            "min_distance": float(np.min(distances)),
            "std_distance": float(np.std(distances))
        }
        
        # Overall similarity score (0-100%)
        # Lower distances mean higher similarity
        max_allowed_distance = 0.1  # Threshold for max distance
        similarity_score = 100 * max(0, 1 - (metrics["mean_distance"] / max_allowed_distance))
        metrics["similarity_score"] = min(100, similarity_score)
        
        return metrics


class DigitalTwinAPI:
    """
    API for the Digital Twin Asset Verification service.
    This is a skeleton implementation for Phase 1.
    """
    
    def __init__(self):
        """Initialize the Digital Twin API."""
        self.twin_manager = DigitalTwinManager()
        
    def register_asset(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new asset and create its digital twin.
        
        Args:
            request_data: Dictionary containing asset data and spatial signature
            
        Returns:
            Digital twin data
        """
        # Extract request parameters
        asset_data = request_data.get("asset_data", {})
        spatial_signature = request_data.get("spatial_signature")
        gps_coordinates = request_data.get("gps_coordinates")
        
        # Validate inputs
        if not asset_data:
            return {"error": "Missing required parameter: asset_data"}
        if not spatial_signature:
            return {"error": "Missing required parameter: spatial_signature"}
            
        # Create digital twin
        try:
            twin = self.twin_manager.create_digital_twin(
                asset_data=asset_data,
                spatial_signature=spatial_signature,
                gps_coordinates=gps_coordinates
            )
            
            return twin
        except Exception as e:
            return {"error": f"Digital twin creation failed: {str(e)}"}
            
    def verify_asset(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify an asset against its digital twin.
        
        Args:
            request_data: Dictionary containing asset_id and current signature
            
        Returns:
            Verification result
        """
        # Extract request parameters
        asset_id = request_data.get("asset_id")
        current_signature = request_data.get("current_signature")
        current_gps = request_data.get("current_gps")
        
        # Validate inputs
        if not asset_id:
            return {"error": "Missing required parameter: asset_id"}
        if not current_signature:
            return {"error": "Missing required parameter: current_signature"}
            
        # Verify asset
        try:
            result = self.twin_manager.verify_asset(
                asset_id=asset_id,
                current_signature=current_signature,
                current_gps=current_gps
            )
            
            return result
        except Exception as e:
            return {"error": f"Asset verification failed: {str(e)}"}
            
    def get_asset_history(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the lifecycle history of an asset.
        
        Args:
            request_data: Dictionary containing asset_id
            
        Returns:
            Asset lifecycle history
        """
        # Extract request parameters
        asset_id = request_data.get("asset_id")
        
        # Validate inputs
        if not asset_id:
            return {"error": "Missing required parameter: asset_id"}
            
        # Get the digital twin
        twin = self.twin_manager.twins_db.get(asset_id)
        if not twin:
            return {"error": f"No digital twin found for asset ID: {asset_id}"}
            
        # Return lifecycle history
        return {
            "asset_id": asset_id,
            "asset_data": twin["asset_data"],
            "created_at": twin["created_at"],
            "verification_status": twin["verification_status"],
            "lifecycle": twin["lifecycle"]
        }
