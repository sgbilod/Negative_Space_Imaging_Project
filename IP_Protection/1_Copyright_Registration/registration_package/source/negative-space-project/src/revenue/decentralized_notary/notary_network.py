"""
Proof-of-View Decentralized Notary - Phase 1 Implementation

This module implements the "Proof-of-View Consensus Mechanism" for the 
Decentralized Time Notary Network. It creates a distributed system that 
requires physical verification of landmarks to participate as a notary node.
"""

import hashlib
import time
import json
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

from ...negative_mapping.spatial_signature_generator import SpatialSignatureGenerator


class LandmarkRegistry:
    """
    Registry of public landmarks that can be used for Proof-of-View validation.
    """
    
    def __init__(self):
        """Initialize the landmark registry."""
        self.landmarks = {}  # In-memory storage for demo purposes
        
    def register_landmark(self,
                         name: str,
                         description: str,
                         location: Dict[str, float],
                         spatial_signature: Union[str, List[List[float]]],
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Register a new public landmark.
        
        Args:
            name: Name of the landmark
            description: Description of the landmark
            location: Dictionary with latitude and longitude
            spatial_signature: Spatial signature or raw coordinates
            metadata: Additional metadata
            
        Returns:
            Landmark registration data
        """
        # Generate landmark ID
        landmark_id = str(uuid.uuid4())
        
        # Generate signature if raw coordinates were provided
        if isinstance(spatial_signature, list):
            signature_generator = SpatialSignatureGenerator()
            signature = signature_generator.generate(spatial_signature)
            coordinates = spatial_signature
        else:
            signature = spatial_signature
            coordinates = None
            
        # Create landmark record
        landmark = {
            "landmark_id": landmark_id,
            "name": name,
            "description": description,
            "location": location,
            "signature": signature,
            "signature_hash": hashlib.sha256(signature.encode()).hexdigest(),
            "created_at": datetime.now().isoformat(),
            "verified_by": 0,
            "active": True
        }
        
        # Add coordinates if available
        if coordinates:
            landmark["coordinates"] = coordinates
            
        # Add metadata if provided
        if metadata:
            landmark["metadata"] = metadata
            
        # Store the landmark
        self.landmarks[landmark_id] = landmark
        
        return landmark
        
    def get_landmark(self, landmark_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a landmark by ID.
        
        Args:
            landmark_id: ID of the landmark
            
        Returns:
            Landmark data or None if not found
        """
        return self.landmarks.get(landmark_id)
        
    def get_landmarks_near(self,
                         latitude: float,
                         longitude: float,
                         radius_km: float = 10.0) -> List[Dict[str, Any]]:
        """
        Get landmarks near a location.
        
        Args:
            latitude: Current latitude
            longitude: Current longitude
            radius_km: Search radius in kilometers
            
        Returns:
            List of landmarks within the radius
        """
        # Calculate nearby landmarks
        nearby = []
        
        for landmark in self.landmarks.values():
            # Calculate distance using Haversine formula
            from math import sin, cos, sqrt, atan2, radians
            
            R = 6371.0  # Earth radius in kilometers
            
            lat1 = radians(latitude)
            lon1 = radians(longitude)
            lat2 = radians(landmark["location"]["latitude"])
            lon2 = radians(landmark["location"]["longitude"])
            
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = R * c
            
            if distance <= radius_km:
                # Add distance to the landmark data
                landmark_with_distance = landmark.copy()
                landmark_with_distance["distance_km"] = distance
                nearby.append(landmark_with_distance)
                
        # Sort by distance
        nearby.sort(key=lambda x: x["distance_km"])
        
        return nearby


class ProofOfViewValidator:
    """
    Validates Proof-of-View submissions against registered landmarks.
    """
    
    def __init__(self, 
                match_threshold: float = 0.85,
                time_validity_hours: int = 24):
        """
        Initialize the validator.
        
        Args:
            match_threshold: Threshold for considering a proof valid (0.0-1.0)
            time_validity_hours: How long a proof is valid for
        """
        self.match_threshold = match_threshold
        self.time_validity_hours = time_validity_hours
        self.signature_generator = SpatialSignatureGenerator()
        
    def validate_proof(self, 
                      landmark: Dict[str, Any], 
                      proof_signature: Union[str, List[List[float]]]) -> Dict[str, Any]:
        """
        Validate a Proof-of-View submission.
        
        Args:
            landmark: Landmark data
            proof_signature: Submitted signature or raw coordinates
            
        Returns:
            Validation result
        """
        # Generate signature if raw coordinates were provided
        if isinstance(proof_signature, list):
            signature = self.signature_generator.generate(proof_signature)
        else:
            signature = proof_signature
            
        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        # Simple string comparison for Phase 1
        # In a real implementation, we would use more sophisticated comparison
        landmark_signature = landmark["signature"]
        
        if landmark_signature == signature:
            match_score = 1.0
        else:
            # Calculate simple similarity based on hash comparison
            # In reality, this would use more advanced signature comparison
            landmark_hash = landmark["signature_hash"]
            proof_hash = hashlib.sha256(signature.encode()).hexdigest()
            
            # Count matching characters in the hash
            matching = sum(1 for a, b in zip(landmark_hash, proof_hash) if a == b)
            match_score = matching / len(landmark_hash)
            
        # Determine if the proof is valid
        is_valid = match_score >= self.match_threshold
        
        # Create validation result
        result = {
            "valid": is_valid,
            "match_score": match_score,
            "threshold": self.match_threshold,
            "landmark_id": landmark["landmark_id"],
            "landmark_name": landmark["name"],
            "validated_at": timestamp,
            "expires_at": (datetime.fromisoformat(timestamp) + 
                           timedelta(hours=self.time_validity_hours)).isoformat()
        }
        
        return result


class NotaryNode:
    """
    A node in the decentralized notary network that can notarize documents.
    """
    
    def __init__(self, 
                node_id: str,
                owner_id: str,
                landmark_registry: LandmarkRegistry,
                pov_validator: ProofOfViewValidator):
        """
        Initialize a notary node.
        
        Args:
            node_id: Unique identifier for this node
            owner_id: ID of the node operator
            landmark_registry: Registry of landmarks
            pov_validator: Validator for Proof-of-View submissions
        """
        self.node_id = node_id
        self.owner_id = owner_id
        self.landmark_registry = landmark_registry
        self.pov_validator = pov_validator
        
        # Node state
        self.reputation_score = 0.0
        self.last_proof_time = None
        self.valid_proofs = []
        self.notarizations = []  # Documents notarized by this node
        
    def submit_proof_of_view(self, 
                           landmark_id: str, 
                           proof_signature: Union[str, List[List[float]]]) -> Dict[str, Any]:
        """
        Submit a Proof-of-View to increase node reputation.
        
        Args:
            landmark_id: ID of the landmark being verified
            proof_signature: Spatial signature or raw coordinates
            
        Returns:
            Submission result
        """
        # Get the landmark
        landmark = self.landmark_registry.get_landmark(landmark_id)
        if not landmark:
            return {
                "success": False,
                "reason": "landmark_not_found",
                "details": f"Landmark not found: {landmark_id}"
            }
            
        # Validate the proof
        validation = self.pov_validator.validate_proof(landmark, proof_signature)
        
        # Update node state if the proof is valid
        if validation["valid"]:
            self.last_proof_time = datetime.fromisoformat(validation["validated_at"])
            self.valid_proofs.append({
                "landmark_id": landmark_id,
                "validated_at": validation["validated_at"],
                "expires_at": validation["expires_at"],
                "match_score": validation["match_score"]
            })
            
            # Increment landmark verification count
            landmark["verified_by"] += 1
            
            # Update reputation score
            # Simple formula: average of match scores, weighted by recency
            if len(self.valid_proofs) > 0:
                total_score = sum(p["match_score"] for p in self.valid_proofs)
                self.reputation_score = total_score / len(self.valid_proofs)
                
        # Return result
        return {
            "success": validation["valid"],
            "validation": validation,
            "node_id": self.node_id,
            "reputation_score": self.reputation_score,
            "valid_proofs_count": len(self.valid_proofs)
        }
        
    def notarize_document(self, 
                        document_hash: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Notarize a document.
        
        Args:
            document_hash: Hash of the document to notarize
            metadata: Additional metadata
            
        Returns:
            Notarization result
        """
        # Check if the node is eligible to notarize
        if not self._is_eligible():
            return {
                "success": False,
                "reason": "node_not_eligible",
                "details": "Node does not have valid Proof-of-View or sufficient reputation"
            }
            
        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        # Create notarization record
        notarization_id = str(uuid.uuid4())
        notarization = {
            "notarization_id": notarization_id,
            "document_hash": document_hash,
            "notarized_at": timestamp,
            "node_id": self.node_id,
            "node_reputation": self.reputation_score,
            "proofs_count": len(self.valid_proofs)
        }
        
        # Add metadata if provided
        if metadata:
            notarization["metadata"] = metadata
            
        # Store the notarization
        self.notarizations.append(notarization)
        
        return {
            "success": True,
            "notarization": notarization
        }
        
    def _is_eligible(self) -> bool:
        """
        Check if the node is eligible to notarize documents.
        """
        # Node must have a valid Proof-of-View
        if not self.last_proof_time:
            return False
            
        # Check if the latest proof is still valid
        now = datetime.now()
        time_since_proof = now - self.last_proof_time
        if time_since_proof.total_seconds() > (self.pov_validator.time_validity_hours * 3600):
            return False
            
        # Check reputation score
        if self.reputation_score < 0.5:  # Minimum reputation threshold
            return False
            
        return True


class NotaryNetwork:
    """
    The decentralized notary network that manages nodes and notarization requests.
    """
    
    def __init__(self):
        """Initialize the notary network."""
        self.landmark_registry = LandmarkRegistry()
        self.pov_validator = ProofOfViewValidator()
        
        # Network state
        self.nodes = {}  # node_id -> NotaryNode
        self.notarizations = {}  # notarization_id -> notarization record
        
    def register_node(self, 
                     owner_id: str,
                     owner_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Register a new notary node.
        
        Args:
            owner_id: ID of the node operator
            owner_data: Additional data about the operator
            
        Returns:
            Node registration data
        """
        # Generate node ID
        node_id = str(uuid.uuid4())
        
        # Create the node
        node = NotaryNode(
            node_id=node_id,
            owner_id=owner_id,
            landmark_registry=self.landmark_registry,
            pov_validator=self.pov_validator
        )
        
        # Store the node
        self.nodes[node_id] = node
        
        # Return registration data
        return {
            "node_id": node_id,
            "owner_id": owner_id,
            "registered_at": datetime.now().isoformat(),
            "owner_data": owner_data
        }
        
    def submit_proof_of_view(self, 
                           node_id: str, 
                           landmark_id: str, 
                           proof_signature: Union[str, List[List[float]]]) -> Dict[str, Any]:
        """
        Submit a Proof-of-View for a node.
        
        Args:
            node_id: ID of the node
            landmark_id: ID of the landmark being verified
            proof_signature: Spatial signature or raw coordinates
            
        Returns:
            Submission result
        """
        # Check if the node exists
        node = self.nodes.get(node_id)
        if not node:
            return {
                "success": False,
                "reason": "node_not_found",
                "details": f"Node not found: {node_id}"
            }
            
        # Submit proof
        return node.submit_proof_of_view(landmark_id, proof_signature)
        
    def notarize_document(self, 
                        document_hash: str, 
                        metadata: Optional[Dict[str, Any]] = None,
                        min_nodes: int = 3) -> Dict[str, Any]:
        """
        Notarize a document using multiple nodes for consensus.
        
        Args:
            document_hash: Hash of the document to notarize
            metadata: Additional metadata
            min_nodes: Minimum number of nodes required for consensus
            
        Returns:
            Notarization result
        """
        # Find eligible nodes
        eligible_nodes = [n for n in self.nodes.values() if n._is_eligible()]
        
        if len(eligible_nodes) < min_nodes:
            return {
                "success": False,
                "reason": "insufficient_nodes",
                "details": f"Not enough eligible nodes ({len(eligible_nodes)}/{min_nodes})"
            }
            
        # Notarize with all eligible nodes
        notarizations = []
        for node in eligible_nodes:
            result = node.notarize_document(document_hash, metadata)
            if result["success"]:
                notarizations.append(result["notarization"])
                
        if len(notarizations) < min_nodes:
            return {
                "success": False,
                "reason": "consensus_failure",
                "details": f"Not enough successful notarizations ({len(notarizations)}/{min_nodes})"
            }
            
        # Create network-level notarization record
        timestamp = datetime.now().isoformat()
        notarization_id = str(uuid.uuid4())
        
        network_notarization = {
            "notarization_id": notarization_id,
            "document_hash": document_hash,
            "notarized_at": timestamp,
            "consensus_nodes": len(notarizations),
            "node_notarizations": notarizations
        }
        
        # Add metadata if provided
        if metadata:
            network_notarization["metadata"] = metadata
            
        # Store the notarization
        self.notarizations[notarization_id] = network_notarization
        
        return {
            "success": True,
            "notarization": network_notarization
        }
        
    def verify_notarization(self, notarization_id: str) -> Dict[str, Any]:
        """
        Verify a document notarization.
        
        Args:
            notarization_id: ID of the notarization to verify
            
        Returns:
            Verification result
        """
        # Check if the notarization exists
        notarization = self.notarizations.get(notarization_id)
        if not notarization:
            return {
                "verified": False,
                "reason": "notarization_not_found",
                "details": f"Notarization not found: {notarization_id}"
            }
            
        # In a real implementation, we would verify the blockchain record
        # For Phase 1, we just check if it exists in our database
        
        return {
            "verified": True,
            "notarization": notarization
        }


class NotaryAPI:
    """
    API for the Decentralized Notary Network.
    This is a skeleton implementation for Phase 1.
    """
    
    def __init__(self):
        """Initialize the Notary API."""
        self.network = NotaryNetwork()
        
    def register_landmark(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new landmark.
        
        Args:
            request_data: Dictionary containing landmark data
            
        Returns:
            Landmark registration data
        """
        # Extract request parameters
        name = request_data.get("name")
        description = request_data.get("description")
        location = request_data.get("location")
        spatial_signature = request_data.get("spatial_signature")
        metadata = request_data.get("metadata")
        
        # Validate inputs
        if not name:
            return {"error": "Missing required parameter: name"}
        if not description:
            return {"error": "Missing required parameter: description"}
        if not location:
            return {"error": "Missing required parameter: location"}
        if not spatial_signature:
            return {"error": "Missing required parameter: spatial_signature"}
            
        # Register landmark
        try:
            landmark = self.network.landmark_registry.register_landmark(
                name=name,
                description=description,
                location=location,
                spatial_signature=spatial_signature,
                metadata=metadata
            )
            
            return landmark
        except Exception as e:
            return {"error": f"Landmark registration failed: {str(e)}"}
            
    def register_node(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new notary node.
        
        Args:
            request_data: Dictionary containing node registration data
            
        Returns:
            Node registration data
        """
        # Extract request parameters
        owner_id = request_data.get("owner_id")
        owner_data = request_data.get("owner_data")
        
        # Validate inputs
        if not owner_id:
            return {"error": "Missing required parameter: owner_id"}
            
        # Register node
        try:
            node = self.network.register_node(
                owner_id=owner_id,
                owner_data=owner_data
            )
            
            return node
        except Exception as e:
            return {"error": f"Node registration failed: {str(e)}"}
            
    def submit_proof_of_view(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a Proof-of-View.
        
        Args:
            request_data: Dictionary containing proof submission data
            
        Returns:
            Submission result
        """
        # Extract request parameters
        node_id = request_data.get("node_id")
        landmark_id = request_data.get("landmark_id")
        proof_signature = request_data.get("proof_signature")
        
        # Validate inputs
        if not node_id:
            return {"error": "Missing required parameter: node_id"}
        if not landmark_id:
            return {"error": "Missing required parameter: landmark_id"}
        if not proof_signature:
            return {"error": "Missing required parameter: proof_signature"}
            
        # Submit proof
        try:
            result = self.network.submit_proof_of_view(
                node_id=node_id,
                landmark_id=landmark_id,
                proof_signature=proof_signature
            )
            
            return result
        except Exception as e:
            return {"error": f"Proof submission failed: {str(e)}"}
            
    def notarize_document(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Notarize a document.
        
        Args:
            request_data: Dictionary containing notarization data
            
        Returns:
            Notarization result
        """
        # Extract request parameters
        document_hash = request_data.get("document_hash")
        metadata = request_data.get("metadata")
        min_nodes = request_data.get("min_nodes", 3)
        
        # Validate inputs
        if not document_hash:
            return {"error": "Missing required parameter: document_hash"}
            
        # Notarize document
        try:
            result = self.network.notarize_document(
                document_hash=document_hash,
                metadata=metadata,
                min_nodes=min_nodes
            )
            
            return result
        except Exception as e:
            return {"error": f"Document notarization failed: {str(e)}"}
            
    def verify_notarization(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a document notarization.
        
        Args:
            request_data: Dictionary containing verification data
            
        Returns:
            Verification result
        """
        # Extract request parameters
        notarization_id = request_data.get("notarization_id")
        
        # Validate inputs
        if not notarization_id:
            return {"error": "Missing required parameter: notarization_id"}
            
        # Verify notarization
        try:
            result = self.network.verify_notarization(notarization_id)
            
            return result
        except Exception as e:
            return {"error": f"Notarization verification failed: {str(e)}"}
