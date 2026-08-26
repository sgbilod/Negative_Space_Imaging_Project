"""
Enhanced Streaming Verification Protocol

This module implements a continuous authentication protocol for live streams, broadcasts,
and data feeds. It enables real-time verification of the temporal and spatial authenticity
of streaming content through embedded signatures and challenge-response mechanisms.
"""

import hashlib
import hmac
import time
import uuid
import random
import json
import threading
import queue
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import base64

from ...negative_mapping.spatial_signature_generator import SpatialSignatureGenerator
from ..temporal_auth.temporal_auth_service import TemporalAuthService
from ..decentralized_notary.notary_network import NotaryNetwork, NotaryAPI


class StreamSignature:
    """
    Represents a cryptographic signature for a stream fragment.
    """
    
    def __init__(self, 
                 stream_id: str,
                 fragment_id: str,
                 timestamp: str,
                 spatial_coordinates: List[List[float]],
                 previous_signature: str = None):
        """
        Initialize a stream signature.
        
        Args:
            stream_id: ID of the stream
            fragment_id: ID of the fragment
            timestamp: ISO format timestamp of the signature
            spatial_coordinates: Spatial coordinates for the signature
            previous_signature: Signature of the previous fragment (for chaining)
        """
        self.stream_id = stream_id
        self.fragment_id = fragment_id
        self.timestamp = timestamp
        self.spatial_coordinates = spatial_coordinates
        self.previous_signature = previous_signature
        
        # Generate a spatial signature
        self.spatial_signature = self._generate_spatial_signature()
        
        # Generate the full signature
        self.signature = self._generate_signature()
        
    def _generate_spatial_signature(self) -> str:
        """
        Generate a spatial signature from the coordinates.
        
        Returns:
            Spatial signature
        """
        generator = SpatialSignatureGenerator()
        return generator.generate(self.spatial_coordinates)
        
    def _generate_signature(self) -> str:
        """
        Generate the full signature by combining all components.
        
        Returns:
            Full signature
        """
        # Create the base for the signature
        base = f"{self.stream_id}:{self.fragment_id}:{self.timestamp}:{self.spatial_signature}"
        
        # If we have a previous signature, incorporate it
        if self.previous_signature:
            base = f"{base}:{self.previous_signature}"
            
        # Hash the base to create the signature
        return hashlib.sha256(base.encode()).hexdigest()
        
    def verify(self, 
              spatial_coordinates: List[List[float]] = None,
              previous_signature: str = None) -> bool:
        """
        Verify the signature.
        
        Args:
            spatial_coordinates: Optional coordinates to verify against
            previous_signature: Optional previous signature to verify against
            
        Returns:
            True if verified, False otherwise
        """
        # If spatial coordinates are provided, verify the spatial signature
        if spatial_coordinates:
            generator = SpatialSignatureGenerator()
            expected_spatial_sig = generator.generate(spatial_coordinates)
            
            if expected_spatial_sig != self.spatial_signature:
                return False
                
        # If a previous signature is provided, verify the chain
        if previous_signature and previous_signature != self.previous_signature:
            return False
            
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the signature to a dictionary."""
        return {
            "stream_id": self.stream_id,
            "fragment_id": self.fragment_id,
            "timestamp": self.timestamp,
            "spatial_signature": self.spatial_signature,
            "previous_signature": self.previous_signature,
            "signature": self.signature
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamSignature':
        """
        Create a signature from a dictionary.
        
        Args:
            data: Dictionary containing signature data
            
        Returns:
            A StreamSignature instance
        """
        signature = cls(
            stream_id=data["stream_id"],
            fragment_id=data["fragment_id"],
            timestamp=data["timestamp"],
            spatial_coordinates=[],  # We don't store the actual coordinates
            previous_signature=data["previous_signature"]
        )
        
        # Override generated values
        signature.spatial_signature = data["spatial_signature"]
        signature.signature = data["signature"]
        
        return signature


class StreamFragment:
    """
    Represents a fragment of a verified stream.
    """
    
    def __init__(self, 
                stream_id: str,
                fragment_id: str,
                data: bytes,
                timestamp: str,
                signature: StreamSignature):
        """
        Initialize a stream fragment.
        
        Args:
            stream_id: ID of the stream
            fragment_id: ID of the fragment
            data: Fragment data
            timestamp: ISO format timestamp of the fragment
            signature: Signature of the fragment
        """
        self.stream_id = stream_id
        self.fragment_id = fragment_id
        self.data = data
        self.timestamp = timestamp
        self.signature = signature
        
        # Generate a hash of the data
        self.data_hash = hashlib.sha256(data).hexdigest()
        
    def verify(self, 
              previous_fragment: 'StreamFragment' = None) -> bool:
        """
        Verify the fragment.
        
        Args:
            previous_fragment: Optional previous fragment for chain verification
            
        Returns:
            True if verified, False otherwise
        """
        # Verify the data hash
        expected_hash = hashlib.sha256(self.data).hexdigest()
        if expected_hash != self.data_hash:
            return False
            
        # Verify the signature
        previous_signature = None
        if previous_fragment:
            previous_signature = previous_fragment.signature.signature
            
        return self.signature.verify(previous_signature=previous_signature)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the fragment to a dictionary."""
        return {
            "stream_id": self.stream_id,
            "fragment_id": self.fragment_id,
            "data_hash": self.data_hash,
            "timestamp": self.timestamp,
            "signature": self.signature.to_dict()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], fragment_data: bytes) -> 'StreamFragment':
        """
        Create a fragment from a dictionary and raw data.
        
        Args:
            data: Dictionary containing fragment metadata
            fragment_data: Raw fragment data
            
        Returns:
            A StreamFragment instance
        """
        signature = StreamSignature.from_dict(data["signature"])
        
        fragment = cls(
            stream_id=data["stream_id"],
            fragment_id=data["fragment_id"],
            data=fragment_data,
            timestamp=data["timestamp"],
            signature=signature
        )
        
        # Override generated values
        fragment.data_hash = data["data_hash"]
        
        return fragment


class ChallengeResponse:
    """
    Represents a challenge-response pair for stream verification.
    """
    
    def __init__(self, 
                stream_id: str,
                challenge_id: str,
                challenge_type: str,
                challenge_data: Dict[str, Any],
                timestamp: str,
                expiration: str):
        """
        Initialize a challenge-response pair.
        
        Args:
            stream_id: ID of the stream
            challenge_id: ID of the challenge
            challenge_type: Type of challenge
            challenge_data: Challenge data
            timestamp: ISO format timestamp of the challenge
            expiration: ISO format timestamp of the expiration
        """
        self.stream_id = stream_id
        self.challenge_id = challenge_id
        self.challenge_type = challenge_type
        self.challenge_data = challenge_data
        self.timestamp = timestamp
        self.expiration = expiration
        
        # Response data
        self.response_data = None
        self.response_timestamp = None
        self.verified = False
        
    def is_expired(self) -> bool:
        """
        Check if the challenge has expired.
        
        Returns:
            True if expired, False otherwise
        """
        now = datetime.now().isoformat()
        return now > self.expiration
        
    def submit_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a response to the challenge.
        
        Args:
            response_data: Response data
            
        Returns:
            Verification result
        """
        # Check if the challenge has expired
        if self.is_expired():
            return {
                "verified": False,
                "reason": "Challenge expired",
                "challenge_id": self.challenge_id
            }
            
        # Store the response
        self.response_data = response_data
        self.response_timestamp = datetime.now().isoformat()
        
        # Verify the response
        verification = self.verify_response()
        self.verified = verification["verified"]
        
        return verification
        
    def verify_response(self) -> Dict[str, Any]:
        """
        Verify the response against the challenge.
        
        Returns:
            Verification result
        """
        # Make sure we have a response
        if not self.response_data:
            return {
                "verified": False,
                "reason": "No response submitted",
                "challenge_id": self.challenge_id
            }
            
        # Verify based on challenge type
        if self.challenge_type == "spatial_signature":
            return self._verify_spatial_signature()
        elif self.challenge_type == "temporal_auth":
            return self._verify_temporal_auth()
        elif self.challenge_type == "content_watermark":
            return self._verify_content_watermark()
        else:
            return {
                "verified": False,
                "reason": f"Unknown challenge type: {self.challenge_type}",
                "challenge_id": self.challenge_id
            }
            
    def _verify_spatial_signature(self) -> Dict[str, Any]:
        """
        Verify a spatial signature challenge.
        
        Returns:
            Verification result
        """
        # Get the expected coordinates from the challenge
        expected_coordinates = self.challenge_data.get("coordinates", [])
        
        # Get the signature from the response
        signature = self.response_data.get("signature", "")
        
        # Verify the signature
        generator = SpatialSignatureGenerator()
        expected_signature = generator.generate(expected_coordinates)
        
        if signature == expected_signature:
            return {
                "verified": True,
                "challenge_id": self.challenge_id,
                "challenge_type": self.challenge_type
            }
        else:
            return {
                "verified": False,
                "reason": "Signature mismatch",
                "challenge_id": self.challenge_id,
                "challenge_type": self.challenge_type
            }
            
    def _verify_temporal_auth(self) -> Dict[str, Any]:
        """
        Verify a temporal authentication challenge.
        
        Returns:
            Verification result
        """
        # Get the expected timestamp from the challenge
        expected_timestamp = self.challenge_data.get("timestamp", "")
        
        # Get the timestamp from the response
        timestamp = self.response_data.get("timestamp", "")
        
        # Calculate the time difference
        try:
            expected_dt = datetime.fromisoformat(expected_timestamp)
            response_dt = datetime.fromisoformat(timestamp)
            
            # Allow a small time difference (e.g., 5 seconds)
            time_diff = abs((expected_dt - response_dt).total_seconds())
            
            if time_diff <= 5:
                return {
                    "verified": True,
                    "challenge_id": self.challenge_id,
                    "challenge_type": self.challenge_type,
                    "time_diff": time_diff
                }
            else:
                return {
                    "verified": False,
                    "reason": "Timestamp mismatch",
                    "challenge_id": self.challenge_id,
                    "challenge_type": self.challenge_type,
                    "time_diff": time_diff
                }
        except ValueError:
            return {
                "verified": False,
                "reason": "Invalid timestamp format",
                "challenge_id": self.challenge_id,
                "challenge_type": self.challenge_type
            }
            
    def _verify_content_watermark(self) -> Dict[str, Any]:
        """
        Verify a content watermark challenge.
        
        Returns:
            Verification result
        """
        # Get the expected watermark from the challenge
        expected_watermark = self.challenge_data.get("watermark", "")
        
        # Get the watermark from the response
        watermark = self.response_data.get("watermark", "")
        
        if watermark == expected_watermark:
            return {
                "verified": True,
                "challenge_id": self.challenge_id,
                "challenge_type": self.challenge_type
            }
        else:
            return {
                "verified": False,
                "reason": "Watermark mismatch",
                "challenge_id": self.challenge_id,
                "challenge_type": self.challenge_type
            }
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert the challenge to a dictionary."""
        return {
            "stream_id": self.stream_id,
            "challenge_id": self.challenge_id,
            "challenge_type": self.challenge_type,
            "challenge_data": self.challenge_data,
            "timestamp": self.timestamp,
            "expiration": self.expiration,
            "response_data": self.response_data,
            "response_timestamp": self.response_timestamp,
            "verified": self.verified
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChallengeResponse':
        """
        Create a challenge from a dictionary.
        
        Args:
            data: Dictionary containing challenge data
            
        Returns:
            A ChallengeResponse instance
        """
        challenge = cls(
            stream_id=data["stream_id"],
            challenge_id=data["challenge_id"],
            challenge_type=data["challenge_type"],
            challenge_data=data["challenge_data"],
            timestamp=data["timestamp"],
            expiration=data["expiration"]
        )
        
        # Override response data if available
        challenge.response_data = data.get("response_data")
        challenge.response_timestamp = data.get("response_timestamp")
        challenge.verified = data.get("verified", False)
        
        return challenge


class StreamVerifier:
    """
    Verifies the authenticity of a stream.
    """
    
    def __init__(self):
        """Initialize the stream verifier."""
        self.notary_api = NotaryAPI()
        self.temporal_auth = TemporalAuthService()
        
    def verify_fragment(self, 
                       fragment: StreamFragment,
                       previous_fragment: StreamFragment = None) -> Dict[str, Any]:
        """
        Verify a stream fragment.
        
        Args:
            fragment: Fragment to verify
            previous_fragment: Optional previous fragment for chain verification
            
        Returns:
            Verification result
        """
        # Verify the fragment
        if not fragment.verify(previous_fragment):
            return {
                "verified": False,
                "reason": "Fragment verification failed",
                "fragment_id": fragment.fragment_id,
                "stream_id": fragment.stream_id
            }
            
        # Verify the timestamp
        try:
            fragment_time = datetime.fromisoformat(fragment.timestamp)
            now = datetime.now()
            
            # Allow a time difference of up to 10 minutes
            time_diff = abs((now - fragment_time).total_seconds())
            
            if time_diff > 600:
                return {
                    "verified": False,
                    "reason": "Timestamp too far from current time",
                    "fragment_id": fragment.fragment_id,
                    "stream_id": fragment.stream_id,
                    "time_diff": time_diff
                }
        except ValueError:
            return {
                "verified": False,
                "reason": "Invalid timestamp format",
                "fragment_id": fragment.fragment_id,
                "stream_id": fragment.stream_id
            }
            
        # Create a notarization
        notarization_data = {
            "type": "stream_fragment",
            "stream_id": fragment.stream_id,
            "fragment_id": fragment.fragment_id,
            "data_hash": fragment.data_hash,
            "signature": fragment.signature.signature,
            "timestamp": fragment.timestamp
        }
        
        notarization_result = self.notary_api.notarize_document(notarization_data)
        
        return {
            "verified": True,
            "fragment_id": fragment.fragment_id,
            "stream_id": fragment.stream_id,
            "notarization": notarization_result
        }
        
    def generate_challenge(self, 
                         stream_id: str,
                         challenge_type: str = None) -> ChallengeResponse:
        """
        Generate a challenge for a stream.
        
        Args:
            stream_id: ID of the stream
            challenge_type: Optional type of challenge to generate
            
        Returns:
            A ChallengeResponse instance
        """
        # If no challenge type is specified, choose one randomly
        if not challenge_type:
            challenge_types = ["spatial_signature", "temporal_auth", "content_watermark"]
            challenge_type = random.choice(challenge_types)
            
        # Generate a challenge ID
        challenge_id = str(uuid.uuid4())
        
        # Set the timestamp and expiration
        timestamp = datetime.now().isoformat()
        expiration = (datetime.now() + timedelta(minutes=5)).isoformat()
        
        # Generate challenge data based on the type
        challenge_data = {}
        
        if challenge_type == "spatial_signature":
            # Generate some random coordinates
            coordinates = []
            for _ in range(5):
                coordinates.append([
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(-10, 10)
                ])
                
            challenge_data = {
                "coordinates": coordinates
            }
        elif challenge_type == "temporal_auth":
            # Use the current time
            challenge_data = {
                "timestamp": timestamp
            }
        elif challenge_type == "content_watermark":
            # Generate a random watermark
            watermark = base64.b64encode(os.urandom(16)).decode()
            
            challenge_data = {
                "watermark": watermark
            }
            
        # Create the challenge
        challenge = ChallengeResponse(
            stream_id=stream_id,
            challenge_id=challenge_id,
            challenge_type=challenge_type,
            challenge_data=challenge_data,
            timestamp=timestamp,
            expiration=expiration
        )
        
        return challenge
        
    def verify_challenge_response(self, 
                                challenge: ChallengeResponse,
                                response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a response to a challenge.
        
        Args:
            challenge: The challenge
            response_data: Response data
            
        Returns:
            Verification result
        """
        return challenge.submit_response(response_data)


class StreamAuthenticator:
    """
    Authenticates streams and manages the verification process.
    """
    
    def __init__(self):
        """Initialize the stream authenticator."""
        self.verifier = StreamVerifier()
        
        # Stream state
        self.streams = {}  # stream_id -> {fragments, challenges, ...}
        
    def register_stream(self, 
                      stream_id: str = None,
                      metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Register a new stream.
        
        Args:
            stream_id: Optional ID for the stream (generated if not provided)
            metadata: Optional metadata for the stream
            
        Returns:
            Registration result
        """
        # Generate a stream ID if not provided
        if not stream_id:
            stream_id = str(uuid.uuid4())
            
        # Check if the stream already exists
        if stream_id in self.streams:
            return {
                "success": False,
                "reason": "Stream already exists",
                "stream_id": stream_id
            }
            
        # Create the stream state
        self.streams[stream_id] = {
            "metadata": metadata or {},
            "fragments": [],
            "challenges": [],
            "verification_status": "REGISTERED",
            "registration_time": datetime.now().isoformat(),
            "last_fragment_time": None,
            "challenge_success_rate": 1.0
        }
        
        return {
            "success": True,
            "stream_id": stream_id,
            "registration_time": self.streams[stream_id]["registration_time"]
        }
        
    def add_fragment(self, 
                   stream_id: str,
                   data: bytes,
                   coordinates: List[List[float]] = None) -> Dict[str, Any]:
        """
        Add a fragment to a stream.
        
        Args:
            stream_id: ID of the stream
            data: Fragment data
            coordinates: Optional spatial coordinates for the fragment
            
        Returns:
            Addition result
        """
        # Check if the stream exists
        if stream_id not in self.streams:
            return {
                "success": False,
                "reason": "Stream not found",
                "stream_id": stream_id
            }
            
        # Get the stream state
        stream = self.streams[stream_id]
        
        # Get the previous fragment (if any)
        previous_fragment = None
        if stream["fragments"]:
            previous_fragment = stream["fragments"][-1]
            
        # Generate a fragment ID
        fragment_id = str(uuid.uuid4())
        
        # Set the timestamp
        timestamp = datetime.now().isoformat()
        
        # If coordinates aren't provided, generate some
        if not coordinates:
            coordinates = []
            for _ in range(5):
                coordinates.append([
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(-10, 10)
                ])
                
        # Create a signature
        previous_signature = None
        if previous_fragment:
            previous_signature = previous_fragment.signature.signature
            
        signature = StreamSignature(
            stream_id=stream_id,
            fragment_id=fragment_id,
            timestamp=timestamp,
            spatial_coordinates=coordinates,
            previous_signature=previous_signature
        )
        
        # Create the fragment
        fragment = StreamFragment(
            stream_id=stream_id,
            fragment_id=fragment_id,
            data=data,
            timestamp=timestamp,
            signature=signature
        )
        
        # Verify the fragment
        verification = self.verifier.verify_fragment(fragment, previous_fragment)
        
        if not verification["verified"]:
            return {
                "success": False,
                "reason": "Fragment verification failed",
                "verification": verification,
                "stream_id": stream_id,
                "fragment_id": fragment_id
            }
            
        # Add the fragment to the stream
        stream["fragments"].append(fragment)
        stream["last_fragment_time"] = timestamp
        
        # Update the verification status
        self._update_stream_verification_status(stream_id)
        
        return {
            "success": True,
            "stream_id": stream_id,
            "fragment_id": fragment_id,
            "verification": verification
        }
        
    def challenge_stream(self, 
                       stream_id: str,
                       challenge_type: str = None) -> Dict[str, Any]:
        """
        Challenge a stream to verify its authenticity.
        
        Args:
            stream_id: ID of the stream to challenge
            challenge_type: Optional type of challenge
            
        Returns:
            Challenge result
        """
        # Check if the stream exists
        if stream_id not in self.streams:
            return {
                "success": False,
                "reason": "Stream not found",
                "stream_id": stream_id
            }
            
        # Generate a challenge
        challenge = self.verifier.generate_challenge(stream_id, challenge_type)
        
        # Add the challenge to the stream
        self.streams[stream_id]["challenges"].append(challenge)
        
        return {
            "success": True,
            "stream_id": stream_id,
            "challenge": challenge.to_dict()
        }
        
    def respond_to_challenge(self, 
                           stream_id: str,
                           challenge_id: str,
                           response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Respond to a challenge.
        
        Args:
            stream_id: ID of the stream
            challenge_id: ID of the challenge
            response_data: Response data
            
        Returns:
            Response result
        """
        # Check if the stream exists
        if stream_id not in self.streams:
            return {
                "success": False,
                "reason": "Stream not found",
                "stream_id": stream_id
            }
            
        # Find the challenge
        challenge = None
        for c in self.streams[stream_id]["challenges"]:
            if c.challenge_id == challenge_id:
                challenge = c
                break
                
        if not challenge:
            return {
                "success": False,
                "reason": "Challenge not found",
                "stream_id": stream_id,
                "challenge_id": challenge_id
            }
            
        # Submit the response
        verification = self.verifier.verify_challenge_response(challenge, response_data)
        
        # Update the stream verification status
        self._update_stream_verification_status(stream_id)
        
        return {
            "success": True,
            "stream_id": stream_id,
            "challenge_id": challenge_id,
            "verification": verification
        }
        
    def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """
        Get the status of a stream.
        
        Args:
            stream_id: ID of the stream
            
        Returns:
            Stream status
        """
        # Check if the stream exists
        if stream_id not in self.streams:
            return {
                "success": False,
                "reason": "Stream not found",
                "stream_id": stream_id
            }
            
        # Get the stream state
        stream = self.streams[stream_id]
        
        # Count the fragments and challenges
        fragment_count = len(stream["fragments"])
        challenge_count = len(stream["challenges"])
        
        # Count the verified challenges
        verified_challenges = sum(1 for c in stream["challenges"] if c.verified)
        
        # Calculate the challenge success rate
        challenge_success_rate = 0.0
        if challenge_count > 0:
            challenge_success_rate = verified_challenges / challenge_count
            
        # Get the last fragment time
        last_fragment_time = stream["last_fragment_time"]
        
        # Calculate the stream age
        stream_age = None
        if stream["registration_time"]:
            try:
                registration_time = datetime.fromisoformat(stream["registration_time"])
                stream_age = (datetime.now() - registration_time).total_seconds()
            except ValueError:
                pass
                
        return {
            "success": True,
            "stream_id": stream_id,
            "verification_status": stream["verification_status"],
            "fragment_count": fragment_count,
            "challenge_count": challenge_count,
            "verified_challenges": verified_challenges,
            "challenge_success_rate": challenge_success_rate,
            "registration_time": stream["registration_time"],
            "last_fragment_time": last_fragment_time,
            "stream_age": stream_age
        }
        
    def _update_stream_verification_status(self, stream_id: str):
        """
        Update the verification status of a stream.
        
        Args:
            stream_id: ID of the stream
        """
        # Check if the stream exists
        if stream_id not in self.streams:
            return
            
        # Get the stream state
        stream = self.streams[stream_id]
        
        # Count the verified challenges
        challenge_count = len(stream["challenges"])
        verified_challenges = sum(1 for c in stream["challenges"] if c.verified)
        
        # Calculate the challenge success rate
        challenge_success_rate = 1.0
        if challenge_count > 0:
            challenge_success_rate = verified_challenges / challenge_count
            
        # Update the success rate
        stream["challenge_success_rate"] = challenge_success_rate
        
        # Determine the verification status
        if challenge_count == 0:
            # No challenges yet
            if len(stream["fragments"]) == 0:
                stream["verification_status"] = "REGISTERED"
            else:
                stream["verification_status"] = "ACTIVE"
        elif challenge_success_rate == 1.0:
            # All challenges verified
            stream["verification_status"] = "VERIFIED"
        elif challenge_success_rate >= 0.7:
            # Most challenges verified
            stream["verification_status"] = "MOSTLY_VERIFIED"
        else:
            # Too many failed challenges
            stream["verification_status"] = "SUSPICIOUS"


class StreamProcessor:
    """
    Processes stream data and manages the verification workflow.
    """
    
    def __init__(self):
        """Initialize the stream processor."""
        self.authenticator = StreamAuthenticator()
        
        # Streaming state
        self.active_streams = {}  # stream_id -> streaming info
        
    def start_stream(self, 
                   metadata: Dict[str, Any] = None,
                   challenge_interval: int = 60,
                   verification_callback: Callable = None) -> Dict[str, Any]:
        """
        Start a new verified stream.
        
        Args:
            metadata: Optional metadata for the stream
            challenge_interval: Interval in seconds between challenges
            verification_callback: Optional callback for verification events
            
        Returns:
            Stream start result
        """
        # Register the stream
        registration = self.authenticator.register_stream(metadata=metadata)
        
        if not registration["success"]:
            return registration
            
        stream_id = registration["stream_id"]
        
        # Start the verification thread
        self.active_streams[stream_id] = {
            "metadata": metadata or {},
            "challenge_interval": challenge_interval,
            "verification_callback": verification_callback,
            "start_time": datetime.now().isoformat(),
            "verification_thread": None,
            "stop_event": threading.Event(),
            "fragments_queue": queue.Queue(),
            "challenges": []
        }
        
        # Start the verification thread
        verification_thread = threading.Thread(
            target=self._verification_worker,
            args=(stream_id,)
        )
        
        verification_thread.daemon = True
        verification_thread.start()
        
        self.active_streams[stream_id]["verification_thread"] = verification_thread
        
        return {
            "success": True,
            "stream_id": stream_id,
            "start_time": self.active_streams[stream_id]["start_time"]
        }
        
    def stop_stream(self, stream_id: str) -> Dict[str, Any]:
        """
        Stop a verified stream.
        
        Args:
            stream_id: ID of the stream
            
        Returns:
            Stream stop result
        """
        # Check if the stream exists
        if stream_id not in self.active_streams:
            return {
                "success": False,
                "reason": "Stream not found",
                "stream_id": stream_id
            }
            
        # Signal the verification thread to stop
        self.active_streams[stream_id]["stop_event"].set()
        
        # Wait for the thread to finish
        self.active_streams[stream_id]["verification_thread"].join(timeout=5)
        
        # Get the stream status
        status = self.authenticator.get_stream_status(stream_id)
        
        # Remove the stream from active streams
        del self.active_streams[stream_id]
        
        return {
            "success": True,
            "stream_id": stream_id,
            "status": status
        }
        
    def add_stream_data(self, 
                      stream_id: str,
                      data: bytes,
                      coordinates: List[List[float]] = None) -> Dict[str, Any]:
        """
        Add data to a stream.
        
        Args:
            stream_id: ID of the stream
            data: Stream data
            coordinates: Optional spatial coordinates
            
        Returns:
            Data addition result
        """
        # Check if the stream exists
        if stream_id not in self.active_streams:
            return {
                "success": False,
                "reason": "Stream not found",
                "stream_id": stream_id
            }
            
        # Add the data to the fragments queue
        self.active_streams[stream_id]["fragments_queue"].put((data, coordinates))
        
        return {
            "success": True,
            "stream_id": stream_id,
            "queued": True
        }
        
    def handle_challenge(self, 
                       stream_id: str,
                       challenge_id: str,
                       coordinates: List[List[float]] = None) -> Dict[str, Any]:
        """
        Handle a challenge for a stream.
        
        Args:
            stream_id: ID of the stream
            challenge_id: ID of the challenge
            coordinates: Optional spatial coordinates for the response
            
        Returns:
            Challenge handling result
        """
        # Check if the stream exists
        if stream_id not in self.active_streams:
            return {
                "success": False,
                "reason": "Stream not found",
                "stream_id": stream_id
            }
            
        # Find the challenge
        challenge = None
        for c in self.active_streams[stream_id]["challenges"]:
            if c["challenge_id"] == challenge_id:
                challenge = c
                break
                
        if not challenge:
            return {
                "success": False,
                "reason": "Challenge not found",
                "stream_id": stream_id,
                "challenge_id": challenge_id
            }
            
        # Prepare the response based on the challenge type
        response_data = {}
        
        if challenge["challenge_type"] == "spatial_signature":
            # Use the provided coordinates or generate some
            coords = coordinates or []
            
            if not coords:
                # Use the coordinates from the challenge
                coords = challenge["challenge_data"].get("coordinates", [])
                
            # Generate a signature
            generator = SpatialSignatureGenerator()
            signature = generator.generate(coords)
            
            response_data = {
                "signature": signature
            }
        elif challenge["challenge_type"] == "temporal_auth":
            # Use the current time
            response_data = {
                "timestamp": datetime.now().isoformat()
            }
        elif challenge["challenge_type"] == "content_watermark":
            # Use the watermark from the challenge
            watermark = challenge["challenge_data"].get("watermark", "")
            
            response_data = {
                "watermark": watermark
            }
            
        # Submit the response
        result = self.authenticator.respond_to_challenge(
            stream_id=stream_id,
            challenge_id=challenge_id,
            response_data=response_data
        )
        
        return result
        
    def _verification_worker(self, stream_id: str):
        """
        Worker thread for stream verification.
        
        Args:
            stream_id: ID of the stream
        """
        # Get the stream state
        stream = self.active_streams[stream_id]
        stop_event = stream["stop_event"]
        fragments_queue = stream["fragments_queue"]
        challenge_interval = stream["challenge_interval"]
        verification_callback = stream["verification_callback"]
        
        # Track the last challenge time
        last_challenge_time = 0
        
        while not stop_event.is_set():
            # Check if it's time for a challenge
            current_time = time.time()
            if current_time - last_challenge_time > challenge_interval:
                # Generate a challenge
                result = self.authenticator.challenge_stream(stream_id)
                
                if result["success"]:
                    # Store the challenge
                    stream["challenges"].append(result["challenge"])
                    
                    # Call the verification callback if provided
                    if verification_callback:
                        verification_callback({
                            "type": "challenge",
                            "stream_id": stream_id,
                            "challenge": result["challenge"]
                        })
                        
                # Update the last challenge time
                last_challenge_time = current_time
                
            # Process any pending fragments
            try:
                # Use a timeout to avoid blocking the thread
                data, coordinates = fragments_queue.get(block=True, timeout=1)
                
                # Add the fragment
                result = self.authenticator.add_fragment(
                    stream_id=stream_id,
                    data=data,
                    coordinates=coordinates
                )
                
                # Call the verification callback if provided
                if verification_callback and result["success"]:
                    verification_callback({
                        "type": "fragment",
                        "stream_id": stream_id,
                        "fragment_id": result["fragment_id"],
                        "verification": result["verification"]
                    })
                    
            except queue.Empty:
                # No fragments to process
                pass
                
            # Sleep briefly to avoid high CPU usage
            time.sleep(0.1)


class EnhancedStreamingVerificationProtocol:
    """
    Main class for the Enhanced Streaming Verification Protocol.
    """
    
    def __init__(self):
        """Initialize the streaming verification protocol."""
        self.processor = StreamProcessor()
        
    def start_verified_stream(self, 
                            metadata: Dict[str, Any] = None,
                            challenge_interval: int = 60,
                            verification_callback: Callable = None) -> Dict[str, Any]:
        """
        Start a new verified stream.
        
        Args:
            metadata: Optional metadata for the stream
            challenge_interval: Interval in seconds between challenges
            verification_callback: Optional callback for verification events
            
        Returns:
            Stream start result
        """
        return self.processor.start_stream(
            metadata=metadata,
            challenge_interval=challenge_interval,
            verification_callback=verification_callback
        )
        
    def stop_verified_stream(self, stream_id: str) -> Dict[str, Any]:
        """
        Stop a verified stream.
        
        Args:
            stream_id: ID of the stream
            
        Returns:
            Stream stop result
        """
        return self.processor.stop_stream(stream_id)
        
    def add_stream_data(self, 
                      stream_id: str,
                      data: bytes,
                      coordinates: List[List[float]] = None) -> Dict[str, Any]:
        """
        Add data to a stream.
        
        Args:
            stream_id: ID of the stream
            data: Stream data
            coordinates: Optional spatial coordinates
            
        Returns:
            Data addition result
        """
        return self.processor.add_stream_data(
            stream_id=stream_id,
            data=data,
            coordinates=coordinates
        )
        
    def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """
        Get the status of a stream.
        
        Args:
            stream_id: ID of the stream
            
        Returns:
            Stream status
        """
        return self.processor.authenticator.get_stream_status(stream_id)
        
    def verify_stream_fragment(self, 
                             stream_id: str,
                             fragment_id: str) -> Dict[str, Any]:
        """
        Verify a specific fragment in a stream.
        
        Args:
            stream_id: ID of the stream
            fragment_id: ID of the fragment
            
        Returns:
            Verification result
        """
        # Check if the stream exists
        if stream_id not in self.processor.authenticator.streams:
            return {
                "success": False,
                "reason": "Stream not found",
                "stream_id": stream_id
            }
            
        # Find the fragment
        fragment = None
        for f in self.processor.authenticator.streams[stream_id]["fragments"]:
            if f.fragment_id == fragment_id:
                fragment = f
                break
                
        if not fragment:
            return {
                "success": False,
                "reason": "Fragment not found",
                "stream_id": stream_id,
                "fragment_id": fragment_id
            }
            
        # Find the previous fragment
        previous_fragment = None
        fragments = self.processor.authenticator.streams[stream_id]["fragments"]
        
        for i, f in enumerate(fragments):
            if f.fragment_id == fragment_id and i > 0:
                previous_fragment = fragments[i - 1]
                break
                
        # Verify the fragment
        verification = self.processor.authenticator.verifier.verify_fragment(
            fragment=fragment,
            previous_fragment=previous_fragment
        )
        
        return {
            "success": True,
            "stream_id": stream_id,
            "fragment_id": fragment_id,
            "verification": verification
        }
