# Documentation for acausal_randomness_oracle.py

```python
"""
Acausal Randomness Oracle

This module implements a system for generating true random numbers using acausal spatial void
signatures. It provides an API for accessing high-quality randomness from spatial-temporal
configurations, ensuring unpredictability and bias-free random values.
"""

import hashlib
import hmac
import random
import uuid
import math
import time
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import base64

from ...negative_mapping.spatial_signature_generator import SpatialSignatureGenerator
from ...negative_mapping.void_signature_extractor import VoidSignatureExtractor
from ..decentralized_notary.notary_network import NotaryAPI


class EntropySource:
    """
    Abstract base class for entropy sources.
    """
    
    def __init__(self, name: str):
        """
        Initialize an entropy source.
        
        Args:
            name: Name of the entropy source
        """
        self.name = name
        
    def get_entropy(self, num_bytes: int = 32) -> bytes:
        """
        Get entropy from the source.
        
        Args:
            num_bytes: Number of bytes of entropy to generate
            
        Returns:
            Entropy bytes
        """
        raise NotImplementedError("Subclasses must implement get_entropy")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the entropy source.
        
        Returns:
            Statistics dictionary
        """
        raise NotImplementedError("Subclasses must implement get_stats")


class VoidSignatureEntropySource(EntropySource):
    """
    Entropy source using void signatures.
    """
    
    def __init__(self, 
               coordinates_provider: Callable[[], List[List[float]]] = None):
        """
        Initialize a void signature entropy source.
        
        Args:
            coordinates_provider: Optional function that provides spatial coordinates
        """
        super().__init__(name="void_signature")
        self.void_extractor = VoidSignatureExtractor()
        
        # Provider for spatial coordinates
        self.coordinates_provider = coordinates_provider or self._default_coordinates
        
        # Statistics
        self.total_extractions = 0
        self.total_bytes = 0
        self.extraction_times = []
        
    def get_entropy(self, num_bytes: int = 32) -> bytes:
        """
        Get entropy from void signatures.
        
        Args:
            num_bytes: Number of bytes of entropy to generate
            
        Returns:
            Entropy bytes
        """
        # Record start time for statistics
        start_time = time.time()
        
        # Generate void signatures until we have enough entropy
        signatures = []
        bytes_needed = num_bytes
        
        while bytes_needed > 0:
            # Get spatial coordinates
            coordinates = self.coordinates_provider()
            
            # Extract void signature
            signature = self.void_extractor.extract(coordinates)
            
            # Add to our collection
            signatures.append(signature)
            
            # Estimate how many bytes we've collected
            # Each signature provides about 16 bytes of entropy
            bytes_needed -= 16
            
        # Combine all signatures into a single entropy pool
        entropy_pool = hashlib.sha512("".join(signatures).encode()).digest()
        
        # If we need more bytes than the hash provides, stretch the output
        if num_bytes > len(entropy_pool):
            # Use HKDF-like construction to stretch the output
            result = bytearray()
            counter = 0
            
            while len(result) < num_bytes:
                counter_bytes = counter.to_bytes(4, byteorder="big")
                block = hmac.new(entropy_pool, counter_bytes, hashlib.sha512).digest()
                result.extend(block)
                counter += 1
                
            entropy_pool = bytes(result[:num_bytes])
        else:
            # Truncate to the requested size
            entropy_pool = entropy_pool[:num_bytes]
            
        # Update statistics
        self.total_extractions += 1
        self.total_bytes += num_bytes
        self.extraction_times.append(time.time() - start_time)
        
        return entropy_pool
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the entropy source.
        
        Returns:
            Statistics dictionary
        """
        # Calculate average extraction time
        avg_time = 0
        if self.extraction_times:
            avg_time = sum(self.extraction_times) / len(self.extraction_times)
            
        return {
            "name": self.name,
            "total_extractions": self.total_extractions,
            "total_bytes": self.total_bytes,
            "average_extraction_time": avg_time
        }
        
    def _default_coordinates(self) -> List[List[float]]:
        """
        Default provider for spatial coordinates.
        
        Returns:
            List of spatial coordinates
        """
        # Generate random coordinates for demonstration
        # In a real implementation, these would come from astronomical data
        coordinates = []
        for _ in range(10):
            coordinates.append([
                random.uniform(-100, 100),
                random.uniform(-100, 100),
                random.uniform(-100, 100)
            ])
            
        return coordinates


class AstronomicalEntropySource(EntropySource):
    """
    Entropy source using astronomical data.
    """
    
    def __init__(self, 
               astronomical_data_provider: Callable[[], Dict[str, Any]] = None):
        """
        Initialize an astronomical entropy source.
        
        Args:
            astronomical_data_provider: Optional function that provides astronomical data
        """
        super().__init__(name="astronomical")
        
        # Provider for astronomical data
        self.astronomical_data_provider = astronomical_data_provider or self._default_data
        
        # Statistics
        self.total_extractions = 0
        self.total_bytes = 0
        self.extraction_times = []
        
    def get_entropy(self, num_bytes: int = 32) -> bytes:
        """
        Get entropy from astronomical data.
        
        Args:
            num_bytes: Number of bytes of entropy to generate
            
        Returns:
            Entropy bytes
        """
        # Record start time for statistics
        start_time = time.time()
        
        # Get astronomical data
        astro_data = self.astronomical_data_provider()
        
        # Convert the data to a string and hash it
        data_str = json.dumps(astro_data, sort_keys=True)
        entropy = hashlib.sha512(data_str.encode()).digest()
        
        # If we need more bytes, stretch the output
        if num_bytes > len(entropy):
            # Use HKDF-like construction to stretch the output
            result = bytearray()
            counter = 0
            
            while len(result) < num_bytes:
                counter_bytes = counter.to_bytes(4, byteorder="big")
                block = hmac.new(entropy, counter_bytes, hashlib.sha512).digest()
                result.extend(block)
                counter += 1
                
            entropy = bytes(result[:num_bytes])
        else:
            # Truncate to the requested size
            entropy = entropy[:num_bytes]
            
        # Update statistics
        self.total_extractions += 1
        self.total_bytes += num_bytes
        self.extraction_times.append(time.time() - start_time)
        
        return entropy
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the entropy source.
        
        Returns:
            Statistics dictionary
        """
        # Calculate average extraction time
        avg_time = 0
        if self.extraction_times:
            avg_time = sum(self.extraction_times) / len(self.extraction_times)
            
        return {
            "name": self.name,
            "total_extractions": self.total_extractions,
            "total_bytes": self.total_bytes,
            "average_extraction_time": avg_time
        }
        
    def _default_data(self) -> Dict[str, Any]:
        """
        Default provider for astronomical data.
        
        Returns:
            Astronomical data dictionary
        """
        # In a real implementation, this would fetch current astronomical data
        # For demonstration, we'll use some simulated data
        
        # Current time as a seed
        now = datetime.now()
        seed = int(now.timestamp() * 1000)
        
        # Set the seed for reproducibility in testing
        random.seed(seed)
        
        # Simulate positions for various celestial objects
        celestial_objects = {
            "sun": {
                "ra": random.uniform(0, 24),  # Right ascension (hours)
                "dec": random.uniform(-90, 90),  # Declination (degrees)
                "distance": 1.0  # Astronomical units
            },
            "moon": {
                "ra": random.uniform(0, 24),
                "dec": random.uniform(-90, 90),
                "distance": random.uniform(0.95, 1.05)  # Varies
            }
        }
        
        # Add planets
        planets = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
        
        for planet in planets:
            celestial_objects[planet] = {
                "ra": random.uniform(0, 24),
                "dec": random.uniform(-90, 90),
                "distance": random.uniform(0.5, 30.0)  # Varies widely
            }
            
        # Add some background radiation measurements
        background_radiation = {
            "cosmic_microwave": random.uniform(2.7, 2.8),  # Kelvin
            "gamma_ray_count": random.randint(10, 100),
            "neutrino_detection": random.uniform(0, 1)
        }
        
        return {
            "timestamp": now.isoformat(),
            "celestial_objects": celestial_objects,
            "background_radiation": background_radiation,
            "seed": seed
        }


class QuantumFluctuationEntropySource(EntropySource):
    """
    Entropy source simulating quantum fluctuations in space.
    """
    
    def __init__(self):
        """Initialize a quantum fluctuation entropy source."""
        super().__init__(name="quantum_fluctuation")
        
        # Statistics
        self.total_extractions = 0
        self.total_bytes = 0
        self.extraction_times = []
        
    def get_entropy(self, num_bytes: int = 32) -> bytes:
        """
        Get entropy from simulated quantum fluctuations.
        
        Args:
            num_bytes: Number of bytes of entropy to generate
            
        Returns:
            Entropy bytes
        """
        # Record start time for statistics
        start_time = time.time()
        
        # Simulate quantum fluctuations
        # In a real implementation, this would use actual quantum measurements
        
        # Create an array of "measurements"
        measurements = []
        
        # We need roughly 8 times as many measurements as bytes
        # since each measurement gives us about 1 bit of entropy
        for _ in range(num_bytes * 8):
            # Simulate a quantum measurement (0 or 1)
            # This is just a simple random number for demonstration
            measurement = 1 if random.random() > 0.5 else 0
            measurements.append(measurement)
            
        # Convert the bits to bytes
        entropy = bytearray()
        
        for i in range(0, len(measurements), 8):
            if i + 8 <= len(measurements):
                # Combine 8 bits into a byte
                byte_value = 0
                for j in range(8):
                    byte_value = (byte_value << 1) | measurements[i + j]
                    
                entropy.append(byte_value)
                
        # If we didn't get enough bytes, hash what we have
        if len(entropy) < num_bytes:
            entropy = hashlib.sha512(entropy).digest()[:num_bytes]
            
        # Update statistics
        self.total_extractions += 1
        self.total_bytes += num_bytes
        self.extraction_times.append(time.time() - start_time)
        
        return bytes(entropy)
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the entropy source.
        
        Returns:
            Statistics dictionary
        """
        # Calculate average extraction time
        avg_time = 0
        if self.extraction_times:
            avg_time = sum(self.extraction_times) / len(self.extraction_times)
            
        return {
            "name": self.name,
            "total_extractions": self.total_extractions,
            "total_bytes": self.total_bytes,
            "average_extraction_time": avg_time
        }


class EntropyPool:
    """
    Pool of entropy from multiple sources.
    """
    
    def __init__(self):
        """Initialize an entropy pool."""
        # Set up entropy sources
        self.sources = {
            "void_signature": VoidSignatureEntropySource(),
            "astronomical": AstronomicalEntropySource(),
            "quantum_fluctuation": QuantumFluctuationEntropySource()
        }
        
        # Pool state
        self.pool = bytearray(64)  # Start with 64 bytes of zeros
        self.pool_refills = 0
        
        # Counters for notarized randomness
        self.notarized_extractions = 0
        self.next_refill_timestamp = 0
        
        # Initialize the pool
        self._refill_pool()
        
    def get_entropy(self, 
                  num_bytes: int = 32,
                  source_weights: Dict[str, float] = None) -> bytes:
        """
        Get entropy from the pool, optionally specifying source weights.
        
        Args:
            num_bytes: Number of bytes of entropy to get
            source_weights: Optional weights for each source (defaults to equal)
            
        Returns:
            Entropy bytes
        """
        # Check if the pool needs refilling
        if time.time() > self.next_refill_timestamp:
            self._refill_pool()
            
        # Use default weights if none provided
        if not source_weights:
            source_weights = {
                "void_signature": 1.0,
                "astronomical": 1.0,
                "quantum_fluctuation": 1.0
            }
            
        # Normalize the weights
        total_weight = sum(source_weights.values())
        if total_weight == 0:
            # Avoid division by zero
            total_weight = 1.0
            
        for source in source_weights:
            source_weights[source] /= total_weight
            
        # Get entropy from each source based on weights
        entropies = []
        
        for source_name, weight in source_weights.items():
            if source_name in self.sources:
                # Calculate how many bytes to get from this source
                source_bytes = int(num_bytes * weight)
                if source_bytes > 0:
                    entropy = self.sources[source_name].get_entropy(source_bytes)
                    entropies.append(entropy)
                    
        # Combine entropies with the pool
        combined = bytearray()
        
        # Add the current pool
        combined.extend(self.pool)
        
        # Add entropies from sources
        for entropy in entropies:
            combined.extend(entropy)
            
        # Mix the combined entropy
        result = hashlib.sha512(combined).digest()
        
        # Update the pool with some of the new entropy
        self.pool = bytearray(result[:64])
        
        # Return the requested number of bytes
        return result[:num_bytes]
        
    def get_notarized_entropy(self, 
                            num_bytes: int = 32,
                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get entropy that is notarized for verification.
        
        Args:
            num_bytes: Number of bytes of entropy to get
            metadata: Optional metadata to include in the notarization
            
        Returns:
            Dictionary with entropy and notarization information
        """
        # Get entropy from the pool
        entropy = self.get_entropy(num_bytes)
        
        # Convert to a hex string for easier handling
        entropy_hex = entropy.hex()
        
        # Prepare notarization data
        notarization_data = {
            "entropy": entropy_hex,
            "timestamp": datetime.now().isoformat(),
            "source": "acausal_randomness_oracle",
            "metadata": metadata or {}
        }
        
        # Get a notarization from the notary network
        notary_api = NotaryAPI()
        notarization = notary_api.notarize_document(notarization_data)
        
        # Update notarization counter
        self.notarized_extractions += 1
        
        return {
            "entropy": entropy_hex,
            "notarization": notarization,
            "timestamp": notarization_data["timestamp"]
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the entropy pool and sources.
        
        Returns:
            Statistics dictionary
        """
        # Get stats from each source
        source_stats = {}
        for name, source in self.sources.items():
            source_stats[name] = source.get_stats()
            
        return {
            "pool_refills": self.pool_refills,
            "notarized_extractions": self.notarized_extractions,
            "sources": source_stats
        }
        
    def _refill_pool(self):
        """Refill the entropy pool from all sources."""
        # Get fresh entropy from each source
        entropies = []
        
        for source in self.sources.values():
            entropies.append(source.get_entropy(64))
            
        # Combine all entropies
        combined = bytearray()
        
        # Start with the current pool
        combined.extend(self.pool)
        
        # Add entropies from sources
        for entropy in entropies:
            combined.extend(entropy)
            
        # Mix the combined entropy
        self.pool = bytearray(hashlib.sha512(combined).digest()[:64])
        
        # Update refill count and next refill time
        self.pool_refills += 1
        
        # Schedule next refill (5 minutes from now)
        self.next_refill_timestamp = time.time() + 300


class RandomGenerator:
    """
    Generator for various types of random values.
    """
    
    def __init__(self, entropy_pool: EntropyPool):
        """
        Initialize a random generator.
        
        Args:
            entropy_pool: Entropy pool to use
        """
        self.entropy_pool = entropy_pool
        
    def get_random_bytes(self, num_bytes: int) -> bytes:
        """
        Get random bytes.
        
        Args:
            num_bytes: Number of random bytes to generate
            
        Returns:
            Random bytes
        """
        return self.entropy_pool.get_entropy(num_bytes)
        
    def get_random_int(self, min_value: int, max_value: int) -> int:
        """
        Get a random integer in the given range.
        
        Args:
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            
        Returns:
            Random integer
        """
        if min_value > max_value:
            min_value, max_value = max_value, min_value
            
        # Calculate how many bytes we need
        range_size = max_value - min_value + 1
        bytes_needed = (range_size.bit_length() + 7) // 8
        
        # Get random bytes
        random_bytes = self.get_random_bytes(bytes_needed)
        
        # Convert to an integer
        value = int.from_bytes(random_bytes, byteorder="big")
        
        # Scale to the desired range
        return min_value + (value % range_size)
        
    def get_random_float(self) -> float:
        """
        Get a random float between 0 and 1.
        
        Returns:
            Random float
        """
        # Get 8 random bytes
        random_bytes = self.get_random_bytes(8)
        
        # Convert to an integer
        value = int.from_bytes(random_bytes, byteorder="big")
        
        # Scale to [0, 1)
        return value / (2**(8*8))
        
    def get_random_choice(self, items: List[Any]) -> Any:
        """
        Get a random item from a list.
        
        Args:
            items: List of items to choose from
            
        Returns:
            Random item
        """
        if not items:
            return None
            
        index = self.get_random_int(0, len(items) - 1)
        return items[index]
        
    def get_random_shuffle(self, items: List[Any]) -> List[Any]:
        """
        Get a randomly shuffled copy of a list.
        
        Args:
            items: List of items to shuffle
            
        Returns:
            Shuffled list
        """
        if not items:
            return []
            
        # Copy the list
        result = items.copy()
        
        # Fisher-Yates shuffle
        for i in range(len(result) - 1, 0, -1):
            j = self.get_random_int(0, i)
            result[i], result[j] = result[j], result[i]
            
        return result
        
    def get_random_sample(self, 
                         items: List[Any],
                         k: int) -> List[Any]:
        """
        Get a random sample of k items from a list.
        
        Args:
            items: List of items to sample from
            k: Number of items to sample
            
        Returns:
            Random sample
        """
        if not items:
            return []
            
        # Ensure k is not larger than the list
        k = min(k, len(items))
        
        # Use reservoir sampling for efficiency
        result = items[:k]
        
        for i in range(k, len(items)):
            j = self.get_random_int(0, i)
            if j < k:
                result[j] = items[i]
                
        return result
        
    def get_random_uuid(self) -> str:
        """
        Get a random UUID.
        
        Returns:
            Random UUID
        """
        # Get 16 random bytes
        random_bytes = self.get_random_bytes(16)
        
        # Set the version (4) and variant (RFC 4122)
        random_bytes = bytearray(random_bytes)
        random_bytes[6] = (random_bytes[6] & 0x0F) | 0x40  # version 4
        random_bytes[8] = (random_bytes[8] & 0x3F) | 0x80  # variant RFC 4122
        
        # Convert to UUID
        return str(uuid.UUID(bytes=bytes(random_bytes)))
        
    def get_random_string(self, 
                         length: int,
                         charset: str = None) -> str:
        """
        Get a random string.
        
        Args:
            length: Length of the string
            charset: Optional charset to use (defaults to alphanumeric)
            
        Returns:
            Random string
        """
        if not charset:
            charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            
        # Get random indices into the charset
        result = []
        for _ in range(length):
            index = self.get_random_int(0, len(charset) - 1)
            result.append(charset[index])
            
        return "".join(result)
        
    def get_random_notarized(self, 
                          type_name: str,
                          **params) -> Dict[str, Any]:
        """
        Get a notarized random value.
        
        Args:
            type_name: Type of random value to generate
            **params: Parameters for the random value generation
            
        Returns:
            Dictionary with random value and notarization
        """
        # Generate the random value based on the type
        value = None
        
        if type_name == "bytes":
            num_bytes = params.get("num_bytes", 32)
            value = self.get_random_bytes(num_bytes).hex()
        elif type_name == "int":
            min_value = params.get("min_value", 0)
            max_value = params.get("max_value", 100)
            value = self.get_random_int(min_value, max_value)
        elif type_name == "float":
            value = self.get_random_float()
        elif type_name == "choice":
            items = params.get("items", [])
            value = self.get_random_choice(items)
        elif type_name == "uuid":
            value = self.get_random_uuid()
        elif type_name == "string":
            length = params.get("length", 10)
            charset = params.get("charset", None)
            value = self.get_random_string(length, charset)
        else:
            raise ValueError(f"Unknown random value type: {type_name}")
            
        # Get a notarized entropy with the value and parameters
        metadata = {
            "type": type_name,
            "params": params,
            "result": value
        }
        
        notarized = self.entropy_pool.get_notarized_entropy(
            num_bytes=32,
            metadata=metadata
        )
        
        # Add the value to the result
        notarized["value"] = value
        
        return notarized


class RandomnessVerifier:
    """
    Verifies notarized random values.
    """
    
    def __init__(self):
        """Initialize a randomness verifier."""
        self.notary_api = NotaryAPI()
        
    def verify_randomness(self, notarized_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a notarized random value.
        
        Args:
            notarized_data: Notarized randomness data
            
        Returns:
            Verification result
        """
        # Check if we have the necessary data
        if "notarization" not in notarized_data:
            return {
                "verified": False,
                "reason": "Missing notarization data"
            }
            
        # Get the notarization
        notarization = notarized_data["notarization"]
        
        # Verify the notarization with the notary network
        verification = self.notary_api.verify_notarization(notarization)
        
        if not verification.get("verified", False):
            return {
                "verified": False,
                "reason": "Notarization verification failed",
                "details": verification
            }
            
        # Check if the entropy matches the value
        metadata = notarization.get("metadata", {})
        value = notarized_data.get("value")
        
        if metadata.get("result") != value:
            return {
                "verified": False,
                "reason": "Value does not match notarized result"
            }
            
        return {
            "verified": True,
            "timestamp": notarization.get("timestamp"),
            "type": metadata.get("type"),
            "params": metadata.get("params")
        }


class AcausalRandomnessOracle:
    """
    Main class for the acausal randomness oracle.
    """
    
    def __init__(self):
        """Initialize the acausal randomness oracle."""
        self.entropy_pool = EntropyPool()
        self.generator = RandomGenerator(self.entropy_pool)
        self.verifier = RandomnessVerifier()
        
    def get_random_bytes(self, num_bytes: int) -> bytes:
        """
        Get random bytes.
        
        Args:
            num_bytes: Number of random bytes to generate
            
        Returns:
            Random bytes
        """
        return self.generator.get_random_bytes(num_bytes)
        
    def get_random_int(self, min_value: int, max_value: int) -> int:
        """
        Get a random integer in the given range.
        
        Args:
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            
        Returns:
            Random integer
        """
        return self.generator.get_random_int(min_value, max_value)
        
    def get_random_float(self) -> float:
        """
        Get a random float between 0 and 1.
        
        Returns:
            Random float
        """
        return self.generator.get_random_float()
        
    def get_random_choice(self, items: List[Any]) -> Any:
        """
        Get a random item from a list.
        
        Args:
            items: List of items to choose from
            
        Returns:
            Random item
        """
        return self.generator.get_random_choice(items)
        
    def get_random_shuffle(self, items: List[Any]) -> List[Any]:
        """
        Get a randomly shuffled copy of a list.
        
        Args:
            items: List of items to shuffle
            
        Returns:
            Shuffled list
        """
        return self.generator.get_random_shuffle(items)
        
    def get_random_sample(self, 
                         items: List[Any],
                         k: int) -> List[Any]:
        """
        Get a random sample of k items from a list.
        
        Args:
            items: List of items to sample from
            k: Number of items to sample
            
        Returns:
            Random sample
        """
        return self.generator.get_random_sample(items, k)
        
    def get_random_uuid(self) -> str:
        """
        Get a random UUID.
        
        Returns:
            Random UUID
        """
        return self.generator.get_random_uuid()
        
    def get_random_string(self, 
                         length: int,
                         charset: str = None) -> str:
        """
        Get a random string.
        
        Args:
            length: Length of the string
            charset: Optional charset to use (defaults to alphanumeric)
            
        Returns:
            Random string
        """
        return self.generator.get_random_string(length, charset)
        
    def get_notarized_randomness(self, 
                               type_name: str,
                               **params) -> Dict[str, Any]:
        """
        Get a notarized random value.
        
        Args:
            type_name: Type of random value to generate
            **params: Parameters for the random value generation
            
        Returns:
            Dictionary with random value and notarization
        """
        return self.generator.get_random_notarized(type_name, **params)
        
    def verify_randomness(self, notarized_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a notarized random value.
        
        Args:
            notarized_data: Notarized randomness data
            
        Returns:
            Verification result
        """
        return self.verifier.verify_randomness(notarized_data)
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the oracle.
        
        Returns:
            Statistics dictionary
        """
        return self.entropy_pool.get_stats()

```