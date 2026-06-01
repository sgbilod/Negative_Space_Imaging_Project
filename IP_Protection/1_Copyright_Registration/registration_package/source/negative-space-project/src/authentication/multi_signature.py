"""
Multi-Signature Authentication Module for Negative Space Signatures

This module provides mechanisms to combine multiple negative space signatures
into a unified authentication token with enhanced security properties.

Classes:
    MultiSignatureManager: Core class for managing multi-signature authentication
    SignatureCombiner: Utility for combining different signature types
    ThresholdVerifier: Implements threshold-based verification (M-of-N signatures)
    HierarchicalVerifier: Implements hierarchical verification with priority levels
"""

import os
import sys
import time
import json
import logging
import hashlib
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from pathlib import Path
import random

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import project modules
from utils.fallbacks import (
    np, WEB3_AVAILABLE, web3 as w3, Web3, NUMPY_AVAILABLE, hashlib as hl
)
from blockchain.smart_contracts import VerificationService, SignatureRegistry

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SignatureCombiner:
    """Utility for combining different negative space signature types"""
    
    def __init__(self, combination_mode: str = 'weighted'):
        """
        Initialize a signature combiner
        
        Args:
            combination_mode: Mode for combining signatures
                - 'weighted': Combine with weighted importance
                - 'concatenate': Simple concatenation
                - 'interleave': Interleave signature elements
                - 'hash': Hash-based combination
        """
        self.combination_mode = combination_mode
        self.supported_modes = ['weighted', 'concatenate', 'interleave', 'hash']
        
        if combination_mode not in self.supported_modes:
            logger.warning(f"Unsupported combination mode: {combination_mode}")
            logger.warning(f"Using default mode: 'weighted'")
            self.combination_mode = 'weighted'
    
    def combine(self, signatures: List[List[float]], 
               weights: Optional[List[float]] = None) -> List[float]:
        """
        Combine multiple signatures into a single composite signature
        
        Args:
            signatures: List of signatures to combine
            weights: Optional weights for each signature (for weighted mode)
            
        Returns:
            List[float]: Combined signature
        """
        if not signatures:
            raise ValueError("No signatures provided for combination")
        
        # If only one signature, return it directly
        if len(signatures) == 1:
            return signatures[0]
        
        # Use the appropriate combination method based on mode
        if self.combination_mode == 'weighted':
            return self._combine_weighted(signatures, weights)
        elif self.combination_mode == 'concatenate':
            return self._combine_concatenate(signatures)
        elif self.combination_mode == 'interleave':
            return self._combine_interleave(signatures)
        elif self.combination_mode == 'hash':
            return self._combine_hash(signatures)
        else:
            # Fallback to weighted combination
            return self._combine_weighted(signatures, weights)
    
    def _combine_weighted(self, signatures: List[List[float]], 
                        weights: Optional[List[float]] = None) -> List[float]:
        """
        Combine signatures using weighted averaging
        
        Args:
            signatures: List of signatures to combine
            weights: Optional weights for each signature
            
        Returns:
            List[float]: Combined signature
        """
        # Normalize all signatures to the same length
        max_length = max(len(sig) for sig in signatures)
        normalized_sigs = []
        
        for sig in signatures:
            if len(sig) < max_length:
                # Pad shorter signatures
                if NUMPY_AVAILABLE:
                    padded = np.zeros(max_length)
                    padded[:len(sig)] = sig
                    normalized_sigs.append(padded)
                else:
                    padded = [0.0] * max_length
                    for i, val in enumerate(sig):
                        padded[i] = val
                    normalized_sigs.append(padded)
            else:
                normalized_sigs.append(sig)
        
        # Set default weights if not provided
        if weights is None:
            weights = [1.0 / len(normalized_sigs)] * len(normalized_sigs)
        
        # Normalize weights to sum to 1
        if NUMPY_AVAILABLE:
            weights = np.array(weights) / np.sum(weights)
        else:
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]
        
        # Combine signatures using weighted average
        if NUMPY_AVAILABLE:
            result = np.zeros(max_length)
            for i, sig in enumerate(normalized_sigs):
                result += sig * weights[i]
            return result.tolist()
        else:
            result = [0.0] * max_length
            for i, sig in enumerate(normalized_sigs):
                for j, val in enumerate(sig):
                    result[j] += val * weights[i]
            return result
    
    def _combine_concatenate(self, signatures: List[List[float]]) -> List[float]:
        """
        Combine signatures by concatenation
        
        Args:
            signatures: List of signatures to combine
            
        Returns:
            List[float]: Combined signature
        """
        # Simple concatenation
        if NUMPY_AVAILABLE:
            return np.concatenate(signatures).tolist()
        else:
            result = []
            for sig in signatures:
                result.extend(sig)
            return result
    
    def _combine_interleave(self, signatures: List[List[float]]) -> List[float]:
        """
        Combine signatures by interleaving elements
        
        Args:
            signatures: List of signatures to combine
            
        Returns:
            List[float]: Combined signature
        """
        # Normalize all signatures to the same length
        max_length = max(len(sig) for sig in signatures)
        normalized_sigs = []
        
        for sig in signatures:
            if len(sig) < max_length:
                # Pad shorter signatures
                if NUMPY_AVAILABLE:
                    padded = np.zeros(max_length)
                    padded[:len(sig)] = sig
                    normalized_sigs.append(padded)
                else:
                    padded = [0.0] * max_length
                    for i, val in enumerate(sig):
                        padded[i] = val
                    normalized_sigs.append(padded)
            else:
                normalized_sigs.append(sig)
        
        # Interleave elements
        result = []
        for i in range(max_length):
            for sig in normalized_sigs:
                result.append(sig[i])
        
        return result
    
    def _combine_hash(self, signatures: List[List[float]]) -> List[float]:
        """
        Combine signatures using hash-based method
        
        Args:
            signatures: List of signatures to combine
            
        Returns:
            List[float]: Combined signature
        """
        # Convert each signature to a hash
        hashes = []
        for sig in signatures:
            sig_str = ','.join(str(x) for x in sig)
            sig_hash = hashlib.sha256(sig_str.encode()).digest()
            hashes.append(sig_hash)
        
        # Combine hashes
        combined_hash = b''
        for h in hashes:
            combined_hash = hashlib.sha256(combined_hash + h).digest()
        
        # Convert hash to a float array (128 elements)
        if NUMPY_AVAILABLE:
            # Use hash as seed for a deterministic random generator
            seed = int.from_bytes(combined_hash[:4], byteorder='big')
            rng = np.random.RandomState(seed)
            return rng.rand(128).tolist()
        else:
            # Manual approach for generating deterministic values from hash
            result = []
            for i in range(0, min(len(combined_hash), 64), 2):
                # Convert pairs of bytes to float between 0 and 1
                val = int.from_bytes(combined_hash[i:i+2], byteorder='big') / 65535.0
                result.append(val)
            
            # Ensure we have 128 elements
            while len(result) < 128:
                result.extend(result[:128-len(result)])
            
            return result[:128]


class ThresholdVerifier:
    """Implements threshold-based verification (M-of-N signatures)"""
    
    def __init__(self, threshold: int, verification_service: Optional[VerificationService] = None):
        """
        Initialize a threshold verifier
        
        Args:
            threshold: Minimum number of valid signatures required (M in M-of-N)
            verification_service: Optional verification service for blockchain verification
        """
        self.threshold = threshold
        self.verification_service = verification_service
    
    def verify(self, signature_ids: List[str], 
              signature_data: List[List[float]]) -> Dict[str, Any]:
        """
        Verify using the threshold model (M-of-N signatures must be valid)
        
        Args:
            signature_ids: List of signature IDs to verify against
            signature_data: List of signature data to verify
            
        Returns:
            Dict[str, Any]: Verification result
        """
        if len(signature_ids) != len(signature_data):
            raise ValueError("Number of signature IDs must match number of signatures")
        
        # If threshold is higher than number of signatures, it's impossible to meet
        if self.threshold > len(signature_ids):
            return {
                'verified': False,
                'reason': f"Threshold ({self.threshold}) exceeds number of signatures ({len(signature_ids)})",
                'valid_count': 0,
                'total_count': len(signature_ids),
                'details': {}
            }
        
        # Verify each signature
        valid_count = 0
        verification_details = {}
        
        for i, (sig_id, sig_data) in enumerate(zip(signature_ids, signature_data)):
            if self.verification_service:
                # Use blockchain verification
                result = self.verification_service.verify_signature(sig_id, sig_data)
                is_valid = result['isValid']
                details = result['details']
            else:
                # Use local verification (for testing/demo)
                is_valid = self._local_verify(sig_id, sig_data)
                details = {'method': 'local', 'confidence': 0.8}
            
            # Track results
            if is_valid:
                valid_count += 1
            
            verification_details[f'signature_{i+1}'] = {
                'id': sig_id,
                'valid': is_valid,
                'details': details
            }
        
        # Check if we meet the threshold
        verified = valid_count >= self.threshold
        
        return {
            'verified': verified,
            'reason': f"{valid_count} of {len(signature_ids)} signatures valid, threshold is {self.threshold}",
            'valid_count': valid_count,
            'total_count': len(signature_ids),
            'details': verification_details
        }
    
    def _local_verify(self, signature_id: str, signature_data: List[float]) -> bool:
        """
        Perform local verification for testing/demo purposes
        
        Args:
            signature_id: ID of the signature
            signature_data: Signature data
            
        Returns:
            bool: True if signature is valid
        """
        # For demo purposes, use a deterministic verification based on signature_id
        # In a real implementation, this would use a proper verification algorithm
        hash_val = hashlib.md5(signature_id.encode()).hexdigest()
        first_byte = int(hash_val[:2], 16)
        
        # 80% chance of returning True (for demo purposes)
        return first_byte < 204  # 204 is ~80% of 255


class HierarchicalVerifier:
    """Implements hierarchical verification with priority levels"""
    
    def __init__(self, level_thresholds: Dict[str, int], 
                verification_service: Optional[VerificationService] = None):
        """
        Initialize a hierarchical verifier
        
        Args:
            level_thresholds: Dictionary mapping priority levels to thresholds
                e.g., {'high': 1, 'medium': 2, 'low': 3}
            verification_service: Optional verification service for blockchain verification
        """
        self.level_thresholds = level_thresholds
        self.verification_service = verification_service
        
        # Create threshold verifiers for each level
        self.level_verifiers = {}
        for level, threshold in level_thresholds.items():
            self.level_verifiers[level] = ThresholdVerifier(threshold, verification_service)
    
    def verify(self, signature_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify using the hierarchical model
        
        Args:
            signature_map: Dictionary mapping signature IDs to:
                {
                    'level': Priority level (e.g., 'high', 'medium', 'low'),
                    'data': Signature data
                }
            
        Returns:
            Dict[str, Any]: Verification result
        """
        # Group signatures by level
        signatures_by_level = {}
        for sig_id, sig_info in signature_map.items():
            level = sig_info.get('level', 'medium')  # Default to medium if not specified
            if level not in signatures_by_level:
                signatures_by_level[level] = {'ids': [], 'data': []}
            
            signatures_by_level[level]['ids'].append(sig_id)
            signatures_by_level[level]['data'].append(sig_info['data'])
        
        # Verify each level
        results_by_level = {}
        verified_levels = set()
        
        for level, threshold in self.level_thresholds.items():
            if level in signatures_by_level:
                # Get signatures for this level
                level_sigs = signatures_by_level[level]
                
                # Verify using the threshold verifier for this level
                level_result = self.level_verifiers[level].verify(
                    level_sigs['ids'], level_sigs['data']
                )
                
                # Track results
                results_by_level[level] = level_result
                if level_result['verified']:
                    verified_levels.add(level)
            else:
                # No signatures for this level
                results_by_level[level] = {
                    'verified': False,
                    'reason': f"No signatures provided for level '{level}'",
                    'valid_count': 0,
                    'total_count': 0,
                    'details': {}
                }
        
        # Determine overall verification result
        # A signature is verified if ANY level is verified
        verified = len(verified_levels) > 0
        
        return {
            'verified': verified,
            'verified_levels': list(verified_levels),
            'level_results': results_by_level
        }


class MultiSignatureManager:
    """Core class for managing multi-signature authentication"""
    
    def __init__(self, verification_service: Optional[VerificationService] = None,
                combination_mode: str = 'weighted', 
                verification_mode: str = 'threshold',
                threshold: int = 2,
                level_thresholds: Optional[Dict[str, int]] = None):
        """
        Initialize a multi-signature manager
        
        Args:
            verification_service: Optional verification service for blockchain verification
            combination_mode: Mode for combining signatures
            verification_mode: Verification mode ('threshold' or 'hierarchical')
            threshold: Threshold for threshold verification mode
            level_thresholds: Level thresholds for hierarchical verification mode
        """
        self.verification_service = verification_service
        self.combination_mode = combination_mode
        self.verification_mode = verification_mode
        
        # Initialize combiners and verifiers
        self.signature_combiner = SignatureCombiner(combination_mode)
        
        # Set up verifiers
        if verification_mode == 'threshold':
            self.verifier = ThresholdVerifier(threshold, verification_service)
        elif verification_mode == 'hierarchical':
            if level_thresholds is None:
                level_thresholds = {'high': 1, 'medium': 2, 'low': 3}
            self.verifier = HierarchicalVerifier(level_thresholds, verification_service)
        else:
            logger.warning(f"Unsupported verification mode: {verification_mode}")
            logger.warning("Using default mode: 'threshold'")
            self.verification_mode = 'threshold'
            self.verifier = ThresholdVerifier(threshold, verification_service)
    
    def register_multi_signature(self, signatures: List[List[float]], 
                               metadata: Optional[Dict[str, Any]] = None,
                               weights: Optional[List[float]] = None) -> str:
        """
        Register a combined multi-signature
        
        Args:
            signatures: List of signatures to combine
            metadata: Optional metadata for the combined signature
            weights: Optional weights for signature combination
            
        Returns:
            str: ID of the registered multi-signature
        """
        # Combine signatures
        combined_signature = self.signature_combiner.combine(signatures, weights)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'type': 'multi_signature',
            'combination_mode': self.combination_mode,
            'component_count': len(signatures),
            'timestamp': time.time()
        })
        
        # Register the combined signature
        if self.verification_service:
            signature_id = self.verification_service.registry.register_signature(
                combined_signature, metadata
            )
            logger.info(f"Multi-signature registered with ID: {signature_id}")
            return signature_id
        else:
            # Generate a mock ID for demo purposes
            mock_id = hashlib.sha256(str(combined_signature).encode()).hexdigest()
            logger.info(f"Multi-signature registered with mock ID: {mock_id}")
            return mock_id
    
    def verify_multi_signature(self, signature_ids: List[str], 
                             signature_data: List[List[float]]) -> Dict[str, Any]:
        """
        Verify a multi-signature
        
        Args:
            signature_ids: List of signature IDs to verify
            signature_data: List of signature data to verify
            
        Returns:
            Dict[str, Any]: Verification result
        """
        if self.verification_mode == 'threshold':
            return self.verifier.verify(signature_ids, signature_data)
        elif self.verification_mode == 'hierarchical':
            # Convert to the format expected by HierarchicalVerifier
            signature_map = {}
            for i, (sig_id, sig_data) in enumerate(zip(signature_ids, signature_data)):
                # For demo purposes, assign levels based on index
                if i == 0:
                    level = 'high'
                elif i < len(signature_ids) // 2:
                    level = 'medium'
                else:
                    level = 'low'
                
                signature_map[sig_id] = {
                    'level': level,
                    'data': sig_data
                }
            
            return self.verifier.verify(signature_map)
        else:
            return {'verified': False, 'reason': f"Unsupported verification mode: {self.verification_mode}"}
    
    def create_authentication_token(self, signatures: List[List[float]], 
                                  metadata: Optional[Dict[str, Any]] = None,
                                  expiration: int = 3600) -> Dict[str, Any]:
        """
        Create an authentication token from multiple signatures
        
        Args:
            signatures: List of signatures to combine
            metadata: Optional metadata for the token
            expiration: Token expiration time in seconds
            
        Returns:
            Dict[str, Any]: Authentication token
        """
        # Combine signatures
        combined_signature = self.signature_combiner.combine(signatures)
        
        # Create token
        token = {
            'signature': combined_signature,
            'created_at': time.time(),
            'expires_at': time.time() + expiration,
            'type': 'multi_signature',
            'combination_mode': self.combination_mode
        }
        
        if metadata:
            token['metadata'] = metadata
        
        # Generate a token ID
        token_id = hashlib.sha256(str(token).encode()).hexdigest()
        token['id'] = token_id
        
        return token
    
    def verify_authentication_token(self, token: Dict[str, Any], 
                                  signatures: List[List[float]]) -> bool:
        """
        Verify an authentication token
        
        Args:
            token: Authentication token to verify
            signatures: List of signatures to verify against
            
        Returns:
            bool: True if token is valid
        """
        # Check expiration
        if token.get('expires_at', 0) < time.time():
            logger.warning("Token has expired")
            return False
        
        # Extract token signature
        token_signature = token.get('signature')
        if not token_signature:
            logger.warning("Token does not contain a signature")
            return False
        
        # Combine provided signatures using the same mode as the token
        combiner = SignatureCombiner(token.get('combination_mode', self.combination_mode))
        test_signature = combiner.combine(signatures)
        
        # Compare signatures
        # In a real implementation, this would use a proper signature comparison algorithm
        # For demo purposes, we'll use a simple similarity check
        similarity = self._calculate_similarity(token_signature, test_signature)
        
        # Check if similarity exceeds threshold
        return similarity > 0.8
    
    def _calculate_similarity(self, sig1: List[float], sig2: List[float]) -> float:
        """
        Calculate similarity between two signatures
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Ensure same length
        max_length = max(len(sig1), len(sig2))
        
        if NUMPY_AVAILABLE:
            # Pad signatures to same length
            padded_sig1 = np.zeros(max_length)
            padded_sig1[:len(sig1)] = sig1
            
            padded_sig2 = np.zeros(max_length)
            padded_sig2[:len(sig2)] = sig2
            
            # Calculate cosine similarity
            dot_product = np.dot(padded_sig1, padded_sig2)
            norm1 = np.linalg.norm(padded_sig1)
            norm2 = np.linalg.norm(padded_sig2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
                
            return dot_product / (norm1 * norm2)
        else:
            # Manual implementation of cosine similarity
            padded_sig1 = [0.0] * max_length
            for i, val in enumerate(sig1):
                padded_sig1[i] = val
                
            padded_sig2 = [0.0] * max_length
            for i, val in enumerate(sig2):
                padded_sig2[i] = val
            
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(padded_sig1, padded_sig2))
            
            # Calculate magnitudes
            mag1 = sum(a * a for a in padded_sig1) ** 0.5
            mag2 = sum(b * b for b in padded_sig2) ** 0.5
            
            if mag1 == 0 or mag2 == 0:
                return 0
                
            return dot_product / (mag1 * mag2)
