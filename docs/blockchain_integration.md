# Documentation for blockchain_integration.py

```python
"""
Blockchain integration for Negative Space Signatures

This module provides functionality for creating and verifying digital
signatures based on negative space analysis, and integrating with blockchain
technologies for secure, tamper-proof records.

Classes:
    NegativeSpaceHasher: Creates unique hashes from negative space signatures
    BlockchainConnector: Connects to blockchain networks
    SignatureVerifier: Verifies the authenticity of negative space signatures
"""

import os
import sys
import hashlib
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NegativeSpaceHasher:
    """Creates unique cryptographic hashes from negative space signatures"""
    
    def __init__(self, hash_algorithm: str = 'sha256'):
        """
        Initialize the hasher
        
        Args:
            hash_algorithm: The hashing algorithm to use
        """
        self.hash_algorithm = hash_algorithm
    
    def hash_signature(self, signature: np.ndarray) -> str:
        """
        Create a hash from a spatial signature
        
        Args:
            signature: The spatial signature to hash
            
        Returns:
            str: The hexadecimal hash string
        """
        # Convert numpy array to bytes
        signature_bytes = signature.tobytes()
        
        # Create hash
        if self.hash_algorithm == 'sha256':
            hash_obj = hashlib.sha256(signature_bytes)
        elif self.hash_algorithm == 'sha512':
            hash_obj = hashlib.sha512(signature_bytes)
        elif self.hash_algorithm == 'md5':
            hash_obj = hashlib.md5(signature_bytes)
        else:
            hash_obj = hashlib.sha256(signature_bytes)
        
        # Return hex digest
        return hash_obj.hexdigest()
    
    def hash_point_cloud(self, point_cloud: Any) -> str:
        """
        Create a hash from a point cloud
        
        Args:
            point_cloud: The point cloud to hash
            
        Returns:
            str: The hexadecimal hash string
        """
        if hasattr(point_cloud, 'compute_spatial_signature'):
            # Use the point cloud's spatial signature
            signature = point_cloud.compute_spatial_signature()
            return self.hash_signature(signature)
        elif hasattr(point_cloud, 'points') and isinstance(point_cloud.points, np.ndarray):
            # Use the point cloud's points directly
            return self.hash_signature(point_cloud.points)
        else:
            logger.warning("Point cloud doesn't have a compatible format for hashing")
            return "0" * 64  # Return a dummy hash
    
    def create_merkle_tree(self, signatures: List[np.ndarray]) -> Dict:
        """
        Create a Merkle tree from multiple signatures
        
        Args:
            signatures: List of signatures to include in the tree
            
        Returns:
            Dict: Merkle tree structure
        """
        if not signatures:
            return {"root": "0" * 64, "leaves": [], "tree": []}
        
        # Create leaf nodes (hashes of individual signatures)
        leaves = [self.hash_signature(sig) for sig in signatures]
        
        # Build the tree
        tree = leaves.copy()
        tree_levels = [leaves]
        
        # Continue until we reach the root
        while len(tree_levels[-1]) > 1:
            level = tree_levels[-1]
            next_level = []
            
            # Process pairs of nodes
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    # Hash the pair
                    combined = level[i] + level[i+1]
                    hash_obj = hashlib.sha256(combined.encode())
                    next_level.append(hash_obj.hexdigest())
                else:
                    # Odd number of nodes, promote the last one
                    next_level.append(level[i])
            
            tree_levels.append(next_level)
            tree.extend(next_level)
        
        # Root is the last node
        root = tree_levels[-1][0]
        
        return {
            "root": root,
            "leaves": leaves,
            "tree": tree
        }


class BlockchainConnector:
    """Connects to blockchain networks for storing and retrieving negative space hashes"""
    
    def __init__(self, blockchain_type: str = 'simulated'):
        """
        Initialize the blockchain connector
        
        Args:
            blockchain_type: Type of blockchain to connect to
                            ('simulated', 'ethereum', 'hyperledger', etc.)
        """
        self.blockchain_type = blockchain_type
        self.simulated_chain = []
        
        # In a real implementation, this would connect to a blockchain network
        logger.info(f"Initialized {blockchain_type} blockchain connector")
        
        # For a real blockchain connection, we would initialize appropriate libraries
        if blockchain_type == 'ethereum':
            try:
                # This would be replaced with actual Ethereum connection
                # import web3
                # self.web3 = web3.Web3()
                pass
            except ImportError:
                logger.warning("web3 package not available, using simulated blockchain")
                self.blockchain_type = 'simulated'
    
    def store_hash(self, hash_value: str, metadata: Dict = None) -> str:
        """
        Store a hash on the blockchain
        
        Args:
            hash_value: The hash to store
            metadata: Additional metadata to store with the hash
            
        Returns:
            str: Transaction ID or reference
        """
        if metadata is None:
            metadata = {}
        
        # Add timestamp
        metadata['timestamp'] = datetime.now().isoformat()
        
        if self.blockchain_type == 'simulated':
            # For simulation, just store in memory
            transaction_id = f"tx_{len(self.simulated_chain)}"
            self.simulated_chain.append({
                'transaction_id': transaction_id,
                'hash': hash_value,
                'metadata': metadata,
                'timestamp': metadata['timestamp']
            })
            logger.info(f"Stored hash {hash_value[:8]}... in simulated blockchain")
            return transaction_id
        elif self.blockchain_type == 'ethereum':
            # This would be implemented for real Ethereum connections
            transaction_id = f"eth_{hash_value[:16]}"
            logger.info(f"Simulating Ethereum transaction for hash {hash_value[:8]}...")
            return transaction_id
        else:
            logger.warning(f"Unsupported blockchain type: {self.blockchain_type}")
            return ""
    
    def verify_hash(self, hash_value: str) -> Dict:
        """
        Verify if a hash exists on the blockchain
        
        Args:
            hash_value: The hash to verify
            
        Returns:
            Dict: Verification result with metadata
        """
        if self.blockchain_type == 'simulated':
            # Search in simulated chain
            for tx in self.simulated_chain:
                if tx['hash'] == hash_value:
                    return {
                        'verified': True,
                        'transaction_id': tx['transaction_id'],
                        'timestamp': tx['timestamp'],
                        'metadata': tx['metadata']
                    }
            
            # Not found
            return {'verified': False}
        
        elif self.blockchain_type == 'ethereum':
            # This would be implemented for real Ethereum connections
            # For now, just return a simulated response
            logger.info(f"Simulating Ethereum verification for hash {hash_value[:8]}...")
            return {
                'verified': True,
                'transaction_id': f"eth_{hash_value[:16]}",
                'timestamp': datetime.now().isoformat(),
                'metadata': {}
            }
        
        else:
            logger.warning(f"Unsupported blockchain type: {self.blockchain_type}")
            return {'verified': False}
    
    def get_transaction_history(self) -> List[Dict]:
        """
        Get the transaction history
        
        Returns:
            List[Dict]: List of transactions
        """
        if self.blockchain_type == 'simulated':
            return self.simulated_chain
        else:
            # Would be implemented for real blockchain connections
            logger.info(f"Transaction history not implemented for {self.blockchain_type}")
            return []


class SignatureVerifier:
    """Verifies the authenticity of negative space signatures"""
    
    def __init__(self, hasher: NegativeSpaceHasher = None, 
                 blockchain: BlockchainConnector = None):
        """
        Initialize the verifier
        
        Args:
            hasher: The hasher to use for signature hashing
            blockchain: The blockchain connector to use for verification
        """
        self.hasher = hasher or NegativeSpaceHasher()
        self.blockchain = blockchain or BlockchainConnector()
    
    def register_signature(self, signature: np.ndarray, metadata: Dict = None) -> str:
        """
        Register a signature on the blockchain
        
        Args:
            signature: The spatial signature to register
            metadata: Additional metadata to store
            
        Returns:
            str: Transaction ID or reference
        """
        # Create hash
        hash_value = self.hasher.hash_signature(signature)
        
        # Store on blockchain
        transaction_id = self.blockchain.store_hash(hash_value, metadata)
        
        return transaction_id
    
    def verify_signature(self, signature: np.ndarray, threshold: float = 0.95) -> Dict:
        """
        Verify if a signature has been registered
        
        Args:
            signature: The spatial signature to verify
            threshold: Similarity threshold for fuzzy matching
            
        Returns:
            Dict: Verification result with metadata
        """
        # Create hash
        hash_value = self.hasher.hash_signature(signature)
        
        # Check blockchain
        result = self.blockchain.verify_hash(hash_value)
        
        # In a real implementation, we would also do fuzzy matching
        # to handle small variations in the signature
        
        return result
    
    def create_authentication_token(self, signature: np.ndarray, 
                                   expiration_seconds: int = 3600) -> Dict:
        """
        Create an authentication token based on a negative space signature
        
        Args:
            signature: The spatial signature to use
            expiration_seconds: Token validity period in seconds
            
        Returns:
            Dict: Authentication token
        """
        # Create hash
        hash_value = self.hasher.hash_signature(signature)
        
        # Generate expiration time
        expiration = datetime.now().timestamp() + expiration_seconds
        
        # Create token
        token = {
            'hash': hash_value,
            'created': datetime.now().isoformat(),
            'expires': datetime.fromtimestamp(expiration).isoformat(),
            'signature_dims': signature.shape
        }
        
        return token
    
    def verify_authentication_token(self, token: Dict, signature: np.ndarray) -> bool:
        """
        Verify an authentication token against a signature
        
        Args:
            token: The authentication token
            signature: The spatial signature to verify
            
        Returns:
            bool: True if token is valid and matches the signature
        """
        # Check expiration
        expiration = datetime.fromisoformat(token['expires']).timestamp()
        if datetime.now().timestamp() > expiration:
            logger.warning("Authentication token has expired")
            return False
        
        # Verify hash
        hash_value = self.hasher.hash_signature(signature)
        if hash_value != token['hash']:
            logger.warning("Authentication token does not match signature")
            return False
        
        return True

```