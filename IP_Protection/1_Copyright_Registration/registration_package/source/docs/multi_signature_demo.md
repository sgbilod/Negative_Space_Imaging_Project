# Documentation for multi_signature_demo.py

```python
"""
Multi-Signature Authentication Demo

This script demonstrates how to use the multi-signature authentication system
for combining and verifying multiple negative space signatures.

Usage:
    python multi_signature_demo.py [--mode {generate|register|verify|threshold|hierarchical}]
                                  [--signatures N]
                                  [--threshold M]
                                  [--provider_url URL]
                                  [--contract_address ADDRESS]

Examples:
    # Generate and register a multi-signature
    python multi_signature_demo.py --mode register --signatures 3

    # Verify a multi-signature using threshold verification
    python multi_signature_demo.py --mode threshold --signatures 5 --threshold 3

    # Verify a multi-signature using hierarchical verification
    python multi_signature_demo.py --mode hierarchical
"""

import os
import sys
import time
import json
import argparse
import logging
import random
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir))
sys.path.append(project_root)

# Import project modules
try:
    from src.authentication.multi_signature import (
        MultiSignatureManager, SignatureCombiner, 
        ThresholdVerifier, HierarchicalVerifier
    )
    from src.blockchain.smart_contracts import (
        SmartContractManager, SignatureRegistry, VerificationService
    )
    from src.utils.fallbacks import (
        np, WEB3_AVAILABLE, web3 as w3, NUMPY_AVAILABLE
    )
    from simplified_demo import SimplePointCloud, generate_test_scene
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure you have installed all required dependencies.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiSignatureDemo:
    """Demo for multi-signature authentication using negative space signatures"""
    
    def __init__(self, provider_url=None, contract_address=None, 
                output_dir='output/multi_sig'):
        """
        Initialize the multi-signature demo
        
        Args:
            provider_url: URL of the Ethereum provider
            contract_address: Address of the deployed contract
            output_dir: Directory for storing output files
        """
        self.provider_url = provider_url
        self.contract_address = contract_address
        self.output_dir = os.path.join(project_root, output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize blockchain components if blockchain support is available
        if WEB3_AVAILABLE and provider_url:
            # Initialize smart contract components
            self.contract_manager = SmartContractManager(
                provider_url=provider_url,
                contract_address=contract_address
            )
            
            self.signature_registry = SignatureRegistry(self.contract_manager)
            self.verification_service = VerificationService(self.signature_registry)
        else:
            logger.warning("Blockchain support not available or not configured")
            logger.warning("Using local simulation for blockchain operations")
            
            self.contract_manager = None
            self.signature_registry = None
            self.verification_service = None
        
        # Initialize multi-signature components
        self.signature_combiner = SignatureCombiner()
        
        # Create managers for different verification modes
        self.threshold_manager = MultiSignatureManager(
            verification_service=self.verification_service,
            verification_mode='threshold',
            threshold=2
        )
        
        self.hierarchical_manager = MultiSignatureManager(
            verification_service=self.verification_service,
            verification_mode='hierarchical',
            level_thresholds={'high': 1, 'medium': 2, 'low': 3}
        )
    
    def generate_signatures(self, count: int = 3) -> List[List[float]]:
        """
        Generate multiple test signatures
        
        Args:
            count: Number of signatures to generate
            
        Returns:
            List[List[float]]: List of generated signatures
        """
        logger.info(f"Generating {count} test signatures...")
        
        signatures = []
        for i in range(count):
            # Generate a test signature
            dimensions = random.choice([64, 96, 128])
            
            if NUMPY_AVAILABLE:
                signature = np.random.rand(dimensions).tolist()
            else:
                signature = [random.random() for _ in range(dimensions)]
            
            signatures.append(signature)
            logger.info(f"  Generated signature {i+1} with {dimensions} dimensions")
        
        # Save signatures
        self._save_signatures(signatures)
        
        return signatures
    
    def _save_signatures(self, signatures: List[List[float]]):
        """
        Save signatures to a file
        
        Args:
            signatures: List of signatures to save
        """
        timestamp = int(time.time())
        filepath = os.path.join(self.output_dir, f"signatures_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump({
                'signatures': signatures,
                'timestamp': timestamp,
                'count': len(signatures)
            }, f, indent=2)
        
        logger.info(f"Signatures saved to: {filepath}")
    
    def load_signatures(self) -> List[List[float]]:
        """
        Load the most recent signatures from file
        
        Returns:
            List[List[float]]: List of loaded signatures
        """
        # Find the most recent signatures file
        signature_files = [f for f in os.listdir(self.output_dir) 
                          if f.startswith("signatures_") and f.endswith(".json")]
        
        if not signature_files:
            logger.warning("No signature files found")
            return []
        
        # Sort by timestamp and get the most recent
        latest_file = sorted(signature_files)[-1]
        filepath = os.path.join(self.output_dir, latest_file)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            signatures = data.get('signatures', [])
            logger.info(f"Loaded {len(signatures)} signatures from: {filepath}")
            
            return signatures
        except Exception as e:
            logger.error(f"Error loading signatures: {e}")
            return []
    
    def register_multi_signature(self, signatures: List[List[float]] = None,
                               combination_mode: str = 'weighted') -> str:
        """
        Register a multi-signature
        
        Args:
            signatures: List of signatures to combine (loads from file if None)
            combination_mode: Mode for combining signatures
            
        Returns:
            str: ID of the registered multi-signature
        """
        # Load signatures if not provided
        if signatures is None:
            signatures = self.load_signatures()
            
            if not signatures:
                logger.error("No signatures available for registration")
                return None
        
        logger.info(f"Registering multi-signature from {len(signatures)} signatures...")
        logger.info(f"Using combination mode: {combination_mode}")
        
        # Create a combiner with the specified mode
        combiner = SignatureCombiner(combination_mode)
        
        # Combine signatures
        combined_signature = combiner.combine(signatures)
        
        # Create metadata
        metadata = {
            'type': 'multi_signature',
            'combination_mode': combination_mode,
            'component_count': len(signatures),
            'dimensions': len(combined_signature),
            'timestamp': time.time()
        }
        
        # Register the multi-signature
        if self.signature_registry:
            # Use blockchain registration
            signature_id = self.signature_registry.register_signature(
                combined_signature, metadata
            )
        else:
            # Generate a mock ID for demo purposes
            signature_id = hashlib.sha256(str(combined_signature).encode()).hexdigest()
        
        logger.info(f"Multi-signature registered with ID: {signature_id}")
        
        # Save registration info
        self._save_registration(signature_id, combined_signature, metadata)
        
        return signature_id
    
    def _save_registration(self, signature_id: str, 
                         signature_data: List[float], 
                         metadata: Dict[str, Any]):
        """
        Save registration information
        
        Args:
            signature_id: ID of the registered signature
            signature_data: Signature data
            metadata: Signature metadata
        """
        registration = {
            'signature_id': signature_id,
            'signature_data': signature_data,
            'metadata': metadata
        }
        
        filepath = os.path.join(
            self.output_dir, f"registration_{int(time.time())}.json"
        )
        
        with open(filepath, 'w') as f:
            json.dump(registration, f, indent=2)
        
        logger.info(f"Registration saved to: {filepath}")
    
    def load_registration(self) -> Dict[str, Any]:
        """
        Load the most recent registration
        
        Returns:
            Dict[str, Any]: Registration information
        """
        # Find the most recent registration file
        registration_files = [f for f in os.listdir(self.output_dir) 
                             if f.startswith("registration_") and f.endswith(".json")]
        
        if not registration_files:
            logger.warning("No registration files found")
            return {}
        
        # Sort by timestamp and get the most recent
        latest_file = sorted(registration_files)[-1]
        filepath = os.path.join(self.output_dir, latest_file)
        
        try:
            with open(filepath, 'r') as f:
                registration = json.load(f)
            
            logger.info(f"Loaded registration from: {filepath}")
            
            return registration
        except Exception as e:
            logger.error(f"Error loading registration: {e}")
            return {}
    
    def verify_threshold(self, signatures: List[List[float]] = None, 
                       threshold: int = 2) -> Dict[str, Any]:
        """
        Verify signatures using threshold verification
        
        Args:
            signatures: List of signatures to verify (loads from file if None)
            threshold: Threshold for verification (M of N signatures)
            
        Returns:
            Dict[str, Any]: Verification result
        """
        # Load signatures if not provided
        if signatures is None:
            signatures = self.load_signatures()
            
            if not signatures:
                logger.error("No signatures available for verification")
                return {'verified': False, 'reason': 'No signatures available'}
        
        logger.info(f"Verifying {len(signatures)} signatures with threshold {threshold}...")
        
        # Create signature IDs (for demo, we'll generate these)
        signature_ids = []
        for i, sig in enumerate(signatures):
            # Generate a deterministic ID based on the signature
            sig_id = hashlib.sha256(str(sig).encode()).hexdigest()
            signature_ids.append(sig_id)
        
        # Create a threshold verifier
        verifier = ThresholdVerifier(threshold, self.verification_service)
        
        # Verify signatures
        result = verifier.verify(signature_ids, signatures)
        
        # Log result
        if result['verified']:
            logger.info("✅ Threshold verification successful!")
            logger.info(f"  {result['valid_count']} of {result['total_count']} signatures valid")
        else:
            logger.info("❌ Threshold verification failed!")
            logger.info(f"  {result['valid_count']} of {result['total_count']} signatures valid")
            logger.info(f"  Threshold: {threshold}")
        
        # Save result
        self._save_verification_result(result, 'threshold')
        
        return result
    
    def verify_hierarchical(self, signatures: List[List[float]] = None) -> Dict[str, Any]:
        """
        Verify signatures using hierarchical verification
        
        Args:
            signatures: List of signatures to verify (loads from file if None)
            
        Returns:
            Dict[str, Any]: Verification result
        """
        # Load signatures if not provided
        if signatures is None:
            signatures = self.load_signatures()
            
            if not signatures:
                logger.error("No signatures available for verification")
                return {'verified': False, 'reason': 'No signatures available'}
        
        logger.info(f"Verifying {len(signatures)} signatures with hierarchical approach...")
        
        # Create signature map with different priority levels
        signature_map = {}
        for i, sig in enumerate(signatures):
            # Generate a deterministic ID based on the signature
            sig_id = hashlib.sha256(str(sig).encode()).hexdigest()
            
            # Assign priority level based on index
            if i == 0:
                level = 'high'
            elif i < len(signatures) // 2:
                level = 'medium'
            else:
                level = 'low'
            
            signature_map[sig_id] = {
                'level': level,
                'data': sig
            }
        
        # Create level thresholds
        level_thresholds = {
            'high': 1,  # Need 1 high-priority signature
            'medium': 2,  # Need 2 medium-priority signatures
            'low': 3  # Need 3 low-priority signatures
        }
        
        # Create a hierarchical verifier
        verifier = HierarchicalVerifier(level_thresholds, self.verification_service)
        
        # Verify signatures
        result = verifier.verify(signature_map)
        
        # Log result
        if result['verified']:
            logger.info("✅ Hierarchical verification successful!")
            logger.info(f"  Verified levels: {', '.join(result['verified_levels'])}")
        else:
            logger.info("❌ Hierarchical verification failed!")
            logger.info("  No levels were verified")
        
        # Save result
        self._save_verification_result(result, 'hierarchical')
        
        return result
    
    def _save_verification_result(self, result: Dict[str, Any], mode: str):
        """
        Save verification result
        
        Args:
            result: Verification result
            mode: Verification mode ('threshold' or 'hierarchical')
        """
        filepath = os.path.join(
            self.output_dir, f"{mode}_verification_{int(time.time())}.json"
        )
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Verification result saved to: {filepath}")
    
    def create_authentication_token(self, signatures: List[List[float]] = None,
                                  expiration: int = 3600) -> Dict[str, Any]:
        """
        Create an authentication token from multiple signatures
        
        Args:
            signatures: List of signatures to combine (loads from file if None)
            expiration: Token expiration time in seconds
            
        Returns:
            Dict[str, Any]: Authentication token
        """
        # Load signatures if not provided
        if signatures is None:
            signatures = self.load_signatures()
            
            if not signatures:
                logger.error("No signatures available for token creation")
                return None
        
        logger.info(f"Creating authentication token from {len(signatures)} signatures...")
        
        # Create metadata
        metadata = {
            'created_at': time.time(),
            'expires_at': time.time() + expiration,
            'signature_count': len(signatures)
        }
        
        # Use the threshold manager to create a token
        token = self.threshold_manager.create_authentication_token(
            signatures, metadata, expiration
        )
        
        logger.info(f"Authentication token created with ID: {token['id']}")
        
        # Save token
        self._save_token(token)
        
        return token
    
    def _save_token(self, token: Dict[str, Any]):
        """
        Save authentication token
        
        Args:
            token: Authentication token to save
        """
        filepath = os.path.join(
            self.output_dir, f"token_{int(time.time())}.json"
        )
        
        with open(filepath, 'w') as f:
            json.dump(token, f, indent=2)
        
        logger.info(f"Authentication token saved to: {filepath}")
    
    def load_token(self) -> Dict[str, Any]:
        """
        Load the most recent authentication token
        
        Returns:
            Dict[str, Any]: Authentication token
        """
        # Find the most recent token file
        token_files = [f for f in os.listdir(self.output_dir) 
                      if f.startswith("token_") and f.endswith(".json")]
        
        if not token_files:
            logger.warning("No token files found")
            return {}
        
        # Sort by timestamp and get the most recent
        latest_file = sorted(token_files)[-1]
        filepath = os.path.join(self.output_dir, latest_file)
        
        try:
            with open(filepath, 'r') as f:
                token = json.load(f)
            
            logger.info(f"Loaded token from: {filepath}")
            
            return token
        except Exception as e:
            logger.error(f"Error loading token: {e}")
            return {}
    
    def verify_token(self, token: Dict[str, Any] = None, 
                   signatures: List[List[float]] = None) -> bool:
        """
        Verify an authentication token
        
        Args:
            token: Authentication token to verify (loads from file if None)
            signatures: List of signatures to verify against (loads from file if None)
            
        Returns:
            bool: True if token is valid
        """
        # Load token if not provided
        if token is None:
            token = self.load_token()
            
            if not token:
                logger.error("No token available for verification")
                return False
        
        # Load signatures if not provided
        if signatures is None:
            signatures = self.load_signatures()
            
            if not signatures:
                logger.error("No signatures available for token verification")
                return False
        
        logger.info(f"Verifying authentication token against {len(signatures)} signatures...")
        
        # Check expiration
        if token.get('expires_at', 0) < time.time():
            logger.warning("❌ Token has expired")
            return False
        
        # Use the threshold manager to verify the token
        is_valid = self.threshold_manager.verify_authentication_token(token, signatures)
        
        if is_valid:
            logger.info("✅ Token verification successful!")
        else:
            logger.info("❌ Token verification failed!")
        
        return is_valid


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-Signature Authentication Demo')
    
    parser.add_argument('--mode', choices=['generate', 'register', 'verify', 'threshold', 'hierarchical'],
                       default='generate', help='Demo mode')
    
    parser.add_argument('--signatures', type=int, default=3,
                       help='Number of signatures to generate')
    
    parser.add_argument('--threshold', type=int, default=2,
                       help='Threshold for verification (M of N signatures)')
    
    parser.add_argument('--provider_url', type=str, default=None,
                       help='Ethereum provider URL')
    
    parser.add_argument('--contract_address', type=str, default=None,
                       help='Address of the deployed contract')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create demo
    demo = MultiSignatureDemo(
        provider_url=args.provider_url,
        contract_address=args.contract_address
    )
    
    # Run demo mode
    if args.mode == 'generate':
        # Generate signatures
        signatures = demo.generate_signatures(args.signatures)
        
        # Create token
        token = demo.create_authentication_token(signatures)
        
    elif args.mode == 'register':
        # Generate signatures if needed
        signatures = demo.load_signatures()
        if not signatures:
            signatures = demo.generate_signatures(args.signatures)
        
        # Register multi-signature
        demo.register_multi_signature(signatures)
        
    elif args.mode == 'verify':
        # Load token
        token = demo.load_token()
        
        if not token:
            # Generate signatures and create token
            signatures = demo.generate_signatures(args.signatures)
            token = demo.create_authentication_token(signatures)
        
        # Verify token
        demo.verify_token(token)
        
    elif args.mode == 'threshold':
        # Generate signatures if needed
        signatures = demo.load_signatures()
        if not signatures:
            signatures = demo.generate_signatures(args.signatures)
        
        # Verify using threshold approach
        demo.verify_threshold(signatures, args.threshold)
        
    elif args.mode == 'hierarchical':
        # Generate signatures if needed
        signatures = demo.load_signatures()
        if not signatures:
            signatures = demo.generate_signatures(args.signatures)
        
        # Verify using hierarchical approach
        demo.verify_hierarchical(signatures)


if __name__ == '__main__':
    main()

```