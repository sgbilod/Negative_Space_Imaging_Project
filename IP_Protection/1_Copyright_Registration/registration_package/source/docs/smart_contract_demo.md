# Documentation for smart_contract_demo.py

```python
"""
Blockchain Smart Contract Integration Demo for Negative Space Signatures

This script demonstrates how to use the smart contract integration module
to register and verify negative space signatures.

Usage:
    python smart_contract_demo.py [--mode {register|verify|list|generate}]
                                 [--signature_id SIGNATURE_ID]
                                 [--provider_url PROVIDER_URL]
                                 [--contract_address CONTRACT_ADDRESS]

Examples:
    # Generate a test signature
    python smart_contract_demo.py --mode generate

    # Register a signature
    python smart_contract_demo.py --mode register

    # Verify a signature
    python smart_contract_demo.py --mode verify --signature_id 0x1234...

    # List registered signatures
    python smart_contract_demo.py --mode list
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import hashlib
import random

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir))
sys.path.append(project_root)

# Import from blockchain module
try:
    from src.blockchain.smart_contracts import (
        SmartContractManager, SignatureRegistry, VerificationService
    )

    # Import from utils.fallbacks
    from src.utils.fallbacks import (
        np, WEB3_AVAILABLE, web3 as w3, NUMPY_AVAILABLE
    )

    # Import demo utilities
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


class BlockchainDemo:
    """Blockchain integration demo for negative space signatures"""
    
    def __init__(self, provider_url: str = None, 
                contract_address: str = None,
                private_key: str = None):
        """
        Initialize a blockchain demo
        
        Args:
            provider_url: URL of the Ethereum provider (optional)
            contract_address: Address of the deployed contract (optional)
            private_key: Private key for transaction signing (optional)
        """
        self.provider_url = provider_url
        self.contract_address = contract_address
        self.private_key = private_key
        
        # Create contract manager
        self.contract_manager = SmartContractManager(
            provider_url=provider_url,
            private_key=private_key,
            contract_address=contract_address
        )
        
        # Initialize or deploy contract
        self._initialize_contract()
        
        # Create signature registry
        self.signature_registry = SignatureRegistry(self.contract_manager)
        
        # Create verification service
        self.verification_service = VerificationService(self.signature_registry)
        
        # Storage for generated test signatures
        self.test_signatures = {}
    
    def _initialize_contract(self):
        """Initialize or deploy contract"""
        if not self.contract_manager.is_initialized():
            if self.contract_address:
                # Load existing contract
                logger.info(f"Loading contract from address: {self.contract_address}")
                self.contract_manager.compile_contract()
                self.contract_manager.load_contract(self.contract_address)
            else:
                # Deploy new contract
                logger.info("Deploying new contract...")
                self.contract_manager.compile_contract()
                self.contract_address = self.contract_manager.deploy_contract()
                
                if self.contract_address:
                    logger.info(f"Contract deployed to address: {self.contract_address}")
                else:
                    logger.warning("Contract deployment failed, using fallback mode")
    
    def generate_test_signature(self, dimensions: int = 128, 
                              save_to_file: bool = True) -> Tuple[List[float], Dict]:
        """
        Generate a test signature
        
        Args:
            dimensions: Number of dimensions for the signature
            save_to_file: Whether to save the signature to a file
            
        Returns:
            Tuple[List[float], Dict]: Signature data and metadata
        """
        logger.info(f"Generating a test signature with {dimensions} dimensions...")
        
        # Generate random signature data
        if NUMPY_AVAILABLE:
            signature_data = np.random.rand(dimensions).tolist()
        else:
            signature_data = [random.random() for _ in range(dimensions)]
        
        # Generate metadata
        metadata = {
            'source': 'test_generator',
            'dimensions': dimensions,
            'description': 'Test signature for blockchain demo',
            'timestamp': time.time(),
            'content_type': 'random_values'
        }
        
        # Save to file if requested
        if save_to_file:
            self._save_signature_to_file(signature_data, metadata)
        
        # Store in memory
        signature_key = hashlib.md5(str(time.time()).encode()).hexdigest()
        self.test_signatures[signature_key] = {
            'data': signature_data,
            'metadata': metadata
        }
        
        logger.info(f"Test signature generated with key: {signature_key}")
        
        return signature_data, metadata
    
    def _save_signature_to_file(self, signature_data: List[float], metadata: Dict):
        """
        Save a signature to a file
        
        Args:
            signature_data: Signature data
            metadata: Signature metadata
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.join(project_root, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for filename
        timestamp = int(time.time())
        
        # Save signature data
        signature_file = os.path.join(output_dir, f'test_signature_{timestamp}.json')
        
        with open(signature_file, 'w') as f:
            json.dump({
                'signature_data': signature_data,
                'metadata': metadata
            }, f, indent=2)
        
        logger.info(f"Signature saved to: {signature_file}")
    
    def load_signature_from_file(self, filepath: str) -> Tuple[List[float], Dict]:
        """
        Load a signature from a file
        
        Args:
            filepath: Path to the signature file
            
        Returns:
            Tuple[List[float], Dict]: Signature data and metadata
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            signature_data = data.get('signature_data', [])
            metadata = data.get('metadata', {})
            
            logger.info(f"Signature loaded from: {filepath}")
            
            return signature_data, metadata
            
        except Exception as e:
            logger.error(f"Error loading signature: {e}")
            return [], {}
    
    def register_signature(self, signature_data: List[float] = None, 
                         metadata: Dict = None) -> str:
        """
        Register a signature on the blockchain
        
        Args:
            signature_data: Signature data (optional, generates if None)
            metadata: Signature metadata (optional, generates if None)
            
        Returns:
            str: Signature ID
        """
        # Generate signature if not provided
        if signature_data is None or metadata is None:
            signature_data, metadata = self.generate_test_signature()
        
        logger.info("Registering signature on the blockchain...")
        
        # Register the signature
        signature_id = self.signature_registry.register_signature(signature_data, metadata)
        
        if signature_id:
            logger.info(f"Signature registered with ID: {signature_id}")
            
            # Save signature ID for later verification
            output_dir = os.path.join(project_root, 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save signature mapping
            signature_key = hashlib.md5(str(signature_data).encode()).hexdigest()
            mapping_file = os.path.join(output_dir, 'signature_mapping.json')
            
            try:
                # Load existing mapping if available
                if os.path.exists(mapping_file):
                    with open(mapping_file, 'r') as f:
                        mapping = json.load(f)
                else:
                    mapping = {}
                
                # Add new mapping
                mapping[signature_key] = {
                    'signature_id': signature_id,
                    'metadata': metadata,
                    'timestamp': time.time()
                }
                
                # Save updated mapping
                with open(mapping_file, 'w') as f:
                    json.dump(mapping, f, indent=2)
                
                logger.info(f"Signature mapping saved to: {mapping_file}")
                
            except Exception as e:
                logger.error(f"Error saving signature mapping: {e}")
        
        else:
            logger.error("Failed to register signature")
        
        return signature_id
    
    def verify_signature(self, signature_id: str, signature_data: List[float] = None) -> bool:
        """
        Verify a signature on the blockchain
        
        Args:
            signature_id: ID of the signature to verify
            signature_data: Signature data to verify (optional, loads from file if None)
            
        Returns:
            bool: True if signature is valid
        """
        # If signature data not provided, try to load from file
        if signature_data is None:
            signature_data = self._find_signature_data_for_id(signature_id)
            
            if not signature_data:
                logger.error(f"No signature data found for ID: {signature_id}")
                return False
        
        logger.info(f"Verifying signature with ID: {signature_id}")
        
        # Verify the signature
        result = self.verification_service.verify_signature(signature_id, signature_data)
        
        # Display result
        if result['isValid']:
            logger.info("✅ Signature is valid!")
        else:
            logger.info("❌ Signature is not valid!")
        
        # Display details
        logger.info("Verification details:")
        for key, value in result['details'].items():
            logger.info(f"  {key}: {value}")
        
        return result['isValid']
    
    def _find_signature_data_for_id(self, signature_id: str) -> List[float]:
        """
        Find signature data for a given ID
        
        Args:
            signature_id: Signature ID to find data for
            
        Returns:
            List[float]: Signature data if found, None otherwise
        """
        # Check if we have the signature in memory
        for key, signature in self.test_signatures.items():
            if key == signature_id:
                return signature['data']
        
        # Try to load from file
        mapping_file = os.path.join(project_root, 'output', 'signature_mapping.json')
        
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                
                # Check if signature ID is in mapping
                for key, value in mapping.items():
                    if value.get('signature_id') == signature_id:
                        # Try to load the associated signature file
                        timestamp = int(value.get('timestamp', 0))
                        signature_file = os.path.join(
                            project_root, 'output', f'test_signature_{timestamp}.json'
                        )
                        
                        if os.path.exists(signature_file):
                            signature_data, _ = self.load_signature_from_file(signature_file)
                            return signature_data
            
            except Exception as e:
                logger.error(f"Error loading signature mapping: {e}")
        
        # Generate a random signature as a fallback
        logger.warning("Generating random signature data for verification (will likely fail)")
        if NUMPY_AVAILABLE:
            return np.random.rand(128).tolist()
        else:
            return [random.random() for _ in range(128)]
    
    def list_signatures(self) -> List[Dict]:
        """
        List registered signatures
        
        Returns:
            List[Dict]: List of signature details
        """
        logger.info("Fetching registered signatures...")
        
        # Get signature count
        count = self.signature_registry.get_signature_count()
        logger.info(f"Found {count} registered signatures")
        
        if count == 0:
            return []
        
        # Get signature IDs
        signature_ids = self.signature_registry.get_signature_ids(0, count)
        
        # Get signature details
        signatures = []
        for signature_id in signature_ids:
            signature = self.signature_registry.get_signature(signature_id)
            if signature:
                signatures.append({
                    'id': signature_id,
                    'registeredBy': signature.get('registeredBy', 'unknown'),
                    'timestamp': signature.get('timestamp', 0),
                    'isRevoked': signature.get('isRevoked', False),
                    'metadata': self._parse_metadata(signature.get('metadata', '{}'))
                })
        
        # Display signatures
        for i, signature in enumerate(signatures):
            logger.info(f"Signature {i+1}:")
            logger.info(f"  ID: {signature['id']}")
            logger.info(f"  Registered by: {signature['registeredBy']}")
            logger.info(f"  Timestamp: {signature['timestamp']}")
            logger.info(f"  Revoked: {signature['isRevoked']}")
            logger.info(f"  Metadata: {signature['metadata']}")
        
        return signatures
    
    def _parse_metadata(self, metadata_str: str) -> Dict:
        """
        Parse metadata JSON string
        
        Args:
            metadata_str: Metadata JSON string
            
        Returns:
            Dict: Parsed metadata
        """
        try:
            return json.loads(metadata_str)
        except:
            return {}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Blockchain Integration Demo')
    
    parser.add_argument('--mode', choices=['register', 'verify', 'list', 'generate'],
                       default='generate', help='Demo mode')
    
    parser.add_argument('--signature_id', type=str, default=None,
                       help='Signature ID for verification')
    
    parser.add_argument('--provider_url', type=str, default=None,
                       help='Ethereum provider URL')
    
    parser.add_argument('--contract_address', type=str, default=None,
                       help='Address of the deployed contract')
    
    parser.add_argument('--private_key', type=str, default=None,
                       help='Private key for transaction signing')
    
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save results')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create demo
    demo = BlockchainDemo(
        provider_url=args.provider_url,
        contract_address=args.contract_address,
        private_key=args.private_key
    )
    
    # Run demo mode
    if args.mode == 'generate':
        # Generate test signature
        demo.generate_test_signature()
        
    elif args.mode == 'register':
        # Register a signature
        demo.register_signature()
        
    elif args.mode == 'verify':
        # Verify a signature
        if not args.signature_id:
            logger.error("Signature ID is required for verification")
            return
        
        demo.verify_signature(args.signature_id)
        
    elif args.mode == 'list':
        # List signatures
        demo.list_signatures()


if __name__ == '__main__':
    main()

```