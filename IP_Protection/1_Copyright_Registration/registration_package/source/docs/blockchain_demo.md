# Documentation for blockchain_demo.py

```python
"""
Blockchain Integration Demo for Negative Space Imaging

This demo showcases how negative space signatures can be securely stored and verified
using blockchain technology, providing a foundation for unique digital authentication.

Usage:
    python blockchain_demo.py [--output_dir DIR] [--mode MODE]

Examples:
    # Run the full demo
    python blockchain_demo.py --output_dir output/blockchain
    
    # Generate and register a signature
    python blockchain_demo.py --mode register
    
    # Verify a signature
    python blockchain_demo.py --mode verify --signature_id <ID>
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import hashlib
import random

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import project modules
try:
    from src.blockchain.blockchain_integration import (
        NegativeSpaceHasher, BlockchainConnector, SignatureVerifier
    )
    from src.blockchain.smart_contracts import (
        SmartContractManager, SignatureRegistry, VerificationService
    )
    from src.utils.fallbacks import (
        np, WEB3_AVAILABLE, web3 as w3, NUMPY_AVAILABLE
    )
    from simplified_demo import SimplePointCloud, generate_test_scene
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running from the project root.")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Blockchain Integration for Negative Space")
    
    parser.add_argument('--output_dir', type=str, default='output/blockchain',
                       help='Directory to save results (default: output/blockchain)')
    
    parser.add_argument('--num_signatures', type=int, default=5,
                       help='Number of signatures to generate (default: 5)')
    
    parser.add_argument('--mode', type=str, default='full',
                      choices=['full', 'register', 'verify', 'list', 'generate'],
                      help='Demo mode (default: full)')
    
    parser.add_argument('--signature_id', type=str, default=None,
                      help='Signature ID for verification')
    
    parser.add_argument('--provider_url', type=str, default=None,
                      help='Ethereum provider URL')
    
    parser.add_argument('--contract_address', type=str, default=None,
                      help='Address of the deployed contract')
    
    return parser.parse_args()

def ensure_directory(directory):
    """Ensure a directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def visualize_blockchain_results(signatures, hashes, verifications, output_path):
    """
    Visualize blockchain verification results
    
    Args:
        signatures: List of signatures
        hashes: List of hashes
        verifications: List of verification results
        output_path: Path to save the visualization
    """
    # Without matplotlib, create a text-based visualization
    logger.info("Creating text-based visualization of blockchain results")
    
    # Save results as JSON
    with open(output_path, 'w') as f:
        json.dump({
            'signatures': [s.tolist() if hasattr(s, 'tolist') else s for s in signatures],
            'hashes': hashes,
            'verifications': verifications
        }, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    
    # Display text-based summary
    logger.info("Blockchain Verification Summary:")
    logger.info("=" * 50)
    
    for i, (sig, hash_val, verified) in enumerate(zip(signatures, hashes, verifications)):
        logger.info(f"Signature {i+1}:")
        logger.info(f"  Hash: {hash_val[:16]}...")
        logger.info(f"  Verified: {'✅' if verified else '❌'}")
        logger.info("-" * 50)
    
    # 2. Second subplot: Visualization of hashes
def run_blockchain_demo(args):
    """Run the blockchain integration demo"""
    logger.info("=== Running Blockchain Integration Demo ===")
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    # Initialize blockchain components
    hasher = NegativeSpaceHasher()
    blockchain = BlockchainConnector()
    verifier = SignatureVerifier(hasher, blockchain)
    
    # Initialize smart contract components
    contract_manager = SmartContractManager(
        provider_url=args.provider_url,
        contract_address=args.contract_address
    )
    
    signature_registry = SignatureRegistry(contract_manager)
    verification_service = VerificationService(signature_registry)
    
    # Generate multiple scenes and extract signatures
    signatures = []
    point_clouds = []
    
    for i in range(args.num_signatures):
        # Generate a slightly different scene each time
        point_cloud = generate_test_scene()
        point_clouds.append(point_cloud)
        
        # Extract signature
        signature = point_cloud.compute_spatial_signature()
        signatures.append(signature)
        
        logger.info(f"Generated signature {i+1}/{args.num_signatures}")
    
    # Create hashes for all signatures
    hashes = [hasher.hash_signature(sig) for sig in signatures]
    
    # Register signatures on the blockchain
    transaction_ids = []
    for i, signature in enumerate(signatures):
        metadata = {
            'description': f'Negative space signature {i+1}',
            'source': 'Blockchain demo',
            'timestamp': time.time()
        }
        
        transaction_id = verifier.register_signature(signature, metadata)
        transaction_ids.append(transaction_id)
        
        logger.info(f"Registered signature {i+1} with transaction ID: {transaction_id}")
    
    # Verify signatures
    verification_results = []
    for i, signature in enumerate(signatures):
        result = verifier.verify_signature(signature)
        verification_results.append(result)
        
        logger.info(f"Verification of signature {i+1}: {'Successful' if result['verified'] else 'Failed'}")
    
    # Create authentication tokens
    tokens = []
    for i, signature in enumerate(signatures):
        token = verifier.create_authentication_token(signature)
        tokens.append(token)
        
        logger.info(f"Created authentication token for signature {i+1}")
    
    # Verify tokens
    for i, (token, signature) in enumerate(zip(tokens, signatures)):
        is_valid = verifier.verify_authentication_token(token, signature)
        logger.info(f"Token {i+1} verification: {'Valid' if is_valid else 'Invalid'}")
    
    # Save blockchain data
    blockchain_data = {
        'hashes': hashes,
        'transaction_ids': transaction_ids,
        'tokens': tokens,
        'verification_results': [
            {k: v for k, v in result.items() if k != 'metadata'}
            for result in verification_results
        ]
    }
    
    data_path = os.path.join(args.output_dir, "blockchain_data.json")
    with open(data_path, 'w') as f:
        json.dump(blockchain_data, f, indent=2)
    
    logger.info(f"Blockchain data saved to {data_path}")
    
    # Create visualizations
    vis_path = os.path.join(args.output_dir, "blockchain_visualization.png")
    visualize_blockchain_results(signatures, hashes, verification_results, vis_path)
    
    # Create Merkle tree and save
    merkle_tree = hasher.create_merkle_tree(signatures)
    merkle_path = os.path.join(args.output_dir, "merkle_tree.json")
    with open(merkle_path, 'w') as f:
        json.dump({
            'root': merkle_tree['root'],
            'leaves': merkle_tree['leaves']
        }, f, indent=2)
    
    logger.info(f"Merkle tree saved to {merkle_path}")
    
    logger.info(f"Blockchain demo completed. Results saved to {args.output_dir}")
    
    return signatures, hashes, verification_results

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Run the blockchain demo
    signatures, hashes, verification_results = run_blockchain_demo(args)
    
    logger.info("Blockchain integration demo completed successfully!")

if __name__ == "__main__":
    main()

```