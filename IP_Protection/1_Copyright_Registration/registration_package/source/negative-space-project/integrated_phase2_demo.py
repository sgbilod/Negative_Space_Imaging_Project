"""
Integrated Phase 2 Demo for Negative Space Imaging

This demo showcases the combined features of Phase 2:
1. Temporal analysis of negative spaces changing over time
2. Blockchain integration for secure storage and verification

Usage:
    python integrated_phase2_demo.py [--output_dir DIR]

Example:
    python integrated_phase2_demo.py --output_dir output/phase2
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import project modules
try:
    from src.temporal_variants.negative_space_tracker import (
        NegativeSpaceTracker, ChangeMetrics, TemporalSignature, ChangeType
    )
    from src.blockchain.blockchain_integration import (
        NegativeSpaceHasher, BlockchainConnector, SignatureVerifier
    )
    from simplified_demo import SimplePointCloud, generate_test_scene
except ImportError:
    logger.error("Failed to import required modules. Make sure you're running from the project root.")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Integrated Phase 2 Demo")
    
    parser.add_argument('--output_dir', type=str, default='output/phase2',
                       help='Directory to save results (default: output/phase2)')
    
    parser.add_argument('--num_frames', type=int, default=5,
                       help='Number of frames to generate (default: 5)')
    
    return parser.parse_args()

def ensure_directory(directory):
    """Ensure a directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def generate_deforming_scene(frame: int, num_frames: int) -> SimplePointCloud:
    """Generate a scene that deforms over time"""
    from temporal_demo import generate_deforming_scene
    return generate_deforming_scene(frame, num_frames)

def run_integrated_demo(args):
    """Run the integrated Phase 2 demo"""
    logger.info("=== Running Integrated Phase 2 Demo ===")
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    # Initialize components
    tracker = NegativeSpaceTracker()
    hasher = NegativeSpaceHasher()
    blockchain = BlockchainConnector()
    verifier = SignatureVerifier(hasher, blockchain)
    
    # Generate a sequence of deforming scenes
    clouds = []
    signatures = []
    hashes = []
    transaction_ids = []
    
    for frame in range(args.num_frames):
        # Generate a point cloud for this frame
        cloud = generate_deforming_scene(frame, args.num_frames)
        clouds.append(cloud)
        
        # Save visualization
        frame_vis_path = os.path.join(args.output_dir, f"frame_{frame:03d}.png")
        cloud.visualize(frame_vis_path)
        
        # Add to tracker and get change metrics
        metrics = tracker.add_point_cloud(cloud)
        change_type = tracker.get_change_type(metrics)
        
        logger.info(f"Frame {frame}: Change type = {change_type.name}, "
                   f"Void count delta = {metrics.void_count_delta}, "
                   f"Volume delta = {metrics.volume_delta:.4f}")
        
        # Extract spatial signature
        signature = cloud.compute_spatial_signature()
        signatures.append(signature)
        
        # Create hash
        hash_value = hasher.hash_signature(signature)
        hashes.append(hash_value)
        
        # Register on blockchain with frame metadata
        metadata = {
            'description': f'Frame {frame} negative space signature',
            'frame': frame,
            'change_type': change_type.name,
            'void_count': len(cloud.void_points),
            'timestamp': time.time()
        }
        
        transaction_id = verifier.register_signature(signature, metadata)
        transaction_ids.append(transaction_id)
        
        logger.info(f"Registered frame {frame} signature with transaction ID: {transaction_id}")
    
    # Get temporal signature
    temporal_signature = tracker.get_temporal_signature()
    
    # Visualize temporal changes
    temporal_vis_path = os.path.join(args.output_dir, "temporal_changes.png")
    tracker.visualize_changes(temporal_vis_path)
    
    # Register the temporal signature on the blockchain
    temporal_hash = hasher.hash_signature(temporal_signature.signature)
    
    temporal_metadata = {
        'description': 'Temporal signature of negative space changes',
        'num_frames': args.num_frames,
        'frame_hashes': hashes,
        'timestamp': time.time()
    }
    
    temporal_tx_id = blockchain.store_hash(temporal_hash, temporal_metadata)
    logger.info(f"Registered temporal signature with transaction ID: {temporal_tx_id}")
    
    # Create a Merkle tree from all signatures
    merkle_tree = hasher.create_merkle_tree(signatures)
    
    # Save Merkle tree
    merkle_path = os.path.join(args.output_dir, "merkle_tree.json")
    with open(merkle_path, 'w') as f:
        json.dump({
            'root': merkle_tree['root'],
            'leaves': merkle_tree['leaves']
        }, f, indent=2)
    
    # Create integrated visualization
    plt.figure(figsize=(12, 10))
    
    # First subplot: Show the temporal signature
    plt.subplot(2, 1, 1)
    plt.plot(temporal_signature.signature[:32])
    plt.title('Temporal Signature')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.grid(True, alpha=0.3)
    
    # Second subplot: Show change metrics over time
    plt.subplot(2, 1, 2)
    frames = list(range(args.num_frames))
    volume_deltas = [0] + [c.volume_delta for c in tracker.change_history[1:]]
    void_count_deltas = [0] + [c.void_count_delta for c in tracker.change_history[1:]]
    
    plt.plot(frames, volume_deltas, 'b-', marker='o', label='Volume Change')
    plt.plot(frames, void_count_deltas, 'r-', marker='s', label='Void Count Change')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title('Negative Space Changes Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Change Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the integrated visualization
    integrated_vis_path = os.path.join(args.output_dir, "integrated_visualization.png")
    plt.savefig(integrated_vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to JSON
    results = {
        'frames': [
            {
                'frame': i,
                'hash': h,
                'transaction_id': tx,
                'void_count': len(cloud.void_points)
            }
            for i, (h, tx, cloud) in enumerate(zip(hashes, transaction_ids, clouds))
        ],
        'temporal_signature': {
            'hash': temporal_hash,
            'transaction_id': temporal_tx_id
        },
        'merkle_root': merkle_tree['root']
    }
    
    results_path = os.path.join(args.output_dir, "integrated_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Integrated Phase 2 demo completed. Results saved to {args.output_dir}")
    
    return tracker, blockchain, signatures

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Run the integrated demo
    tracker, blockchain, signatures = run_integrated_demo(args)
    
    logger.info("Integrated Phase 2 demo completed successfully!")

if __name__ == "__main__":
    main()
