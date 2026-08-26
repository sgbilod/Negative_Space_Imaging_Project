# Negative Space Imaging Project - Phase 2

## Phase 2 Overview

Phase 2 extends the core negative space analysis capabilities developed in Phase 1 with:

1. **Temporal Analysis**: Track and analyze how negative spaces change over time
2. **Blockchain Integration**: Secure and authenticate negative space signatures
3. **Advanced Spatial Signatures**: Enhanced algorithms for unique pattern generation

This document explains how to use the new Phase 2 features and provides examples of their applications.

## Temporal Analysis

The temporal analysis module tracks changes in negative spaces over time, enabling:

- Detection of object movement through void space changes
- Identification of emerging or dissolving voids
- Quantification of spatial deformation
- Generation of temporal signatures that capture change patterns

### Using the Temporal Analysis Module

```python
from src.temporal_variants.negative_space_tracker import NegativeSpaceTracker

# Initialize tracker
tracker = NegativeSpaceTracker()

# Add point clouds over time
for frame in range(num_frames):
    # Generate or capture a point cloud
    point_cloud = capture_point_cloud()
    
    # Add to tracker and get change metrics
    metrics = tracker.add_point_cloud(point_cloud)
    
    # Determine type of change
    change_type = tracker.get_change_type(metrics)
    print(f"Frame {frame}: Change type = {change_type.name}")

# Get the temporal signature
temporal_signature = tracker.get_temporal_signature()

# Visualize changes over time
tracker.visualize_changes("output/temporal_changes.png")
```

### Running the Temporal Demo

```bash
python temporal_demo.py --num_frames 10 --output_dir output/temporal
```

This demo generates a sequence of deforming 3D scenes and analyzes how their negative spaces change over time. The output includes:

- Visualization of each frame
- Metrics for each change (volume delta, centroid displacement, etc.)
- A temporal signature that captures the pattern of changes
- Visualization of changes over time

## Blockchain Integration

The blockchain integration module provides tools for:

- Creating cryptographic hashes from negative space signatures
- Registering signatures on a blockchain (simulated or real)
- Verifying the authenticity of signatures
- Generating authentication tokens based on negative space patterns

### Using the Blockchain Module

```python
from src.blockchain.blockchain_integration import (
    NegativeSpaceHasher, BlockchainConnector, SignatureVerifier
)

# Initialize components
hasher = NegativeSpaceHasher()
blockchain = BlockchainConnector()
verifier = SignatureVerifier(hasher, blockchain)

# Extract a signature from a point cloud
point_cloud = generate_point_cloud()
signature = point_cloud.compute_spatial_signature()

# Register on blockchain with metadata
transaction_id = verifier.register_signature(signature, {
    'description': 'Product authentication signature',
    'source': 'Manufacturing line 3'
})

# Later, verify the signature
result = verifier.verify_signature(signature)
if result['verified']:
    print(f"Signature verified! Registered on: {result['timestamp']}")
else:
    print("Signature verification failed")

# Create an authentication token
token = verifier.create_authentication_token(signature)

# Verify a token against a signature
is_valid = verifier.verify_authentication_token(token, signature)
```

### Running the Blockchain Demo

```bash
python blockchain_demo.py --num_signatures 5 --output_dir output/blockchain
```

This demo:

1. Generates multiple negative space signatures
2. Creates unique cryptographic hashes for each signature
3. Registers them on a simulated blockchain
4. Verifies the signatures
5. Creates authentication tokens
6. Visualizes the process including a Merkle tree structure

## Applications of Phase 2 Features

### Supply Chain Authentication

Negative space signatures can be used to authenticate products in a supply chain:

1. During manufacturing, capture the negative space signature of each product
2. Register the signature on a blockchain with product metadata
3. At any point in the supply chain, verify the product by:
   - Scanning it to generate a new negative space signature
   - Comparing against the blockchain-registered signature

### Temporal Change Detection

Track changes in environments or objects over time:

1. Set up a negative space tracker for a scene
2. Periodically capture new point clouds
3. Analyze the changes to detect:
   - Unauthorized object movement
   - Structural deformation
   - New objects appearing or disappearing

### Advanced Authentication Tokens

Create unforgeable authentication tokens based on physical reality:

1. Generate a negative space signature from a physical object or environment
2. Create an authentication token with an expiration time
3. Use the token for secure access control that requires:
   - Possession of the physical object
   - Being in the same physical environment

## Next Steps

Phase 3 will build upon these features to create a complete end-to-end system with:

1. Real-time negative space tracking
2. Integration with smart contracts
3. Multi-signature authentication
4. Mobile applications for field verification
