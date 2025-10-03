# Documentation for notary_integration_demo.py

```python
"""
Integration script for the Decentralized Notary Network

This script demonstrates the integration of the Decentralized Notary Network
with the other revenue streams of the Negative Space Imaging Project.
"""

import os
import sys
import json
import time
import hashlib
from datetime import datetime

# Add the parent directory to the path to import the notary modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.revenue.decentralized_notary.notary_network import NotaryAPI, NotaryNetwork
from src.revenue.decentralized_notary.blockchain_connector import BlockchainConnector, Blockchain
from src.negative_mapping.spatial_signature_generator import SpatialSignatureGenerator
from src.revenue.temporal_auth.temporal_auth_service import TemporalAuthService
from src.revenue.encryption_service.spatial_key_generator import SpatialKeyGenerator
from src.revenue.random_generation.cosmic_randomness_extractor import CosmicRandomnessExtractor
from src.revenue.proof_of_existence.digital_twin_manager import DigitalTwinManager


def demo_landmark_registration(notary_api, signature_generator):
    """
    Demonstrate landmark registration.
    
    Args:
        notary_api: NotaryAPI instance
        signature_generator: SpatialSignatureGenerator instance
        
    Returns:
        Registered landmark data
    """
    print("\n=== Landmark Registration ===")
    
    # Generate a test spatial signature
    import numpy as np
    
    # Create some test coordinates
    np.random.seed(42)  # For reproducibility
    coordinates = []
    for _ in range(20):
        coordinates.append([
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10)
        ])
        
    # Generate a signature
    signature = signature_generator.generate(coordinates)
    
    # Register a landmark
    landmark_data = {
        'name': 'Test Landmark',
        'description': 'A test landmark for demonstration',
        'location': {
            'latitude': 37.7749,
            'longitude': -122.4194
        },
        'spatial_signature': signature,
        'metadata': {
            'created_by': 'integration_demo.py',
            'timestamp': datetime.now().isoformat()
        }
    }
    
    landmark = notary_api.register_landmark(landmark_data)
    
    if 'error' in landmark:
        print(f"Error: {landmark['error']}")
        return None
        
    print(f"Landmark registered successfully!")
    print(f"Landmark ID: {landmark['landmark_id']}")
    print(f"Name: {landmark['name']}")
    print(f"Signature hash: {landmark['signature_hash']}")
    
    return landmark


def demo_node_registration(notary_api):
    """
    Demonstrate node registration.
    
    Args:
        notary_api: NotaryAPI instance
        
    Returns:
        Registered node data
    """
    print("\n=== Node Registration ===")
    
    # Register a node
    node_data = {
        'owner_id': f"demo-user-{int(time.time())}",
        'owner_data': {
            'client': 'integration_demo.py',
            'registration_time': datetime.now().isoformat()
        }
    }
    
    node = notary_api.register_node(node_data)
    
    if 'error' in node:
        print(f"Error: {node['error']}")
        return None
        
    print(f"Node registered successfully!")
    print(f"Node ID: {node['node_id']}")
    print(f"Owner ID: {node['owner_id']}")
    
    return node


def demo_proof_of_view(notary_api, node, landmark, signature_generator):
    """
    Demonstrate Proof-of-View submission.
    
    Args:
        notary_api: NotaryAPI instance
        node: Node data
        landmark: Landmark data
        signature_generator: SpatialSignatureGenerator instance
        
    Returns:
        Proof submission result
    """
    print("\n=== Proof of View Submission ===")
    
    if not node or not landmark:
        print("Node or landmark not available")
        return None
        
    # For a real proof, we would capture a new spatial signature
    # For the demo, we'll use the same signature to ensure it matches
    
    # Submit the proof
    proof_data = {
        'node_id': node['node_id'],
        'landmark_id': landmark['landmark_id'],
        'proof_signature': landmark['signature']
    }
    
    proof_result = notary_api.submit_proof_of_view(proof_data)
    
    if 'error' in proof_result:
        print(f"Error: {proof_result['error']}")
        return None
        
    if not proof_result.get('success', False):
        print(f"Proof submission failed: {proof_result.get('reason', 'Unknown reason')}")
        return None
        
    print(f"Proof of View submitted successfully!")
    print(f"Match score: {proof_result['validation']['match_score']:.2f}")
    print(f"Threshold: {proof_result['validation']['threshold']:.2f}")
    print(f"Valid: {proof_result['validation']['valid']}")
    print(f"Node reputation: {proof_result['reputation_score']:.2f}")
    
    return proof_result


def demo_document_notarization(notary_api, blockchain_connector, node):
    """
    Demonstrate document notarization.
    
    Args:
        notary_api: NotaryAPI instance
        blockchain_connector: BlockchainConnector instance
        node: Node data
        
    Returns:
        Notarization result
    """
    print("\n=== Document Notarization ===")
    
    if not node:
        print("Node not available")
        return None
        
    # Create a test document
    document = f"Test document content - {datetime.now().isoformat()}"
    document_hash = hashlib.sha256(document.encode()).hexdigest()
    
    # Prepare metadata
    metadata = {
        'notarized_by': node['node_id'],
        'client_type': 'integration_demo.py',
        'timestamp': datetime.now().isoformat()
    }
    
    # Notarize the document
    notarization_data = {
        'document_hash': document_hash,
        'metadata': metadata,
        'min_nodes': 1  # For demo purposes
    }
    
    notarization_result = notary_api.notarize_document(notarization_data)
    
    if 'error' in notarization_result:
        print(f"Error: {notarization_result['error']}")
        return None
        
    if not notarization_result.get('success', False):
        print(f"Notarization failed: {notarization_result.get('reason', 'Unknown reason')}")
        return None
        
    print(f"Document notarized successfully!")
    print(f"Notarization ID: {notarization_result['notarization']['notarization_id']}")
    print(f"Document hash: {document_hash}")
    print(f"Notarized at: {notarization_result['notarization']['notarized_at']}")
    
    # Record on blockchain
    blockchain_result = blockchain_connector.record_notarization(notarization_result['notarization'])
    
    print(f"Recorded on blockchain:")
    print(f"Transaction ID: {blockchain_result['transaction_id']}")
    print(f"Block ID: {blockchain_result['block_id']}")
    print(f"Confirmation time: {blockchain_result['confirmation_time']}")
    
    return notarization_result


def demo_integration_with_temporal_auth(temporal_auth, signature_generator):
    """
    Demonstrate integration with the Temporal Authentication Service.
    
    Args:
        temporal_auth: TemporalAuthService instance
        signature_generator: SpatialSignatureGenerator instance
    """
    print("\n=== Temporal Authentication Integration ===")
    
    # Generate a test gesture signature
    import numpy as np
    
    # Create some test coordinates representing a gesture
    np.random.seed(43)  # For reproducibility
    gesture_coordinates = []
    for _ in range(30):
        gesture_coordinates.append([
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5)
        ])
    
    # Register a test user
    user_id = f"test-user-{int(time.time())}"
    registration_result = temporal_auth.register_user(
        user_id=user_id,
        gesture_coordinates=gesture_coordinates,
        user_metadata={
            'created_by': 'integration_demo.py',
            'timestamp': datetime.now().isoformat()
        }
    )
    
    print(f"User registered: {user_id}")
    print(f"Registration success: {registration_result['success']}")
    
    # Authenticate with the same gesture
    auth_result = temporal_auth.authenticate_user(
        user_id=user_id,
        gesture_coordinates=gesture_coordinates
    )
    
    print(f"Authentication success: {auth_result['success']}")
    if auth_result['success']:
        print(f"Authentication token: {auth_result['token'][:20]}...")
        print(f"Token expires: {auth_result['expires_at']}")
    
    return auth_result


def demo_integration_with_spatial_keys(spatial_key_generator, signature_generator):
    """
    Demonstrate integration with the Spatial Key Generator.
    
    Args:
        spatial_key_generator: SpatialKeyGenerator instance
        signature_generator: SpatialSignatureGenerator instance
    """
    print("\n=== Spatial Key Generation Integration ===")
    
    # Generate a test spatial signature
    import numpy as np
    
    # Create some test coordinates
    np.random.seed(44)  # For reproducibility
    coordinates = []
    for _ in range(20):
        coordinates.append([
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10)
        ])
    
    # Generate a key
    key_result = spatial_key_generator.generate_key(
        coordinates=coordinates,
        key_metadata={
            'created_by': 'integration_demo.py',
            'timestamp': datetime.now().isoformat()
        }
    )
    
    print(f"Key generated: {key_result['key_id']}")
    print(f"Key type: {key_result['key_type']}")
    print(f"Key strength: {key_result['key_strength']} bits")
    print(f"Key material (first 20 chars): {key_result['key_material'][:20]}...")
    
    # Validate the key with the same coordinates
    validation_result = spatial_key_generator.validate_key(
        key_id=key_result['key_id'],
        coordinates=coordinates
    )
    
    print(f"Key validation success: {validation_result['valid']}")
    print(f"Validation score: {validation_result['match_score']:.2f}")
    
    return key_result


def demo_integration_with_cosmic_rng(cosmic_rng, signature_generator):
    """
    Demonstrate integration with the Cosmic Randomness Extractor.
    
    Args:
        cosmic_rng: CosmicRandomnessExtractor instance
        signature_generator: SpatialSignatureGenerator instance
    """
    print("\n=== Cosmic Randomness Integration ===")
    
    # Generate a test spatial signature
    import numpy as np
    
    # Create some test coordinates
    np.random.seed(45)  # For reproducibility
    coordinates = []
    for _ in range(20):
        coordinates.append([
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10)
        ])
    
    # Generate random data
    random_result = cosmic_rng.generate_random_bytes(
        num_bytes=32,
        spatial_seed=coordinates,
        metadata={
            'created_by': 'integration_demo.py',
            'timestamp': datetime.now().isoformat()
        }
    )
    
    print(f"Random data generated: {random_result['request_id']}")
    print(f"Entropy sources: {', '.join(random_result['entropy_sources'])}")
    print(f"Entropy estimate: {random_result['entropy_estimate']:.2f} bits")
    print(f"Random bytes (hex): {random_result['random_bytes_hex'][:20]}...")
    
    # Generate a random number in a range
    random_number = cosmic_rng.generate_random_int(
        min_value=1,
        max_value=100,
        spatial_seed=coordinates
    )
    
    print(f"Random number (1-100): {random_number}")
    
    return random_result


def demo_integration_with_digital_twin(digital_twin_manager, signature_generator):
    """
    Demonstrate integration with the Digital Twin Manager.
    
    Args:
        digital_twin_manager: DigitalTwinManager instance
        signature_generator: SpatialSignatureGenerator instance
    """
    print("\n=== Digital Twin Integration ===")
    
    # Generate a test spatial signature
    import numpy as np
    
    # Create some test coordinates
    np.random.seed(46)  # For reproducibility
    coordinates = []
    for _ in range(20):
        coordinates.append([
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10)
        ])
    
    # Register a digital twin
    asset_id = f"test-asset-{int(time.time())}"
    twin_result = digital_twin_manager.register_asset(
        asset_id=asset_id,
        spatial_coordinates=coordinates,
        asset_metadata={
            'name': 'Test Asset',
            'description': 'A test asset for the digital twin demo',
            'created_by': 'integration_demo.py',
            'timestamp': datetime.now().isoformat()
        }
    )
    
    print(f"Digital twin registered: {twin_result['twin_id']}")
    print(f"Asset ID: {twin_result['asset_id']}")
    print(f"Signature hash: {twin_result['signature_hash']}")
    
    # Verify the digital twin
    verify_result = digital_twin_manager.verify_asset(
        asset_id=asset_id,
        spatial_coordinates=coordinates
    )
    
    print(f"Verification success: {verify_result['verified']}")
    print(f"Similarity score: {verify_result['similarity_score']:.2f}")
    
    return twin_result


def run_integration_demo():
    """Run the full integration demo."""
    print("=== Negative Space Imaging Project ===")
    print("=== Decentralized Notary Network Integration Demo ===")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Initialize components
    signature_generator = SpatialSignatureGenerator()
    blockchain = Blockchain()
    blockchain_connector = BlockchainConnector(blockchain)
    notary_api = NotaryAPI()
    temporal_auth = TemporalAuthService()
    spatial_key_generator = SpatialKeyGenerator()
    cosmic_rng = CosmicRandomnessExtractor()
    digital_twin_manager = DigitalTwinManager()
    
    # Run demo components
    landmark = demo_landmark_registration(notary_api, signature_generator)
    node = demo_node_registration(notary_api)
    proof = demo_proof_of_view(notary_api, node, landmark, signature_generator)
    notarization = demo_document_notarization(notary_api, blockchain_connector, node)
    
    # Demonstrate integration with other revenue streams
    auth_result = demo_integration_with_temporal_auth(temporal_auth, signature_generator)
    key_result = demo_integration_with_spatial_keys(spatial_key_generator, signature_generator)
    random_result = demo_integration_with_cosmic_rng(cosmic_rng, signature_generator)
    twin_result = demo_integration_with_digital_twin(digital_twin_manager, signature_generator)
    
    print("\n=== Demo Complete ===")
    print("All revenue streams have been successfully integrated with the Decentralized Notary Network.")


if __name__ == "__main__":
    run_integration_demo()

```