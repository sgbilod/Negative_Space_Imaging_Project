# Documentation for cli.py

```python
"""
Command Line Interface for the Decentralized Notary Network

This module provides a CLI for interacting with the Decentralized Time Notary Network.
"""

import argparse
import json
import os
import sys
import hashlib
import uuid
from datetime import datetime
import requests

# Add the parent directory to the path to import the notary modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from revenue.decentralized_notary.notary_network import NotaryAPI
from negative_mapping.spatial_signature_generator import SpatialSignatureGenerator


class NotaryNetworkCLI:
    """
    Command Line Interface for the Decentralized Notary Network.
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.api = NotaryAPI()
        self.signature_generator = SpatialSignatureGenerator()
        self.config_dir = os.path.expanduser("~/.negative_space_notary")
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.node_id = None
        self.owner_id = None
        
        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Load configuration if it exists
        self.load_config()
        
    def load_config(self):
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.node_id = config.get('node_id')
                self.owner_id = config.get('owner_id')
                
    def save_config(self):
        """Save configuration to file."""
        config = {
            'node_id': self.node_id,
            'owner_id': self.owner_id
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
    def register_landmark(self, args):
        """
        Register a new landmark.
        
        Args:
            args: Command line arguments
        """
        # Parse spatial signature data
        if args.signature_file:
            with open(args.signature_file, 'r') as f:
                signature_data = json.load(f)
                
            if isinstance(signature_data, list):
                # Raw coordinates
                spatial_signature = signature_data
            else:
                # Precomputed signature
                spatial_signature = signature_data
        else:
            # Generate a random signature for testing
            import random
            coordinates = []
            for _ in range(10):
                coordinates.append([
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(-10, 10)
                ])
                
            spatial_signature = coordinates
            
        # Parse location
        location = {
            'latitude': args.latitude,
            'longitude': args.longitude
        }
        
        # Register the landmark
        result = self.api.register_landmark({
            'name': args.name,
            'description': args.description,
            'location': location,
            'spatial_signature': spatial_signature,
            'metadata': args.metadata
        })
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
            
        print(f"Landmark registered successfully!")
        print(f"Landmark ID: {result['landmark_id']}")
        print(f"Name: {result['name']}")
        
    def list_landmarks(self, args):
        """
        List all landmarks.
        
        Args:
            args: Command line arguments
        """
        landmarks = self.api.network.landmark_registry.landmarks
        
        if not landmarks:
            print("No landmarks registered.")
            return
            
        print(f"Found {len(landmarks)} landmarks:")
        
        for landmark_id, landmark in landmarks.items():
            print(f"\nID: {landmark_id}")
            print(f"Name: {landmark['name']}")
            print(f"Description: {landmark['description']}")
            print(f"Location: {landmark['location']['latitude']}, {landmark['location']['longitude']}")
            print(f"Created: {landmark['created_at']}")
            print(f"Verified by: {landmark['verified_by']} nodes")
            
    def register_node(self, args):
        """
        Register a new notary node.
        
        Args:
            args: Command line arguments
        """
        if self.node_id:
            print(f"You already have a registered node: {self.node_id}")
            print("Use --force to register a new one.")
            
            if not args.force:
                return
                
        # Generate a unique owner ID if not provided
        if not args.owner_id:
            owner_id = f"cli-user-{uuid.uuid4()}"
        else:
            owner_id = args.owner_id
            
        # Register the node
        result = self.api.register_node({
            'owner_id': owner_id,
            'owner_data': {
                'client': 'cli',
                'registration_time': datetime.now().isoformat()
            }
        })
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
            
        # Save the node ID
        self.node_id = result['node_id']
        self.owner_id = owner_id
        self.save_config()
        
        print(f"Node registered successfully!")
        print(f"Node ID: {self.node_id}")
        print(f"Owner ID: {self.owner_id}")
        
    def submit_proof_of_view(self, args):
        """
        Submit a Proof-of-View for a landmark.
        
        Args:
            args: Command line arguments
        """
        if not self.node_id:
            print("You need to register a node first.")
            print("Use 'register-node' command.")
            return
            
        # Parse spatial signature data
        if args.signature_file:
            with open(args.signature_file, 'r') as f:
                signature_data = json.load(f)
                
            if isinstance(signature_data, list):
                # Raw coordinates
                spatial_signature = signature_data
            else:
                # Precomputed signature
                spatial_signature = signature_data
        else:
            # Generate a random signature for testing
            import random
            coordinates = []
            for _ in range(10):
                coordinates.append([
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(-10, 10)
                ])
                
            spatial_signature = coordinates
            
        # Submit the proof
        result = self.api.submit_proof_of_view({
            'node_id': self.node_id,
            'landmark_id': args.landmark_id,
            'proof_signature': spatial_signature
        })
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
            
        if not result.get('success', False):
            print(f"Proof submission failed: {result.get('reason', 'Unknown reason')}")
            return
            
        print(f"Proof of View submitted successfully!")
        print(f"Match score: {result['validation']['match_score']:.2f}")
        print(f"Threshold: {result['validation']['threshold']:.2f}")
        print(f"Valid: {result['validation']['valid']}")
        print(f"Node reputation: {result['reputation_score']:.2f}")
        
    def notarize_document(self, args):
        """
        Notarize a document.
        
        Args:
            args: Command line arguments
        """
        if not self.node_id:
            print("You need to register a node first.")
            print("Use 'register-node' command.")
            return
            
        # Calculate document hash
        if args.file:
            with open(args.file, 'rb') as f:
                document_hash = hashlib.sha256(f.read()).hexdigest()
        else:
            document_hash = args.hash
            
        # Prepare metadata
        metadata = {
            'notarized_by': self.node_id,
            'client_type': 'cli',
            'timestamp': datetime.now().isoformat()
        }
        
        if args.metadata:
            metadata.update(args.metadata)
            
        # Notarize the document
        result = self.api.notarize_document({
            'document_hash': document_hash,
            'metadata': metadata,
            'min_nodes': args.min_nodes
        })
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
            
        if not result.get('success', False):
            print(f"Notarization failed: {result.get('reason', 'Unknown reason')}")
            return
            
        print(f"Document notarized successfully!")
        print(f"Notarization ID: {result['notarization']['notarization_id']}")
        print(f"Document hash: {document_hash}")
        print(f"Notarized at: {result['notarization']['notarized_at']}")
        print(f"Consensus nodes: {result['notarization']['consensus_nodes']}")
        
        # Save notarization details to file
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result['notarization'], f, indent=2)
                
            print(f"Notarization details saved to: {args.output}")
            
    def verify_notarization(self, args):
        """
        Verify a document notarization.
        
        Args:
            args: Command line arguments
        """
        # Verify the notarization
        result = self.api.verify_notarization({
            'notarization_id': args.id
        })
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
            
        if not result.get('verified', False):
            print(f"Verification failed: {result.get('reason', 'Unknown reason')}")
            return
            
        print(f"Notarization verified successfully!")
        print(f"Notarization ID: {result['notarization']['notarization_id']}")
        print(f"Document hash: {result['notarization']['document_hash']}")
        print(f"Notarized at: {result['notarization']['notarized_at']}")
        print(f"Consensus nodes: {result['notarization']['consensus_nodes']}")
        
    def run(self):
        """Run the CLI application."""
        parser = argparse.ArgumentParser(
            description="Decentralized Notary Network CLI",
            formatter_class=argparse.RawTextHelpFormatter
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Command to run')
        
        # Register landmark command
        landmark_parser = subparsers.add_parser(
            'register-landmark',
            help='Register a new landmark for Proof-of-View validation'
        )
        landmark_parser.add_argument('--name', '-n', required=True, help='Name of the landmark')
        landmark_parser.add_argument('--description', '-d', required=True, help='Description of the landmark')
        landmark_parser.add_argument('--latitude', '-lat', required=True, type=float, help='Latitude of the landmark')
        landmark_parser.add_argument('--longitude', '-lon', required=True, type=float, help='Longitude of the landmark')
        landmark_parser.add_argument('--signature-file', '-s', help='File containing spatial signature data')
        landmark_parser.add_argument('--metadata', '-m', type=json.loads, help='Additional metadata as JSON')
        
        # List landmarks command
        list_parser = subparsers.add_parser(
            'list-landmarks',
            help='List all registered landmarks'
        )
        
        # Register node command
        node_parser = subparsers.add_parser(
            'register-node',
            help='Register a new notary node'
        )
        node_parser.add_argument('--owner-id', '-o', help='Owner ID for the node')
        node_parser.add_argument('--force', '-f', action='store_true', help='Force registration even if already registered')
        
        # Submit proof of view command
        proof_parser = subparsers.add_parser(
            'submit-proof',
            help='Submit a Proof-of-View for a landmark'
        )
        proof_parser.add_argument('--landmark-id', '-l', required=True, help='ID of the landmark')
        proof_parser.add_argument('--signature-file', '-s', help='File containing spatial signature data')
        
        # Notarize document command
        notarize_parser = subparsers.add_parser(
            'notarize',
            help='Notarize a document'
        )
        notarize_parser.add_argument('--file', '-f', help='File to notarize')
        notarize_parser.add_argument('--hash', '-h', help='Document hash if file is not provided')
        notarize_parser.add_argument('--metadata', '-m', type=json.loads, help='Additional metadata as JSON')
        notarize_parser.add_argument('--min-nodes', '-n', type=int, default=3, help='Minimum number of nodes for consensus')
        notarize_parser.add_argument('--output', '-o', help='Output file for notarization details')
        
        # Verify notarization command
        verify_parser = subparsers.add_parser(
            'verify',
            help='Verify a document notarization'
        )
        verify_parser.add_argument('--id', '-i', required=True, help='Notarization ID to verify')
        
        # Parse arguments
        args = parser.parse_args()
        
        if args.command == 'register-landmark':
            self.register_landmark(args)
        elif args.command == 'list-landmarks':
            self.list_landmarks(args)
        elif args.command == 'register-node':
            self.register_node(args)
        elif args.command == 'submit-proof':
            self.submit_proof_of_view(args)
        elif args.command == 'notarize':
            self.notarize_document(args)
        elif args.command == 'verify':
            self.verify_notarization(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    cli = NotaryNetworkCLI()
    cli.run()

```