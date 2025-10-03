# Documentation for authentication_demo.py

```python
"""
Authentication Demo using Blockchain-Verified Negative Space Signatures

This script demonstrates how to use blockchain-verified negative space signatures
for authentication purposes. It shows a full authentication flow from signature
capture to blockchain verification.

Usage:
    python authentication_demo.py [--mode {setup|authenticate|list}]
                                 [--user_id USER_ID]
                                 [--provider_url PROVIDER_URL]
                                 [--contract_address CONTRACT_ADDRESS]

Examples:
    # Set up a new authentication profile
    python authentication_demo.py --mode setup --user_id "alice"

    # Authenticate a user
    python authentication_demo.py --mode authenticate --user_id "alice"

    # List registered authentication profiles
    python authentication_demo.py --mode list
"""

import os
import sys
import time
import json
import hashlib
import random
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir))
sys.path.append(project_root)

# Import project modules
try:
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


class AuthenticationDemo:
    """Demo for authentication using negative space signatures"""
    
    def __init__(self, provider_url=None, contract_address=None, profiles_dir='output/profiles'):
        """
        Initialize the authentication demo
        
        Args:
            provider_url: URL of the Ethereum provider
            contract_address: Address of the deployed contract
            profiles_dir: Directory for storing authentication profiles
        """
        self.provider_url = provider_url
        self.contract_address = contract_address
        self.profiles_dir = os.path.join(project_root, profiles_dir)
        
        # Create profiles directory if it doesn't exist
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        # Initialize smart contract components
        self.contract_manager = SmartContractManager(
            provider_url=provider_url,
            contract_address=contract_address
        )
        
        self.signature_registry = SignatureRegistry(self.contract_manager)
        self.verification_service = VerificationService(self.signature_registry)
        
        # Create a log file
        self.log_file = os.path.join(self.profiles_dir, 'authentication_log.txt')
    
    def setup_profile(self, user_id: str) -> str:
        """
        Set up a new authentication profile
        
        Args:
            user_id: ID of the user to set up
            
        Returns:
            str: Profile ID
        """
        logger.info(f"Setting up authentication profile for user: {user_id}")
        
        # Generate signature based on the user's negative space
        user_signature = self._capture_signature(user_id)
        
        # Create metadata
        metadata = {
            'user_id': user_id,
            'timestamp': time.time(),
            'purpose': 'authentication',
            'device_info': {
                'type': 'simulation',
                'version': '1.0.0'
            }
        }
        
        # Register on the blockchain
        logger.info("Registering authentication signature on the blockchain...")
        signature_id = self.signature_registry.register_signature(user_signature, metadata)
        
        if signature_id:
            logger.info(f"Authentication signature registered with ID: {signature_id}")
            
            # Save profile
            profile = {
                'user_id': user_id,
                'signature_id': signature_id,
                'signature_data': user_signature,
                'metadata': metadata,
                'created_at': time.time()
            }
            
            profile_path = os.path.join(self.profiles_dir, f"{user_id}_profile.json")
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            logger.info(f"Authentication profile saved to: {profile_path}")
            
            # Log the setup
            self._log_event('setup', user_id, True, signature_id)
            
            return signature_id
        
        else:
            logger.error("Failed to register authentication signature")
            self._log_event('setup', user_id, False, None)
            return None
    
    def authenticate_user(self, user_id: str) -> bool:
        """
        Authenticate a user
        
        Args:
            user_id: ID of the user to authenticate
            
        Returns:
            bool: True if authentication successful
        """
        logger.info(f"Authenticating user: {user_id}")
        
        # Load the user's profile
        profile_path = os.path.join(self.profiles_dir, f"{user_id}_profile.json")
        
        if not os.path.exists(profile_path):
            logger.error(f"No profile found for user: {user_id}")
            self._log_event('authenticate', user_id, False, None, "Profile not found")
            return False
        
        # Load profile
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        # Get the signature ID and stored signature
        signature_id = profile.get('signature_id')
        stored_signature = profile.get('signature_data')
        
        if not signature_id or not stored_signature:
            logger.error(f"Invalid profile for user: {user_id}")
            self._log_event('authenticate', user_id, False, signature_id, "Invalid profile")
            return False
        
        # Capture current signature
        current_signature = self._capture_signature(user_id)
        
        # Verify on the blockchain
        logger.info(f"Verifying signature with ID: {signature_id}")
        result = self.verification_service.verify_signature(signature_id, current_signature)
        
        # Display result
        if result['isValid']:
            logger.info("✅ Authentication successful!")
            self._log_event('authenticate', user_id, True, signature_id)
            return True
        else:
            logger.info("❌ Authentication failed!")
            self._log_event('authenticate', user_id, False, signature_id, 
                          result['details'].get('reason', 'Unknown reason'))
            return False
    
    def list_profiles(self) -> List[Dict]:
        """
        List all authentication profiles
        
        Returns:
            List[Dict]: List of profiles
        """
        logger.info("Listing authentication profiles...")
        
        profiles = []
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('_profile.json'):
                profile_path = os.path.join(self.profiles_dir, filename)
                
                try:
                    with open(profile_path, 'r') as f:
                        profile = json.load(f)
                        
                    # Extract key information
                    profiles.append({
                        'user_id': profile.get('user_id'),
                        'signature_id': profile.get('signature_id'),
                        'created_at': profile.get('created_at')
                    })
                except Exception as e:
                    logger.error(f"Error loading profile {filename}: {e}")
        
        # Display profiles
        for i, profile in enumerate(profiles):
            logger.info(f"Profile {i+1}:")
            logger.info(f"  User ID: {profile['user_id']}")
            logger.info(f"  Signature ID: {profile['signature_id']}")
            logger.info(f"  Created at: {time.ctime(profile['created_at'])}")
            logger.info("-" * 50)
        
        return profiles
    
    def _capture_signature(self, user_id: str) -> List[float]:
        """
        Capture a negative space signature
        
        In a real implementation, this would capture from a sensor.
        For the demo, we generate a consistent signature based on the user ID.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List[float]: Signature data
        """
        logger.info(f"Capturing negative space signature for user: {user_id}")
        
        # Use the user ID as a seed for random generation to ensure consistency
        seed = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 10000
        
        if NUMPY_AVAILABLE:
            # Set the seed for numpy random
            np.random.seed(seed)
            # Generate a signature
            signature = np.random.rand(128).tolist()
            # Add some noise for repeated captures (in real world, captures would vary slightly)
            noise_level = 0.05
            noise = np.random.rand(128) * noise_level
            signature = (np.array(signature) + noise).tolist()
        else:
            # Set the seed for random
            random.seed(seed)
            # Generate a signature
            signature = [random.random() for _ in range(128)]
            # Add some noise for repeated captures
            random.seed(time.time())  # Reset seed for noise
            noise_level = 0.05
            signature = [s + random.random() * noise_level for s in signature]
        
        logger.info(f"Signature captured with {len(signature)} features")
        
        return signature
    
    def _log_event(self, event_type: str, user_id: str, success: bool, 
                 signature_id: str = None, reason: str = None):
        """
        Log an authentication event
        
        Args:
            event_type: Type of event ('setup' or 'authenticate')
            user_id: ID of the user
            success: Whether the event was successful
            signature_id: ID of the signature (optional)
            reason: Reason for failure (optional)
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        status = "SUCCESS" if success else "FAILURE"
        
        log_entry = f"{timestamp} | {event_type.upper()} | {user_id} | {status}"
        
        if signature_id:
            log_entry += f" | Signature ID: {signature_id}"
        
        if reason and not success:
            log_entry += f" | Reason: {reason}"
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Authentication Demo')
    
    parser.add_argument('--mode', choices=['setup', 'authenticate', 'list'],
                       default='setup', help='Demo mode')
    
    parser.add_argument('--user_id', type=str, default=None,
                       help='ID of the user')
    
    parser.add_argument('--provider_url', type=str, default=None,
                       help='Ethereum provider URL')
    
    parser.add_argument('--contract_address', type=str, default=None,
                       help='Address of the deployed contract')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create demo
    demo = AuthenticationDemo(
        provider_url=args.provider_url,
        contract_address=args.contract_address
    )
    
    # Run demo mode
    if args.mode == 'setup':
        # Set up a new authentication profile
        if not args.user_id:
            user_id = f"user_{int(time.time())}"
            logger.info(f"No user ID provided, using generated ID: {user_id}")
        else:
            user_id = args.user_id
        
        demo.setup_profile(user_id)
        
    elif args.mode == 'authenticate':
        # Authenticate a user
        if not args.user_id:
            logger.error("User ID is required for authentication")
            return
        
        success = demo.authenticate_user(args.user_id)
        if success:
            logger.info("Authentication successful! User is verified.")
        else:
            logger.info("Authentication failed! User could not be verified.")
        
    elif args.mode == 'list':
        # List profiles
        demo.list_profiles()


if __name__ == '__main__':
    main()

```