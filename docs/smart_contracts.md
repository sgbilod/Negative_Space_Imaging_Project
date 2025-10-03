# Documentation for smart_contracts.py

```python
"""
Smart Contract Implementation for Negative Space Signatures

This module provides Ethereum smart contract integration for storing and verifying
negative space signatures. It includes both the Solidity contract code and Python
interfaces for interacting with deployed contracts.

Classes:
    SmartContractManager: Manages deployment and interaction with smart contracts
    SignatureRegistry: Interface to the signature registry contract
    VerificationService: Service for verifying signatures against the blockchain
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
from datetime import datetime
import hashlib

# Import from centralized fallbacks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.fallbacks import (
    np, WEB3_AVAILABLE, web3 as w3, NUMPY_AVAILABLE
)

# Define get_web3_fallback since it's not in fallbacks.py
def get_web3_fallback():
    """Get fallback Web3 implementation for testing"""
    if WEB3_AVAILABLE:
        return w3
    else:
        logger.warning("Using fallback Web3 implementation")
        # Create minimal Web3-like functionality for simulation
        class DummyWeb3:
            def __init__(self):
                self.eth = DummyEth()
                
        class DummyEth:
            def __init__(self):
                self.accounts = ["0x0000000000000000000000000000000000000001"]
                self.default_account = self.accounts[0]
                self.contract = lambda abi, address: DummyContract()
        
        class DummyContract:
            def __init__(self):
                pass
                
            def functions(self):
                return self
        
        return DummyWeb3()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Smart contract source code
SIGNATURE_REGISTRY_CONTRACT = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title NegativeSpaceSignatureRegistry
 * @dev Smart contract for registering and verifying negative space signatures
 */
contract NegativeSpaceSignatureRegistry {
    // Owner of the contract
    address public owner;

    // Signature data structure
    struct Signature {
        bytes32 signatureHash;    // Hash of the signature data
        address registeredBy;     // Address that registered this signature
        uint256 timestamp;        // Registration timestamp
        string metadata;          // Additional metadata (JSON string)
        bool isRevoked;           // Whether the signature has been revoked
    }

    // Mapping from signature ID to Signature
    mapping(bytes32 => Signature) public signatures;
    
    // Array to keep track of all signature IDs
    bytes32[] public signatureIds;
    
    // Events
    event SignatureRegistered(bytes32 indexed id, address indexed registeredBy, uint256 timestamp);
    event SignatureRevoked(bytes32 indexed id, address indexed revokedBy, uint256 timestamp);
    event SignatureVerified(bytes32 indexed id, address indexed verifiedBy, bool isValid, uint256 timestamp);
    
    // Constructor
    constructor() {
        owner = msg.sender;
    }
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }
    
    /**
     * @dev Register a new negative space signature
     * @param signatureHash Hash of the signature data
     * @param metadata Additional metadata (JSON string)
     * @return id Unique identifier for the registered signature
     */
    function registerSignature(bytes32 signatureHash, string memory metadata) public returns (bytes32) {
        // Generate a unique ID for this signature
        bytes32 id = keccak256(abi.encodePacked(signatureHash, msg.sender, block.timestamp));
        
        // Ensure this ID doesn't already exist
        require(signatures[id].timestamp == 0, "Signature ID already exists");
        
        // Create and store the signature
        signatures[id] = Signature({
            signatureHash: signatureHash,
            registeredBy: msg.sender,
            timestamp: block.timestamp,
            metadata: metadata,
            isRevoked: false
        });
        
        // Add to the list of signature IDs
        signatureIds.push(id);
        
        // Emit event
        emit SignatureRegistered(id, msg.sender, block.timestamp);
        
        return id;
    }
    
    /**
     * @dev Revoke a signature
     * @param id ID of the signature to revoke
     */
    function revokeSignature(bytes32 id) public {
        // Ensure the signature exists
        require(signatures[id].timestamp > 0, "Signature does not exist");
        
        // Ensure the caller is either the owner or the one who registered it
        require(
            msg.sender == owner || msg.sender == signatures[id].registeredBy,
            "Only the owner or registrar can revoke a signature"
        );
        
        // Mark as revoked
        signatures[id].isRevoked = true;
        
        // Emit event
        emit SignatureRevoked(id, msg.sender, block.timestamp);
    }
    
    /**
     * @dev Verify a signature
     * @param id ID of the signature to verify
     * @param signatureHash Hash of the signature data to verify against
     * @return isValid Whether the signature is valid
     */
    function verifySignature(bytes32 id, bytes32 signatureHash) public returns (bool) {
        // Ensure the signature exists
        require(signatures[id].timestamp > 0, "Signature does not exist");
        
        // Check if the signature is valid
        bool isValid = !signatures[id].isRevoked && 
                      signatures[id].signatureHash == signatureHash;
        
        // Emit event
        emit SignatureVerified(id, msg.sender, isValid, block.timestamp);
        
        return isValid;
    }
    
    /**
     * @dev Get a signature by ID
     * @param id ID of the signature to get
     * @return signatureHash Hash of the signature data
     * @return registeredBy Address that registered this signature
     * @return timestamp Registration timestamp
     * @return metadata Additional metadata
     * @return isRevoked Whether the signature has been revoked
     */
    function getSignature(bytes32 id) public view returns (
        bytes32 signatureHash, 
        address registeredBy, 
        uint256 timestamp, 
        string memory metadata, 
        bool isRevoked
    ) {
        // Ensure the signature exists
        require(signatures[id].timestamp > 0, "Signature does not exist");
        
        // Return the signature data
        Signature storage sig = signatures[id];
        return (
            sig.signatureHash,
            sig.registeredBy,
            sig.timestamp,
            sig.metadata,
            sig.isRevoked
        );
    }
    
    /**
     * @dev Get the number of registered signatures
     * @return count Number of signatures
     */
    function getSignatureCount() public view returns (uint256) {
        return signatureIds.length;
    }
    
    /**
     * @dev Get a list of signature IDs within a range
     * @param startIndex Start index
     * @param count Number of signatures to return
     * @return ids Array of signature IDs
     */
    function getSignatureIds(uint256 startIndex, uint256 count) public view returns (bytes32[] memory) {
        // Ensure the range is valid
        require(startIndex < signatureIds.length, "Start index out of range");
        
        // Calculate the actual count
        uint256 actualCount = count;
        if (startIndex + count > signatureIds.length) {
            actualCount = signatureIds.length - startIndex;
        }
        
        // Create and populate the array
        bytes32[] memory ids = new bytes32[](actualCount);
        for (uint256 i = 0; i < actualCount; i++) {
            ids[i] = signatureIds[startIndex + i];
        }
        
        return ids;
    }
    
    /**
     * @dev Transfer ownership of the contract
     * @param newOwner Address of the new owner
     */
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner cannot be the zero address");
        owner = newOwner;
    }
}
"""


class SmartContractManager:
    """Manages deployment and interaction with smart contracts"""
    
    def __init__(self, provider_url: str = None, 
                 private_key: str = None,
                 contract_address: str = None):
        """
        Initialize a smart contract manager
        
        Args:
            provider_url: URL of the Ethereum provider (e.g., Infura)
            private_key: Private key for transaction signing
            contract_address: Address of an existing contract (if any)
        """
        self.provider_url = provider_url
        self.private_key = private_key
        self.contract_address = contract_address
        self.web3 = None
        self.contract = None
        self.abi = None
        self.bytecode = None
        self.initialized = False
        self.using_fallback = False
        
        # Initialize Web3 connection
        self._initialize_web3()
    
    def _initialize_web3(self):
        """Initialize Web3 connection"""
        if not WEB3_AVAILABLE:
            logger.warning("Web3 library not available. Using fallback implementation.")
            self.web3 = get_web3_fallback()
            self.using_fallback = True
            return
        
        try:
            # Connect to Ethereum node
            if self.provider_url:
                if self.provider_url.startswith('http'):
                    self.web3 = w3.Web3(w3.Web3.HTTPProvider(self.provider_url))
                elif self.provider_url.startswith('ws'):
                    self.web3 = w3.Web3(w3.Web3.WebsocketProvider(self.provider_url))
                else:
                    logger.error(f"Unsupported provider URL: {self.provider_url}")
                    self.web3 = get_web3_fallback()
                    self.using_fallback = True
                    return
            else:
                # Use local node or fallback
                try:
                    self.web3 = w3.Web3(w3.Web3.IPCProvider())
                except:
                    logger.warning("No provider URL specified and IPC connection failed. Using HTTP.")
                    self.web3 = w3.Web3(w3.Web3.HTTPProvider('http://localhost:8545'))
            
            # Check connection
            if not self.web3.is_connected():
                logger.warning("Could not connect to Ethereum node. Using fallback.")
                self.web3 = get_web3_fallback()
                self.using_fallback = True
            else:
                logger.info(f"Connected to Ethereum node. Chain ID: {self.web3.eth.chain_id}")
        
        except Exception as e:
            logger.error(f"Error initializing Web3: {e}")
            self.web3 = get_web3_fallback()
            self.using_fallback = True
    
    def compile_contract(self) -> bool:
        """
        Compile the contract source code
        
        Returns:
            bool: True if compilation was successful
        """
        if self.using_fallback:
            logger.info("Using fallback Web3 implementation. Contract will be simulated.")
            
            # Simulate compilation
            self.abi = [
                {
                    "name": "registerSignature",
                    "type": "function",
                    "inputs": [
                        {"name": "signatureHash", "type": "bytes32"},
                        {"name": "metadata", "type": "string"}
                    ],
                    "outputs": [
                        {"name": "", "type": "bytes32"}
                    ]
                },
                {
                    "name": "verifySignature",
                    "type": "function",
                    "inputs": [
                        {"name": "id", "type": "bytes32"},
                        {"name": "signatureHash", "type": "bytes32"}
                    ],
                    "outputs": [
                        {"name": "", "type": "bool"}
                    ]
                }
            ]
            self.bytecode = "0x"
            return True
        
        try:
            # Check if solc is available for compilation
            try:
                # Define SOLCX_AVAILABLE flag
                SOLCX_AVAILABLE = False
                try:
                    from solcx import compile_source, install_solc
                    SOLCX_AVAILABLE = True
                except ImportError:
                    logger.warning("solcx not available. Using hardcoded ABI and bytecode.")
                
                if SOLCX_AVAILABLE:
                    # Install specific solc version if needed
                    try:
                        install_solc(version='0.8.0')
                    except:
                        logger.warning("Could not install solc 0.8.0. Using default version.")
                    
                    # Compile the contract
                    compiled_sol = compile_source(
                        SIGNATURE_REGISTRY_CONTRACT,
                        output_values=['abi', 'bin']
                    )
                    
                    # Extract the contract data
                    contract_id, contract_interface = compiled_sol.popitem()
                    self.abi = contract_interface['abi']
                    self.bytecode = contract_interface['bin']
                    
                    logger.info(f"Contract compiled successfully: {contract_id}")
                    return True
                else:
                    # If compilation is not possible, use pre-compiled data
                    self._use_hardcoded_contract()
                    return True
                
            except ImportError:
                logger.warning("solcx not available. Using hardcoded ABI and bytecode.")
                # If compilation is not possible, use pre-compiled data
                self._use_hardcoded_contract()
                return True
                
        except Exception as e:
            logger.error(f"Error compiling contract: {e}")
            return False
    
    def _use_hardcoded_contract(self):
        """Use hardcoded ABI and bytecode for the contract"""
        # This is a simplified ABI, in a real implementation this would be the full ABI
        self.abi = [
            {
                "inputs": [],
                "stateMutability": "nonpayable",
                "type": "constructor"
            },
            {
                "inputs": [
                    {
                        "internalType": "bytes32",
                        "name": "signatureHash",
                        "type": "bytes32"
                    },
                    {
                        "internalType": "string",
                        "name": "metadata",
                        "type": "string"
                    }
                ],
                "name": "registerSignature",
                "outputs": [
                    {
                        "internalType": "bytes32",
                        "name": "",
                        "type": "bytes32"
                    }
                ],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "bytes32",
                        "name": "id",
                        "type": "bytes32"
                    }
                ],
                "name": "revokeSignature",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "bytes32",
                        "name": "id",
                        "type": "bytes32"
                    },
                    {
                        "internalType": "bytes32",
                        "name": "signatureHash",
                        "type": "bytes32"
                    }
                ],
                "name": "verifySignature",
                "outputs": [
                    {
                        "internalType": "bool",
                        "name": "",
                        "type": "bool"
                    }
                ],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "bytes32",
                        "name": "id",
                        "type": "bytes32"
                    }
                ],
                "name": "getSignature",
                "outputs": [
                    {
                        "internalType": "bytes32",
                        "name": "signatureHash",
                        "type": "bytes32"
                    },
                    {
                        "internalType": "address",
                        "name": "registeredBy",
                        "type": "address"
                    },
                    {
                        "internalType": "uint256",
                        "name": "timestamp",
                        "type": "uint256"
                    },
                    {
                        "internalType": "string",
                        "name": "metadata",
                        "type": "string"
                    },
                    {
                        "internalType": "bool",
                        "name": "isRevoked",
                        "type": "bool"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        # A placeholder bytecode (would be the real bytecode in a production implementation)
        self.bytecode = "0x608060405234801561001057600080fd5b50610f8a806100206000396000f3fe608060405234801561001057600080fd5b506004361061004c5760003560e01c8063209652551461005157806330d9c9161461006f578063715018a61461008b578063f2fde38b146100a5575b600080fd5b6100596100c1565b6040516100669190610c8a565b60405180910390f35b61008960048036038101906100849190610a7e565b6100c7565b005b6100a3600480360381019061009e9190610a55565b610236565b005b6100bf60048036038101906100ba9190610a55565b6102a2565b005b60005481565b61010961010483838080601f016020809104026020016040519081016040528093929190818152602001838380828437600081840152601f19601f8201169050808301925050505050505061036a565b82610389565b6101a1576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040180806020018281038252604081526020018061100f6040913960400191505060405180910390fd5b42826000018190555081816001019080519060200190610100929190610e69565b5080826002019080519060200190610100929190610e69565b506001600080828254019250508190555080827f25a311358326fb7a11906ecb7db53adf9b5a9da92114cfbaba2b297299f193e960405160405180910390a35050565b61023e6103e8565b73ffffffffffffffffffffffffffffffffffffffff166000809054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff161461029f576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040180806020018281038252602281526020018061104f6022913960400191505060405180910390fd5b565b6102aa6103e8565b73ffffffffffffffffffffffffffffffffffffffff166000809054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff161461030b576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040180806020018281038252602281526020018061104f6022913960400191505060405180910390fd5b600073ffffffffffffffffffffffffffffffffffffffff168173ffffffffffffffffffffffffffffffffffffffff16141561034b57600080fd5b806000806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff16021790555050565b6000816040516020018082805190602001908083835b602083106103a35780518252602082019150602081019050602083039250610380565b6001836020036101000a03801982511681845116808217855250505050505090500191505060405160208183030381529060405280519060200120905092915050565b6000600180600001541015610401576001905061037f565b6000600180600001541415610418576000905061037f565b60006001806000015414156102a15760005b81518110156104575761044a82828151811061044257fe5b60200101517f010000000000000000000000000000000000000000000000000000000000000090047f01000000000000000000000000000000000000000000000000000000000000006103f0565b915060010161042a565b5060008090506000541561046e576001905061037f565b6000905061037f565b600080fd5b600081359050610489816100a5565b92915050565b600081519050610100816100a5565b600082601f83011261100f578081fd5b813561110d6110088261009a565b61006f565b915080825260208301602083018583830111156112e8578384fd5b610130838284610169565b50505092915050565b60006020828403121561114d578081fd5b600061115d8482850161047a565b91505092915050565b60006020828403121561117a578081fd5b600082013567ffffffffffffffff81111561119457600080fd5b61100d848285016110ee565b600080604083850312156111b4578182fd5b60006111c28582860161047a565b92505060206111d3858286016110ee565b9150509250929050565b60006111ea8383610270565b905092915050565b60006112008383610289565b905092915050565b6000610c8a82516102ce565b6000610c8a825161030a565b6000610c8a825161034a565b6000601f19601f830116905091905056fea265627a7a723058207f6ee8c7a1abb25a62b0a29a6f10eca6b3dea1abb49e74449a539dcc5b605ff264736f6c634300050a0032";
    
    def deploy_contract(self, account_address: str = None) -> Optional[str]:
        """
        Deploy the contract to the blockchain
        
        Args:
            account_address: Address to deploy from (if not using private key)
            
        Returns:
            str: Deployed contract address, or None if deployment failed
        """
        if not self.abi or not self.bytecode:
            if not self.compile_contract():
                logger.error("Contract compilation failed")
                return None
        
        if self.using_fallback:
            logger.info("Using fallback Web3 implementation. Contract will be simulated.")
            self.contract_address = "0x" + "0" * 40  # Dummy address
            self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.abi)
            self.initialized = True
            return self.contract_address
        
        try:
            # Create contract object
            contract = self.web3.eth.contract(abi=self.abi, bytecode=self.bytecode)
            
            # Prepare transaction
            if self.private_key:
                # Use private key for transaction signing
                acct = self.web3.eth.account.from_key(self.private_key)
                tx_hash = self._deploy_with_private_key(contract, acct)
            else:
                # Use specified account or default account
                if not account_address:
                    account_address = self.web3.eth.accounts[0]
                tx_hash = self._deploy_with_account(contract, account_address)
            
            if not tx_hash:
                logger.error("Transaction hash is None, deployment failed")
                return None
            
            # Wait for transaction receipt
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            self.contract_address = tx_receipt.contractAddress
            
            # Create contract instance
            self.contract = self.web3.eth.contract(
                address=self.contract_address,
                abi=self.abi
            )
            
            self.initialized = True
            logger.info(f"Contract deployed to address: {self.contract_address}")
            return self.contract_address
            
        except Exception as e:
            logger.error(f"Error deploying contract: {e}")
            return None
    
    def _deploy_with_private_key(self, contract, account):
        """
        Deploy contract using a private key
        
        Args:
            contract: Contract object
            account: Account object from private key
            
        Returns:
            Transaction hash
        """
        try:
            # Get transaction count
            nonce = self.web3.eth.get_transaction_count(account.address)
            
            # Build transaction
            transaction = contract.constructor().build_transaction({
                'from': account.address,
                'nonce': nonce,
                'gas': 2000000,
                'gasPrice': self.web3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"Error deploying with private key: {e}")
            return None
    
    def _deploy_with_account(self, contract, account_address):
        """
        Deploy contract using an unlocked account
        
        Args:
            contract: Contract object
            account_address: Address of the account
            
        Returns:
            Transaction hash
        """
        try:
            # Build and send transaction
            tx_hash = contract.constructor().transact({
                'from': account_address,
                'gas': 2000000,
                'gasPrice': self.web3.eth.gas_price
            })
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"Error deploying with account: {e}")
            return None
    
    def load_contract(self, address: str, abi: List = None) -> bool:
        """
        Load an existing contract
        
        Args:
            address: Contract address
            abi: Contract ABI (if not already set)
            
        Returns:
            bool: True if successful
        """
        if not self.web3:
            logger.error("Web3 not initialized")
            return False
        
        try:
            if abi:
                self.abi = abi
            elif not self.abi:
                if not self.compile_contract():
                    logger.error("Contract compilation failed")
                    return False
            
            self.contract_address = address
            self.contract = self.web3.eth.contract(
                address=address,
                abi=self.abi
            )
            
            self.initialized = True
            logger.info(f"Contract loaded from address: {address}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading contract: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """
        Check if the contract manager is initialized
        
        Returns:
            bool: True if initialized
        """
        return self.initialized


class SignatureRegistry:
    """Interface to the signature registry contract"""
    
    def __init__(self, contract_manager: SmartContractManager):
        """
        Initialize a signature registry
        
        Args:
            contract_manager: Smart contract manager
        """
        self.contract_manager = contract_manager
        self.using_fallback = self.contract_manager.using_fallback
        
        # For fallback implementation
        if self.using_fallback:
            self.signatures = {}
            self.signature_ids = []
    
    def register_signature(self, signature_data: Union[List[float], bytes], 
                          metadata: Dict = None,
                          account_address: str = None) -> Optional[str]:
        """
        Register a signature on the blockchain
        
        Args:
            signature_data: Signature data (float array or bytes)
            metadata: Additional metadata
            account_address: Address to register from (if not using private key)
            
        Returns:
            str: Signature ID, or None if registration failed
        """
        if not self.contract_manager.is_initialized():
            logger.error("Contract not initialized")
            return None
        
        try:
            # Convert signature to bytes32 hash
            signature_hash = self._hash_signature(signature_data)
            
            # Convert metadata to JSON string
            metadata_str = "{}" if metadata is None else json.dumps(metadata)
            
            if self.using_fallback:
                # Fallback implementation
                signature_id = self._fallback_register(signature_hash, metadata_str)
                return signature_id
            
            # Prepare transaction
            if self.contract_manager.private_key:
                # Use private key for transaction signing
                signature_id = self._register_with_private_key(signature_hash, metadata_str)
            else:
                # Use specified account or default account
                if not account_address:
                    account_address = self.contract_manager.web3.eth.accounts[0]
                signature_id = self._register_with_account(signature_hash, metadata_str, account_address)
            
            return signature_id
            
        except Exception as e:
            logger.error(f"Error registering signature: {e}")
            return None
    
    def _fallback_register(self, signature_hash: bytes, metadata_str: str) -> str:
        """
        Register a signature using fallback implementation
        
        Args:
            signature_hash: Hash of the signature data
            metadata_str: Metadata JSON string
            
        Returns:
            str: Signature ID
        """
        # Generate a unique ID
        timestamp = int(time.time())
        address = "0x" + "0" * 40  # Dummy address
        signature_id = hashlib.sha256(
            signature_hash + address.encode() + str(timestamp).encode()
        ).hexdigest()
        
        # Store the signature
        self.signatures[signature_id] = {
            'signatureHash': signature_hash,
            'registeredBy': address,
            'timestamp': timestamp,
            'metadata': metadata_str,
            'isRevoked': False
        }
        
        # Add to list of signature IDs
        self.signature_ids.append(signature_id)
        
        logger.info(f"Signature registered with ID: {signature_id} (fallback mode)")
        return signature_id
    
    def _register_with_private_key(self, signature_hash: bytes, metadata_str: str) -> Optional[str]:
        """
        Register a signature using a private key
        
        Args:
            signature_hash: Hash of the signature data
            metadata_str: Metadata JSON string
            
        Returns:
            str: Signature ID
        """
        try:
            # Get account from private key
            acct = self.contract_manager.web3.eth.account.from_key(self.contract_manager.private_key)
            
            # Get transaction count
            nonce = self.contract_manager.web3.eth.get_transaction_count(acct.address)
            
            # Build transaction
            tx = self.contract_manager.contract.functions.registerSignature(
                signature_hash, metadata_str
            ).build_transaction({
                'from': acct.address,
                'nonce': nonce,
                'gas': 2000000,
                'gasPrice': self.contract_manager.web3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = acct.sign_transaction(tx)
            tx_hash = self.contract_manager.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            tx_receipt = self.contract_manager.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Extract signature ID from event logs
            logs = self.contract_manager.contract.events.SignatureRegistered().process_receipt(tx_receipt)
            if logs:
                signature_id = logs[0]['args']['id'].hex()
                logger.info(f"Signature registered with ID: {signature_id}")
                return signature_id
            
            logger.error("No signature ID found in transaction logs")
            return None
            
        except Exception as e:
            logger.error(f"Error registering with private key: {e}")
            return None
    
    def _register_with_account(self, signature_hash: bytes, metadata_str: str, 
                              account_address: str) -> Optional[str]:
        """
        Register a signature using an unlocked account
        
        Args:
            signature_hash: Hash of the signature data
            metadata_str: Metadata JSON string
            account_address: Address of the account
            
        Returns:
            str: Signature ID
        """
        try:
            # Call the contract function
            tx_hash = self.contract_manager.contract.functions.registerSignature(
                signature_hash, metadata_str
            ).transact({
                'from': account_address,
                'gas': 2000000,
                'gasPrice': self.contract_manager.web3.eth.gas_price
            })
            
            # Wait for receipt
            tx_receipt = self.contract_manager.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Extract signature ID from event logs
            logs = self.contract_manager.contract.events.SignatureRegistered().process_receipt(tx_receipt)
            if logs:
                signature_id = logs[0]['args']['id'].hex()
                logger.info(f"Signature registered with ID: {signature_id}")
                return signature_id
            
            logger.error("No signature ID found in transaction logs")
            return None
            
        except Exception as e:
            logger.error(f"Error registering with account: {e}")
            return None
    
    def verify_signature(self, signature_id: str, signature_data: Union[List[float], bytes],
                        account_address: str = None) -> bool:
        """
        Verify a signature against the blockchain
        
        Args:
            signature_id: ID of the signature to verify
            signature_data: Signature data to verify
            account_address: Address to verify from (if not using private key)
            
        Returns:
            bool: True if signature is valid
        """
        if not self.contract_manager.is_initialized():
            logger.error("Contract not initialized")
            return False
        
        try:
            # Convert signature to bytes32 hash
            signature_hash = self._hash_signature(signature_data)
            
            if self.using_fallback:
                # Fallback implementation
                return self._fallback_verify(signature_id, signature_hash)
            
            # Prepare transaction
            if self.contract_manager.private_key:
                # Use private key for transaction signing
                return self._verify_with_private_key(signature_id, signature_hash)
            else:
                # Use specified account or default account
                if not account_address:
                    account_address = self.contract_manager.web3.eth.accounts[0]
                return self._verify_with_account(signature_id, signature_hash, account_address)
            
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False
    
    def _fallback_verify(self, signature_id: str, signature_hash: bytes) -> bool:
        """
        Verify a signature using fallback implementation
        
        Args:
            signature_id: ID of the signature to verify
            signature_hash: Hash of the signature data
            
        Returns:
            bool: True if signature is valid
        """
        # Check if signature exists
        if signature_id not in self.signatures:
            logger.error(f"Signature with ID {signature_id} does not exist")
            return False
        
        # Get the signature
        signature = self.signatures[signature_id]
        
        # Check if revoked
        if signature['isRevoked']:
            logger.info(f"Signature with ID {signature_id} has been revoked")
            return False
        
        # Verify the hash
        is_valid = signature['signatureHash'] == signature_hash
        
        logger.info(f"Signature verification result: {is_valid} (fallback mode)")
        return is_valid
    
    def _verify_with_private_key(self, signature_id: str, signature_hash: bytes) -> bool:
        """
        Verify a signature using a private key
        
        Args:
            signature_id: ID of the signature to verify
            signature_hash: Hash of the signature data
            
        Returns:
            bool: True if signature is valid
        """
        try:
            # Get account from private key
            acct = self.contract_manager.web3.eth.account.from_key(self.contract_manager.private_key)
            
            # Get transaction count
            nonce = self.contract_manager.web3.eth.get_transaction_count(acct.address)
            
            # Build transaction
            tx = self.contract_manager.contract.functions.verifySignature(
                bytes.fromhex(signature_id.replace('0x', '')), signature_hash
            ).build_transaction({
                'from': acct.address,
                'nonce': nonce,
                'gas': 2000000,
                'gasPrice': self.contract_manager.web3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = acct.sign_transaction(tx)
            tx_hash = self.contract_manager.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            tx_receipt = self.contract_manager.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Extract result from event logs
            logs = self.contract_manager.contract.events.SignatureVerified().process_receipt(tx_receipt)
            if logs:
                is_valid = logs[0]['args']['isValid']
                logger.info(f"Signature verification result: {is_valid}")
                return is_valid
            
            # If no logs, call the function directly (view function)
            is_valid = self.contract_manager.contract.functions.verifySignature(
                bytes.fromhex(signature_id.replace('0x', '')), signature_hash
            ).call({'from': acct.address})
            
            logger.info(f"Signature verification result: {is_valid}")
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying with private key: {e}")
            return False
    
    def _verify_with_account(self, signature_id: str, signature_hash: bytes, 
                            account_address: str) -> bool:
        """
        Verify a signature using an unlocked account
        
        Args:
            signature_id: ID of the signature to verify
            signature_hash: Hash of the signature data
            account_address: Address of the account
            
        Returns:
            bool: True if signature is valid
        """
        try:
            # Call the contract function
            is_valid = self.contract_manager.contract.functions.verifySignature(
                bytes.fromhex(signature_id.replace('0x', '')), signature_hash
            ).call({'from': account_address})
            
            logger.info(f"Signature verification result: {is_valid}")
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying with account: {e}")
            return False
    
    def get_signature(self, signature_id: str) -> Optional[Dict]:
        """
        Get signature details
        
        Args:
            signature_id: ID of the signature to get
            
        Returns:
            Dict: Signature details, or None if not found
        """
        if not self.contract_manager.is_initialized():
            logger.error("Contract not initialized")
            return None
        
        try:
            if self.using_fallback:
                # Fallback implementation
                return self._fallback_get_signature(signature_id)
            
            # Call the contract function
            result = self.contract_manager.contract.functions.getSignature(
                bytes.fromhex(signature_id.replace('0x', ''))
            ).call()
            
            # Parse the result
            signature = {
                'signatureHash': result[0].hex(),
                'registeredBy': result[1],
                'timestamp': result[2],
                'metadata': result[3],
                'isRevoked': result[4]
            }
            
            return signature
            
        except Exception as e:
            logger.error(f"Error getting signature: {e}")
            return None
    
    def _fallback_get_signature(self, signature_id: str) -> Optional[Dict]:
        """
        Get signature details using fallback implementation
        
        Args:
            signature_id: ID of the signature to get
            
        Returns:
            Dict: Signature details, or None if not found
        """
        # Check if signature exists
        if signature_id not in self.signatures:
            logger.error(f"Signature with ID {signature_id} does not exist")
            return None
        
        # Return a copy of the signature
        return dict(self.signatures[signature_id])
    
    def revoke_signature(self, signature_id: str, account_address: str = None) -> bool:
        """
        Revoke a signature
        
        Args:
            signature_id: ID of the signature to revoke
            account_address: Address to revoke from (if not using private key)
            
        Returns:
            bool: True if successful
        """
        if not self.contract_manager.is_initialized():
            logger.error("Contract not initialized")
            return False
        
        try:
            if self.using_fallback:
                # Fallback implementation
                return self._fallback_revoke(signature_id)
            
            # Prepare transaction
            if self.contract_manager.private_key:
                # Use private key for transaction signing
                return self._revoke_with_private_key(signature_id)
            else:
                # Use specified account or default account
                if not account_address:
                    account_address = self.contract_manager.web3.eth.accounts[0]
                return self._revoke_with_account(signature_id, account_address)
            
        except Exception as e:
            logger.error(f"Error revoking signature: {e}")
            return False
    
    def _fallback_revoke(self, signature_id: str) -> bool:
        """
        Revoke a signature using fallback implementation
        
        Args:
            signature_id: ID of the signature to revoke
            
        Returns:
            bool: True if successful
        """
        # Check if signature exists
        if signature_id not in self.signatures:
            logger.error(f"Signature with ID {signature_id} does not exist")
            return False
        
        # Mark as revoked
        self.signatures[signature_id]['isRevoked'] = True
        
        logger.info(f"Signature with ID {signature_id} revoked (fallback mode)")
        return True
    
    def _revoke_with_private_key(self, signature_id: str) -> bool:
        """
        Revoke a signature using a private key
        
        Args:
            signature_id: ID of the signature to revoke
            
        Returns:
            bool: True if successful
        """
        try:
            # Get account from private key
            acct = self.contract_manager.web3.eth.account.from_key(self.contract_manager.private_key)
            
            # Get transaction count
            nonce = self.contract_manager.web3.eth.get_transaction_count(acct.address)
            
            # Build transaction
            tx = self.contract_manager.contract.functions.revokeSignature(
                bytes.fromhex(signature_id.replace('0x', ''))
            ).build_transaction({
                'from': acct.address,
                'nonce': nonce,
                'gas': 2000000,
                'gasPrice': self.contract_manager.web3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = acct.sign_transaction(tx)
            tx_hash = self.contract_manager.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            tx_receipt = self.contract_manager.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Check if successful
            if tx_receipt.status == 1:
                logger.info(f"Signature with ID {signature_id} revoked")
                return True
            else:
                logger.error(f"Transaction failed: {tx_hash.hex()}")
                return False
            
        except Exception as e:
            logger.error(f"Error revoking with private key: {e}")
            return False
    
    def _revoke_with_account(self, signature_id: str, account_address: str) -> bool:
        """
        Revoke a signature using an unlocked account
        
        Args:
            signature_id: ID of the signature to revoke
            account_address: Address of the account
            
        Returns:
            bool: True if successful
        """
        try:
            # Call the contract function
            tx_hash = self.contract_manager.contract.functions.revokeSignature(
                bytes.fromhex(signature_id.replace('0x', ''))
            ).transact({
                'from': account_address,
                'gas': 2000000,
                'gasPrice': self.contract_manager.web3.eth.gas_price
            })
            
            # Wait for receipt
            tx_receipt = self.contract_manager.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Check if successful
            if tx_receipt.status == 1:
                logger.info(f"Signature with ID {signature_id} revoked")
                return True
            else:
                logger.error(f"Transaction failed: {tx_hash.hex()}")
                return False
            
        except Exception as e:
            logger.error(f"Error revoking with account: {e}")
            return False
    
    def get_signature_count(self) -> int:
        """
        Get the number of registered signatures
        
        Returns:
            int: Number of signatures
        """
        if not self.contract_manager.is_initialized():
            logger.error("Contract not initialized")
            return 0
        
        try:
            if self.using_fallback:
                # Fallback implementation
                return len(self.signature_ids)
            
            # Call the contract function
            return self.contract_manager.contract.functions.getSignatureCount().call()
            
        except Exception as e:
            logger.error(f"Error getting signature count: {e}")
            return 0
    
    def get_signature_ids(self, start_index: int = 0, count: int = 10) -> List[str]:
        """
        Get a list of signature IDs
        
        Args:
            start_index: Start index
            count: Number of signatures to return
            
        Returns:
            List[str]: List of signature IDs
        """
        if not self.contract_manager.is_initialized():
            logger.error("Contract not initialized")
            return []
        
        try:
            if self.using_fallback:
                # Fallback implementation
                end_index = min(start_index + count, len(self.signature_ids))
                return self.signature_ids[start_index:end_index]
            
            # Call the contract function
            ids = self.contract_manager.contract.functions.getSignatureIds(start_index, count).call()
            
            # Convert to hex strings
            return [id_bytes.hex() for id_bytes in ids]
            
        except Exception as e:
            logger.error(f"Error getting signature IDs: {e}")
            return []
    
    def _hash_signature(self, signature_data: Union[List[float], bytes]) -> bytes:
        """
        Hash a signature
        
        Args:
            signature_data: Signature data
            
        Returns:
            bytes: Signature hash
        """
        if isinstance(signature_data, bytes):
            # Already in bytes format
            data = signature_data
        elif isinstance(signature_data, list):
            # Convert from float list to bytes
            if NUMPY_AVAILABLE:
                data = np.array(signature_data, dtype=np.float32).tobytes()
            else:
                # Manual conversion
                data = b"".join(float(x).hex().encode() for x in signature_data)
        else:
            raise ValueError("Signature data must be a list of floats or bytes")
        
        # Hash the data
        hash_bytes = hashlib.sha256(data).digest()
        
        # Convert to bytes32
        return hash_bytes


class VerificationService:
    """Service for verifying signatures against the blockchain"""
    
    def __init__(self, signature_registry: SignatureRegistry):
        """
        Initialize a verification service
        
        Args:
            signature_registry: Signature registry
        """
        self.signature_registry = signature_registry
        self.verification_cache = {}
        self.cache_timeout = 3600  # 1 hour in seconds
    
    def verify_signature(self, signature_id: str, signature_data: Union[List[float], bytes]) -> Dict:
        """
        Verify a signature and return detailed results
        
        Args:
            signature_id: ID of the signature to verify
            signature_data: Signature data to verify
            
        Returns:
            Dict: Verification results
        """
        # Check cache first
        cache_key = f"{signature_id}:{self._hash_for_cache(signature_data)}"
        if cache_key in self.verification_cache:
            cache_entry = self.verification_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_timeout:
                logger.info(f"Using cached verification result for {signature_id}")
                return cache_entry['result']
        
        # Not in cache or expired, perform verification
        is_valid = self.signature_registry.verify_signature(signature_id, signature_data)
        
        # Get signature details
        signature = self.signature_registry.get_signature(signature_id)
        
        # Prepare result
        if signature:
            result = {
                'isValid': is_valid,
                'signatureId': signature_id,
                'timestamp': datetime.now().isoformat(),
                'details': {
                    'registeredBy': signature.get('registeredBy', 'unknown'),
                    'registrationTime': self._format_timestamp(signature.get('timestamp', 0)),
                    'isRevoked': signature.get('isRevoked', False),
                    'metadata': self._parse_metadata(signature.get('metadata', '{}'))
                }
            }
        else:
            result = {
                'isValid': False,
                'signatureId': signature_id,
                'timestamp': datetime.now().isoformat(),
                'details': {
                    'error': 'Signature not found'
                }
            }
        
        # Cache the result
        self.verification_cache[cache_key] = {
            'timestamp': time.time(),
            'result': result
        }
        
        return result
    
    def verify_multiple_signatures(self, verifications: List[Dict]) -> Dict:
        """
        Verify multiple signatures and return aggregated results
        
        Args:
            verifications: List of {signatureId, signatureData} dictionaries
            
        Returns:
            Dict: Aggregated verification results
        """
        results = []
        all_valid = True
        
        for verification in verifications:
            signature_id = verification.get('signatureId')
            signature_data = verification.get('signatureData')
            
            if not signature_id or not signature_data:
                logger.error("Missing signatureId or signatureData")
                continue
            
            result = self.verify_signature(signature_id, signature_data)
            results.append(result)
            
            if not result['isValid']:
                all_valid = False
        
        return {
            'allValid': all_valid,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
    
    def clear_cache(self):
        """Clear the verification cache"""
        self.verification_cache = {}
        logger.info("Verification cache cleared")
    
    def _hash_for_cache(self, signature_data: Union[List[float], bytes]) -> str:
        """
        Create a hash of signature data for caching
        
        Args:
            signature_data: Signature data
            
        Returns:
            str: Hash for caching
        """
        if isinstance(signature_data, bytes):
            data = signature_data
        elif isinstance(signature_data, list):
            if NUMPY_AVAILABLE:
                data = np.array(signature_data, dtype=np.float32).tobytes()
            else:
                # Manual conversion
                data = b"".join(str(x).encode() for x in signature_data)
        else:
            data = str(signature_data).encode()
        
        return hashlib.md5(data).hexdigest()
    
    def _format_timestamp(self, timestamp: int) -> str:
        """
        Format a Unix timestamp as ISO string
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            str: Formatted timestamp
        """
        try:
            return datetime.fromtimestamp(timestamp).isoformat()
        except:
            return str(timestamp)
    
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

```