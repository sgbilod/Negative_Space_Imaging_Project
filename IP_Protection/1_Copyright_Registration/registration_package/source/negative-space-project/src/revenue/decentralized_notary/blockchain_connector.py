"""
Blockchain Connector for Decentralized Notary Network

This module provides blockchain integration for the Decentralized Time Notary Network,
allowing for secure, immutable recording of notarized documents and proof-of-view validations.
"""

import hashlib
import json
import time
import uuid
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# In a production environment, we would use an actual blockchain library
# For Phase 1, we'll simulate blockchain functionality


class BlockchainTransaction:
    """
    Represents a transaction to be added to the blockchain.
    """
    
    def __init__(self, transaction_type: str, data: Dict[str, Any]):
        """
        Initialize a blockchain transaction.
        
        Args:
            transaction_type: Type of transaction (notarization, proof, etc.)
            data: Transaction data
        """
        self.transaction_id = str(uuid.uuid4())
        self.transaction_type = transaction_type
        self.data = data
        self.timestamp = datetime.now().isoformat()
        self.confirmed = False
        self.block_id = None
        self.confirmation_time = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the transaction to a dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "transaction_type": self.transaction_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "confirmed": self.confirmed,
            "block_id": self.block_id,
            "confirmation_time": self.confirmation_time
        }
        
    def __str__(self) -> str:
        """String representation of the transaction."""
        return json.dumps(self.to_dict(), indent=2)


class Block:
    """
    Represents a block in the blockchain.
    """
    
    def __init__(self, previous_hash: str):
        """
        Initialize a block.
        
        Args:
            previous_hash: Hash of the previous block
        """
        self.block_id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.previous_hash = previous_hash
        self.transactions = []
        self.nonce = 0
        self.hash = None
        
    def add_transaction(self, transaction: BlockchainTransaction) -> None:
        """
        Add a transaction to the block.
        
        Args:
            transaction: Transaction to add
        """
        self.transactions.append(transaction)
        
    def calculate_hash(self) -> str:
        """
        Calculate the hash of the block.
        
        Returns:
            Block hash
        """
        block_data = {
            "block_id": self.block_id,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "transactions": [t.to_dict() for t in self.transactions],
            "nonce": self.nonce
        }
        
        block_string = json.dumps(block_data, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
        
    def mine_block(self, difficulty: int = 2) -> str:
        """
        Mine the block (find a hash with the required difficulty).
        
        Args:
            difficulty: Mining difficulty (number of leading zeros)
            
        Returns:
            Block hash
        """
        target = '0' * difficulty
        
        while True:
            self.hash = self.calculate_hash()
            
            if self.hash.startswith(target):
                break
                
            self.nonce += 1
            
        # Mark all transactions as confirmed
        confirmation_time = datetime.now().isoformat()
        
        for transaction in self.transactions:
            transaction.confirmed = True
            transaction.block_id = self.block_id
            transaction.confirmation_time = confirmation_time
            
        return self.hash
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the block to a dictionary."""
        return {
            "block_id": self.block_id,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "transactions": [t.to_dict() for t in self.transactions],
            "nonce": self.nonce,
            "hash": self.hash
        }
        
    def __str__(self) -> str:
        """String representation of the block."""
        return json.dumps(self.to_dict(), indent=2)


class Blockchain:
    """
    A simple blockchain implementation.
    """
    
    def __init__(self, difficulty: int = 2):
        """
        Initialize the blockchain.
        
        Args:
            difficulty: Mining difficulty (number of leading zeros)
        """
        self.chain = []
        self.pending_transactions = []
        self.difficulty = difficulty
        self.mining_reward = 1.0
        
        # Create the genesis block
        self._create_genesis_block()
        
    def _create_genesis_block(self) -> None:
        """Create the genesis block."""
        genesis_block = Block("0")
        genesis_block.timestamp = "2025-01-01T00:00:00.000000"
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
        
    def get_latest_block(self) -> Block:
        """
        Get the latest block in the chain.
        
        Returns:
            Latest block
        """
        return self.chain[-1]
        
    def add_transaction(self, transaction: BlockchainTransaction) -> str:
        """
        Add a transaction to the pending transactions.
        
        Args:
            transaction: Transaction to add
            
        Returns:
            Transaction ID
        """
        self.pending_transactions.append(transaction)
        return transaction.transaction_id
        
    def mine_pending_transactions(self, miner_address: str) -> Optional[Block]:
        """
        Mine pending transactions and add a new block to the chain.
        
        Args:
            miner_address: Address to send the mining reward to
            
        Returns:
            The newly mined block, or None if there are no pending transactions
        """
        if not self.pending_transactions:
            return None
            
        # Create a new block
        new_block = Block(self.get_latest_block().hash)
        
        # Add pending transactions to the block
        # In a real blockchain, we would limit the number of transactions per block
        for transaction in self.pending_transactions:
            new_block.add_transaction(transaction)
            
        # Mine the block
        new_block.mine_block(self.difficulty)
        
        # Add the block to the chain
        self.chain.append(new_block)
        
        # Clear the pending transactions
        self.pending_transactions = []
        
        # Add a mining reward transaction for the next block
        reward_transaction = BlockchainTransaction(
            transaction_type="mining_reward",
            data={
                "miner": miner_address,
                "amount": self.mining_reward
            }
        )
        
        self.pending_transactions.append(reward_transaction)
        
        return new_block
        
    def is_chain_valid(self) -> bool:
        """
        Validate the blockchain.
        
        Returns:
            True if the chain is valid, False otherwise
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Validate hash
            if current_block.hash != current_block.calculate_hash():
                return False
                
            # Validate previous hash
            if current_block.previous_hash != previous_block.hash:
                return False
                
        return True
        
    def get_transaction(self, transaction_id: str) -> Optional[BlockchainTransaction]:
        """
        Get a transaction by ID.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Transaction or None if not found
        """
        # Check pending transactions
        for transaction in self.pending_transactions:
            if transaction.transaction_id == transaction_id:
                return transaction
                
        # Check confirmed transactions
        for block in self.chain:
            for transaction in block.transactions:
                if transaction.transaction_id == transaction_id:
                    return transaction
                    
        return None
        
    def get_transaction_history(self, address: str) -> List[BlockchainTransaction]:
        """
        Get transaction history for an address.
        
        Args:
            address: Address to get history for
            
        Returns:
            List of transactions
        """
        transactions = []
        
        # Check confirmed transactions
        for block in self.chain:
            for transaction in block.transactions:
                # Check if the address is involved in the transaction
                if (transaction.transaction_type == "notarization" and 
                    transaction.data.get("owner_id") == address):
                    transactions.append(transaction)
                    
                if (transaction.transaction_type == "proof_of_view" and 
                    transaction.data.get("node_id") == address):
                    transactions.append(transaction)
                    
                if (transaction.transaction_type == "mining_reward" and 
                    transaction.data.get("miner") == address):
                    transactions.append(transaction)
                    
        return transactions
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the blockchain to a dictionary."""
        return {
            "chain": [block.to_dict() for block in self.chain],
            "pending_transactions": [tx.to_dict() for tx in self.pending_transactions],
            "difficulty": self.difficulty,
            "mining_reward": self.mining_reward
        }


class BlockchainConnector:
    """
    Connector for interacting with the blockchain.
    """
    
    def __init__(self, blockchain: Optional[Blockchain] = None):
        """
        Initialize the blockchain connector.
        
        Args:
            blockchain: Existing blockchain instance or None to create a new one
        """
        self.blockchain = blockchain or Blockchain()
        self.node_id = str(uuid.uuid4())
        
    def record_notarization(self, 
                          notarization_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a document notarization on the blockchain.
        
        Args:
            notarization_data: Notarization data
            
        Returns:
            Transaction details
        """
        transaction = BlockchainTransaction(
            transaction_type="notarization",
            data=notarization_data
        )
        
        transaction_id = self.blockchain.add_transaction(transaction)
        
        # Mine the block to confirm the transaction
        # In a real implementation, mining would be done by a separate process
        self.blockchain.mine_pending_transactions(self.node_id)
        
        # Get the confirmed transaction
        confirmed_tx = self.blockchain.get_transaction(transaction_id)
        
        return {
            "success": True,
            "transaction_id": transaction_id,
            "confirmed": confirmed_tx.confirmed,
            "block_id": confirmed_tx.block_id,
            "timestamp": confirmed_tx.timestamp,
            "confirmation_time": confirmed_tx.confirmation_time
        }
        
    def record_proof_of_view(self, 
                           proof_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a Proof-of-View submission on the blockchain.
        
        Args:
            proof_data: Proof-of-View data
            
        Returns:
            Transaction details
        """
        transaction = BlockchainTransaction(
            transaction_type="proof_of_view",
            data=proof_data
        )
        
        transaction_id = self.blockchain.add_transaction(transaction)
        
        # Mine the block to confirm the transaction
        self.blockchain.mine_pending_transactions(self.node_id)
        
        # Get the confirmed transaction
        confirmed_tx = self.blockchain.get_transaction(transaction_id)
        
        return {
            "success": True,
            "transaction_id": transaction_id,
            "confirmed": confirmed_tx.confirmed,
            "block_id": confirmed_tx.block_id,
            "timestamp": confirmed_tx.timestamp,
            "confirmation_time": confirmed_tx.confirmation_time
        }
        
    def verify_notarization(self, 
                          transaction_id: str) -> Dict[str, Any]:
        """
        Verify a document notarization on the blockchain.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Verification result
        """
        transaction = self.blockchain.get_transaction(transaction_id)
        
        if not transaction:
            return {
                "verified": False,
                "reason": "transaction_not_found",
                "details": f"Transaction not found: {transaction_id}"
            }
            
        if not transaction.confirmed:
            return {
                "verified": False,
                "reason": "transaction_not_confirmed",
                "details": "Transaction is not confirmed yet"
            }
            
        if transaction.transaction_type != "notarization":
            return {
                "verified": False,
                "reason": "invalid_transaction_type",
                "details": f"Transaction is not a notarization: {transaction.transaction_type}"
            }
            
        # In a real implementation, we would verify the chain is valid
        chain_valid = self.blockchain.is_chain_valid()
        
        if not chain_valid:
            return {
                "verified": False,
                "reason": "invalid_blockchain",
                "details": "The blockchain has been tampered with"
            }
            
        return {
            "verified": True,
            "transaction": transaction.to_dict(),
            "notarization": transaction.data
        }
        
    def get_blockchain_status(self) -> Dict[str, Any]:
        """
        Get the current status of the blockchain.
        
        Returns:
            Blockchain status
        """
        latest_block = self.blockchain.get_latest_block()
        
        return {
            "chain_length": len(self.blockchain.chain),
            "pending_transactions": len(self.blockchain.pending_transactions),
            "difficulty": self.blockchain.difficulty,
            "latest_block": {
                "block_id": latest_block.block_id,
                "timestamp": latest_block.timestamp,
                "hash": latest_block.hash,
                "transactions": len(latest_block.transactions)
            },
            "is_valid": self.blockchain.is_chain_valid()
        }
