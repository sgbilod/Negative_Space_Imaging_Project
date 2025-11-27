"""
Persistent Memory System with Decay

Implements a sophisticated memory management system for the Negative Space
Imaging Project. Provides persistent storage with configurable decay functions,
spatial signature caching using Locality-Sensitive Hashing (LSH), and
efficient retrieval mechanisms.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# =============================================================================
# Enums
# =============================================================================


class MemoryType(Enum):
    """Types of memories stored in the system."""
    SPATIAL_SIGNATURE = auto()
    RECONSTRUCTION_PARAMS = auto()
    ANALYSIS_RESULT = auto()
    AGENT_EXPERIENCE = auto()
    CALIBRATION_DATA = auto()


class DecayStrategy(Enum):
    """Available decay strategies for memory entries."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    STEP = "step"
    NONE = "none"


# =============================================================================
# Decay Functions
# =============================================================================


class DecayFunction:
    """
    Static class providing decay function implementations.

    Decay functions determine how memory relevance decreases over time.
    Each function takes the initial relevance, time elapsed, and decay rate.
    """

    @staticmethod
    def exponential(
        initial_relevance: float,
        time_elapsed_hours: float,
        decay_rate: float = 0.01
    ) -> float:
        """
        Exponential decay: R(t) = R0 * e^(-λt)

        Fast initial decay, slowing over time. Good for rapidly
        aging information.

        Args:
            initial_relevance: Starting relevance (0-1)
            time_elapsed_hours: Hours since memory creation
            decay_rate: Lambda (λ) - higher = faster decay

        Returns:
            Current relevance (0-1)
        """
        return initial_relevance * math.exp(-decay_rate * time_elapsed_hours)

    @staticmethod
    def linear(
        initial_relevance: float,
        time_elapsed_hours: float,
        decay_rate: float = 0.001
    ) -> float:
        """
        Linear decay: R(t) = R0 - λt

        Constant rate of decay. Simple and predictable.

        Args:
            initial_relevance: Starting relevance (0-1)
            time_elapsed_hours: Hours since memory creation
            decay_rate: Rate of decay per hour

        Returns:
            Current relevance (0-1), minimum 0
        """
        return max(0.0, initial_relevance - decay_rate * time_elapsed_hours)

    @staticmethod
    def logarithmic(
        initial_relevance: float,
        time_elapsed_hours: float,
        decay_rate: float = 0.1
    ) -> float:
        """
        Logarithmic decay: R(t) = R0 / (1 + λ * ln(1 + t))

        Slow initial decay, accelerating over time. Good for
        information that remains relevant longer.

        Args:
            initial_relevance: Starting relevance (0-1)
            time_elapsed_hours: Hours since memory creation
            decay_rate: Scaling factor

        Returns:
            Current relevance (0-1)
        """
        if time_elapsed_hours <= 0:
            return initial_relevance
        denominator = 1 + decay_rate * math.log(1 + time_elapsed_hours)
        return initial_relevance / denominator

    @staticmethod
    def step(
        initial_relevance: float,
        time_elapsed_hours: float,
        decay_rate: float = 24.0  # Hours per step
    ) -> float:
        """
        Step decay: R(t) = R0 * (0.5 ^ floor(t / step_size))

        Halves relevance at fixed intervals. Good for
        staged deprecation.

        Args:
            initial_relevance: Starting relevance (0-1)
            time_elapsed_hours: Hours since memory creation
            decay_rate: Hours between steps (default: 24)

        Returns:
            Current relevance (0-1)
        """
        steps = int(time_elapsed_hours / decay_rate)
        return initial_relevance * (0.5 ** steps)

    @staticmethod
    def none(
        initial_relevance: float,
        time_elapsed_hours: float,
        decay_rate: float = 0.0
    ) -> float:
        """
        No decay: R(t) = R0

        Memory remains at initial relevance forever.

        Args:
            initial_relevance: Starting relevance (0-1)
            time_elapsed_hours: Ignored
            decay_rate: Ignored

        Returns:
            Initial relevance unchanged
        """
        return initial_relevance

    @classmethod
    def get_function(
        cls,
        strategy: DecayStrategy
    ) -> Callable[[float, float, float], float]:
        """Get the decay function for a strategy."""
        mapping = {
            DecayStrategy.EXPONENTIAL: cls.exponential,
            DecayStrategy.LINEAR: cls.linear,
            DecayStrategy.LOGARITHMIC: cls.logarithmic,
            DecayStrategy.STEP: cls.step,
            DecayStrategy.NONE: cls.none,
        }
        return mapping.get(strategy, cls.exponential)


# =============================================================================
# Memory Entry
# =============================================================================


@dataclass
class MemoryEntry:
    """
    Represents a single memory entry with decay properties.

    Attributes:
        memory_id: Unique identifier
        memory_type: Type classification
        key: Lookup key
        data: The stored data
        created_at: Creation timestamp
        last_accessed: Last access timestamp
        access_count: Number of times accessed
        initial_relevance: Starting relevance score
        current_relevance: Current relevance after decay
        decay_rate: Rate parameter for decay function
        decay_strategy: Which decay function to use
        metadata: Additional metadata
        tags: Searchable tags
    """
    memory_id: str
    memory_type: MemoryType
    key: str
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    initial_relevance: float = 1.0
    current_relevance: float = 1.0
    decay_rate: float = 0.01
    decay_strategy: DecayStrategy = DecayStrategy.EXPONENTIAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Validate and normalize fields."""
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
        if isinstance(self.decay_strategy, str):
            self.decay_strategy = DecayStrategy(self.decay_strategy)

    def calculate_relevance(self, current_time: Optional[datetime] = None) -> float:
        """
        Calculate current relevance based on decay.

        Args:
            current_time: Time to calculate for (default: now)

        Returns:
            Current relevance score (0-1)
        """
        if current_time is None:
            current_time = datetime.now()

        elapsed = current_time - self.created_at
        hours = elapsed.total_seconds() / 3600

        decay_fn = DecayFunction.get_function(self.decay_strategy)
        self.current_relevance = decay_fn(
            self.initial_relevance,
            hours,
            self.decay_rate
        )

        return self.current_relevance

    def reinforce(self, boost: float = 0.1) -> float:
        """
        Reinforce memory relevance after access.

        Args:
            boost: Amount to boost relevance

        Returns:
            New relevance score
        """
        self.access_count += 1
        self.last_accessed = datetime.now()

        # Boost relevance, but don't exceed initial
        self.current_relevance = min(
            self.initial_relevance,
            self.current_relevance + boost
        )

        return self.current_relevance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.name,
            "key": self.key,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "initial_relevance": self.initial_relevance,
            "current_relevance": self.current_relevance,
            "decay_rate": self.decay_rate,
            "decay_strategy": self.decay_strategy.value,
            "metadata": self.metadata,
            "tags": list(self.tags)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryEntry:
        """Create from dictionary."""
        return cls(
            memory_id=data["memory_id"],
            memory_type=MemoryType[data["memory_type"]],
            key=data["key"],
            data=data["data"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            initial_relevance=data.get("initial_relevance", 1.0),
            current_relevance=data.get("current_relevance", 1.0),
            decay_rate=data.get("decay_rate", 0.01),
            decay_strategy=DecayStrategy(data.get("decay_strategy", "exponential")),
            metadata=data.get("metadata", {}),
            tags=set(data.get("tags", []))
        )


# =============================================================================
# Spatial Signature Cache (LSH)
# =============================================================================


class SpatialSignatureCache:
    """
    Locality-Sensitive Hashing cache for spatial signatures.

    Enables fast approximate nearest neighbor search for high-dimensional
    spatial signature vectors. Uses random hyperplane LSH.
    """

    def __init__(
        self,
        dimension: int = 128,
        num_tables: int = 10,
        num_bits: int = 16,
        seed: int = 42
    ) -> None:
        """
        Initialize the LSH cache.

        Args:
            dimension: Dimensionality of signature vectors
            num_tables: Number of hash tables (more = higher recall)
            num_bits: Bits per hash (more = higher precision)
            seed: Random seed for reproducibility
        """
        self.dimension = dimension
        self.num_tables = num_tables
        self.num_bits = num_bits

        np.random.seed(seed)

        # Generate random hyperplanes for each table
        self.hyperplanes = [
            np.random.randn(num_bits, dimension)
            for _ in range(num_tables)
        ]

        # Hash tables: table_idx -> hash_value -> list of (memory_id, vector)
        self.tables: List[Dict[int, List[Tuple[str, np.ndarray]]]] = [
            {} for _ in range(num_tables)
        ]

        # Store all signatures for exact distance when needed
        self.signatures: Dict[str, np.ndarray] = {}

        logger.info(
            f"Initialized SpatialSignatureCache: "
            f"{dimension}D, {num_tables} tables, {num_bits} bits"
        )

    def _hash_vector(
        self,
        vector: np.ndarray,
        table_idx: int
    ) -> int:
        """Compute LSH hash for a vector using specified table."""
        projections = np.dot(self.hyperplanes[table_idx], vector)
        bits = (projections > 0).astype(int)
        # Convert bit array to integer
        return int("".join(str(b) for b in bits), 2)

    def add(self, memory_id: str, signature: np.ndarray) -> None:
        """
        Add a signature to the cache.

        Args:
            memory_id: Unique identifier for this signature
            signature: The spatial signature vector
        """
        if len(signature) != self.dimension:
            raise ValueError(
                f"Signature dimension {len(signature)} != "
                f"expected {self.dimension}"
            )

        # Normalize vector
        norm = np.linalg.norm(signature)
        if norm > 0:
            normalized = signature / norm
        else:
            normalized = signature

        # Store in all tables
        for table_idx in range(self.num_tables):
            hash_val = self._hash_vector(normalized, table_idx)
            if hash_val not in self.tables[table_idx]:
                self.tables[table_idx][hash_val] = []
            self.tables[table_idx][hash_val].append((memory_id, normalized.copy()))

        # Store full signature
        self.signatures[memory_id] = normalized.copy()

        logger.debug(f"Added signature {memory_id} to LSH cache")

    def remove(self, memory_id: str) -> bool:
        """
        Remove a signature from the cache.

        Args:
            memory_id: ID of signature to remove

        Returns:
            True if found and removed
        """
        if memory_id not in self.signatures:
            return False

        signature = self.signatures.pop(memory_id)

        # Remove from all tables
        for table_idx in range(self.num_tables):
            hash_val = self._hash_vector(signature, table_idx)
            if hash_val in self.tables[table_idx]:
                self.tables[table_idx][hash_val] = [
                    (mid, sig) for mid, sig in self.tables[table_idx][hash_val]
                    if mid != memory_id
                ]
                if not self.tables[table_idx][hash_val]:
                    del self.tables[table_idx][hash_val]

        return True

    def query(
        self,
        signature: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors to a query signature.

        Args:
            signature: Query vector
            k: Number of neighbors to return
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of (memory_id, similarity) tuples, sorted by similarity
        """
        if len(signature) != self.dimension:
            raise ValueError(
                f"Query dimension {len(signature)} != "
                f"expected {self.dimension}"
            )

        # Normalize query
        norm = np.linalg.norm(signature)
        if norm > 0:
            normalized = signature / norm
        else:
            normalized = signature

        # Collect candidates from all tables
        candidates: Set[str] = set()
        for table_idx in range(self.num_tables):
            hash_val = self._hash_vector(normalized, table_idx)
            if hash_val in self.tables[table_idx]:
                for memory_id, _ in self.tables[table_idx][hash_val]:
                    candidates.add(memory_id)

        # Calculate exact similarities for candidates
        results = []
        for memory_id in candidates:
            stored = self.signatures[memory_id]
            similarity = float(np.dot(normalized, stored))
            if similarity >= min_similarity:
                results.append((memory_id, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = sum(
            sum(len(bucket) for bucket in table.values())
            for table in self.tables
        )

        bucket_sizes = [
            len(bucket)
            for table in self.tables
            for bucket in table.values()
        ]

        return {
            "dimension": self.dimension,
            "num_tables": self.num_tables,
            "num_bits": self.num_bits,
            "unique_signatures": len(self.signatures),
            "total_entries": total_entries,
            "avg_bucket_size": np.mean(bucket_sizes) if bucket_sizes else 0,
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0
        }


# =============================================================================
# Persistent Memory Manager
# =============================================================================


class PersistentMemoryManager:
    """
    SQLite-backed persistent memory manager with decay processing.

    Provides:
    - Persistent storage of memory entries
    - Automatic decay processing
    - Relevance-based retrieval
    - Spatial signature indexing
    - Memory reinforcement
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        signature_dimension: int = 128,
        min_relevance: float = 0.01,
        decay_interval_hours: float = 1.0
    ) -> None:
        """
        Initialize the memory manager.

        Args:
            db_path: Path to SQLite database file
            signature_dimension: Dimension for spatial signatures
            min_relevance: Minimum relevance before pruning
            decay_interval_hours: How often to process decay
        """
        self.db_path = db_path
        self.min_relevance = min_relevance
        self.decay_interval_hours = decay_interval_hours

        # Initialize spatial signature cache
        self.signature_cache = SpatialSignatureCache(
            dimension=signature_dimension
        )

        # In-memory cache for fast access
        self._cache: Dict[str, MemoryEntry] = {}

        # Initialize database
        self._init_db()

        # Load existing data
        self._load_from_db()

        logger.info(f"PersistentMemoryManager initialized: {db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                key TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                initial_relevance REAL DEFAULT 1.0,
                current_relevance REAL DEFAULT 1.0,
                decay_rate REAL DEFAULT 0.01,
                decay_strategy TEXT DEFAULT 'exponential',
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signatures (
                memory_id TEXT PRIMARY KEY,
                signature BLOB NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
                    ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type
            ON memories(memory_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_key
            ON memories(key)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_relevance
            ON memories(current_relevance)
        """)

        conn.commit()
        conn.close()

    def _load_from_db(self) -> None:
        """Load memories from database into cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM memories")
        rows = cursor.fetchall()

        for row in rows:
            entry = MemoryEntry(
                memory_id=row[0],
                memory_type=MemoryType[row[1]],
                key=row[2],
                data=json.loads(row[3]),
                created_at=datetime.fromisoformat(row[4]),
                last_accessed=datetime.fromisoformat(row[5]),
                access_count=row[6],
                initial_relevance=row[7],
                current_relevance=row[8],
                decay_rate=row[9],
                decay_strategy=DecayStrategy(row[10]),
                metadata=json.loads(row[11]),
                tags=set(json.loads(row[12]))
            )
            self._cache[entry.memory_id] = entry

        # Load signatures
        cursor.execute("SELECT memory_id, signature FROM signatures")
        for memory_id, sig_bytes in cursor.fetchall():
            signature = np.frombuffer(sig_bytes, dtype=np.float64)
            self.signature_cache.add(memory_id, signature)

        conn.close()

        logger.info(f"Loaded {len(self._cache)} memories from database")

    def _save_entry(self, entry: MemoryEntry) -> None:
        """Save a memory entry to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO memories
            (memory_id, memory_type, key, data, created_at, last_accessed,
             access_count, initial_relevance, current_relevance, decay_rate,
             decay_strategy, metadata, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.memory_id,
            entry.memory_type.name,
            entry.key,
            json.dumps(entry.data),
            entry.created_at.isoformat(),
            entry.last_accessed.isoformat(),
            entry.access_count,
            entry.initial_relevance,
            entry.current_relevance,
            entry.decay_rate,
            entry.decay_strategy.value,
            json.dumps(entry.metadata),
            json.dumps(list(entry.tags))
        ))

        conn.commit()
        conn.close()

    def store(
        self,
        key: str,
        data: Dict[str, Any],
        memory_type: MemoryType,
        initial_relevance: float = 1.0,
        decay_rate: float = 0.01,
        decay_strategy: DecayStrategy = DecayStrategy.EXPONENTIAL,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        spatial_signature: Optional[np.ndarray] = None
    ) -> MemoryEntry:
        """
        Store a new memory entry.

        Args:
            key: Lookup key
            data: Data to store
            memory_type: Type classification
            initial_relevance: Starting relevance (0-1)
            decay_rate: Decay rate parameter
            decay_strategy: Which decay function to use
            metadata: Additional metadata
            tags: Searchable tags
            spatial_signature: Optional signature vector for similarity search

        Returns:
            The created memory entry
        """
        import uuid

        memory_id = str(uuid.uuid4())

        entry = MemoryEntry(
            memory_id=memory_id,
            memory_type=memory_type,
            key=key,
            data=data,
            initial_relevance=initial_relevance,
            current_relevance=initial_relevance,
            decay_rate=decay_rate,
            decay_strategy=decay_strategy,
            metadata=metadata or {},
            tags=tags or set()
        )

        # Store in cache
        self._cache[memory_id] = entry

        # Save to database
        self._save_entry(entry)

        # Handle spatial signature
        if spatial_signature is not None:
            self._store_signature(memory_id, spatial_signature)

        logger.debug(f"Stored memory: {memory_id} ({key})")

        return entry

    def _store_signature(
        self,
        memory_id: str,
        signature: np.ndarray
    ) -> None:
        """Store a spatial signature."""
        # Add to LSH cache
        self.signature_cache.add(memory_id, signature)

        # Persist to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO signatures (memory_id, signature)
            VALUES (?, ?)
        """, (memory_id, signature.astype(np.float64).tobytes()))

        conn.commit()
        conn.close()

    def retrieve(
        self,
        key: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        memory_id: Optional[str] = None,
        min_relevance: Optional[float] = None,
        tags: Optional[Set[str]] = None,
        limit: int = 100
    ) -> List[MemoryEntry]:
        """
        Retrieve memory entries matching criteria.

        Args:
            key: Filter by key
            memory_type: Filter by type
            memory_id: Get specific memory
            min_relevance: Minimum relevance threshold
            tags: Required tags (any match)
            limit: Maximum entries to return

        Returns:
            List of matching entries, sorted by relevance
        """
        if memory_id and memory_id in self._cache:
            entry = self._cache[memory_id]
            entry.calculate_relevance()
            entry.reinforce(0.05)
            self._save_entry(entry)
            return [entry]

        results = []
        min_rel = min_relevance if min_relevance is not None else 0.0

        for entry in self._cache.values():
            # Calculate current relevance
            entry.calculate_relevance()

            # Apply filters
            if entry.current_relevance < min_rel:
                continue
            if key is not None and entry.key != key:
                continue
            if memory_type is not None and entry.memory_type != memory_type:
                continue
            if tags is not None and not tags.intersection(entry.tags):
                continue

            results.append(entry)

        # Sort by relevance
        results.sort(key=lambda e: e.current_relevance, reverse=True)

        # Reinforce accessed entries
        for entry in results[:limit]:
            entry.reinforce(0.02)
            self._save_entry(entry)

        return results[:limit]

    def retrieve_similar(
        self,
        signature: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.5,
        min_relevance: float = 0.0
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve memories with similar spatial signatures.

        Args:
            signature: Query signature vector
            k: Number of results
            min_similarity: Minimum cosine similarity
            min_relevance: Minimum relevance threshold

        Returns:
            List of (entry, similarity) tuples
        """
        # Query LSH cache
        candidates = self.signature_cache.query(
            signature, k=k * 2, min_similarity=min_similarity
        )

        results = []
        for memory_id, similarity in candidates:
            if memory_id not in self._cache:
                continue

            entry = self._cache[memory_id]
            entry.calculate_relevance()

            if entry.current_relevance >= min_relevance:
                results.append((entry, similarity))
                entry.reinforce(0.05)
                self._save_entry(entry)

        # Sort by combined score
        results.sort(
            key=lambda x: x[0].current_relevance * x[1],
            reverse=True
        )

        return results[:k]

    def reinforce(
        self,
        memory_id: str,
        boost: float = 0.1
    ) -> Optional[float]:
        """
        Reinforce a memory's relevance.

        Args:
            memory_id: ID of memory to reinforce
            boost: Amount to boost relevance

        Returns:
            New relevance, or None if not found
        """
        if memory_id not in self._cache:
            return None

        entry = self._cache[memory_id]
        new_relevance = entry.reinforce(boost)
        self._save_entry(entry)

        logger.debug(f"Reinforced {memory_id}: {new_relevance:.3f}")

        return new_relevance

    def process_decay(self) -> Dict[str, Any]:
        """
        Process decay for all memories and prune low-relevance entries.

        Returns:
            Statistics about the decay processing
        """
        pruned = 0
        updated = 0

        for memory_id in list(self._cache.keys()):
            entry = self._cache[memory_id]
            old_relevance = entry.current_relevance
            new_relevance = entry.calculate_relevance()

            if new_relevance < self.min_relevance:
                # Prune this memory
                self._delete_entry(memory_id)
                pruned += 1
            elif new_relevance != old_relevance:
                self._save_entry(entry)
                updated += 1

        logger.info(
            f"Decay processing: {updated} updated, {pruned} pruned"
        )

        return {
            "updated": updated,
            "pruned": pruned,
            "remaining": len(self._cache)
        }

    def _delete_entry(self, memory_id: str) -> None:
        """Delete a memory entry."""
        # Remove from cache
        self._cache.pop(memory_id, None)

        # Remove from signature cache
        self.signature_cache.remove(memory_id)

        # Remove from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM signatures WHERE memory_id = ?",
            (memory_id,)
        )
        cursor.execute(
            "DELETE FROM memories WHERE memory_id = ?",
            (memory_id,)
        )

        conn.commit()
        conn.close()

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if found and deleted
        """
        if memory_id not in self._cache:
            return False

        self._delete_entry(memory_id)
        logger.debug(f"Deleted memory: {memory_id}")
        return True

    def clear(self) -> int:
        """
        Clear all memories.

        Returns:
            Number of memories cleared
        """
        count = len(self._cache)

        self._cache.clear()
        self.signature_cache = SpatialSignatureCache(
            dimension=self.signature_cache.dimension
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM signatures")
        cursor.execute("DELETE FROM memories")
        conn.commit()
        conn.close()

        logger.info(f"Cleared {count} memories")
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        type_counts = {}
        for entry in self._cache.values():
            type_name = entry.memory_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        relevance_values = [
            entry.current_relevance for entry in self._cache.values()
        ]

        return {
            "total_memories": len(self._cache),
            "by_type": type_counts,
            "avg_relevance": np.mean(relevance_values) if relevance_values else 0,
            "min_relevance_threshold": self.min_relevance,
            "db_path": self.db_path,
            "signature_cache": self.signature_cache.get_statistics()
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_memory_manager(
    db_path: str = "data/memory.db",
    **kwargs: Any
) -> PersistentMemoryManager:
    """
    Create a configured memory manager.

    Args:
        db_path: Path to database file
        **kwargs: Additional configuration

    Returns:
        Configured PersistentMemoryManager
    """
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    return PersistentMemoryManager(
        db_path=db_path,
        **kwargs
    )


# =============================================================================
# Demonstration
# =============================================================================


async def main() -> None:
    """Demonstrate memory system functionality."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 70)
    print("Persistent Memory System - Demonstration")
    print("=" * 70)

    # Create memory manager (use temp file for demo)
    import tempfile
    db_path = os.path.join(tempfile.gettempdir(), "demo_memory.db")

    manager = create_memory_manager(
        db_path=db_path,
        signature_dimension=64,
        min_relevance=0.05
    )

    # Store some memories
    print("\n1. Storing memories...")

    # Spatial signature memory
    signature1 = np.random.randn(64)
    entry1 = manager.store(
        key="patient_001_scan_001",
        data={
            "patient_id": "P001",
            "scan_type": "CT",
            "findings": ["normal", "no_anomalies"],
            "confidence": 0.95
        },
        memory_type=MemoryType.SPATIAL_SIGNATURE,
        tags={"ct", "normal", "chest"},
        spatial_signature=signature1
    )
    print(f"   Stored: {entry1.key} (ID: {entry1.memory_id[:8]}...)")

    # Reconstruction params memory
    entry2 = manager.store(
        key="recon_profile_standard",
        data={
            "algorithm": "filtered_backprojection",
            "iterations": 50,
            "filter": "ram-lak",
            "resolution": [512, 512, 512]
        },
        memory_type=MemoryType.RECONSTRUCTION_PARAMS,
        decay_strategy=DecayStrategy.LOGARITHMIC,
        tags={"reconstruction", "standard"}
    )
    print(f"   Stored: {entry2.key} (ID: {entry2.memory_id[:8]}...)")

    # Agent experience memory
    entry3 = manager.store(
        key="agent_exp_001",
        data={
            "agent_id": "acq_agent_001",
            "task_type": "dicom_import",
            "avg_time_ms": 150,
            "success_rate": 0.98
        },
        memory_type=MemoryType.AGENT_EXPERIENCE,
        decay_rate=0.005,  # Slower decay
        tags={"agent", "performance"}
    )
    print(f"   Stored: {entry3.key} (ID: {entry3.memory_id[:8]}...)")

    # Store another spatial signature for similarity search
    signature2 = signature1 + np.random.randn(64) * 0.1  # Similar to signature1
    entry4 = manager.store(
        key="patient_002_scan_001",
        data={
            "patient_id": "P002",
            "scan_type": "CT",
            "findings": ["normal"],
            "confidence": 0.92
        },
        memory_type=MemoryType.SPATIAL_SIGNATURE,
        tags={"ct", "normal", "chest"},
        spatial_signature=signature2
    )
    print(f"   Stored: {entry4.key} (ID: {entry4.memory_id[:8]}...)")

    # Display statistics
    print("\n2. Memory Statistics:")
    stats = manager.get_statistics()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   By type: {stats['by_type']}")
    print(f"   Avg relevance: {stats['avg_relevance']:.3f}")

    # Retrieve by type
    print("\n3. Retrieving by type...")
    results = manager.retrieve(memory_type=MemoryType.SPATIAL_SIGNATURE)
    print(f"   Found {len(results)} spatial signatures")
    for entry in results:
        print(f"     - {entry.key}: relevance={entry.current_relevance:.3f}")

    # Similarity search
    print("\n4. Similarity search...")
    query_signature = signature1 + np.random.randn(64) * 0.05
    similar = manager.retrieve_similar(
        signature=query_signature,
        k=5,
        min_similarity=0.5
    )
    print(f"   Found {len(similar)} similar signatures")
    for entry, similarity in similar:
        print(f"     - {entry.key}: similarity={similarity:.3f}")

    # Reinforce a memory
    print("\n5. Reinforcing memory...")
    before = entry1.current_relevance
    manager.reinforce(entry1.memory_id, boost=0.2)
    after = manager.retrieve(memory_id=entry1.memory_id)[0].current_relevance
    print(f"   Relevance: {before:.3f} -> {after:.3f}")

    # Process decay
    print("\n6. Processing decay...")
    decay_stats = manager.process_decay()
    print(f"   Updated: {decay_stats['updated']}")
    print(f"   Pruned: {decay_stats['pruned']}")
    print(f"   Remaining: {decay_stats['remaining']}")

    # Clean up
    manager.clear()

    print("\n" + "=" * 70)
    print("Memory system demonstration complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
