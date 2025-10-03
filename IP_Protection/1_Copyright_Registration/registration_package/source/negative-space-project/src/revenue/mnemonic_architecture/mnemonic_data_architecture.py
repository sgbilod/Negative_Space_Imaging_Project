"""
Mnemonic Data Architecture (Project "Mnemosyne")

This module implements a new data storage architecture that encodes information within virtual
negative spaces, creating a spatial-mnemonic system that leverages the human brain's powerful
spatial memory (the "Method of Loci"). It's a file system you can 'walk through'.
"""

import hashlib
import json
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import numpy as np

from ...negative_mapping.spatial_signature_generator import SpatialSignatureGenerator
from ...negative_mapping.void_signature_extractor import VoidSignatureExtractor


@dataclass
class SpatialNode:
    """
    A node in the spatial data architecture.
    Each node represents a piece of data placed in a specific position in 3D space.
    """
    node_id: str
    position: List[float]  # [x, y, z]
    data_type: str  # "file", "folder", "tag", "connection", etc.
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    connections: List[str] = field(default_factory=list)  # List of node_ids
    
    def distance_to(self, other: 'SpatialNode') -> float:
        """Calculate Euclidean distance to another node."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, other.position)))
    
    def semantic_distance_to(self, other: 'SpatialNode', semantic_space: Dict[str, List[float]]) -> float:
        """
        Calculate semantic distance to another node based on content similarity.
        
        Args:
            other: The other node to compare with
            semantic_space: Dictionary mapping content to semantic vectors
            
        Returns:
            Semantic distance score (lower means more similar)
        """
        # Get semantic vectors
        my_vector = semantic_space.get(str(self.content), None)
        other_vector = semantic_space.get(str(other.content), None)
        
        # If either doesn't have a vector, return a high distance
        if my_vector is None or other_vector is None:
            return 1000.0
            
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(my_vector, other_vector))
        norm_a = math.sqrt(sum(a * a for a in my_vector))
        norm_b = math.sqrt(sum(b * b for b in other_vector))
        
        # Convert similarity to distance (1 - similarity)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - (dot_product / (norm_a * norm_b))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "position": self.position,
            "data_type": self.data_type,
            "content": self.content,
            "metadata": self.metadata,
            "creation_time": self.creation_time,
            "last_modified": self.last_modified,
            "connections": self.connections
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpatialNode':
        """Create from dictionary after deserialization."""
        return cls(
            node_id=data["node_id"],
            position=data["position"],
            data_type=data["data_type"],
            content=data["content"],
            metadata=data["metadata"],
            creation_time=data["creation_time"],
            last_modified=data["last_modified"],
            connections=data["connections"]
        )


@dataclass
class Cluster:
    """
    A cluster of related nodes in the spatial architecture.
    Clusters group semantically related data in proximity.
    """
    cluster_id: str
    name: str
    center: List[float]  # Central position [x, y, z]
    radius: float  # Radius of influence
    theme: str  # Visual/conceptual theme
    member_nodes: List[str] = field(default_factory=list)  # List of node_ids
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "center": self.center,
            "radius": self.radius,
            "theme": self.theme,
            "member_nodes": self.member_nodes,
            "metadata": self.metadata,
            "creation_time": self.creation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Cluster':
        """Create from dictionary after deserialization."""
        return cls(
            cluster_id=data["cluster_id"],
            name=data["name"],
            center=data["center"],
            radius=data["radius"],
            theme=data["theme"],
            member_nodes=data["member_nodes"],
            metadata=data["metadata"],
            creation_time=data["creation_time"]
        )


@dataclass
class Path:
    """
    A path represents a journey through the spatial data structure.
    Paths can be used for guided tours, presentations, or navigation aids.
    """
    path_id: str
    name: str
    nodes: List[str]  # Ordered list of node_ids to visit
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path_id": self.path_id,
            "name": self.name,
            "nodes": self.nodes,
            "metadata": self.metadata,
            "creation_time": self.creation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Path':
        """Create from dictionary after deserialization."""
        return cls(
            path_id=data["path_id"],
            name=data["name"],
            nodes=data["nodes"],
            metadata=data["metadata"],
            creation_time=data["creation_time"]
        )


class AIDataCartographer:
    """
    AI system that organizes data into an intuitive 3D spatial layout.
    This system analyzes data relationships and builds a 3D "memory palace".
    """
    
    def __init__(self, 
                 initial_volume_size: List[float] = None,
                 learning_rate: float = 0.05,
                 similarity_threshold: float = 0.7):
        """
        Initialize the AI Data Cartographer.
        
        Args:
            initial_volume_size: Size of the initial 3D space [x, y, z]
            learning_rate: Rate of adjustment for spatial positioning
            similarity_threshold: Threshold for considering items similar
        """
        self.initial_volume_size = initial_volume_size or [100.0, 100.0, 100.0]
        self.learning_rate = learning_rate
        self.similarity_threshold = similarity_threshold
        
        # Initialize semantic space
        self.semantic_space = {}  # content_hash -> semantic_vector
        
        # Working memory
        self.current_analysis = None
        self.pending_placements = []
        
    def analyze_data_relationships(self, 
                                 data_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze relationships between data items to determine spatial positions.
        
        Args:
            data_items: List of data items to analyze
            
        Returns:
            Analysis results
        """
        # Create a matrix for similarity calculations
        n_items = len(data_items)
        similarity_matrix = np.zeros((n_items, n_items))
        
        # Calculate pairwise similarities
        for i in range(n_items):
            for j in range(i+1, n_items):
                similarity = self._calculate_similarity(data_items[i], data_items[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Use dimensionality reduction to place items in 3D space
        # In a real implementation, this would use t-SNE, UMAP, or a similar algorithm
        # For this demo, we'll use a simplified approach
        
        # Generate initial random positions
        positions = []
        for _ in range(n_items):
            positions.append([
                random.uniform(0, self.initial_volume_size[0]),
                random.uniform(0, self.initial_volume_size[1]),
                random.uniform(0, self.initial_volume_size[2])
            ])
        
        # Iteratively adjust positions based on similarities
        positions = self._optimize_positions(positions, similarity_matrix)
        
        # Store results
        self.current_analysis = {
            "data_items": data_items,
            "similarity_matrix": similarity_matrix.tolist(),
            "positions": positions,
            "timestamp": datetime.now().isoformat()
        }
        
        # Determine clusters
        clusters = self._identify_clusters(positions, similarity_matrix)
        
        # Prepare results
        results = {
            "positions": positions,
            "clusters": clusters,
            "similarity_matrix": similarity_matrix.tolist()
        }
        
        return results
    
    def _calculate_similarity(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> float:
        """
        Calculate semantic similarity between two data items.
        
        Args:
            item1: First data item
            item2: Second data item
            
        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        # In a real implementation, this would use NLP techniques,
        # embedding models, and other ML approaches
        
        # For this demo, we'll simulate with a simplified approach
        
        # Check common tags/metadata
        tags1 = set(item1.get("tags", []))
        tags2 = set(item2.get("tags", []))
        common_tags = tags1.intersection(tags2)
        tag_similarity = len(common_tags) / max(1, len(tags1.union(tags2)))
        
        # Check type similarity
        type_similarity = 1.0 if item1.get("type") == item2.get("type") else 0.0
        
        # Check name/title similarity (crude approximation)
        name1 = item1.get("name", "").lower()
        name2 = item2.get("name", "").lower()
        
        # Count common words
        words1 = set(name1.split())
        words2 = set(name2.split())
        common_words = words1.intersection(words2)
        name_similarity = len(common_words) / max(1, len(words1.union(words2)))
        
        # Combine similarities with weights
        combined_similarity = (
            0.5 * tag_similarity +
            0.3 * name_similarity +
            0.2 * type_similarity
        )
        
        return combined_similarity
    
    def _optimize_positions(self, 
                          positions: List[List[float]], 
                          similarity_matrix: np.ndarray) -> List[List[float]]:
        """
        Optimize positions of data items in 3D space based on similarity.
        
        Args:
            positions: Initial positions
            similarity_matrix: Pairwise similarity matrix
            
        Returns:
            Optimized positions
        """
        # Number of iterations for position optimization
        n_iterations = 100
        n_items = len(positions)
        
        # Copy positions to avoid modifying the original
        optimized = [pos.copy() for pos in positions]
        
        # Iteratively adjust positions
        for _ in range(n_iterations):
            # For each pair of items
            for i in range(n_items):
                for j in range(n_items):
                    if i == j:
                        continue
                        
                    # Current distance
                    dist = math.sqrt(sum((optimized[i][k] - optimized[j][k])**2 for k in range(3)))
                    
                    # Ideal distance based on similarity
                    # Similar items should be closer
                    ideal_dist = 10.0 * (1.0 - similarity_matrix[i, j])
                    
                    # Direction vector
                    direction = [(optimized[j][k] - optimized[i][k]) / max(0.001, dist) for k in range(3)]
                    
                    # Adjust position
                    if dist > ideal_dist:
                        # Move closer
                        for k in range(3):
                            optimized[i][k] += direction[k] * self.learning_rate * (dist - ideal_dist)
                    else:
                        # Move away
                        for k in range(3):
                            optimized[i][k] -= direction[k] * self.learning_rate * (ideal_dist - dist)
                    
                    # Ensure within bounds
                    for k in range(3):
                        optimized[i][k] = max(0, min(self.initial_volume_size[k], optimized[i][k]))
        
        return optimized
    
    def _identify_clusters(self, 
                         positions: List[List[float]], 
                         similarity_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify natural clusters in the data based on position and similarity.
        
        Args:
            positions: Optimized positions
            similarity_matrix: Pairwise similarity matrix
            
        Returns:
            List of cluster information
        """
        # In a real implementation, this would use a clustering algorithm like DBSCAN
        # For this demo, we'll use a simplified approach
        
        n_items = len(positions)
        assigned = [False] * n_items
        clusters = []
        
        # Find clusters
        for i in range(n_items):
            if assigned[i]:
                continue
                
            # Start a new cluster
            cluster_members = [i]
            assigned[i] = True
            
            # Find similar items
            for j in range(n_items):
                if i == j or assigned[j]:
                    continue
                    
                if similarity_matrix[i, j] > self.similarity_threshold:
                    cluster_members.append(j)
                    assigned[j] = True
            
            # Only create clusters with at least 2 members
            if len(cluster_members) > 1:
                # Calculate cluster center
                center = [0, 0, 0]
                for idx in cluster_members:
                    for k in range(3):
                        center[k] += positions[idx][k]
                
                for k in range(3):
                    center[k] /= len(cluster_members)
                
                # Calculate radius
                max_dist = 0
                for idx in cluster_members:
                    dist = math.sqrt(sum((positions[idx][k] - center[k])**2 for k in range(3)))
                    max_dist = max(max_dist, dist)
                
                # Add cluster
                clusters.append({
                    "id": str(uuid.uuid4()),
                    "center": center,
                    "radius": max_dist * 1.2,  # Add some padding
                    "members": cluster_members
                })
        
        return clusters
    
    def suggest_node_placement(self, 
                             data_item: Dict[str, Any],
                             existing_nodes: List[SpatialNode]) -> Dict[str, Any]:
        """
        Suggest where to place a new data item within the existing spatial structure.
        
        Args:
            data_item: The data item to place
            existing_nodes: Existing nodes in the spatial structure
            
        Returns:
            Placement suggestion
        """
        # Calculate similarities with existing nodes
        similarities = []
        for node in existing_nodes:
            sim = self._calculate_similarity_to_node(data_item, node)
            similarities.append((node, sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # If we have similar nodes, place near them
        if similarities and similarities[0][1] > self.similarity_threshold:
            similar_node = similarities[0][0]
            
            # Generate position near the similar node
            # Add some randomness to avoid exact overlap
            position = [
                similar_node.position[0] + random.uniform(-5, 5),
                similar_node.position[1] + random.uniform(-5, 5),
                similar_node.position[2] + random.uniform(-5, 5)
            ]
            
            # Ensure within bounds
            for i in range(3):
                position[i] = max(0, min(self.initial_volume_size[i], position[i]))
            
            return {
                "position": position,
                "similar_to": similar_node.node_id,
                "similarity_score": similarities[0][1],
                "confidence": "high" if similarities[0][1] > 0.8 else "medium"
            }
        else:
            # If no similar nodes, find a less crowded area
            position = self._find_open_position(existing_nodes)
            
            return {
                "position": position,
                "similar_to": None,
                "similarity_score": 0.0,
                "confidence": "low"
            }
    
    def _calculate_similarity_to_node(self, data_item: Dict[str, Any], node: SpatialNode) -> float:
        """
        Calculate similarity between a data item and an existing node.
        
        Args:
            data_item: Data item
            node: Existing node
            
        Returns:
            Similarity score (0-1)
        """
        # Convert node to a comparable format
        node_data = {
            "type": node.data_type,
            "name": node.metadata.get("name", ""),
            "tags": node.metadata.get("tags", [])
        }
        
        return self._calculate_similarity(data_item, node_data)
    
    def _find_open_position(self, existing_nodes: List[SpatialNode]) -> List[float]:
        """
        Find a position in the space that isn't too crowded.
        
        Args:
            existing_nodes: Existing nodes
            
        Returns:
            Position [x, y, z]
        """
        if not existing_nodes:
            # If no existing nodes, just place in the center
            return [
                self.initial_volume_size[0] / 2,
                self.initial_volume_size[1] / 2,
                self.initial_volume_size[2] / 2
            ]
        
        # Create a grid representation of the space
        grid_size = 10
        grid = np.zeros((grid_size, grid_size, grid_size))
        
        # Mark areas near existing nodes
        for node in existing_nodes:
            # Convert position to grid coordinates
            grid_x = int(node.position[0] / self.initial_volume_size[0] * (grid_size-1))
            grid_y = int(node.position[1] / self.initial_volume_size[1] * (grid_size-1))
            grid_z = int(node.position[2] / self.initial_volume_size[2] * (grid_size-1))
            
            # Mark this cell and neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        x, y, z = grid_x + dx, grid_y + dy, grid_z + dz
                        if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
                            grid[x, y, z] += 1
        
        # Find the cell with lowest occupancy
        min_val = float('inf')
        min_pos = [0, 0, 0]
        
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    if grid[x, y, z] < min_val:
                        min_val = grid[x, y, z]
                        min_pos = [x, y, z]
        
        # Convert back to actual coordinates
        position = [
            min_pos[0] / (grid_size-1) * self.initial_volume_size[0],
            min_pos[1] / (grid_size-1) * self.initial_volume_size[1],
            min_pos[2] / (grid_size-1) * self.initial_volume_size[2]
        ]
        
        # Add some randomness
        position = [
            position[0] + random.uniform(-2, 2),
            position[1] + random.uniform(-2, 2),
            position[2] + random.uniform(-2, 2)
        ]
        
        # Ensure within bounds
        for i in range(3):
            position[i] = max(0, min(self.initial_volume_size[i], position[i]))
        
        return position
    
    def generate_semantic_vector(self, content: Any) -> List[float]:
        """
        Generate a semantic vector for content.
        
        Args:
            content: The content to generate a vector for
            
        Returns:
            Semantic vector
        """
        # In a real implementation, this would use embeddings from language models
        # For this demo, we'll simulate with a simplified approach
        
        # Hash the content to get a stable representation
        content_str = str(content)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        
        # Check if we already have a vector for this content
        if content_hash in self.semantic_space:
            return self.semantic_space[content_hash]
        
        # Generate a random 10-dimensional vector
        # In reality, this would be a meaningful embedding
        vector = [random.uniform(-1, 1) for _ in range(10)]
        
        # Normalize the vector
        norm = math.sqrt(sum(x*x for x in vector))
        vector = [x/norm for x in vector]
        
        # Store in semantic space
        self.semantic_space[content_hash] = vector
        
        return vector


class MnemonicDataArchitecture:
    """
    Main class for the Mnemonic Data Architecture system.
    This system creates a 3D spatial organization of data for intuitive navigation and recall.
    """
    
    def __init__(self):
        """Initialize the Mnemonic Data Architecture system."""
        # Core components
        self.signature_generator = SpatialSignatureGenerator()
        self.ai_cartographer = AIDataCartographer()
        
        # Data storage
        self.nodes = {}  # node_id -> SpatialNode
        self.clusters = {}  # cluster_id -> Cluster
        self.paths = {}  # path_id -> Path
        
        # Spatial indices for efficient querying
        self.spatial_index = {}  # grid_cell_key -> List[node_id]
        self.grid_size = 10.0  # Size of grid cells for spatial indexing
        
        # Metadata
        self.metadata = {
            "name": "Mnemonic Data Architecture",
            "created": datetime.now().isoformat(),
            "version": "1.0",
            "volume_size": [100.0, 100.0, 100.0]  # Default volume size
        }
    
    def add_node(self, 
                data_type: str,
                content: Any,
                position: Optional[List[float]] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a new node to the spatial structure.
        
        Args:
            data_type: Type of data ("file", "folder", "tag", etc.)
            content: The data content
            position: Optional manual position, otherwise AI will determine
            metadata: Additional metadata
            
        Returns:
            Node creation result
        """
        # Generate a new node ID
        node_id = str(uuid.uuid4())
        
        # If position not provided, use AI to suggest placement
        if position is None:
            # Prepare data item for AI
            data_item = {
                "type": data_type,
                "name": metadata.get("name", "") if metadata else "",
                "tags": metadata.get("tags", []) if metadata else []
            }
            
            # Get suggestion
            suggestion = self.ai_cartographer.suggest_node_placement(
                data_item=data_item,
                existing_nodes=list(self.nodes.values())
            )
            
            position = suggestion["position"]
        
        # Create the node
        node = SpatialNode(
            node_id=node_id,
            position=position,
            data_type=data_type,
            content=content,
            metadata=metadata or {}
        )
        
        # Store the node
        self.nodes[node_id] = node
        
        # Update spatial index
        self._update_spatial_index_for_node(node)
        
        # Check if this node belongs in any existing clusters
        for cluster in self.clusters.values():
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(position, cluster.center)))
            if distance <= cluster.radius:
                cluster.member_nodes.append(node_id)
        
        return {
            "success": True,
            "node_id": node_id,
            "position": position
        }
    
    def update_node(self, 
                   node_id: str,
                   updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing node.
        
        Args:
            node_id: ID of the node to update
            updates: Dictionary of fields to update
            
        Returns:
            Update result
        """
        # Check if the node exists
        if node_id not in self.nodes:
            return {
                "success": False,
                "error": "Node not found",
                "node_id": node_id
            }
        
        # Get the node
        node = self.nodes[node_id]
        
        # Handle position update separately
        if "position" in updates:
            # Remove from old position in spatial index
            self._remove_from_spatial_index(node)
            
            # Update position
            node.position = updates["position"]
            
            # Add to new position in spatial index
            self._update_spatial_index_for_node(node)
            
            # Update cluster memberships
            self._update_cluster_memberships(node)
        
        # Update other fields
        for key, value in updates.items():
            if key == "position":
                continue  # Already handled
            elif key == "content":
                node.content = value
            elif key == "metadata":
                node.metadata.update(value)
            elif key == "data_type":
                node.data_type = value
            elif key == "connections":
                node.connections = value
        
        # Update last modified time
        node.last_modified = datetime.now().isoformat()
        
        return {
            "success": True,
            "node_id": node_id,
            "updated_fields": list(updates.keys())
        }
    
    def delete_node(self, node_id: str) -> Dict[str, Any]:
        """
        Delete a node from the spatial structure.
        
        Args:
            node_id: ID of the node to delete
            
        Returns:
            Deletion result
        """
        # Check if the node exists
        if node_id not in self.nodes:
            return {
                "success": False,
                "error": "Node not found",
                "node_id": node_id
            }
        
        # Get the node
        node = self.nodes[node_id]
        
        # Remove from spatial index
        self._remove_from_spatial_index(node)
        
        # Remove from clusters
        for cluster in self.clusters.values():
            if node_id in cluster.member_nodes:
                cluster.member_nodes.remove(node_id)
        
        # Remove from paths
        for path in self.paths.values():
            if node_id in path.nodes:
                path.nodes.remove(node_id)
        
        # Remove connections to this node from other nodes
        for other_node in self.nodes.values():
            if node_id in other_node.connections:
                other_node.connections.remove(node_id)
        
        # Delete the node
        del self.nodes[node_id]
        
        return {
            "success": True,
            "node_id": node_id
        }
    
    def create_cluster(self, 
                      name: str,
                      center: List[float],
                      radius: float,
                      theme: str,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new cluster in the spatial structure.
        
        Args:
            name: Name of the cluster
            center: Center position [x, y, z]
            radius: Radius of influence
            theme: Visual/conceptual theme
            metadata: Additional metadata
            
        Returns:
            Cluster creation result
        """
        # Generate a new cluster ID
        cluster_id = str(uuid.uuid4())
        
        # Find nodes that belong in this cluster
        member_nodes = []
        for node_id, node in self.nodes.items():
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(node.position, center)))
            if distance <= radius:
                member_nodes.append(node_id)
        
        # Create the cluster
        cluster = Cluster(
            cluster_id=cluster_id,
            name=name,
            center=center,
            radius=radius,
            theme=theme,
            member_nodes=member_nodes,
            metadata=metadata or {}
        )
        
        # Store the cluster
        self.clusters[cluster_id] = cluster
        
        return {
            "success": True,
            "cluster_id": cluster_id,
            "member_count": len(member_nodes)
        }
    
    def create_path(self, 
                   name: str,
                   nodes: List[str],
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new path through the spatial structure.
        
        Args:
            name: Name of the path
            nodes: Ordered list of node_ids to visit
            metadata: Additional metadata
            
        Returns:
            Path creation result
        """
        # Generate a new path ID
        path_id = str(uuid.uuid4())
        
        # Validate nodes
        valid_nodes = [node_id for node_id in nodes if node_id in self.nodes]
        
        if len(valid_nodes) != len(nodes):
            return {
                "success": False,
                "error": "Some nodes not found",
                "valid_nodes": len(valid_nodes),
                "total_nodes": len(nodes)
            }
        
        # Create the path
        path = Path(
            path_id=path_id,
            name=name,
            nodes=nodes,
            metadata=metadata or {}
        )
        
        # Store the path
        self.paths[path_id] = path
        
        return {
            "success": True,
            "path_id": path_id,
            "node_count": len(nodes)
        }
    
    def auto_organize_data(self, data_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Automatically organize a set of data items into the spatial structure.
        
        Args:
            data_items: List of data items to organize
            
        Returns:
            Organization results
        """
        # Use AI to analyze data relationships
        analysis = self.ai_cartographer.analyze_data_relationships(data_items)
        
        # Add nodes based on the analysis
        node_ids = []
        for i, item in enumerate(data_items):
            position = analysis["positions"][i]
            
            # Extract data from item
            data_type = item.get("type", "unknown")
            content = item.get("content", {})
            metadata = {
                "name": item.get("name", ""),
                "tags": item.get("tags", []),
                "description": item.get("description", ""),
                "source": item.get("source", "auto-organization")
            }
            
            # Create the node
            result = self.add_node(
                data_type=data_type,
                content=content,
                position=position,
                metadata=metadata
            )
            
            if result["success"]:
                node_ids.append(result["node_id"])
        
        # Create clusters based on the analysis
        cluster_ids = []
        for cluster_info in analysis["clusters"]:
            # Get a representative node for the cluster
            if cluster_info["members"]:
                rep_idx = cluster_info["members"][0]
                rep_item = data_items[rep_idx]
                
                # Use its name or generate a name
                cluster_name = rep_item.get("name", f"Cluster-{len(cluster_ids)+1}")
            else:
                cluster_name = f"Cluster-{len(cluster_ids)+1}"
            
            # Create the cluster
            result = self.create_cluster(
                name=cluster_name,
                center=cluster_info["center"],
                radius=cluster_info["radius"],
                theme="auto-generated",
                metadata={
                    "source": "auto-organization",
                    "member_count": len(cluster_info["members"])
                }
            )
            
            if result["success"]:
                cluster_ids.append(result["cluster_id"])
        
        # Create connections between similar nodes
        similarity_matrix = np.array(analysis["similarity_matrix"])
        for i in range(len(node_ids)):
            connections = []
            for j in range(len(node_ids)):
                if i != j and similarity_matrix[i, j] > self.ai_cartographer.similarity_threshold:
                    connections.append(node_ids[j])
            
            # Update node connections
            if connections:
                self.update_node(
                    node_id=node_ids[i],
                    updates={"connections": connections}
                )
        
        return {
            "success": True,
            "nodes_created": len(node_ids),
            "clusters_created": len(cluster_ids),
            "node_ids": node_ids,
            "cluster_ids": cluster_ids
        }
    
    def query_spatial_region(self, 
                            center: List[float],
                            radius: float) -> Dict[str, Any]:
        """
        Query nodes within a spatial region.
        
        Args:
            center: Center of the region [x, y, z]
            radius: Radius of the region
            
        Returns:
            Query results
        """
        # Find grid cells that intersect with the region
        grid_cells = self._get_grid_cells_for_region(center, radius)
        
        # Collect candidate node IDs from these cells
        candidate_ids = set()
        for cell_key in grid_cells:
            if cell_key in self.spatial_index:
                candidate_ids.update(self.spatial_index[cell_key])
        
        # Filter candidates by exact distance
        results = []
        for node_id in candidate_ids:
            node = self.nodes[node_id]
            
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(node.position, center)))
            
            if distance <= radius:
                results.append({
                    "node_id": node_id,
                    "distance": distance,
                    "data_type": node.data_type,
                    "position": node.position,
                    "metadata": node.metadata
                })
        
        # Sort by distance
        results.sort(key=lambda x: x["distance"])
        
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    
    def get_node_by_id(self, node_id: str) -> Dict[str, Any]:
        """
        Get a node by its ID.
        
        Args:
            node_id: ID of the node to get
            
        Returns:
            Node data
        """
        if node_id not in self.nodes:
            return {
                "success": False,
                "error": "Node not found",
                "node_id": node_id
            }
        
        node = self.nodes[node_id]
        
        return {
            "success": True,
            "node": node.to_dict()
        }
    
    def get_cluster_by_id(self, cluster_id: str) -> Dict[str, Any]:
        """
        Get a cluster by its ID.
        
        Args:
            cluster_id: ID of the cluster to get
            
        Returns:
            Cluster data
        """
        if cluster_id not in self.clusters:
            return {
                "success": False,
                "error": "Cluster not found",
                "cluster_id": cluster_id
            }
        
        cluster = self.clusters[cluster_id]
        
        # Get all nodes in the cluster
        nodes = []
        for node_id in cluster.member_nodes:
            if node_id in self.nodes:
                nodes.append(self.nodes[node_id].to_dict())
        
        return {
            "success": True,
            "cluster": cluster.to_dict(),
            "nodes": nodes
        }
    
    def get_path_by_id(self, path_id: str) -> Dict[str, Any]:
        """
        Get a path by its ID.
        
        Args:
            path_id: ID of the path to get
            
        Returns:
            Path data
        """
        if path_id not in self.paths:
            return {
                "success": False,
                "error": "Path not found",
                "path_id": path_id
            }
        
        path = self.paths[path_id]
        
        # Get all nodes in the path
        nodes = []
        for node_id in path.nodes:
            if node_id in self.nodes:
                nodes.append(self.nodes[node_id].to_dict())
        
        return {
            "success": True,
            "path": path.to_dict(),
            "nodes": nodes
        }
    
    def search_by_content(self, 
                         query: str,
                         max_results: int = 10) -> Dict[str, Any]:
        """
        Search for nodes by content.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Search results
        """
        # In a real implementation, this would use proper search algorithms
        # For this demo, we'll use a simplified approach
        
        results = []
        
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        for node_id, node in self.nodes.items():
            # Check if query matches content
            content_str = str(node.content).lower()
            
            # Check if query matches node name
            name = node.metadata.get("name", "").lower()
            
            # Check if query matches tags
            tags = [tag.lower() for tag in node.metadata.get("tags", [])]
            
            # Check if query matches description
            description = node.metadata.get("description", "").lower()
            
            # Calculate match score
            score = 0
            
            if query_lower in content_str:
                score += 3
            
            if query_lower in name:
                score += 5
                
            for tag in tags:
                if query_lower in tag:
                    score += 4
                    break
            
            if query_lower in description:
                score += 2
            
            # If we have a match, add to results
            if score > 0:
                results.append({
                    "node_id": node_id,
                    "score": score,
                    "data_type": node.data_type,
                    "position": node.position,
                    "metadata": node.metadata
                })
        
        # Sort by score (highest first) and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:max_results]
        
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    
    def query_cognitive_api(self, 
                           query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a cognitive query on the spatial data structure.
        
        Args:
            query: Query parameters
            
        Returns:
            Query results
        """
        query_type = query.get("type", "unknown")
        
        if query_type == "related_to":
            # Find nodes related to a specific node
            node_id = query.get("node_id")
            if not node_id or node_id not in self.nodes:
                return {
                    "success": False,
                    "error": "Invalid node_id",
                    "query_type": query_type
                }
            
            # Get the node
            node = self.nodes[node_id]
            
            # Get direct connections
            direct_connections = []
            for conn_id in node.connections:
                if conn_id in self.nodes:
                    direct_connections.append({
                        "node_id": conn_id,
                        "data_type": self.nodes[conn_id].data_type,
                        "metadata": self.nodes[conn_id].metadata,
                        "relationship": "direct"
                    })
            
            # Get nodes in the same clusters
            cluster_connections = []
            for cluster in self.clusters.values():
                if node_id in cluster.member_nodes:
                    for member_id in cluster.member_nodes:
                        if member_id != node_id and member_id not in node.connections:
                            cluster_connections.append({
                                "node_id": member_id,
                                "data_type": self.nodes[member_id].data_type,
                                "metadata": self.nodes[member_id].metadata,
                                "relationship": "cluster",
                                "cluster_id": cluster.cluster_id
                            })
            
            # Get spatially close nodes
            spatial_connections = []
            spatial_results = self.query_spatial_region(
                center=node.position,
                radius=10.0
            )
            
            for result in spatial_results.get("results", []):
                result_id = result["node_id"]
                if (result_id != node_id and 
                    result_id not in node.connections and
                    not any(c["node_id"] == result_id for c in cluster_connections)):
                    spatial_connections.append({
                        "node_id": result_id,
                        "data_type": self.nodes[result_id].data_type,
                        "metadata": self.nodes[result_id].metadata,
                        "relationship": "spatial",
                        "distance": result["distance"]
                    })
            
            return {
                "success": True,
                "query_type": query_type,
                "node_id": node_id,
                "direct_connections": direct_connections,
                "cluster_connections": cluster_connections,
                "spatial_connections": spatial_connections
            }
            
        elif query_type == "path_between":
            # Find a path between two nodes
            start_id = query.get("start_node_id")
            end_id = query.get("end_node_id")
            
            if not start_id or not end_id or start_id not in self.nodes or end_id not in self.nodes:
                return {
                    "success": False,
                    "error": "Invalid start or end node_id",
                    "query_type": query_type
                }
            
            # Find a path
            path = self._find_path(start_id, end_id)
            
            return {
                "success": True,
                "query_type": query_type,
                "start_node_id": start_id,
                "end_node_id": end_id,
                "path_found": len(path) > 0,
                "path": path
            }
            
        elif query_type == "semantic_similar":
            # Find semantically similar nodes
            content = query.get("content")
            if not content:
                return {
                    "success": False,
                    "error": "No content provided",
                    "query_type": query_type
                }
            
            # Generate semantic vector
            vector = self.ai_cartographer.generate_semantic_vector(content)
            
            # Get semantic vectors for all nodes
            semantic_space = {}
            for node_id, node in self.nodes.items():
                node_vector = self.ai_cartographer.generate_semantic_vector(node.content)
                semantic_space[str(node.content)] = node_vector
            
            # Calculate similarity with all nodes
            similarities = []
            for node_id, node in self.nodes.items():
                # Create a dummy node for the query content
                query_node = SpatialNode(
                    node_id="query",
                    position=[0, 0, 0],
                    data_type="query",
                    content=content
                )
                
                # Calculate semantic distance
                distance = node.semantic_distance_to(query_node, semantic_space)
                
                # Convert to similarity (1 - distance)
                similarity = 1.0 - distance
                
                if similarity > 0.5:  # Only include somewhat similar nodes
                    similarities.append({
                        "node_id": node_id,
                        "similarity": similarity,
                        "data_type": node.data_type,
                        "metadata": node.metadata
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "success": True,
                "query_type": query_type,
                "results": similarities[:10]  # Limit to top 10
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown query type: {query_type}",
                "supported_types": ["related_to", "path_between", "semantic_similar"]
            }
    
    def _update_spatial_index_for_node(self, node: SpatialNode):
        """
        Update the spatial index for a node.
        
        Args:
            node: Node to update in the index
        """
        # Get the grid cell key
        cell_key = self._get_grid_cell_key(node.position)
        
        # Add to the index
        if cell_key not in self.spatial_index:
            self.spatial_index[cell_key] = []
        
        if node.node_id not in self.spatial_index[cell_key]:
            self.spatial_index[cell_key].append(node.node_id)
    
    def _remove_from_spatial_index(self, node: SpatialNode):
        """
        Remove a node from the spatial index.
        
        Args:
            node: Node to remove from the index
        """
        # Get the grid cell key
        cell_key = self._get_grid_cell_key(node.position)
        
        # Remove from the index
        if cell_key in self.spatial_index and node.node_id in self.spatial_index[cell_key]:
            self.spatial_index[cell_key].remove(node.node_id)
    
    def _get_grid_cell_key(self, position: List[float]) -> str:
        """
        Get the grid cell key for a position.
        
        Args:
            position: Position [x, y, z]
            
        Returns:
            Grid cell key
        """
        # Convert position to grid cell coordinates
        cell_x = int(position[0] / self.grid_size)
        cell_y = int(position[1] / self.grid_size)
        cell_z = int(position[2] / self.grid_size)
        
        return f"{cell_x},{cell_y},{cell_z}"
    
    def _get_grid_cells_for_region(self, center: List[float], radius: float) -> List[str]:
        """
        Get grid cell keys that intersect with a region.
        
        Args:
            center: Center of the region [x, y, z]
            radius: Radius of the region
            
        Returns:
            List of grid cell keys
        """
        # Calculate the bounding box of the region
        min_x = max(0, center[0] - radius)
        max_x = center[0] + radius
        min_y = max(0, center[1] - radius)
        max_y = center[1] + radius
        min_z = max(0, center[2] - radius)
        max_z = center[2] + radius
        
        # Convert to grid cell coordinates
        min_cell_x = int(min_x / self.grid_size)
        max_cell_x = int(max_x / self.grid_size) + 1
        min_cell_y = int(min_y / self.grid_size)
        max_cell_y = int(max_y / self.grid_size) + 1
        min_cell_z = int(min_z / self.grid_size)
        max_cell_z = int(max_z / self.grid_size) + 1
        
        # Generate all cell keys in the bounding box
        cell_keys = []
        for x in range(min_cell_x, max_cell_x):
            for y in range(min_cell_y, max_cell_y):
                for z in range(min_cell_z, max_cell_z):
                    cell_keys.append(f"{x},{y},{z}")
        
        return cell_keys
    
    def _update_cluster_memberships(self, node: SpatialNode):
        """
        Update cluster memberships for a node after position change.
        
        Args:
            node: Node to update
        """
        # Check all clusters
        for cluster_id, cluster in self.clusters.items():
            # Calculate distance to cluster center
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(node.position, cluster.center)))
            
            # Check if node should be in this cluster
            if distance <= cluster.radius:
                # Add to cluster if not already a member
                if node.node_id not in cluster.member_nodes:
                    cluster.member_nodes.append(node.node_id)
            else:
                # Remove from cluster if a member
                if node.node_id in cluster.member_nodes:
                    cluster.member_nodes.remove(node.node_id)
    
    def _find_path(self, start_id: str, end_id: str) -> List[Dict[str, Any]]:
        """
        Find a path between two nodes using a simple breadth-first search.
        
        Args:
            start_id: Starting node ID
            end_id: Ending node ID
            
        Returns:
            List of nodes in the path
        """
        # If same node, return empty path
        if start_id == end_id:
            return [{
                "node_id": start_id,
                "data_type": self.nodes[start_id].data_type,
                "metadata": self.nodes[start_id].metadata
            }]
        
        # Queue for BFS
        queue = [(start_id, [start_id])]
        visited = set([start_id])
        
        while queue:
            (node_id, path) = queue.pop(0)
            
            # Get the node
            node = self.nodes[node_id]
            
            # Check all connections
            for conn_id in node.connections:
                if conn_id not in self.nodes:
                    continue
                    
                if conn_id == end_id:
                    # Found the end, construct the path
                    result = []
                    for p_id in path + [end_id]:
                        result.append({
                            "node_id": p_id,
                            "data_type": self.nodes[p_id].data_type,
                            "metadata": self.nodes[p_id].metadata
                        })
                    return result
                    
                if conn_id not in visited:
                    visited.add(conn_id)
                    queue.append((conn_id, path + [conn_id]))
        
        # No path found, try spatial proximity
        # Get nodes that are spatially close to both start and end
        start_node = self.nodes[start_id]
        end_node = self.nodes[end_id]
        
        # Query nodes near the start
        start_region = self.query_spatial_region(
            center=start_node.position,
            radius=20.0
        )
        
        # Query nodes near the end
        end_region = self.query_spatial_region(
            center=end_node.position,
            radius=20.0
        )
        
        # Find common nodes
        start_ids = set(r["node_id"] for r in start_region.get("results", []))
        end_ids = set(r["node_id"] for r in end_region.get("results", []))
        
        common_ids = start_ids.intersection(end_ids)
        
        if common_ids:
            # Use the first common node to create a path
            common_id = next(iter(common_ids))
            
            # Create the path
            result = [
                {
                    "node_id": start_id,
                    "data_type": self.nodes[start_id].data_type,
                    "metadata": self.nodes[start_id].metadata
                },
                {
                    "node_id": common_id,
                    "data_type": self.nodes[common_id].data_type,
                    "metadata": self.nodes[common_id].metadata
                },
                {
                    "node_id": end_id,
                    "data_type": self.nodes[end_id].data_type,
                    "metadata": self.nodes[end_id].metadata
                }
            ]
            return result
        
        # No path found
        return []
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export the current state of the mnemonic data architecture.
        
        Returns:
            The complete state
        """
        return {
            "metadata": self.metadata,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "clusters": {cluster_id: cluster.to_dict() for cluster_id, cluster in self.clusters.items()},
            "paths": {path_id: path.to_dict() for path_id, path in self.paths.items()}
        }
    
    def import_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import a state into the mnemonic data architecture.
        
        Args:
            state: The state to import
            
        Returns:
            Import result
        """
        try:
            # Clear current state
            self.nodes = {}
            self.clusters = {}
            self.paths = {}
            self.spatial_index = {}
            
            # Import metadata
            self.metadata = state.get("metadata", {})
            
            # Import nodes
            for node_id, node_data in state.get("nodes", {}).items():
                self.nodes[node_id] = SpatialNode.from_dict(node_data)
                self._update_spatial_index_for_node(self.nodes[node_id])
            
            # Import clusters
            for cluster_id, cluster_data in state.get("clusters", {}).items():
                self.clusters[cluster_id] = Cluster.from_dict(cluster_data)
            
            # Import paths
            for path_id, path_data in state.get("paths", {}).items():
                self.paths[path_id] = Path.from_dict(path_data)
            
            return {
                "success": True,
                "node_count": len(self.nodes),
                "cluster_count": len(self.clusters),
                "path_count": len(self.paths)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Import failed: {str(e)}"
            }
