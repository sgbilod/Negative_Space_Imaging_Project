#!/usr/bin/env python
"""
Semantic Context System
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements semantic context analysis for understanding relationships
between negative spaces and their surroundings.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


class RelationType(Enum):
    """Types of semantic relationships."""
    CONTAINS = "contains"
    ADJACENT = "adjacent"
    OVERLAPS = "overlaps"
    CONNECTED = "connected"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"


@dataclass
class SemanticRelation:
    """Represents a semantic relationship between regions."""
    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float
    attributes: Dict[str, float]


@dataclass
class SemanticContext:
    """Semantic context for a negative space region."""
    region_id: str
    semantic_embedding: torch.Tensor
    relations: List[SemanticRelation]
    context_score: float
    description: str


class SemanticGraphNetwork(nn.Module):
    """Neural network for learning semantic relationships."""
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Node feature processing
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Multi-head attention for neighborhood aggregation
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )
        
        # Relationship classification
        self.relation_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, len(RelationType))
        )
        
        # Context embedding
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, feature_dim)
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process semantic graph.
        
        Args:
            node_features: Node feature tensor [num_nodes, feature_dim]
            adjacency: Adjacency matrix [num_nodes, num_nodes]
            edge_index: Edge index tensor [2, num_edges]
            
        Returns:
            relation_logits: Relationship classification logits
            context_embeddings: Context-aware node embeddings
            attention_weights: Attention weights for interpretability
        """
        # Encode node features
        node_hidden = self.node_encoder(node_features)
        
        # Create edge features
        edge_features = torch.cat([
            node_hidden[edge_index[0]],
            node_hidden[edge_index[1]]
        ], dim=-1)
        edge_hidden = self.edge_encoder(edge_features)
        
        # Apply attention over neighborhood
        attended_features, attention_weights = self.attention(
            node_hidden.unsqueeze(0),
            node_hidden.unsqueeze(0),
            node_hidden.unsqueeze(0),
            key_padding_mask=(adjacency == 0)
        )
        attended_features = attended_features.squeeze(0)
        
        # Classify relationships
        relation_logits = self.relation_classifier(edge_hidden)
        
        # Generate context embeddings
        context_embeddings = self.context_encoder(attended_features)
        
        return relation_logits, context_embeddings, attention_weights


class SemanticContextAnalyzer:
    """Analyzes semantic relationships in negative spaces."""
    
    def __init__(
        self,
        feature_dim: int = 256,
        device: Optional[torch.device] = None,
        language_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize semantic graph network
        self.graph_net = SemanticGraphNetwork(
            feature_dim=feature_dim
        ).to(self.device)
        
        # Initialize language model for descriptions
        self.tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.language_model = AutoModel.from_pretrained(
            language_model
        ).to(self.device)
        
        # Graph analysis
        self.graph = nx.Graph()
    
    def analyze_context(
        self,
        regions: Dict[str, np.ndarray],
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, SemanticContext]:
        """
        Analyze semantic context of negative space regions.
        
        Args:
            regions: Dictionary of region masks
            features: Dictionary of region features
            
        Returns:
            Dictionary mapping region IDs to semantic contexts
        """
        # Build adjacency graph
        self._build_graph(regions)
        
        # Prepare tensors
        region_ids = list(regions.keys())
        node_features = torch.stack(
            [features[rid] for rid in region_ids]
        ).to(self.device)
        
        adjacency = torch.tensor(
            nx.adjacency_matrix(self.graph).todense(),
            device=self.device
        )
        
        edge_index = torch.tensor(
            [[i, j] for i, j in self.graph.edges()],
            device=self.device
        ).t()
        
        # Process through graph network
        with torch.no_grad():
            relation_logits, context_embeds, attention = self.graph_net(
                node_features,
                adjacency,
                edge_index
            )
        
        # Extract relationships
        relations = self._extract_relations(
            region_ids,
            relation_logits,
            edge_index
        )
        
        # Generate descriptions
        descriptions = self._generate_descriptions(
            regions,
            relations,
            context_embeds
        )
        
        # Build semantic contexts
        contexts = {}
        for i, rid in enumerate(region_ids):
            region_relations = [
                r for r in relations
                if r.source_id == rid or r.target_id == rid
            ]
            
            # Compute context score based on relationship confidences
            context_score = np.mean([
                r.confidence for r in region_relations
            ]) if region_relations else 0.0
            
            contexts[rid] = SemanticContext(
                region_id=rid,
                semantic_embedding=context_embeds[i],
                relations=region_relations,
                context_score=float(context_score),
                description=descriptions[rid]
            )
        
        return contexts
    
    def _build_graph(self, regions: Dict[str, np.ndarray]):
        """Build graph from region spatial relationships."""
        self.graph.clear()
        region_ids = list(regions.keys())
        
        # Add nodes
        self.graph.add_nodes_from(region_ids)
        
        # Add edges based on spatial relationships
        for i, rid1 in enumerate(region_ids):
            for rid2 in region_ids[i+1:]:
                mask1 = regions[rid1]
                mask2 = regions[rid2]
                
                # Check for adjacency
                dilated1 = cv2.dilate(
                    mask1.astype(np.uint8),
                    np.ones((3, 3))
                )
                if np.any(dilated1 & mask2):
                    self.graph.add_edge(rid1, rid2)
    
    def _extract_relations(
        self,
        region_ids: List[str],
        relation_logits: torch.Tensor,
        edge_index: torch.Tensor
    ) -> List[SemanticRelation]:
        """Extract semantic relations from network outputs."""
        relations = []
        
        # Convert logits to probabilities
        relation_probs = F.softmax(relation_logits, dim=-1)
        
        for i, (src, dst) in enumerate(edge_index.t()):
            # Get highest probability relation
            rel_type_idx = relation_probs[i].argmax().item()
            confidence = relation_probs[i, rel_type_idx].item()
            
            # Create relation object
            relation = SemanticRelation(
                source_id=region_ids[src],
                target_id=region_ids[dst],
                relation_type=list(RelationType)[rel_type_idx],
                confidence=confidence,
                attributes={
                    "attention_weight": float(
                        self.graph[region_ids[src]][region_ids[dst]].get(
                            "weight",
                            0.0
                        )
                    )
                }
            )
            relations.append(relation)
        
        return relations
    
    def _generate_descriptions(
        self,
        regions: Dict[str, np.ndarray],
        relations: List[SemanticRelation],
        context_embeds: torch.Tensor
    ) -> Dict[str, str]:
        """Generate natural language descriptions of semantic contexts."""
        descriptions = {}
        
        for rid, mask in regions.items():
            # Get region's relations
            region_relations = [
                r for r in relations
                if r.source_id == rid or r.target_id == rid
            ]
            
            # Create description template
            desc_parts = [f"Region {rid}"]
            
            if region_relations:
                # Add relationship descriptions
                rel_descs = []
                for rel in region_relations:
                    other_id = (
                        rel.target_id if rel.source_id == rid
                        else rel.source_id
                    )
                    rel_descs.append(
                        f"{rel.relation_type.value} region {other_id}"
                    )
                
                if rel_descs:
                    desc_parts.append("is")
                    desc_parts.append(", ".join(rel_descs))
            
            # Join description parts
            descriptions[rid] = " ".join(desc_parts)
        
        return descriptions
