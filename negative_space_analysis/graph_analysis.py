#!/usr/bin/env python
"""
Graph-based Pattern Analysis Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements sophisticated graph-based analysis of negative space
patterns using Graph Neural Networks and topological features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay


@dataclass
class GraphFeatures:
    """Features extracted from graph analysis."""
    centrality_measures: Dict[str, np.ndarray]
    clustering_coefficients: np.ndarray
    spectral_features: np.ndarray
    topological_features: Dict[str, float]
    graph_embeddings: torch.Tensor
    community_labels: np.ndarray
    pattern_scores: Dict[str, float]


class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer with attention mechanism."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        # Multi-head attention
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)
        
        # Output transformation
        self.output_transform = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Dropout(dropout),
            nn.LayerNorm(out_channels)
        )
        
        # Edge feature network
        self.edge_network = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, num_heads)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with graph attention.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, num_features]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        
        # Compute query, key, value
        q = self.query(x).view(-1, self.num_heads, self.head_dim)
        k = self.key(x).view(-1, self.num_heads, self.head_dim)
        v = self.value(x).view(-1, self.num_heads, self.head_dim)
        
        # Get source and target nodes
        src, dst = edge_index
        
        # Compute edge features if not provided
        if edge_attr is None:
            edge_features = torch.cat([x[src], x[dst]], dim=-1)
            edge_weights = self.edge_network(edge_features)
            edge_weights = edge_weights.view(-1, self.num_heads, 1)
        else:
            edge_weights = edge_attr.unsqueeze(-1)
        
        # Compute attention scores
        attn = (q[dst] * k[src]).sum(dim=-1, keepdim=True)
        attn = attn * edge_weights / np.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=0)
        
        # Apply attention to values
        out = attn * v[src]
        
        # Aggregate messages
        out = scatter_add(
            out,
            dst,
            dim=0,
            dim_size=num_nodes
        )
        
        # Reshape and transform output
        out = out.view(-1, self.out_channels)
        out = self.output_transform(out)
        
        return out


class GraphNN(nn.Module):
    """Graph Neural Network for pattern analysis."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            GraphConvLayer(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # Global pooling and output layers
        self.pool = GlobalAttentionPooling(hidden_channels)
        self.pattern_head = nn.Linear(hidden_channels, 1)
        self.feature_head = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge features [num_edges, num_features]
            batch: Optional batch assignment for multiple graphs
            
        Returns:
            pattern_scores: Pattern detection scores
            node_embeddings: Per-node embeddings
            graph_embedding: Global graph embedding
        """
        # Apply graph convolution layers
        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr)
            x = F.relu(x)
        
        # Get node embeddings
        node_embeddings = self.feature_head(x)
        
        # Pool to get graph embedding
        graph_embedding = self.pool(node_embeddings, batch)
        
        # Get pattern scores
        pattern_scores = self.pattern_head(graph_embedding).sigmoid()
        
        return pattern_scores, node_embeddings, graph_embedding


class GlobalAttentionPooling(nn.Module):
    """Global attention pooling layer."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, 1, bias=False)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply attention pooling.
        
        Args:
            x: Node features [num_nodes, channels]
            batch: Optional batch assignment for multiple graphs
            
        Returns:
            Pooled features [batch_size, channels]
        """
        attention_weights = self.attention(x)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        if batch is not None:
            out = scatter_add(
                attention_weights * x,
                batch,
                dim=0
            )
        else:
            out = (attention_weights * x).sum(dim=0, keepdim=True)
        
        return out


class NegativeSpaceGraphAnalyzer:
    """Analyzes negative space patterns using graph-based methods."""
    
    def __init__(
        self,
        feature_dim: int = 128,
        min_region_size: int = 50,
        num_gnn_layers: int = 3,
        device: Optional[torch.device] = None
    ):
        self.feature_dim = feature_dim
        self.min_region_size = min_region_size
        self.device = device or torch.device('cpu')
        
        # Initialize GNN
        self.gnn = GraphNN(
            in_channels=feature_dim,
            hidden_channels=feature_dim,
            num_layers=num_gnn_layers
        ).to(self.device)
    
    def analyze_pattern(
        self,
        regions: Dict[str, np.ndarray],
        features: Dict[str, np.ndarray]
    ) -> GraphFeatures:
        """
        Analyze patterns in negative spaces using graph representation.
        
        Args:
            regions: Dictionary of region masks
            features: Dictionary of region features
            
        Returns:
            GraphFeatures object
        """
        # Build graph from regions
        graph = self._build_region_graph(regions)
        
        # Convert to PyTorch geometric format
        node_features, edge_index, edge_attr = self._prepare_graph_data(
            graph,
            features
        )
        
        # Apply GNN
        with torch.no_grad():
            pattern_scores, node_embeddings, graph_embedding = self.gnn(
                node_features.to(self.device),
                edge_index.to(self.device),
                edge_attr.to(self.device) if edge_attr is not None else None
            )
        
        # Compute graph features
        centrality = self._compute_centrality(graph)
        clustering = self._compute_clustering(graph)
        spectral = self._compute_spectral_features(graph)
        topology = self._compute_topological_features(graph)
        communities = self._detect_communities(graph)
        
        # Create pattern scores dictionary
        pattern_dict = {
            f"pattern_{i}": float(score)
            for i, score in enumerate(pattern_scores.cpu().numpy())
        }
        
        return GraphFeatures(
            centrality_measures=centrality,
            clustering_coefficients=clustering,
            spectral_features=spectral,
            topological_features=topology,
            graph_embeddings=graph_embedding.cpu(),
            community_labels=communities,
            pattern_scores=pattern_dict
        )
    
    def _build_region_graph(
        self,
        regions: Dict[str, np.ndarray]
    ) -> nx.Graph:
        """Build graph from region masks."""
        graph = nx.Graph()
        
        # Add nodes
        for region_id, mask in regions.items():
            centroid = np.array(ndimage.center_of_mass(mask))
            graph.add_node(
                region_id,
                pos=centroid,
                area=float(np.sum(mask))
            )
        
        # Add edges based on Delaunay triangulation
        points = np.array([
            graph.nodes[node]["pos"]
            for node in graph.nodes
        ])
        
        if len(points) >= 4:  # Minimum points for Delaunay
            tri = Delaunay(points)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        node1 = list(graph.nodes)[simplex[i]]
                        node2 = list(graph.nodes)[simplex[j]]
                        
                        # Add edge with distance weight
                        dist = np.linalg.norm(
                            graph.nodes[node1]["pos"] -
                            graph.nodes[node2]["pos"]
                        )
                        graph.add_edge(node1, node2, weight=float(dist))
        
        return graph
    
    def _prepare_graph_data(
        self,
        graph: nx.Graph,
        features: Dict[str, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Prepare graph data for PyTorch Geometric."""
        # Node features
        node_features = []
        for node in graph.nodes:
            if node in features:
                node_features.append(features[node])
            else:
                node_features.append(np.zeros(self.feature_dim))
        
        node_features = torch.FloatTensor(node_features)
        
        # Edge indices
        edge_index = torch.tensor([
            [graph.nodes.index(u), graph.nodes.index(v)]
            for u, v in graph.edges
        ]).t().contiguous()
        
        # Edge attributes
        if nx.get_edge_attributes(graph, "weight"):
            edge_attr = torch.tensor([
                graph[u][v]["weight"]
                for u, v in graph.edges
            ]).float().unsqueeze(-1)
        else:
            edge_attr = None
        
        return node_features, edge_index, edge_attr
    
    def _compute_centrality(
        self,
        graph: nx.Graph
    ) -> Dict[str, np.ndarray]:
        """Compute various centrality measures."""
        return {
            "degree": np.array(list(dict(nx.degree_centrality(graph)).values())),
            "betweenness": np.array(
                list(dict(nx.betweenness_centrality(graph)).values())
            ),
            "closeness": np.array(
                list(dict(nx.closeness_centrality(graph)).values())
            )
        }
    
    def _compute_clustering(self, graph: nx.Graph) -> np.ndarray:
        """Compute clustering coefficients."""
        return np.array(list(dict(nx.clustering(graph)).values()))
    
    def _compute_spectral_features(self, graph: nx.Graph) -> np.ndarray:
        """Compute spectral features of the graph."""
        # Get adjacency matrix
        adj = nx.adjacency_matrix(graph).todense()
        
        # Compute Laplacian
        degree = np.diag(np.sum(adj, axis=1))
        laplacian = degree - adj
        
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(laplacian)
        
        return eigenvalues
    
    def _compute_topological_features(
        self,
        graph: nx.Graph
    ) -> Dict[str, float]:
        """Compute topological features."""
        return {
            "num_nodes": float(graph.number_of_nodes()),
            "num_edges": float(graph.number_of_edges()),
            "avg_degree": float(
                sum(dict(graph.degree()).values()) / graph.number_of_nodes()
            ),
            "density": float(nx.density(graph)),
            "avg_clustering": float(nx.average_clustering(graph)),
            "assortativity": float(nx.degree_assortativity_coefficient(graph))
        }
    
    def _detect_communities(self, graph: nx.Graph) -> np.ndarray:
        """Detect communities using Louvain method."""
        communities = nx.community.louvain_communities(graph)
        
        # Convert to label array
        labels = np.zeros(graph.number_of_nodes(), dtype=int)
        for i, community in enumerate(communities):
            for node in community:
                labels[list(graph.nodes).index(node)] = i
        
        return labels
