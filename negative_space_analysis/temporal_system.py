#!/usr/bin/env python
"""
Temporal Analysis System
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements temporal analysis for tracking and understanding
changes in negative spaces over time sequences.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class ChangeType(Enum):
    """Types of temporal changes in negative spaces."""
    APPEAR = "appear"
    DISAPPEAR = "disappear"
    MERGE = "merge"
    SPLIT = "split"
    DEFORM = "deform"
    STABLE = "stable"
    MOVE = "move"


@dataclass
class TemporalChange:
    """Represents a temporal change between frames."""
    frame_idx: int
    region_ids: List[str]
    change_type: ChangeType
    confidence: float
    metrics: Dict[str, float]


@dataclass
class TrajectoryPoint:
    """A point in a region's temporal trajectory."""
    frame_idx: int
    position: np.ndarray
    area: float
    features: torch.Tensor


class Trajectory:
    """Represents the temporal trajectory of a negative space region."""
    
    def __init__(self, region_id: str):
        self.region_id = region_id
        self.points: List[TrajectoryPoint] = []
        self.active = True
        self.last_update = -1
        
    def add_point(self, point: TrajectoryPoint):
        """Add a new point to the trajectory."""
        self.points.append(point)
        self.last_update = point.frame_idx
        
    def get_duration(self) -> int:
        """Get trajectory duration in frames."""
        if not self.points:
            return 0
        return self.points[-1].frame_idx - self.points[0].frame_idx + 1
    
    def get_velocity(self) -> Optional[np.ndarray]:
        """Calculate current velocity vector."""
        if len(self.points) < 2:
            return None
        
        p1 = self.points[-2]
        p2 = self.points[-1]
        dt = p2.frame_idx - p1.frame_idx
        
        if dt == 0:
            return None
            
        return (p2.position - p1.position) / dt


class TemporalEncoder(nn.Module):
    """Encodes temporal sequences of region features."""
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Temporal encoding
        self.position_encoding = nn.Parameter(
            torch.randn(1, 1000, feature_dim)
        )
        
        # Feature processing
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Temporal processing
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(2 * hidden_dim, feature_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Process temporal sequence.
        
        Args:
            features: Feature sequences [batch, max_len, feature_dim]
            lengths: Sequence lengths [batch]
            
        Returns:
            Encoded temporal features
        """
        batch_size, max_len = features.shape[:2]
        
        # Add temporal position encoding
        pos_enc = self.position_encoding[:, :max_len]
        features = features + pos_enc
        
        # Encode features
        hidden = self.feature_encoder(features)
        
        # Pack for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Process temporal sequence
        output, _ = self.lstm(packed)
        
        # Unpack output
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output,
            batch_first=True
        )
        
        # Project output
        return self.output_proj(output)


class TemporalAnalyzer:
    """Analyzes temporal patterns in negative spaces."""
    
    def __init__(
        self,
        feature_dim: int = 256,
        max_track_age: int = 10,
        match_threshold: float = 0.7,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize temporal encoder
        self.encoder = TemporalEncoder(
            feature_dim=feature_dim
        ).to(self.device)
        
        # Tracking parameters
        self.max_track_age = max_track_age
        self.match_threshold = match_threshold
        
        # State
        self.frame_idx = 0
        self.trajectories: Dict[str, Trajectory] = {}
        self.changes: List[TemporalChange] = []
    
    def update(
        self,
        regions: Dict[str, np.ndarray],
        features: Dict[str, torch.Tensor]
    ) -> List[TemporalChange]:
        """
        Update temporal analysis with new frame.
        
        Args:
            regions: Dictionary of region masks
            features: Dictionary of region features
            
        Returns:
            List of detected temporal changes
        """
        # Convert regions to centroids
        centroids = {
            rid: self._compute_centroid(mask)
            for rid, mask in regions.items()
        }
        
        # Match current regions to trajectories
        matches = self._match_regions(
            regions,
            features,
            centroids
        )
        
        # Update trajectories
        changes = self._update_trajectories(
            matches,
            regions,
            features,
            centroids
        )
        
        # Update frame index
        self.frame_idx += 1
        
        return changes
    
    def get_trajectory(self, region_id: str) -> Optional[Trajectory]:
        """Get trajectory for a region if it exists."""
        return self.trajectories.get(region_id)
    
    def _compute_centroid(self, mask: np.ndarray) -> np.ndarray:
        """Compute centroid of a region mask."""
        y, x = np.nonzero(mask)
        if len(x) == 0:
            return np.zeros(2)
        return np.array([np.mean(x), np.mean(y)])
    
    def _match_regions(
        self,
        regions: Dict[str, np.ndarray],
        features: Dict[str, torch.Tensor],
        centroids: Dict[str, np.ndarray]
    ) -> List[Tuple[str, str]]:
        """Match current regions to existing trajectories."""
        if not self.trajectories:
            return []
            
        # Build cost matrix
        active_trajectories = [
            tid for tid, traj in self.trajectories.items()
            if traj.active
        ]
        
        cost_matrix = np.zeros(
            (len(active_trajectories), len(regions))
        )
        
        for i, tid in enumerate(active_trajectories):
            traj = self.trajectories[tid]
            last_point = traj.points[-1]
            
            for j, (rid, feat) in enumerate(features.items()):
                # Feature similarity
                feat_sim = F.cosine_similarity(
                    last_point.features,
                    feat,
                    dim=0
                ).item()
                
                # Position distance
                pos_dist = np.linalg.norm(
                    last_point.position - centroids[rid]
                )
                
                # Combined cost
                cost = -(feat_sim - 0.1 * pos_dist)
                cost_matrix[i, j] = cost
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter matches by threshold
        matches = []
        for i, j in zip(row_ind, col_ind):
            if -cost_matrix[i, j] >= self.match_threshold:
                matches.append(
                    (active_trajectories[i], list(regions.keys())[j])
                )
        
        return matches
    
    def _update_trajectories(
        self,
        matches: List[Tuple[str, str]],
        regions: Dict[str, np.ndarray],
        features: Dict[str, torch.Tensor],
        centroids: Dict[str, np.ndarray]
    ) -> List[TemporalChange]:
        """Update trajectories and detect changes."""
        changes = []
        matched_current = set()
        matched_tracks = set()
        
        # Process matches
        for tid, rid in matches:
            matched_current.add(rid)
            matched_tracks.add(tid)
            
            traj = self.trajectories[tid]
            
            # Add new point
            traj.add_point(TrajectoryPoint(
                frame_idx=self.frame_idx,
                position=centroids[rid],
                area=float(regions[rid].sum()),
                features=features[rid]
            ))
            
            # Detect deformation
            if len(traj.points) >= 2:
                area_change = abs(
                    traj.points[-1].area - traj.points[-2].area
                ) / traj.points[-2].area
                
                if area_change > 0.2:
                    changes.append(TemporalChange(
                        frame_idx=self.frame_idx,
                        region_ids=[rid],
                        change_type=ChangeType.DEFORM,
                        confidence=min(1.0, area_change),
                        metrics={"area_change": area_change}
                    ))
        
        # Handle unmatched current regions (appearances)
        for rid in regions.keys():
            if rid not in matched_current:
                # Create new trajectory
                traj = Trajectory(rid)
                traj.add_point(TrajectoryPoint(
                    frame_idx=self.frame_idx,
                    position=centroids[rid],
                    area=float(regions[rid].sum()),
                    features=features[rid]
                ))
                
                self.trajectories[rid] = traj
                
                changes.append(TemporalChange(
                    frame_idx=self.frame_idx,
                    region_ids=[rid],
                    change_type=ChangeType.APPEAR,
                    confidence=1.0,
                    metrics={}
                ))
        
        # Handle unmatched trajectories (disappearances)
        for tid, traj in self.trajectories.items():
            if not traj.active:
                continue
                
            if tid not in matched_tracks:
                if self.frame_idx - traj.last_update > self.max_track_age:
                    traj.active = False
                    changes.append(TemporalChange(
                        frame_idx=self.frame_idx,
                        region_ids=[tid],
                        change_type=ChangeType.DISAPPEAR,
                        confidence=1.0,
                        metrics={"age": self.frame_idx - traj.last_update}
                    ))
        
        return changes


class TemporalPredictor:
    """Predicts future states of negative spaces."""
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Feature processing
        self.encoder = TemporalEncoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)
        
        # Prediction heads
        self.position_pred = nn.Linear(feature_dim, 2).to(self.device)
        self.area_pred = nn.Linear(feature_dim, 1).to(self.device)
        self.feature_pred = nn.Linear(
            feature_dim,
            feature_dim
        ).to(self.device)
    
    def predict(
        self,
        trajectory: Trajectory,
        num_steps: int = 5
    ) -> List[TrajectoryPoint]:
        """
        Predict future trajectory points.
        
        Args:
            trajectory: Input trajectory
            num_steps: Number of steps to predict
            
        Returns:
            List of predicted trajectory points
        """
        if len(trajectory.points) < 2:
            return []
            
        # Prepare input sequence
        features = torch.stack([
            p.features for p in trajectory.points
        ]).unsqueeze(0)
        
        lengths = torch.tensor([len(trajectory.points)])
        
        # Encode sequence
        encoded = self.encoder(features, lengths)
        last_hidden = encoded[0, -1]
        
        # Generate predictions
        predictions = []
        current = last_hidden
        
        for i in range(num_steps):
            # Predict next state
            pos_delta = self.position_pred(current)
            area = torch.exp(self.area_pred(current))
            next_features = self.feature_pred(current)
            
            # Create trajectory point
            last_point = trajectory.points[-1]
            next_position = (
                last_point.position +
                pos_delta.detach().cpu().numpy()
            )
            
            point = TrajectoryPoint(
                frame_idx=last_point.frame_idx + i + 1,
                position=next_position,
                area=float(area.item()),
                features=next_features
            )
            
            predictions.append(point)
            current = next_features
        
        return predictions
