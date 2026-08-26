#!/usr/bin/env python
"""
Interactive Refinement System
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements interactive refinement capabilities for negative space
analysis, allowing user feedback to improve results.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
import torch.nn.functional as F


class FeedbackType(Enum):
    """Types of user feedback."""
    REGION_MERGE = "merge"
    REGION_SPLIT = "split"
    BOUNDARY_ADJUST = "boundary"
    REGION_ADD = "add"
    REGION_REMOVE = "remove"
    IMPORTANCE_ADJUST = "importance"


@dataclass
class UserFeedback:
    """Represents user feedback for refinement."""
    feedback_type: FeedbackType
    region_ids: List[str]
    parameters: Dict[str, Any]
    timestamp: float
    confidence: float


@dataclass
class RefinementSuggestion:
    """Represents a suggested refinement."""
    suggestion_type: FeedbackType
    region_ids: List[str]
    description: str
    confidence: float
    parameters: Dict[str, Any]


class FeedbackEncoder(nn.Module):
    """Encodes user feedback into learnable representations."""
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        num_feedback_types: int = len(FeedbackType)
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Feedback type embedding
        self.type_embedding = nn.Embedding(
            num_feedback_types,
            hidden_dim
        )
        
        # Parameter encoding
        self.parameter_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Confidence encoding
        self.confidence_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Combined feedback representation
        self.output_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(
        self,
        feedback_types: torch.Tensor,
        parameters: torch.Tensor,
        confidences: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode feedback information.
        
        Args:
            feedback_types: Feedback type indices [batch]
            parameters: Parameter tensors [batch, feature_dim]
            confidences: Confidence values [batch, 1]
            
        Returns:
            Encoded feedback representations
        """
        # Encode components
        type_embed = self.type_embedding(feedback_types)
        param_embed = self.parameter_encoder(parameters)
        conf_embed = self.confidence_encoder(confidences)
        
        # Combine representations
        combined = torch.cat(
            [type_embed, param_embed, conf_embed],
            dim=-1
        )
        
        return self.output_proj(combined)


class RefinementNetwork(nn.Module):
    """Neural network for learning refinement patterns."""
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Feedback encoding
        self.feedback_encoder = FeedbackEncoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )
        
        # Region feature processing
        self.region_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Cross-attention for feedback-region interaction
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )
        
        # Refinement prediction
        self.refinement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(FeedbackType))
        )
        
        # Parameter prediction
        self.parameter_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, feature_dim)
        )
    
    def forward(
        self,
        region_features: torch.Tensor,
        feedback_history: List[UserFeedback]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process regions and feedback for refinement.
        
        Args:
            region_features: Region feature tensors [num_regions, feature_dim]
            feedback_history: List of previous user feedback
            
        Returns:
            Refinement logits and parameter predictions
        """
        if not feedback_history:
            # No feedback to process
            return None, None
        
        # Prepare feedback tensors
        feedback_types = torch.tensor([
            list(FeedbackType).index(f.feedback_type)
            for f in feedback_history
        ], device=region_features.device)
        
        feedback_params = torch.stack([
            torch.zeros(self.feature_dim)  # Simplified parameter encoding
            for f in feedback_history
        ]).to(region_features.device)
        
        feedback_conf = torch.tensor([
            [f.confidence] for f in feedback_history
        ], device=region_features.device)
        
        # Encode feedback
        feedback_embed = self.feedback_encoder(
            feedback_types,
            feedback_params,
            feedback_conf
        )
        
        # Encode regions
        region_embed = self.region_encoder(region_features)
        
        # Cross-attention between feedback and regions
        attended_regions, _ = self.attention(
            region_embed.unsqueeze(0),
            feedback_embed.unsqueeze(0),
            feedback_embed.unsqueeze(0)
        )
        attended_regions = attended_regions.squeeze(0)
        
        # Generate refinement predictions
        refinement_logits = self.refinement_head(attended_regions)
        parameters = self.parameter_head(attended_regions)
        
        return refinement_logits, parameters


class InteractiveRefinement:
    """Manages interactive refinement of negative space analysis."""
    
    def __init__(
        self,
        feature_dim: int = 256,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize refinement network
        self.network = RefinementNetwork(
            feature_dim=feature_dim
        ).to(self.device)
        
        # State
        self.feedback_history: List[UserFeedback] = []
        self.region_states: Dict[str, Dict] = {}
        self.refinement_cache: Dict[str, Set[FeedbackType]] = {}
    
    def add_feedback(
        self,
        feedback: UserFeedback,
        regions: Dict[str, np.ndarray],
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Process new user feedback.
        
        Args:
            feedback: User feedback to process
            regions: Current region masks
            features: Region features
            
        Returns:
            Updated region masks
        """
        # Add to history
        self.feedback_history.append(feedback)
        
        # Apply feedback
        updated_regions = self._apply_feedback(
            feedback,
            regions.copy(),
            features
        )
        
        # Update region states
        self._update_states(feedback, updated_regions)
        
        return updated_regions
    
    def get_suggestions(
        self,
        regions: Dict[str, np.ndarray],
        features: Dict[str, torch.Tensor]
    ) -> List[RefinementSuggestion]:
        """
        Generate refinement suggestions.
        
        Args:
            regions: Current region masks
            features: Region features
            
        Returns:
            List of refinement suggestions
        """
        if not self.feedback_history:
            return []
        
        # Prepare region features
        region_features = torch.stack(
            list(features.values())
        ).to(self.device)
        
        # Generate refinements
        with torch.no_grad():
            refinement_logits, parameters = self.network(
                region_features,
                self.feedback_history
            )
            
        if refinement_logits is None:
            return []
            
        # Convert to suggestions
        suggestions = []
        region_ids = list(regions.keys())
        
        for i, region_id in enumerate(region_ids):
            # Skip if already refined
            if region_id in self.refinement_cache:
                continue
                
            # Get highest confidence refinement
            ref_type_idx = refinement_logits[i].argmax().item()
            confidence = F.softmax(
                refinement_logits[i],
                dim=0
            )[ref_type_idx].item()
            
            if confidence > 0.7:  # Confidence threshold
                ref_type = list(FeedbackType)[ref_type_idx]
                
                suggestion = RefinementSuggestion(
                    suggestion_type=ref_type,
                    region_ids=[region_id],
                    description=self._get_suggestion_description(
                        ref_type,
                        region_id
                    ),
                    confidence=confidence,
                    parameters={
                        "features": parameters[i].cpu()
                    }
                )
                
                suggestions.append(suggestion)
        
        return suggestions
    
    def _apply_feedback(
        self,
        feedback: UserFeedback,
        regions: Dict[str, np.ndarray],
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """Apply feedback to modify regions."""
        if feedback.feedback_type == FeedbackType.REGION_MERGE:
            # Merge regions
            if len(feedback.region_ids) >= 2:
                primary_id = feedback.region_ids[0]
                merged_mask = regions[primary_id].copy()
                
                for rid in feedback.region_ids[1:]:
                    merged_mask |= regions[rid]
                    regions.pop(rid)
                
                regions[primary_id] = merged_mask
        
        elif feedback.feedback_type == FeedbackType.REGION_SPLIT:
            # Split region (simplified)
            if len(feedback.region_ids) == 1:
                rid = feedback.region_ids[0]
                mask = regions[rid]
                
                # Simple vertical split
                h, w = mask.shape
                left_mask = mask.copy()
                right_mask = mask.copy()
                left_mask[:, w//2:] = 0
                right_mask[:, :w//2] = 0
                
                regions[f"{rid}_1"] = left_mask
                regions[f"{rid}_2"] = right_mask
                regions.pop(rid)
        
        elif feedback.feedback_type == FeedbackType.BOUNDARY_ADJUST:
            # Adjust region boundary
            if len(feedback.region_ids) == 1:
                rid = feedback.region_ids[0]
                dilation = feedback.parameters.get("dilation", 1)
                
                if dilation > 0:
                    regions[rid] = self._dilate_mask(
                        regions[rid],
                        dilation
                    )
                else:
                    regions[rid] = self._erode_mask(
                        regions[rid],
                        abs(dilation)
                    )
        
        elif feedback.feedback_type == FeedbackType.REGION_REMOVE:
            # Remove regions
            for rid in feedback.region_ids:
                regions.pop(rid, None)
        
        return regions
    
    def _update_states(
        self,
        feedback: UserFeedback,
        regions: Dict[str, np.ndarray]
    ):
        """Update region states based on feedback."""
        # Track refined regions
        for rid in feedback.region_ids:
            if rid not in self.refinement_cache:
                self.refinement_cache[rid] = set()
            self.refinement_cache[rid].add(feedback.feedback_type)
        
        # Update region states
        for rid, mask in regions.items():
            if rid not in self.region_states:
                self.region_states[rid] = {
                    "feedback_count": 0,
                    "last_feedback": None,
                    "stability_score": 1.0
                }
            
            if rid in feedback.region_ids:
                state = self.region_states[rid]
                state["feedback_count"] += 1
                state["last_feedback"] = feedback.feedback_type
                state["stability_score"] *= 0.9  # Reduce stability
    
    def _get_suggestion_description(
        self,
        ref_type: FeedbackType,
        region_id: str
    ) -> str:
        """Generate description for refinement suggestion."""
        if ref_type == FeedbackType.REGION_MERGE:
            return f"Consider merging region {region_id} with adjacent regions"
        elif ref_type == FeedbackType.REGION_SPLIT:
            return f"Region {region_id} might benefit from splitting"
        elif ref_type == FeedbackType.BOUNDARY_ADJUST:
            return f"Consider adjusting boundaries of region {region_id}"
        elif ref_type == FeedbackType.REGION_REMOVE:
            return f"Region {region_id} might be noise and could be removed"
        else:
            return f"Suggested refinement for region {region_id}"
    
    @staticmethod
    def _dilate_mask(mask: np.ndarray, amount: int) -> np.ndarray:
        """Dilate binary mask."""
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(
            mask.astype(np.uint8),
            kernel,
            iterations=amount
        ).astype(bool)
    
    @staticmethod
    def _erode_mask(mask: np.ndarray, amount: int) -> np.ndarray:
        """Erode binary mask."""
        kernel = np.ones((3, 3), np.uint8)
        return cv2.erode(
            mask.astype(np.uint8),
            kernel,
            iterations=amount
        ).astype(bool)
