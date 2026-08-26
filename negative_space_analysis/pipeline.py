#!/usr/bin/env python
"""
Negative Space Analysis Pipeline
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements the main pipeline that integrates all subsystems
for comprehensive negative space analysis.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

from .multimodal_system import (
    MultiModalAnalyzer,
    ModalityType,
    MultiModalFeatures
)
from .semantic_system import (
    SemanticContextAnalyzer,
    SemanticContext,
    RelationType
)
from .temporal_system import (
    TemporalAnalyzer,
    TemporalChange,
    Trajectory
)
from .interactive_system import (
    InteractiveRefinement,
    UserFeedback,
    RefinementSuggestion
)


@dataclass
class AnalysisConfig:
    """Configuration for the analysis pipeline."""
    feature_dim: int = 256
    batch_size: int = 32
    device: Optional[torch.device] = None
    enable_temporal: bool = True
    enable_refinement: bool = True
    refinement_threshold: float = 0.7
    max_regions: int = 100
    logging_level: int = logging.INFO


@dataclass
class AnalysisResult:
    """Complete analysis results for negative spaces."""
    region_ids: List[str]
    region_masks: Dict[str, np.ndarray]
    multimodal_features: Dict[str, MultiModalFeatures]
    semantic_contexts: Dict[str, SemanticContext]
    temporal_changes: List[TemporalChange]
    refinement_suggestions: List[RefinementSuggestion]
    analysis_metadata: Dict[str, Any]


class NegativeSpaceAnalyzer:
    """Main pipeline for negative space analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = config.device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Configure logging
        logging.basicConfig(
            level=config.logging_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize subsystems
        self.multimodal = MultiModalAnalyzer(
            feature_dim=config.feature_dim,
            device=self.device
        )
        
        self.semantic = SemanticContextAnalyzer(
            feature_dim=config.feature_dim,
            device=self.device
        )
        
        if config.enable_temporal:
            self.temporal = TemporalAnalyzer(
                feature_dim=config.feature_dim,
                device=self.device
            )
        else:
            self.temporal = None
        
        if config.enable_refinement:
            self.refinement = InteractiveRefinement(
                feature_dim=config.feature_dim,
                device=self.device
            )
        else:
            self.refinement = None
        
        # State
        self.frame_idx = 0
        self.analysis_cache = {}
    
    def analyze_frame(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> AnalysisResult:
        """
        Analyze negative spaces in a new frame.
        
        Args:
            image: RGB image array
            depth: Optional depth map
            audio: Optional audio features
            text: Optional text description
            metadata: Optional metadata dictionary
            
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        self.logger.info(f"Starting analysis for frame {self.frame_idx}")
        
        try:
            # 1. Multi-modal Analysis
            self.logger.debug("Running multi-modal analysis")
            multimodal_result = self._run_multimodal_analysis(
                image, depth, audio, text
            )
            
            region_ids = multimodal_result.region_ids
            region_masks = multimodal_result.region_masks
            multimodal_features = multimodal_result.features
            
            # Limit number of regions
            if len(region_ids) > self.config.max_regions:
                self.logger.warning(
                    f"Limiting analysis to {self.config.max_regions} regions"
                )
                region_ids = region_ids[:self.config.max_regions]
                region_masks = {
                    rid: region_masks[rid] for rid in region_ids
                }
                multimodal_features = {
                    rid: multimodal_features[rid] for rid in region_ids
                }
            
            # 2. Semantic Analysis
            self.logger.debug("Running semantic analysis")
            semantic_contexts = self.semantic.analyze_context(
                region_masks,
                {rid: feat.combined for rid, feat in multimodal_features.items()}
            )
            
            # 3. Temporal Analysis
            temporal_changes = []
            if self.temporal is not None:
                self.logger.debug("Running temporal analysis")
                temporal_changes = self.temporal.update(
                    region_masks,
                    {rid: feat.combined for rid, feat in multimodal_features.items()}
                )
            
            # 4. Interactive Refinement
            refinement_suggestions = []
            if self.refinement is not None:
                self.logger.debug("Generating refinement suggestions")
                refinement_suggestions = self.refinement.get_suggestions(
                    region_masks,
                    {rid: feat.combined for rid, feat in multimodal_features.items()}
                )
                
                # Filter low confidence suggestions
                refinement_suggestions = [
                    s for s in refinement_suggestions
                    if s.confidence >= self.config.refinement_threshold
                ]
            
            # Prepare metadata
            analysis_metadata = {
                "frame_idx": self.frame_idx,
                "timestamp": time.time(),
                "processing_time": time.time() - start_time,
                "device": str(self.device),
                "num_regions": len(region_ids),
                "input_modalities": self._get_active_modalities(
                    image, depth, audio, text
                )
            }
            if metadata:
                analysis_metadata.update(metadata)
            
            # Create result
            result = AnalysisResult(
                region_ids=region_ids,
                region_masks=region_masks,
                multimodal_features=multimodal_features,
                semantic_contexts=semantic_contexts,
                temporal_changes=temporal_changes,
                refinement_suggestions=refinement_suggestions,
                analysis_metadata=analysis_metadata
            )
            
            # Cache result
            self.analysis_cache[self.frame_idx] = result
            
            self.logger.info(
                f"Completed analysis for frame {self.frame_idx} "
                f"with {len(region_ids)} regions"
            )
            
            # Increment frame counter
            self.frame_idx += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in analysis pipeline: {str(e)}")
            raise
    
    def apply_feedback(
        self,
        feedback: UserFeedback
    ) -> Optional[AnalysisResult]:
        """
        Apply user feedback and reanalyze.
        
        Args:
            feedback: User feedback to incorporate
            
        Returns:
            Updated analysis results if successful
        """
        if self.refinement is None:
            self.logger.warning("Refinement system is disabled")
            return None
            
        try:
            # Get latest result
            if not self.analysis_cache:
                self.logger.error("No previous analysis to refine")
                return None
                
            latest_result = self.analysis_cache[self.frame_idx - 1]
            
            # Apply feedback
            updated_masks = self.refinement.add_feedback(
                feedback,
                latest_result.region_masks,
                {
                    rid: feat.combined
                    for rid, feat in latest_result.multimodal_features.items()
                }
            )
            
            # Reanalyze with updated masks
            return self._reanalyze_with_masks(
                updated_masks,
                latest_result.analysis_metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error applying feedback: {str(e)}")
            return None
    
    def get_region_trajectory(
        self,
        region_id: str
    ) -> Optional[Trajectory]:
        """Get temporal trajectory for a region."""
        if self.temporal is None:
            return None
        return self.temporal.get_trajectory(region_id)
    
    def _run_multimodal_analysis(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        audio: Optional[np.ndarray],
        text: Optional[str]
    ) -> Any:
        """Run multi-modal analysis stage."""
        # Prepare inputs
        inputs = {ModalityType.VISUAL: image}
        
        if depth is not None:
            inputs[ModalityType.DEPTH] = depth
        if audio is not None:
            inputs[ModalityType.AUDIO] = audio
        if text is not None:
            inputs[ModalityType.TEXT] = text
            
        # Run analysis
        return self.multimodal.analyze(inputs)
    
    def _reanalyze_with_masks(
        self,
        masks: Dict[str, np.ndarray],
        metadata: Dict[str, Any]
    ) -> AnalysisResult:
        """Reanalyze with updated masks."""
        # Extract features
        features = self.multimodal.extract_features(masks)
        
        # Update semantic analysis
        semantic_contexts = self.semantic.analyze_context(
            masks,
            {rid: feat.combined for rid, feat in features.items()}
        )
        
        # Update temporal if enabled
        temporal_changes = []
        if self.temporal is not None:
            temporal_changes = self.temporal.update(
                masks,
                {rid: feat.combined for rid, feat in features.items()}
            )
        
        # Get refinement suggestions
        refinement_suggestions = []
        if self.refinement is not None:
            refinement_suggestions = self.refinement.get_suggestions(
                masks,
                {rid: feat.combined for rid, feat in features.items()}
            )
        
        # Update metadata
        metadata.update({
            "reanalysis_time": time.time(),
            "num_regions": len(masks)
        })
        
        return AnalysisResult(
            region_ids=list(masks.keys()),
            region_masks=masks,
            multimodal_features=features,
            semantic_contexts=semantic_contexts,
            temporal_changes=temporal_changes,
            refinement_suggestions=refinement_suggestions,
            analysis_metadata=metadata
        )
    
    @staticmethod
    def _get_active_modalities(
        image: np.ndarray,
        depth: Optional[np.ndarray],
        audio: Optional[np.ndarray],
        text: Optional[str]
    ) -> List[str]:
        """Get list of active input modalities."""
        modalities = ["visual"]
        if depth is not None:
            modalities.append("depth")
        if audio is not None:
            modalities.append("audio")
        if text is not None:
            modalities.append("text")
        return modalities
