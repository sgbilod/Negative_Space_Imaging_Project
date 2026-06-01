#!/usr/bin/env python
"""
Negative Space Analysis Example
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This script demonstrates how to use the negative space analysis pipeline
with various input modalities and features.
"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Optional

from negative_space_analysis.pipeline import (
    NegativeSpaceAnalyzer,
    AnalysisConfig
)
from negative_space_analysis.interactive_system import (
    UserFeedback,
    FeedbackType
)


def load_sample_data(
    image_path: str,
    depth_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    text: Optional[str] = None
) -> Dict:
    """Load sample data for analysis."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load depth if available
    depth = None
    if depth_path:
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            print(f"Warning: Failed to load depth map: {depth_path}")
    
    # Load audio if available
    audio = None
    if audio_path:
        try:
            audio = np.load(audio_path)
        except Exception as e:
            print(f"Warning: Failed to load audio: {audio_path}")
    
    return {
        "image": image,
        "depth": depth,
        "audio": audio,
        "text": text
    }


def visualize_results(result, frame_idx: int):
    """Visualize analysis results."""
    plt.figure(figsize=(15, 10))
    
    # Plot original image with region overlays
    plt.subplot(2, 2, 1)
    plt.imshow(result.analysis_metadata["original_image"])
    for rid, mask in result.region_masks.items():
        # Create colored overlay for each region
        overlay = np.zeros_like(
            result.analysis_metadata["original_image"],
            dtype=np.float32
        )
        color = np.random.rand(3)
        overlay[mask] = color
        plt.imshow(overlay, alpha=0.3)
    plt.title("Negative Space Regions")
    
    # Plot semantic relationships
    plt.subplot(2, 2, 2)
    G = nx.Graph()
    for rid, context in result.semantic_contexts.items():
        G.add_node(rid)
        for relation in context.relations:
            G.add_edge(
                relation.source_id,
                relation.target_id,
                weight=relation.confidence
            )
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color='lightblue',
        node_size=500,
        font_size=8
    )
    plt.title("Semantic Relationships")
    
    # Plot temporal changes
    plt.subplot(2, 2, 3)
    changes = [(c.frame_idx, len(c.region_ids)) for c in result.temporal_changes]
    if changes:
        frames, counts = zip(*changes)
        plt.plot(frames, counts, '-o')
        plt.xlabel("Frame")
        plt.ylabel("Number of Changes")
        plt.title("Temporal Changes")
    
    # Plot refinement suggestions
    plt.subplot(2, 2, 4)
    suggestions = result.refinement_suggestions
    if suggestions:
        types = [s.suggestion_type.value for s in suggestions]
        confidences = [s.confidence for s in suggestions]
        plt.bar(range(len(types)), confidences)
        plt.xticks(range(len(types)), types, rotation=45)
        plt.ylabel("Confidence")
        plt.title("Refinement Suggestions")
    
    plt.tight_layout()
    plt.savefig(f"analysis_result_{frame_idx}.png")
    plt.close()


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize analyzer with configuration
    config = AnalysisConfig(
        feature_dim=256,
        batch_size=32,
        enable_temporal=True,
        enable_refinement=True,
        refinement_threshold=0.7,
        logging_level=logging.INFO
    )
    
    analyzer = NegativeSpaceAnalyzer(config)
    logger.info("Initialized negative space analyzer")
    
    # Sample data paths
    data_dir = Path("sample_data")
    image_paths = sorted(data_dir.glob("*.jpg"))
    depth_paths = sorted(data_dir.glob("depth/*.png"))
    audio_paths = sorted(data_dir.glob("audio/*.npy"))
    
    # Process sequence of frames
    for frame_idx, image_path in enumerate(image_paths):
        logger.info(f"Processing frame {frame_idx}")
        
        # Load corresponding data
        depth_path = depth_paths[frame_idx] if frame_idx < len(depth_paths) else None
        audio_path = audio_paths[frame_idx] if frame_idx < len(audio_paths) else None
        text = f"Frame {frame_idx} analysis"
        
        try:
            # Load data
            data = load_sample_data(
                str(image_path),
                str(depth_path) if depth_path else None,
                str(audio_path) if audio_path else None,
                text
            )
            
            # Store original image for visualization
            data["metadata"] = {"original_image": data["image"]}
            
            # Run analysis
            result = analyzer.analyze_frame(
                image=data["image"],
                depth=data["depth"],
                audio=data["audio"],
                text=data["text"],
                metadata=data["metadata"]
            )
            
            logger.info(
                f"Frame {frame_idx} analysis complete: "
                f"found {len(result.region_ids)} regions"
            )
            
            # Visualize results
            visualize_results(result, frame_idx)
            
            # Example of applying user feedback
            if result.refinement_suggestions:
                suggestion = result.refinement_suggestions[0]
                feedback = UserFeedback(
                    feedback_type=suggestion.suggestion_type,
                    region_ids=suggestion.region_ids,
                    parameters=suggestion.parameters,
                    timestamp=time.time(),
                    confidence=1.0
                )
                
                # Apply feedback and get updated results
                updated_result = analyzer.apply_feedback(feedback)
                if updated_result:
                    logger.info(
                        f"Applied feedback: regions updated "
                        f"from {len(result.region_ids)} "
                        f"to {len(updated_result.region_ids)}"
                    )
                    
                    # Visualize updated results
                    visualize_results(updated_result, f"{frame_idx}_refined")
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {str(e)}")
            continue
    
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
