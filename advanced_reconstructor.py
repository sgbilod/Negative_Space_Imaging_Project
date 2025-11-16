#!/usr/bin/env python
"""
Advanced Reconstruction Module for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

Upgrades the minimal prototype with:
- Semantic segmentation strategies (adaptive threshold, watershed)
- Morphological operations for noise reduction
- Connected-component labeling for accurate region detection
- Rich metrics (15+ per reconstruction vs 5 in prototype)
- Multi-scale analysis capability
- Comprehensive provenance tracking

Usage:
    from advanced_reconstructor import (
        AdvancedReconstructor,
        AdaptiveThresholdSegmenter,
        MorphologyConfig
    )

    reconstructor = AdvancedReconstructor(
        segmenter=AdaptiveThresholdSegmenter(method='otsu'),
        morphology=MorphologyConfig(operations=[('open', 3), ('close', 5)])
    )

    result = reconstructor.reconstruct(
        'processed_image.png',
        output_dir='reconstruction_outputs'
    )
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import label as scipy_label
from skimage import filters, morphology, measure, segmentation


# ============================================================================
# Configuration & Metadata
# ============================================================================

@dataclass
class MorphologyConfig:
    """Configuration for morphological operations."""
    operations: List[Tuple[str, int]] = field(default_factory=lambda: [
        ('open', 3),   # Remove noise
        ('close', 5)   # Fill holes
    ])


@dataclass
class ProvenanceMetadata:
    """Provenance tracking for reconstruction."""
    reconstruction_id: str
    timestamp: str
    segmentation_strategy: str
    morphology_operations: List[Tuple[str, int]]
    processing_time_ms: float
    image_dimensions: Tuple[int, int]
    security: Optional[Dict[str, Any]] = None


# ============================================================================
# Metrics Data Classes
# ============================================================================

@dataclass
class RegionMetrics:
    """Metrics for a single negative space region."""
    region_id: int
    area: int
    perimeter: float
    centroid: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]
    convexity: float
    circularity: float
    eccentricity: float
    aspect_ratio: float
    solidity: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SpatialDistribution:
    """Spatial distribution metrics."""
    density_map: List[List[int]]  # 4x4 grid of region counts
    spatial_entropy: float
    clustering_coefficient: float


@dataclass
class ReconstructionMetrics:
    """Aggregate reconstruction metrics."""
    total_regions: int
    negative_space_ratio: float
    mean_region_area: float
    median_region_area: float
    region_area_std: float
    largest_region_area: int
    smallest_region_area: int
    mean_convexity: float
    mean_circularity: float
    mean_eccentricity: float
    spatial_distribution: SpatialDistribution

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AdvancedReconstructionResult:
    """Complete reconstruction output."""
    image_path: str
    segmentation_strategy: str
    morphology_operations: List[Tuple[str, int]]
    regions: List[RegionMetrics]
    aggregate_metrics: ReconstructionMetrics
    artifacts: Dict[str, str]
    provenance: ProvenanceMetadata

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            'image_path': self.image_path,
            'segmentation_strategy': self.segmentation_strategy,
            'morphology_operations': self.morphology_operations,
            'regions': [r.to_dict() for r in self.regions],
            'aggregate_metrics': self.aggregate_metrics.to_dict(),
            'artifacts': self.artifacts,
            'provenance': asdict(self.provenance)
        }


# ============================================================================
# Segmentation Strategies
# ============================================================================

class SegmentationStrategy(ABC):
    """Abstract base class for segmentation algorithms."""

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment image into negative space regions.

        Args:
            image: Grayscale image as numpy array

        Returns:
            Binary mask where True = negative space
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name for provenance."""
        pass


class AdaptiveThresholdSegmenter(SegmentationStrategy):
    """
    Adaptive threshold-based segmentation.

    Supports multiple methods:
    - 'otsu': Otsu's automatic threshold selection
    - 'mean': Mean-based local adaptive threshold
    - 'gaussian': Gaussian-weighted local adaptive threshold
    """

    def __init__(self, method: str = 'otsu', block_size: int = 35):
        """
        Initialize segmenter.

        Args:
            method: Thresholding method ('otsu', 'mean', 'gaussian')
            block_size: Block size for local adaptive methods
        """
        self.method = method
        self.block_size = block_size

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment using adaptive thresholding."""
        if self.method == 'otsu':
            threshold = filters.threshold_otsu(image)
            return image < threshold
        elif self.method == 'mean':
            threshold = filters.threshold_local(
                image,
                block_size=self.block_size,
                method='mean'
            )
            return image < threshold
        elif self.method == 'gaussian':
            threshold = filters.threshold_local(
                image,
                block_size=self.block_size,
                method='gaussian'
            )
            return image < threshold
        else:
            raise ValueError(f"Unknown method: {self.method}")

    @property
    def name(self) -> str:
        return f"AdaptiveThreshold({self.method})"


class WatershedSegmenter(SegmentationStrategy):
    """
    Watershed-based segmentation.

    Uses morphological gradient and watershed transform for
    accurate boundary localization.
    """

    def __init__(self, min_distance: int = 10):
        """
        Initialize watershed segmenter.

        Args:
            min_distance: Minimum distance between markers
        """
        self.min_distance = min_distance

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment using watershed transform."""
        # Compute gradient
        gradient = filters.sobel(image)

        # Find markers (peaks in distance transform)
        distance = ndimage.distance_transform_edt(image < filters.threshold_otsu(image))
        local_max = morphology.local_maxima(distance, indices=False)
        markers = ndimage.label(local_max)[0]

        # Watershed
        labels = segmentation.watershed(gradient, markers)

        # Return binary mask of labeled regions
        return labels > 0

    @property
    def name(self) -> str:
        return f"Watershed(min_dist={self.min_distance})"


# ============================================================================
# Morphological Processing
# ============================================================================

class MorphologyProcessor:
    """Apply morphological operations to clean segmentation masks."""

    def __init__(self, config: MorphologyConfig):
        """
        Initialize processor.

        Args:
            config: Morphology configuration
        """
        self.config = config

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply configured morphological operations.

        Args:
            mask: Binary mask

        Returns:
            Processed mask
        """
        result = mask.copy()

        for operation, size in self.config.operations:
            struct = morphology.disk(size)

            if operation == 'open':
                result = morphology.binary_opening(result, struct)
            elif operation == 'close':
                result = morphology.binary_closing(result, struct)
            elif operation == 'erode':
                result = morphology.binary_erosion(result, struct)
            elif operation == 'dilate':
                result = morphology.binary_dilation(result, struct)
            else:
                raise ValueError(f"Unknown operation: {operation}")

        return result


# ============================================================================
# Component Analysis
# ============================================================================

class ComponentAnalyzer:
    """Analyze connected components and extract region metrics."""

    def analyze(
        self,
        mask: np.ndarray,
        min_area: int = 10
    ) -> List[RegionMetrics]:
        """
        Analyze connected components in binary mask.

        Args:
            mask: Binary mask
            min_area: Minimum region area to include

        Returns:
            List of RegionMetrics for each component
        """
        # Label connected components
        labeled_mask = morphology.label(mask, connectivity=2)

        # Extract region properties
        regions = measure.regionprops(labeled_mask)

        metrics_list = []
        for idx, region in enumerate(regions, start=1):
            if region.area < min_area:
                continue

            # Calculate metrics
            metrics = self._calculate_region_metrics(region, idx)
            metrics_list.append(metrics)

        return metrics_list

    def _calculate_region_metrics(
        self,
        region: Any,
        region_id: int
    ) -> RegionMetrics:
        """Calculate all metrics for a single region."""
        # Basic properties
        area = region.area
        perimeter = region.perimeter
        centroid = region.centroid
        bbox = region.bbox  # (min_row, min_col, max_row, max_col)

        # Convexity
        convex_area = region.convex_area
        convexity = area / convex_area if convex_area > 0 else 0.0

        # Circularity (compactness)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

        # Eccentricity (from ellipse fit)
        eccentricity = region.eccentricity

        # Aspect ratio (from bounding box)
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        aspect_ratio = width / height if height > 0 else 0.0

        # Solidity
        solidity = region.solidity

        return RegionMetrics(
            region_id=region_id,
            area=area,
            perimeter=perimeter,
            centroid=(centroid[1], centroid[0]),  # (x, y)
            bounding_box=(bbox[1], bbox[0], bbox[3], bbox[2]),  # (x1, y1, x2, y2)
            convexity=convexity,
            circularity=circularity,
            eccentricity=eccentricity,
            aspect_ratio=aspect_ratio,
            solidity=solidity
        )


# ============================================================================
# Metrics Calculator
# ============================================================================

class MetricsCalculator:
    """Calculate aggregate metrics from region data."""

    def calculate(
        self,
        regions: List[RegionMetrics],
        mask: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> ReconstructionMetrics:
        """
        Calculate aggregate metrics.

        Args:
            regions: List of region metrics
            mask: Binary mask
            image_shape: (height, width)

        Returns:
            ReconstructionMetrics
        """
        if not regions:
            # No regions found
            return self._empty_metrics(image_shape)

        # Region statistics
        areas = [r.area for r in regions]
        total_regions = len(regions)
        mean_area = np.mean(areas)
        median_area = np.median(areas)
        std_area = np.std(areas)
        largest_area = max(areas)
        smallest_area = min(areas)

        # Negative space ratio
        total_pixels = image_shape[0] * image_shape[1]
        negative_pixels = np.sum(mask)
        negative_space_ratio = negative_pixels / total_pixels if total_pixels > 0 else 0.0

        # Shape metrics
        mean_convexity = np.mean([r.convexity for r in regions])
        mean_circularity = np.mean([r.circularity for r in regions])
        mean_eccentricity = np.mean([r.eccentricity for r in regions])

        # Spatial distribution
        spatial_dist = self._calculate_spatial_distribution(regions, image_shape)

        return ReconstructionMetrics(
            total_regions=total_regions,
            negative_space_ratio=negative_space_ratio,
            mean_region_area=mean_area,
            median_region_area=median_area,
            region_area_std=std_area,
            largest_region_area=largest_area,
            smallest_region_area=smallest_area,
            mean_convexity=mean_convexity,
            mean_circularity=mean_circularity,
            mean_eccentricity=mean_eccentricity,
            spatial_distribution=spatial_dist
        )

    def _calculate_spatial_distribution(
        self,
        regions: List[RegionMetrics],
        image_shape: Tuple[int, int]
    ) -> SpatialDistribution:
        """Calculate spatial distribution metrics."""
        height, width = image_shape

        # 4x4 density map
        grid_h, grid_w = 4, 4
        cell_h = height / grid_h
        cell_w = width / grid_w

        density_map = [[0 for _ in range(grid_w)] for _ in range(grid_h)]

        for region in regions:
            cx, cy = region.centroid
            grid_x = min(int(cx / cell_w), grid_w - 1)
            grid_y = min(int(cy / cell_h), grid_h - 1)
            density_map[grid_y][grid_x] += 1

        # Spatial entropy (uniformity)
        flat_density = [count for row in density_map for count in row]
        total_regions = sum(flat_density)
        if total_regions > 0:
            probs = [c / total_regions for c in flat_density if c > 0]
            spatial_entropy = -sum(p * np.log2(p) for p in probs)
        else:
            spatial_entropy = 0.0

        # Clustering coefficient (simplified)
        # Higher values = more clustered
        if len(regions) > 1:
            centroids = np.array([r.centroid for r in regions])
            distances = []
            for i, c1 in enumerate(centroids):
                for c2 in centroids[i+1:]:
                    distances.append(np.linalg.norm(c1 - c2))
            mean_dist = np.mean(distances)
            # Normalize by image diagonal
            diagonal = np.sqrt(height**2 + width**2)
            clustering_coefficient = 1.0 - (mean_dist / diagonal)
        else:
            clustering_coefficient = 0.0

        return SpatialDistribution(
            density_map=density_map,
            spatial_entropy=spatial_entropy,
            clustering_coefficient=max(0.0, clustering_coefficient)
        )

    def _empty_metrics(self, image_shape: Tuple[int, int]) -> ReconstructionMetrics:
        """Return empty metrics when no regions found."""
        return ReconstructionMetrics(
            total_regions=0,
            negative_space_ratio=0.0,
            mean_region_area=0.0,
            median_region_area=0.0,
            region_area_std=0.0,
            largest_region_area=0,
            smallest_region_area=0,
            mean_convexity=0.0,
            mean_circularity=0.0,
            mean_eccentricity=0.0,
            spatial_distribution=SpatialDistribution(
                density_map=[[0]*4 for _ in range(4)],
                spatial_entropy=0.0,
                clustering_coefficient=0.0
            )
        )


# ============================================================================
# Main Reconstructor
# ============================================================================

class AdvancedReconstructor:
    """
    Advanced reconstruction orchestrator.

    Coordinates segmentation, morphology, component analysis, and
    metrics calculation.
    """

    def __init__(
        self,
        segmenter: Optional[SegmentationStrategy] = None,
        morphology: Optional[MorphologyConfig] = None,
        min_region_area: int = 10
    ):
        """
        Initialize reconstructor.

        Args:
            segmenter: Segmentation strategy (default: AdaptiveThresholdSegmenter)
            morphology: Morphology configuration (default: open+close)
            min_region_area: Minimum region area to include
        """
        self.segmenter = segmenter or AdaptiveThresholdSegmenter(method='otsu')
        self.morphology_config = morphology or MorphologyConfig()
        self.min_region_area = min_region_area

        self.morphology_processor = MorphologyProcessor(self.morphology_config)
        self.component_analyzer = ComponentAnalyzer()
        self.metrics_calculator = MetricsCalculator()

    def reconstruct(
        self,
        image_path: str,
        output_dir: Optional[str] = None
    ) -> AdvancedReconstructionResult:
        """
        Perform advanced reconstruction on processed image.

        Args:
            image_path: Path to processed image
            output_dir: Optional directory for artifacts

        Returns:
            AdvancedReconstructionResult
        """
        start_time = time.time()

        # Load image
        image = self._load_image(image_path)
        height, width = image.shape

        # Segmentation
        raw_mask = self.segmenter.segment(image)

        # Morphological processing
        cleaned_mask = self.morphology_processor.apply(raw_mask)

        # Component analysis
        regions = self.component_analyzer.analyze(
            cleaned_mask,
            min_area=self.min_region_area
        )

        # Aggregate metrics
        aggregate_metrics = self.metrics_calculator.calculate(
            regions,
            cleaned_mask,
            (height, width)
        )

        # Processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Generate provenance
        provenance = ProvenanceMetadata(
            reconstruction_id=self._generate_id(),
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            segmentation_strategy=self.segmenter.name,
            morphology_operations=self.morphology_config.operations,
            processing_time_ms=processing_time_ms,
            image_dimensions=(width, height)
        )

        # Save artifacts
        artifacts = {}
        if output_dir:
            artifacts = self._save_artifacts(
                output_dir,
                raw_mask,
                cleaned_mask,
                regions,
                (height, width)
            )
            # Security signing of artifacts
            try:
                from security_module import load_or_create_keys, sign_artifact
                private_key, public_key, key_id = load_or_create_keys()
                signed = []
                for name, path in artifacts.items():
                    signed.append(sign_artifact(private_key, name, path))
                provenance.security = {
                    'key_id': key_id,
                    'hash_algorithm': 'sha256',
                    'artifacts': [s.to_dict() for s in signed]
                }
            except Exception as e:  # noqa: BLE001
                # Non-fatal security error
                provenance.security = {'error': f'signing_failed: {e}'}

        return AdvancedReconstructionResult(
            image_path=image_path,
            segmentation_strategy=self.segmenter.name,
            morphology_operations=self.morphology_config.operations,
            regions=regions,
            aggregate_metrics=aggregate_metrics,
            artifacts=artifacts,
            provenance=provenance
        )

    def _load_image(self, path: str) -> np.ndarray:
        """Load image as grayscale numpy array."""
        img = Image.open(path).convert('L')
        return np.array(img)

    def _generate_id(self) -> str:
        """Generate unique reconstruction ID."""
        import hashlib
        timestamp = datetime.now(timezone.utc).isoformat()
        hash_input = f"{timestamp}{time.time()}"
        hash_suffix = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        return f"recon-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{hash_suffix}"

    def _save_artifacts(
        self,
        output_dir: str,
        raw_mask: np.ndarray,
        cleaned_mask: np.ndarray,
        regions: List[RegionMetrics],
        image_shape: Tuple[int, int]
    ) -> Dict[str, str]:
        """Save reconstruction artifacts."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        artifacts = {}

        # Save raw mask
        raw_mask_path = output_path / 'mask_raw.png'
        Image.fromarray((raw_mask * 255).astype(np.uint8)).save(raw_mask_path)
        artifacts['mask_raw'] = str(raw_mask_path)

        # Save cleaned mask
        cleaned_mask_path = output_path / 'mask_morphed.png'
        Image.fromarray((cleaned_mask * 255).astype(np.uint8)).save(cleaned_mask_path)
        artifacts['mask_morphed'] = str(cleaned_mask_path)

        # Save labeled regions (color-coded)
        labeled_mask = morphology.label(cleaned_mask, connectivity=2)
        labeled_path = output_path / 'labeled_regions.png'
        # Normalize labels for visualization
        if labeled_mask.max() > 0:
            labeled_viz = ((labeled_mask / labeled_mask.max()) * 255).astype(np.uint8)
        else:
            labeled_viz = np.zeros_like(labeled_mask, dtype=np.uint8)
        Image.fromarray(labeled_viz).save(labeled_path)
        artifacts['labeled_regions'] = str(labeled_path)

        return artifacts


# ============================================================================
# Helper Functions
# ============================================================================

def reconstruct_image(
    image_path: str,
    output_dir: str,
    segmentation_method: str = 'otsu',
    morphology_ops: Optional[List[Tuple[str, int]]] = None
) -> AdvancedReconstructionResult:
    """
    Convenience function for reconstruction.

    Args:
        image_path: Path to processed image
        output_dir: Output directory for artifacts
        segmentation_method: 'otsu', 'mean', 'gaussian', or 'watershed'
        morphology_ops: List of (operation, size) tuples

    Returns:
        AdvancedReconstructionResult
    """
    # Create segmenter
    if segmentation_method == 'watershed':
        segmenter = WatershedSegmenter()
    else:
        segmenter = AdaptiveThresholdSegmenter(method=segmentation_method)

    # Create morphology config
    if morphology_ops is None:
        morphology_ops = [('open', 3), ('close', 5)]
    morphology_config = MorphologyConfig(operations=morphology_ops)

    # Create reconstructor and run
    reconstructor = AdvancedReconstructor(
        segmenter=segmenter,
        morphology=morphology_config
    )

    return reconstructor.reconstruct(image_path, output_dir)


if __name__ == '__main__':
    # Example usage
    import sys
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python advanced_reconstructor.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'reconstruction_outputs'

    print(f"\nRunning advanced reconstruction on: {image_path}")
    print(f"Output directory: {output_dir}\n")

    result = reconstruct_image(image_path, output_dir)

    print("=== Reconstruction Complete ===")
    print(f"Regions detected: {result.aggregate_metrics.total_regions}")
    print(f"Negative space ratio: {result.aggregate_metrics.negative_space_ratio:.3f}")
    print(f"Mean region area: {result.aggregate_metrics.mean_region_area:.1f} pixels")
    print(f"Spatial entropy: {result.aggregate_metrics.spatial_distribution.spatial_entropy:.3f}")
    print(f"Processing time: {result.provenance.processing_time_ms:.2f}ms")
    print(f"\nArtifacts saved:")
    for name, path in result.artifacts.items():
        print(f"  - {name}: {path}")

    # Save JSON summary
    json_path = Path(output_dir) / 'reconstruction_result.json'
    with open(json_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nJSON summary: {json_path}")
