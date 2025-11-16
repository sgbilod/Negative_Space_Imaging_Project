# Advanced Reconstruction Design

**Track C – Enhanced Negative Space Reconstruction**

## Overview

The Advanced Reconstruction module (`advanced_reconstructor.py`) upgrades the minimal prototype with:
- **Semantic Segmentation** for intelligent region detection
- **Morphological Operations** for noise reduction and boundary refinement
- **Connected-Component Labeling** for accurate region counting
- **Richer Metrics** (convexity, circularity, perimeter ratios, spatial distribution)
- **Multi-Scale Analysis** for hierarchical negative space detection
- **Benchmarking Hooks** for performance tracking and optimization

## Design Goals

1. **Accuracy**: Move from heuristic thresholding to principled segmentation
2. **Richness**: Expand metrics to support downstream analysis and ML
3. **Performance**: Maintain < 1s processing for 1024×1024 images
4. **Modularity**: Plugin-based segmentation strategies (threshold, watershed, ML-based)
5. **Traceability**: Detailed reconstruction provenance and intermediate artifacts

## Current Limitations (Prototype)

The existing `negative_space_reconstructor.py`:
- Uses 25th percentile threshold (arbitrary, no adaptation)
- Counts regions via row-scan transitions (misses 2D connectivity)
- Provides only 5 basic metrics (mean intensity, ratio, region count, dimensions)
- No morphological cleanup (noise artifacts persist)
- No spatial distribution analysis
- No multi-resolution support

## Proposed Enhancements

### 1. Segmentation Pipeline

```python
class SegmentationStrategy(ABC):
    """Abstract base for segmentation algorithms."""
    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Return binary mask of negative space."""
        pass

# Implementations:
- AdaptiveThresholdSegmenter (Otsu, local adaptive)
- WatershedSegmenter (marker-based watershed)
- MLSegmenter (semantic segmentation model - future)
```

### 2. Morphological Processing

```python
class MorphologyProcessor:
    """Clean and refine segmentation masks."""

    def apply_operations(
        self,
        mask: np.ndarray,
        operations: List[str]
    ) -> np.ndarray:
        """
        Apply sequence of morphological ops:
        - 'open': Remove small objects (noise)
        - 'close': Fill small holes
        - 'erode': Shrink boundaries
        - 'dilate': Expand boundaries
        """
        pass
```

### 3. Connected-Component Analysis

Replace naive region counting with proper connected-component labeling:

```python
def analyze_components(mask: np.ndarray) -> ComponentAnalysis:
    """
    Use scipy.ndimage.label or cv2.connectedComponentsWithStats.

    Returns:
        ComponentAnalysis with per-region:
        - area, centroid, bounding box
        - convexity, circularity
        - perimeter, eccentricity
    """
    pass
```

### 4. Enhanced Metrics

Expand from 5 to 20+ metrics:

**Region Metrics:**
- Area (pixels, percentage)
- Perimeter, perimeter-to-area ratio
- Convexity (convex hull area / region area)
- Circularity (4π × area / perimeter²)
- Eccentricity, aspect ratio
- Centroid, bounding box

**Spatial Distribution:**
- Density map (regions per quadrant)
- Clustering coefficient (nearest-neighbor distances)
- Spatial entropy (uniformity of distribution)

**Multi-Scale:**
- Pyramid analysis (detect at 3 scales: 1x, 0.5x, 0.25x)
- Hierarchical region relationships (parent/child containment)

**Texture:**
- Local Binary Patterns (LBP) statistics
- Gray-level co-occurrence matrix (GLCM) features

### 5. Reconstruction Artifacts

Output multiple artifact types:

```
reconstruction_outputs/
├── mask_raw.png              # Initial segmentation
├── mask_morphed.png          # After morphological ops
├── labeled_regions.png       # Color-coded components
├── component_analysis.json   # Per-region detailed metrics
├── reconstruction_result.json # Aggregate summary
└── provenance.json           # Processing metadata
```

## Architecture

```
AdvancedReconstructor (orchestrator)
├── SegmentationStrategy (plugin interface)
│   ├── AdaptiveThresholdSegmenter
│   ├── WatershedSegmenter
│   └── MLSegmenter (future)
├── MorphologyProcessor (mask cleanup)
├── ComponentAnalyzer (region metrics)
├── MetricsCalculator (aggregate statistics)
└── ArtifactExporter (save outputs)
```

### Core Interfaces

```python
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

@dataclass
class ReconstructionMetrics:
    """Aggregate reconstruction metrics."""
    total_regions: int
    negative_space_ratio: float
    mean_region_area: float
    spatial_entropy: float
    clustering_coefficient: float
    pyramid_levels: int

@dataclass
class AdvancedReconstructionResult:
    """Complete reconstruction output."""
    image_path: str
    segmentation_strategy: str
    morphology_operations: List[str]
    regions: List[RegionMetrics]
    aggregate_metrics: ReconstructionMetrics
    artifacts: Dict[str, str]  # artifact_type -> file_path
    provenance: ProvenanceMetadata
    processing_time_ms: float
```

## Implementation Phases

### Phase 1 (Current Track C)
- [x] Design doc
- [ ] Core `AdvancedReconstructor` class
- [ ] Segmentation strategy interface + adaptive threshold impl
- [ ] Morphology processor
- [ ] Connected-component analyzer with basic metrics
- [ ] Enhanced JSON output with 15+ metrics
- [ ] Unit tests for each component
- [ ] Integration with `end_to_end_demo.py`

### Phase 2 (Track C Extension)
- [ ] Watershed segmentation
- [ ] Multi-scale pyramid analysis
- [ ] Spatial distribution metrics
- [ ] Texture feature extraction
- [ ] Performance benchmarks
- [ ] Comparison vs prototype (accuracy, speed)

### Phase 3 (Future)
- [ ] ML-based semantic segmentation
- [ ] Real-time processing mode
- [ ] GPU acceleration
- [ ] 3D reconstruction from multiple views

## Segmentation Strategies

### 1. Adaptive Threshold (Default)

**Algorithm**: Otsu's method or local adaptive thresholding
**Pros**: Fast, no training required, handles varying illumination
**Cons**: Struggles with complex textures, gradual transitions

```python
class AdaptiveThresholdSegmenter(SegmentationStrategy):
    def __init__(self, method: str = "otsu"):
        self.method = method  # 'otsu', 'gaussian', 'mean'

    def segment(self, image: np.ndarray) -> np.ndarray:
        if self.method == "otsu":
            threshold = threshold_otsu(image)
            return image < threshold
        elif self.method == "gaussian":
            return threshold_local(image, block_size=35, method='gaussian')
        # ...
```

### 2. Watershed Segmentation

**Algorithm**: Marker-based watershed transform
**Pros**: Good boundary localization, handles touching objects
**Cons**: Sensitive to markers, computationally intensive

```python
class WatershedSegmenter(SegmentationStrategy):
    def segment(self, image: np.ndarray) -> np.ndarray:
        # 1. Gradient computation
        gradient = sobel(image)

        # 2. Marker generation (local minima)
        markers = find_markers(gradient)

        # 3. Watershed transform
        labels = watershed(gradient, markers)

        return labels > 0  # binary mask
```

### 3. ML-Based Segmentation (Future)

**Algorithm**: U-Net or DeepLabv3+ trained on annotated data
**Pros**: State-of-art accuracy, learns domain-specific features
**Cons**: Requires training data, slower inference, model management

## Morphological Operations

Standard sequence:
1. **Open** (erode → dilate): Remove small noise specks
2. **Close** (dilate → erode): Fill small holes in regions
3. **Optional**: Additional erosion/dilation for boundary adjustment

```python
def apply_morphology(
    mask: np.ndarray,
    operations: List[Tuple[str, int]]
) -> np.ndarray:
    """
    Args:
        mask: Binary mask
        operations: List of (op_name, kernel_size) tuples

    Example:
        apply_morphology(mask, [
            ('open', 3),   # Remove noise
            ('close', 5),  # Fill holes
            ('erode', 1)   # Slight shrink
        ])
    """
    from scipy.ndimage import binary_opening, binary_closing, binary_erosion

    for op, size in operations:
        struct = np.ones((size, size))
        if op == 'open':
            mask = binary_opening(mask, structure=struct)
        elif op == 'close':
            mask = binary_closing(mask, structure=struct)
        elif op == 'erode':
            mask = binary_erosion(mask, structure=struct)
        # ...

    return mask
```

## Metrics Definitions

### Region-Level Metrics

1. **Area**: Number of pixels in region
2. **Perimeter**: Boundary length (approximate via contour)
3. **Convexity**: `convex_hull_area / region_area` (1.0 = perfectly convex)
4. **Circularity**: `4π × area / perimeter²` (1.0 = perfect circle)
5. **Eccentricity**: Ratio of focal distance to major axis (ellipse fit)
6. **Aspect Ratio**: `width / height` of bounding box
7. **Solidity**: `area / convex_hull_area`

### Aggregate Metrics

1. **Total Regions**: Count of connected components
2. **Negative Space Ratio**: `negative_pixels / total_pixels`
3. **Mean Region Area**: Average region size
4. **Region Area Variance**: Size distribution spread
5. **Spatial Entropy**: Uniformity of region distribution across image
6. **Clustering Coefficient**: Measure of region clustering vs random distribution

### Spatial Distribution

Divide image into 4×4 grid, compute:
- Regions per cell
- Area per cell
- Density heatmap

## Performance Targets

| Image Size | Processing Time | Memory | Regions Handled |
|------------|----------------|--------|-----------------|
| 256×256    | < 50ms         | < 20MB | 100+            |
| 512×512    | < 200ms        | < 50MB | 500+            |
| 1024×1024  | < 1s           | < 150MB| 2000+           |

## Testing Strategy

### Unit Tests
- Each segmentation strategy independently
- Morphology operations correctness
- Metric calculations (known ground truth)
- Component labeling accuracy

### Integration Tests
- Full pipeline with synthetic images
- Compare advanced vs prototype results
- Artifact generation and schema validation

### Performance Tests
- Speed benchmarks for different image sizes
- Memory profiling
- Scalability with region count

### Accuracy Tests
- Annotated test images with ground truth
- Precision, recall, F1 for segmentation
- Metric correlation with manual measurements

## Dependencies

**Core**:
- `numpy` (arrays)
- `scipy` (connected components, morphology, image processing)
- `scikit-image` (segmentation, metrics, transforms)
- `Pillow` (I/O)

**Optional**:
- `opencv-python` (alternative implementations, visualization)
- `matplotlib` (debugging visualizations)

## Integration with Pipeline

Replace simple reconstructor call in `end_to_end_demo.py`:

```python
# Before (prototype)
from negative_space_reconstructor import NegativeSpaceReconstructor
reconstructor = NegativeSpaceReconstructor()
result = reconstructor.reconstruct(processed_image_path)

# After (advanced)
from advanced_reconstructor import (
    AdvancedReconstructor,
    AdaptiveThresholdSegmenter,
    MorphologyConfig
)

reconstructor = AdvancedReconstructor(
    segmenter=AdaptiveThresholdSegmenter(method='otsu'),
    morphology=MorphologyConfig(
        operations=[('open', 3), ('close', 5)]
    )
)

result = reconstructor.reconstruct(
    processed_image_path,
    output_dir='reconstruction_outputs'
)

# Access richer data
print(f"Regions: {result.aggregate_metrics.total_regions}")
print(f"Spatial entropy: {result.aggregate_metrics.spatial_entropy:.3f}")
print(f"Top 5 largest regions:")
for region in sorted(result.regions, key=lambda r: r.area, reverse=True)[:5]:
    print(f"  - ID {region.region_id}: {region.area}px, "
          f"circularity={region.circularity:.2f}")
```

## Success Metrics

- ✅ 15+ metrics per reconstruction (vs 5 in prototype)
- ✅ Connected-component labeling (vs naive row-scan)
- ✅ Morphological cleanup reduces false positives by 30%+
- ✅ Processing time < 1s for 1024×1024 images
- ✅ 90%+ test coverage
- ✅ Drop-in compatible with existing pipeline

## Related Documents

- `negative_space_reconstructor.py` – Minimal prototype (to be superseded)
- `ARCHITECTURE.md` – System overview
- `TESTING_FRAMEWORK.md` – Test strategy
- `docs/ACQUISITION_SERVICE_DESIGN.md` – Acquisition architecture

---

**Status**: Design complete, ready for implementation
**Owner**: Stephen Bilodeau
**Created**: 2025-11-15
**Track**: C – Advanced Reconstruction
