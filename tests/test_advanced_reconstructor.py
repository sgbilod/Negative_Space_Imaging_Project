#!/usr/bin/env python
"""
Unit tests for Advanced Reconstruction Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

Tests cover:
- Segmentation strategies (adaptive threshold, watershed)
- Morphological processing
- Component analysis and metrics
- Full reconstruction pipeline
- Artifact generation
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image

from advanced_reconstructor import (
    AdvancedReconstructor,
    AdaptiveThresholdSegmenter,
    WatershedSegmenter,
    MorphologyConfig,
    MorphologyProcessor,
    ComponentAnalyzer,
    MetricsCalculator,
    RegionMetrics,
    ReconstructionMetrics,
    SpatialDistribution,
    reconstruct_image
)


@pytest.fixture
def synthetic_image():
    """Create synthetic test image with clear negative space regions."""
    # Create 256x256 image with 3 dark circles (negative space)
    img = np.ones((256, 256), dtype=np.uint8) * 200  # Light background

    # Add 3 dark circles
    y, x = np.ogrid[:256, :256]

    # Circle 1 (top-left)
    mask1 = (x - 64)**2 + (y - 64)**2 <= 20**2
    img[mask1] = 50

    # Circle 2 (top-right)
    mask2 = (x - 192)**2 + (y - 64)**2 <= 30**2
    img[mask2] = 40

    # Circle 3 (bottom-center)
    mask3 = (x - 128)**2 + (y - 192)**2 <= 25**2
    img[mask3] = 45

    return img


@pytest.fixture
def temp_image_file(synthetic_image):
    """Save synthetic image to temporary file."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        Image.fromarray(synthetic_image).save(f.name)
        yield f.name
    Path(f.name).unlink()


class TestSegmentationStrategies:
    """Test suite for segmentation strategies."""

    def test_adaptive_threshold_otsu(self, synthetic_image):
        """Test Otsu thresholding."""
        segmenter = AdaptiveThresholdSegmenter(method='otsu')
        mask = segmenter.segment(synthetic_image)

        assert mask.shape == synthetic_image.shape
        assert mask.dtype == bool
        assert np.any(mask)  # Should find some negative space
        assert segmenter.name == "AdaptiveThreshold(otsu)"

    def test_adaptive_threshold_mean(self, synthetic_image):
        """Test mean adaptive thresholding."""
        segmenter = AdaptiveThresholdSegmenter(method='mean', block_size=35)
        mask = segmenter.segment(synthetic_image)

        assert mask.shape == synthetic_image.shape
        assert mask.dtype == bool
        assert np.any(mask)

    def test_adaptive_threshold_gaussian(self, synthetic_image):
        """Test Gaussian adaptive thresholding."""
        segmenter = AdaptiveThresholdSegmenter(method='gaussian', block_size=35)
        mask = segmenter.segment(synthetic_image)

        assert mask.shape == synthetic_image.shape
        assert mask.dtype == bool
        assert np.any(mask)

    def test_watershed_segmenter(self, synthetic_image):
        """Test watershed segmentation."""
        segmenter = WatershedSegmenter(min_distance=10)
        mask = segmenter.segment(synthetic_image)

        assert mask.shape == synthetic_image.shape
        assert mask.dtype == bool
        assert "Watershed" in segmenter.name

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        segmenter = AdaptiveThresholdSegmenter(method='invalid')

        with pytest.raises(ValueError):
            segmenter.segment(np.zeros((10, 10)))


class TestMorphologyProcessor:
    """Test suite for morphological processing."""

    def test_open_operation(self):
        """Test opening operation removes noise."""
        # Create noisy mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True  # Large region
        mask[10, 10] = True  # Single noise pixel

        config = MorphologyConfig(operations=[('open', 3)])
        processor = MorphologyProcessor(config)
        result = processor.apply(mask)

        assert not result[10, 10]  # Noise removed
        assert np.any(result[40:60, 40:60])  # Large region preserved

    def test_close_operation(self):
        """Test closing operation fills holes."""
        # Create mask with hole
        mask = np.ones((100, 100), dtype=bool)
        mask[49:51, 49:51] = False  # Small hole

        config = MorphologyConfig(operations=[('close', 3)])
        processor = MorphologyProcessor(config)
        result = processor.apply(mask)

        assert np.all(result[49:51, 49:51])  # Hole filled

    def test_multiple_operations(self):
        """Test sequential operations."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True

        config = MorphologyConfig(operations=[
            ('open', 2),
            ('close', 2),
            ('erode', 1)
        ])
        processor = MorphologyProcessor(config)
        result = processor.apply(mask)

        assert result.shape == mask.shape
        assert result.dtype == bool

    def test_invalid_operation_raises_error(self):
        """Test invalid operation raises ValueError."""
        config = MorphologyConfig(operations=[('invalid', 3)])
        processor = MorphologyProcessor(config)

        with pytest.raises(ValueError):
            processor.apply(np.zeros((10, 10), dtype=bool))


class TestComponentAnalyzer:
    """Test suite for component analysis."""

    def test_analyze_single_region(self):
        """Test analysis of single region."""
        # Create mask with single rectangle
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 30:70] = True

        analyzer = ComponentAnalyzer()
        regions = analyzer.analyze(mask, min_area=10)

        assert len(regions) == 1
        region = regions[0]

        assert region.region_id == 1
        assert region.area == 20 * 40  # 800 pixels
        assert region.perimeter > 0
        assert 0 <= region.convexity <= 1.0
        assert 0 <= region.circularity <= 1.0
        assert 0 <= region.eccentricity <= 1.0

    def test_analyze_multiple_regions(self):
        """Test analysis of multiple regions."""
        # Create mask with 3 circles
        mask = np.zeros((200, 200), dtype=bool)
        y, x = np.ogrid[:200, :200]

        mask1 = (x - 50)**2 + (y - 50)**2 <= 20**2
        mask2 = (x - 150)**2 + (y - 50)**2 <= 20**2
        mask3 = (x - 100)**2 + (y - 150)**2 <= 20**2

        mask = mask1 | mask2 | mask3

        analyzer = ComponentAnalyzer()
        regions = analyzer.analyze(mask, min_area=10)

        assert len(regions) == 3

        # Check unique IDs
        ids = [r.region_id for r in regions]
        assert len(set(ids)) == 3

        # Check all have positive area
        assert all(r.area > 0 for r in regions)

    def test_min_area_filtering(self):
        """Test minimum area filtering."""
        mask = np.zeros((100, 100), dtype=bool)

        # Large region
        mask[10:30, 10:30] = True

        # Small region (should be filtered)
        mask[50:52, 50:52] = True

        analyzer = ComponentAnalyzer()
        regions = analyzer.analyze(mask, min_area=100)

        assert len(regions) == 1  # Only large region
        assert regions[0].area >= 100


class TestMetricsCalculator:
    """Test suite for metrics calculation."""

    def test_calculate_with_regions(self):
        """Test metrics calculation with valid regions."""
        # Create sample regions
        regions = [
            RegionMetrics(
                region_id=1,
                area=100,
                perimeter=40.0,
                centroid=(50.0, 50.0),
                bounding_box=(40, 40, 60, 60),
                convexity=0.9,
                circularity=0.8,
                eccentricity=0.5,
                aspect_ratio=1.0,
                solidity=0.85
            ),
            RegionMetrics(
                region_id=2,
                area=200,
                perimeter=60.0,
                centroid=(150.0, 150.0),
                bounding_box=(140, 140, 160, 160),
                convexity=0.85,
                circularity=0.75,
                eccentricity=0.6,
                aspect_ratio=1.0,
                solidity=0.80
            )
        ]

        mask = np.zeros((200, 200), dtype=bool)
        mask[40:60, 40:60] = True
        mask[140:160, 140:160] = True

        calculator = MetricsCalculator()
        metrics = calculator.calculate(regions, mask, (200, 200))

        assert metrics.total_regions == 2
        assert metrics.mean_region_area == 150.0
        assert metrics.largest_region_area == 200
        assert metrics.smallest_region_area == 100
        assert 0 <= metrics.negative_space_ratio <= 1.0
        assert metrics.mean_convexity > 0
        assert metrics.mean_circularity > 0

    def test_calculate_empty_regions(self):
        """Test metrics calculation with no regions."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate([], np.zeros((100, 100), dtype=bool), (100, 100))

        assert metrics.total_regions == 0
        assert metrics.mean_region_area == 0.0
        assert metrics.negative_space_ratio == 0.0

    def test_spatial_distribution(self):
        """Test spatial distribution calculation."""
        # Create regions in different quadrants
        regions = [
            RegionMetrics(1, 100, 40, (25.0, 25.0), (0, 0, 50, 50),
                         0.9, 0.8, 0.5, 1.0, 0.85),
            RegionMetrics(2, 100, 40, (175.0, 25.0), (150, 0, 200, 50),
                         0.9, 0.8, 0.5, 1.0, 0.85),
            RegionMetrics(3, 100, 40, (25.0, 175.0), (0, 150, 50, 200),
                         0.9, 0.8, 0.5, 1.0, 0.85),
            RegionMetrics(4, 100, 40, (175.0, 175.0), (150, 150, 200, 200),
                         0.9, 0.8, 0.5, 1.0, 0.85)
        ]

        calculator = MetricsCalculator()
        mask = np.zeros((200, 200), dtype=bool)
        metrics = calculator.calculate(regions, mask, (200, 200))

        spatial_dist = metrics.spatial_distribution

        assert isinstance(spatial_dist, SpatialDistribution)
        assert len(spatial_dist.density_map) == 4
        assert len(spatial_dist.density_map[0]) == 4
        assert spatial_dist.spatial_entropy >= 0
        assert 0 <= spatial_dist.clustering_coefficient <= 1.0


class TestAdvancedReconstructor:
    """Test suite for full reconstruction pipeline."""

    def test_reconstruct_basic(self, temp_image_file):
        """Test basic reconstruction."""
        reconstructor = AdvancedReconstructor()
        result = reconstructor.reconstruct(temp_image_file)

        assert result.image_path == temp_image_file
        assert result.segmentation_strategy is not None
        assert len(result.morphology_operations) > 0
        assert result.aggregate_metrics.total_regions >= 0
        assert result.provenance.processing_time_ms > 0

    def test_reconstruct_with_output_dir(self, temp_image_file):
        """Test reconstruction with artifact output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reconstructor = AdvancedReconstructor()
            result = reconstructor.reconstruct(temp_image_file, output_dir=tmpdir)

            assert len(result.artifacts) > 0

            # Check artifacts exist
            for artifact_path in result.artifacts.values():
                assert Path(artifact_path).exists()

    def test_reconstruct_with_custom_segmenter(self, temp_image_file):
        """Test reconstruction with custom segmenter."""
        segmenter = AdaptiveThresholdSegmenter(method='mean')
        reconstructor = AdvancedReconstructor(segmenter=segmenter)

        result = reconstructor.reconstruct(temp_image_file)

        assert "mean" in result.segmentation_strategy.lower()

    def test_reconstruct_with_custom_morphology(self, temp_image_file):
        """Test reconstruction with custom morphology."""
        morphology = MorphologyConfig(operations=[
            ('open', 2),
            ('close', 3)
        ])
        reconstructor = AdvancedReconstructor(morphology=morphology)

        result = reconstructor.reconstruct(temp_image_file)

        assert result.morphology_operations == [('open', 2), ('close', 3)]

    def test_reconstruct_detects_regions(self, temp_image_file):
        """Test that reconstruction detects expected regions."""
        reconstructor = AdvancedReconstructor(min_region_area=50)
        result = reconstructor.reconstruct(temp_image_file)

        # Should detect 3 dark circles from synthetic image
        assert result.aggregate_metrics.total_regions >= 1
        assert result.aggregate_metrics.negative_space_ratio > 0

    def test_region_metrics_completeness(self, temp_image_file):
        """Test that region metrics are complete."""
        reconstructor = AdvancedReconstructor()
        result = reconstructor.reconstruct(temp_image_file)

        if result.regions:
            region = result.regions[0]

            assert region.region_id > 0
            assert region.area > 0
            assert region.perimeter > 0
            assert len(region.centroid) == 2
            assert len(region.bounding_box) == 4
            assert 0 <= region.convexity <= 1.0
            assert 0 <= region.circularity <= 1.0
            assert 0 <= region.eccentricity <= 1.0
            assert region.aspect_ratio > 0
            assert 0 <= region.solidity <= 1.0

    def test_result_serialization(self, temp_image_file):
        """Test that result can be serialized to dict."""
        reconstructor = AdvancedReconstructor()
        result = reconstructor.reconstruct(temp_image_file)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert 'image_path' in result_dict
        assert 'segmentation_strategy' in result_dict
        assert 'regions' in result_dict
        assert 'aggregate_metrics' in result_dict
        assert 'provenance' in result_dict


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_reconstruct_image_default(self, temp_image_file):
        """Test convenience function with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = reconstruct_image(temp_image_file, tmpdir)

            assert result.aggregate_metrics.total_regions >= 0
            assert len(result.artifacts) > 0

    def test_reconstruct_image_custom_method(self, temp_image_file):
        """Test convenience function with custom method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = reconstruct_image(
                temp_image_file,
                tmpdir,
                segmentation_method='mean'
            )

            assert "mean" in result.segmentation_strategy.lower()

    def test_reconstruct_image_watershed(self, temp_image_file):
        """Test convenience function with watershed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = reconstruct_image(
                temp_image_file,
                tmpdir,
                segmentation_method='watershed'
            )

            assert "watershed" in result.segmentation_strategy.lower()

    def test_reconstruct_image_custom_morphology(self, temp_image_file):
        """Test convenience function with custom morphology."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = reconstruct_image(
                temp_image_file,
                tmpdir,
                morphology_ops=[('erode', 2), ('dilate', 2)]
            )

            assert result.morphology_operations == [('erode', 2), ('dilate', 2)]


class TestPerformance:
    """Performance and scalability tests."""

    @pytest.mark.parametrize("size", [128, 256, 512])
    def test_processing_time_scales(self, size):
        """Test that processing time scales reasonably."""
        # Create test image
        img = np.random.randint(0, 256, (size, size), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            Image.fromarray(img).save(f.name)
            temp_path = f.name

        try:
            reconstructor = AdvancedReconstructor()
            result = reconstructor.reconstruct(temp_path)

            # Time should be reasonable
            assert result.provenance.processing_time_ms < 5000  # 5 seconds max

            print(f"\n{size}x{size}: {result.provenance.processing_time_ms:.2f}ms")

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
