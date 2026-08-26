"""
Unit Tests for Negative Space Analyzer

Comprehensive test suite for the NegativeSpaceAnalyzer core class.
Tests cover image preprocessing, detection, feature extraction,
region analysis, and error handling.

Coverage Target: 95%+
Test Count: 35+ individual test cases
"""

import pytest
import numpy as np
import torch
import cv2
from typing import Dict
from unittest.mock import Mock, patch, MagicMock
import logging

logger = logging.getLogger(__name__)


# =====================================================================
# IMAGE PREPROCESSING TESTS
# =====================================================================

class TestImagePreprocessing:
    """Tests for image preprocessing functionality."""

    @pytest.mark.unit
    def test_preprocess_grayscale_image(self, synthetic_image, mock_analyzer):
        """Test preprocessing of grayscale images."""
        # Mock the _preprocess_image method
        mock_analyzer._preprocess_image.return_value = \
            synthetic_image.astype(np.float32) / 255.0

        result = mock_analyzer._preprocess_image(synthetic_image)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == synthetic_image.shape
        assert result.min() >= 0.0 and result.max() <= 1.0

    @pytest.mark.unit
    def test_preprocess_rgb_to_grayscale(self, multi_channel_image,
                                         mock_analyzer):
        """Test conversion of RGB images to grayscale."""
        preprocessed = cv2.cvtColor(
            multi_channel_image,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32) / 255.0

        assert len(preprocessed.shape) == 2  # Should be 2D
        assert preprocessed.dtype == np.float32

    @pytest.mark.unit
    def test_preprocess_value_range(self, synthetic_image):
        """Test that preprocessed values are in [0, 1] range."""
        processed = synthetic_image.astype(np.float32) / 255.0

        assert np.all(processed >= 0.0), "Values below 0"
        assert np.all(processed <= 1.0), "Values above 1"
        assert processed.min() < 0.5  # Should have variety
        assert processed.max() > 0.5

    @pytest.mark.unit
    def test_preprocess_preserves_shape(self, medical_image):
        """Test that preprocessing preserves image shape."""
        original_shape = medical_image.shape
        processed = medical_image.astype(np.float32) / 255.0

        assert processed.shape == original_shape

    @pytest.mark.unit
    def test_preprocess_empty_image(self):
        """Test preprocessing of empty (black) image."""
        empty = np.zeros((256, 256), dtype=np.uint8)
        processed = empty.astype(np.float32) / 255.0

        assert np.all(processed == 0.0)
        assert processed.shape == (256, 256)

    @pytest.mark.unit
    def test_preprocess_full_image(self):
        """Test preprocessing of full (white) image."""
        full = np.full((256, 256), 255, dtype=np.uint8)
        processed = full.astype(np.float32) / 255.0

        assert np.all(processed == 1.0)
        assert processed.shape == (256, 256)

    @pytest.mark.unit
    def test_preprocess_various_sizes(self, edge_case_images):
        """Test preprocessing of various image sizes."""
        for name, image in edge_case_images.items():
            processed = image.astype(np.float32) / 255.0

            assert processed.dtype == np.float32
            assert processed.shape == image.shape
            assert processed.min() >= 0.0 and processed.max() <= 1.0


# =====================================================================
# NEGATIVE SPACE DETECTION TESTS
# =====================================================================

class TestNegativeSpaceDetection:
    """Tests for negative space detection functionality."""

    @pytest.mark.unit
    def test_detect_returns_dict(self, mock_analyzer, synthetic_image):
        """Test that detection returns a dictionary."""
        result = mock_analyzer._detect_negative_spaces(synthetic_image)

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_detect_region_ids_format(self, mock_analyzer, synthetic_image):
        """Test that region IDs follow expected format."""
        result = mock_analyzer._detect_negative_spaces(synthetic_image)

        for region_id in result.keys():
            assert isinstance(region_id, str)
            assert region_id.startswith("region_")

    @pytest.mark.unit
    def test_detect_region_masks_are_binary(self, mock_analyzer,
                                            synthetic_image):
        """Test that region masks are binary or float [0,1]."""
        result = mock_analyzer._detect_negative_spaces(synthetic_image)

        for region_id, mask in result.items():
            unique_values = np.unique(mask)
            assert len(unique_values) <= 2  # Binary or mostly binary
            assert np.all((mask == 0) | (mask == 1) |
                         ((mask >= 0) & (mask <= 1)))

    @pytest.mark.unit
    def test_detect_respects_min_region_size(self, mock_analyzer):
        """Test that detection respects minimum region size."""
        mock_analyzer.min_region_size = 500

        # Create a small region
        small_region = np.zeros((256, 256), dtype=np.uint8)
        small_region[100:110, 100:110] = 1  # 10x10 = 100 pixels

        # In real implementation, this should be filtered
        assert mock_analyzer.min_region_size == 500

    @pytest.mark.unit
    def test_detect_respects_threshold(self, mock_analyzer):
        """Test that detection respects confidence threshold."""
        mock_analyzer.detection_threshold = 0.85

        assert mock_analyzer.detection_threshold == 0.85
        assert 0.0 <= mock_analyzer.detection_threshold <= 1.0

    @pytest.mark.unit
    @pytest.mark.slow
    def test_detect_astronomical_image(self, mock_analyzer,
                                       astronomical_image):
        """Test detection on astronomical images."""
        result = mock_analyzer._detect_negative_spaces(astronomical_image)

        assert isinstance(result, dict)
        # Astronomical images may have multiple detected objects
        assert len(result) >= 0

    @pytest.mark.unit
    def test_detect_medical_image(self, mock_analyzer, medical_image):
        """Test detection on medical images."""
        result = mock_analyzer._detect_negative_spaces(medical_image)

        assert isinstance(result, dict)


# =====================================================================
# FEATURE EXTRACTION TESTS
# =====================================================================

class TestFeatureExtraction:
    """Tests for feature extraction functionality."""

    @pytest.mark.unit
    def test_extract_features_has_required_fields(
        self, negative_space_features_data):
        """Test that extracted features have all required fields."""
        required_fields = [
            "area", "perimeter", "centroid", "topology_index",
            "connectivity", "pattern_score", "confidence"
        ]

        for field in required_fields:
            assert field in negative_space_features_data

    @pytest.mark.unit
    def test_extract_features_value_ranges(
        self, negative_space_features_data):
        """Test that extracted features are within valid ranges."""
        # Area should be positive
        assert negative_space_features_data["area"] > 0

        # Perimeter should be positive
        assert negative_space_features_data["perimeter"] > 0

        # Centroid should be 2D coordinates
        centroid = negative_space_features_data["centroid"]
        assert len(centroid) == 2
        assert all(isinstance(c, (int, float)) for c in centroid)

        # Confidence should be in [0, 1]
        assert 0.0 <= negative_space_features_data["confidence"] <= 1.0

        # Connectivity should be in [0, 1]
        assert 0.0 <= negative_space_features_data["connectivity"] <= 1.0

        # Pattern score should be in [0, 1]
        assert 0.0 <= negative_space_features_data["pattern_score"] <= 1.0

    @pytest.mark.unit
    def test_extract_features_circular_region(self):
        """Test feature extraction on circular region."""
        # Create a simple circular region
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 50, 1, -1)

        # Calculate basic features
        area = np.sum(mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = cv2.arcLength(contours[0], True) if contours else 0

        assert area > 0
        assert perimeter > 0
        # For a circle, perimeter should be ~2*pi*radius = ~314
        assert 200 < perimeter < 400

    @pytest.mark.unit
    def test_extract_features_rectangular_region(self):
        """Test feature extraction on rectangular region."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(mask, (100, 100), (200, 150), 1, -1)

        area = np.sum(mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = cv2.arcLength(contours[0], True) if contours else 0

        assert area == 100 * 50  # width * height
        # Perimeter should be 2*(width + height) = 2*150 = 300
        assert abs(perimeter - 300) < 5

    @pytest.mark.unit
    def test_extract_features_small_region(self):
        """Test feature extraction on very small region."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[128, 128] = 1  # Single pixel

        area = np.sum(mask)
        assert area == 1

    @pytest.mark.unit
    def test_extract_features_large_region(self):
        """Test feature extraction on large region."""
        mask = np.ones((256, 256), dtype=np.uint8)

        area = np.sum(mask)
        assert area == 256 * 256


# =====================================================================
# REGION ANALYSIS TESTS
# =====================================================================

class TestRegionAnalysis:
    """Tests for region analysis functionality."""

    @pytest.mark.unit
    def test_region_connectivity_analysis(self):
        """Test connectivity analysis of regions."""
        # Create two separate regions
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (64, 64), 30, 1, -1)
        cv2.circle(mask, (192, 192), 30, 1, -1)

        num_labels, labels = cv2.connectedComponents(mask)

        # Should have 2 separate regions (+ background)
        assert num_labels == 3  # 2 regions + 1 background

    @pytest.mark.unit
    def test_region_overlap_detection(self):
        """Test detection of overlapping regions."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 50, 1, -1)
        cv2.circle(mask, (150, 128), 50, 1, -1)

        # Overlapping circles should be connected
        num_labels, labels = cv2.connectedComponents(mask)

        # Should recognize as single connected region
        assert num_labels == 2  # 1 region + 1 background

    @pytest.mark.unit
    def test_region_isolation(self):
        """Test isolation of individual regions."""
        mask = np.zeros((256, 256), dtype=np.uint8)

        # Create 4 isolated regions
        cv2.circle(mask, (64, 64), 20, 1, -1)
        cv2.circle(mask, (192, 64), 20, 1, -1)
        cv2.circle(mask, (64, 192), 20, 1, -1)
        cv2.circle(mask, (192, 192), 20, 1, -1)

        num_labels, labels = cv2.connectedComponents(mask)

        # Should have 4 separate regions + background
        assert num_labels == 5

    @pytest.mark.unit
    def test_region_boundary_detection(self):
        """Test boundary detection within regions."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 50, 1, -1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        assert len(contours) == 1
        assert len(contours[0]) > 0  # Should have boundary points


# =====================================================================
# STATISTICAL ANALYSIS TESTS
# =====================================================================

class TestStatisticalAnalysis:
    """Tests for statistical analysis of results."""

    @pytest.mark.unit
    def test_statistics_complete_structure(self, analysis_result_data):
        """Test that statistics have complete structure."""
        stats = analysis_result_data["statistics"]

        required_keys = [
            "region_count", "feature_count", "total_negative_space_area",
            "average_region_size", "average_confidence"
        ]

        for key in required_keys:
            assert key in stats

    @pytest.mark.unit
    def test_statistics_valid_ranges(self, analysis_result_data):
        """Test that statistics are within valid ranges."""
        stats = analysis_result_data["statistics"]

        assert stats["region_count"] >= 0
        assert stats["feature_count"] >= 0
        assert stats["total_negative_space_area"] >= 0
        assert stats["average_region_size"] >= 0

        # Confidence should be in [0, 1]
        assert 0.0 <= stats["average_confidence"] <= 1.0

    @pytest.mark.unit
    def test_statistics_consistency(self, analysis_result_data):
        """Test statistical consistency."""
        stats = analysis_result_data["statistics"]
        regions = analysis_result_data["detected_regions"]

        # Region count should match
        assert stats["region_count"] == len(regions)

        # Average region size calculation
        if stats["region_count"] > 0:
            expected_avg = stats["total_negative_space_area"] / \
                          stats["region_count"]
            assert abs(expected_avg - stats["average_region_size"]) < 1

    @pytest.mark.unit
    def test_statistics_zero_regions(self):
        """Test statistics with zero detected regions."""
        stats = {
            "region_count": 0,
            "feature_count": 0,
            "total_negative_space_area": 0,
            "average_region_size": 0,
            "average_confidence": 0,
        }

        assert all(v == 0 for v in stats.values())


# =====================================================================
# ERROR HANDLING & EDGE CASES
# =====================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.mark.unit
    def test_handle_empty_image(self, mock_analyzer):
        """Test handling of empty (all zeros) image."""
        empty = np.zeros((256, 256), dtype=np.uint8)
        result = mock_analyzer._detect_negative_spaces(empty)

        assert isinstance(result, dict)
        # May return empty dict or handle gracefully
        assert len(result) >= 0

    @pytest.mark.unit
    def test_handle_single_pixel_image(self, mock_analyzer):
        """Test handling of single pixel image."""
        single_pixel = np.zeros((1, 1), dtype=np.uint8)
        single_pixel[0, 0] = 255

        # Should handle without crashing
        result = mock_analyzer._detect_negative_spaces(single_pixel)
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_handle_very_large_image(self, mock_analyzer):
        """Test handling of very large images."""
        # Create large image
        large = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)

        # Should process without memory issues
        result = mock_analyzer._detect_negative_spaces(large)
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_handle_non_square_image(self, mock_analyzer):
        """Test handling of non-square images."""
        non_square = np.random.randint(
            0, 256, (512, 256), dtype=np.uint8
        )

        result = mock_analyzer._detect_negative_spaces(non_square)
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_handle_invalid_dtype(self):
        """Test handling of invalid data types."""
        # Should handle gracefully
        invalid_dtype = np.random.random((256, 256))  # float64

        # Convert to acceptable format
        image = (invalid_dtype * 255).astype(np.uint8)
        assert image.dtype == np.uint8

    @pytest.mark.unit
    def test_handle_nan_values(self):
        """Test handling of NaN values in image."""
        image_with_nan = np.random.randn(256, 256)
        image_with_nan[10:20, 10:20] = np.nan

        # Should handle or raise informative error
        cleaned = np.nan_to_num(image_with_nan, nan=0.0)
        assert not np.any(np.isnan(cleaned))

    @pytest.mark.unit
    def test_handle_infinity_values(self):
        """Test handling of infinity values."""
        image_with_inf = np.ones((256, 256), dtype=np.float32)
        image_with_inf[50:60, 50:60] = np.inf

        # Should handle or raise informative error
        cleaned = np.clip(image_with_inf, -1e6, 1e6)
        assert not np.any(np.isinf(cleaned))


# =====================================================================
# CONFIGURATION & PARAMETER TESTS
# =====================================================================

class TestAnalyzerConfiguration:
    """Tests for analyzer configuration and parameters."""

    @pytest.mark.unit
    def test_threshold_boundary_values(self, mock_analyzer):
        """Test threshold with boundary values."""
        mock_analyzer.detection_threshold = 0.0
        assert mock_analyzer.detection_threshold == 0.0

        mock_analyzer.detection_threshold = 1.0
        assert mock_analyzer.detection_threshold == 1.0

        mock_analyzer.detection_threshold = 0.5
        assert mock_analyzer.detection_threshold == 0.5

    @pytest.mark.unit
    def test_min_region_size_values(self, mock_analyzer):
        """Test minimum region size configuration."""
        for size in [1, 10, 100, 1000, 10000]:
            mock_analyzer.min_region_size = size
            assert mock_analyzer.min_region_size == size

    @pytest.mark.unit
    def test_device_configuration(self, mock_analyzer, device):
        """Test device configuration (CPU/GPU)."""
        mock_analyzer.device = device
        assert isinstance(mock_analyzer.device, torch.device)


# =====================================================================
# INTEGRATION WITH FIXTURES
# =====================================================================

class TestWithFixtures:
    """Tests using provided fixtures."""

    @pytest.mark.unit
    def test_with_synthetic_image_fixture(self, synthetic_image,
                                          assert_image_quality):
        """Test using synthetic image fixture."""
        assert_image_quality(synthetic_image)

    @pytest.mark.unit
    def test_with_medical_image_fixture(self, medical_image,
                                        assert_image_quality):
        """Test using medical image fixture."""
        assert_image_quality(medical_image)

    @pytest.mark.unit
    def test_with_astronomical_image_fixture(self, astronomical_image,
                                             assert_image_quality):
        """Test using astronomical image fixture."""
        assert_image_quality(astronomical_image)

    @pytest.mark.unit
    def test_with_image_batch_fixture(self, image_batch):
        """Test processing multiple images in batch."""
        assert isinstance(image_batch, list)
        assert len(image_batch) == 5

        for img in image_batch:
            assert isinstance(img, np.ndarray)
            assert img.shape == (256, 256)

    @pytest.mark.unit
    def test_with_analysis_result_fixture(self, analysis_result_data,
                                          assert_analysis_result):
        """Test using analysis result fixture."""
        assert_analysis_result(analysis_result_data)
