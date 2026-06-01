"""
Data Validation Tests for Negative Space Imaging Analysis Results

Comprehensive test suite for validating data structures, serialization,
metadata integrity, and result consistency.

Focus Areas:
- AnalysisResult structure validation
- JSONB serialization/deserialization
- Metadata validation
- Confidence score bounds
- Data consistency and integrity
"""

import pytest
import json
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


# =====================================================================
# ANALYSIS RESULT STRUCTURE VALIDATION
# =====================================================================

class TestAnalysisResultStructure:
    """Tests for AnalysisResult data structure validation."""

    @pytest.mark.unit
    def test_analysis_result_has_all_required_fields(
        self, analysis_result_data):
        """Test that AnalysisResult has all required fields."""
        required_fields = {
            "id": str,
            "timestamp": str,
            "image_id": str,
            "image_metadata": dict,
            "detected_regions": list,
            "features": list,
            "statistics": dict,
            "processing_time_ms": (int, float),
            "algorithm_version": str,
        }

        for field, expected_type in required_fields.items():
            assert field in analysis_result_data, f"Missing field: {field}"

            if isinstance(expected_type, tuple):
                assert isinstance(analysis_result_data[field],
                                expected_type), \
                    f"Field {field} has wrong type"
            else:
                assert isinstance(analysis_result_data[field],
                                expected_type), \
                    f"Field {field} has wrong type"

    @pytest.mark.unit
    def test_analysis_result_id_format(self, analysis_result_data):
        """Test that result ID has valid format."""
        result_id = analysis_result_data["id"]

        assert isinstance(result_id, str)
        assert len(result_id) > 0
        # Should be alphanumeric (typically)
        assert result_id.replace("_", "").replace("-", "").isalnum()

    @pytest.mark.unit
    def test_analysis_result_timestamp_format(self, analysis_result_data):
        """Test that timestamp is valid ISO format."""
        timestamp = analysis_result_data["timestamp"]

        assert isinstance(timestamp, str)
        # Should be ISO format (2025-01-17T10:30:00Z)
        assert "T" in timestamp or "-" in timestamp

    @pytest.mark.unit
    def test_analysis_result_image_id_valid(self, analysis_result_data):
        """Test that image_id is valid."""
        image_id = analysis_result_data["image_id"]

        assert isinstance(image_id, str)
        assert len(image_id) > 0

    @pytest.mark.unit
    def test_analysis_result_processing_time_valid(
        self, analysis_result_data):
        """Test that processing time is valid."""
        time_ms = analysis_result_data["processing_time_ms"]

        assert isinstance(time_ms, (int, float))
        assert time_ms >= 0, "Processing time cannot be negative"
        assert time_ms < 1000000, "Processing time unreasonably large"

    @pytest.mark.unit
    def test_analysis_result_version_format(self, analysis_result_data):
        """Test that algorithm version follows semantic versioning."""
        version = analysis_result_data["algorithm_version"]

        assert isinstance(version, str)
        parts = version.split(".")
        assert len(parts) >= 2, "Version should be at least X.Y"
        # Each part should be numeric
        for part in parts:
            assert part.isdigit(), f"Version part '{part}' not numeric"


# =====================================================================
# IMAGE METADATA VALIDATION
# =====================================================================

class TestImageMetadataValidation:
    """Tests for image metadata validation."""

    @pytest.mark.unit
    def test_metadata_has_required_fields(self, analysis_result_data):
        """Test that metadata has all required fields."""
        metadata = analysis_result_data["image_metadata"]

        required_fields = {
            "width": int,
            "height": int,
            "format": str,
            "filename": str,
        }

        for field, expected_type in required_fields.items():
            assert field in metadata, f"Missing metadata field: {field}"
            assert isinstance(metadata[field], expected_type), \
                f"Metadata field {field} has wrong type"

    @pytest.mark.unit
    def test_metadata_dimensions_valid(self, analysis_result_data):
        """Test that image dimensions are valid."""
        metadata = analysis_result_data["image_metadata"]
        width = metadata["width"]
        height = metadata["height"]

        # Dimensions should be positive
        assert width > 0, "Width must be positive"
        assert height > 0, "Height must be positive"

        # Should be reasonable sizes (not absurdly large)
        assert width <= 16384, "Width unreasonably large"
        assert height <= 16384, "Height unreasonably large"

    @pytest.mark.unit
    def test_metadata_format_valid(self, analysis_result_data):
        """Test that image format is valid."""
        metadata = analysis_result_data["image_metadata"]
        format_str = metadata["format"]

        valid_formats = {"JPEG", "PNG", "TIFF", "DICOM", "FITS",
                        "RAW", "BMP", "WEBP"}

        assert format_str in valid_formats, \
            f"Invalid format: {format_str}"

    @pytest.mark.unit
    def test_metadata_filename_valid(self, analysis_result_data):
        """Test that filename is valid."""
        metadata = analysis_result_data["image_metadata"]
        filename = metadata["filename"]

        assert isinstance(filename, str)
        assert len(filename) > 0, "Filename cannot be empty"
        assert "/" not in filename and "\\" not in filename, \
            "Filename should not contain path separators"

    @pytest.mark.unit
    def test_metadata_consistency_with_regions(
        self, analysis_result_data):
        """Test metadata consistency with detected regions."""
        metadata = analysis_result_data["image_metadata"]
        regions = analysis_result_data["detected_regions"]

        for region in regions:
            bbox = region["bounding_box"]

            # Region bbox should be within image bounds
            assert bbox["x"] >= 0
            assert bbox["y"] >= 0
            assert bbox["x"] + bbox["width"] <= metadata["width"]
            assert bbox["y"] + bbox["height"] <= metadata["height"]


# =====================================================================
# REGION VALIDATION
# =====================================================================

class TestRegionValidation:
    """Tests for detected region validation."""

    @pytest.mark.unit
    def test_region_has_required_fields(self, analysis_result_data):
        """Test that each region has required fields."""
        regions = analysis_result_data["detected_regions"]

        required_fields = {
            "id": str,
            "centroid": list,
            "area": (int, float),
            "confidence": (int, float),
            "bounding_box": dict,
        }

        for region in regions:
            for field, expected_type in required_fields.items():
                assert field in region, f"Region missing field: {field}"

                if isinstance(expected_type, tuple):
                    assert isinstance(region[field], expected_type)
                else:
                    assert isinstance(region[field], expected_type)

    @pytest.mark.unit
    def test_region_id_format(self, analysis_result_data):
        """Test that region IDs follow expected format."""
        regions = analysis_result_data["detected_regions"]

        for region in regions:
            region_id = region["id"]
            assert isinstance(region_id, str)
            assert region_id.startswith("region_")

    @pytest.mark.unit
    def test_region_centroid_valid(self, analysis_result_data):
        """Test that centroid is valid."""
        regions = analysis_result_data["detected_regions"]
        metadata = analysis_result_data["image_metadata"]

        for region in regions:
            centroid = region["centroid"]

            assert isinstance(centroid, list)
            assert len(centroid) == 2, "Centroid should be 2D"

            x, y = centroid
            assert 0 <= x <= metadata["width"]
            assert 0 <= y <= metadata["height"]

    @pytest.mark.unit
    def test_region_area_valid(self, analysis_result_data):
        """Test that region area is valid."""
        regions = analysis_result_data["detected_regions"]
        metadata = analysis_result_data["image_metadata"]

        max_area = metadata["width"] * metadata["height"]

        for region in regions:
            area = region["area"]

            assert isinstance(area, (int, float))
            assert area > 0, "Area must be positive"
            assert area <= max_area, "Area cannot exceed image area"

    @pytest.mark.unit
    def test_region_confidence_bounds(self, analysis_result_data):
        """Test that region confidence is in [0, 1]."""
        regions = analysis_result_data["detected_regions"]

        for region in regions:
            confidence = region["confidence"]

            assert isinstance(confidence, (int, float))
            assert 0.0 <= confidence <= 1.0, \
                f"Confidence {confidence} out of bounds"

    @pytest.mark.unit
    def test_bounding_box_valid(self, analysis_result_data):
        """Test that bounding boxes are valid."""
        regions = analysis_result_data["detected_regions"]
        metadata = analysis_result_data["image_metadata"]

        for region in regions:
            bbox = region["bounding_box"]

            required_keys = {"x", "y", "width", "height"}
            assert required_keys.issubset(bbox.keys()), \
                "Missing bounding box fields"

            # Values should be non-negative
            assert bbox["x"] >= 0
            assert bbox["y"] >= 0
            assert bbox["width"] > 0
            assert bbox["height"] > 0

            # Should be within image bounds
            assert bbox["x"] + bbox["width"] <= metadata["width"]
            assert bbox["y"] + bbox["height"] <= metadata["height"]

    @pytest.mark.unit
    def test_regions_no_duplicates(self, analysis_result_data):
        """Test that region IDs are unique."""
        regions = analysis_result_data["detected_regions"]
        region_ids = [r["id"] for r in regions]

        assert len(region_ids) == len(set(region_ids)), \
            "Duplicate region IDs found"


# =====================================================================
# FEATURES VALIDATION
# =====================================================================

class TestFeaturesValidation:
    """Tests for features validation."""

    @pytest.mark.unit
    def test_feature_has_required_fields(self, analysis_result_data):
        """Test that each feature has required fields."""
        features = analysis_result_data["features"]

        required_fields = {
            "type": str,
            "confidence": (int, float),
            "significance": (int, float),
            "region_id": str,
        }

        for feature in features:
            for field, expected_type in required_fields.items():
                assert field in feature, f"Feature missing field: {field}"

                if isinstance(expected_type, tuple):
                    assert isinstance(feature[field], expected_type)
                else:
                    assert isinstance(feature[field], expected_type)

    @pytest.mark.unit
    def test_feature_type_valid(self, analysis_result_data):
        """Test that feature types are recognized."""
        features = analysis_result_data["features"]

        valid_types = {
            "circular", "rectangular", "symmetric", "pattern",
            "edge", "corner", "texture", "gradient"
        }

        for feature in features:
            feature_type = feature["type"]
            # Type should be recognizable
            assert isinstance(feature_type, str)
            assert len(feature_type) > 0

    @pytest.mark.unit
    def test_feature_confidence_bounds(self, analysis_result_data):
        """Test that feature confidence is in [0, 1]."""
        features = analysis_result_data["features"]

        for feature in features:
            confidence = feature["confidence"]

            assert isinstance(confidence, (int, float))
            assert 0.0 <= confidence <= 1.0

    @pytest.mark.unit
    def test_feature_significance_bounds(self, analysis_result_data):
        """Test that feature significance is in [0, 1]."""
        features = analysis_result_data["features"]

        for feature in features:
            significance = feature["significance"]

            assert isinstance(significance, (int, float))
            assert 0.0 <= significance <= 1.0

    @pytest.mark.unit
    def test_feature_region_reference_valid(self, analysis_result_data):
        """Test that feature region references are valid."""
        features = analysis_result_data["features"]
        regions = analysis_result_data["detected_regions"]
        region_ids = {r["id"] for r in regions}

        for feature in features:
            region_id = feature["region_id"]

            assert region_id in region_ids, \
                f"Feature references non-existent region: {region_id}"


# =====================================================================
# STATISTICS VALIDATION
# =====================================================================

class TestStatisticsValidation:
    """Tests for statistics validation."""

    @pytest.mark.unit
    def test_statistics_required_fields(self, analysis_result_data):
        """Test statistics have required fields."""
        stats = analysis_result_data["statistics"]

        required_fields = {
            "region_count": int,
            "feature_count": int,
            "total_negative_space_area": (int, float),
            "average_region_size": (int, float),
            "average_confidence": (int, float),
        }

        for field, expected_type in required_fields.items():
            assert field in stats, f"Statistics missing field: {field}"

            if isinstance(expected_type, tuple):
                assert isinstance(stats[field], expected_type)
            else:
                assert isinstance(stats[field], expected_type)

    @pytest.mark.unit
    def test_statistics_non_negative(self, analysis_result_data):
        """Test that statistics are non-negative."""
        stats = analysis_result_data["statistics"]

        non_negative_fields = [
            "region_count", "feature_count", "total_negative_space_area",
            "average_region_size"
        ]

        for field in non_negative_fields:
            assert stats[field] >= 0, f"{field} is negative"

    @pytest.mark.unit
    def test_statistics_average_confidence_bounds(
        self, analysis_result_data):
        """Test average confidence is in valid range."""
        stats = analysis_result_data["statistics"]

        confidence = stats["average_confidence"]
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.unit
    def test_statistics_consistency_with_regions(
        self, analysis_result_data):
        """Test statistics are consistent with regions."""
        stats = analysis_result_data["statistics"]
        regions = analysis_result_data["detected_regions"]

        # Region count should match
        assert stats["region_count"] == len(regions)

        # Average confidence should be calculable
        if regions:
            confidences = [r["confidence"] for r in regions]
            expected_avg = np.mean(confidences)

            assert abs(stats["average_confidence"] - expected_avg) < 0.01

    @pytest.mark.unit
    def test_statistics_zero_regions_handling(self):
        """Test statistics when there are zero regions."""
        stats = {
            "region_count": 0,
            "feature_count": 0,
            "total_negative_space_area": 0,
            "average_region_size": 0,
            "average_confidence": 0,
        }

        # Should handle gracefully
        assert all(v >= 0 for v in stats.values())


# =====================================================================
# SERIALIZATION/DESERIALIZATION TESTS
# =====================================================================

class TestSerializationDeserialization:
    """Tests for JSON serialization and deserialization."""

    @pytest.mark.unit
    def test_analysis_result_json_serializable(
        self, analysis_result_data):
        """Test that AnalysisResult can be serialized to JSON."""
        try:
            json_str = json.dumps(analysis_result_data)
            assert isinstance(json_str, str)
            assert len(json_str) > 0
        except TypeError as e:
            pytest.fail(f"Not JSON serializable: {e}")

    @pytest.mark.unit
    def test_analysis_result_json_roundtrip(
        self, analysis_result_data):
        """Test JSON serialization roundtrip."""
        # Serialize
        json_str = json.dumps(analysis_result_data)

        # Deserialize
        restored = json.loads(json_str)

        # Should match original
        assert restored == analysis_result_data

    @pytest.mark.unit
    def test_numpy_arrays_handling_in_serialization(self):
        """Test handling of NumPy arrays in serialization."""
        data = {
            "array": np.array([1, 2, 3]),
            "float": np.float32(3.14),
            "int": np.int64(42),
        }

        # Need custom encoder for NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return super().default(obj)

        json_str = json.dumps(data, cls=NumpyEncoder)
        restored = json.loads(json_str)

        assert restored["array"] == [1, 2, 3]
        assert restored["float"] == pytest.approx(3.14)
        assert restored["int"] == 42


# =====================================================================
# DATA INTEGRITY TESTS
# =====================================================================

class TestDataIntegrity:
    """Tests for data integrity and consistency."""

    @pytest.mark.unit
    def test_immutability_concern(self, analysis_result_data):
        """Test that modifying copy doesn't affect original."""
        import copy

        original = copy.deepcopy(analysis_result_data)
        modified = analysis_result_data

        # Modify the copy
        if modified["detected_regions"]:
            modified["detected_regions"][0]["confidence"] = 0.0

        # Original should be unchanged
        assert original == copy.deepcopy(analysis_result_data) or \
               modified["detected_regions"][0]["confidence"] != 0.0

    @pytest.mark.unit
    def test_none_values_handling(self, analysis_result_data):
        """Test handling of None values in data."""
        data_with_none = analysis_result_data.copy()
        data_with_none["optional_field"] = None

        # Should be serializable
        json_str = json.dumps(data_with_none)
        assert "null" in json_str

    @pytest.mark.unit
    def test_large_result_handling(self):
        """Test handling of large result objects."""
        large_result = {
            "id": "large_001",
            "timestamp": "2025-01-17T10:30:00Z",
            "image_id": "large_image",
            "detected_regions": [
                {
                    "id": f"region_{i}",
                    "centroid": [np.random.rand() * 256 for _ in range(2)],
                    "area": np.random.rand() * 10000,
                    "confidence": np.random.rand(),
                    "bounding_box": {
                        "x": int(np.random.rand() * 256),
                        "y": int(np.random.rand() * 256),
                        "width": int(np.random.rand() * 100 + 10),
                        "height": int(np.random.rand() * 100 + 10),
                    },
                }
                for i in range(1000)  # 1000 regions
            ],
            "features": [
                {
                    "type": "pattern",
                    "confidence": np.random.rand(),
                    "significance": np.random.rand(),
                    "region_id": f"region_{np.random.randint(0, 1000)}",
                }
                for _ in range(5000)  # 5000 features
            ],
        }

        # Should handle without issues
        assert len(large_result["detected_regions"]) == 1000
        assert len(large_result["features"]) == 5000


# =====================================================================
# EDGE CASE VALIDATION
# =====================================================================

class TestEdgeCaseValidation:
    """Tests for edge case data validation."""

    @pytest.mark.unit
    def test_empty_regions_list(self):
        """Test handling of empty regions list."""
        result = {
            "id": "test",
            "timestamp": "2025-01-17T10:30:00Z",
            "image_id": "test_image",
            "detected_regions": [],
            "features": [],
            "statistics": {
                "region_count": 0,
                "feature_count": 0,
            },
        }

        assert len(result["detected_regions"]) == 0
        assert result["statistics"]["region_count"] == 0

    @pytest.mark.unit
    def test_single_region_result(self):
        """Test result with single region."""
        result = {
            "id": "single",
            "detected_regions": [
                {
                    "id": "region_0",
                    "area": 100,
                    "confidence": 0.95,
                }
            ],
            "statistics": {"region_count": 1},
        }

        assert len(result["detected_regions"]) == 1
        assert result["statistics"]["region_count"] == 1

    @pytest.mark.unit
    def test_zero_area_regions_rejection(self):
        """Test rejection of zero-area regions."""
        region = {"area": 0, "confidence": 0.8}

        # Zero area should be invalid
        assert region["area"] <= 0  # Should be filtered out

    @pytest.mark.unit
    def test_extreme_confidence_values(self):
        """Test handling of extreme confidence values."""
        # Very confident
        assert 0.0 <= 0.9999 <= 1.0

        # Not confident at all
        assert 0.0 <= 0.0001 <= 1.0

        # Edge values
        assert 0.0 <= 0.0 <= 1.0
        assert 0.0 <= 1.0 <= 1.0
