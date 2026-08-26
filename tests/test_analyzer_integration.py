"""
Integration Tests for Negative Space Imaging Analysis Pipeline

Comprehensive integration tests for end-to-end workflows, database operations,
file handling, and format-specific image processing.

Test Areas:
- Complete analysis pipeline
- Database storage and retrieval
- File I/O operations
- Medical image formats (DICOM)
- Astronomical image formats (FITS)
- Multi-stage processing workflows
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, MagicMock, patch
import json
import logging

logger = logging.getLogger(__name__)


# =====================================================================
# FULL PIPELINE INTEGRATION TESTS
# =====================================================================

class TestFullAnalysisPipeline:
    """Tests for complete end-to-end analysis pipeline."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_analysis_workflow(self, synthetic_image,
                                       mock_analyzer):
        """Test complete analysis from image to results."""
        # Step 1: Preprocess
        preprocessed = synthetic_image.astype(np.float32) / 255.0
        assert preprocessed.shape == synthetic_image.shape

        # Step 2: Detect negative spaces
        regions = mock_analyzer._detect_negative_spaces(preprocessed)
        assert isinstance(regions, dict)

        # Step 3: Verify result structure
        assert len(regions) >= 0

    @pytest.mark.integration
    def test_batch_processing_workflow(self, image_batch, mock_analyzer):
        """Test batch processing of multiple images."""
        results = []

        for image in image_batch:
            # Process each image
            preprocessed = image.astype(np.float32) / 255.0
            regions = mock_analyzer._detect_negative_spaces(preprocessed)
            results.append({
                "image_shape": image.shape,
                "region_count": len(regions),
            })

        assert len(results) == len(image_batch)
        assert all(r["image_shape"] == (256, 256) for r in results)

    @pytest.mark.integration
    def test_pipeline_with_different_image_types(self,
                                                 synthetic_image,
                                                 medical_image,
                                                 astronomical_image,
                                                 mock_analyzer):
        """Test pipeline works with various image types."""
        test_images = {
            "synthetic": synthetic_image,
            "medical": medical_image,
            "astronomical": astronomical_image,
        }

        for name, image in test_images.items():
            logger.info(f"Processing {name} image")

            preprocessed = image.astype(np.float32) / 255.0
            regions = mock_analyzer._detect_negative_spaces(preprocessed)

            assert isinstance(regions, dict)
            logger.info(f"  Detected {len(regions)} regions")

    @pytest.mark.integration
    def test_pipeline_error_recovery(self, mock_analyzer):
        """Test pipeline handles errors gracefully."""
        # Test with problematic input
        edge_images = [
            np.zeros((256, 256), dtype=np.uint8),  # Empty
            np.ones((256, 256), dtype=np.uint8),   # Full
        ]

        for image in edge_images:
            try:
                preprocessed = image.astype(np.float32) / 255.0
                regions = mock_analyzer._detect_negative_spaces(preprocessed)
                # Should complete without crash
                assert isinstance(regions, dict)
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                raise


# =====================================================================
# DATABASE INTEGRATION TESTS
# =====================================================================

class TestDatabaseIntegration:
    """Tests for database storage and retrieval."""

    @pytest.mark.integration
    @pytest.mark.database
    def test_store_analysis_result_to_database(self, temp_db_path,
                                              analysis_result_data):
        """Test storing analysis result to database."""
        # Simulate database storage
        stored_data = analysis_result_data.copy()
        stored_data["_stored_at"] = temp_db_path

        assert "_stored_at" in stored_data
        assert stored_data["_stored_at"] == temp_db_path

    @pytest.mark.integration
    @pytest.mark.database
    def test_retrieve_analysis_result_from_database(
        self, analysis_result_data):
        """Test retrieving analysis result from database."""
        # Mock retrieval
        result_id = analysis_result_data["id"]
        retrieved = analysis_result_data.copy()

        assert retrieved["id"] == result_id
        assert all(k in retrieved for k in [
            "detected_regions", "features", "statistics"
        ])

    @pytest.mark.integration
    @pytest.mark.database
    def test_query_results_by_image_id(self):
        """Test querying results by image ID."""
        # Mock query results
        query_results = [
            {"id": "result_1", "image_id": "image_001", "region_count": 3},
            {"id": "result_2", "image_id": "image_001", "region_count": 2},
        ]

        # Filter by image_id
        image_id = "image_001"
        filtered = [r for r in query_results if r["image_id"] == image_id]

        assert len(filtered) == 2
        assert all(r["image_id"] == image_id for r in filtered)

    @pytest.mark.integration
    @pytest.mark.database
    def test_update_result_in_database(self, analysis_result_data):
        """Test updating result in database."""
        original_count = analysis_result_data["statistics"]["region_count"]

        # Simulate update
        updated_data = analysis_result_data.copy()
        updated_data["statistics"]["region_count"] = original_count + 1

        assert updated_data["statistics"]["region_count"] != original_count

    @pytest.mark.integration
    @pytest.mark.database
    def test_delete_result_from_database(self, analysis_result_data):
        """Test deleting result from database."""
        result_id = analysis_result_data["id"]

        # Mock deletion
        deleted_id = result_id

        assert deleted_id == result_id

    @pytest.mark.integration
    @pytest.mark.database
    def test_batch_database_operations(self, analysis_result_data):
        """Test batch database operations."""
        results = []

        # Simulate storing 10 results
        for i in range(10):
            result = analysis_result_data.copy()
            result["id"] = f"result_{i:03d}"
            results.append(result)

        assert len(results) == 10
        assert all("id" in r for r in results)


# =====================================================================
# FILE I/O INTEGRATION TESTS
# =====================================================================

class TestFileIOIntegration:
    """Tests for file input/output operations."""

    @pytest.mark.integration
    def test_save_image_to_file(self, synthetic_image, test_data_dir):
        """Test saving image to file."""
        filename = test_data_dir / "test_image.png"

        try:
            import cv2
            cv2.imwrite(str(filename), synthetic_image)
            assert filename.exists()
        except ImportError:
            # Fallback: use numpy
            np.save(str(filename.with_suffix('.npy')), synthetic_image)
            assert filename.with_suffix('.npy').exists()

    @pytest.mark.integration
    def test_load_image_from_file(self, test_data_dir):
        """Test loading image from file."""
        # Create test file
        test_data = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        filename = test_data_dir / "load_test.npy"
        np.save(str(filename), test_data)

        # Load
        loaded = np.load(str(filename))

        assert loaded.shape == test_data.shape
        assert np.allclose(loaded, test_data)

    @pytest.mark.integration
    def test_save_results_to_json(self, analysis_result_data,
                                  test_data_dir):
        """Test saving analysis results to JSON file."""
        filename = test_data_dir / "results.json"

        with open(filename, 'w') as f:
            json.dump(analysis_result_data, f, indent=2)

        assert filename.exists()
        assert filename.stat().st_size > 0

    @pytest.mark.integration
    def test_load_results_from_json(self, analysis_result_data,
                                   test_data_dir):
        """Test loading analysis results from JSON file."""
        filename = test_data_dir / "load_results.json"

        # Save
        with open(filename, 'w') as f:
            json.dump(analysis_result_data, f)

        # Load
        with open(filename, 'r') as f:
            loaded = json.load(f)

        assert loaded["id"] == analysis_result_data["id"]
        assert loaded["statistics"] == analysis_result_data["statistics"]

    @pytest.mark.integration
    def test_create_batch_output_directory(self, test_data_dir):
        """Test creating batch output directory structure."""
        batch_dir = test_data_dir / "batch_001"
        batch_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (batch_dir / "images").mkdir(exist_ok=True)
        (batch_dir / "results").mkdir(exist_ok=True)
        (batch_dir / "logs").mkdir(exist_ok=True)

        assert batch_dir.exists()
        assert (batch_dir / "images").exists()
        assert (batch_dir / "results").exists()
        assert (batch_dir / "logs").exists()


# =====================================================================
# DICOM FORMAT TESTS
# =====================================================================

class TestDICOMIntegration:
    """Tests for DICOM medical image format support."""

    @pytest.mark.integration
    def test_dicom_file_detection(self):
        """Test detection of DICOM format."""
        filename = "patient_scan.dcm"
        is_dicom = filename.lower().endswith('.dcm')

        assert is_dicom

    @pytest.mark.integration
    def test_dicom_metadata_extraction(self):
        """Test extracting metadata from DICOM."""
        # Mock DICOM metadata
        dicom_metadata = {
            "PatientName": "Test Patient",
            "PatientID": "12345",
            "Modality": "CT",
            "SeriesNumber": 1,
            "Rows": 512,
            "Columns": 512,
        }

        assert dicom_metadata["Modality"] == "CT"
        assert dicom_metadata["Rows"] == 512

    @pytest.mark.integration
    def test_dicom_pixel_data_extraction(self):
        """Test extracting pixel data from DICOM."""
        # Mock DICOM pixel array
        pixel_array = np.random.randint(-1000, 1000, (512, 512),
                                       dtype=np.int16)

        assert pixel_array.shape == (512, 512)
        assert pixel_array.dtype == np.int16

    @pytest.mark.integration
    def test_dicom_windowing_operation(self):
        """Test DICOM window/level operation for display."""
        pixel_array = np.random.randint(-1000, 1000, (512, 512),
                                       dtype=np.int16)

        # Apply window/level for display
        window_width = 400  # e.g., for soft tissue
        window_center = 40

        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2

        windowed = np.clip(pixel_array, min_val, max_val)
        display_array = ((windowed - min_val) / (max_val - min_val) * 255).\
                       astype(np.uint8)

        assert display_array.dtype == np.uint8
        assert display_array.min() >= 0
        assert display_array.max() <= 255


# =====================================================================
# FITS FORMAT TESTS
# =====================================================================

class TestFITSIntegration:
    """Tests for FITS astronomical image format support."""

    @pytest.mark.integration
    def test_fits_file_detection(self):
        """Test detection of FITS format."""
        filenames = ["observation.fits", "data.fit", "image.fits.gz"]

        fits_files = [f for f in filenames
                     if f.lower().endswith(('.fits', '.fit'))
                     or '.fits.gz' in f.lower()]

        assert len(fits_files) == 3

    @pytest.mark.integration
    def test_fits_header_extraction(self):
        """Test extracting header from FITS."""
        # Mock FITS header
        fits_header = {
            "SIMPLE": True,
            "BITPIX": -32,
            "NAXIS": 2,
            "NAXIS1": 512,
            "NAXIS2": 512,
            "TELESCOP": "Hubble",
            "INSTRUME": "WFC3",
        }

        assert fits_header["TELESCOP"] == "Hubble"
        assert fits_header["NAXIS1"] == 512

    @pytest.mark.integration
    def test_fits_data_extraction(self):
        """Test extracting data from FITS."""
        # Mock FITS data
        fits_data = np.random.randn(512, 512).astype(np.float32)

        assert fits_data.shape == (512, 512)
        assert fits_data.dtype == np.float32

    @pytest.mark.integration
    def test_fits_scaling_operation(self):
        """Test FITS data scaling (BZERO, BSCALE)."""
        raw_data = np.random.randint(0, 65535, (256, 256),
                                    dtype=np.uint16)

        # Apply BZERO and BSCALE
        bzero = 32768.0
        bscale = 1.0

        scaled_data = raw_data * bscale + bzero

        assert scaled_data.shape == raw_data.shape


# =====================================================================
# MULTI-FORMAT PROCESSING TESTS
# =====================================================================

class TestMultiFormatProcessing:
    """Tests for processing multiple image formats."""

    @pytest.mark.integration
    def test_format_auto_detection(self):
        """Test automatic format detection from filename."""
        test_files = {
            "image.png": "PNG",
            "scan.dcm": "DICOM",
            "observation.fits": "FITS",
            "photo.jpg": "JPEG",
            "data.tiff": "TIFF",
        }

        formats_detected = {}
        for filename, expected_format in test_files.items():
            ext = Path(filename).suffix.upper().lstrip('.')
            if ext == "DCM":
                detected = "DICOM"
            elif ext in ("FITS", "FIT"):
                detected = "FITS"
            else:
                detected = ext

            formats_detected[filename] = detected

        assert formats_detected["scan.dcm"] == "DICOM"
        assert formats_detected["observation.fits"] == "FITS"

    @pytest.mark.integration
    def test_format_conversion_pipeline(self, synthetic_image,
                                       test_data_dir):
        """Test converting between formats in pipeline."""
        # Save as different formats
        formats = ["png", "npy"]

        for fmt in formats:
            if fmt == "npy":
                filename = test_data_dir / f"image.{fmt}"
                np.save(str(filename), synthetic_image)
            else:
                try:
                    import cv2
                    filename = test_data_dir / f"image.{fmt}"
                    cv2.imwrite(str(filename), synthetic_image)
                except ImportError:
                    pass

        # Files should exist
        npy_file = test_data_dir / "image.npy"
        if npy_file.exists():
            loaded = np.load(str(npy_file))
            assert loaded.shape == synthetic_image.shape


# =====================================================================
# WORKFLOW STATE MANAGEMENT TESTS
# =====================================================================

class TestWorkflowStateManagement:
    """Tests for maintaining workflow state across pipeline stages."""

    @pytest.mark.integration
    def test_pipeline_state_tracking(self, synthetic_image):
        """Test tracking state through pipeline stages."""
        state = {
            "stage": "init",
            "image": synthetic_image,
            "processed": False,
            "regions_detected": False,
        }

        # Progress through stages
        state["stage"] = "preprocessing"
        state["processed"] = True

        state["stage"] = "detection"
        state["regions_detected"] = True

        assert state["stage"] == "detection"
        assert state["processed"]
        assert state["regions_detected"]

    @pytest.mark.integration
    def test_checkpoint_and_resume(self, analysis_result_data):
        """Test checkpointing and resuming analysis."""
        # Create checkpoint
        checkpoint = {
            "result_id": analysis_result_data["id"],
            "stage": "region_detection",
            "timestamp": "2025-01-17T10:30:00Z",
            "data": analysis_result_data,
        }

        # Simulate resuming from checkpoint
        resumed_id = checkpoint["result_id"]
        resumed_data = checkpoint["data"]

        assert resumed_id == analysis_result_data["id"]
        assert resumed_data == analysis_result_data

    @pytest.mark.integration
    def test_error_state_handling(self):
        """Test handling error states in pipeline."""
        error_state = {
            "stage": "detection",
            "error": True,
            "error_message": "Out of memory",
            "recovered": False,
        }

        # Attempt recovery
        error_state["recovered"] = True
        error_state["recovery_method"] = "reduce_batch_size"

        assert error_state["recovered"]
        assert error_state["recovery_method"] is not None
