"""
Pytest Configuration and Fixtures for Negative Space Imaging Project Tests

This module provides comprehensive pytest configuration, reusable fixtures, mocking
utilities, and test data generators for the entire test suite.

Features:
- Image generation fixtures (synthetic, medical, astronomical)
- Mock object fixtures for all major components
- Database fixtures with cleanup
- Performance profiling fixtures
- Concurrent testing utilities
- Test data persistence and caching
"""

import pytest
import numpy as np
import torch
import tempfile
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator, Any
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
import sys

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =====================================================================
# PYTEST CONFIGURATION & MARKERS
# =====================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "database: mark test as requiring database"
    )
    config.addinivalue_line(
        "markers", "concurrent: mark test as testing concurrent operations"
    )


# =====================================================================
# SCOPE & SESSION FIXTURES
# =====================================================================

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create and return a session-level test data directory."""
    test_dir = Path(tempfile.gettempdir()) / "nsip_test_data"
    test_dir.mkdir(exist_ok=True)
    logger.info(f"Test data directory: {test_dir}")
    yield test_dir
    # Cleanup is optional - comment out to inspect test data
    # import shutil
    # shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Check if CUDA is available for GPU tests."""
    available = torch.cuda.is_available()
    logger.info(f"CUDA available: {available}")
    return available


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get the appropriate device (CPU or GPU) for testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


# =====================================================================
# IMAGE GENERATION FIXTURES
# =====================================================================

@pytest.fixture
def synthetic_image() -> np.ndarray:
    """Generate a synthetic test image with known patterns."""
    image = np.zeros((256, 256), dtype=np.uint8)

    # Add circular patterns (negative space candidates)
    cv2.circle(image, (64, 64), 30, 255, -1)
    cv2.circle(image, (192, 64), 30, 255, -1)
    cv2.circle(image, (64, 192), 30, 255, -1)
    cv2.circle(image, (192, 192), 30, 255, -1)

    # Add rectangular patterns
    cv2.rectangle(image, (40, 120), (100, 160), 200, -1)
    cv2.rectangle(image, (156, 120), (216, 160), 200, -1)

    # Add noise
    noise = np.random.normal(0, 5, image.shape)
    image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

    return image


@pytest.fixture
def medical_image() -> np.ndarray:
    """Generate a realistic medical image (e.g., CT scan-like)."""
    image = np.random.randint(0, 100, (512, 512), dtype=np.uint8)

    # Simulate tissue density variations
    y, x = np.ogrid[:512, :512]
    mask = (x - 256)**2 + (y - 256)**2 <= 150**2
    image[mask] = np.random.randint(150, 220, np.sum(mask), dtype=np.uint8)

    # Add some anatomical-like features
    cv2.ellipse(image, (256, 256), (120, 100), 0, 0, 360, 180, -1)

    # Gaussian blur to simulate real imaging
    image = cv2.GaussianBlur(image, (5, 5), 1.0)

    return image


@pytest.fixture
def astronomical_image() -> np.ndarray:
    """Generate an astronomical image (e.g., deep space with objects)."""
    # Create dark space background
    image = np.random.randint(0, 20, (256, 256), dtype=np.uint8)

    # Add bright objects (stars, galaxies)
    for _ in range(15):
        x = np.random.randint(20, 236)
        y = np.random.randint(20, 236)
        radius = np.random.randint(3, 12)
        brightness = np.random.randint(150, 255)
        cv2.circle(image, (x, y), radius, brightness, -1)

    # Add Gaussian noise
    noise = np.random.normal(0, 2, image.shape)
    image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

    return image


@pytest.fixture
def multi_channel_image() -> np.ndarray:
    """Generate a multi-channel (RGB) test image."""
    base = np.zeros((256, 256, 3), dtype=np.uint8)

    # Red channel: circular objects
    cv2.circle(base[:,:,0], (128, 128), 50, 200, -1)

    # Green channel: rectangular objects
    cv2.rectangle(base[:,:,1], (50, 50), (150, 150), 200, -1)

    # Blue channel: gradient
    for i in range(256):
        base[:, i, 2] = int(i)

    return base


@pytest.fixture
def image_batch(synthetic_image: np.ndarray) -> List[np.ndarray]:
    """Generate a batch of test images."""
    batch = []
    for i in range(5):
        img = synthetic_image.copy()
        # Apply slight variations
        noise = np.random.normal(0, i+1, img.shape)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        batch.append(img)
    return batch


@pytest.fixture
def edge_case_images() -> Dict[str, np.ndarray]:
    """Generate edge case images for robustness testing."""
    return {
        "empty": np.zeros((256, 256), dtype=np.uint8),
        "full": np.full((256, 256), 255, dtype=np.uint8),
        "single_pixel": np.zeros((256, 256), dtype=np.uint8),
        "small": np.random.randint(0, 256, (16, 16), dtype=np.uint8),
        "large": np.random.randint(0, 256, (2048, 2048), dtype=np.uint8),
        "non_square": np.random.randint(0, 256, (256, 512), dtype=np.uint8),
        "sparse": np.zeros((256, 256), dtype=np.uint8),
    }


# =====================================================================
# MOCK OBJECT FIXTURES
# =====================================================================

@pytest.fixture
def mock_analyzer():
    """Create a mock NegativeSpaceAnalyzer."""
    mock = MagicMock()
    mock.device = torch.device("cpu")
    mock.detection_threshold = 0.85
    mock.min_region_size = 100

    # Mock the analysis result
    mock._detect_negative_spaces.return_value = {
        "region_0_1": np.ones((256, 256), dtype=np.uint8) * 0.5,
        "region_0_2": np.ones((256, 256), dtype=np.uint8) * 0.3,
    }

    return mock


@pytest.fixture
def mock_segmenter():
    """Create a mock SemanticNegativeSpaceSegmenter."""
    mock = MagicMock()

    # Create mock segmentation result
    result = MagicMock()
    result.probabilities = torch.randn((1, 2, 256, 256))
    mock.return_value = result

    return mock


@pytest.fixture
def mock_region_grower():
    """Create a mock AdaptiveRegionGrower."""
    mock = MagicMock()

    growth_result = MagicMock()
    growth_result.mask = np.ones((256, 256), dtype=np.uint8)
    growth_result.confidence = 0.85
    mock.grow_region.return_value = growth_result

    return mock


@pytest.fixture
def mock_graph_analyzer():
    """Create a mock NegativeSpaceGraphAnalyzer."""
    mock = MagicMock()

    graph_result = MagicMock()
    graph_result.pattern_score = 0.75
    graph_result.connectivity = 0.82
    mock.analyze_pattern.return_value = graph_result

    return mock


@pytest.fixture
def mock_topology_analyzer():
    """Create a mock TopologicalAnalyzer."""
    mock = MagicMock()

    topo_result = MagicMock()
    topo_result.index = 1.5
    topo_result.betti_numbers = [1, 2]
    mock.analyze.return_value = topo_result

    return mock


# =====================================================================
# DATA STRUCTURE FIXTURES
# =====================================================================

@pytest.fixture
def analysis_result_data() -> Dict[str, Any]:
    """Create sample AnalysisResult data for testing."""
    return {
        "id": "test_analysis_001",
        "timestamp": "2025-01-17T10:30:00Z",
        "image_id": "image_001",
        "image_metadata": {
            "width": 256,
            "height": 256,
            "format": "PNG",
            "filename": "test_image.png",
        },
        "detected_regions": [
            {
                "id": "region_0_1",
                "centroid": [128, 128],
                "area": 5000,
                "confidence": 0.92,
                "bounding_box": {"x": 100, "y": 100, "width": 60, "height": 60},
            },
            {
                "id": "region_0_2",
                "centroid": [64, 64],
                "area": 3000,
                "confidence": 0.85,
                "bounding_box": {"x": 40, "y": 40, "width": 50, "height": 50},
            },
        ],
        "features": [
            {
                "type": "circular",
                "confidence": 0.89,
                "significance": 0.75,
                "region_id": "region_0_1",
            },
            {
                "type": "symmetric",
                "confidence": 0.82,
                "significance": 0.68,
                "region_id": "region_0_2",
            },
        ],
        "statistics": {
            "region_count": 2,
            "feature_count": 2,
            "total_negative_space_area": 8000,
            "average_confidence": 0.885,
            "average_region_size": 4000,
        },
        "processing_time_ms": 245,
        "algorithm_version": "2.1.0",
    }


@pytest.fixture
def negative_space_features_data() -> Dict[str, Any]:
    """Create sample NegativeSpaceFeatures data."""
    return {
        "area": 5000.0,
        "perimeter": 250.0,
        "centroid": (128.0, 128.0),
        "topology_index": 1.5,
        "connectivity": 0.82,
        "pattern_score": 0.75,
        "confidence": 0.92,
        "topological_features": {
            "index": 1.5,
            "betti_numbers": [1, 2],
        },
        "uncertainty_metrics": {
            "mean": 0.89,
            "std": 0.05,
            "confidence": 0.92,
        },
        "scale_params": {
            "scale_factor": 1.0,
            "target_resolution": (256, 256),
        },
    }


# =====================================================================
# DATABASE FIXTURES
# =====================================================================

@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    logger.info(f"Using temporary database: {db_path}")
    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)
        logger.info(f"Cleaned up database: {db_path}")


@pytest.fixture
def mock_db_connection():
    """Create a mock database connection."""
    mock = MagicMock()
    mock.execute.return_value = MagicMock(fetchall=lambda: [])
    mock.commit.return_value = None
    mock.close.return_value = None

    return mock


# =====================================================================
# PERFORMANCE & PROFILING FIXTURES
# =====================================================================

@pytest.fixture
def benchmark_timer():
    """Fixture for benchmarking execution time."""
    class Timer:
        def __init__(self):
            self.times = []

        def start(self) -> float:
            self.start_time = time.time()
            return self.start_time

        def stop(self) -> float:
            elapsed = time.time() - self.start_time
            self.times.append(elapsed)
            return elapsed

        def mean(self) -> float:
            return np.mean(self.times) if self.times else 0.0

        def std(self) -> float:
            return np.std(self.times) if self.times else 0.0

        def summary(self) -> Dict[str, float]:
            return {
                "mean_ms": self.mean() * 1000,
                "std_ms": self.std() * 1000,
                "min_ms": min(self.times) * 1000 if self.times else 0,
                "max_ms": max(self.times) * 1000 if self.times else 0,
                "count": len(self.times),
            }

    return Timer()


@pytest.fixture
def memory_profiler():
    """Fixture for memory profiling."""
    class MemoryProfiler:
        def __init__(self):
            self.snapshots = []

        def take_snapshot(self, label: str = "") -> Dict[str, float]:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            snapshot = {
                "timestamp": time.time(),
                "label": label,
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
            }
            self.snapshots.append(snapshot)
            return snapshot

        def get_peak_memory(self) -> float:
            return max(s["rss_mb"] for s in self.snapshots) if self.snapshots else 0.0

        def get_memory_delta(self) -> float:
            if len(self.snapshots) < 2:
                return 0.0
            return self.snapshots[-1]["rss_mb"] - self.snapshots[0]["rss_mb"]

    return MemoryProfiler()


# =====================================================================
# CONCURRENCY FIXTURES
# =====================================================================

@pytest.fixture
def thread_pool_executor():
    """Provide a thread pool executor for concurrent testing."""
    executor = ThreadPoolExecutor(max_workers=4)
    yield executor
    executor.shutdown(wait=True)


@pytest.fixture
def concurrent_test_runner():
    """Fixture for running concurrent tests."""
    class ConcurrentRunner:
        def __init__(self):
            self.results = []
            self.errors = []

        def run(self, func, args_list: List[Tuple], max_workers: int = 4):
            """Run function with different arguments concurrently."""
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(func, *args)
                    for args in args_list
                ]

                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        result = future.result(timeout=30)
                        self.results.append(result)
                    except Exception as e:
                        self.errors.append((i, str(e)))
                        logger.error(f"Error in concurrent task {i}: {e}")

            return self.results, self.errors

    return ConcurrentRunner()


# =====================================================================
# CONTEXT & MOCK PATCHES
# =====================================================================

@pytest.fixture
def patch_torch_cuda():
    """Mock torch.cuda availability for testing."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def patch_opencv():
    """Mock OpenCV functions for isolated testing."""
    with patch("cv2.cvtColor"), \
         patch("cv2.equalizeHist"), \
         patch("cv2.connectedComponents"), \
         patch("cv2.findContours"):
        yield


# =====================================================================
# CLEANUP & ASSERTION UTILITIES
# =====================================================================

@pytest.fixture
def assert_image_quality():
    """Fixture for image quality assertions."""
    def _assert(image: np.ndarray, expected_dtype: type = np.uint8,
                expected_range: Tuple[int, int] = (0, 255),
                expected_shape: Optional[Tuple] = None):
        """
        Assert image meets quality criteria.

        Args:
            image: Image array to validate
            expected_dtype: Expected data type
            expected_range: Expected value range
            expected_shape: Expected shape (optional)
        """
        assert isinstance(image, np.ndarray), "Not an ndarray"
        assert image.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {image.dtype}"
        assert np.min(image) >= expected_range[0], f"Min value {np.min(image)} below {expected_range[0]}"
        assert np.max(image) <= expected_range[1], f"Max value {np.max(image)} above {expected_range[1]}"

        if expected_shape:
            assert image.shape == expected_shape, f"Expected shape {expected_shape}, got {image.shape}"

    return _assert


@pytest.fixture
def assert_analysis_result():
    """Fixture for analysis result assertions."""
    def _assert(result: Dict[str, Any], has_regions: bool = True,
                has_features: bool = True, has_statistics: bool = True):
        """
        Assert analysis result meets expected criteria.

        Args:
            result: Analysis result dictionary
            has_regions: Whether to check for detected regions
            has_features: Whether to check for features
            has_statistics: Whether to check for statistics
        """
        assert isinstance(result, dict), "Result is not a dictionary"

        # Check required fields
        required_fields = ["id", "timestamp", "image_id", "processing_time_ms"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Check optional fields
        if has_regions:
            assert "detected_regions" in result, "Missing detected_regions"
            assert isinstance(result["detected_regions"], list), "detected_regions is not a list"

        if has_features:
            assert "features" in result, "Missing features"
            assert isinstance(result["features"], list), "features is not a list"

        if has_statistics:
            assert "statistics" in result, "Missing statistics"
            assert isinstance(result["statistics"], dict), "statistics is not a dict"

    return _assert


# =====================================================================
# UTILITY FIXTURES
# =====================================================================

@pytest.fixture
def random_seed():
    """Set random seeds for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture
def capture_logs(caplog):
    """Fixture for capturing and inspecting log messages."""
    with caplog.at_level(logging.DEBUG):
        yield caplog


# =====================================================================
# ENVIRONMENT FIXTURES
# =====================================================================

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup and cleanup test environment."""
    logger.info("=" * 60)
    logger.info("Starting test")
    logger.info("=" * 60)

    yield

    logger.info("=" * 60)
    logger.info("Test completed")
    logger.info("=" * 60)


# Import cv2 for image generation fixtures
try:
    import cv2
except ImportError:
    logger.warning("OpenCV not available - some fixtures may not work")
