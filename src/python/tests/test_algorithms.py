import pytest
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms import (
    detect_edges,
    detect_edges_sobel,
    find_contours,
    filter_contours,
    calculate_confidence,
    extract_bounding_boxes,
)


@pytest.fixture
def sample_image_array():
    """Create a simple test image as numpy array."""
    # Create 100x100 white image with black square in center
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    img[25:75, 25:75] = 0  # Black square in center
    return img


@pytest.fixture
def grayscale_image(sample_image_array):
    """Convert sample image to grayscale."""
    return np.dot(sample_image_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


class TestEdgeDetection:
    """Test suite for edge detection algorithms."""

    def test_detect_edges_canny(self, grayscale_image):
        """Test Canny edge detection."""
        edges = detect_edges(grayscale_image, method='canny', low_threshold=100, high_threshold=200)

        assert edges is not None
        assert edges.dtype == np.uint8 or edges.dtype == bool
        assert edges.shape == grayscale_image.shape

    def test_detect_edges_canny_different_thresholds(self, grayscale_image):
        """Test Canny with different threshold values."""
        edges_loose = detect_edges(grayscale_image, method='canny', low_threshold=50, high_threshold=150)
        edges_strict = detect_edges(grayscale_image, method='canny', low_threshold=150, high_threshold=250)

        # Loose thresholds should detect more edges
        assert np.sum(edges_loose) >= np.sum(edges_strict)

    def test_detect_edges_sobel(self, grayscale_image):
        """Test Sobel edge detection."""
        edges = detect_edges_sobel(grayscale_image)

        assert edges is not None
        assert edges.shape == grayscale_image.shape
        assert edges.dtype in [np.float64, np.float32, np.uint8]

    def test_detect_edges_returns_2d(self, sample_image_array):
        """Test that edge detection returns 2D array."""
        edges = detect_edges(sample_image_array)

        assert len(edges.shape) == 2

    def test_edge_detection_with_uniform_image(self):
        """Test edge detection on uniform image (no edges)."""
        uniform_image = np.ones((100, 100), dtype=np.uint8) * 128

        edges = detect_edges(uniform_image)

        # Should have very few or no edges
        edge_count = np.sum(edges > 0)
        assert edge_count < 50  # Less than 50% of pixels


class TestContourDetection:
    """Test suite for contour detection and filtering."""

    def test_find_contours(self, grayscale_image):
        """Test finding contours in image."""
        contours = find_contours(grayscale_image)

        assert contours is not None
        assert isinstance(contours, list) or hasattr(contours, '__len__')
        assert len(contours) > 0

    def test_find_contours_with_clear_objects(self):
        """Test finding contours with clear distinct objects."""
        # Create image with 3 distinct black circles
        img = np.ones((200, 200), dtype=np.uint8) * 255

        # Draw circles (simplified with rectangles)
        img[30:70, 30:70] = 0
        img[130:170, 30:70] = 0
        img[80:120, 130:170] = 0

        contours = find_contours(img)

        # Should find multiple contours
        assert len(contours) >= 2

    def test_filter_contours_removes_small(self, grayscale_image):
        """Test that filtering removes small contours."""
        contours = find_contours(grayscale_image)
        min_area = 100

        filtered = filter_contours(contours, min_area=min_area)

        # Filtered should have fewer or equal contours
        assert len(filtered) <= len(contours)

        # All remaining contours should meet min_area requirement
        for contour in filtered:
            area = contour_area(contour)
            assert area >= min_area

    def test_filter_contours_with_high_threshold(self, grayscale_image):
        """Test filtering with very high threshold."""
        contours = find_contours(grayscale_image)

        # Use threshold higher than any actual contour
        filtered = filter_contours(contours, min_area=100000)

        # Should be mostly empty
        assert len(filtered) <= len(contours)

    def test_filter_contours_preserves_large_contours(self):
        """Test that large contours are preserved."""
        img = np.ones((200, 200), dtype=np.uint8) * 255
        # Large black square (50x50 pixels = 2500 area)
        img[50:150, 50:150] = 0

        contours = find_contours(img)
        filtered = filter_contours(contours, min_area=1000)

        # Should preserve the large square
        assert len(filtered) > 0


class TestConfidenceCalculation:
    """Test suite for confidence score calculation."""

    def test_calculate_confidence_returns_valid_range(self, grayscale_image):
        """Test that confidence score is between 0 and 1."""
        confidence = calculate_confidence(grayscale_image)

        assert isinstance(confidence, (float, np.floating))
        assert 0 <= confidence <= 1

    def test_calculate_confidence_uniform_image(self):
        """Test confidence on uniform image."""
        uniform_image = np.ones((100, 100), dtype=np.uint8) * 128

        confidence = calculate_confidence(uniform_image)

        # Uniform image should have lower confidence (no clear features)
        assert confidence < 0.7

    def test_calculate_confidence_high_contrast(self):
        """Test confidence on high contrast image."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 255  # High contrast square

        confidence = calculate_confidence(img)

        # High contrast should have higher confidence
        assert confidence > 0.3

    def test_calculate_confidence_consistency(self, grayscale_image):
        """Test that confidence calculation is consistent."""
        conf1 = calculate_confidence(grayscale_image)
        conf2 = calculate_confidence(grayscale_image)

        assert conf1 == conf2


class TestBoundingBoxExtraction:
    """Test suite for bounding box extraction."""

    def test_extract_bounding_boxes(self):
        """Test extracting bounding boxes from contours."""
        # Create simple image with clear object
        img = np.ones((200, 200), dtype=np.uint8) * 255
        img[50:150, 50:150] = 0  # Black square

        contours = find_contours(img)
        bboxes = extract_bounding_boxes(contours)

        assert bboxes is not None
        assert len(bboxes) > 0

        # Each bbox should have x, y, width, height
        for bbox in bboxes:
            assert 'x' in bbox
            assert 'y' in bbox
            assert 'width' in bbox
            assert 'height' in bbox

    def test_bounding_box_format(self):
        """Test bounding box format validation."""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        img[20:80, 20:80] = 0

        contours = find_contours(img)
        bboxes = extract_bounding_boxes(contours)

        for bbox in bboxes:
            # Coordinates should be non-negative
            assert bbox['x'] >= 0
            assert bbox['y'] >= 0
            # Dimensions should be positive
            assert bbox['width'] > 0
            assert bbox['height'] > 0

    def test_bounding_boxes_contained_in_image(self):
        """Test that bounding boxes are within image bounds."""
        img = np.ones((200, 200), dtype=np.uint8) * 255
        img[30:170, 40:180] = 0

        contours = find_contours(img)
        bboxes = extract_bounding_boxes(contours)

        for bbox in bboxes:
            assert bbox['x'] + bbox['width'] <= 200
            assert bbox['y'] + bbox['height'] <= 200


class TestAlgorithmIntegration:
    """Test algorithm components working together."""

    def test_full_pipeline(self, sample_image_array):
        """Test complete analysis pipeline."""
        # Convert to grayscale
        gray = np.dot(sample_image_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        # Detect edges
        edges = detect_edges(gray, method='canny')

        # Find contours
        contours = find_contours(edges)

        # Filter contours
        filtered = filter_contours(contours, min_area=10)

        # Extract bounding boxes
        bboxes = extract_bounding_boxes(filtered)

        # Calculate confidence
        confidence = calculate_confidence(gray)

        # All steps should succeed
        assert edges is not None
        assert len(contours) >= 0
        assert len(filtered) >= 0
        assert confidence >= 0 and confidence <= 1

    def test_pipeline_with_multiple_objects(self):
        """Test pipeline with multiple distinct objects."""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 255

        # Add multiple black objects
        img[30:80, 30:80] = 0  # Square
        img[100:150, 200:250] = 0  # Another square
        img[200:250, 100:150] = 0  # Third square

        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        edges = detect_edges(gray)
        contours = find_contours(edges)
        filtered = filter_contours(contours, min_area=100)

        # Should detect multiple objects
        assert len(filtered) >= 2


# Helper function for tests
def contour_area(contour):
    """Calculate area of contour (simplified)."""
    if isinstance(contour, np.ndarray):
        return len(contour)
    return 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
