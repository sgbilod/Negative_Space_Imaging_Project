import pytest
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.negative_space_analyzer import NegativeSpaceAnalyzer
from src.image_processor import ImageProcessor


@pytest.fixture
def sample_image():
    """Create a simple test image (100x100 white image with black border)."""
    img = Image.new('RGB', (100, 100), color='white')
    pixels = img.load()

    # Add black border
    for i in range(100):
        pixels[i, 0] = (0, 0, 0)
        pixels[i, 99] = (0, 0, 0)
        pixels[0, i] = (0, 0, 0)
        pixels[99, i] = (0, 0, 0)

    return img


@pytest.fixture
def sample_image_path(tmp_path, sample_image):
    """Save sample image to temporary path."""
    path = tmp_path / "test_image.jpg"
    sample_image.save(path)
    return str(path)


@pytest.fixture
def analyzer():
    """Create NegativeSpaceAnalyzer instance."""
    return NegativeSpaceAnalyzer()


class TestNegativeSpaceAnalyzer:
    """Test suite for NegativeSpaceAnalyzer class."""

    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze')
        assert hasattr(analyzer, 'analyze_bytes')

    def test_analyze_with_valid_image(self, analyzer, sample_image_path):
        """Test analyze() with valid image file."""
        result = analyzer.analyze(sample_image_path)

        assert result is not None
        assert 'negative_space_percentage' in result
        assert 'positive_space_percentage' in result
        assert 'regions_count' in result
        assert 'processing_time_ms' in result
        assert 'confidence_score' in result

    def test_analyze_with_nonexistent_file(self, analyzer):
        """Test analyze() raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            analyzer.analyze('/nonexistent/path/image.jpg')

    def test_analyze_with_invalid_image_format(self, analyzer, tmp_path):
        """Test analyze() raises error for invalid image format."""
        invalid_file = tmp_path / "not_an_image.txt"
        invalid_file.write_text("This is not an image")

        with pytest.raises(Exception):
            analyzer.analyze(str(invalid_file))

    def test_analyze_bytes_with_valid_image(self, analyzer, sample_image):
        """Test analyze_bytes() with image bytes."""
        img_byte_arr = BytesIO()
        sample_image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        result = analyzer.analyze_bytes(img_bytes)

        assert result is not None
        assert 'negative_space_percentage' in result
        assert isinstance(result['negative_space_percentage'], (int, float))

    def test_analyze_bytes_with_invalid_bytes(self, analyzer):
        """Test analyze_bytes() with invalid image bytes."""
        invalid_bytes = b'not an image'

        with pytest.raises(Exception):
            analyzer.analyze_bytes(invalid_bytes)

    def test_batch_analyze(self, analyzer, tmp_path, sample_image):
        """Test batch_analyze() with multiple images."""
        # Create test images
        image_paths = []
        for i in range(3):
            path = tmp_path / f"test_image_{i}.jpg"
            sample_image.save(path)
            image_paths.append(str(path))

        results = analyzer.batch_analyze(image_paths)

        assert len(results) == 3
        for result in results:
            assert 'negative_space_percentage' in result
            assert result['status'] == 'completed'

    def test_batch_analyze_with_mixed_valid_invalid(self, analyzer, tmp_path, sample_image):
        """Test batch_analyze() with mix of valid and invalid paths."""
        valid_path = tmp_path / "valid_image.jpg"
        sample_image.save(valid_path)

        image_paths = [str(valid_path), '/nonexistent/path.jpg']

        results = analyzer.batch_analyze(image_paths)

        # Should process valid image, skip/mark invalid one
        assert len(results) == 2
        assert results[0]['status'] == 'completed'
        assert results[1]['status'] in ['failed', 'error']

    def test_result_metrics_are_valid(self, analyzer, sample_image_path):
        """Test that returned metrics are within valid ranges."""
        result = analyzer.analyze(sample_image_path)

        neg_space = result['negative_space_percentage']
        pos_space = result['positive_space_percentage']
        confidence = result['confidence_score']

        # Validate ranges
        assert 0 <= neg_space <= 100
        assert 0 <= pos_space <= 100
        assert 0 <= confidence <= 1
        assert abs(neg_space + pos_space - 100) < 1  # Should sum to ~100

    def test_processing_time_is_recorded(self, analyzer, sample_image_path):
        """Test that processing time is recorded."""
        result = analyzer.analyze(sample_image_path)

        assert 'processing_time_ms' in result
        assert result['processing_time_ms'] > 0
        assert isinstance(result['processing_time_ms'], (int, float))

    def test_regions_count_is_valid(self, analyzer, sample_image_path):
        """Test that regions count is a valid non-negative integer."""
        result = analyzer.analyze(sample_image_path)

        assert 'regions_count' in result
        assert isinstance(result['regions_count'], int)
        assert result['regions_count'] >= 0

    def test_confidence_score_calculation(self, analyzer, sample_image_path):
        """Test confidence score is properly calculated."""
        result = analyzer.analyze(sample_image_path)

        confidence = result['confidence_score']

        # Confidence should be between 0 and 1
        assert 0 <= confidence <= 1

        # Should be float
        assert isinstance(confidence, float)


class TestImageProcessor:
    """Test suite for ImageProcessor utility."""

    def test_processor_load_image(self, sample_image_path):
        """Test loading image file."""
        processor = ImageProcessor()
        image = processor.load_image(sample_image_path)

        assert image is not None
        assert isinstance(image, np.ndarray)

    def test_processor_convert_to_grayscale(self, sample_image):
        """Test image conversion to grayscale."""
        processor = ImageProcessor()
        img_array = np.array(sample_image)

        gray = processor.to_grayscale(img_array)

        assert len(gray.shape) == 2  # Grayscale should be 2D

    def test_processor_with_different_image_sizes(self, tmp_path):
        """Test processor with various image sizes."""
        processor = ImageProcessor()
        sizes = [(100, 100), (500, 300), (1920, 1080)]

        for width, height in sizes:
            img = Image.new('RGB', (width, height), color='white')
            path = tmp_path / f"image_{width}x{height}.jpg"
            img.save(path)

            loaded = processor.load_image(str(path))
            assert loaded.shape[:2] == (height, width)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
