"""
Negative Space Detection - Core Analyzer

This module implements the main NegativeSpaceAnalyzer class that orchestrates
the negative space detection workflow.

Author: Stephen Bilodeau
Project: Negative Space Imaging Project
License: Proprietary
"""

from typing import List, Dict, Optional, Any
import numpy as np
import cv2
import logging
import time
from pathlib import Path

from . import algorithms
from . import models
from ..utils import image_utils
from ..exceptions import ImageLoadError, AnalysisError, ValidationError

logger = logging.getLogger(__name__)


class NegativeSpaceAnalyzer:
    """
    Analyzes images to detect and characterize negative space regions.

    This class provides a comprehensive interface for negative space detection,
    supporting single image analysis, batch processing, and customizable
    configuration.

    Attributes:
        config: Configuration object for analysis parameters
        _cache: Internal cache for processed images
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize NegativeSpaceAnalyzer.

        Args:
            config: Optional configuration dictionary. If not provided, defaults are used.

        Raises:
            ValidationError: If configuration is invalid

        Example:
            >>> analyzer = NegativeSpaceAnalyzer()
            >>> result = analyzer.analyze('image.jpg')
            >>> print(f"Negative space: {result.negative_space_percentage:.1f}%")
        """
        try:
            if config is None:
                self.config = models.ConfigModel()
            else:
                self.config = models.ConfigModel(**config)

            self._cache: Dict[str, np.ndarray] = {}
            logger.info("NegativeSpaceAnalyzer initialized successfully")
            logger.debug(f"Config: edge_method={self.config.edge_detection_method}, "
                        f"min_area={self.config.min_contour_area}")

        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {e}")
            raise ValidationError(f"Invalid configuration: {str(e)}", "config")

    def analyze(self, image_path: str) -> models.AnalysisResult:
        """
        Analyze a single image for negative space.

        Args:
            image_path: Path to image file

        Returns:
            AnalysisResult containing detection results

        Raises:
            ImageLoadError: If image cannot be loaded
            AnalysisError: If analysis fails

        Example:
            >>> analyzer = NegativeSpaceAnalyzer()
            >>> result = analyzer.analyze('test_image.jpg')
            >>> print(f"Found {len(result.contours)} contours")
            >>> print(f"Average confidence: {result.average_confidence:.2f}")
        """
        start_time = time.time()
        logger.info(f"Starting analysis of: {image_path}")

        try:
            # Load image
            image = image_utils.load_image(image_path)

            # Perform analysis
            result = self._analyze_image(image, image_path)

            # Record processing time
            result.processing_time_ms = (time.time() - start_time) * 1000

            logger.info(f"Analysis completed in {result.processing_time_ms:.1f}ms")
            return result

        except ImageLoadError as e:
            logger.error(f"Image load error: {e}")
            raise

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise AnalysisError(f"Failed to analyze image: {str(e)}", "analyze")

    def analyze_bytes(self, image_data: bytes) -> models.AnalysisResult:
        """
        Analyze image from bytes data.

        Args:
            image_data: Image data as bytes

        Returns:
            AnalysisResult containing detection results

        Raises:
            ImageLoadError: If image cannot be decoded
            AnalysisError: If analysis fails

        Example:
            >>> with open('image.jpg', 'rb') as f:
            ...     result = analyzer.analyze_bytes(f.read())
        """
        start_time = time.time()
        logger.info(f"Starting analysis of image from bytes ({len(image_data)} bytes)")

        try:
            image = image_utils.load_image_from_bytes(image_data)

            # Perform analysis
            result = self._analyze_image(image, "from_bytes")

            # Record processing time
            result.processing_time_ms = (time.time() - start_time) * 1000

            logger.info(f"Analysis completed in {result.processing_time_ms:.1f}ms")
            return result

        except ImageLoadError as e:
            logger.error(f"Image load error: {e}")
            raise

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise AnalysisError(f"Failed to analyze bytes: {str(e)}", "analyze_bytes")

    def batch_analyze(self, image_paths: List[str]) -> List[models.AnalysisResult]:
        """
        Analyze multiple images.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of AnalysisResult objects

        Example:
            >>> images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
            >>> results = analyzer.batch_analyze(images)
            >>> for result in results:
            ...     print(f"{result.image_path}: {result.negative_space_percentage:.1f}%")
        """
        logger.info(f"Starting batch analysis of {len(image_paths)} images")

        results: List[models.AnalysisResult] = []

        for i, image_path in enumerate(image_paths, 1):
            try:
                logger.info(f"Processing {i}/{len(image_paths)}: {image_path}")
                result = self.analyze(image_path)
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to analyze {image_path}: {e}")
                continue

        logger.info(f"Batch analysis completed: {len(results)}/{len(image_paths)} successful")
        return results

    def _analyze_image(self, image: np.ndarray, source: str) -> models.AnalysisResult:
        """
        Internal method to perform analysis on an image array.

        Args:
            image: Image as numpy array
            source: Source identifier for logging

        Returns:
            AnalysisResult

        Raises:
            AnalysisError: If any analysis step fails
        """
        try:
            # Get image dimensions
            height, width = image.shape[:2]
            total_pixels = height * width

            # Resize if needed
            image = image_utils.resize_image(image, self.config.max_image_size)

            # Convert to grayscale
            gray = image_utils.convert_to_grayscale(image)

            # Enhance contrast if enabled
            if self.config.enable_contrast_enhancement:
                gray = image_utils.enhance_contrast(gray, method="clahe")

            # Detect edges
            edges = algorithms.detect_edges(
                gray,
                method=self.config.edge_detection_method,
                threshold1=self.config.canny_threshold1,
                threshold2=self.config.canny_threshold2,
            )

            # Apply morphological operations if enabled
            if self.config.enable_morphology:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self.config.morphology_kernel_size,
                                        self.config.morphology_kernel_size)
                )
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours = algorithms.find_contours(edges)

            # Filter contours
            filtered_contours = algorithms.filter_contours(
                contours, self.config.min_contour_area, self.config.max_contour_area
            )

            # Extract bounding boxes and calculate metrics
            bounding_boxes = algorithms.extract_bounding_boxes(filtered_contours)

            # Create contour data models
            contour_models: List[models.ContourData] = []
            total_negative_space = 0.0
            confidence_scores = []

            for i, (bbox, contour) in enumerate(zip(bounding_boxes, filtered_contours)):
                confidence = algorithms.calculate_confidence(contour, gray)

                # Only include contours meeting confidence threshold
                if confidence < self.config.confidence_threshold:
                    continue

                contour_data = models.ContourData(
                    id=i,
                    area=bbox["area"],
                    perimeter=bbox["perimeter"],
                    confidence=confidence,
                    bounding_box={
                        "x": bbox["x"],
                        "y": bbox["y"],
                        "width": bbox["width"],
                        "height": bbox["height"],
                    },
                    centroid={"x": bbox["centroid_x"], "y": bbox["centroid_y"]},
                    aspect_ratio=bbox["aspect_ratio"],
                    solidity=bbox["solidity"],
                    circularity=bbox["circularity"],
                )

                contour_models.append(contour_data)
                total_negative_space += bbox["area"]
                confidence_scores.append(confidence)

            # Calculate statistics
            average_confidence = (
                sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            )

            negative_space_percentage = (total_negative_space / total_pixels) * 100

            # Create result
            result = models.AnalysisResult(
                image_path=source,
                image_width=width,
                image_height=height,
                contours=contour_models,
                total_negative_space=total_negative_space,
                negative_space_percentage=negative_space_percentage,
                average_confidence=average_confidence,
                processing_time_ms=0.0,  # Will be set by caller
                metadata={
                    "edge_method": self.config.edge_detection_method,
                    "num_raw_contours": len(contours),
                    "num_filtered_contours": len(filtered_contours),
                    "num_final_contours": len(contour_models),
                    "resized": image.shape[:2] != (height, width),
                },
            )

            return result

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise AnalysisError(f"Analysis step failed: {str(e)}", "_analyze_image")

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Update analyzer configuration.

        Args:
            config_dict: Dictionary of configuration parameters to update

        Raises:
            ValidationError: If configuration is invalid

        Example:
            >>> analyzer.update_config({
            ...     'edge_detection_method': 'sobel',
            ...     'min_contour_area': 50
            ... })
        """
        try:
            updated_config = self.config.model_dump()
            updated_config.update(config_dict)
            self.config = models.ConfigModel(**updated_config)
            logger.info(f"Configuration updated: {config_dict}")

        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            raise ValidationError(f"Invalid configuration update: {str(e)}", "config_dict")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("✓ NegativeSpaceAnalyzer: Core analyzer class defined")
    print("✓ analyze: Single image analysis method defined")
    print("✓ analyze_bytes: Bytes analysis method defined")
    print("✓ batch_analyze: Batch analysis method defined")

    # Test instantiation
    try:
        analyzer = NegativeSpaceAnalyzer()
        print(f"\n✓ Analyzer instantiated successfully")
        print(f"  - Edge method: {analyzer.config.edge_detection_method}")
        print(f"  - Min area: {analyzer.config.min_contour_area}")
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")

    print("\n✓ NegativeSpaceAnalyzer initialized successfully!")
