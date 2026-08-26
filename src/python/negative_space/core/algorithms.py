"""
Negative Space Detection - Image Processing Algorithms

This module implements core algorithms for edge detection, contour analysis,
and confidence scoring for negative space detection.

Author: Stephen Bilodeau
Project: Negative Space Imaging Project
License: Proprietary
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


def detect_edges(image: np.ndarray, method: str = "canny", **kwargs) -> np.ndarray:
    """
    Detect edges in an image using specified method.

    Args:
        image: Input grayscale image (2D array)
        method: Edge detection method ('canny' or 'sobel')
        **kwargs: Additional arguments for the specific method

    Returns:
        Binary edge map

    Raises:
        ValueError: If method is unsupported or image is invalid

    Example:
        >>> gray_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
        >>> edges = detect_edges(gray_image, method='canny')
        >>> print(f"Edge map shape: {edges.shape}")
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid")

    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D)")

    logger.info(f"Detecting edges using {method} method")

    if method == "canny":
        threshold1 = kwargs.get("threshold1", 100)
        threshold2 = kwargs.get("threshold2", 200)
        apertureSize = kwargs.get("apertureSize", 3)
        L2gradient = kwargs.get("L2gradient", True)

        edges = cv2.Canny(
            image, threshold1, threshold2, apertureSize=apertureSize, L2gradient=L2gradient
        )
        logger.debug(f"Canny edge detection: {threshold1}-{threshold2}")

    elif method == "sobel":
        kernel_size = kwargs.get("kernel_size", 3)
        scale = kwargs.get("scale", 1.0)

        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)

        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.uint8(np.clip(magnitude * scale, 0, 255))

        _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        logger.debug(f"Sobel edge detection: kernel_size={kernel_size}")

    else:
        raise ValueError(f"Unsupported edge detection method: {method}")

    logger.info(f"Edge detection completed. Non-zero pixels: {np.count_nonzero(edges)}")
    return edges


def find_contours(edges: np.ndarray) -> List[np.ndarray]:
    """
    Find contours in an edge map.

    Args:
        edges: Binary edge map from edge detection

    Returns:
        List of contours as numpy arrays

    Raises:
        ValueError: If edges array is invalid

    Example:
        >>> edges = detect_edges(gray_image, method='canny')
        >>> contours = find_contours(edges)
        >>> print(f"Found {len(contours)} contours")
    """
    if edges is None or edges.size == 0:
        raise ValueError("Input edges array is empty or invalid")

    logger.info("Finding contours in edge map")

    # OpenCV 4.0+ returns (contours, hierarchy), older versions return (image, contours, hierarchy)
    contour_result = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contour_result) == 3:
        _, contours, _ = contour_result
    else:
        contours, _ = contour_result

    logger.info(f"Found {len(contours)} contours")
    return list(contours)


def filter_contours(
    contours: List[np.ndarray],
    min_area: int = 100,
    max_area: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Filter contours by area criteria.

    Args:
        contours: List of contours from find_contours
        min_area: Minimum contour area in pixels
        max_area: Maximum contour area in pixels (None for no limit)

    Returns:
        Filtered list of contours

    Example:
        >>> contours = find_contours(edges)
        >>> filtered = filter_contours(contours, min_area=100, max_area=10000)
        >>> print(f"Filtered to {len(filtered)} contours")
    """
    if not contours:
        logger.warning("No contours provided for filtering")
        return []

    logger.info(f"Filtering {len(contours)} contours (min_area={min_area}, max_area={max_area})")

    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        if max_area is not None and area > max_area:
            continue

        filtered.append(contour)

    logger.info(f"Filtered contours: {len(filtered)} remaining")
    return filtered


def calculate_confidence(contour: np.ndarray, image: np.ndarray) -> float:
    """
    Calculate confidence score for a contour based on shape properties.

    Args:
        contour: Single contour array
        image: Original grayscale image

    Returns:
        Confidence score between 0 and 1

    Example:
        >>> confidence = calculate_confidence(contour, gray_image)
        >>> print(f"Confidence: {confidence:.2f}")
    """
    if contour is None or len(contour) < 4:
        logger.warning("Contour too small for confidence calculation")
        return 0.0

    try:
        # Calculate area
        area = cv2.contourArea(contour)
        if area == 0:
            return 0.0

        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0

        # Calculate circularity (4π * area / perimeter²)
        circularity = 4 * np.pi * area / (perimeter**2)
        circularity = min(circularity, 1.0)  # Clamp to 1.0

        # Fit ellipse if possible (requires at least 5 points)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (width, height), angle = ellipse

            # Calculate aspect ratio
            aspect_ratio = min(width, height) / (max(width, height) + 1e-6)

            # Calculate solidity (contour area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)

            # Combined confidence score
            confidence = (circularity + aspect_ratio + solidity) / 3.0
        else:
            confidence = circularity

        confidence = float(np.clip(confidence, 0.0, 1.0))
        logger.debug(f"Calculated confidence: {confidence:.3f}")
        return confidence

    except Exception as e:
        logger.error(f"Error calculating confidence: {e}")
        return 0.0


def extract_bounding_boxes(contours: List[np.ndarray]) -> List[Dict[str, float]]:
    """
    Extract bounding box information from contours.

    Args:
        contours: List of contours

    Returns:
        List of dictionaries containing bounding box information

    Example:
        >>> boxes = extract_bounding_boxes(contours)
        >>> for box in boxes:
        ...     print(f"Box at ({box['x']}, {box['y']}): {box['width']}x{box['height']}")
    """
    if not contours:
        logger.warning("No contours provided for bounding box extraction")
        return []

    boxes = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate moments for centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = x + w / 2
            cy = y + h / 2

        # Calculate aspect ratio
        aspect_ratio = w / (h + 1e-6)

        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        solidity = contour_area / (hull_area + 1e-6)

        # Circularity
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * contour_area / (perimeter**2 + 1e-6)
        circularity = min(circularity, 1.0)

        box = {
            "id": i,
            "x": float(x),
            "y": float(y),
            "width": float(w),
            "height": float(h),
            "area": float(contour_area),
            "perimeter": float(perimeter),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "aspect_ratio": float(aspect_ratio),
            "solidity": float(solidity),
            "circularity": float(circularity),
        }
        boxes.append(box)

    logger.info(f"Extracted {len(boxes)} bounding boxes")
    return boxes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("✓ detect_edges: Edge detection function defined")
    print("✓ find_contours: Contour finding function defined")
    print("✓ filter_contours: Contour filtering function defined")
    print("✓ calculate_confidence: Confidence scoring function defined")
    print("✓ extract_bounding_boxes: Bounding box extraction function defined")

    print("\n✓ All image processing algorithms initialized successfully!")
