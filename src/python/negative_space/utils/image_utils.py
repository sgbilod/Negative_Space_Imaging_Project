"""
Negative Space Detection - Image Utilities

This module provides utility functions for image loading, preprocessing,
and visualization.

Author: Stephen Bilodeau
Project: Negative Space Imaging Project
License: Proprietary
"""

from typing import Optional, List, Tuple
import numpy as np
import cv2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """
    Load image from file path.

    Args:
        path: Path to image file

    Returns:
        Image as numpy array (BGR format)

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If image cannot be read

    Example:
        >>> image = load_image('image.jpg')
        >>> print(f"Image shape: {image.shape}")
    """
    path_obj = Path(path)

    if not path_obj.exists():
        logger.error(f"Image file not found: {path}")
        raise FileNotFoundError(f"Image file not found: {path}")

    logger.info(f"Loading image from: {path}")

    image = cv2.imread(path)

    if image is None:
        logger.error(f"Failed to read image: {path}")
        raise IOError(f"Failed to read image from: {path}")

    logger.info(f"Image loaded successfully. Shape: {image.shape}")
    return image


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """
    Load image from bytes data.

    Args:
        data: Image data as bytes

    Returns:
        Image as numpy array (BGR format)

    Raises:
        ValueError: If data cannot be decoded as image

    Example:
        >>> with open('image.jpg', 'rb') as f:
        ...     image = load_image_from_bytes(f.read())
    """
    if not data:
        raise ValueError("Input data is empty")

    logger.info(f"Loading image from {len(data)} bytes")

    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        logger.error("Failed to decode image from bytes")
        raise ValueError("Failed to decode image from bytes data")

    logger.info(f"Image loaded from bytes. Shape: {image.shape}")
    return image


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resize image to fit within max_size while maintaining aspect ratio.

    Args:
        image: Input image
        max_size: Maximum dimension size

    Returns:
        Resized image

    Example:
        >>> image = load_image('large_image.jpg')
        >>> resized = resize_image(image, max_size=512)
        >>> print(f"Original: {image.shape}, Resized: {resized.shape}")
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid")

    height, width = image.shape[:2]

    if height <= max_size and width <= max_size:
        logger.debug(f"Image already within max_size: {width}x{height}")
        return image

    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)

    logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    logger.debug(f"Resize completed. New shape: {resized.shape}")
    return resized


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.

    Args:
        image: Input image (BGR or already grayscale)

    Returns:
        Grayscale image

    Example:
        >>> bgr_image = load_image('image.jpg')
        >>> gray = convert_to_grayscale(bgr_image)
        >>> print(f"Grayscale shape: {gray.shape}")
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid")

    if len(image.shape) == 2:
        logger.debug("Image already grayscale")
        return image

    if len(image.shape) != 3:
        raise ValueError("Input image must be 2D or 3D array")

    logger.info("Converting image to grayscale")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    logger.debug(f"Grayscale conversion completed. Shape: {gray.shape}")
    return gray


def enhance_contrast(
    image: np.ndarray, method: str = "clahe", clip_limit: float = 2.0
) -> np.ndarray:
    """
    Enhance image contrast.

    Args:
        image: Input grayscale image
        method: Enhancement method ('clahe' or 'histogram')
        clip_limit: CLAHE clip limit

    Returns:
        Contrast-enhanced image

    Example:
        >>> gray = convert_to_grayscale(image)
        >>> enhanced = enhance_contrast(gray)
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid")

    if len(image.shape) != 2:
        logger.warning("Expected grayscale image, converting...")
        image = convert_to_grayscale(image)

    logger.info(f"Enhancing contrast using {method} method")

    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        logger.debug(f"CLAHE enhancement completed (clip_limit={clip_limit})")

    elif method == "histogram":
        enhanced = cv2.equalizeHist(image)
        logger.debug("Histogram equalization completed")

    else:
        logger.warning(f"Unknown enhancement method: {method}, returning original")
        return image

    return enhanced


def save_visualization(
    image: np.ndarray, contours: List[np.ndarray], path: str, thickness: int = 2
) -> None:
    """
    Save image with contours drawn on it.

    Args:
        image: Input image
        contours: List of contours to draw
        path: Output file path
        thickness: Contour line thickness

    Raises:
        IOError: If save operation fails

    Example:
        >>> image = load_image('image.jpg')
        >>> contours = find_contours(edges)
        >>> save_visualization(image, contours, 'output.jpg')
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid")

    logger.info(f"Creating visualization with {len(contours)} contours")

    # Create a copy to avoid modifying original
    vis_image = image.copy()

    # Draw contours
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), thickness)

    logger.info(f"Saving visualization to: {path}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(path, vis_image)

    if not success:
        logger.error(f"Failed to save visualization to: {path}")
        raise IOError(f"Failed to save visualization to: {path}")

    logger.debug(f"Visualization saved successfully")


def get_image_info(image: np.ndarray) -> dict:
    """
    Extract information about image.

    Args:
        image: Input image

    Returns:
        Dictionary containing image information

    Example:
        >>> image = load_image('image.jpg')
        >>> info = get_image_info(image)
        >>> print(f"Size: {info['width']}x{info['height']}")
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid")

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    dtype = str(image.dtype)
    pixel_count = height * width
    memory_mb = (image.nbytes / (1024 * 1024))

    info = {
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": dtype,
        "pixel_count": pixel_count,
        "memory_mb": memory_mb,
        "aspect_ratio": width / height,
    }

    logger.debug(f"Image info: {width}x{height}, {channels} channels, {memory_mb:.2f} MB")

    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("✓ load_image: Image loading function defined")
    print("✓ load_image_from_bytes: Bytes loading function defined")
    print("✓ resize_image: Image resizing function defined")
    print("✓ convert_to_grayscale: Grayscale conversion function defined")
    print("✓ enhance_contrast: Contrast enhancement function defined")
    print("✓ save_visualization: Visualization saving function defined")
    print("✓ get_image_info: Image information function defined")

    print("\n✓ All image utility functions initialized successfully!")
