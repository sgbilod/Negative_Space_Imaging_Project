#!/usr/bin/env python
"""
Image Acquisition Demo for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This script demonstrates the image acquisition pipeline for the Negative Space
Imaging System, including:

1. Loading an image from various sources (file, URL, camera)
2. Preprocessing and normalizing the image
3. Applying initial filters and enhancements
4. Preparing the image for negative space analysis
5. Storing the image with appropriate metadata

Usage:
    python demo_acquisition.py --source file --path ./Hoag's_object.jpg --output ./processed/
"""

import argparse
import json
import os
import sys
import time
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Add parent directory to path to allow imports from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
try:
    from utils.logger import setup_logger
except ImportError:
    # Define a simple logger if the project logger is not available
    import logging
    def setup_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

# Initialize logger
logger = setup_logger("image_acquisition_demo")

# Constants
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "tiff", "bmp", "webp"]
DEFAULT_OUTPUT_DIR = "./processed"
DEFAULT_QUALITY = 95
MAX_IMAGE_SIZE = (4096, 4096)  # Max dimensions for processing

class ImageAcquisitionError(Exception):
    """Custom exception for image acquisition errors."""
    pass

class ImageAcquisition:
    """
    Class for handling image acquisition, preprocessing, and storage.
    """

    def __init__(self, source_type: str, source_path: str, output_dir: str = DEFAULT_OUTPUT_DIR,
                 quality: int = DEFAULT_QUALITY, metadata: Optional[Dict] = None):
        """
        Initialize the image acquisition process.

        Args:
            source_type: Type of source ('file', 'url', 'camera')
            source_path: Path to the source (file path, URL, or camera ID)
            output_dir: Directory to save processed images
            quality: Quality for saved images (1-100)
            metadata: Additional metadata to store with the image
        """
        self.source_type = source_type.lower()
        self.source_path = source_path
        self.output_dir = output_dir
        self.quality = min(max(quality, 1), 100)  # Ensure quality is between 1-100
        self.metadata = metadata or {}
        self.image_id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize image containers
        self.original_image = None
        self.processed_image = None

        logger.info(f"Initialized image acquisition with ID: {self.image_id}")
        logger.info(f"Source: {self.source_type} - {self.source_path}")

    def acquire(self) -> Image.Image:
        """
        Acquire the image from the specified source.

        Returns:
            The acquired PIL Image
        """
        logger.info(f"Acquiring image from {self.source_type}: {self.source_path}")

        try:
            if self.source_type == "file":
                return self._acquire_from_file()
            elif self.source_type == "url":
                return self._acquire_from_url()
            elif self.source_type == "camera":
                return self._acquire_from_camera()
            else:
                raise ImageAcquisitionError(f"Unsupported source type: {self.source_type}")
        except Exception as e:
            logger.error(f"Error acquiring image: {str(e)}")
            raise ImageAcquisitionError(f"Failed to acquire image: {str(e)}") from e

    def _acquire_from_file(self) -> Image.Image:
        """Load image from a file."""
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Image file not found: {self.source_path}")

        try:
            image = Image.open(self.source_path)
            logger.info(f"Loaded image from file: {self.source_path}")
            logger.info(f"Image dimensions: {image.width}x{image.height}, Format: {image.format}")

            # Extract file metadata
            self.metadata.update({
                "original_filename": os.path.basename(self.source_path),
                "original_format": image.format,
                "original_size": os.path.getsize(self.source_path),
                "original_dimensions": f"{image.width}x{image.height}",
            })

            self.original_image = image
            return image
        except Exception as e:
            raise ImageAcquisitionError(f"Failed to load image from file: {str(e)}") from e

    def _acquire_from_url(self) -> Image.Image:
        """Load image from a URL."""
        try:
            import requests
            from io import BytesIO

            response = requests.get(self.source_path, stream=True, timeout=10)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))
            logger.info(f"Loaded image from URL: {self.source_path}")
            logger.info(f"Image dimensions: {image.width}x{image.height}, Format: {image.format}")

            # Extract URL metadata
            self.metadata.update({
                "source_url": self.source_path,
                "content_type": response.headers.get("Content-Type"),
                "content_length": response.headers.get("Content-Length"),
                "original_format": image.format,
                "original_dimensions": f"{image.width}x{image.height}",
            })

            self.original_image = image
            return image
        except ImportError:
            raise ImageAcquisitionError("Failed to load image from URL: requests module not installed")
        except Exception as e:
            raise ImageAcquisitionError(f"Failed to load image from URL: {str(e)}") from e

    def _acquire_from_camera(self) -> Image.Image:
        """
        Acquire image from a camera.
        Note: This is a simplified demo that simulates camera acquisition.
        """
        try:
            # In a real implementation, this would use a camera library like OpenCV
            # For this demo, we'll create a simple test image
            logger.info("Simulating camera acquisition (demo mode)")

            # Create a gradient test image with a timestamp
            width, height = 800, 600
            image = Image.new("RGB", (width, height), color="black")

            # Create a simple gradient background
            pixels = image.load()
            for x in range(width):
                for y in range(height):
                    r = int(255 * x / width)
                    g = int(255 * y / height)
                    b = 128
                    pixels[x, y] = (r, g, b)

            # Add timestamp text
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            text = f"Camera Acquisition: {timestamp}"

            # Use default font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            text_width, text_height = draw.textsize(text, font=font)
            position = ((width - text_width) // 2, (height - text_height) // 2)
            draw.text(position, text, fill="white", font=font)

            # Add camera metadata
            self.metadata.update({
                "camera_id": self.source_path,
                "acquisition_time": timestamp,
                "simulated": True,
                "original_dimensions": f"{width}x{height}",
            })

            logger.info(f"Generated simulated camera image: {width}x{height}")
            self.original_image = image
            return image
        except Exception as e:
            raise ImageAcquisitionError(f"Failed to acquire image from camera: {str(e)}") from e

    def preprocess(self, resize: bool = True, enhance: bool = True,
                   denoise: bool = True) -> Image.Image:
        """
        Preprocess the acquired image for analysis.

        Args:
            resize: Whether to resize large images
            enhance: Whether to apply enhancement
            denoise: Whether to apply noise reduction

        Returns:
            Processed PIL Image
        """
        if self.original_image is None:
            raise ImageAcquisitionError("No image acquired yet. Call acquire() first.")

        logger.info("Preprocessing image...")
        image = self.original_image.copy()

        # Resize large images to improve processing speed
        if resize and (image.width > MAX_IMAGE_SIZE[0] or image.height > MAX_IMAGE_SIZE[1]):
            logger.info(f"Resizing image from {image.width}x{image.height} to fit within {MAX_IMAGE_SIZE}")
            image.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
            logger.info(f"Resized to {image.width}x{image.height}")

        # Apply enhancements
        if enhance:
            logger.info("Applying image enhancements")
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)  # Slight contrast boost

            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)  # Slight sharpness boost

        # Apply noise reduction if needed
        if denoise:
            logger.info("Applying noise reduction")
            image = image.filter(ImageFilter.MedianFilter(size=3))

        # Convert to RGB mode if not already
        if image.mode != "RGB":
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert("RGB")

        self.processed_image = image
        logger.info(f"Preprocessing complete. Final dimensions: {image.width}x{image.height}")

        # Update metadata with preprocessing info
        self.metadata.update({
            "preprocessing": {
                "resized": resize and (self.original_image.width > MAX_IMAGE_SIZE[0] or
                                       self.original_image.height > MAX_IMAGE_SIZE[1]),
                "enhanced": enhance,
                "denoised": denoise,
                "final_dimensions": f"{image.width}x{image.height}",
            }
        })

        return image

    def save(self, format: str = "jpg") -> str:
        """
        Save the processed image with metadata.

        Args:
            format: Output format (jpg, png, etc.)

        Returns:
            Path to the saved image file
        """
        if self.processed_image is None:
            raise ImageAcquisitionError("No processed image available. Call preprocess() first.")

        # Ensure format is lowercase and valid
        format = format.lower()
        if format not in SUPPORTED_FORMATS:
            logger.warning(f"Unsupported format '{format}', defaulting to jpg")
            format = "jpg"

        # Generate output filename
        output_filename = f"{self.image_id}_{int(time.time())}.{format}"
        output_path = os.path.join(self.output_dir, output_filename)

        # Prepare metadata file path
        metadata_filename = f"{self.image_id}_metadata.json"
        metadata_path = os.path.join(self.output_dir, metadata_filename)

        try:
            # Save the image
            logger.info(f"Saving processed image to {output_path}")

            # Different save parameters based on format
            if format in ["jpg", "jpeg"]:
                self.processed_image.save(output_path, format="JPEG", quality=self.quality)
            elif format == "png":
                self.processed_image.save(output_path, format="PNG", optimize=True)
            elif format == "webp":
                self.processed_image.save(output_path, format="WEBP", quality=self.quality)
            else:
                self.processed_image.save(output_path)

            # Update metadata with final info
            self.metadata.update({
                "image_id": self.image_id,
                "timestamp": self.timestamp,
                "saved_path": output_path,
                "saved_format": format,
                "saved_quality": self.quality,
                "saved_size": os.path.getsize(output_path),
            })

            # Save metadata as JSON
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)

            logger.info(f"Saved image metadata to {metadata_path}")
            logger.info(f"Image acquisition and processing complete for {self.image_id}")

            return output_path
        except Exception as e:
            logger.error(f"Error saving processed image: {str(e)}")
            raise ImageAcquisitionError(f"Failed to save processed image: {str(e)}") from e

def main():
    """Main entry point for the image acquisition demo."""
    parser = argparse.ArgumentParser(description="Negative Space Imaging - Image Acquisition Demo")
    parser.add_argument("--source", choices=["file", "url", "camera"], default="file",
                        help="Source type for image acquisition")
    parser.add_argument("--path", required=True,
                        help="Path to the source (file path, URL, or camera ID)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory for processed images (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY,
                        help=f"Quality for saved images, 1-100 (default: {DEFAULT_QUALITY})")
    parser.add_argument("--format", choices=SUPPORTED_FORMATS, default="jpg",
                        help="Output format for the processed image (default: jpg)")
    parser.add_argument("--no-resize", action="store_true",
                        help="Disable automatic resizing of large images")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Disable image enhancement")
    parser.add_argument("--no-denoise", action="store_true",
                        help="Disable noise reduction")
    args = parser.parse_args()

    try:
        # Create image acquisition instance
        acquisition = ImageAcquisition(
            source_type=args.source,
            source_path=args.path,
            output_dir=args.output,
            quality=args.quality
        )

        # Acquire the image
        acquisition.acquire()

        # Preprocess the image
        acquisition.preprocess(
            resize=not args.no_resize,
            enhance=not args.no_enhance,
            denoise=not args.no_denoise
        )

        # Save the processed image
        output_path = acquisition.save(format=args.format)

        print(f"\nImage acquisition completed successfully!")
        print(f"Image ID: {acquisition.image_id}")
        print(f"Saved to: {output_path}")
        print(f"Metadata saved to: {os.path.join(args.output, f'{acquisition.image_id}_metadata.json')}")

        return 0
    except Exception as e:
        logger.error(f"Error in image acquisition demo: {str(e)}")
        print(f"\nError: {str(e)}")
        return 1


def acquire_image(
    source: str = None,
    mode: str = "simulation",
    width: int = 512,
    height: int = 512,
    **kwargs
) -> Tuple[bytes, Dict[str, any]]:
    """
    Acquire an image for the Negative Space Imaging project.

    This function provides a simplified interface to acquire images for use
    in the secure imaging workflow. It handles various acquisition modes:
    - Simulation: Generate synthetic test images
    - File: Load from local file
    - URL: Download from remote URL

    Args:
        source: Source identifier (file path, URL, etc.)
        mode: Acquisition mode ("simulation", "file", "url")
        width: Width for simulated images
        height: Height for simulated images
        **kwargs: Additional acquisition parameters

    Returns:
        Tuple of (image_data, metadata)
    """
    try:
        # Map string mode to proper enum if needed
        if mode == "simulation":
            logger.info("Simulating image acquisition...")
            print("Simulating image acquisition...")

            # Generate a synthetic image with negative space patterns
            data = np.zeros((height, width), dtype=np.uint8)

            # Create a background pattern (grayish)
            data.fill(180)

            # Add negative space regions (3 dark circular regions)
            for i in range(3):
                # Random center and radius for each region
                cx = np.random.randint(width // 4, 3 * width // 4)
                cy = np.random.randint(height // 4, 3 * height // 4)
                radius = np.random.randint(30, min(width, height) // 4)

                # Create a circular mask
                y, x = np.ogrid[:height, :width]
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                mask = dist <= radius

                # Apply dark value to create negative space
                data[mask] = 30

                # Add some noise/texture to the region
                noise = np.random.randint(0, 20, size=np.count_nonzero(mask), dtype=np.uint8)
                data[mask] = np.minimum(data[mask] + noise, 255).astype(np.uint8)

            # Convert to bytes
            image_data = data.tobytes()

            # Create metadata
            metadata = {
                "image_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "source": "simulation",
                "width": width,
                "height": height,
                "size_bytes": len(image_data),
                "negative_space_regions": 3,
                "hash": hashlib.sha256(image_data).hexdigest()
            }

        elif mode == "file":
            logger.info(f"Acquiring image from file: {source}")

            if not os.path.exists(source):
                raise FileNotFoundError(f"Image file not found: {source}")

            # Load image file
            with open(source, "rb") as f:
                image_data = f.read()

                # Create metadata
                metadata = {
                    "image_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "source": source,
                    "size_bytes": len(image_data),
                    "hash": hashlib.sha256(image_data).hexdigest()
                }            # Try to get image dimensions if it's a supported format
            try:
                with Image.open(source) as img:
                    metadata["width"] = img.width
                    metadata["height"] = img.height
                    metadata["format"] = img.format
            except Exception:
                # If we can't get dimensions, just continue
                pass

        elif mode == "url":
            logger.info(f"Acquiring image from URL: {source}")

            # This would normally use requests to download the image
            # For now, just simulate it
            logger.warning("URL acquisition is simulated in this demo")

            # Simulate downloading an image
            time.sleep(1)  # Simulate network delay

            # Generate a random image as a placeholder
            data = np.random.randint(0, 255, (height, width), dtype=np.uint8)
            image_data = data.tobytes()

            # Create metadata
            metadata = {
                "image_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "width": width,
                "height": height,
                "size_bytes": len(image_data),
                "hash": hashlib.sha256(image_data).hexdigest()
            }

        else:
            raise ValueError(f"Unsupported acquisition mode: {mode}")

        # Return the acquired image data and metadata
        return image_data, metadata

    except Exception as e:
        logger.error(f"Error during image acquisition: {str(e)}")
        raise


def get_acquisition_metadata():
    """
    Get the metadata from the most recent image acquisition.

    Returns:
        Dictionary containing metadata or None if no acquisition has been performed
    """
    # Look for the most recent metadata file
    metadata_dir = os.path.join(os.getcwd(), "data", "metadata")

    if not os.path.exists(metadata_dir):
        return None

    # List all metadata files
    metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith("_metadata.json")]

    if not metadata_files:
        return None

    # Sort by modification time (most recent first)
    metadata_files.sort(key=lambda f: os.path.getmtime(os.path.join(metadata_dir, f)), reverse=True)

    # Load the most recent metadata
    try:
        with open(os.path.join(metadata_dir, metadata_files[0]), "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading acquisition metadata: {str(e)}")
        return None


if __name__ == "__main__":
    sys.exit(main())
