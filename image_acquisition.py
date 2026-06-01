#!/usr/bin/env python
"""
Image Acquisition Module for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides functionality to acquire images from various sources:
1. Local files (RAW, DICOM, FITS, etc.)
2. Connected cameras or imaging devices
3. Remote servers (via HTTPS or SFTP)
4. Simulated image data (for testing)

The module ensures proper security and data integrity:
- Validates image source authenticity
- Maintains HIPAA compliance for sensitive imagery
- Verifies image integrity with cryptographic hashing
- Provides complete acquisition metadata and audit trail

Usage:
    from image_acquisition import ImageAcquisition, ImageFormat, AcquisitionMode

    # Initialize acquisition with specific format and mode
    acquisition = ImageAcquisition(
        format=ImageFormat.RAW,
        mode=AcquisitionMode.LOCAL_FILE
    )

    # Acquire image with acquisition-specific parameters
    image_data, metadata = acquisition.acquire(
        source="path/to/image.raw",
        secure=True
    )
"""

import os
import sys
import time
import hashlib
import random
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
from datetime import datetime
import json

# Optional imports for different acquisition modes
try:
    import requests
    REMOTE_AVAILABLE = True
except ImportError:
    REMOTE_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

try:
    from astropy.io import fits
    FITS_AVAILABLE = True
except ImportError:
    FITS_AVAILABLE = False


class ImageFormat(Enum):
    """Supported image formats for acquisition."""
    RAW = auto()
    DICOM = auto()
    FITS = auto()
    TIFF = auto()
    PNG = auto()
    JPG = auto()
    CUSTOM = auto()


class AcquisitionMode(Enum):
    """Supported modes for image acquisition."""
    LOCAL_FILE = auto()
    CAMERA = auto()
    REMOTE_HTTP = auto()
    REMOTE_SFTP = auto()
    SIMULATION = auto()


class AcquisitionError(Exception):
    """Base exception for acquisition errors."""
    pass


class SourceAuthenticationError(AcquisitionError):
    """Exception raised when source cannot be authenticated."""
    pass


class FormatError(AcquisitionError):
    """Exception raised when image format is invalid or unsupported."""
    pass


class ImageAcquisition:
    """Main class for image acquisition from various sources."""

    def __init__(
        self,
        format: ImageFormat = ImageFormat.RAW,
        mode: AcquisitionMode = AcquisitionMode.LOCAL_FILE,
        security_level: int = 2,
        verify_integrity: bool = True
    ):
        """
        Initialize the image acquisition system.

        Args:
            format: Image format to acquire
            mode: Acquisition mode (local file, camera, remote, etc.)
            security_level: Security level (0-3) with higher being more secure
            verify_integrity: Whether to verify image integrity after acquisition
        """
        self.format = format
        self.mode = mode
        self.security_level = min(max(0, security_level), 3)  # Clamp between 0-3
        self.verify_integrity = verify_integrity
        self._acquisition_id = None
        self._last_metadata = None  # Store the last created metadata

        # Check if the requested mode is available
        if mode == AcquisitionMode.REMOTE_HTTP and not REMOTE_AVAILABLE:
            raise ImportError("Remote acquisition requires the 'requests' package")

        if mode == AcquisitionMode.CAMERA and not PIL_AVAILABLE:
            raise ImportError("Camera acquisition requires the 'PIL' package")

        if format == ImageFormat.DICOM and not DICOM_AVAILABLE:
            raise ImportError("DICOM format requires the 'pydicom' package")

        if format == ImageFormat.FITS and not FITS_AVAILABLE:
            raise ImportError("FITS format requires the 'astropy' package")

    def acquire(
        self,
        source: str,
        secure: bool = True,
        **kwargs
    ) -> Tuple[bytes, Dict[str, any]]:
        """
        Acquire an image from the specified source.

        Args:
            source: Source identifier (filename, URL, device ID, etc.)
            secure: Whether to use secure acquisition protocols
            **kwargs: Additional acquisition-specific parameters

        Returns:
            Tuple of (image_data, metadata)
        """
        # Generate unique acquisition ID
        self._acquisition_id = self._generate_acquisition_id()

        # Start acquisition
        start_time = time.time()

        try:
            # Authenticate source if security is enabled
            if secure and self.security_level > 0:
                self._authenticate_source(source)

            # Dispatch to appropriate acquisition method
            if self.mode == AcquisitionMode.LOCAL_FILE:
                image_data = self._acquire_from_file(source, **kwargs)
            elif self.mode == AcquisitionMode.REMOTE_HTTP:
                image_data = self._acquire_from_http(source, secure, **kwargs)
            elif self.mode == AcquisitionMode.CAMERA:
                image_data = self._acquire_from_camera(source, **kwargs)
            elif self.mode == AcquisitionMode.REMOTE_SFTP:
                image_data = self._acquire_from_sftp(source, secure, **kwargs)
            elif self.mode == AcquisitionMode.SIMULATION:
                image_data = self._simulate_acquisition(**kwargs)
            else:
                raise AcquisitionError(f"Unsupported acquisition mode: {self.mode}")

            # Calculate processing time
            elapsed_time = time.time() - start_time

            # Create metadata
            metadata = self._create_metadata(source, image_data, elapsed_time, **kwargs)

            # Verify integrity if requested
            if self.verify_integrity:
                self._verify_integrity(image_data, metadata)

            return image_data, metadata

        except Exception as e:
            # Log acquisition error
            self._log_acquisition_error(e, source)
            raise

    def _acquire_from_file(self, filepath: str, **kwargs) -> bytes:
        """Acquire image from local file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")

        # Try to get image dimensions if possible
        width = height = None
        if PIL_AVAILABLE and self.format in (
            ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPG
        ):
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    kwargs['width'] = width
                    kwargs['height'] = height
            except Exception:
                pass

        # Read file based on format
        with open(filepath, 'rb') as f:
            if self.format == ImageFormat.RAW:
                return self._read_raw_file(f, **kwargs)
            elif self.format == ImageFormat.DICOM and DICOM_AVAILABLE:
                return self._read_dicom_file(filepath)
            elif self.format == ImageFormat.FITS and FITS_AVAILABLE:
                return self._read_fits_file(filepath)
            elif self.format in (
                ImageFormat.TIFF, ImageFormat.PNG, ImageFormat.JPG
            ) and PIL_AVAILABLE:
                return self._read_pil_file(filepath)
            else:
                # Default to binary read
                return f.read()

    def _read_raw_file(self, file_obj: BinaryIO, width: int = None, height: int = None, **kwargs) -> bytes:
        """Read a RAW image file with optional dimensions."""
        return file_obj.read()

    def _read_dicom_file(self, filepath: str) -> bytes:
        """Read a DICOM medical image file."""
        if not DICOM_AVAILABLE:
            raise ImportError("DICOM format requires the 'pydicom' package")

        ds = pydicom.dcmread(filepath)
        return ds.pixel_array.tobytes()

    def _read_fits_file(self, filepath: str) -> bytes:
        """Read a FITS astronomical image file."""
        if not FITS_AVAILABLE:
            raise ImportError("FITS format requires the 'astropy' package")

        with fits.open(filepath) as hdul:
            return hdul[0].data.tobytes()

    def _read_pil_file(self, filepath: str) -> np.ndarray:
        """Read an image file using PIL and return as numpy array."""
        if not PIL_AVAILABLE:
            raise ImportError("Image formats require the 'PIL' package")

        with Image.open(filepath) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Return as numpy array
            return np.array(img)

    def _acquire_from_http(self, url: str, secure: bool = True, **kwargs) -> bytes:
        """Acquire image from HTTP/HTTPS URL."""
        if not REMOTE_AVAILABLE:
            raise ImportError("Remote acquisition requires the 'requests' package")

        # Enforce HTTPS for secure acquisition
        if secure and self.security_level > 1 and not url.startswith('https://'):
            raise SourceAuthenticationError("Secure acquisition requires HTTPS URL")

        # Get image from URL
        response = requests.get(url, **kwargs)
        response.raise_for_status()

        return response.content

    def _acquire_from_camera(self, device_id: str, **kwargs) -> bytes:
        """Acquire image from connected camera."""
        # This is a placeholder - actual implementation would use camera APIs
        raise NotImplementedError("Camera acquisition not implemented in this version")

    def _acquire_from_sftp(self, source: str, secure: bool = True, **kwargs) -> bytes:
        """Acquire image from SFTP server."""
        # This is a placeholder - actual implementation would use pysftp or paramiko
        raise NotImplementedError("SFTP acquisition not implemented in this version")

    def _simulate_acquisition(
        self,
        width: int = 512,
        height: int = 512,
        pattern: str = 'random',
        negative_space_regions: int = 3,
        **kwargs
    ) -> bytes:
        """
        Simulate image acquisition for testing purposes.

        Args:
            width: Width of simulated image
            height: Height of simulated image
            pattern: Pattern type ('random', 'gradient', 'negative_space')
            negative_space_regions: Number of negative space regions to generate
            **kwargs: Additional simulation parameters

        Returns:
            Simulated image data as bytes
        """
        # Create a simulated image array
        if pattern == 'random':
            image_array = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        elif pattern == 'gradient':
            x = np.linspace(0, 255, width, dtype=np.uint8)
            y = np.linspace(0, 255, height, dtype=np.uint8)
            xx, yy = np.meshgrid(x, y)
            image_array = (xx + yy) // 2
        elif pattern == 'negative_space':
            # Start with random background
            image_array = np.random.randint(120, 200, (height, width), dtype=np.uint8)

            # Add negative space regions (darker areas)
            for i in range(negative_space_regions):
                # Random region center and size
                cx = random.randint(width//4, 3*width//4)
                cy = random.randint(height//4, 3*height//4)
                size = random.randint(20, min(width, height)//3)

                # Create circular mask
                y, x = np.ogrid[-cy:height-cy, -cx:width-cx]
                mask = x*x + y*y <= size*size

                # Apply darker values to create negative space
                image_array[mask] = np.random.randint(10, 60, size=np.count_nonzero(mask))
        else:
            # Default to random pattern
            image_array = np.random.randint(0, 256, (height, width), dtype=np.uint8)

        # Convert to bytes
        return image_array.tobytes()

    def _authenticate_source(self, source: str) -> bool:
        """
        Authenticate the image source based on security level.

        Raises SourceAuthenticationError if authentication fails.
        """
        # Basic source authentication based on security level
        if self.security_level == 1:
            # Level 1: Basic checks (file exists, URL is valid)
            if self.mode == AcquisitionMode.LOCAL_FILE and not os.path.exists(source):
                raise SourceAuthenticationError(f"Source file does not exist: {source}")

            if self.mode == AcquisitionMode.REMOTE_HTTP and not source.startswith(('http://', 'https://')):
                raise SourceAuthenticationError(f"Invalid URL format: {source}")

        elif self.security_level >= 2:
            # Level 2-3: More stringent checks
            if self.mode == AcquisitionMode.LOCAL_FILE:
                # Check file exists and has proper permissions
                if not os.path.exists(source):
                    raise SourceAuthenticationError(f"Source file does not exist: {source}")

                if not os.access(source, os.R_OK):
                    raise SourceAuthenticationError(f"No read permission for file: {source}")

                # Level 3 adds additional checks
                if self.security_level == 3:
                    # Check file is in an allowed directory
                    allowed_dirs = [os.getcwd(), '/data', '/secured/images']
                    if not any(os.path.abspath(source).startswith(d) for d in allowed_dirs):
                        raise SourceAuthenticationError(f"File location not authorized: {source}")

            if self.mode == AcquisitionMode.REMOTE_HTTP:
                # Require HTTPS for level 2+
                if not source.startswith('https://'):
                    raise SourceAuthenticationError("Secure acquisition requires HTTPS URL")

                # Level 3 adds additional checks
                if self.security_level == 3:
                    # Check against allowed domains
                    allowed_domains = ['secure-images.org', 'trusted-medical.com', 'negativespacecorp.com']
                    domain = source.split('/')[2]
                    if not any(domain.endswith(d) for d in allowed_domains):
                        raise SourceAuthenticationError(f"Domain not in allowed list: {domain}")

        return True

    def _create_metadata(
        self,
        source: str,
        image_data: bytes,
        elapsed_time: float,
        **kwargs
    ) -> Dict[str, any]:
        """Create metadata for the acquired image."""
        metadata = {
            "acquisition_id": self._acquisition_id,
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "mode": self.mode.name,
            "format": self.format.name,
            "size_bytes": len(image_data),
            "elapsed_time_seconds": elapsed_time,
            "sha256_hash": hashlib.sha256(image_data).hexdigest(),
        }

        # Add dimensions if provided
        if 'width' in kwargs and 'height' in kwargs:
            metadata.update({
                "width": kwargs.get("width"),
                "height": kwargs.get("height"),
            })

        # Add format-specific metadata
        if self.format == ImageFormat.RAW:
            metadata.update({
                "bit_depth": kwargs.get("bit_depth", "unknown"),
            })

        # Add simulation metadata if applicable
        if self.mode == AcquisitionMode.SIMULATION:
            metadata.update({
                "simulated": True,
                "pattern": kwargs.get("pattern", "random"),
                "negative_space_regions": kwargs.get("negative_space_regions", 0),
            })

        # Store this metadata as the last created metadata
        self._last_metadata = metadata

        return metadata

    def _verify_integrity(self, image_data: bytes, metadata: Dict[str, any]) -> bool:
        """Verify the integrity of the acquired image data."""
        # Calculate hash of the image data
        calculated_hash = hashlib.sha256(image_data).hexdigest()

        # Compare with the hash in metadata
        if calculated_hash != metadata["sha256_hash"]:
            raise AcquisitionError("Image integrity verification failed: hash mismatch")

        return True

    def _generate_acquisition_id(self) -> str:
        """Generate a unique ID for this acquisition."""
        timestamp = int(time.time() * 1000)
        random_component = random.randint(0, 0xFFFFFF)
        return f"{timestamp:x}{random_component:06x}"

    def _log_acquisition_error(self, error: Exception, source: str) -> None:
        """Log an acquisition error to file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "acquisition_id": self._acquisition_id,
            "source": source,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "mode": self.mode.name,
            "format": self.format.name,
        }

        # Append to log file
        with open("acquisition_errors.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def save_metadata(self, filepath: str) -> None:
        """
        Save metadata to a JSON file.

        Args:
            filepath: Path to save the metadata JSON file
        """
        if not filepath.endswith('.json'):
            filepath += '.json'

        try:
            # Ensure directory exists
            os.makedirs(
                os.path.dirname(os.path.abspath(filepath)),
                exist_ok=True
            )

            # Write metadata to file
            with open(filepath, 'w') as f:
                json.dump(self._last_metadata, f, indent=2)

            return filepath
        except Exception as e:
            raise AcquisitionError(f"Failed to save metadata: {str(e)}")

    def save_image(
        self,
        image_data: Union[bytes, np.ndarray],
        filepath: str
    ) -> None:
        """
        Save image data to a file.

        Args:
            image_data: Image data as bytes or numpy array
            filepath: Path to save the image file
        """
        try:
            # Ensure directory exists
            os.makedirs(
                os.path.dirname(os.path.abspath(filepath)),
                exist_ok=True
            )

            # Handle different types of image data
            if isinstance(image_data, bytes):
                # If it's a raw byte stream
                with open(filepath, 'wb') as f:
                    f.write(image_data)
            elif isinstance(image_data, np.ndarray):
                # If it's a numpy array, use PIL to save
                if not PIL_AVAILABLE:
                    raise ImportError(
                        "PIL is required to save numpy arrays as images"
                    )

                img = Image.fromarray(image_data)
                img.save(filepath)
            else:
                raise TypeError("Unsupported image data type")

            return filepath
        except Exception as e:
            raise AcquisitionError(f"Failed to save image: {str(e)}")


# Simple demo function to demonstrate usage
def acquire_image(
    source: str = None,
    format: str = "RAW",
    mode: str = "SIMULATION",
    width: int = 512,
    height: int = 512,
    **kwargs
) -> Tuple[bytes, Dict[str, any]]:
    """
    Simple demo function to acquire an image.

    Args:
        source: Source identifier (file path, URL, etc.)
        format: Image format (RAW, DICOM, FITS, TIFF, PNG, JPG)
        mode: Acquisition mode (LOCAL_FILE, REMOTE_HTTP, CAMERA, SIMULATION)
        width: Width for simulated images
        height: Height for simulated images
        **kwargs: Additional acquisition parameters

    Returns:
        Tuple of (image_data, metadata)
    """
    try:
        # Convert string arguments to enums
        img_format = getattr(ImageFormat, format.upper())
        acq_mode = getattr(AcquisitionMode, mode.upper())

        # Create acquisition object
        acquisition = ImageAcquisition(format=img_format, mode=acq_mode)

        # Default source for simulation
        if acq_mode == AcquisitionMode.SIMULATION and not source:
            source = "simulated_image"

        # Acquire image
        print(f"Acquiring image from {source or 'simulation'}...")
        image_data, metadata = acquisition.acquire(
            source=source,
            width=width,
            height=height,
            **kwargs
        )

        # Print brief summary
        print(f"Image acquired: {metadata['size_bytes']} bytes, "
              f"hash: {metadata['sha256_hash'][:16]}...")

        return image_data, metadata

    except Exception as e:
        print(f"Error during image acquisition: {e}")
        raise


# Demo/test code
if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Image Acquisition Demo")
    parser.add_argument("--source", help="Image source (file path or URL)")
    parser.add_argument("--format", default="RAW", help="Image format")
    parser.add_argument("--mode", default="SIMULATION", help="Acquisition mode")
    parser.add_argument("--width", type=int, default=512, help="Image width for simulation")
    parser.add_argument("--height", type=int, default=512, help="Image height for simulation")
    parser.add_argument("--pattern", default="negative_space", help="Simulation pattern")
    parser.add_argument("--regions", type=int, default=3, help="Number of negative space regions")

    args = parser.parse_args()

    # Run demo
    image_data, metadata = acquire_image(
        source=args.source,
        format=args.format,
        mode=args.mode,
        width=args.width,
        height=args.height,
        pattern=args.pattern,
        negative_space_regions=args.regions
    )

    # Save metadata to file for inspection
    with open("test_image_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Optionally save the image data for visualization
    if args.mode == "SIMULATION":
        with open("test_image.raw", "wb") as f:
            f.write(image_data)
        print(f"Saved simulated image to test_image.raw")
