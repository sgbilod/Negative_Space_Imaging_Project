#!/usr/bin/env python
"""
Image Format Handlers Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides format handlers for various image types:
- DICOM (medical imaging)
- FITS (astronomical imaging)
- RAW camera formats
- TIFF, PNG, JPEG (standard formats)
- HDF5 (large scientific datasets)

Each handler provides:
- Reading and parsing image data
- Writing with format-specific options
- Metadata extraction and embedding
- Format validation
"""

from __future__ import annotations

import io
import logging
import os
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Type, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
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

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    RAWPY_AVAILABLE = False


class ImageFormatType(Enum):
    """Supported image format types."""
    DICOM = "dicom"
    FITS = "fits"
    RAW = "raw"
    TIFF = "tiff"
    PNG = "png"
    JPEG = "jpeg"
    HDF5 = "hdf5"
    UNKNOWN = "unknown"


@dataclass
class ImageMetadata:
    """Container for image metadata."""
    format: ImageFormatType
    width: int
    height: int
    channels: int = 1
    bit_depth: int = 8
    color_space: str = "grayscale"
    creation_date: Optional[datetime] = None
    source: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format": self.format.value,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "color_space": self.color_space,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "source": self.source,
            "custom": self.custom,
        }


class ImageFormatHandler(ABC):
    """Abstract base class for image format handlers."""

    format_type: ImageFormatType = ImageFormatType.UNKNOWN
    extensions: List[str] = []
    
    @abstractmethod
    def read(self, path: Union[str, Path, BinaryIO]) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Read image from file.
        
        Args:
            path: File path or file-like object
            
        Returns:
            Tuple of (image_array, metadata)
        """
        pass

    @abstractmethod
    def write(
        self,
        path: Union[str, Path],
        data: np.ndarray,
        metadata: Optional[ImageMetadata] = None,
        **kwargs: Any,
    ) -> None:
        """
        Write image to file.
        
        Args:
            path: Output file path
            data: Image data as numpy array
            metadata: Optional metadata to embed
            **kwargs: Format-specific options
        """
        pass

    @abstractmethod
    def validate(self, path: Union[str, Path, BinaryIO]) -> bool:
        """
        Validate if file is in the expected format.
        
        Args:
            path: File path or file-like object
            
        Returns:
            True if file is valid for this format
        """
        pass

    @classmethod
    def supports_extension(cls, extension: str) -> bool:
        """Check if handler supports the given extension."""
        return extension.lower().lstrip(".") in [e.lower() for e in cls.extensions]


class DICOMHandler(ImageFormatHandler):
    """Handler for DICOM medical imaging format."""

    format_type = ImageFormatType.DICOM
    extensions = ["dcm", "dicom"]

    def read(self, path: Union[str, Path, BinaryIO]) -> Tuple[np.ndarray, ImageMetadata]:
        """Read DICOM file."""
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom is required for DICOM support")

        ds = pydicom.dcmread(path)
        
        # Get pixel data
        pixel_array = ds.pixel_array.copy()
        
        # Handle windowing if present
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            center = ds.WindowCenter
            width = ds.WindowWidth
            if isinstance(center, pydicom.multival.MultiValue):
                center = center[0]
            if isinstance(width, pydicom.multival.MultiValue):
                width = width[0]
            
            # Apply windowing
            min_val = center - width / 2
            max_val = center + width / 2
            pixel_array = np.clip(pixel_array, min_val, max_val)
        
        # Build metadata
        height, width = pixel_array.shape[:2]
        channels = 1 if len(pixel_array.shape) == 2 else pixel_array.shape[2]
        
        metadata = ImageMetadata(
            format=self.format_type,
            width=width,
            height=height,
            channels=channels,
            bit_depth=int(ds.BitsAllocated) if hasattr(ds, "BitsAllocated") else 16,
            color_space="grayscale" if channels == 1 else "rgb",
            creation_date=None,
            source=str(path) if isinstance(path, (str, Path)) else None,
            custom={
                "patient_id": getattr(ds, "PatientID", None),
                "modality": getattr(ds, "Modality", None),
                "study_description": getattr(ds, "StudyDescription", None),
                "series_description": getattr(ds, "SeriesDescription", None),
            },
        )
        
        return pixel_array, metadata

    def write(
        self,
        path: Union[str, Path],
        data: np.ndarray,
        metadata: Optional[ImageMetadata] = None,
        **kwargs: Any,
    ) -> None:
        """Write DICOM file."""
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom is required for DICOM support")

        # Create new DICOM dataset
        ds = pydicom.Dataset()
        
        # File meta information
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta = file_meta
        
        # Required DICOM elements
        ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.PatientName = kwargs.get("patient_name", "Anonymous")
        ds.PatientID = kwargs.get("patient_id", "000000")
        ds.Modality = kwargs.get("modality", "OT")
        
        # Image properties
        ds.Rows, ds.Columns = data.shape[:2]
        ds.SamplesPerPixel = 1 if len(data.shape) == 2 else data.shape[2]
        ds.PhotometricInterpretation = "MONOCHROME2" if ds.SamplesPerPixel == 1 else "RGB"
        ds.BitsAllocated = 16 if data.dtype == np.uint16 else 8
        ds.BitsStored = ds.BitsAllocated
        ds.HighBit = ds.BitsStored - 1
        ds.PixelRepresentation = 0
        
        # Pixel data
        ds.PixelData = data.tobytes()
        
        # Save
        ds.save_as(str(path))
        logger.info(f"DICOM file saved to {path}")

    def validate(self, path: Union[str, Path, BinaryIO]) -> bool:
        """Validate DICOM file."""
        if not DICOM_AVAILABLE:
            return False
        
        try:
            pydicom.dcmread(path, stop_before_pixels=True)
            return True
        except Exception:
            return False


class FITSHandler(ImageFormatHandler):
    """Handler for FITS astronomical imaging format."""

    format_type = ImageFormatType.FITS
    extensions = ["fits", "fit", "fts"]

    def read(self, path: Union[str, Path, BinaryIO]) -> Tuple[np.ndarray, ImageMetadata]:
        """Read FITS file."""
        if not FITS_AVAILABLE:
            raise ImportError("astropy is required for FITS support")

        with fits.open(path) as hdul:
            # Get primary HDU or first image HDU
            for hdu in hdul:
                if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)) and hdu.data is not None:
                    data = hdu.data.copy()
                    header = hdu.header
                    break
            else:
                raise ValueError("No image data found in FITS file")
        
        # Handle dimensions
        if len(data.shape) == 2:
            height, width = data.shape
            channels = 1
        elif len(data.shape) == 3:
            channels, height, width = data.shape
            # Transpose to HWC format
            data = np.transpose(data, (1, 2, 0))
        else:
            raise ValueError(f"Unsupported FITS array shape: {data.shape}")
        
        # Build metadata
        metadata = ImageMetadata(
            format=self.format_type,
            width=width,
            height=height,
            channels=channels,
            bit_depth=data.dtype.itemsize * 8,
            color_space="grayscale" if channels == 1 else "rgb",
            source=str(path) if isinstance(path, (str, Path)) else None,
            custom={
                "object": header.get("OBJECT"),
                "telescope": header.get("TELESCOP"),
                "instrument": header.get("INSTRUME"),
                "exposure_time": header.get("EXPTIME"),
                "filter": header.get("FILTER"),
                "ra": header.get("RA"),
                "dec": header.get("DEC"),
            },
        )
        
        return data, metadata

    def write(
        self,
        path: Union[str, Path],
        data: np.ndarray,
        metadata: Optional[ImageMetadata] = None,
        **kwargs: Any,
    ) -> None:
        """Write FITS file."""
        if not FITS_AVAILABLE:
            raise ImportError("astropy is required for FITS support")

        # Create primary HDU
        hdu = fits.PrimaryHDU(data)
        
        # Add header keywords
        if metadata:
            for key, value in metadata.custom.items():
                if value is not None and len(key) <= 8:
                    hdu.header[key.upper()] = value
        
        # Add standard keywords
        hdu.header["ORIGIN"] = "Negative Space Imaging"
        hdu.header["DATE"] = datetime.utcnow().isoformat()
        
        for key, value in kwargs.items():
            if value is not None and len(key) <= 8:
                hdu.header[key.upper()] = value
        
        # Write file
        hdul = fits.HDUList([hdu])
        hdul.writeto(str(path), overwrite=True)
        logger.info(f"FITS file saved to {path}")

    def validate(self, path: Union[str, Path, BinaryIO]) -> bool:
        """Validate FITS file."""
        if not FITS_AVAILABLE:
            return False
        
        try:
            with fits.open(path):
                return True
        except Exception:
            return False


class TIFFHandler(ImageFormatHandler):
    """Handler for TIFF format."""

    format_type = ImageFormatType.TIFF
    extensions = ["tiff", "tif"]

    def read(self, path: Union[str, Path, BinaryIO]) -> Tuple[np.ndarray, ImageMetadata]:
        """Read TIFF file."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for TIFF support")

        with Image.open(path) as img:
            data = np.array(img)
            
            height, width = data.shape[:2]
            channels = 1 if len(data.shape) == 2 else data.shape[2]
            
            metadata = ImageMetadata(
                format=self.format_type,
                width=width,
                height=height,
                channels=channels,
                bit_depth=data.dtype.itemsize * 8,
                color_space=img.mode,
                source=str(path) if isinstance(path, (str, Path)) else None,
            )
            
        return data, metadata

    def write(
        self,
        path: Union[str, Path],
        data: np.ndarray,
        metadata: Optional[ImageMetadata] = None,
        **kwargs: Any,
    ) -> None:
        """Write TIFF file."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for TIFF support")

        img = Image.fromarray(data)
        
        compression = kwargs.get("compression", "lzw")
        img.save(str(path), compression=compression if compression else None)
        logger.info(f"TIFF file saved to {path}")

    def validate(self, path: Union[str, Path, BinaryIO]) -> bool:
        """Validate TIFF file."""
        if not PIL_AVAILABLE:
            return False
        
        try:
            with Image.open(path) as img:
                return img.format == "TIFF"
        except Exception:
            return False


class PNGHandler(ImageFormatHandler):
    """Handler for PNG format."""

    format_type = ImageFormatType.PNG
    extensions = ["png"]

    def read(self, path: Union[str, Path, BinaryIO]) -> Tuple[np.ndarray, ImageMetadata]:
        """Read PNG file."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for PNG support")

        with Image.open(path) as img:
            data = np.array(img)
            
            height, width = data.shape[:2]
            channels = 1 if len(data.shape) == 2 else data.shape[2]
            
            metadata = ImageMetadata(
                format=self.format_type,
                width=width,
                height=height,
                channels=channels,
                bit_depth=8,
                color_space=img.mode,
                source=str(path) if isinstance(path, (str, Path)) else None,
            )
            
        return data, metadata

    def write(
        self,
        path: Union[str, Path],
        data: np.ndarray,
        metadata: Optional[ImageMetadata] = None,
        **kwargs: Any,
    ) -> None:
        """Write PNG file."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for PNG support")

        img = Image.fromarray(data)
        img.save(str(path), format="PNG")
        logger.info(f"PNG file saved to {path}")

    def validate(self, path: Union[str, Path, BinaryIO]) -> bool:
        """Validate PNG file."""
        if not PIL_AVAILABLE:
            return False
        
        try:
            with Image.open(path) as img:
                return img.format == "PNG"
        except Exception:
            return False


class JPEGHandler(ImageFormatHandler):
    """Handler for JPEG format."""

    format_type = ImageFormatType.JPEG
    extensions = ["jpg", "jpeg"]

    def read(self, path: Union[str, Path, BinaryIO]) -> Tuple[np.ndarray, ImageMetadata]:
        """Read JPEG file."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for JPEG support")

        with Image.open(path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            data = np.array(img)
            
            height, width = data.shape[:2]
            channels = data.shape[2] if len(data.shape) > 2 else 1
            
            metadata = ImageMetadata(
                format=self.format_type,
                width=width,
                height=height,
                channels=channels,
                bit_depth=8,
                color_space="RGB",
                source=str(path) if isinstance(path, (str, Path)) else None,
            )
            
        return data, metadata

    def write(
        self,
        path: Union[str, Path],
        data: np.ndarray,
        metadata: Optional[ImageMetadata] = None,
        **kwargs: Any,
    ) -> None:
        """Write JPEG file."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for JPEG support")

        img = Image.fromarray(data)
        quality = kwargs.get("quality", 95)
        img.save(str(path), format="JPEG", quality=quality)
        logger.info(f"JPEG file saved to {path}")

    def validate(self, path: Union[str, Path, BinaryIO]) -> bool:
        """Validate JPEG file."""
        if not PIL_AVAILABLE:
            return False
        
        try:
            with Image.open(path) as img:
                return img.format == "JPEG"
        except Exception:
            return False


class HDF5Handler(ImageFormatHandler):
    """Handler for HDF5 format for large datasets."""

    format_type = ImageFormatType.HDF5
    extensions = ["h5", "hdf5", "hdf"]

    def read(
        self,
        path: Union[str, Path, BinaryIO],
        dataset: str = "image",
    ) -> Tuple[np.ndarray, ImageMetadata]:
        """Read HDF5 file."""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 support")

        with h5py.File(path, "r") as f:
            if dataset not in f:
                # Try to find image data
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        dataset = key
                        break
                else:
                    raise ValueError(f"No dataset found in HDF5 file")
            
            data = f[dataset][:]
            attrs = dict(f[dataset].attrs)
        
        height, width = data.shape[:2]
        channels = 1 if len(data.shape) == 2 else data.shape[2]
        
        metadata = ImageMetadata(
            format=self.format_type,
            width=width,
            height=height,
            channels=channels,
            bit_depth=data.dtype.itemsize * 8,
            color_space="grayscale" if channels == 1 else "rgb",
            source=str(path) if isinstance(path, (str, Path)) else None,
            custom=attrs,
        )
        
        return data, metadata

    def write(
        self,
        path: Union[str, Path],
        data: np.ndarray,
        metadata: Optional[ImageMetadata] = None,
        dataset: str = "image",
        **kwargs: Any,
    ) -> None:
        """Write HDF5 file."""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 support")

        compression = kwargs.get("compression", "gzip")
        compression_opts = kwargs.get("compression_opts", 4)
        
        with h5py.File(str(path), "w") as f:
            dset = f.create_dataset(
                dataset,
                data=data,
                compression=compression,
                compression_opts=compression_opts,
            )
            
            # Add metadata as attributes
            if metadata:
                for key, value in metadata.custom.items():
                    if value is not None:
                        try:
                            dset.attrs[key] = value
                        except TypeError:
                            dset.attrs[key] = str(value)
            
            dset.attrs["created"] = datetime.utcnow().isoformat()
        
        logger.info(f"HDF5 file saved to {path}")

    def validate(self, path: Union[str, Path, BinaryIO]) -> bool:
        """Validate HDF5 file."""
        if not HDF5_AVAILABLE:
            return False
        
        try:
            with h5py.File(path, "r"):
                return True
        except Exception:
            return False


class RAWHandler(ImageFormatHandler):
    """Handler for camera RAW formats."""

    format_type = ImageFormatType.RAW
    extensions = ["raw", "cr2", "cr3", "nef", "arw", "dng", "orf", "rw2"]

    def read(self, path: Union[str, Path, BinaryIO]) -> Tuple[np.ndarray, ImageMetadata]:
        """Read RAW file."""
        if RAWPY_AVAILABLE:
            return self._read_with_rawpy(path)
        else:
            return self._read_raw_bytes(path)

    def _read_with_rawpy(self, path: Union[str, Path, BinaryIO]) -> Tuple[np.ndarray, ImageMetadata]:
        """Read RAW file using rawpy."""
        with rawpy.imread(str(path)) as raw:
            # Get RGB image
            data = raw.postprocess()
            
            height, width = data.shape[:2]
            channels = data.shape[2] if len(data.shape) > 2 else 1
            
            metadata = ImageMetadata(
                format=self.format_type,
                width=width,
                height=height,
                channels=channels,
                bit_depth=16,
                color_space="RGB",
                source=str(path) if isinstance(path, (str, Path)) else None,
                custom={
                    "camera_make": raw.camera_make,
                    "camera_model": raw.camera_model,
                    "iso": raw.raw_value(raw.raw_pattern[0, 0], 0, 0),
                },
            )
            
        return data, metadata

    def _read_raw_bytes(self, path: Union[str, Path, BinaryIO]) -> Tuple[np.ndarray, ImageMetadata]:
        """Read RAW file as bytes (fallback)."""
        if isinstance(path, (str, Path)):
            with open(path, "rb") as f:
                raw_bytes = f.read()
        else:
            raw_bytes = path.read()
        
        # Basic RAW reading - assumes 16-bit grayscale
        data = np.frombuffer(raw_bytes, dtype=np.uint16)
        
        # Try to infer dimensions
        size = int(np.sqrt(len(data)))
        if size * size == len(data):
            data = data.reshape(size, size)
        else:
            # Can't determine dimensions
            data = data.reshape(-1, 1)
        
        height, width = data.shape[:2]
        
        metadata = ImageMetadata(
            format=self.format_type,
            width=width,
            height=height,
            channels=1,
            bit_depth=16,
            color_space="grayscale",
            source=str(path) if isinstance(path, (str, Path)) else None,
        )
        
        return data, metadata

    def write(
        self,
        path: Union[str, Path],
        data: np.ndarray,
        metadata: Optional[ImageMetadata] = None,
        **kwargs: Any,
    ) -> None:
        """Write RAW file (as raw bytes)."""
        with open(str(path), "wb") as f:
            f.write(data.tobytes())
        logger.info(f"RAW file saved to {path}")

    def validate(self, path: Union[str, Path, BinaryIO]) -> bool:
        """Validate RAW file."""
        if RAWPY_AVAILABLE:
            try:
                with rawpy.imread(str(path)):
                    return True
            except Exception:
                return False
        
        # Fallback: check file extension
        if isinstance(path, (str, Path)):
            ext = Path(path).suffix.lower().lstrip(".")
            return ext in self.extensions
        return False


class ImageFormatRegistry:
    """Registry for image format handlers."""

    _handlers: Dict[ImageFormatType, Type[ImageFormatHandler]] = {
        ImageFormatType.DICOM: DICOMHandler,
        ImageFormatType.FITS: FITSHandler,
        ImageFormatType.TIFF: TIFFHandler,
        ImageFormatType.PNG: PNGHandler,
        ImageFormatType.JPEG: JPEGHandler,
        ImageFormatType.HDF5: HDF5Handler,
        ImageFormatType.RAW: RAWHandler,
    }

    @classmethod
    def get_handler(cls, format_type: ImageFormatType) -> ImageFormatHandler:
        """Get handler for format type."""
        handler_class = cls._handlers.get(format_type)
        if handler_class is None:
            raise ValueError(f"No handler for format: {format_type}")
        return handler_class()

    @classmethod
    def get_handler_for_file(cls, path: Union[str, Path]) -> ImageFormatHandler:
        """Get appropriate handler based on file extension."""
        ext = Path(path).suffix.lower().lstrip(".")
        
        for handler_class in cls._handlers.values():
            if handler_class.supports_extension(ext):
                return handler_class()
        
        raise ValueError(f"No handler for extension: {ext}")

    @classmethod
    def read_image(cls, path: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """Read image using appropriate handler."""
        handler = cls.get_handler_for_file(path)
        return handler.read(path)

    @classmethod
    def write_image(
        cls,
        path: Union[str, Path],
        data: np.ndarray,
        metadata: Optional[ImageMetadata] = None,
        **kwargs: Any,
    ) -> None:
        """Write image using appropriate handler."""
        handler = cls.get_handler_for_file(path)
        handler.write(path, data, metadata, **kwargs)


# Convenience functions
def read_image(path: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
    """Read image from file."""
    return ImageFormatRegistry.read_image(path)


def write_image(
    path: Union[str, Path],
    data: np.ndarray,
    metadata: Optional[ImageMetadata] = None,
    **kwargs: Any,
) -> None:
    """Write image to file."""
    ImageFormatRegistry.write_image(path, data, metadata, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Image Format Handlers")
    print("=" * 40)
    print("\nSupported formats:")
    for fmt, handler_class in ImageFormatRegistry._handlers.items():
        handler = handler_class()
        exts = ", ".join(handler.extensions)
        print(f"  {fmt.value}: {exts}")
    
    print("\nLibrary availability:")
    print(f"  PIL: {PIL_AVAILABLE}")
    print(f"  pydicom: {DICOM_AVAILABLE}")
    print(f"  astropy: {FITS_AVAILABLE}")
    print(f"  h5py: {HDF5_AVAILABLE}")
    print(f"  rawpy: {RAWPY_AVAILABLE}")
