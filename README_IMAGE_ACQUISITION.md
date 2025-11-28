# Image Acquisition Pipeline

**Copyright © 2025 Stephen Bilodeau. All rights reserved.**

This document provides comprehensive documentation for the image acquisition system in the Negative Space Imaging Project.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Supported Image Formats](#supported-image-formats)
- [Acquisition Modes](#acquisition-modes)
- [Data Ingestion Workflows](#data-ingestion-workflows)
- [Configuration Options](#configuration-options)
- [Integration with External Systems](#integration-with-external-systems)
- [Security and Compliance](#security-and-compliance)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Image Acquisition Pipeline is a core component of the Negative Space Imaging System. It handles the secure capture, validation, and ingestion of images from multiple sources. The system is designed to support both medical imaging (HIPAA-compliant) and astronomical data acquisition workflows.

### Key Capabilities

- **Multi-source acquisition**: Local files, cameras, remote servers, simulated data
- **Multiple format support**: RAW, DICOM, FITS, TIFF, PNG, JPEG
- **Security-first design**: Cryptographic integrity verification and source authentication
- **Real-time processing**: Thread-based architecture for concurrent acquisition
- **Complete audit trail**: Full metadata and logging for compliance

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Image Acquisition Pipeline                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   Local Files    │   │  Remote Sources  │   │    Cameras       │
│                  │   │                  │   │                  │
│  • RAW images    │   │  • HTTP/HTTPS    │   │  • USB devices   │
│  • DICOM files   │   │  • SFTP servers  │   │  • IP cameras    │
│  • FITS data     │   │  • S3 buckets    │   │  • Telescopes    │
│  • TIFF/PNG/JPG  │   │  • Kafka streams │   │  • Medical scanners│
└────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │  Source Validation   │
                    │  & Authentication    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Format Detection    │
                    │  & Preprocessing     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Integrity Check     │
                    │  (SHA-256 Hash)      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Metadata Creation   │
                    │  & Audit Logging     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Processing Queue   │
                    │   (Redis/Bull)       │
                    └──────────────────────┘
```

### Components

| Component | File | Description |
|-----------|------|-------------|
| **ImageAcquisition** | `image_acquisition.py` | Main acquisition class with multi-source support |
| **ImageFormat** | `image_acquisition.py` | Enum defining supported image formats |
| **AcquisitionMode** | `image_acquisition.py` | Enum defining acquisition modes |
| **RealtimePreprocessor** | `realtime_preprocessing.py` | Thread-based real-time preprocessing |
| **AcquisitionProfiles** | `acquisition_profiles.py` | JSON-based configuration profiles |
| **IntegratedSystem** | `integrated_acquisition_system.py` | Unified acquisition system |

---

## Supported Image Formats

### RAW Format

- **Extensions**: `.raw`, `.nef`, `.cr2`, `.arw`, `.dng`
- **Use Case**: High-fidelity sensor data from cameras
- **Bit Depths**: 8, 12, 14, 16-bit

```python
from image_acquisition import ImageAcquisition, ImageFormat

acq = ImageAcquisition(format=ImageFormat.RAW)
image_data, metadata = acq.acquire(source="image.raw", width=4096, height=3072)
```

### DICOM Format (Medical Imaging)

- **Extensions**: `.dcm`, `.dicom`
- **Use Case**: Medical imaging (X-ray, CT, MRI)
- **Requirements**: `pydicom` package
- **Compliance**: HIPAA-ready with PHI handling

```python
acq = ImageAcquisition(format=ImageFormat.DICOM)
image_data, metadata = acq.acquire(source="scan.dcm")
```

### FITS Format (Astronomical)

- **Extensions**: `.fits`, `.fit`, `.fts`
- **Use Case**: Astronomical observations
- **Requirements**: `astropy` package
- **Features**: Multi-extension, WCS coordinates

```python
acq = ImageAcquisition(format=ImageFormat.FITS)
image_data, metadata = acq.acquire(source="observation.fits")
```

### Standard Formats

| Format | Extensions | Use Case |
|--------|------------|----------|
| **TIFF** | `.tiff`, `.tif` | Lossless high-quality images |
| **PNG** | `.png` | Lossless with transparency |
| **JPEG** | `.jpg`, `.jpeg` | Compressed photographs |

---

## Acquisition Modes

### LOCAL_FILE Mode

Acquire images from the local filesystem.

```python
from image_acquisition import ImageAcquisition, ImageFormat, AcquisitionMode

acq = ImageAcquisition(
    format=ImageFormat.PNG,
    mode=AcquisitionMode.LOCAL_FILE,
    security_level=2
)

image_data, metadata = acq.acquire(source="/path/to/image.png")
```

### REMOTE_HTTP Mode

Acquire images from HTTP/HTTPS URLs.

```python
acq = ImageAcquisition(
    format=ImageFormat.JPG,
    mode=AcquisitionMode.REMOTE_HTTP,
    security_level=3  # Requires HTTPS
)

image_data, metadata = acq.acquire(
    source="https://secure-server.com/images/scan.jpg",
    secure=True
)
```

### SIMULATION Mode

Generate simulated images for testing and development.

```python
acq = ImageAcquisition(
    format=ImageFormat.RAW,
    mode=AcquisitionMode.SIMULATION
)

image_data, metadata = acq.acquire(
    source="simulated",
    width=512,
    height=512,
    pattern="negative_space",  # 'random', 'gradient', 'negative_space'
    negative_space_regions=5
)
```

### CAMERA Mode

Acquire images from connected camera devices (placeholder for hardware integration).

```python
acq = ImageAcquisition(
    format=ImageFormat.RAW,
    mode=AcquisitionMode.CAMERA
)

# Note: Requires hardware-specific SDK implementation
image_data, metadata = acq.acquire(source="device://camera0")
```

### REMOTE_SFTP Mode

Acquire images from SFTP servers (placeholder for secure remote acquisition).

```python
acq = ImageAcquisition(
    format=ImageFormat.TIFF,
    mode=AcquisitionMode.REMOTE_SFTP
)

# Note: Requires paramiko or pysftp implementation
image_data, metadata = acq.acquire(source="sftp://server.com/images/scan.tiff")
```

---

## Data Ingestion Workflows

### Batch Processing Workflow

```python
import os
from image_acquisition import ImageAcquisition, ImageFormat, AcquisitionMode

def batch_acquire(input_dir, output_dir):
    """Process all images in a directory."""
    acq = ImageAcquisition(
        format=ImageFormat.RAW,
        mode=AcquisitionMode.LOCAL_FILE
    )
    
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        image_data, metadata = acq.acquire(source=filepath)
        
        # Save processed data
        output_path = os.path.join(output_dir, f"{filename}_processed.raw")
        acq.save_image(image_data, output_path)
        acq.save_metadata(os.path.join(output_dir, f"{filename}_metadata.json"))
```

### Streaming Workflow

The system supports streaming ingestion from Kafka and other streaming platforms:

```yaml
# ingestion/data_ingestion.yaml
ingestion:
  mode: "streaming"
  batch_size: 10000
  parallel_streams: 32
  buffer_size: "64Gi"
  compression: "lz4"

sources:
  - type: "kafka"
    brokers: ["kafka-1:9092", "kafka-2:9092"]
    topics: ["image-data-raw"]
    consumer_group: "image-processor"
```

### CLI Workflow

```bash
# Acquire and process a single image
python cli.py acquire --source path/to/image.raw --output acquired.raw

# Run complete secure workflow
python cli.py workflow --mode threshold --signatures 5 --threshold 3

# Simulate acquisition for testing
python image_acquisition.py --mode SIMULATION --pattern negative_space --regions 5
```

---

## Configuration Options

### Security Levels

| Level | Description | Requirements |
|-------|-------------|--------------|
| **0** | No security checks | None |
| **1** | Basic validation | File exists, URL format valid |
| **2** | Standard security | Level 1 + HTTPS required, read permissions |
| **3** | Maximum security | Level 2 + Allowed directories/domains only |

### Acquisition Profiles

Store and load configuration profiles:

```python
# acquisition_profiles.py usage
from acquisition_profiles import AcquisitionProfiles

profiles = AcquisitionProfiles()

# Load a preset profile
high_speed_config = profiles.load("high_speed_capture")

# Apply to acquisition
acq = ImageAcquisition(**high_speed_config)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `IMAGE_SECURITY_LEVEL` | Default security level | `2` |
| `ALLOWED_IMAGE_DIRS` | Comma-separated allowed directories | Current dir |
| `ALLOWED_DOMAINS` | Comma-separated allowed domains | None |
| `MAX_IMAGE_SIZE` | Maximum image size in bytes | `1073741824` (1GB) |

---

## Integration with External Systems

### Medical Imaging Systems (DICOM)

```python
# Integration with PACS/DICOM servers
from image_acquisition import ImageAcquisition, ImageFormat

# Configure for HIPAA compliance
acq = ImageAcquisition(
    format=ImageFormat.DICOM,
    security_level=3,
    verify_integrity=True
)

# Acquire from PACS
image_data, metadata = acq.acquire(
    source="path/to/dicom/series",
    hipaa_audit=True
)
```

### Telescope Integration

```python
# Integration with astronomical instruments
acq = ImageAcquisition(
    format=ImageFormat.FITS,
    mode=AcquisitionMode.CAMERA
)

# Acquire from telescope
image_data, metadata = acq.acquire(
    source="telescope://primary",
    exposure_time=300,  # seconds
    filter="H-alpha"
)
```

### Cloud Storage Integration

```yaml
# S3 integration configuration
sources:
  - type: "s3"
    bucket: "negative-space-data"
    prefix: "raw/"
    max_concurrent_requests: 100
    credentials:
      use_iam_role: true
```

---

## Security and Compliance

### Integrity Verification

All acquired images are verified using SHA-256 hashing:

```python
# Automatic integrity check
acq = ImageAcquisition(verify_integrity=True)
image_data, metadata = acq.acquire(source="image.raw")

# Manual verification
import hashlib
calculated_hash = hashlib.sha256(image_data).hexdigest()
assert calculated_hash == metadata["sha256_hash"]
```

### Source Authentication

```python
# Level 3 authentication example
acq = ImageAcquisition(security_level=3)

# Only allows files from approved directories
# Only allows URLs from approved domains
# Enforces HTTPS for all remote sources
```

### Audit Logging

All acquisition errors are logged to `acquisition_errors.log`:

```json
{
  "timestamp": "2025-11-28T12:00:00.000Z",
  "acquisition_id": "1a2b3c4d5e6f",
  "source": "/path/to/image.raw",
  "error_type": "SourceAuthenticationError",
  "error_message": "File location not authorized",
  "mode": "LOCAL_FILE",
  "format": "RAW"
}
```

---

## API Reference

### ImageAcquisition Class

```python
class ImageAcquisition:
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
            format: Image format to acquire (RAW, DICOM, FITS, etc.)
            mode: Acquisition mode (LOCAL_FILE, REMOTE_HTTP, etc.)
            security_level: Security level 0-3 (higher = more secure)
            verify_integrity: Whether to verify image integrity
        """

    def acquire(
        self,
        source: str,
        secure: bool = True,
        **kwargs
    ) -> Tuple[bytes, Dict[str, any]]:
        """
        Acquire an image from the specified source.
        
        Args:
            source: Source identifier (filepath, URL, device ID)
            secure: Whether to use secure protocols
            **kwargs: Additional parameters (width, height, pattern, etc.)
        
        Returns:
            Tuple of (image_data, metadata)
        """

    def save_image(self, image_data: bytes, filepath: str) -> str:
        """Save image data to a file."""

    def save_metadata(self, filepath: str) -> str:
        """Save metadata to a JSON file."""
```

### Metadata Structure

```python
{
    "acquisition_id": "1a2b3c4d5e6f7890",
    "timestamp": "2025-11-28T12:00:00.000000",
    "source": "/path/to/image.raw",
    "mode": "LOCAL_FILE",
    "format": "RAW",
    "size_bytes": 16777216,
    "elapsed_time_seconds": 0.123,
    "sha256_hash": "abc123...",
    "width": 4096,  # Optional
    "height": 3072,  # Optional
    "bit_depth": 16  # Optional, format-specific
}
```

---

## Troubleshooting

### Common Issues

#### 1. "Image file not found" Error

```bash
# Verify file exists
ls -la /path/to/image.raw

# Check file permissions
chmod 644 /path/to/image.raw
```

#### 2. "Source authentication failed" Error

```python
# Lower security level for development
acq = ImageAcquisition(security_level=1)
```

#### 3. "DICOM format requires pydicom" Error

```bash
pip install pydicom
```

#### 4. "FITS format requires astropy" Error

```bash
pip install astropy
```

#### 5. Remote acquisition timeout

```python
# Increase timeout for slow connections
image_data, metadata = acq.acquire(
    source="https://slow-server.com/large-image.fits",
    timeout=120  # seconds
)
```

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

acq = ImageAcquisition(format=ImageFormat.RAW)
```

---

## Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture overview
- [REQUIREMENTS.md](./REQUIREMENTS.md) - Dependency requirements
- [GETTING_STARTED.md](./GETTING_STARTED.md) - Quickstart guide
- [cli.py](./cli.py) - Command-line interface
- [image_acquisition_integration_report.md](./image_acquisition_integration_report.md) - Integration report

---

Last Updated: November 2025
