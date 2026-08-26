"""
Negative Space Imaging Project - Core Module

A comprehensive Python module for detecting and analyzing negative space
in images using advanced computer vision techniques.

This package provides:
- NegativeSpaceAnalyzer: Main analysis class
- Configurable edge detection (Canny, Sobel)
- Contour analysis and filtering
- Confidence scoring system
- JSON report generation

Author: Stephen Bilodeau
Project: Negative Space Imaging Project
License: Proprietary
Version: 1.0.0

Example:
    >>> from negative_space import NegativeSpaceAnalyzer
    >>> analyzer = NegativeSpaceAnalyzer()
    >>> result = analyzer.analyze('image.jpg')
    >>> print(f"Negative space: {result.negative_space_percentage:.1f}%")
    >>> print(result.to_json())
"""

__version__ = "1.0.0"
__author__ = "Stephen Bilodeau"
__license__ = "Proprietary"

from .core.analyzer import NegativeSpaceAnalyzer
from .core.models import AnalysisResult, ContourData, ConfigModel
from .exceptions import (
    NegativeSpaceError,
    ImageLoadError,
    AnalysisError,
    ValidationError,
)

__all__ = [
    "NegativeSpaceAnalyzer",
    "AnalysisResult",
    "ContourData",
    "ConfigModel",
    "NegativeSpaceError",
    "ImageLoadError",
    "AnalysisError",
    "ValidationError",
]

if __name__ == "__main__":
    print(f"✓ Negative Space Imaging Project v{__version__}")
    print(f"✓ Author: {__author__}")
    print(f"✓ License: {__license__}")
    print("\n✓ Available Classes:")
    print("  - NegativeSpaceAnalyzer: Main analysis engine")
    print("  - AnalysisResult: Result data model")
    print("  - ContourData: Contour data model")
    print("  - ConfigModel: Configuration model")
    print("\n✓ Available Exceptions:")
    print("  - NegativeSpaceError: Base exception")
    print("  - ImageLoadError: Image loading error")
    print("  - AnalysisError: Analysis error")
    print("  - ValidationError: Validation error")
    print("\n✓ Module initialized successfully!")
