"""
Negative Space Detection - Data Models

This module defines Pydantic models for data validation and serialization
throughout the negative space detection system.

Author: Stephen Bilodeau
Project: Negative Space Imaging Project
License: Proprietary
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import json


class ContourData(BaseModel):
    """Model representing a single contour region."""

    id: int = Field(..., description="Unique contour identifier")
    area: float = Field(..., ge=0, description="Area of the contour in pixels")
    perimeter: float = Field(..., ge=0, description="Perimeter of the contour")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    bounding_box: Dict[str, float] = Field(..., description="Bounding box coordinates")
    centroid: Dict[str, float] = Field(..., description="Centroid coordinates")
    aspect_ratio: float = Field(..., ge=0, description="Width/height ratio")
    solidity: float = Field(..., ge=0, le=1, description="Solidity measure")
    circularity: float = Field(..., ge=0, le=1, description="Circularity measure")

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "id": 1,
                "area": 1500.5,
                "perimeter": 150.3,
                "confidence": 0.95,
                "bounding_box": {"x": 10, "y": 20, "width": 100, "height": 80},
                "centroid": {"x": 60, "y": 60},
                "aspect_ratio": 1.25,
                "solidity": 0.85,
                "circularity": 0.78,
            }
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    @validator("confidence", "solidity", "circularity")
    def validate_normalized_values(cls, v: float) -> float:
        """Validate that normalized values are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Value must be between 0 and 1")
        return v


class AnalysisResult(BaseModel):
    """Model representing complete analysis results."""

    image_path: str = Field(..., description="Path to analyzed image")
    image_width: int = Field(..., gt=0, description="Image width in pixels")
    image_height: int = Field(..., gt=0, description="Image height in pixels")
    contours: List[ContourData] = Field(default_factory=list, description="Detected contours")
    total_negative_space: float = Field(..., ge=0, description="Total negative space area")
    negative_space_percentage: float = Field(
        ..., ge=0, le=100, description="Percentage of image that is negative space"
    )
    average_confidence: float = Field(..., ge=0, le=1, description="Average confidence score")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}
        schema_extra = {
            "example": {
                "image_path": "/path/to/image.jpg",
                "image_width": 1024,
                "image_height": 768,
                "contours": [],
                "total_negative_space": 250000.0,
                "negative_space_percentage": 32.5,
                "average_confidence": 0.87,
                "processing_time_ms": 145.3,
                "timestamp": "2025-11-08T10:30:00",
                "metadata": {},
            }
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization."""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @validator("negative_space_percentage")
    def validate_percentage(cls, v: float) -> float:
        """Validate that percentage is 0-100."""
        if not 0 <= v <= 100:
            raise ValueError("Percentage must be between 0 and 100")
        return v


class ConfigModel(BaseModel):
    """Model for analyzer configuration."""

    edge_detection_method: str = Field(
        default="canny", description="Edge detection method: 'canny' or 'sobel'"
    )
    canny_threshold1: int = Field(default=100, ge=0, le=255, description="Canny lower threshold")
    canny_threshold2: int = Field(default=200, ge=0, le=255, description="Canny upper threshold")
    sobel_kernel_size: int = Field(
        default=3, description="Sobel kernel size (must be odd)", ge=1
    )
    min_contour_area: int = Field(default=100, ge=0, description="Minimum contour area in pixels")
    max_contour_area: Optional[int] = Field(
        default=None, ge=0, description="Maximum contour area in pixels"
    )
    enable_morphology: bool = Field(default=True, description="Enable morphological operations")
    morphology_kernel_size: int = Field(
        default=5, description="Morphological kernel size", ge=1
    )
    confidence_threshold: float = Field(
        default=0.5, ge=0, le=1, description="Minimum confidence score threshold"
    )
    max_image_size: int = Field(
        default=1024, ge=256, description="Maximum image dimension for resizing"
    )
    enable_contrast_enhancement: bool = Field(
        default=True, description="Enable contrast enhancement"
    )

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "edge_detection_method": "canny",
                "canny_threshold1": 100,
                "canny_threshold2": 200,
                "sobel_kernel_size": 3,
                "min_contour_area": 100,
                "max_contour_area": None,
                "enable_morphology": True,
                "morphology_kernel_size": 5,
                "confidence_threshold": 0.5,
                "max_image_size": 1024,
                "enable_contrast_enhancement": True,
            }
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    @validator("edge_detection_method")
    def validate_edge_method(cls, v: str) -> str:
        """Validate edge detection method."""
        if v not in ["canny", "sobel"]:
            raise ValueError("edge_detection_method must be 'canny' or 'sobel'")
        return v

    @validator("sobel_kernel_size", "morphology_kernel_size")
    def validate_odd_kernel_size(cls, v: int) -> int:
        """Validate that kernel sizes are odd."""
        if v % 2 == 0:
            raise ValueError("Kernel size must be odd")
        return v


if __name__ == "__main__":
    print("✓ ContourData: Contour model defined")
    print("✓ AnalysisResult: Analysis result model defined")
    print("✓ ConfigModel: Configuration model defined")

    # Test instantiation
    config = ConfigModel()
    print(f"\n✓ Default config created successfully:")
    print(f"  - Edge method: {config.edge_detection_method}")
    print(f"  - Min contour area: {config.min_contour_area}")
    print(f"  - Confidence threshold: {config.confidence_threshold}")

    print("\n✓ All Pydantic models initialized successfully!")
