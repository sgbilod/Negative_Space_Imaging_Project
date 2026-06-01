"""
Negative Space Detection - Custom Exception Classes

This module defines custom exception classes for the negative space
detection system, providing specific error types for different failure scenarios.

Author: Stephen Bilodeau
Project: Negative Space Imaging Project
License: Proprietary
"""

from typing import Optional


class NegativeSpaceError(Exception):
    """Base exception class for all negative space detection errors."""

    def __init__(self, message: str, code: Optional[str] = None) -> None:
        """
        Initialize NegativeSpaceError.

        Args:
            message: Descriptive error message
            code: Optional error code for categorization

        Example:
            >>> raise NegativeSpaceError("Processing failed", code="E001")
        """
        self.message = message
        self.code = code
        super().__init__(f"[{self.code}] {message}" if code else message)


class ImageLoadError(NegativeSpaceError):
    """Raised when image loading or reading fails."""

    def __init__(self, message: str, file_path: Optional[str] = None) -> None:
        """
        Initialize ImageLoadError.

        Args:
            message: Descriptive error message
            file_path: Path to the file that failed to load

        Example:
            >>> raise ImageLoadError("File not found", "test.jpg")
        """
        self.file_path = file_path
        super().__init__(message, code="IMG001")


class AnalysisError(NegativeSpaceError):
    """Raised when image analysis fails."""

    def __init__(self, message: str, analysis_step: Optional[str] = None) -> None:
        """
        Initialize AnalysisError.

        Args:
            message: Descriptive error message
            analysis_step: Name of the analysis step that failed

        Example:
            >>> raise AnalysisError("Edge detection failed", "canny_edge")
        """
        self.analysis_step = analysis_step
        super().__init__(message, code="ANL001")


class ValidationError(NegativeSpaceError):
    """Raised when data validation fails."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        """
        Initialize ValidationError.

        Args:
            message: Descriptive error message
            field: Name of the field that failed validation

        Example:
            >>> raise ValidationError("Invalid confidence score", "confidence")
        """
        self.field = field
        super().__init__(message, code="VAL001")


if __name__ == "__main__":
    print("✓ NegativeSpaceError: Base exception defined")
    print("✓ ImageLoadError: Image loading exception defined")
    print("✓ AnalysisError: Analysis error exception defined")
    print("✓ ValidationError: Validation error exception defined")
    print("\n✓ All custom exceptions initialized successfully!")
