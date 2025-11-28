#!/usr/bin/env python
"""
Acquisition Profiles Management Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides profile management for different imaging acquisition scenarios
including medical imaging, astronomical imaging, industrial inspection, and research.

Each profile encapsulates optimal settings for:
- Image format and resolution
- Acquisition timing and exposure
- Preprocessing parameters
- Quality thresholds
- Storage and metadata handling
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="AcquisitionProfile")


class ProfileType(Enum):
    """Types of acquisition profiles."""
    MEDICAL = "medical"
    ASTRONOMICAL = "astronomical"
    INDUSTRIAL = "industrial"
    RESEARCH = "research"
    CUSTOM = "custom"


class ImageQuality(Enum):
    """Image quality levels."""
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass
class ExposureSettings:
    """Exposure configuration for image acquisition."""
    exposure_time_ms: float = 100.0
    gain: float = 1.0
    iso: Optional[int] = None
    aperture: Optional[float] = None
    auto_exposure: bool = False
    bracketing_enabled: bool = False
    bracketing_steps: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "exposure_time_ms": self.exposure_time_ms,
            "gain": self.gain,
            "iso": self.iso,
            "aperture": self.aperture,
            "auto_exposure": self.auto_exposure,
            "bracketing_enabled": self.bracketing_enabled,
            "bracketing_steps": self.bracketing_steps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExposureSettings":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ResolutionSettings:
    """Resolution and format settings."""
    width: int = 1920
    height: int = 1080
    bit_depth: int = 16
    channels: int = 1
    color_space: str = "grayscale"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
            "color_space": self.color_space,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResolutionSettings":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PreprocessingSettings:
    """Preprocessing pipeline settings."""
    noise_reduction: bool = True
    noise_reduction_strength: float = 0.5
    dark_frame_subtraction: bool = False
    flat_field_correction: bool = False
    hot_pixel_removal: bool = True
    cosmic_ray_removal: bool = False
    debayering: bool = False
    debayer_algorithm: str = "bilinear"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "noise_reduction": self.noise_reduction,
            "noise_reduction_strength": self.noise_reduction_strength,
            "dark_frame_subtraction": self.dark_frame_subtraction,
            "flat_field_correction": self.flat_field_correction,
            "hot_pixel_removal": self.hot_pixel_removal,
            "cosmic_ray_removal": self.cosmic_ray_removal,
            "debayering": self.debayering,
            "debayer_algorithm": self.debayer_algorithm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingSettings":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class QualitySettings:
    """Quality control settings."""
    min_snr: float = 10.0
    max_noise_level: float = 0.05
    sharpness_threshold: float = 0.7
    contrast_threshold: float = 0.3
    auto_reject_poor_quality: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_snr": self.min_snr,
            "max_noise_level": self.max_noise_level,
            "sharpness_threshold": self.sharpness_threshold,
            "contrast_threshold": self.contrast_threshold,
            "auto_reject_poor_quality": self.auto_reject_poor_quality,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualitySettings":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class StorageSettings:
    """Storage and output settings."""
    output_format: str = "TIFF"
    compression: Optional[str] = None
    compression_level: int = 6
    output_directory: str = "./output"
    filename_pattern: str = "{timestamp}_{profile}_{sequence}"
    metadata_format: str = "json"
    create_thumbnails: bool = True
    thumbnail_size: int = 256

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_format": self.output_format,
            "compression": self.compression,
            "compression_level": self.compression_level,
            "output_directory": self.output_directory,
            "filename_pattern": self.filename_pattern,
            "metadata_format": self.metadata_format,
            "create_thumbnails": self.create_thumbnails,
            "thumbnail_size": self.thumbnail_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageSettings":
        """Create from dictionary."""
        return cls(**data)


class AcquisitionProfile(ABC):
    """
    Abstract base class for acquisition profiles.

    Provides the foundation for domain-specific acquisition configurations.
    """

    def __init__(
        self,
        name: str,
        profile_type: ProfileType,
        description: str = "",
    ):
        """
        Initialize acquisition profile.

        Args:
            name: Profile name
            profile_type: Type of profile
            description: Profile description
        """
        self.name = name
        self.profile_type = profile_type
        self.description = description
        self.created_at = datetime.utcnow()
        self.modified_at = datetime.utcnow()
        self.exposure = ExposureSettings()
        self.resolution = ResolutionSettings()
        self.preprocessing = PreprocessingSettings()
        self.quality = QualitySettings()
        self.storage = StorageSettings()
        self.custom_settings: Dict[str, Any] = {}

    @abstractmethod
    def validate(self) -> bool:
        """Validate profile settings."""
        pass

    @abstractmethod
    def get_optimal_settings(self) -> Dict[str, Any]:
        """Get optimal settings for this profile type."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "name": self.name,
            "profile_type": self.profile_type.value,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "exposure": self.exposure.to_dict(),
            "resolution": self.resolution.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "quality": self.quality.to_dict(),
            "storage": self.storage.to_dict(),
            "custom_settings": self.custom_settings,
        }

    def save(self, path: Path) -> None:
        """Save profile to file."""
        self.modified_at = datetime.utcnow()
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Profile saved to {path}")


class MedicalImagingProfile(AcquisitionProfile):
    """
    Profile for medical imaging applications.

    Optimized for DICOM output, HIPAA compliance, and diagnostic quality.
    """

    def __init__(
        self,
        name: str = "Medical Imaging Default",
        modality: str = "XR",
        body_part: str = "general",
    ):
        """
        Initialize medical imaging profile.

        Args:
            name: Profile name
            modality: Imaging modality (XR, CT, MRI, etc.)
            body_part: Target body part
        """
        super().__init__(
            name=name,
            profile_type=ProfileType.MEDICAL,
            description=f"Medical imaging profile for {modality} - {body_part}",
        )
        self.modality = modality
        self.body_part = body_part

        # Medical-specific defaults
        self.resolution = ResolutionSettings(
            width=4096, height=4096, bit_depth=16, channels=1
        )
        self.storage = StorageSettings(
            output_format="DICOM",
            compression="lossless",
            metadata_format="dicom",
        )
        self.quality = QualitySettings(
            min_snr=20.0,
            auto_reject_poor_quality=True,
        )
        self.preprocessing = PreprocessingSettings(
            noise_reduction=True,
            noise_reduction_strength=0.3,
        )

        # HIPAA compliance settings
        self.custom_settings = {
            "hipaa_compliant": True,
            "anonymize_metadata": False,
            "audit_logging": True,
            "encryption_required": True,
            "modality": modality,
            "body_part": body_part,
        }

    def validate(self) -> bool:
        """Validate medical profile settings."""
        if self.resolution.bit_depth < 12:
            logger.warning("Medical imaging typically requires 12+ bit depth")
            return False
        if not self.custom_settings.get("hipaa_compliant"):
            logger.warning("HIPAA compliance is recommended for medical imaging")
        return True

    def get_optimal_settings(self) -> Dict[str, Any]:
        """Get optimal settings for medical imaging."""
        return {
            "resolution": {"width": 4096, "height": 4096, "bit_depth": 16},
            "exposure": {"auto_exposure": True},
            "quality": {"min_snr": 20.0},
            "storage": {"output_format": "DICOM", "compression": "lossless"},
        }


class AstronomicalImagingProfile(AcquisitionProfile):
    """
    Profile for astronomical imaging applications.

    Optimized for FITS output, long exposures, and celestial object detection.
    """

    def __init__(
        self,
        name: str = "Astronomical Imaging Default",
        target_type: str = "deep_sky",
        telescope_focal_length: float = 1000.0,
    ):
        """
        Initialize astronomical imaging profile.

        Args:
            name: Profile name
            target_type: Type of target (deep_sky, planetary, solar, etc.)
            telescope_focal_length: Focal length in mm
        """
        super().__init__(
            name=name,
            profile_type=ProfileType.ASTRONOMICAL,
            description=f"Astronomical imaging profile for {target_type}",
        )
        self.target_type = target_type
        self.telescope_focal_length = telescope_focal_length

        # Astronomical-specific defaults
        self.resolution = ResolutionSettings(
            width=4656, height=3520, bit_depth=16, channels=1
        )
        self.exposure = ExposureSettings(
            exposure_time_ms=30000.0,  # 30 seconds default
            gain=1.0,
            bracketing_enabled=False,
        )
        self.storage = StorageSettings(
            output_format="FITS",
            compression=None,
            metadata_format="fits_header",
        )
        self.preprocessing = PreprocessingSettings(
            dark_frame_subtraction=True,
            flat_field_correction=True,
            hot_pixel_removal=True,
            cosmic_ray_removal=True,
        )

        self.custom_settings = {
            "target_type": target_type,
            "telescope_focal_length": telescope_focal_length,
            "tracking_enabled": True,
            "dithering_enabled": True,
            "plate_solving": True,
            "wcs_coordinates": True,
        }

    def validate(self) -> bool:
        """Validate astronomical profile settings."""
        if self.exposure.exposure_time_ms < 100 and self.target_type == "deep_sky":
            logger.warning("Deep sky imaging typically requires longer exposures")
            return False
        return True

    def get_optimal_settings(self) -> Dict[str, Any]:
        """Get optimal settings for astronomical imaging."""
        if self.target_type == "deep_sky":
            return {
                "exposure": {"exposure_time_ms": 120000.0, "gain": 1.0},
                "preprocessing": {
                    "dark_frame_subtraction": True,
                    "flat_field_correction": True,
                    "cosmic_ray_removal": True,
                },
            }
        elif self.target_type == "planetary":
            return {
                "exposure": {"exposure_time_ms": 10.0, "gain": 2.0},
                "resolution": {"bit_depth": 12},
            }
        return {}


class IndustrialInspectionProfile(AcquisitionProfile):
    """
    Profile for industrial inspection applications.

    Optimized for defect detection, high throughput, and quality control.
    """

    def __init__(
        self,
        name: str = "Industrial Inspection Default",
        inspection_type: str = "surface",
        production_speed: float = 1.0,
    ):
        """
        Initialize industrial inspection profile.

        Args:
            name: Profile name
            inspection_type: Type of inspection (surface, dimensional, etc.)
            production_speed: Production line speed factor (1.0 = normal)
        """
        super().__init__(
            name=name,
            profile_type=ProfileType.INDUSTRIAL,
            description=f"Industrial inspection profile for {inspection_type}",
        )
        self.inspection_type = inspection_type
        self.production_speed = production_speed

        # Industrial-specific defaults
        self.resolution = ResolutionSettings(
            width=2048, height=2048, bit_depth=8, channels=3, color_space="rgb"
        )
        self.exposure = ExposureSettings(
            exposure_time_ms=5.0,  # Fast exposure for production line
            auto_exposure=False,
        )
        self.storage = StorageSettings(
            output_format="PNG",
            compression="lossless",
            create_thumbnails=False,
        )
        self.quality = QualitySettings(
            sharpness_threshold=0.8,
            auto_reject_poor_quality=True,
        )

        self.custom_settings = {
            "inspection_type": inspection_type,
            "production_speed": production_speed,
            "real_time_analysis": True,
            "defect_classification": True,
            "pass_fail_threshold": 0.95,
        }

    def validate(self) -> bool:
        """Validate industrial profile settings."""
        if self.exposure.exposure_time_ms > 50 and self.production_speed > 0.5:
            logger.warning("Exposure may be too long for production speed")
            return False
        return True

    def get_optimal_settings(self) -> Dict[str, Any]:
        """Get optimal settings for industrial inspection."""
        return {
            "exposure": {"exposure_time_ms": 5.0 / self.production_speed},
            "quality": {"sharpness_threshold": 0.8},
        }


class ResearchImagingProfile(AcquisitionProfile):
    """
    Profile for research and scientific imaging applications.

    Flexible profile with maximum customization options.
    """

    def __init__(
        self,
        name: str = "Research Imaging Default",
        research_domain: str = "general",
    ):
        """
        Initialize research imaging profile.

        Args:
            name: Profile name
            research_domain: Research domain/field
        """
        super().__init__(
            name=name,
            profile_type=ProfileType.RESEARCH,
            description=f"Research imaging profile for {research_domain}",
        )
        self.research_domain = research_domain

        # Research-specific defaults (maximum quality/flexibility)
        self.resolution = ResolutionSettings(
            width=4096, height=4096, bit_depth=16, channels=1
        )
        self.storage = StorageSettings(
            output_format="HDF5",
            compression="gzip",
            metadata_format="json",
        )

        self.custom_settings = {
            "research_domain": research_domain,
            "raw_data_preservation": True,
            "metadata_extensive": True,
            "reproducibility_tracking": True,
        }

    def validate(self) -> bool:
        """Validate research profile settings."""
        return True

    def get_optimal_settings(self) -> Dict[str, Any]:
        """Get optimal settings for research imaging."""
        return {
            "storage": {"raw_data_preservation": True},
            "quality": {"auto_reject_poor_quality": False},
        }


class ProfileManager:
    """
    Manages acquisition profiles.

    Provides profile loading, saving, and lookup functionality.
    """

    def __init__(self, profiles_directory: Optional[Path] = None):
        """
        Initialize profile manager.

        Args:
            profiles_directory: Directory for storing profiles
        """
        self.profiles_directory = profiles_directory or Path("./profiles")
        self.profiles: Dict[str, AcquisitionProfile] = {}
        self._profile_types: Dict[ProfileType, Type[AcquisitionProfile]] = {
            ProfileType.MEDICAL: MedicalImagingProfile,
            ProfileType.ASTRONOMICAL: AstronomicalImagingProfile,
            ProfileType.INDUSTRIAL: IndustrialInspectionProfile,
            ProfileType.RESEARCH: ResearchImagingProfile,
        }

        # Register default profiles
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default profiles."""
        self.register(MedicalImagingProfile())
        self.register(AstronomicalImagingProfile())
        self.register(IndustrialInspectionProfile())
        self.register(ResearchImagingProfile())

    def register(self, profile: AcquisitionProfile) -> None:
        """Register a profile."""
        self.profiles[profile.name] = profile
        logger.info(f"Registered profile: {profile.name}")

    def get(self, name: str) -> Optional[AcquisitionProfile]:
        """Get a profile by name."""
        return self.profiles.get(name)

    def list_profiles(self) -> List[str]:
        """List all registered profile names."""
        return list(self.profiles.keys())

    def create_profile(
        self,
        profile_type: ProfileType,
        name: str,
        **kwargs: Any,
    ) -> AcquisitionProfile:
        """
        Create a new profile.

        Args:
            profile_type: Type of profile to create
            name: Profile name
            **kwargs: Additional profile arguments

        Returns:
            Created profile
        """
        profile_class = self._profile_types.get(profile_type)
        if profile_class is None:
            raise ValueError(f"Unknown profile type: {profile_type}")

        profile = profile_class(name=name, **kwargs)
        self.register(profile)
        return profile

    def save_all(self) -> None:
        """Save all profiles to the profiles directory."""
        self.profiles_directory.mkdir(parents=True, exist_ok=True)
        for name, profile in self.profiles.items():
            safe_name = name.replace(" ", "_").lower()
            path = self.profiles_directory / f"{safe_name}.json"
            profile.save(path)

    def load_from_directory(self) -> None:
        """Load profiles from the profiles directory."""
        if not self.profiles_directory.exists():
            logger.warning(f"Profiles directory not found: {self.profiles_directory}")
            return

        for path in self.profiles_directory.glob("*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                profile_type = ProfileType(data["profile_type"])
                profile_class = self._profile_types.get(profile_type)
                if profile_class:
                    profile = profile_class(name=data["name"])
                    # Apply loaded settings
                    profile.exposure = ExposureSettings.from_dict(data["exposure"])
                    profile.resolution = ResolutionSettings.from_dict(data["resolution"])
                    profile.preprocessing = PreprocessingSettings.from_dict(
                        data["preprocessing"]
                    )
                    profile.quality = QualitySettings.from_dict(data["quality"])
                    profile.storage = StorageSettings.from_dict(data["storage"])
                    profile.custom_settings = data.get("custom_settings", {})
                    self.register(profile)
            except Exception as e:
                logger.error(f"Failed to load profile from {path}: {e}")


def get_default_manager() -> ProfileManager:
    """Get the default profile manager instance."""
    return ProfileManager()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo: Create and use profiles
    manager = get_default_manager()

    print("Available profiles:")
    for name in manager.list_profiles():
        profile = manager.get(name)
        if profile:
            print(f"  - {name} ({profile.profile_type.value})")
            print(f"    Resolution: {profile.resolution.width}x{profile.resolution.height}")
            print(f"    Output: {profile.storage.output_format}")
