"""
Metadata Extractor Module for Negative Space Imaging Project

This module extracts and manages metadata from images that is specifically
relevant for spatial referencing and negative space analysis.
"""

import os
import json
import numpy as np
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum
import re
import hashlib
import uuid
from pathlib import Path
import exifread
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SpatialMetadata:
    """Container for spatial reference metadata"""
    # Camera information
    camera_id: str = ""
    camera_model: str = ""
    camera_serial: str = ""
    focal_length: Optional[float] = None
    f_number: Optional[float] = None
    
    # Image capture details
    timestamp: str = ""
    exposure_time: Optional[float] = None
    iso: Optional[int] = None
    
    # Spatial information
    gps_coordinates: Optional[Tuple[float, float, float]] = None
    altitude: Optional[float] = None
    compass_direction: Optional[float] = None
    roll: Optional[float] = None
    pitch: Optional[float] = None
    
    # Camera calibration data
    camera_matrix: Optional[np.ndarray] = None
    distortion_coefficients: Optional[np.ndarray] = None
    
    # Negative space specific metadata
    reference_objects: List[Dict[str, Any]] = None
    void_regions: List[Dict[str, Any]] = None
    
    # Processing information
    preprocessing_mode: str = ""
    preprocessing_params: Dict[str, Any] = None
    
    # Unique identifiers
    image_hash: str = ""
    session_id: str = ""
    
    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.reference_objects is None:
            self.reference_objects = []
        if self.void_regions is None:
            self.void_regions = []
        if self.preprocessing_params is None:
            self.preprocessing_params = {}

class MetadataExtractor:
    """
    Extracts and manages metadata from images for spatial referencing.
    
    This class handles extraction of EXIF data, sensor information, and
    other metadata needed for accurate spatial mapping and negative space
    analysis.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize the metadata extractor.
        
        Args:
            session_id: Optional session identifier. If None, a new UUID is generated.
        """
        self.session_id = session_id or str(uuid.uuid4())
        logger.info(f"MetadataExtractor initialized with session ID: {self.session_id}")
        
        # Store processed metadata for the session
        self.metadata_collection = {}
        
        # Mapping of camera models to known sensor sizes (width, height in mm)
        # This can be extended with more camera models
        self.known_sensors = {
            # Common DSLRs
            "NIKON D850": (35.9, 23.9),
            "NIKON Z6": (35.9, 23.9),
            "Canon EOS 5D Mark IV": (36.0, 24.0),
            "SONY ILCE-7M3": (35.6, 23.8),  # Sony A7 III
            
            # Common smartphones
            "iPhone 12 Pro": (6.86, 5.14),
            "iPhone 13 Pro": (6.86, 5.14),
            "Pixel 6": (6.2, 4.65),
            "Galaxy S21": (6.3, 4.7),
        }
    
    def extract_from_file(self, image_path: str) -> SpatialMetadata:
        """
        Extract metadata from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            SpatialMetadata object with extracted information
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        metadata = SpatialMetadata(session_id=self.session_id)
        
        # Generate image hash
        metadata.image_hash = self._generate_file_hash(image_path)
        
        # Extract EXIF data
        self._extract_exif(image_path, metadata)
        
        # Store in collection
        self.metadata_collection[metadata.image_hash] = metadata
        
        return metadata
    
    def extract_from_image(self, image: np.ndarray, 
                          additional_info: Optional[Dict[str, Any]] = None) -> SpatialMetadata:
        """
        Extract metadata from an image array.
        
        Args:
            image: Numpy array containing the image
            additional_info: Optional dictionary with additional metadata
            
        Returns:
            SpatialMetadata object with extracted information
        """
        metadata = SpatialMetadata(session_id=self.session_id)
        
        # Generate image hash from content
        metadata.image_hash = self._generate_image_hash(image)
        
        # Set timestamp
        metadata.timestamp = datetime.datetime.now().isoformat()
        
        # Add any additional provided information
        if additional_info:
            for key, value in additional_info.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
        
        # Store in collection
        self.metadata_collection[metadata.image_hash] = metadata
        
        return metadata
    
    def _extract_exif(self, image_path: str, metadata: SpatialMetadata):
        """
        Extract EXIF metadata from an image file.
        
        Args:
            image_path: Path to the image file
            metadata: SpatialMetadata object to populate
        """
        try:
            # Open image file for EXIF extraction
            with open(image_path, 'rb') as f:
                exif_tags = exifread.process_file(f, details=False)
            
            # Extract basic camera info
            if 'Image Make' in exif_tags:
                metadata.camera_id = str(exif_tags['Image Make'])
            
            if 'Image Model' in exif_tags:
                metadata.camera_model = str(exif_tags['Image Model'])
            
            if 'EXIF BodySerialNumber' in exif_tags:
                metadata.camera_serial = str(exif_tags['EXIF BodySerialNumber'])
            
            # Extract lens and exposure info
            if 'EXIF FocalLength' in exif_tags:
                focal_str = str(exif_tags['EXIF FocalLength'])
                if '/' in focal_str:
                    num, denom = map(float, focal_str.split('/'))
                    metadata.focal_length = num / denom
                else:
                    metadata.focal_length = float(focal_str)
            
            if 'EXIF FNumber' in exif_tags:
                fnumber_str = str(exif_tags['EXIF FNumber'])
                if '/' in fnumber_str:
                    num, denom = map(float, fnumber_str.split('/'))
                    metadata.f_number = num / denom
                else:
                    metadata.f_number = float(fnumber_str)
            
            # Extract timestamp
            if 'EXIF DateTimeOriginal' in exif_tags:
                metadata.timestamp = str(exif_tags['EXIF DateTimeOriginal'])
            
            if 'EXIF ExposureTime' in exif_tags:
                exp_str = str(exif_tags['EXIF ExposureTime'])
                if '/' in exp_str:
                    num, denom = map(float, exp_str.split('/'))
                    metadata.exposure_time = num / denom
                else:
                    metadata.exposure_time = float(exp_str)
            
            if 'EXIF ISOSpeedRatings' in exif_tags:
                metadata.iso = int(str(exif_tags['EXIF ISOSpeedRatings']))
            
            # Extract GPS information
            gps_info = {}
            for key, value in exif_tags.items():
                if key.startswith('GPS'):
                    gps_info[key] = value
            
            if gps_info:
                self._parse_gps_info(gps_info, metadata)
            
            # Use PIL for additional metadata
            with Image.open(image_path) as img:
                exif = img._getexif()
                if exif:
                    labeled_exif = {
                        TAGS.get(tag_id, tag_id): value
                        for tag_id, value in exif.items()
                    }
                    
                    # Extract orientation
                    if 'Orientation' in labeled_exif:
                        # This could be used to determine camera roll
                        orientation = labeled_exif['Orientation']
                        # Process orientation if needed
            
        except Exception as e:
            logger.error(f"Error extracting EXIF data from {image_path}: {str(e)}")
    
    def _parse_gps_info(self, gps_info: Dict, metadata: SpatialMetadata):
        """
        Parse GPS information from EXIF data.
        
        Args:
            gps_info: Dictionary of GPS EXIF tags
            metadata: SpatialMetadata object to populate
        """
        try:
            # Convert GPS coordinates to decimal degrees
            if 'GPS GPSLatitude' in gps_info and 'GPS GPSLatitudeRef' in gps_info:
                lat_data = gps_info['GPS GPSLatitude']
                lat_ref = str(gps_info['GPS GPSLatitudeRef'])
                
                lat = self._convert_to_decimal_degrees(lat_data)
                if lat_ref == 'S':
                    lat = -lat
                
                if 'GPS GPSLongitude' in gps_info and 'GPS GPSLongitudeRef' in gps_info:
                    lon_data = gps_info['GPS GPSLongitude']
                    lon_ref = str(gps_info['GPS GPSLongitudeRef'])
                    
                    lon = self._convert_to_decimal_degrees(lon_data)
                    if lon_ref == 'W':
                        lon = -lon
                    
                    # Set latitude and longitude
                    if 'GPS GPSAltitude' in gps_info:
                        alt_data = gps_info['GPS GPSAltitude']
                        alt = self._convert_to_decimal_value(alt_data)
                        metadata.gps_coordinates = (lat, lon, alt)
                        metadata.altitude = alt
                    else:
                        metadata.gps_coordinates = (lat, lon, 0.0)
            
            # Extract compass direction
            if 'GPS GPSImgDirection' in gps_info:
                dir_data = gps_info['GPS GPSImgDirection']
                metadata.compass_direction = self._convert_to_decimal_value(dir_data)
            
        except Exception as e:
            logger.error(f"Error parsing GPS info: {str(e)}")
    
    def _convert_to_decimal_degrees(self, dms_data) -> float:
        """
        Convert degrees, minutes, seconds format to decimal degrees.
        
        Args:
            dms_data: Degrees, minutes, seconds data from EXIF
            
        Returns:
            Decimal degrees value
        """
        dms_str = str(dms_data)
        
        # Try to handle different formats
        # Format: [degrees, minutes, seconds]
        match = re.match(r'\[(\d+),\s*(\d+),\s*(\d+(?:\.\d+)?)\]', dms_str)
        if match:
            degrees = int(match.group(1))
            minutes = int(match.group(2))
            seconds = float(match.group(3))
            return degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        # Format: degrees/1 minutes/1 seconds/n
        parts = re.findall(r'(\d+)/(\d+)', dms_str)
        if len(parts) == 3:
            degrees = float(parts[0][0]) / float(parts[0][1])
            minutes = float(parts[1][0]) / float(parts[1][1])
            seconds = float(parts[2][0]) / float(parts[2][1])
            return degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        # If parsing fails, try to extract raw numbers
        nums = re.findall(r'\d+(?:\.\d+)?', dms_str)
        if len(nums) >= 3:
            degrees = float(nums[0])
            minutes = float(nums[1])
            seconds = float(nums[2])
            return degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        # If all else fails
        logger.warning(f"Could not parse DMS data: {dms_str}")
        return 0.0
    
    def _convert_to_decimal_value(self, rational_data) -> float:
        """
        Convert rational EXIF value to decimal.
        
        Args:
            rational_data: Rational data from EXIF
            
        Returns:
            Decimal value
        """
        rational_str = str(rational_data)
        
        # Try to handle different formats
        # Format: num/denom
        match = re.match(r'(\d+)/(\d+)', rational_str)
        if match:
            numerator = int(match.group(1))
            denominator = int(match.group(2))
            if denominator == 0:
                return 0.0
            return numerator / denominator
        
        # If it's already a decimal
        try:
            return float(rational_str)
        except ValueError:
            logger.warning(f"Could not parse rational data: {rational_str}")
            return 0.0
    
    def _generate_file_hash(self, file_path: str) -> str:
        """
        Generate a hash for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hash string
        """
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _generate_image_hash(self, image: np.ndarray) -> str:
        """
        Generate a hash for an image array.
        
        Args:
            image: Numpy array containing the image
            
        Returns:
            Hash string
        """
        hasher = hashlib.sha256()
        hasher.update(image.tobytes())
        return hasher.hexdigest()
    
    def add_spatial_reference(self, image_hash: str, 
                             reference_object: Dict[str, Any]) -> bool:
        """
        Add a reference object to the metadata for a specific image.
        
        Args:
            image_hash: Hash of the image
            reference_object: Dictionary with reference object information
            
        Returns:
            True if successful, False otherwise
        """
        if image_hash not in self.metadata_collection:
            logger.error(f"Image hash {image_hash} not found in metadata collection")
            return False
        
        metadata = self.metadata_collection[image_hash]
        metadata.reference_objects.append(reference_object)
        return True
    
    def add_void_region(self, image_hash: str, 
                       void_region: Dict[str, Any]) -> bool:
        """
        Add a void region to the metadata for a specific image.
        
        Args:
            image_hash: Hash of the image
            void_region: Dictionary with void region information
            
        Returns:
            True if successful, False otherwise
        """
        if image_hash not in self.metadata_collection:
            logger.error(f"Image hash {image_hash} not found in metadata collection")
            return False
        
        metadata = self.metadata_collection[image_hash]
        metadata.void_regions.append(void_region)
        return True
    
    def add_camera_calibration(self, image_hash: str, 
                              camera_matrix: np.ndarray,
                              distortion_coefficients: np.ndarray) -> bool:
        """
        Add camera calibration data to the metadata for a specific image.
        
        Args:
            image_hash: Hash of the image
            camera_matrix: Camera matrix from calibration
            distortion_coefficients: Distortion coefficients from calibration
            
        Returns:
            True if successful, False otherwise
        """
        if image_hash not in self.metadata_collection:
            logger.error(f"Image hash {image_hash} not found in metadata collection")
            return False
        
        metadata = self.metadata_collection[image_hash]
        metadata.camera_matrix = camera_matrix
        metadata.distortion_coefficients = distortion_coefficients
        return True
    
    def save_metadata(self, image_hash: str, file_path: str) -> bool:
        """
        Save metadata for a specific image to a JSON file.
        
        Args:
            image_hash: Hash of the image
            file_path: Path to save the metadata
            
        Returns:
            True if successful, False otherwise
        """
        if image_hash not in self.metadata_collection:
            logger.error(f"Image hash {image_hash} not found in metadata collection")
            return False
        
        metadata = self.metadata_collection[image_hash]
        
        # Convert metadata to a serializable dictionary
        metadata_dict = {}
        for key, value in metadata.__dict__.items():
            if isinstance(value, np.ndarray):
                metadata_dict[key] = value.tolist()
            else:
                metadata_dict[key] = value
        
        try:
            with open(file_path, 'w') as f:
                json.dump(metadata_dict, f, indent=4)
            logger.info(f"Metadata saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving metadata to {file_path}: {str(e)}")
            return False
    
    def load_metadata(self, file_path: str) -> Optional[SpatialMetadata]:
        """
        Load metadata from a JSON file.
        
        Args:
            file_path: Path to the metadata file
            
        Returns:
            SpatialMetadata object if successful, None otherwise
        """
        try:
            with open(file_path, 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = SpatialMetadata()
            
            # Populate the metadata object
            for key, value in metadata_dict.items():
                if key in ['camera_matrix', 'distortion_coefficients'] and value is not None:
                    setattr(metadata, key, np.array(value))
                else:
                    setattr(metadata, key, value)
            
            # Add to collection
            self.metadata_collection[metadata.image_hash] = metadata
            
            logger.info(f"Metadata loaded from {file_path}")
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata from {file_path}: {str(e)}")
            return None
    
    def export_collection(self, directory: str) -> bool:
        """
        Export all metadata in the collection to a directory.
        
        Args:
            directory: Directory to save metadata files
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {str(e)}")
                return False
        
        success = True
        for image_hash, metadata in self.metadata_collection.items():
            file_path = os.path.join(directory, f"{image_hash}.json")
            if not self.save_metadata(image_hash, file_path):
                success = False
        
        return success
    
    def get_metadata(self, image_hash: str) -> Optional[SpatialMetadata]:
        """
        Get metadata for a specific image.
        
        Args:
            image_hash: Hash of the image
            
        Returns:
            SpatialMetadata object if found, None otherwise
        """
        return self.metadata_collection.get(image_hash)
    
    def calculate_sensor_size(self, metadata: SpatialMetadata) -> Optional[Tuple[float, float]]:
        """
        Calculate or look up the sensor size based on camera model.
        
        Args:
            metadata: SpatialMetadata object
            
        Returns:
            Tuple of (width, height) in mm if found, None otherwise
        """
        if not metadata.camera_model:
            return None
        
        # Check if we have this camera model in our database
        for model, size in self.known_sensors.items():
            if model.lower() in metadata.camera_model.lower():
                return size
        
        # If we don't know the exact model, make an educated guess
        if "iphone" in metadata.camera_model.lower():
            return (6.86, 5.14)  # Generic iPhone sensor size
        elif "pixel" in metadata.camera_model.lower():
            return (6.2, 4.65)  # Generic Pixel sensor size
        elif "galaxy" in metadata.camera_model.lower():
            return (6.3, 4.7)  # Generic Galaxy sensor size
        
        # For DSLR/mirrorless, try to guess based on keywords
        if any(keyword in metadata.camera_model.lower() for keyword in ["nikon", "canon", "sony", "fuji"]):
            if "full" in metadata.camera_model.lower() or "fx" in metadata.camera_model.lower():
                return (36.0, 24.0)  # Full frame
            else:
                return (23.6, 15.7)  # APS-C
        
        # Unknown camera
        logger.warning(f"Unknown camera model: {metadata.camera_model}, can't determine sensor size")
        return None
