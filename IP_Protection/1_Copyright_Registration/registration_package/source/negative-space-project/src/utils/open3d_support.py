"""
Open3D Support Module

This module provides type checking support for Open3D
and handles common import issues.

Usage:
    from src.utils.open3d_support import o3d, np
"""

import os
import sys
import logging
from typing import Optional, Any, Dict, List, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import Open3D with fallbacks
try:
    import open3d as o3d
    logger.info("Successfully imported Open3D")
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("Open3D import failed. Providing fallback implementation.")
    OPEN3D_AVAILABLE = False
    
    # Create a basic fallback for Open3D
    class FallbackOpen3D:
        """Fallback implementation for Open3D when not available"""
        
        class geometry:
            class PointCloud:
                def __init__(self):
                    self.points = None
                    self.colors = None
                    self.normals = None
                    logger.warning("Using fallback PointCloud - Open3D not available")
                    
            class TriangleMesh:
                def __init__(self):
                    self.vertices = None
                    self.triangles = None
                    logger.warning("Using fallback TriangleMesh - Open3D not available")
        
        class utility:
            class Vector3dVector:
                def __init__(self, vector):
                    self.vector = vector
                    
            class Vector3iVector:
                def __init__(self, vector):
                    self.vector = vector
        
        class visualization:
            class Visualizer:
                def create_window(self):
                    logger.warning("Visualization not available - Open3D not available")
                    return False
    
    # Use the fallback
    o3d = FallbackOpen3D()

# Import numpy
import numpy as np

def check_open3d_installation() -> bool:
    """
    Check if Open3D is properly installed.
    
    Returns:
        bool: True if Open3D is available, False otherwise
    """
    return OPEN3D_AVAILABLE

def get_open3d_version() -> str:
    """
    Get the version of Open3D.
    
    Returns:
        str: Open3D version or "Not available"
    """
    if OPEN3D_AVAILABLE:
        try:
            return o3d.__version__
        except AttributeError:
            return "Unknown"
    return "Not available"

# Additional utility functions for Open3D
def create_coordinate_frame(size: float = 1.0, origin: List[float] = None) -> Any:
    """
    Create a coordinate frame for visualization.
    
    Args:
        size: Size of the coordinate frame
        origin: Origin of the coordinate frame
        
    Returns:
        Open3D geometry object
    """
    if not OPEN3D_AVAILABLE:
        logger.warning("Cannot create coordinate frame - Open3D not available")
        return None
        
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin if origin else [0, 0, 0])
    return mesh
