# Documentation for temporal_demo.py

```python
"""
Temporal Analysis Demo for Negative Space Imaging

This demo shows how negative spaces can be tracked and analyzed over time,
demonstrating temporal signature generation and change detection.

Usage:
    python temporal_demo.py [--num_frames NUM] [--output_dir DIR]

Example:
    python temporal_demo.py --num_frames 10 --output_dir output/temporal
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import project modules
try:
    from src.temporal_variants.negative_space_tracker import (
        NegativeSpaceTracker, ChangeMetrics, TemporalSignature, ChangeType
    )
    from simplified_demo import SimplePointCloud, generate_test_scene
except ImportError:
    logger.error("Failed to import required modules. Make sure you're running from the project root.")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Temporal Analysis for Negative Space")
    
    parser.add_argument('--num_frames', type=int, default=10,
                       help='Number of frames to generate and analyze (default: 10)')
    
    parser.add_argument('--output_dir', type=str, default='output/temporal',
                       help='Directory to save results (default: output/temporal)')
    
    parser.add_argument('--deformation_factor', type=float, default=0.1,
                       help='Factor controlling deformation amount (default: 0.1)')
    
    return parser.parse_args()

def ensure_directory(directory):
    """Ensure a directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def generate_deforming_scene(frame: int, num_frames: int, deformation_factor: float = 0.1) -> SimplePointCloud:
    """
    Generate a scene that changes over time
    
    Args:
        frame: Current frame number
        num_frames: Total number of frames
        deformation_factor: Controls how much the scene deforms
        
    Returns:
        SimplePointCloud: Point cloud for the current frame
    """
    logger.info(f"Generating scene for frame {frame}/{num_frames}")
    
    # Create a point cloud
    point_cloud = SimplePointCloud()
    
    # Progress parameter (0 to 1)
    t = frame / max(1, num_frames - 1)
    
    # Generate points for multiple objects with interesting negative space between them
    
    # Object 1: Sphere on the left that grows
    sphere_points = []
    sphere_colors = []
    sphere_labels = []
    
    # Sphere grows over time
    radius = 0.5 + 0.3 * t
    
    for _ in range(1000):
        # Random point on a sphere
        theta = np.random.random() * 2 * np.pi
        phi = np.random.random() * np.pi
        r = radius + np.random.random() * 0.1  # Slight variation in radius
        
        x = r * np.sin(phi) * np.cos(theta) - 1.5  # Shift left
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        sphere_points.append([x, y, z])
        sphere_colors.append([0, 0, 1])  # Blue
        sphere_labels.append(0)  # Object 0
    
    # Object 2: Cube in the middle that rotates
    cube_points = []
    cube_colors = []
    cube_labels = []
    
    # Rotation angle increases over time
    angle = t * np.pi / 2  # Rotate up to 90 degrees
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    for _ in range(1000):
        # Random point near a cube surface
        face = np.random.randint(0, 6)
        
        if face == 0:  # Front
            x = -0.25 + np.random.random() * 0.1
            y = np.random.random() - 0.5
            z = np.random.random() - 0.5
        elif face == 1:  # Back
            x = 0.25 + np.random.random() * 0.1
            y = np.random.random() - 0.5
            z = np.random.random() - 0.5
        elif face == 2:  # Left
            x = -0.25 + np.random.random() * 0.5
            y = -0.5 - np.random.random() * 0.1
            z = np.random.random() - 0.5
        elif face == 3:  # Right
            x = -0.25 + np.random.random() * 0.5
            y = 0.5 + np.random.random() * 0.1
            z = np.random.random() - 0.5
        elif face == 4:  # Bottom
            x = -0.25 + np.random.random() * 0.5
            y = np.random.random() - 0.5
            z = -0.5 - np.random.random() * 0.1
        else:  # Top
            x = -0.25 + np.random.random() * 0.5
            y = np.random.random() - 0.5
            z = 0.5 + np.random.random() * 0.1
        
        # Apply rotation
        x_rot = x
        y_rot = y * cos_angle - z * sin_angle
        z_rot = y * sin_angle + z * cos_angle
        
        cube_points.append([x_rot, y_rot, z_rot])
        cube_colors.append([0, 1, 0])  # Green
        cube_labels.append(1)  # Object 1
    
    # Object 3: Cylinder on the right that moves
    cylinder_points = []
    cylinder_colors = []
    cylinder_labels = []
    
    # Cylinder moves up over time
    z_offset = -0.5 + t  # Move from -0.5 to +0.5
    
    for _ in range(1000):
        # Random point on a cylinder
        theta = np.random.random() * 2 * np.pi
        h = np.random.random() - 0.5  # Height along cylinder axis
        r = 0.4 + np.random.random() * 0.1  # Slight variation in radius
        
        x = 1.5 + r * np.cos(theta)  # Shift right
        y = r * np.sin(theta)
        z = h + z_offset  # Apply vertical offset
        
        cylinder_points.append([x, y, z])
        cylinder_colors.append([1, 0, 0])  # Red
        cylinder_labels.append(2)  # Object 2
    
    # Combine points
    all_points = np.array(sphere_points + cube_points + cylinder_points)
    all_colors = np.array(sphere_colors + cube_colors + cylinder_colors)
    all_labels = np.array(sphere_labels + cube_labels + cylinder_labels)
    
    # Add points to the point cloud
    point_cloud.add_points(all_points, all_colors, all_labels)
    
    # Classify points
    point_cloud.classify_points()
    
    # Generate void points
    point_cloud.generate_void_points()
    
    logger.info(f"Generated point cloud with {len(point_cloud.points)} points")
    logger.info(f"  Object points: {len(point_cloud.object_points)}")
    logger.info(f"  Void points: {len(point_cloud.void_points)}")
    logger.info(f"  Boundary points: {len(point_cloud.boundary_points)}")
    
    return point_cloud

def run_temporal_demo(args):
    """Run the temporal analysis demo"""
    logger.info("=== Running Temporal Analysis Demo ===")
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    # Initialize tracker
    tracker = NegativeSpaceTracker()
    
    # Generate sequence of point clouds with temporal changes
    clouds = []
    for frame in range(args.num_frames):
        cloud = generate_deforming_scene(frame, args.num_frames, args.deformation_factor)
        
        # Save visualization of each frame
        frame_vis_path = os.path.join(args.output_dir, f"frame_{frame:03d}.png")
        cloud.visualize(frame_vis_path)
        
        # Add to tracker and get change metrics
        metrics = tracker.add_point_cloud(cloud)
        change_type = tracker.get_change_type(metrics)
        
        logger.info(f"Frame {frame}: Change type = {change_type.name}, "
                   f"Void count delta = {metrics.void_count_delta}, "
                   f"Volume delta = {metrics.volume_delta:.4f}")
        
        # Store cloud for later use
        clouds.append(cloud)
    
    # Visualize changes over time
    changes_vis_path = os.path.join(args.output_dir, "temporal_changes.png")
    tracker.visualize_changes(changes_vis_path)
    
    # Get and save the temporal signature
    temporal_signature = tracker.get_temporal_signature()
    signature_path = os.path.join(args.output_dir, "temporal_signature.npy")
    temporal_signature.save(signature_path)
    
    logger.info(f"Temporal demo completed. Results saved to {args.output_dir}")
    
    return clouds, tracker

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Run the temporal demo
    clouds, tracker = run_temporal_demo(args)
    
    logger.info("Temporal analysis demo completed successfully!")

if __name__ == "__main__":
    main()

```