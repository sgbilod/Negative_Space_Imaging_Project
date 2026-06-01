# Documentation for negative_space_demo.py

```python
"""
Negative Space Analyzer Demo

This advanced demo showcases the innovative negative space analysis capabilities
of the project, including:
1. Point cloud generation with focus on negative space
2. Interstitial space analysis between objects
3. Spatial signature generation and visualization
4. Void mesh generation for negative space visualization

Usage:
    python negative_space_demo.py [--demo_type TYPE] [--output_dir DIR]

Example:
    python negative_space_demo.py --demo_type interstitial --output_dir output/demos
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import open3d as o3d

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reconstruction.point_cloud_generator import (
    PointCloudGenerator, PointCloudType, PointCloudParams,
    NegativeSpacePointCloud
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Negative Space Analysis Demo")
    
    parser.add_argument('--demo_type', type=str, default='interstitial',
                       choices=['basic', 'interstitial', 'signature', 'void_mesh', 'all'],
                       help='Type of demo to run (default: interstitial)')
    
    parser.add_argument('--output_dir', type=str, default='output/demos',
                       help='Directory to save results (default: output/demos)')
    
    parser.add_argument('--num_points', type=int, default=5000,
                       help='Number of points to generate (default: 5000)')
    
    parser.add_argument('--show_visualizations', action='store_true',
                       help='Show interactive visualizations')
    
    return parser.parse_args()

def ensure_directory(directory):
    """Ensure a directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def generate_test_scene(num_points: int = 5000) -> NegativeSpacePointCloud:
    """
    Generate a test scene with multiple objects and interesting negative space.
    
    Args:
        num_points: Number of points to generate
        
    Returns:
        NegativeSpacePointCloud object
    """
    print("Generating test scene with multiple objects...")
    
    # Create a point cloud
    point_cloud = NegativeSpacePointCloud()
    
    # Generate points for multiple objects with interesting negative space between them
    
    # Object 1: Sphere on the left
    sphere_points = []
    sphere_colors = []
    
    for _ in range(num_points // 4):
        # Random point on a sphere
        theta = np.random.random() * 2 * np.pi
        phi = np.random.random() * np.pi
        r = 0.5 + np.random.random() * 0.1  # Slight variation in radius
        
        x = r * np.sin(phi) * np.cos(theta) - 1.5  # Shift left
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        sphere_points.append([x, y, z])
        sphere_colors.append([0, 0, 1])  # Blue
    
    # Object 2: Cube in the middle
    cube_points = []
    cube_colors = []
    
    for _ in range(num_points // 4):
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
        
        cube_points.append([x, y, z])
        cube_colors.append([0, 1, 0])  # Green
    
    # Object 3: Cylinder on the right
    cylinder_points = []
    cylinder_colors = []
    
    for _ in range(num_points // 4):
        # Random point on a cylinder
        theta = np.random.random() * 2 * np.pi
        h = np.random.random() - 0.5  # Height along cylinder axis
        r = 0.4 + np.random.random() * 0.1  # Slight variation in radius
        
        x = 1.5 + r * np.cos(theta)  # Shift right
        y = r * np.sin(theta)
        z = h
        
        cylinder_points.append([x, y, z])
        cylinder_colors.append([1, 0, 0])  # Red
    
    # Object 4: Torus at the bottom
    torus_points = []
    torus_colors = []
    
    for _ in range(num_points // 4):
        # Random point on a torus
        u = np.random.random() * 2 * np.pi
        v = np.random.random() * 2 * np.pi
        r = 0.2  # Minor radius
        R = 0.7  # Major radius
        
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v) - 1.5  # Shift down
        
        torus_points.append([x, y, z])
        torus_colors.append([1, 1, 0])  # Yellow
    
    # Combine points
    all_points = np.array(sphere_points + cube_points + cylinder_points + torus_points)
    all_colors = np.array(sphere_colors + cube_colors + cylinder_colors + torus_colors)
    
    # Add points to the point cloud
    point_cloud.add_points(all_points, all_colors)
    
    # Classify points
    point_cloud.classify_points()
    
    print(f"Generated point cloud with {len(point_cloud.points)} points")
    print(f"  Object points: {len(point_cloud.object_points)}")
    print(f"  Boundary points: {len(point_cloud.boundary_points)}")
    print(f"  Void points: {len(point_cloud.void_points)}")
    
    return point_cloud

def run_basic_demo(args):
    """Run the basic negative space demo"""
    print("\n=== Running Basic Negative Space Demo ===")
    
    # Create a point cloud generator with negative space optimization
    generator = PointCloudGenerator(
        cloud_type=PointCloudType.NEGATIVE_SPACE_OPTIMIZED,
        params=PointCloudParams(
            point_density=2000,
            void_sampling_ratio=0.6
        )
    )
    
    # Generate a point cloud
    point_cloud = generate_test_scene(args.num_points)
    
    # Save point cloud
    output_path = os.path.join(args.output_dir, "basic_negative_space.ply")
    ensure_directory(args.output_dir)
    point_cloud.save(output_path)
    
    print(f"Basic point cloud saved to {output_path}")
    
    # Visualize
    if args.show_visualizations:
        point_cloud.visualize(show_classification=True)
    
    return point_cloud

def run_interstitial_demo(args):
    """Run the interstitial space analysis demo"""
    print("\n=== Running Interstitial Space Analysis Demo ===")
    
    # Create a point cloud generator with interstitial optimization
    generator = PointCloudGenerator(
        cloud_type=PointCloudType.INTERSTITIAL,
        params=PointCloudParams(
            point_density=2000,
            void_sampling_ratio=0.7,
            min_interstitial_distance=0.1,
            max_interstitial_distance=2.0
        )
    )
    
    # Generate a point cloud
    point_cloud = generate_test_scene(args.num_points)
    
    # Compute interstitial spaces
    interstitial_spaces = point_cloud.compute_interstitial_spaces()
    
    print(f"Identified {len(point_cloud.interstitial_regions)} interstitial regions")
    
    # Save point cloud
    output_path = os.path.join(args.output_dir, "interstitial_analysis.ply")
    ensure_directory(args.output_dir)
    point_cloud.save(output_path)
    
    print(f"Interstitial analysis point cloud saved to {output_path}")
    
    # Visualize
    if args.show_visualizations:
        point_cloud.visualize(show_classification=True, show_interstitial=True)
    
    return point_cloud

def run_signature_demo(args):
    """Run the spatial signature generation demo"""
    print("\n=== Running Spatial Signature Generation Demo ===")
    
    # Generate a point cloud
    point_cloud = generate_test_scene(args.num_points)
    
    # Compute spatial signature
    signature = point_cloud.compute_spatial_signature()
    
    print(f"Generated spatial signature with {len(signature)} features")
    
    # Visualize signature
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(signature)), signature)
    plt.title("Negative Space Spatial Signature")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.grid(True, alpha=0.3)
    
    # Save visualization
    output_path = os.path.join(args.output_dir, "spatial_signature.png")
    ensure_directory(args.output_dir)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Spatial signature visualization saved to {output_path}")
    
    # Save point cloud
    pc_output_path = os.path.join(args.output_dir, "signature_analysis.ply")
    point_cloud.save(pc_output_path)
    
    # Show plot
    if args.show_visualizations:
        plt.show()
    else:
        plt.close()
    
    return point_cloud, signature

def run_void_mesh_demo(args):
    """Run the void mesh generation demo"""
    print("\n=== Running Void Mesh Generation Demo ===")
    
    # Generate a point cloud
    point_cloud = generate_test_scene(args.num_points)
    
    # Generate void mesh
    print("Generating mesh representation of negative space...")
    void_mesh = point_cloud.generate_void_mesh()
    
    if void_mesh is None:
        print("Failed to generate void mesh")
        return point_cloud
    
    print("Void mesh generated successfully")
    
    # Save mesh
    output_path = os.path.join(args.output_dir, "void_mesh.ply")
    ensure_directory(args.output_dir)
    o3d.io.write_triangle_mesh(output_path, void_mesh)
    
    print(f"Void mesh saved to {output_path}")
    
    # Visualize
    if args.show_visualizations:
        # Visualize the mesh
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add original points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.points)
        pcd.colors = o3d.utility.Vector3dVector(point_cloud.colors)
        vis.add_geometry(pcd)
        
        # Add void mesh with transparency
        void_mesh.paint_uniform_color([1, 0, 0])  # Red
        vis.add_geometry(void_mesh)
        
        # Set rendering options
        opt = vis.get_render_option()
        opt.mesh_show_wireframe = True
        opt.mesh_show_back_face = True
        
        # Run visualization
        vis.run()
        vis.destroy_window()
    
    return point_cloud, void_mesh

def compare_signatures(point_cloud, args):
    """Compare spatial signatures from different configurations"""
    print("\n=== Running Signature Comparison Demo ===")
    
    # Create directory for signature comparison
    comparison_dir = os.path.join(args.output_dir, "signature_comparison")
    ensure_directory(comparison_dir)
    
    # Generate the original signature
    original_signature = point_cloud.compute_spatial_signature()
    
    # Create variations of the point cloud
    variations = []
    
    # Variation 1: Move one object
    variation1 = NegativeSpacePointCloud(point_cloud.points.copy(), point_cloud.colors.copy())
    mask = variation1.points[:, 0] < -1.0  # Sphere on the left
    variation1.points[mask, 0] += 0.5  # Move right
    variations.append(("Object Moved", variation1))
    
    # Variation 2: Add a new object
    variation2 = NegativeSpacePointCloud(point_cloud.points.copy(), point_cloud.colors.copy())
    # Add a small pyramid
    pyramid_points = []
    pyramid_colors = []
    for _ in range(1000):
        # Random point on a pyramid
        h = np.random.random()  # Height factor
        base_x = np.random.random() - 0.5
        base_y = np.random.random() - 0.5
        
        x = 0.5 * (1 - h) * base_x
        y = 0.5 * (1 - h) * base_y
        z = h - 0.5
        
        pyramid_points.append([x, y, z])
        pyramid_colors.append([0.5, 0.5, 1.0])  # Light blue
    
    variation2.add_points(np.array(pyramid_points), np.array(pyramid_colors))
    variations.append(("Object Added", variation2))
    
    # Variation 3: Remove an object
    variation3 = NegativeSpacePointCloud()
    mask = point_cloud.points[:, 0] < 1.0  # Everything except cylinder
    variation3.add_points(point_cloud.points[mask], point_cloud.colors[mask])
    variations.append(("Object Removed", variation3))
    
    # Compute signatures for all variations
    signatures = [original_signature]
    
    for name, var_cloud in variations:
        # Classify points
        var_cloud.classify_points()
        
        # Compute signature
        signature = var_cloud.compute_spatial_signature()
        signatures.append(signature)
        
        # Save point cloud
        output_path = os.path.join(comparison_dir, f"{name.lower().replace(' ', '_')}.ply")
        var_cloud.save(output_path)
        
        print(f"Generated signature for variation: {name}")
    
    # Visualize and compare signatures
    plt.figure(figsize=(12, 8))
    
    labels = ["Original"] + [name for name, _ in variations]
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for i, (signature, label, color) in enumerate(zip(signatures, labels, colors)):
        plt.plot(range(len(signature)), signature, label=label, color=color, marker='o', linestyle='-', alpha=0.7)
    
    plt.title("Comparison of Negative Space Signatures")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save visualization
    output_path = os.path.join(comparison_dir, "signature_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Signature comparison visualization saved to {output_path}")
    
    # Show plot
    if args.show_visualizations:
        plt.show()
    else:
        plt.close()
    
    return signatures

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Run the requested demo
    if args.demo_type == 'basic' or args.demo_type == 'all':
        point_cloud = run_basic_demo(args)
    
    if args.demo_type == 'interstitial' or args.demo_type == 'all':
        point_cloud = run_interstitial_demo(args)
    
    if args.demo_type == 'signature' or args.demo_type == 'all':
        point_cloud, signature = run_signature_demo(args)
        
        # Run signature comparison if we're doing the signature demo
        compare_signatures(point_cloud, args)
    
    if args.demo_type == 'void_mesh' or args.demo_type == 'all':
        point_cloud, void_mesh = run_void_mesh_demo(args)
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()

```