# Documentation for model_assembler_demo.py

```python
"""
Model Assembler Demo

This demo showcases the complete Phase 1 pipeline for Negative Space Imaging,
bringing together all components:
1. Point cloud generation with specialized void space detection
2. Interstitial space analysis and classification
3. Full model assembly with both objects and negative spaces
4. Spatial signature generation and visualization

Usage:
    python model_assembler_demo.py [--output_dir DIR] [--show_vis]

Example:
    python model_assembler_demo.py --output_dir output/assembled_models --show_vis
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import open3d as o3d
import logging

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reconstruction.point_cloud_generator import (
    PointCloudGenerator, PointCloudType, PointCloudParams,
    NegativeSpacePointCloud
)
from src.reconstruction.interstitial_analyzer import (
    InterstitialAnalyzer, InterstitialRegion
)
from src.reconstruction.model_assembler import (
    ModelAssembler, ModelComponent, NegativeSpaceComponent, ComponentType
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Model Assembler Demo")
    
    parser.add_argument('--output_dir', type=str, default='output/assembled_models',
                       help='Directory to save results (default: output/assembled_models)')
    
    parser.add_argument('--num_points', type=int, default=8000,
                       help='Number of points to generate (default: 8000)')
    
    parser.add_argument('--show_vis', action='store_true',
                       help='Show interactive visualizations')
    
    parser.add_argument('--interstitial_method', type=str, default='voronoi',
                       choices=['voronoi', 'dbscan', 'kmeans'],
                       help='Method for interstitial space analysis (default: voronoi)')
    
    parser.add_argument('--mesh_method', type=str, default='alpha_shape',
                       choices=['alpha_shape', 'ball_pivoting', 'poisson'],
                       help='Method for mesh creation (default: alpha_shape)')
    
    return parser.parse_args()

def ensure_directory(directory):
    """Ensure a directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def generate_test_scene(num_points: int = 8000) -> NegativeSpacePointCloud:
    """
    Generate a test scene with multiple objects and interesting negative space.
    
    Args:
        num_points: Number of points to generate
        
    Returns:
        NegativeSpacePointCloud object with labeled points
    """
    print("Generating test scene with multiple objects...")
    
    # Create a point cloud
    point_cloud = NegativeSpacePointCloud()
    
    # Generate points for multiple objects with interesting negative space between them
    
    # Object 1: Sphere on the left
    sphere_points = []
    sphere_colors = []
    sphere_labels = []
    
    for _ in range(num_points // 5):
        # Random point on a sphere
        theta = np.random.random() * 2 * np.pi
        phi = np.random.random() * np.pi
        r = 0.5 + np.random.random() * 0.1  # Slight variation in radius
        
        x = r * np.sin(phi) * np.cos(theta) - 1.5  # Shift left
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        sphere_points.append([x, y, z])
        sphere_colors.append([0, 0, 1])  # Blue
        sphere_labels.append(0)  # Object 0
    
    # Object 2: Cube in the middle
    cube_points = []
    cube_colors = []
    cube_labels = []
    
    for _ in range(num_points // 5):
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
        cube_labels.append(1)  # Object 1
    
    # Object 3: Cylinder on the right
    cylinder_points = []
    cylinder_colors = []
    cylinder_labels = []
    
    for _ in range(num_points // 5):
        # Random point on a cylinder
        theta = np.random.random() * 2 * np.pi
        h = np.random.random() - 0.5  # Height along cylinder axis
        r = 0.4 + np.random.random() * 0.1  # Slight variation in radius
        
        x = 1.5 + r * np.cos(theta)  # Shift right
        y = r * np.sin(theta)
        z = h
        
        cylinder_points.append([x, y, z])
        cylinder_colors.append([1, 0, 0])  # Red
        cylinder_labels.append(2)  # Object 2
    
    # Object 4: Torus at the bottom
    torus_points = []
    torus_colors = []
    torus_labels = []
    
    for _ in range(num_points // 5):
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
        torus_labels.append(3)  # Object 3
    
    # Object 5: Small pyramid at the top
    pyramid_points = []
    pyramid_colors = []
    pyramid_labels = []
    
    for _ in range(num_points // 5):
        # Random point on a pyramid
        h = np.random.random()  # Height factor
        base_x = np.random.random() - 0.5
        base_y = np.random.random() - 0.5
        
        x = 0.5 * (1 - h) * base_x
        y = 0.5 * (1 - h) * base_y
        z = h + 1.5  # Shift up
        
        pyramid_points.append([x, y, z])
        pyramid_colors.append([1, 0, 1])  # Magenta
        pyramid_labels.append(4)  # Object 4
    
    # Combine points
    all_points = np.array(sphere_points + cube_points + cylinder_points + torus_points + pyramid_points)
    all_colors = np.array(sphere_colors + cube_colors + cylinder_colors + torus_colors + pyramid_colors)
    all_labels = np.array(sphere_labels + cube_labels + cylinder_labels + torus_labels + pyramid_labels)
    
    # Add points to the point cloud
    point_cloud.add_points(all_points, all_colors)
    
    # Classify points
    point_cloud.classify_points()
    
    # Generate void points
    point_cloud.generate_void_points(density=1000)
    
    print(f"Generated point cloud with {len(point_cloud.points)} points")
    print(f"  Object points: {len(point_cloud.object_points)}")
    print(f"  Boundary points: {len(point_cloud.boundary_points)}")
    print(f"  Void points: {len(point_cloud.void_points)}")
    
    # Store object labels
    point_cloud.object_labels = all_labels
    
    return point_cloud

def analyze_interstitial_spaces(point_cloud: NegativeSpacePointCloud, 
                               method: str = 'voronoi') -> InterstitialAnalyzer:
    """
    Analyze interstitial spaces in the point cloud.
    
    Args:
        point_cloud: Input point cloud
        method: Method for interstitial space analysis
        
    Returns:
        InterstitialAnalyzer object
    """
    print("\n=== Analyzing Interstitial Spaces ===")
    
    # Create interstitial analyzer
    analyzer = InterstitialAnalyzer()
    
    # Set object points
    analyzer.set_object_points(point_cloud.object_points, point_cloud.object_labels)
    
    # Set void points
    analyzer.set_void_points(point_cloud.void_points)
    
    # Analyze interstitial spaces
    regions = analyzer.analyze(method=method)
    
    print(f"Identified {len(regions)} interstitial regions")
    
    return analyzer

def assemble_model(point_cloud: NegativeSpacePointCloud, 
                  analyzer: InterstitialAnalyzer,
                  mesh_method: str = 'alpha_shape') -> ModelAssembler:
    """
    Assemble a complete model from the point cloud and interstitial analysis.
    
    Args:
        point_cloud: Input point cloud
        analyzer: Interstitial analyzer with identified regions
        mesh_method: Method for mesh creation
        
    Returns:
        ModelAssembler object
    """
    print("\n=== Assembling Complete Model ===")
    
    # Create model assembler
    assembler = ModelAssembler()
    
    # Add object components
    unique_labels = np.unique(point_cloud.object_labels)
    for label in unique_labels:
        mask = point_cloud.object_labels == label
        obj_points = point_cloud.object_points[mask]
        
        # Generate color based on label
        r = (label * 67) % 255 / 255.0
        g = (label * 101) % 255 / 255.0
        b = (label * 191) % 255 / 255.0
        color = np.array([r, g, b])
        
        # Create component
        assembler.create_component_from_points(
            id=int(label),
            type=ComponentType.OBJECT,
            points=obj_points,
            color=color,
            name=f"Object_{label}",
            mesh_method=mesh_method
        )
    
    # Add negative space components from interstitial regions
    for i, region in enumerate(analyzer.regions):
        # Create negative space component
        r = 0.7 + (i * 0.05) % 0.3  # Reddish color
        g = 0.3 + (i * 0.1) % 0.3   # Some green
        b = 0.3 + (i * 0.15) % 0.3  # Some blue
        color = np.array([r, g, b])
        
        ns_comp = assembler.create_component_from_points(
            id=i + 100,  # Offset to avoid conflict with object IDs
            type=ComponentType.NEGATIVE_SPACE,
            points=region.points,
            color=color,
            name=f"NegativeSpace_{i}",
            mesh_method=mesh_method
        )
        
        # Set adjacent objects
        if isinstance(ns_comp, NegativeSpaceComponent):
            ns_comp.adjacent_objects = region.adjacent_objects
    
    # Assemble the model
    assembler.assemble()
    
    print(f"Model assembled with {len(assembler.components)} components")
    print(f"  Object components: {len([c for c in assembler.components if c.type == ComponentType.OBJECT])}")
    print(f"  Negative space components: {len(assembler.negative_space_components)}")
    
    return assembler

def create_visualizations(point_cloud: NegativeSpacePointCloud,
                         analyzer: InterstitialAnalyzer,
                         assembler: ModelAssembler,
                         output_dir: str,
                         show_vis: bool = False):
    """
    Create visualizations of the point cloud, interstitial regions, and assembled model.
    
    Args:
        point_cloud: Input point cloud
        analyzer: Interstitial analyzer
        assembler: Model assembler
        output_dir: Directory to save visualizations
        show_vis: Whether to show interactive visualizations
    """
    print("\n=== Creating Visualizations ===")
    
    # Create directories
    vis_dir = os.path.join(output_dir, "visualizations")
    ensure_directory(vis_dir)
    
    # 1. Point cloud visualization
    if show_vis:
        point_cloud.visualize(show_classification=True)
    
    # Save point cloud
    pc_vis_path = os.path.join(vis_dir, "point_cloud.png")
    
    # Create a static visualization
    # (This will be handled by Open3D capture_screen functionality)
    
    # 2. Interstitial regions visualization
    if show_vis:
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Interstitial Regions")
        
        # Add object point cloud
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(point_cloud.object_points)
        obj_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([0.5, 0.5, 0.5]), (len(point_cloud.object_points), 1))
        )
        vis.add_geometry(obj_pcd)
        
        # Add region meshes
        region_meshes = analyzer.visualize_all_regions()
        for mesh in region_meshes:
            vis.add_geometry(mesh)
        
        # Set rendering options
        opt = vis.get_render_option()
        opt.mesh_show_wireframe = True
        opt.mesh_show_back_face = True
        
        # Run visualization
        vis.run()
        vis.destroy_window()
    
    # 3. Assembled model visualization
    if show_vis:
        assembler.visualize(show_negative_space=True, negative_space_opacity=0.5)
    
    # 4. Spatial signature visualization
    plt.figure(figsize=(12, 6))
    plt.plot(assembler.global_signature, marker='o', linestyle='-', color='b')
    plt.title("Global Spatial Signature")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.grid(True, alpha=0.3)
    
    # Save signature visualization
    sig_vis_path = os.path.join(vis_dir, "global_signature.png")
    plt.savefig(sig_vis_path, dpi=300, bbox_inches='tight')
    
    if show_vis:
        plt.show()
    else:
        plt.close()
    
    # 5. Adjacency visualization
    plt.figure(figsize=(10, 8))
    
    # Create adjacency matrix
    num_objs = len([c for c in assembler.components if c.type == ComponentType.OBJECT])
    num_ns = len(assembler.negative_space_components)
    adj_matrix = np.zeros((num_ns, num_objs))
    
    for i, ns_comp in enumerate(assembler.negative_space_components):
        for obj_id in ns_comp.adjacent_objects:
            if obj_id < num_objs:  # Ensure valid object ID
                adj_matrix[i, obj_id] = 1
    
    # Plot adjacency matrix
    plt.imshow(adj_matrix, cmap='Blues', aspect='auto')
    plt.colorbar(label='Adjacency')
    plt.title("Negative Space - Object Adjacency")
    plt.xlabel("Object ID")
    plt.ylabel("Negative Space ID")
    plt.grid(False)
    
    # Add text annotations
    for i in range(num_ns):
        for j in range(num_objs):
            if adj_matrix[i, j] > 0:
                plt.text(j, i, "X", ha="center", va="center", color="w")
    
    # Save adjacency visualization
    adj_vis_path = os.path.join(vis_dir, "adjacency.png")
    plt.savefig(adj_vis_path, dpi=300, bbox_inches='tight')
    
    if show_vis:
        plt.show()
    else:
        plt.close()
    
    print(f"Visualizations saved to {vis_dir}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Ensure output directory exists
    ensure_directory(args.output_dir)
    
    # Generate test scene
    point_cloud = generate_test_scene(args.num_points)
    
    # Analyze interstitial spaces
    analyzer = analyze_interstitial_spaces(point_cloud, args.interstitial_method)
    
    # Assemble model
    assembler = assemble_model(point_cloud, analyzer, args.mesh_method)
    
    # Save assembled model
    assembler.save(args.output_dir)
    
    # Create visualizations
    create_visualizations(point_cloud, analyzer, assembler, args.output_dir, args.show_vis)
    
    print("\nDemo completed successfully!")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

```