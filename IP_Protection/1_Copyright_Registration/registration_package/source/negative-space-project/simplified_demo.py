"""
Simplified Negative Space Demo

This version of the demo works without Open3D, focusing on the concepts
rather than full 3D visualization.

Usage:
    python simplified_demo.py [--demo_type TYPE] [--output_dir DIR]

Example:
    python simplified_demo.py --demo_type basic
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Simplified Negative Space Demo")
    
    parser.add_argument('--demo_type', type=str, default='basic',
                       choices=['basic', 'signature', 'all'],
                       help='Type of demo to run (default: basic)')
    
    parser.add_argument('--output_dir', type=str, default='output/demos',
                       help='Directory to save results (default: output/demos)')
    
    return parser.parse_args()

def ensure_directory(directory):
    """Ensure a directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

class SimplePointCloud:
    """A simplified point cloud class that doesn't rely on Open3D"""
    
    def __init__(self):
        """Initialize an empty point cloud"""
        self.points = np.array([])
        self.colors = np.array([])
        self.labels = np.array([])
        self.object_points = np.array([])
        self.void_points = np.array([])
        self.boundary_points = np.array([])
    
    def add_points(self, points, colors=None, labels=None):
        """Add points to the point cloud"""
        if len(self.points) == 0:
            self.points = points
            if colors is not None:
                self.colors = colors
            if labels is not None:
                self.labels = labels
        else:
            self.points = np.vstack([self.points, points])
            if colors is not None:
                if len(self.colors) == 0:
                    self.colors = colors
                else:
                    self.colors = np.vstack([self.colors, colors])
            if labels is not None:
                if len(self.labels) == 0:
                    self.labels = labels
                else:
                    self.labels = np.append(self.labels, labels)
    
    def classify_points(self):
        """
        Classify points as objects, boundaries, or voids.
        In this simplified version, we just simulate the classification.
        """
        logger.info("Classifying points...")
        
        # If we have labels, use them for classification
        if len(self.labels) > 0:
            # Points with label >= 0 are object points
            self.object_points = self.points[self.labels >= 0]
            # Simulate boundary points (random subset of object points)
            boundary_indices = np.random.choice(
                len(self.object_points), 
                size=min(100, len(self.object_points)), 
                replace=False
            )
            self.boundary_points = self.object_points[boundary_indices]
            # No void points in this case
            self.void_points = np.array([])
        else:
            # Randomly classify points in a simplified manner
            num_points = len(self.points)
            if num_points == 0:
                return
                
            # 70% object points, 20% void points, 10% boundary points
            object_ratio = 0.7
            void_ratio = 0.2
            
            # Generate random indices for each category
            indices = np.random.permutation(num_points)
            object_count = int(num_points * object_ratio)
            void_count = int(num_points * void_ratio)
            
            object_indices = indices[:object_count]
            void_indices = indices[object_count:object_count+void_count]
            boundary_indices = indices[object_count+void_count:]
            
            self.object_points = self.points[object_indices]
            self.void_points = self.points[void_indices]
            self.boundary_points = self.points[boundary_indices]
        
        logger.info(f"Classification complete: {len(self.object_points)} object points, "
                   f"{len(self.void_points)} void points, {len(self.boundary_points)} boundary points")
    
    def generate_void_points(self, density=100):
        """Generate void points in the spaces between objects"""
        logger.info("Generating void points...")
        
        if len(self.object_points) == 0:
            logger.warning("No object points available for void point generation")
            return
            
        # Find the bounding box of the object points
        min_coords = np.min(self.object_points, axis=0)
        max_coords = np.max(self.object_points, axis=0)
        
        # Generate random points within the bounding box
        num_void_points = int(len(self.object_points) * 0.5)  # 50% of object count
        
        # Generate random coordinates
        void_points = np.random.uniform(
            min_coords, max_coords, size=(num_void_points, 3)
        )
        
        # Filter points that are too close to object points
        # (simplified approach - in reality we'd use distance calculations)
        filtered_void_points = []
        for point in void_points:
            # Calculate minimum distance to any object point
            min_dist = np.min(np.linalg.norm(self.object_points - point, axis=1))
            # Keep points that are not too close but also not too far
            if 0.2 < min_dist < 1.0:
                filtered_void_points.append(point)
        
        if filtered_void_points:
            self.void_points = np.array(filtered_void_points)
            logger.info(f"Generated {len(self.void_points)} void points")
        else:
            logger.warning("No void points generated after filtering")
    
    def compute_spatial_signature(self, num_features=32):
        """
        Compute a spatial signature based on the distribution of points.
        This is a simplified version that doesn't rely on 3D geometry.
        """
        logger.info("Computing spatial signature...")
        
        # Initialize features
        features = np.zeros(num_features)
        
        if len(self.points) == 0:
            logger.warning("No points available for signature computation")
            return features
            
        # 1. Basic point statistics
        features[0] = len(self.object_points) / max(1, len(self.points))  # Object ratio
        features[1] = len(self.void_points) / max(1, len(self.points))    # Void ratio
        features[2] = len(self.boundary_points) / max(1, len(self.points))  # Boundary ratio
        
        # 2. Spatial distribution features
        if len(self.object_points) > 0:
            # Compute center of mass
            center = np.mean(self.object_points, axis=0)
            
            # Compute average distance from center
            distances = np.linalg.norm(self.object_points - center, axis=1)
            features[3] = np.mean(distances)
            features[4] = np.std(distances)
            features[5] = np.max(distances)
            
            # Compute covariance matrix and its eigenvalues (shape descriptors)
            if len(self.object_points) > 1:
                cov = np.cov(self.object_points, rowvar=False)
                if cov.shape[0] == 3:
                    eigenvalues, _ = np.linalg.eigh(cov)
                    features[6:9] = sorted(eigenvalues, reverse=True)
                    
                    # Shape factors
                    if eigenvalues[0] > 0:
                        features[9] = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]  # Planarity
                        features[10] = eigenvalues[2] / eigenvalues[0]  # Sphericity
                        features[11] = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]  # Linearity
        
        # 3. Void space features (if available)
        if len(self.void_points) > 0:
            # Compute center of mass for void points
            void_center = np.mean(self.void_points, axis=0)
            
            # Distance between object center and void center
            if len(self.object_points) > 0:
                features[12] = np.linalg.norm(center - void_center)
            
            # Average distance between void points
            if len(self.void_points) > 1:
                # Sample pairs of void points to compute average distance
                num_samples = min(100, len(self.void_points))
                sample_indices = np.random.choice(len(self.void_points), num_samples, replace=False)
                sample_points = self.void_points[sample_indices]
                
                distances = []
                for i in range(num_samples):
                    for j in range(i+1, num_samples):
                        dist = np.linalg.norm(sample_points[i] - sample_points[j])
                        distances.append(dist)
                
                if distances:
                    features[13] = np.mean(distances)
                    features[14] = np.std(distances)
                    features[15] = np.max(distances)
        
        # Normalize the features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        logger.info("Spatial signature computed successfully")
        return features
    
    def visualize(self, output_path=None):
        """
        Create a 2D visualization of the point cloud.
        This is a simplified version that doesn't rely on 3D visualization.
        """
        if len(self.points) == 0:
            logger.warning("No points to visualize")
            return
            
        # Create figure for 2D projections
        fig = plt.figure(figsize=(15, 5))
        
        # Create 3 subplots for different projections
        ax1 = fig.add_subplot(131)  # XY plane
        ax2 = fig.add_subplot(132)  # XZ plane
        ax3 = fig.add_subplot(133)  # YZ plane
        
        # Plot object points
        if len(self.object_points) > 0:
            ax1.scatter(self.object_points[:, 0], self.object_points[:, 1], 
                      c='blue', s=2, label='Objects')
            ax2.scatter(self.object_points[:, 0], self.object_points[:, 2], 
                      c='blue', s=2, label='Objects')
            ax3.scatter(self.object_points[:, 1], self.object_points[:, 2], 
                      c='blue', s=2, label='Objects')
        
        # Plot void points
        if len(self.void_points) > 0:
            ax1.scatter(self.void_points[:, 0], self.void_points[:, 1], 
                      c='red', s=2, label='Voids')
            ax2.scatter(self.void_points[:, 0], self.void_points[:, 2], 
                      c='red', s=2, label='Voids')
            ax3.scatter(self.void_points[:, 1], self.void_points[:, 2], 
                      c='red', s=2, label='Voids')
        
        # Plot boundary points
        if len(self.boundary_points) > 0:
            ax1.scatter(self.boundary_points[:, 0], self.boundary_points[:, 1], 
                      c='green', s=2, label='Boundaries')
            ax2.scatter(self.boundary_points[:, 0], self.boundary_points[:, 2], 
                      c='green', s=2, label='Boundaries')
            ax3.scatter(self.boundary_points[:, 1], self.boundary_points[:, 2], 
                      c='green', s=2, label='Boundaries')
        
        # Set labels and titles
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('XY Projection')
        ax1.legend()
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('XZ Projection')
        ax2.legend()
        
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        ax3.set_title('YZ Projection')
        ax3.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()

def generate_test_scene():
    """
    Generate a test scene with multiple objects and interesting negative space.
    
    Returns:
        SimplePointCloud object
    """
    logger.info("Generating test scene with multiple objects...")
    
    # Create a point cloud
    point_cloud = SimplePointCloud()
    
    # Generate points for multiple objects with interesting negative space between them
    
    # Object 1: Sphere on the left
    sphere_points = []
    sphere_colors = []
    sphere_labels = []
    
    for _ in range(1000):
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
        
        cube_points.append([x, y, z])
        cube_colors.append([0, 1, 0])  # Green
        cube_labels.append(1)  # Object 1
    
    # Object 3: Cylinder on the right
    cylinder_points = []
    cylinder_colors = []
    cylinder_labels = []
    
    for _ in range(1000):
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

def run_basic_demo(args):
    """Run the basic demo"""
    logger.info("=== Running Basic Demo ===")
    
    # Generate test scene
    point_cloud = generate_test_scene()
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    # Visualize point cloud
    vis_path = os.path.join(args.output_dir, "basic_visualization.png")
    point_cloud.visualize(vis_path)
    
    logger.info(f"Basic demo completed. Results saved to {args.output_dir}")
    
    return point_cloud

def run_signature_demo(args):
    """Run the signature demo"""
    logger.info("=== Running Signature Demo ===")
    
    # Generate test scene
    point_cloud = generate_test_scene()
    
    # Compute spatial signature
    signature = point_cloud.compute_spatial_signature()
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    # Visualize point cloud
    vis_path = os.path.join(args.output_dir, "signature_visualization.png")
    point_cloud.visualize(vis_path)
    
    # Visualize signature
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(signature)), signature)
    plt.title("Spatial Signature")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.grid(True, alpha=0.3)
    
    sig_path = os.path.join(args.output_dir, "signature.png")
    plt.savefig(sig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save signature as CSV
    csv_path = os.path.join(args.output_dir, "signature.csv")
    np.savetxt(csv_path, signature, delimiter=',')
    
    logger.info(f"Signature demo completed. Results saved to {args.output_dir}")
    
    return point_cloud, signature

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Run the requested demo
    if args.demo_type == 'basic' or args.demo_type == 'all':
        point_cloud = run_basic_demo(args)
    
    if args.demo_type == 'signature' or args.demo_type == 'all':
        point_cloud, signature = run_signature_demo(args)
    
    logger.info("Demo completed successfully!")

if __name__ == "__main__":
    main()
