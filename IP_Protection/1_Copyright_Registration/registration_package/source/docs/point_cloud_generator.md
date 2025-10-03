# Documentation for point_cloud_generator.py

```python
"""
Point Cloud Generator Module for Negative Space Imaging Project

This module implements specialized methods for generating point clouds
with a focus on representing and characterizing negative space regions.
"""

import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum
import time
import os
import random
from scipy.spatial import Delaunay, ConvexHull
from sklearn.cluster import DBSCAN

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PointCloudType(Enum):
    """Types of point cloud generation methods"""
    STANDARD = "standard"  # Standard structure from motion
    NEGATIVE_SPACE_OPTIMIZED = "negative_space_optimized"  # Optimized for negative space
    VOID_FOCUSED = "void_focused"  # Focus specifically on void regions
    BOUNDARY_ENHANCED = "boundary_enhanced"  # Enhanced sampling at object boundaries
    INTERSTITIAL = "interstitial"  # Focus on space between multiple objects
    VOLUMETRIC = "volumetric"  # Create volumetric representation of negative space

@dataclass
class PointCloudParams:
    """Parameters for point cloud generation"""
    # General parameters
    point_density: int = 1000  # Target number of points per cubic meter
    min_points: int = 1000  # Minimum number of points to generate
    max_points: int = 100000  # Maximum number of points to generate
    
    # Reconstruction parameters
    reconstruction_quality: str = "medium"  # "low", "medium", "high"
    depth_filtering: bool = True  # Filter outliers in depth
    
    # Negative space parameters
    void_sampling_ratio: float = 0.7  # Ratio of points to generate in void spaces
    boundary_emphasis: float = 2.0  # Emphasis factor for boundary regions
    void_density_falloff: float = 0.5  # How quickly density falls off in void regions
    
    # Advanced parameters
    cluster_tolerance: float = 0.05  # For clustering points
    min_cluster_size: int = 50  # Minimum points to form a cluster
    outlier_removal_std: float = 2.0  # Standard deviation multiplier for outlier removal
    
    # Volumetric parameters
    voxel_size: float = 0.05  # Size of voxels for volumetric representation
    tsdf_truncation: float = 0.1  # Truncation value for TSDF
    
    # Interstitial parameters
    min_interstitial_distance: float = 0.1  # Minimum distance between objects
    max_interstitial_distance: float = 2.0  # Maximum distance between objects

class NegativeSpacePointCloud:
    """
    Point cloud class specifically optimized for negative space representation.
    
    This class extends the standard point cloud concept with specialized
    attributes and methods for working with negative space.
    """
    
    def __init__(self, points: Optional[np.ndarray] = None, 
                colors: Optional[np.ndarray] = None):
        """
        Initialize the negative space point cloud.
        
        Args:
            points: Optional array of 3D points
            colors: Optional array of colors corresponding to points
        """
        # Core point cloud data
        self.points = points if points is not None else np.array([])
        self.colors = colors if colors is not None else np.array([])
        
        # Negative space specific attributes
        self.void_points = np.array([])  # Points in void regions
        self.boundary_points = np.array([])  # Points along boundaries
        self.object_points = np.array([])  # Points on objects
        
        # Classification masks
        self.void_mask = np.array([])
        self.boundary_mask = np.array([])
        self.object_mask = np.array([])
        
        # Spatial structures
        self.void_voxels = None  # Voxelized representation of void spaces
        self.boundary_mesh = None  # Mesh representation of boundaries
        self.interstitial_graph = None  # Graph of connections between void regions
        
        # Metadata
        self.metadata = {}
    
    def add_points(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        """
        Add points to the point cloud.
        
        Args:
            points: Array of 3D points to add
            colors: Optional array of colors corresponding to points
        """
        if len(self.points) == 0:
            self.points = points
            if colors is not None:
                self.colors = colors
            else:
                # Default color (gray)
                self.colors = np.ones((len(points), 3)) * 0.5
        else:
            self.points = np.vstack((self.points, points))
            if colors is not None:
                self.colors = np.vstack((self.colors, colors))
            else:
                # Default color (gray)
                new_colors = np.ones((len(points), 3)) * 0.5
                self.colors = np.vstack((self.colors, new_colors))
    
    def classify_points(self, distance_threshold: float = 0.1, 
                      boundary_thickness: float = 0.05):
        """
        Classify points as object, boundary, or void.
        
        Args:
            distance_threshold: Distance threshold for classification
            boundary_thickness: Thickness of boundary regions
        """
        if len(self.points) == 0:
            logger.warning("No points to classify")
            return
        
        # Create a more efficient spatial data structure
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        
        # Estimate normals if not already estimated
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        
        # Use DBSCAN to identify clusters (objects)
        clustering = DBSCAN(eps=distance_threshold, min_samples=10).fit(self.points)
        labels = clustering.labels_
        
        # Initialize masks
        self.object_mask = np.zeros(len(self.points), dtype=bool)
        self.boundary_mask = np.zeros(len(self.points), dtype=bool)
        self.void_mask = np.zeros(len(self.points), dtype=bool)
        
        # Points in clusters are object points
        self.object_mask[labels >= 0] = True
        
        # Find boundary points
        # These are points that are close to object points but not part of an object
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        
        for i, point in enumerate(self.points):
            if self.object_mask[i]:
                continue
                
            # Find nearest neighbors
            [k, idx, _] = kdtree.search_radius_vector_3d(point, distance_threshold)
            
            # Check if any neighbors are object points
            nearby_objects = np.any(self.object_mask[idx])
            
            if nearby_objects:
                self.boundary_mask[i] = True
            else:
                self.void_mask[i] = True
        
        # Extract point subsets
        self.object_points = self.points[self.object_mask]
        self.boundary_points = self.points[self.boundary_mask]
        self.void_points = self.points[self.void_mask]
        
        # Update metadata
        self.metadata["num_object_points"] = len(self.object_points)
        self.metadata["num_boundary_points"] = len(self.boundary_points)
        self.metadata["num_void_points"] = len(self.void_points)
        
        # Create color coding for visualization
        # Object: blue, Boundary: green, Void: red
        colors = np.zeros_like(self.colors)
        colors[self.object_mask] = [0, 0, 1]  # Blue
        colors[self.boundary_mask] = [0, 1, 0]  # Green
        colors[self.void_mask] = [1, 0, 0]  # Red
        
        # Store original colors
        self.original_colors = self.colors.copy()
        # Update colors to show classification
        self.colors = colors
    
    def generate_void_mesh(self):
        """
        Generate a mesh representation of the void spaces.
        
        This creates a 3D mesh model of the negative space regions,
        providing a tangible visualization of the "invisible" volume.
        """
        if len(self.void_points) == 0:
            logger.warning("No void points available for mesh generation")
            return None
        
        try:
            # Create point cloud from void points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.void_points)
            
            # Estimate normals
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
            
            # Create mesh using Poisson surface reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8)
            
            # Filter low-density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            return mesh
        except Exception as e:
            logger.error(f"Error generating void mesh: {str(e)}")
            return None
    
    def compute_interstitial_spaces(self):
        """
        Identify and analyze the interstitial spaces between objects.
        
        This method finds the void regions between multiple objects and
        characterizes their spatial relationships.
        """
        if len(self.object_points) == 0 or len(self.void_points) == 0:
            logger.warning("Insufficient points for interstitial space analysis")
            return
        
        try:
            # Cluster object points to identify distinct objects
            clustering = DBSCAN(eps=0.2, min_samples=20).fit(self.object_points)
            object_labels = clustering.labels_
            
            # Filter out noise
            valid_objects = object_labels >= 0
            object_clusters = {}
            
            for i, label in enumerate(object_labels):
                if label >= 0:
                    if label not in object_clusters:
                        object_clusters[label] = []
                    object_clusters[label].append(self.object_points[i])
            
            # Convert lists to arrays
            for label in object_clusters:
                object_clusters[label] = np.array(object_clusters[label])
            
            # Need at least 2 objects for interstitial space
            if len(object_clusters) < 2:
                logger.warning("Fewer than 2 distinct objects found")
                return
            
            # Compute centroids of each object
            centroids = {}
            for label, points in object_clusters.items():
                centroids[label] = np.mean(points, axis=0)
            
            # Create a graph of connections between objects
            connections = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    # Calculate distance between centroids
                    distance = np.linalg.norm(centroids[i] - centroids[j])
                    connections.append((i, j, distance))
            
            # Find void points in interstitial spaces
            interstitial_regions = {}
            
            for i, j, distance in connections:
                # Calculate midpoint between centroids
                midpoint = (centroids[i] + centroids[j]) / 2
                
                # Find void points near the midpoint
                nearby_void_indices = []
                for k, void_point in enumerate(self.void_points):
                    # Calculate distances to the line connecting centroids
                    v = centroids[j] - centroids[i]  # Vector from i to j
                    v_norm = v / np.linalg.norm(v)  # Normalized
                    
                    # Vector from centroid i to the void point
                    w = void_point - centroids[i]
                    
                    # Calculate projection of w onto v
                    proj_length = np.dot(w, v_norm)
                    
                    # Only consider points between the centroids
                    if 0 <= proj_length <= distance:
                        # Calculate perpendicular distance to the line
                        proj_point = centroids[i] + proj_length * v_norm
                        perp_distance = np.linalg.norm(void_point - proj_point)
                        
                        # If within threshold, consider it part of the interstitial space
                        if perp_distance < distance * 0.3:  # Adjustable threshold
                            nearby_void_indices.append(k)
                
                # Store interstitial region
                if nearby_void_indices:
                    region_points = self.void_points[nearby_void_indices]
                    interstitial_regions[(i, j)] = region_points
            
            # Store results
            self.interstitial_regions = interstitial_regions
            self.object_centroids = centroids
            self.object_connections = connections
            
            # Update metadata
            self.metadata["num_objects"] = len(centroids)
            self.metadata["num_interstitial_regions"] = len(interstitial_regions)
            
            return interstitial_regions
            
        except Exception as e:
            logger.error(f"Error computing interstitial spaces: {str(e)}")
            return None
    
    def compute_spatial_signature(self):
        """
        Compute a unique spatial signature based on the negative space configuration.
        
        This signature represents the unique properties of the void regions and
        their relationships to objects, creating a "fingerprint" of the scene.
        """
        if len(self.void_points) == 0:
            logger.warning("No void points available for signature generation")
            return None
        
        try:
            # Create feature vector from multiple spatial characteristics
            features = []
            
            # 1. Void volume (approximated by point count)
            void_ratio = len(self.void_points) / len(self.points) if len(self.points) > 0 else 0
            features.append(void_ratio)
            
            # 2. Void distribution statistics
            if len(self.void_points) > 0:
                # Calculate bounding box
                min_coords = np.min(self.void_points, axis=0)
                max_coords = np.max(self.void_points, axis=0)
                bbox_size = max_coords - min_coords
                
                # Volume of bounding box
                bbox_volume = np.prod(bbox_size)
                
                # Void point density
                void_density = len(self.void_points) / bbox_volume if bbox_volume > 0 else 0
                features.append(void_density)
                
                # Spatial distribution statistics
                void_centroid = np.mean(self.void_points, axis=0)
                distances_from_centroid = np.linalg.norm(self.void_points - void_centroid, axis=1)
                
                mean_distance = np.mean(distances_from_centroid)
                std_distance = np.std(distances_from_centroid)
                skewness = np.mean((distances_from_centroid - mean_distance)**3) / (std_distance**3) if std_distance > 0 else 0
                
                features.extend([mean_distance, std_distance, skewness])
            
            # 3. Boundary characteristics
            if len(self.boundary_points) > 0:
                boundary_ratio = len(self.boundary_points) / len(self.points) if len(self.points) > 0 else 0
                features.append(boundary_ratio)
                
                # Surface area approximation
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.boundary_points)
                
                # Estimate normals
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30))
                
                # Convert to numpy array
                normals = np.asarray(pcd.normals)
                
                # Calculate normal consistency (how aligned the normals are)
                normal_dots = np.abs(np.dot(normals, normals.T))
                normal_consistency = np.mean(normal_dots)
                features.append(normal_consistency)
            
            # 4. Interstitial space characteristics
            if hasattr(self, 'interstitial_regions') and self.interstitial_regions:
                num_regions = len(self.interstitial_regions)
                features.append(num_regions)
                
                # Average size of interstitial regions
                region_sizes = [len(points) for points in self.interstitial_regions.values()]
                avg_region_size = np.mean(region_sizes) if region_sizes else 0
                features.append(avg_region_size)
                
                # Connectivity measure
                if hasattr(self, 'object_connections'):
                    avg_connection_distance = np.mean([d for _, _, d in self.object_connections])
                    features.append(avg_connection_distance)
            
            # Normalize features
            features = np.array(features)
            
            # Store signature
            self.spatial_signature = features
            
            # Update metadata
            self.metadata["spatial_signature"] = features.tolist()
            self.metadata["signature_timestamp"] = time.time()
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing spatial signature: {str(e)}")
            return None
    
    def visualize(self, show_classification: bool = True, 
                 show_interstitial: bool = False):
        """
        Visualize the point cloud with optional features.
        
        Args:
            show_classification: Whether to color-code points by classification
            show_interstitial: Whether to highlight interstitial regions
        """
        if len(self.points) == 0:
            logger.warning("No points to visualize")
            return
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        
        # Set colors based on classification if requested
        if show_classification and hasattr(self, 'original_colors'):
            pcd.colors = o3d.utility.Vector3dVector(self.colors)
        else:
            # Use original or default colors
            if hasattr(self, 'original_colors'):
                pcd.colors = o3d.utility.Vector3dVector(self.original_colors)
            else:
                # Default to gray
                pcd.colors = o3d.utility.Vector3dVector(
                    np.ones((len(self.points), 3)) * 0.5)
        
        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        
        # Add interstitial spaces if requested
        if show_interstitial and hasattr(self, 'interstitial_regions'):
            # Add lines connecting object centroids
            if hasattr(self, 'object_centroids'):
                for i, j, _ in self.object_connections:
                    points = np.array([self.object_centroids[i], self.object_centroids[j]])
                    
                    line = o3d.geometry.LineSet()
                    line.points = o3d.utility.Vector3dVector(points)
                    line.lines = o3d.utility.Vector2iVector([[0, 1]])
                    line.colors = o3d.utility.Vector3dVector([[1, 1, 0]])  # Yellow
                    
                    vis.add_geometry(line)
            
            # Add interstitial regions with distinct colors
            colors = [
                [1, 0, 1],    # Magenta
                [0, 1, 1],    # Cyan
                [1, 0.5, 0],  # Orange
                [0.5, 0, 1],  # Purple
                [0, 0.5, 0.5] # Teal
            ]
            
            for i, ((obj1, obj2), points) in enumerate(self.interstitial_regions.items()):
                interstitial_pcd = o3d.geometry.PointCloud()
                interstitial_pcd.points = o3d.utility.Vector3dVector(points)
                
                # Cycle through colors
                color = colors[i % len(colors)]
                interstitial_pcd.colors = o3d.utility.Vector3dVector(
                    np.tile(color, (len(points), 1)))
                
                vis.add_geometry(interstitial_pcd)
        
        # Render
        vis.run()
        vis.destroy_window()
    
    def save(self, filepath: str):
        """
        Save the point cloud to a file.
        
        Args:
            filepath: Path to save the file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Determine file extension
        _, ext = os.path.splitext(filepath)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        
        # Save to file
        if ext.lower() == '.pcd':
            o3d.io.write_point_cloud(filepath, pcd)
        elif ext.lower() == '.ply':
            o3d.io.write_point_cloud(filepath, pcd)
        else:
            # Default to PLY
            filepath = filepath + '.ply' if not ext else filepath
            o3d.io.write_point_cloud(filepath, pcd)
        
        logger.info(f"Point cloud saved to {filepath}")
        
        # Save metadata
        metadata_path = os.path.splitext(filepath)[0] + '_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    @staticmethod
    def load(filepath: str):
        """
        Load a point cloud from a file.
        
        Args:
            filepath: Path to the point cloud file
            
        Returns:
            NegativeSpacePointCloud object
        """
        # Check if file exists
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        try:
            # Load with Open3D
            pcd = o3d.io.read_point_cloud(filepath)
            
            # Convert to numpy arrays
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # Create NegativeSpacePointCloud
            negative_pcd = NegativeSpacePointCloud(points, colors)
            
            # Try to load metadata
            metadata_path = os.path.splitext(filepath)[0] + '_metadata.json'
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    negative_pcd.metadata = json.load(f)
            
            return negative_pcd
            
        except Exception as e:
            logger.error(f"Error loading point cloud: {str(e)}")
            return None

class PointCloudGenerator:
    """
    Specialized point cloud generator with focus on negative space.
    
    This class implements various methods for generating point clouds,
    with specific optimizations for capturing and characterizing the
    negative space between objects.
    """
    
    def __init__(self, cloud_type: Union[str, PointCloudType] = PointCloudType.STANDARD,
                 params: Optional[PointCloudParams] = None):
        """
        Initialize the point cloud generator.
        
        Args:
            cloud_type: Type of point cloud generation method
            params: Parameters for point cloud generation
        """
        if isinstance(cloud_type, str):
            try:
                self.cloud_type = PointCloudType(cloud_type)
            except ValueError:
                logger.warning(f"Invalid cloud type '{cloud_type}'. Using STANDARD instead.")
                self.cloud_type = PointCloudType.STANDARD
        else:
            self.cloud_type = cloud_type
        
        self.params = params or PointCloudParams()
        
        # Storage for camera parameters
        self.camera_matrices = []
        self.projection_matrices = []
        self.camera_positions = []
        
        # Storage for image data
        self.images = []
        self.depth_maps = []
        self.masks = []
        
        # Storage for feature data
        self.keypoints_list = []
        self.descriptors_list = []
        
        # Storage for point cloud data
        self.points_3d = []
        self.colors_3d = []
        
        # Metadata
        self.metadata = {}
    
    def add_camera(self, camera_matrix: np.ndarray, 
                  projection_matrix: np.ndarray,
                  position: np.ndarray):
        """
        Add camera parameters.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            projection_matrix: 3x4 or 4x4 projection matrix
            position: 3D position of the camera
        """
        self.camera_matrices.append(camera_matrix)
        self.projection_matrices.append(projection_matrix)
        self.camera_positions.append(position)
    
    def add_image_data(self, image: np.ndarray,
                      depth_map: Optional[np.ndarray] = None,
                      mask: Optional[np.ndarray] = None):
        """
        Add image data for point cloud generation.
        
        Args:
            image: RGB image
            depth_map: Optional depth map
            mask: Optional mask
        """
        self.images.append(image)
        self.depth_maps.append(depth_map)
        self.masks.append(mask if mask is not None else np.ones(image.shape[:2], dtype=bool))
    
    def add_features(self, keypoints: List, descriptors: np.ndarray):
        """
        Add feature data for point cloud generation.
        
        Args:
            keypoints: List of keypoints
            descriptors: Feature descriptors
        """
        self.keypoints_list.append(keypoints)
        self.descriptors_list.append(descriptors)
    
    def triangulate_points(self, keypoint_matches: List[List[Tuple[int, int]]]) -> np.ndarray:
        """
        Triangulate 3D points from matched keypoints.
        
        Args:
            keypoint_matches: List of keypoint matches between image pairs
                Each match is a list of (image_idx, keypoint_idx) tuples
            
        Returns:
            Array of 3D points
        """
        if not self.camera_matrices or not self.projection_matrices:
            logger.error("No camera parameters available for triangulation")
            return np.array([])
        
        points_3d = []
        
        for match in keypoint_matches:
            if len(match) < 2:
                continue
            
            # Collect 2D points and projection matrices
            image_points = []
            proj_matrices = []
            
            for img_idx, kp_idx in match:
                if img_idx >= len(self.keypoints_list) or kp_idx >= len(self.keypoints_list[img_idx]):
                    continue
                
                # Get keypoint coordinates
                kp = self.keypoints_list[img_idx][kp_idx]
                image_points.append(kp.pt)
                
                # Get projection matrix
                proj_matrices.append(self.projection_matrices[img_idx])
            
            if len(image_points) < 2 or len(proj_matrices) < 2:
                continue
            
            # Convert to numpy arrays
            image_points = np.array(image_points, dtype=np.float64)
            proj_matrices = np.array(proj_matrices, dtype=np.float64)
            
            # Triangulate the point
            point_3d = cv2.triangulatePoints(
                proj_matrices[0], proj_matrices[1],
                image_points[0], image_points[1]
            )
            
            # Convert from homogeneous coordinates
            point_3d = point_3d[:3] / point_3d[3]
            
            points_3d.append(point_3d.ravel())
        
        return np.array(points_3d)
    
    def generate_point_cloud(self) -> NegativeSpacePointCloud:
        """
        Generate a point cloud using the current data and parameters.
        
        Returns:
            NegativeSpacePointCloud object
        """
        start_time = time.time()
        
        # Create an empty point cloud
        point_cloud = NegativeSpacePointCloud()
        
        # Generate based on point cloud type
        if self.cloud_type == PointCloudType.STANDARD:
            self._generate_standard_point_cloud(point_cloud)
        elif self.cloud_type == PointCloudType.NEGATIVE_SPACE_OPTIMIZED:
            self._generate_negative_space_optimized(point_cloud)
        elif self.cloud_type == PointCloudType.VOID_FOCUSED:
            self._generate_void_focused(point_cloud)
        elif self.cloud_type == PointCloudType.BOUNDARY_ENHANCED:
            self._generate_boundary_enhanced(point_cloud)
        elif self.cloud_type == PointCloudType.INTERSTITIAL:
            self._generate_interstitial(point_cloud)
        elif self.cloud_type == PointCloudType.VOLUMETRIC:
            self._generate_volumetric(point_cloud)
        else:
            logger.warning(f"Unknown point cloud type: {self.cloud_type}. Using STANDARD instead.")
            self._generate_standard_point_cloud(point_cloud)
        
        # Update metadata
        processing_time = time.time() - start_time
        point_cloud.metadata["generation_time"] = processing_time
        point_cloud.metadata["point_cloud_type"] = self.cloud_type.value
        point_cloud.metadata["num_points"] = len(point_cloud.points)
        
        return point_cloud
    
    def _generate_standard_point_cloud(self, point_cloud: NegativeSpacePointCloud):
        """
        Generate a standard point cloud from the available data.
        
        Args:
            point_cloud: NegativeSpacePointCloud to populate
        """
        # Check if we have enough data
        if len(self.images) < 2 or len(self.camera_matrices) < 2:
            logger.warning("Insufficient data for point cloud generation")
            
            # Generate a sample point cloud for demo purposes
            self._generate_sample_point_cloud(point_cloud)
            return
        
        # TODO: Implement actual point cloud generation from images
        # This would involve feature matching, triangulation, etc.
        
        # For now, generate a sample point cloud
        self._generate_sample_point_cloud(point_cloud)
    
    def _generate_negative_space_optimized(self, point_cloud: NegativeSpacePointCloud):
        """
        Generate a point cloud optimized for negative space analysis.
        
        Args:
            point_cloud: NegativeSpacePointCloud to populate
        """
        # Start with standard point cloud
        self._generate_standard_point_cloud(point_cloud)
        
        # Skip if no points were generated
        if len(point_cloud.points) == 0:
            return
        
        # Classify points to identify void regions
        point_cloud.classify_points()
        
        # Generate additional points in void regions
        if len(point_cloud.void_points) > 0:
            # Estimate the size of the void regions
            if len(point_cloud.void_points) > 3:
                # Compute convex hull of void points
                hull = ConvexHull(point_cloud.void_points)
                
                # Generate points inside the hull
                num_void_points = min(
                    int(len(point_cloud.points) * self.params.void_sampling_ratio),
                    self.params.max_points - len(point_cloud.points)
                )
                
                void_points = self._generate_points_in_hull(
                    hull, num_void_points)
                
                # Add generated void points
                void_colors = np.zeros((len(void_points), 3))
                void_colors[:, 0] = 1.0  # Red for void points
                
                point_cloud.add_points(void_points, void_colors)
                
                # Re-classify points
                point_cloud.classify_points()
    
    def _generate_void_focused(self, point_cloud: NegativeSpacePointCloud):
        """
        Generate a point cloud specifically focused on void regions.
        
        Args:
            point_cloud: NegativeSpacePointCloud to populate
        """
        # Start with negative space optimized point cloud
        self._generate_negative_space_optimized(point_cloud)
        
        # Skip if no points were generated
        if len(point_cloud.points) == 0:
            return
        
        # If we have depth maps, use them to identify void regions
        if any(depth is not None for depth in self.depth_maps):
            # Use depth maps to identify void regions
            void_points = []
            
            for i, depth_map in enumerate(self.depth_maps):
                if depth_map is None:
                    continue
                
                # Skip if we don't have camera parameters
                if i >= len(self.camera_matrices) or i >= len(self.projection_matrices):
                    continue
                
                # Normalize depth map
                if depth_map.dtype != np.uint8:
                    normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    normalized_depth = depth_map
                
                # Threshold to find far regions (likely void)
                _, binary = cv2.threshold(normalized_depth, 200, 255, cv2.THRESH_BINARY)
                
                # Find contours of void regions
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Skip if no contours found
                if not contours:
                    continue
                
                # Generate 3D points from depth map within void regions
                camera_matrix = self.camera_matrices[i]
                
                for contour in contours:
                    # Create mask for this contour
                    mask = np.zeros_like(binary)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    
                    # Sample points within the contour
                    y_indices, x_indices = np.where(mask > 0)
                    
                    # Skip if no points found
                    if len(x_indices) == 0:
                        continue
                    
                    # Sample a subset of points
                    num_samples = min(1000, len(x_indices))
                    sample_indices = np.random.choice(len(x_indices), num_samples, replace=False)
                    
                    x_samples = x_indices[sample_indices]
                    y_samples = y_indices[sample_indices]
                    
                    for x, y in zip(x_samples, y_samples):
                        # Get depth value
                        z = depth_map[y, x]
                        
                        # Skip if invalid depth
                        if z <= 0:
                            continue
                        
                        # Convert to 3D point
                        x3d = (x - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
                        y3d = (y - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
                        z3d = z
                        
                        void_points.append([x3d, y3d, z3d])
            
            if void_points:
                void_points = np.array(void_points)
                
                # Add void points to point cloud
                void_colors = np.zeros((len(void_points), 3))
                void_colors[:, 0] = 1.0  # Red for void points
                
                point_cloud.add_points(void_points, void_colors)
                
                # Re-classify points
                point_cloud.classify_points()
    
    def _generate_boundary_enhanced(self, point_cloud: NegativeSpacePointCloud):
        """
        Generate a point cloud with enhanced sampling at object boundaries.
        
        Args:
            point_cloud: NegativeSpacePointCloud to populate
        """
        # Start with standard point cloud
        self._generate_standard_point_cloud(point_cloud)
        
        # Skip if no points were generated
        if len(point_cloud.points) == 0:
            return
        
        # Classify points to identify boundary regions
        point_cloud.classify_points()
        
        # If we have boundary points, enhance sampling around them
        if len(point_cloud.boundary_points) > 0:
            # Number of additional boundary points to generate
            num_boundary_points = min(
                int(len(point_cloud.boundary_points) * self.params.boundary_emphasis),
                self.params.max_points - len(point_cloud.points)
            )
            
            # Generate points around boundary points
            boundary_enhancements = []
            
            for point in point_cloud.boundary_points:
                # Add small random offsets to create variations
                num_variations = max(1, num_boundary_points // len(point_cloud.boundary_points))
                
                for _ in range(num_variations):
                    # Random offset with decreasing magnitude away from boundary
                    offset = (np.random.random(3) - 0.5) * 0.05
                    boundary_enhancements.append(point + offset)
            
            if boundary_enhancements:
                boundary_enhancements = np.array(boundary_enhancements)
                
                # Add enhanced boundary points
                boundary_colors = np.zeros((len(boundary_enhancements), 3))
                boundary_colors[:, 1] = 1.0  # Green for boundary points
                
                point_cloud.add_points(boundary_enhancements, boundary_colors)
                
                # Re-classify points
                point_cloud.classify_points()
    
    def _generate_interstitial(self, point_cloud: NegativeSpacePointCloud):
        """
        Generate a point cloud focused on interstitial spaces between objects.
        
        Args:
            point_cloud: NegativeSpacePointCloud to populate
        """
        # Start with negative space optimized point cloud
        self._generate_negative_space_optimized(point_cloud)
        
        # Skip if no points were generated
        if len(point_cloud.points) == 0:
            return
        
        # Compute interstitial spaces
        point_cloud.compute_interstitial_spaces()
        
        # If interstitial regions were identified, enhance them
        if hasattr(point_cloud, 'interstitial_regions') and point_cloud.interstitial_regions:
            # Generate additional points in interstitial regions
            interstitial_enhancements = []
            
            for region_points in point_cloud.interstitial_regions.values():
                if len(region_points) < 3:
                    continue
                
                # Create a convex hull of the region
                try:
                    hull = ConvexHull(region_points)
                    
                    # Number of points to generate in this region
                    num_points = min(500, self.params.max_points // len(point_cloud.interstitial_regions))
                    
                    # Generate points within the hull
                    new_points = self._generate_points_in_hull(hull, num_points)
                    interstitial_enhancements.extend(new_points)
                except Exception as e:
                    logger.warning(f"Error enhancing interstitial region: {str(e)}")
            
            if interstitial_enhancements:
                interstitial_enhancements = np.array(interstitial_enhancements)
                
                # Add interstitial enhancement points
                interstitial_colors = np.zeros((len(interstitial_enhancements), 3))
                interstitial_colors[:, 0] = 1.0  # Red for interstitial points
                interstitial_colors[:, 2] = 1.0  # Add blue to make purple
                
                point_cloud.add_points(interstitial_enhancements, interstitial_colors)
                
                # Re-compute interstitial spaces with enhanced points
                point_cloud.compute_interstitial_spaces()
    
    def _generate_volumetric(self, point_cloud: NegativeSpacePointCloud):
        """
        Generate a volumetric representation of negative space.
        
        Args:
            point_cloud: NegativeSpacePointCloud to populate
        """
        # Start with negative space optimized point cloud
        self._generate_negative_space_optimized(point_cloud)
        
        # Skip if no points were generated
        if len(point_cloud.points) == 0:
            return
        
        # Create a voxel grid
        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud.points)
            
            # Create voxel grid
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                pcd, voxel_size=self.params.voxel_size)
            
            # Extract voxel coordinates and colors
            voxels = []
            voxel_colors = []
            
            for i, voxel in enumerate(voxel_grid.get_voxels()):
                voxels.append(voxel.grid_index)
                
                # Color based on point classification
                if point_cloud.void_mask is not None and len(point_cloud.void_mask) > 0:
                    if i < len(point_cloud.void_mask) and point_cloud.void_mask[i]:
                        voxel_colors.append([1.0, 0, 0])  # Red for void
                    elif i < len(point_cloud.boundary_mask) and point_cloud.boundary_mask[i]:
                        voxel_colors.append([0, 1.0, 0])  # Green for boundary
                    elif i < len(point_cloud.object_mask) and point_cloud.object_mask[i]:
                        voxel_colors.append([0, 0, 1.0])  # Blue for object
                    else:
                        voxel_colors.append([0.5, 0.5, 0.5])  # Gray for unknown
                else:
                    voxel_colors.append([0.5, 0.5, 0.5])  # Gray
            
            # Store voxel grid
            point_cloud.metadata["voxel_size"] = self.params.voxel_size
            point_cloud.void_voxels = voxel_grid
            
        except Exception as e:
            logger.warning(f"Error creating volumetric representation: {str(e)}")
    
    def _generate_sample_point_cloud(self, point_cloud: NegativeSpacePointCloud):
        """
        Generate a sample point cloud for demo purposes.
        
        Args:
            point_cloud: NegativeSpacePointCloud to populate
        """
        # Create two objects with void space between them
        num_points = 5000
        
        # Object 1: Sphere on the left
        sphere_points = []
        sphere_colors = []
        
        for _ in range(num_points // 2):
            # Random point on a sphere
            theta = np.random.random() * 2 * np.pi
            phi = np.random.random() * np.pi
            r = 0.5 + np.random.random() * 0.1  # Slight variation in radius
            
            x = r * np.sin(phi) * np.cos(theta) - 1.0  # Shift left
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            sphere_points.append([x, y, z])
            sphere_colors.append([0, 0, 1])  # Blue
        
        # Object 2: Cube on the right
        cube_points = []
        cube_colors = []
        
        for _ in range(num_points // 2):
            # Random point near a cube surface
            face = np.random.randint(0, 6)
            
            if face == 0:  # Front
                x = 1.0 + np.random.random() * 0.1  # Shift right
                y = np.random.random() - 0.5
                z = np.random.random() - 0.5
            elif face == 1:  # Back
                x = 1.5 + np.random.random() * 0.1
                y = np.random.random() - 0.5
                z = np.random.random() - 0.5
            elif face == 2:  # Left
                x = 1.0 + np.random.random() * 0.5
                y = -0.5 - np.random.random() * 0.1
                z = np.random.random() - 0.5
            elif face == 3:  # Right
                x = 1.0 + np.random.random() * 0.5
                y = 0.5 + np.random.random() * 0.1
                z = np.random.random() - 0.5
            elif face == 4:  # Bottom
                x = 1.0 + np.random.random() * 0.5
                y = np.random.random() - 0.5
                z = -0.5 - np.random.random() * 0.1
            else:  # Top
                x = 1.0 + np.random.random() * 0.5
                y = np.random.random() - 0.5
                z = 0.5 + np.random.random() * 0.1
            
            cube_points.append([x, y, z])
            cube_colors.append([0, 1, 0])  # Green
        
        # Combine points
        all_points = np.array(sphere_points + cube_points)
        all_colors = np.array(sphere_colors + cube_colors)
        
        # Add points to the point cloud
        point_cloud.add_points(all_points, all_colors)
        
        # Classify points
        point_cloud.classify_points()
    
    def _generate_points_in_hull(self, hull: ConvexHull, num_points: int) -> np.ndarray:
        """
        Generate random points inside a convex hull.
        
        Args:
            hull: Convex hull
            num_points: Number of points to generate
            
        Returns:
            Array of points inside the hull
        """
        # Get dimensions
        ndim = hull.points.shape[1]
        
        # Find a point inside the hull (centroid)
        centroid = np.mean(hull.points, axis=0)
        
        # Generate points
        points = []
        attempts = 0
        max_attempts = num_points * 10  # Limit attempts to avoid infinite loop
        
        while len(points) < num_points and attempts < max_attempts:
            attempts += 1
            
            # Generate random point in the bounding box
            min_coords = np.min(hull.points, axis=0)
            max_coords = np.max(hull.points, axis=0)
            
            # Random point with potential noise
            point = min_coords + np.random.random(ndim) * (max_coords - min_coords)
            
            # Check if the point is inside the hull
            equations = hull.equations
            inside = True
            
            for eq in equations:
                # Last element is the constant term
                dist = np.dot(eq[:-1], point) + eq[-1]
                if dist > 1e-6:  # Allow for some numerical error
                    inside = False
                    break
            
            if inside:
                points.append(point)
        
        return np.array(points)

```