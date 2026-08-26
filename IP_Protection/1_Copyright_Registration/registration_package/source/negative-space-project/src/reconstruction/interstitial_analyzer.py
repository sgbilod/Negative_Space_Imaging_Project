"""
Interstitial Space Analyzer

This module provides functionality for analyzing interstitial spaces between objects
in a 3D scene, with a specific focus on negative space analysis.

Interstitial spaces are the regions between objects that form important spatial
relationships and can be used to characterize scenes in a unique way that focuses
on what isn't there rather than what is.

Classes:
    InterstitialRegion: Represents a single interstitial region between objects
    InterstitialAnalyzer: Analyzes and characterizes interstitial spaces
    
Functions:
    compute_voronoi_regions: Computes Voronoi regions to identify interstitial spaces
    identify_region_boundaries: Identifies boundaries between interstitial regions
"""

import numpy as np
import open3d as o3d
from scipy.spatial import Voronoi, Delaunay
from sklearn.cluster import DBSCAN, KMeans
from typing import List, Dict, Tuple, Optional, Union, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InterstitialRegion:
    """
    Represents a single interstitial region between objects.
    
    An interstitial region is a void space between objects that has specific
    characteristics and relationships to the surrounding objects.
    
    Attributes:
        id (int): Unique identifier for the region
        center (np.ndarray): 3D coordinates of the region center
        points (np.ndarray): Points belonging to this region
        volume (float): Estimated volume of the region
        surface_area (float): Estimated surface area of the region
        adjacent_objects (List[int]): IDs of objects adjacent to this region
        boundary_points (np.ndarray): Points on the boundary of this region
        signature (np.ndarray): Spatial signature of this region
    """
    
    def __init__(self, id: int, points: np.ndarray):
        """
        Initialize an interstitial region.
        
        Args:
            id: Unique identifier for the region
            points: Points belonging to this region (Nx3 numpy array)
        """
        self.id = id
        self.points = points
        self.center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)
        self.volume = 0.0
        self.surface_area = 0.0
        self.adjacent_objects: List[int] = []
        self.boundary_points = np.array([])
        self.signature = np.array([])
        
        # Compute basic properties
        self._compute_basic_properties()
    
    def _compute_basic_properties(self):
        """Compute basic properties of the region"""
        if len(self.points) < 4:
            return
            
        try:
            # Create Open3D point cloud for computations
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            
            # Compute approximate volume using convex hull
            hull, _ = pcd.compute_convex_hull()
            self.volume = hull.get_volume()
            
            # Compute surface area
            self.surface_area = hull.get_surface_area()
            
            # Extract boundary points
            hull_vertices = np.asarray(hull.vertices)
            self.boundary_points = self.points[hull_vertices]
            
        except Exception as e:
            logger.warning(f"Failed to compute properties for region {self.id}: {str(e)}")
    
    def compute_signature(self, num_features: int = 32) -> np.ndarray:
        """
        Compute a spatial signature for this interstitial region.
        
        The signature is a fixed-length feature vector that characterizes
        the shape and properties of the interstitial space.
        
        Args:
            num_features: Number of features in the signature vector
            
        Returns:
            Numpy array containing the signature vector
        """
        if len(self.points) < 10:
            self.signature = np.zeros(num_features)
            return self.signature
            
        try:
            # Initialize feature vector
            features = []
            
            # Basic geometric features
            features.append(self.volume)
            features.append(self.surface_area)
            features.append(len(self.points))
            features.append(len(self.adjacent_objects))
            
            # Shape features - eigenvalues of covariance matrix
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            covariance = np.cov(self.points, rowvar=False)
            eigenvalues, _ = np.linalg.eigh(covariance)
            eigenvalues = sorted(eigenvalues, reverse=True)
            
            # Ensure we have 3 eigenvalues even if we get fewer
            eigenvalues = list(eigenvalues) + [0] * (3 - len(eigenvalues))
            
            # Add eigenvalues to features
            features.extend(eigenvalues)
            
            # Shape factors
            if eigenvalues[0] > 0:
                planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
                sphericity = eigenvalues[2] / eigenvalues[0]
                linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
            else:
                planarity, sphericity, linearity = 0, 0, 0
                
            features.extend([planarity, sphericity, linearity])
            
            # Spatial distribution features
            distances = np.linalg.norm(self.points - self.center, axis=1)
            features.append(np.mean(distances))  # Mean distance from center
            features.append(np.std(distances))   # Std of distances from center
            features.append(np.max(distances))   # Max distance from center
            
            # Distance ratios
            if len(self.points) > 1:
                pairwise_distances = []
                sample_size = min(100, len(self.points))
                sample_indices = np.random.choice(len(self.points), sample_size, replace=False)
                sample_points = self.points[sample_indices]
                
                for i in range(sample_size):
                    for j in range(i+1, sample_size):
                        dist = np.linalg.norm(sample_points[i] - sample_points[j])
                        pairwise_distances.append(dist)
                        
                if pairwise_distances:
                    features.append(np.mean(pairwise_distances))
                    features.append(np.std(pairwise_distances))
                    features.append(np.max(pairwise_distances))
                    features.append(np.min(pairwise_distances))
                else:
                    features.extend([0, 0, 0, 0])
            else:
                features.extend([0, 0, 0, 0])
            
            # Ensure we have exactly num_features
            if len(features) > num_features:
                features = features[:num_features]
            elif len(features) < num_features:
                features.extend([0] * (num_features - len(features)))
            
            # Normalize features to unit length
            features_array = np.array(features)
            norm = np.linalg.norm(features_array)
            if norm > 0:
                features_array = features_array / norm
                
            self.signature = features_array
            return self.signature
            
        except Exception as e:
            logger.warning(f"Failed to compute signature for region {self.id}: {str(e)}")
            self.signature = np.zeros(num_features)
            return self.signature
    
    def compute_adjacency(self, object_points: List[np.ndarray], threshold: float = 0.1) -> List[int]:
        """
        Compute which objects are adjacent to this interstitial region.
        
        Args:
            object_points: List of points for each object
            threshold: Distance threshold for considering an object adjacent
            
        Returns:
            List of object IDs that are adjacent to this region
        """
        if len(self.points) == 0:
            return []
            
        self.adjacent_objects = []
        
        # Use boundary points for efficiency
        boundary_points = self.boundary_points if len(self.boundary_points) > 0 else self.points
        
        for obj_id, points in enumerate(object_points):
            if len(points) == 0:
                continue
                
            # Compute minimum distance between boundary points and object points
            min_dist = float('inf')
            
            # Sample boundary points and object points if there are too many
            max_samples = 100
            boundary_sample = boundary_points
            object_sample = points
            
            if len(boundary_sample) > max_samples:
                indices = np.random.choice(len(boundary_sample), max_samples, replace=False)
                boundary_sample = boundary_sample[indices]
                
            if len(object_sample) > max_samples:
                indices = np.random.choice(len(object_sample), max_samples, replace=False)
                object_sample = object_sample[indices]
            
            # Compute pairwise distances
            for bp in boundary_sample:
                for op in object_sample:
                    dist = np.linalg.norm(bp - op)
                    if dist < min_dist:
                        min_dist = dist
                        
                    # Early exit if distance is below threshold
                    if dist < threshold:
                        self.adjacent_objects.append(obj_id)
                        break
                        
                if obj_id in self.adjacent_objects:
                    break
            
            # If distance is below threshold, consider the object adjacent
            if min_dist < threshold and obj_id not in self.adjacent_objects:
                self.adjacent_objects.append(obj_id)
                
        return self.adjacent_objects
        
    def visualize(self) -> o3d.geometry.TriangleMesh:
        """
        Generate a visualization of this interstitial region.
        
        Returns:
            Open3D mesh representing the region
        """
        if len(self.points) < 4:
            return None
            
        # Create point cloud from points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        
        try:
            # Create alpha shape (more accurate than convex hull for complex shapes)
            alpha = 0.5  # Alpha value controls the level of detail
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            
            # Compute normals for proper rendering
            mesh.compute_vertex_normals()
            
            return mesh
        except Exception as e:
            logger.warning(f"Failed to create visualization for region {self.id}: {str(e)}")
            
            # Fall back to convex hull if alpha shape fails
            try:
                mesh, _ = pcd.compute_convex_hull()
                mesh.compute_vertex_normals()
                return mesh
            except:
                return None


class InterstitialAnalyzer:
    """
    Analyzes and characterizes interstitial spaces in a point cloud.
    
    This class provides methods to identify, analyze, and characterize
    the empty spaces between objects in a 3D scene, with a focus on
    negative space analysis.
    
    Attributes:
        regions (List[InterstitialRegion]): List of identified interstitial regions
        object_points (List[np.ndarray]): Points belonging to each object
        object_labels (np.ndarray): Object label for each point
        void_points (np.ndarray): Points representing void space
    """
    
    def __init__(self):
        """Initialize the interstitial analyzer"""
        self.regions: List[InterstitialRegion] = []
        self.object_points: List[np.ndarray] = []
        self.object_labels: np.ndarray = np.array([])
        self.void_points: np.ndarray = np.array([])
        
    def set_object_points(self, points: np.ndarray, labels: np.ndarray):
        """
        Set the object points and their labels.
        
        Args:
            points: Points in the scene (Nx3 numpy array)
            labels: Object label for each point (N numpy array)
        """
        self.object_labels = labels
        
        # Group points by object
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]  # Exclude noise (-1)
        
        self.object_points = []
        for label in unique_labels:
            mask = labels == label
            obj_points = points[mask]
            self.object_points.append(obj_points)
            
        logger.info(f"Set object points: {len(self.object_points)} objects")
    
    def set_void_points(self, points: np.ndarray):
        """
        Set points representing void space.
        
        Args:
            points: Points representing void space (Nx3 numpy array)
        """
        self.void_points = points
        logger.info(f"Set void points: {len(self.void_points)} points")
    
    def analyze(self, 
                method: str = 'voronoi', 
                min_points_per_region: int = 10,
                min_region_volume: float = 0.01,
                eps: float = 0.2,
                adjacency_threshold: float = 0.1) -> List[InterstitialRegion]:
        """
        Analyze interstitial spaces in the scene.
        
        Args:
            method: Method to use for analysis ('voronoi', 'dbscan', or 'kmeans')
            min_points_per_region: Minimum number of points required for a valid region
            min_region_volume: Minimum volume for a valid region
            eps: Distance parameter for DBSCAN clustering
            adjacency_threshold: Distance threshold for adjacency computation
            
        Returns:
            List of identified interstitial regions
        """
        if len(self.void_points) == 0:
            logger.warning("No void points available for analysis")
            return []
            
        logger.info(f"Analyzing interstitial spaces using method: {method}")
        
        # Cluster void points into regions
        if method == 'voronoi':
            self._cluster_by_voronoi()
        elif method == 'dbscan':
            self._cluster_by_dbscan(eps=eps)
        elif method == 'kmeans':
            # Estimate number of clusters based on object count
            n_clusters = max(2, len(self.object_points) * 2)
            self._cluster_by_kmeans(n_clusters=n_clusters)
        else:
            logger.warning(f"Unknown method: {method}, falling back to DBSCAN")
            self._cluster_by_dbscan(eps=eps)
        
        # Filter regions by size and volume
        filtered_regions = []
        for region in self.regions:
            if (len(region.points) >= min_points_per_region and 
                region.volume >= min_region_volume):
                # Compute adjacency to objects
                region.compute_adjacency(self.object_points, threshold=adjacency_threshold)
                # Compute signature
                region.compute_signature()
                filtered_regions.append(region)
        
        self.regions = filtered_regions
        logger.info(f"Identified {len(self.regions)} valid interstitial regions")
        
        return self.regions
    
    def _cluster_by_dbscan(self, eps: float = 0.2, min_samples: int = 5):
        """
        Cluster void points using DBSCAN algorithm.
        
        Args:
            eps: The maximum distance between two samples for them to be considered neighbors
            min_samples: The number of samples in a neighborhood for a point to be a core point
        """
        if len(self.void_points) < min_samples:
            logger.warning("Not enough void points for DBSCAN clustering")
            return
            
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.void_points)
        
        # Create regions from clusters
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]  # Exclude noise (-1)
        
        self.regions = []
        for i, label in enumerate(unique_labels):
            mask = labels == label
            region_points = self.void_points[mask]
            region = InterstitialRegion(id=i, points=region_points)
            self.regions.append(region)
    
    def _cluster_by_kmeans(self, n_clusters: int = 5):
        """
        Cluster void points using K-means algorithm.
        
        Args:
            n_clusters: Number of clusters to form
        """
        if len(self.void_points) < n_clusters:
            logger.warning(f"Not enough void points for K-means clustering with {n_clusters} clusters")
            return
            
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(self.void_points)
        
        # Create regions from clusters
        self.regions = []
        for i in range(n_clusters):
            mask = labels == i
            region_points = self.void_points[mask]
            if len(region_points) > 0:
                region = InterstitialRegion(id=i, points=region_points)
                self.regions.append(region)
    
    def _cluster_by_voronoi(self):
        """
        Cluster void points using Voronoi tessellation.
        
        This method identifies interstitial regions by creating a Voronoi
        tessellation of the object points and then assigning void points
        to the nearest object.
        """
        if len(self.void_points) == 0 or len(self.object_points) == 0:
            logger.warning("Not enough points for Voronoi tessellation")
            return
            
        # Create centroids for each object
        centroids = np.array([np.mean(points, axis=0) for points in self.object_points])
        
        if len(centroids) < 4:
            # Need at least 4 points for 3D Voronoi
            logger.warning("Not enough centroids for Voronoi tessellation, adding dummy points")
            # Add dummy points far away
            dummy_points = np.array([
                [1000, 0, 0],
                [0, 1000, 0],
                [0, 0, 1000],
                [-1000, 0, 0],
                [0, -1000, 0],
                [0, 0, -1000]
            ])
            centroids = np.vstack([centroids, dummy_points])
        
        try:
            # Compute Voronoi regions
            vor = Voronoi(centroids)
            
            # For each void point, find the nearest centroid
            distances = np.zeros((len(self.void_points), len(centroids)))
            for i, centroid in enumerate(centroids):
                distances[:, i] = np.linalg.norm(self.void_points - centroid, axis=1)
                
            # Get the second nearest centroid for each void point
            sorted_indices = np.argsort(distances, axis=1)
            nearest_indices = sorted_indices[:, 0]
            second_nearest_indices = sorted_indices[:, 1]
            
            # Create regions for each pair of adjacent objects
            region_map = {}
            
            for i, (first, second) in enumerate(zip(nearest_indices, second_nearest_indices)):
                # Skip dummy points
                if first >= len(self.object_points) or second >= len(self.object_points):
                    continue
                    
                # Create a unique key for this pair of objects
                key = tuple(sorted([int(first), int(second)]))
                
                if key not in region_map:
                    region_map[key] = []
                    
                region_map[key].append(i)
            
            # Create interstitial regions
            self.regions = []
            for i, (key, indices) in enumerate(region_map.items()):
                if len(indices) < 5:  # Skip regions with too few points
                    continue
                    
                region_points = self.void_points[indices]
                region = InterstitialRegion(id=i, points=region_points)
                
                # Set adjacent objects
                region.adjacent_objects = list(key)
                
                self.regions.append(region)
                
        except Exception as e:
            logger.error(f"Error during Voronoi tessellation: {str(e)}")
            # Fall back to DBSCAN
            logger.info("Falling back to DBSCAN clustering")
            self._cluster_by_dbscan()
    
    def get_region_by_id(self, id: int) -> Optional[InterstitialRegion]:
        """
        Get an interstitial region by its ID.
        
        Args:
            id: ID of the region to retrieve
            
        Returns:
            InterstitialRegion or None if not found
        """
        for region in self.regions:
            if region.id == id:
                return region
        return None
    
    def compute_global_signature(self, num_features: int = 64) -> np.ndarray:
        """
        Compute a global spatial signature for all interstitial spaces.
        
        Args:
            num_features: Number of features in the signature vector
            
        Returns:
            Numpy array containing the global signature vector
        """
        if len(self.regions) == 0:
            return np.zeros(num_features)
            
        # Compute signatures for all regions
        for region in self.regions:
            if len(region.signature) == 0:
                region.compute_signature()
        
        # Combine region signatures into a global signature
        # Strategy 1: Average of all region signatures
        region_signatures = [region.signature for region in self.regions]
        if not region_signatures:
            return np.zeros(num_features)
            
        avg_signature = np.mean(region_signatures, axis=0)
        
        # Strategy 2: Additional global features
        global_features = []
        
        # Number of regions
        global_features.append(len(self.regions))
        
        # Total volume and surface area
        total_volume = sum(region.volume for region in self.regions)
        total_surface_area = sum(region.surface_area for region in self.regions)
        global_features.extend([total_volume, total_surface_area])
        
        # Average number of adjacent objects per region
        avg_adjacency = np.mean([len(region.adjacent_objects) for region in self.regions])
        global_features.append(avg_adjacency)
        
        # Distribution of region volumes
        volumes = [region.volume for region in self.regions]
        if volumes:
            global_features.extend([
                np.mean(volumes),
                np.std(volumes),
                np.max(volumes),
                np.min(volumes)
            ])
        else:
            global_features.extend([0, 0, 0, 0])
        
        # Combine features
        combined_features = np.concatenate([avg_signature, global_features])
        
        # Ensure we have exactly num_features
        if len(combined_features) > num_features:
            combined_features = combined_features[:num_features]
        elif len(combined_features) < num_features:
            combined_features = np.pad(combined_features, 
                                       (0, num_features - len(combined_features)),
                                       'constant')
        
        # Normalize
        norm = np.linalg.norm(combined_features)
        if norm > 0:
            combined_features = combined_features / norm
            
        return combined_features
    
    def visualize_all_regions(self) -> List[o3d.geometry.TriangleMesh]:
        """
        Generate visualizations for all interstitial regions.
        
        Returns:
            List of Open3D meshes representing each region
        """
        meshes = []
        
        for region in self.regions:
            mesh = region.visualize()
            if mesh is not None:
                # Color based on region ID for differentiation
                r = (region.id * 67) % 255 / 255.0
                g = (region.id * 101) % 255 / 255.0
                b = (region.id * 191) % 255 / 255.0
                mesh.paint_uniform_color([r, g, b])
                meshes.append(mesh)
                
        return meshes


def compute_voronoi_regions(points: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Compute Voronoi regions from a set of points.
    
    Args:
        points: Input points (Nx3 numpy array)
        
    Returns:
        Dictionary mapping region ID to an array of vertices
    """
    if len(points) < 4:
        return {}
        
    try:
        # Compute Voronoi diagram
        vor = Voronoi(points)
        
        # Extract regions
        regions = {}
        for i, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if -1 not in region and len(region) > 0:
                vertices = np.array([vor.vertices[v] for v in region])
                regions[i] = vertices
                
        return regions
    except Exception as e:
        logger.error(f"Error computing Voronoi regions: {str(e)}")
        return {}

def identify_region_boundaries(regions: List[InterstitialRegion], 
                              threshold: float = 0.1) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Identify boundaries between interstitial regions.
    
    Args:
        regions: List of interstitial regions
        threshold: Distance threshold for considering points on a boundary
        
    Returns:
        Dictionary mapping region pairs to boundary points
    """
    if len(regions) < 2:
        return {}
        
    boundaries = {}
    
    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions[i+1:], i+1):
            boundary_points = []
            
            # Sample points for efficiency
            max_samples = 100
            sample1 = region1.points
            sample2 = region2.points
            
            if len(sample1) > max_samples:
                indices = np.random.choice(len(sample1), max_samples, replace=False)
                sample1 = sample1[indices]
                
            if len(sample2) > max_samples:
                indices = np.random.choice(len(sample2), max_samples, replace=False)
                sample2 = sample2[indices]
            
            # Find points in region1 that are close to region2
            for p1 in sample1:
                distances = np.linalg.norm(sample2 - p1, axis=1)
                if np.min(distances) < threshold:
                    boundary_points.append(p1)
            
            # Find points in region2 that are close to region1
            for p2 in sample2:
                distances = np.linalg.norm(sample1 - p2, axis=1)
                if np.min(distances) < threshold:
                    boundary_points.append(p2)
            
            if boundary_points:
                boundaries[(region1.id, region2.id)] = np.array(boundary_points)
                
    return boundaries
