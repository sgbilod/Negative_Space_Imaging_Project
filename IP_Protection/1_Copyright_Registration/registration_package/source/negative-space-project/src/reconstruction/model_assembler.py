"""
Model Assembler

This module provides functionality for assembling 3D models from point clouds,
with specialized focus on negative space reconstruction and analysis.

The model assembler takes point clouds generated in previous steps and creates
cohesive 3D models that emphasize both the physical objects and the negative
spaces between them, allowing for detailed analysis of spatial relationships.

Classes:
    ModelComponent: Represents a single component of the assembled model
    NegativeSpaceComponent: Specialized component for negative space regions
    ModelAssembler: Main class for assembling complete models
    
Functions:
    create_mesh_from_point_cloud: Creates a mesh from a point cloud
    optimize_mesh: Optimizes a mesh for visualization and analysis
"""

import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple, Optional, Union, Set
import logging
import os
import time
from enum import Enum
from scipy.spatial import Delaunay

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """Enum for component types"""
    OBJECT = 1
    NEGATIVE_SPACE = 2
    BOUNDARY = 3
    INTERSTITIAL = 4

class ModelComponent:
    """
    Represents a single component of the assembled model.
    
    Attributes:
        id (int): Unique identifier for the component
        type (ComponentType): Type of the component
        mesh (o3d.geometry.TriangleMesh): Mesh representation of the component
        points (np.ndarray): Original points used to create the component
        color (np.ndarray): Color of the component (RGB)
        name (str): Name of the component
        metadata (Dict): Additional metadata for the component
    """
    
    def __init__(self, id: int, type: ComponentType, points: np.ndarray, 
                 color: np.ndarray = None, name: str = ""):
        """
        Initialize a model component.
        
        Args:
            id: Unique identifier for the component
            type: Type of the component
            points: Points belonging to this component (Nx3 numpy array)
            color: Color of the component (RGB)
            name: Name of the component
        """
        self.id = id
        self.type = type
        self.points = points
        self.color = color if color is not None else np.array([0.5, 0.5, 0.5])
        self.name = name if name else f"{type.name}_{id}"
        self.mesh = None
        self.metadata = {}
        
        # Create point cloud
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        if color is not None:
            self.point_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
    
    def create_mesh(self, method: str = "alpha_shape", alpha: float = 0.5, 
                    depth: int = 8, scale: float = 1.2):
        """
        Create a mesh representation of the component.
        
        Args:
            method: Method to use for mesh creation ("alpha_shape", "ball_pivoting", or "poisson")
            alpha: Alpha value for alpha shape reconstruction
            depth: Depth parameter for Poisson reconstruction
            scale: Scale parameter for Poisson reconstruction
            
        Returns:
            Open3D mesh
        """
        if len(self.points) < 4:
            logger.warning(f"Not enough points to create mesh for component {self.id}")
            return None
            
        try:
            # Ensure we have normals for the point cloud
            if not self.point_cloud.has_normals():
                self.point_cloud.estimate_normals()
                self.point_cloud.orient_normals_consistent_tangent_plane(100)
            
            # Create mesh using specified method
            if method == "alpha_shape":
                self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    self.point_cloud, alpha)
            elif method == "ball_pivoting":
                # Estimate radius for ball pivoting
                distances = self.point_cloud.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radii = [avg_dist * 2, avg_dist * 5, avg_dist * 10]
                self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    self.point_cloud, o3d.utility.DoubleVector(radii))
            elif method == "poisson":
                self.mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    self.point_cloud, depth=depth, scale=scale, linear_fit=True)
            else:
                logger.warning(f"Unknown mesh creation method: {method}, falling back to alpha shape")
                self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    self.point_cloud, alpha)
            
            # Compute normals for proper rendering
            self.mesh.compute_vertex_normals()
            
            # Set color
            self.mesh.paint_uniform_color(self.color)
            
            logger.info(f"Created mesh for component {self.id} with {len(self.mesh.triangles)} triangles")
            return self.mesh
            
        except Exception as e:
            logger.error(f"Failed to create mesh for component {self.id}: {str(e)}")
            
            # Try with convex hull as fallback
            try:
                logger.info(f"Trying convex hull as fallback for component {self.id}")
                self.mesh, _ = self.point_cloud.compute_convex_hull()
                self.mesh.compute_vertex_normals()
                self.mesh.paint_uniform_color(self.color)
                return self.mesh
            except Exception as e2:
                logger.error(f"Convex hull fallback also failed for component {self.id}: {str(e2)}")
                return None
    
    def optimize_mesh(self, target_reduction: float = 0.5, preserve_boundaries: bool = True):
        """
        Optimize the mesh for visualization and analysis.
        
        Args:
            target_reduction: Target reduction ratio (0-1)
            preserve_boundaries: Whether to preserve mesh boundaries
            
        Returns:
            Optimized mesh
        """
        if self.mesh is None:
            logger.warning(f"No mesh to optimize for component {self.id}")
            return None
            
        try:
            # Remove duplicate vertices
            self.mesh.remove_duplicated_vertices()
            
            # Remove degenerate triangles
            self.mesh.remove_degenerate_triangles()
            
            # Fix normal orientation
            self.mesh.orient_triangles()
            
            # Simplify mesh
            if target_reduction > 0 and len(self.mesh.triangles) > 100:
                target_triangles = int(len(self.mesh.triangles) * (1 - target_reduction))
                self.mesh = self.mesh.simplify_quadric_decimation(target_triangles)
                
            # Recompute normals
            self.mesh.compute_vertex_normals()
            
            # Apply color again
            self.mesh.paint_uniform_color(self.color)
            
            logger.info(f"Optimized mesh for component {self.id}, now has {len(self.mesh.triangles)} triangles")
            return self.mesh
            
        except Exception as e:
            logger.error(f"Failed to optimize mesh for component {self.id}: {str(e)}")
            return self.mesh
    
    def save(self, output_dir: str):
        """
        Save the component to disk.
        
        Args:
            output_dir: Directory to save the component
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save point cloud
        pc_path = os.path.join(output_dir, f"{self.name}_point_cloud.ply")
        o3d.io.write_point_cloud(pc_path, self.point_cloud)
        
        # Save mesh if available
        if self.mesh is not None:
            mesh_path = os.path.join(output_dir, f"{self.name}_mesh.ply")
            o3d.io.write_triangle_mesh(mesh_path, self.mesh)
            
        logger.info(f"Saved component {self.id} to {output_dir}")


class NegativeSpaceComponent(ModelComponent):
    """
    Specialized component for negative space regions.
    
    This class extends ModelComponent with additional functionality
    specific to negative space analysis.
    
    Attributes:
        adjacent_objects (List[int]): IDs of adjacent object components
        volume (float): Volume of the negative space
        signature (np.ndarray): Spatial signature of the negative space
    """
    
    def __init__(self, id: int, points: np.ndarray, color: np.ndarray = None, name: str = ""):
        """
        Initialize a negative space component.
        
        Args:
            id: Unique identifier for the component
            points: Points belonging to this component (Nx3 numpy array)
            color: Color of the component (RGB)
            name: Name of the component
        """
        super().__init__(id, ComponentType.NEGATIVE_SPACE, points, color, name)
        self.adjacent_objects: List[int] = []
        self.volume = 0.0
        self.signature = np.array([])
    
    def compute_volume(self):
        """
        Compute the volume of the negative space.
        
        Returns:
            Volume of the negative space
        """
        if self.mesh is None:
            # Try to create a mesh if not available
            self.create_mesh()
            
        if self.mesh is None:
            logger.warning(f"Cannot compute volume for component {self.id}: no mesh available")
            return 0.0
            
        try:
            # Make sure the mesh is watertight
            self.mesh.compute_vertex_normals()
            self.mesh.orient_triangles()
            
            # Compute volume
            self.volume = self.mesh.get_volume()
            logger.info(f"Computed volume for component {self.id}: {self.volume}")
            return self.volume
            
        except Exception as e:
            logger.error(f"Failed to compute volume for component {self.id}: {str(e)}")
            
            # Try with convex hull as fallback
            try:
                hull, _ = self.point_cloud.compute_convex_hull()
                self.volume = hull.get_volume()
                return self.volume
            except:
                return 0.0
    
    def compute_signature(self, num_features: int = 32):
        """
        Compute a spatial signature for this negative space component.
        
        Args:
            num_features: Number of features in the signature
            
        Returns:
            Signature vector
        """
        if len(self.points) < 10:
            self.signature = np.zeros(num_features)
            return self.signature
            
        try:
            # Initialize feature vector
            features = []
            
            # Compute volume if not already done
            if self.volume == 0.0:
                self.compute_volume()
                
            # Basic features
            features.append(self.volume)
            features.append(len(self.points))
            features.append(len(self.adjacent_objects))
            
            # Shape features - eigenvalues of covariance matrix
            covariance = np.cov(self.points, rowvar=False)
            eigenvalues, _ = np.linalg.eigh(covariance)
            eigenvalues = sorted(eigenvalues, reverse=True)
            
            # Ensure we have 3 eigenvalues
            eigenvalues = list(eigenvalues) + [0] * (3 - len(eigenvalues))
            features.extend(eigenvalues)
            
            # Shape factors
            if eigenvalues[0] > 0:
                planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
                sphericity = eigenvalues[2] / eigenvalues[0]
                linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
            else:
                planarity, sphericity, linearity = 0, 0, 0
                
            features.extend([planarity, sphericity, linearity])
            
            # Center of mass
            center = np.mean(self.points, axis=0)
            
            # Spatial distribution features
            distances = np.linalg.norm(self.points - center, axis=1)
            features.append(np.mean(distances))  # Mean distance from center
            features.append(np.std(distances))   # Std of distances from center
            features.append(np.max(distances))   # Max distance from center
            
            # Ensure we have exactly num_features
            if len(features) > num_features:
                features = features[:num_features]
            elif len(features) < num_features:
                features.extend([0] * (num_features - len(features)))
            
            # Normalize features
            features_array = np.array(features)
            norm = np.linalg.norm(features_array)
            if norm > 0:
                features_array = features_array / norm
                
            self.signature = features_array
            return self.signature
            
        except Exception as e:
            logger.error(f"Failed to compute signature for component {self.id}: {str(e)}")
            self.signature = np.zeros(num_features)
            return self.signature


class ModelAssembler:
    """
    Main class for assembling complete models from components.
    
    This class takes point clouds and other data generated in previous
    steps and creates a cohesive 3D model that includes both physical
    objects and negative spaces.
    
    Attributes:
        components (List[ModelComponent]): List of model components
        negative_space_components (List[NegativeSpaceComponent]): List of negative space components
        global_signature (np.ndarray): Global signature of the assembled model
    """
    
    def __init__(self):
        """Initialize the model assembler"""
        self.components: List[ModelComponent] = []
        self.negative_space_components: List[NegativeSpaceComponent] = []
        self.global_signature = np.array([])
    
    def add_component(self, component: ModelComponent):
        """
        Add a component to the model.
        
        Args:
            component: Component to add
        """
        self.components.append(component)
        
        # Add to negative space components if applicable
        if isinstance(component, NegativeSpaceComponent):
            self.negative_space_components.append(component)
            
        logger.info(f"Added component {component.id} of type {component.type.name} to model")
    
    def create_component_from_points(self, id: int, type: ComponentType, 
                                    points: np.ndarray, color: np.ndarray = None,
                                    name: str = "", create_mesh: bool = True,
                                    mesh_method: str = "alpha_shape"):
        """
        Create and add a new component from points.
        
        Args:
            id: Unique identifier for the component
            type: Type of the component
            points: Points belonging to this component (Nx3 numpy array)
            color: Color of the component (RGB)
            name: Name of the component
            create_mesh: Whether to create a mesh for the component
            mesh_method: Method to use for mesh creation
            
        Returns:
            Created component
        """
        if len(points) < 4:
            logger.warning(f"Not enough points to create component {id}")
            return None
            
        # Create appropriate component type
        if type == ComponentType.NEGATIVE_SPACE or type == ComponentType.INTERSTITIAL:
            component = NegativeSpaceComponent(id, points, color, name)
        else:
            component = ModelComponent(id, type, points, color, name)
        
        # Create mesh if requested
        if create_mesh:
            component.create_mesh(method=mesh_method)
            component.optimize_mesh()
        
        # Add component to model
        self.add_component(component)
        
        return component
    
    def assemble(self, optimize: bool = True):
        """
        Assemble the complete model from components.
        
        Args:
            optimize: Whether to optimize the model
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Assembling model with {len(self.components)} components")
        
        # Create meshes for components that don't have them
        for component in self.components:
            if component.mesh is None:
                component.create_mesh()
                
                if optimize:
                    component.optimize_mesh()
        
        # Compute volumes for negative space components
        for component in self.negative_space_components:
            if component.volume == 0.0:
                component.compute_volume()
        
        # Compute adjacency between negative space and object components
        self._compute_adjacency()
        
        # Compute global signature
        self.compute_global_signature()
        
        logger.info("Model assembly complete")
        return self
    
    def _compute_adjacency(self, threshold: float = 0.1):
        """
        Compute adjacency between negative space and object components.
        
        Args:
            threshold: Distance threshold for adjacency
        """
        # Get object components
        object_components = [c for c in self.components 
                            if c.type == ComponentType.OBJECT]
        
        # Compute adjacency for each negative space component
        for ns_comp in self.negative_space_components:
            ns_comp.adjacent_objects = []
            
            # Sample points for efficiency
            ns_points = ns_comp.points
            if len(ns_points) > 100:
                indices = np.random.choice(len(ns_points), 100, replace=False)
                ns_points = ns_points[indices]
            
            for obj_comp in object_components:
                # Sample object points
                obj_points = obj_comp.points
                if len(obj_points) > 100:
                    indices = np.random.choice(len(obj_points), 100, replace=False)
                    obj_points = obj_points[indices]
                
                # Check if any points are close enough
                min_dist = float('inf')
                for ns_p in ns_points:
                    for obj_p in obj_points:
                        dist = np.linalg.norm(ns_p - obj_p)
                        if dist < min_dist:
                            min_dist = dist
                        
                        if dist < threshold:
                            ns_comp.adjacent_objects.append(obj_comp.id)
                            break
                    
                    if obj_comp.id in ns_comp.adjacent_objects:
                        break
        
        logger.info("Computed adjacency between components")
    
    def compute_global_signature(self, num_features: int = 64):
        """
        Compute a global signature for the assembled model.
        
        Args:
            num_features: Number of features in the signature
            
        Returns:
            Global signature vector
        """
        if len(self.components) == 0:
            logger.warning("No components available for global signature computation")
            self.global_signature = np.zeros(num_features)
            return self.global_signature
            
        try:
            # Ensure all negative space components have signatures
            for comp in self.negative_space_components:
                if len(comp.signature) == 0:
                    comp.compute_signature()
            
            # Combine negative space signatures
            if self.negative_space_components:
                ns_signatures = [comp.signature for comp in self.negative_space_components]
                avg_ns_signature = np.mean(ns_signatures, axis=0)
            else:
                avg_ns_signature = np.zeros(32)  # Default size
            
            # Global features
            global_features = []
            
            # Component counts
            object_count = sum(1 for c in self.components if c.type == ComponentType.OBJECT)
            ns_count = len(self.negative_space_components)
            global_features.extend([object_count, ns_count])
            
            # Volumetric information
            total_ns_volume = sum(c.volume for c in self.negative_space_components)
            global_features.append(total_ns_volume)
            
            if self.negative_space_components:
                volumes = [c.volume for c in self.negative_space_components]
                global_features.extend([
                    np.mean(volumes),
                    np.std(volumes),
                    np.max(volumes),
                    np.min(volumes)
                ])
            else:
                global_features.extend([0, 0, 0, 0])
            
            # Adjacency information
            adjacencies = [len(c.adjacent_objects) for c in self.negative_space_components]
            if adjacencies:
                global_features.extend([
                    np.mean(adjacencies),
                    np.std(adjacencies),
                    np.max(adjacencies),
                    np.min(adjacencies)
                ])
            else:
                global_features.extend([0, 0, 0, 0])
            
            # Combine all features
            combined_features = np.concatenate([avg_ns_signature, global_features])
            
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
                
            self.global_signature = combined_features
            logger.info(f"Computed global signature with {len(self.global_signature)} features")
            return self.global_signature
            
        except Exception as e:
            logger.error(f"Failed to compute global signature: {str(e)}")
            self.global_signature = np.zeros(num_features)
            return self.global_signature
    
    def save(self, output_dir: str, save_components: bool = True):
        """
        Save the assembled model to disk.
        
        Args:
            output_dir: Directory to save the model
            save_components: Whether to save individual components
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual components
        if save_components:
            components_dir = os.path.join(output_dir, "components")
            os.makedirs(components_dir, exist_ok=True)
            
            for component in self.components:
                component_dir = os.path.join(components_dir, component.name)
                component.save(component_dir)
        
        # Save combined visualization
        try:
            # Create a combined mesh
            combined_mesh = o3d.geometry.TriangleMesh()
            
            for component in self.components:
                if component.mesh is not None:
                    # Create a copy of the mesh
                    mesh_copy = o3d.geometry.TriangleMesh(component.mesh)
                    combined_mesh += mesh_copy
            
            # Save combined mesh
            combined_path = os.path.join(output_dir, "combined_model.ply")
            o3d.io.write_triangle_mesh(combined_path, combined_mesh)
            
            # Save global signature
            if len(self.global_signature) > 0:
                np.save(os.path.join(output_dir, "global_signature.npy"), self.global_signature)
                
            logger.info(f"Saved assembled model to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save combined model: {str(e)}")
    
    def visualize(self, show_negative_space: bool = True, 
                 negative_space_opacity: float = 0.5):
        """
        Visualize the assembled model.
        
        Args:
            show_negative_space: Whether to show negative space components
            negative_space_opacity: Opacity for negative space components (0-1)
            
        Returns:
            Open3D visualizer
        """
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add object components first
        for component in self.components:
            if component.type == ComponentType.OBJECT:
                if component.mesh is not None:
                    vis.add_geometry(component.mesh)
        
        # Add negative space components if requested
        if show_negative_space:
            for component in self.negative_space_components:
                if component.mesh is not None:
                    vis.add_geometry(component.mesh)
        
        # Set rendering options
        opt = vis.get_render_option()
        opt.mesh_show_wireframe = False
        opt.mesh_show_back_face = True
        
        # Run visualization
        vis.run()
        vis.destroy_window()
        
        return vis


def create_mesh_from_point_cloud(points: np.ndarray, colors: np.ndarray = None,
                                method: str = "alpha_shape", alpha: float = 0.5,
                                depth: int = 8, scale: float = 1.2) -> o3d.geometry.TriangleMesh:
    """
    Create a mesh from a point cloud.
    
    Args:
        points: Input points (Nx3 numpy array)
        colors: Input colors (Nx3 numpy array)
        method: Method to use for mesh creation ("alpha_shape", "ball_pivoting", or "poisson")
        alpha: Alpha value for alpha shape reconstruction
        depth: Depth parameter for Poisson reconstruction
        scale: Scale parameter for Poisson reconstruction
        
    Returns:
        Open3D mesh
    """
    if len(points) < 4:
        logger.warning("Not enough points to create mesh")
        return None
        
    try:
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals if needed
        if not pcd.has_normals():
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(100)
        
        # Create mesh using specified method
        if method == "alpha_shape":
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        elif method == "ball_pivoting":
            # Estimate radius for ball pivoting
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist * 2, avg_dist * 5, avg_dist * 10]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
        elif method == "poisson":
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, scale=scale, linear_fit=True)
        else:
            logger.warning(f"Unknown mesh creation method: {method}, falling back to alpha shape")
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        
        # Compute normals for proper rendering
        mesh.compute_vertex_normals()
        
        logger.info(f"Created mesh with {len(mesh.triangles)} triangles")
        return mesh
        
    except Exception as e:
        logger.error(f"Failed to create mesh: {str(e)}")
        
        # Try with convex hull as fallback
        try:
            logger.info("Trying convex hull as fallback")
            mesh, _ = pcd.compute_convex_hull()
            mesh.compute_vertex_normals()
            return mesh
        except Exception as e2:
            logger.error(f"Convex hull fallback also failed: {str(e2)}")
            return None

def optimize_mesh(mesh: o3d.geometry.TriangleMesh, target_reduction: float = 0.5,
                preserve_boundaries: bool = True) -> o3d.geometry.TriangleMesh:
    """
    Optimize a mesh for visualization and analysis.
    
    Args:
        mesh: Input mesh
        target_reduction: Target reduction ratio (0-1)
        preserve_boundaries: Whether to preserve mesh boundaries
        
    Returns:
        Optimized mesh
    """
    if mesh is None:
        logger.warning("No mesh to optimize")
        return None
        
    try:
        # Make a copy of the mesh
        result = o3d.geometry.TriangleMesh(mesh)
        
        # Remove duplicate vertices
        result.remove_duplicated_vertices()
        
        # Remove degenerate triangles
        result.remove_degenerate_triangles()
        
        # Fix normal orientation
        result.orient_triangles()
        
        # Simplify mesh
        if target_reduction > 0 and len(result.triangles) > 100:
            target_triangles = int(len(result.triangles) * (1 - target_reduction))
            result = result.simplify_quadric_decimation(target_triangles)
            
        # Recompute normals
        result.compute_vertex_normals()
        
        logger.info(f"Optimized mesh, reduced from {len(mesh.triangles)} to {len(result.triangles)} triangles")
        return result
        
    except Exception as e:
        logger.error(f"Failed to optimize mesh: {str(e)}")
        return mesh
