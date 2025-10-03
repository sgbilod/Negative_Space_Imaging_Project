# Documentation for visual_explorer.py

```python
"""
Mnemonic Data Architecture Visual Explorer

This script provides a basic 3D visualization of the Mnemonic Data Architecture using PyVista,
a Pythonic interface to the VTK (Visualization Toolkit).

Requirements:
- pyvista
- numpy

Install with: pip install pyvista numpy
"""

import json
import os
import sys
import numpy as np
import pyvista as pv

# Adjust the import path based on your project structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.revenue.mnemonic_architecture.mnemonic_data_architecture import MnemonicDataArchitecture


class MnemonicVisualExplorer:
    """
    A class for visualizing the Mnemonic Data Architecture in 3D space.
    """
    
    # Node type to color mapping
    NODE_COLORS = {
        "document": "blue",
        "image": "green",
        "code": "red",
        "spreadsheet": "yellow",
        "email": "orange",
        "default": "gray"
    }
    
    def __init__(self, mda):
        """
        Initialize the visual explorer.
        
        Args:
            mda: MnemonicDataArchitecture instance
        """
        self.mda = mda
        self.plotter = pv.Plotter()
        self.nodes_actors = {}  # node_id -> actor
        self.clusters_actors = {}  # cluster_id -> actor
        self.connection_actors = []  # List of connection actors
        
        # Set up the plotter
        self.plotter.set_background("white")
        self.plotter.add_axes()
        
    def visualize(self):
        """
        Visualize the current state of the mnemonic data architecture.
        """
        # Clear any existing actors
        self.plotter.clear()
        self.nodes_actors = {}
        self.clusters_actors = {}
        self.connection_actors = []
        
        # Add clusters first (so they appear behind nodes)
        self._add_clusters()
        
        # Add nodes
        self._add_nodes()
        
        # Add connections between nodes
        self._add_connections()
        
        # Add a title
        self.plotter.add_title("Mnemonic Data Architecture Visualization", font_size=18)
        
        # Show the visualization
        self.plotter.show()
    
    def _add_nodes(self):
        """Add nodes to the visualization."""
        for node_id, node in self.mda.nodes.items():
            # Create a sphere for the node
            center = node.position
            radius = 1.0  # Base radius
            
            # Adjust radius based on connections
            radius += 0.1 * len(node.connections)
            
            # Create the sphere
            sphere = pv.Sphere(radius=radius, center=center)
            
            # Get color based on data type
            color = self.NODE_COLORS.get(node.data_type, self.NODE_COLORS["default"])
            
            # Add the sphere to the plotter
            self.nodes_actors[node_id] = self.plotter.add_mesh(
                sphere, 
                color=color, 
                smooth_shading=True,
                name=f"node_{node_id}"
            )
            
            # Add a label with the node name
            name = node.metadata.get("name", f"Node {node_id[:8]}")
            self.plotter.add_point_labels(
                [center], 
                [name], 
                font_size=10, 
                point_color=color, 
                shape_opacity=0.5
            )
    
    def _add_clusters(self):
        """Add clusters to the visualization."""
        for cluster_id, cluster in self.mda.clusters.items():
            # Create a transparent sphere for the cluster
            center = cluster.center
            radius = cluster.radius
            
            # Create the sphere
            sphere = pv.Sphere(radius=radius, center=center)
            
            # Add the sphere to the plotter with transparency
            self.clusters_actors[cluster_id] = self.plotter.add_mesh(
                sphere, 
                color="lightblue", 
                opacity=0.2, 
                smooth_shading=True,
                name=f"cluster_{cluster_id}"
            )
            
            # Add a label with the cluster name
            self.plotter.add_point_labels(
                [center], 
                [cluster.name], 
                font_size=14, 
                bold=True,
                point_color="blue", 
                shape_opacity=0.2
            )
    
    def _add_connections(self):
        """Add connections between nodes."""
        # Track which connections we've already drawn
        drawn_connections = set()
        
        for node_id, node in self.mda.nodes.items():
            for conn_id in node.connections:
                # Create a unique key for this connection
                conn_key = tuple(sorted([node_id, conn_id]))
                
                # Skip if we've already drawn this connection
                if conn_key in drawn_connections:
                    continue
                
                # Mark as drawn
                drawn_connections.add(conn_key)
                
                # Get the connected node
                if conn_id not in self.mda.nodes:
                    continue
                    
                conn_node = self.mda.nodes[conn_id]
                
                # Create a line between the nodes
                line = pv.Line(node.position, conn_node.position)
                
                # Add the line to the plotter
                actor = self.plotter.add_mesh(
                    line, 
                    color="lightgray", 
                    line_width=2,
                    name=f"conn_{node_id}_{conn_id}"
                )
                
                self.connection_actors.append(actor)
    
    def visualize_path(self, path_id):
        """
        Visualize a specific path through the data.
        
        Args:
            path_id: ID of the path to visualize
        """
        if path_id not in self.mda.paths:
            print(f"Path {path_id} not found")
            return
            
        path = self.mda.paths[path_id]
        
        # Clear any existing actors
        self.plotter.clear()
        
        # Add all nodes with reduced opacity
        for node_id, node in self.mda.nodes.items():
            # Create a sphere for the node
            center = node.position
            radius = 0.5  # Smaller radius for non-path nodes
            
            # Create the sphere
            sphere = pv.Sphere(radius=radius, center=center)
            
            # Get color based on data type
            color = self.NODE_COLORS.get(node.data_type, self.NODE_COLORS["default"])
            
            # Add the sphere to the plotter with reduced opacity
            self.plotter.add_mesh(
                sphere, 
                color=color, 
                opacity=0.3,
                smooth_shading=True,
                name=f"node_{node_id}"
            )
        
        # Highlight the path nodes and create the path
        points = []
        for i, node_id in enumerate(path.nodes):
            if node_id not in self.mda.nodes:
                continue
                
            node = self.mda.nodes[node_id]
            points.append(node.position)
            
            # Create a larger sphere for the path node
            center = node.position
            radius = 1.2  # Larger radius for path nodes
            
            # Create the sphere
            sphere = pv.Sphere(radius=radius, center=center)
            
            # Get color based on data type
            color = self.NODE_COLORS.get(node.data_type, self.NODE_COLORS["default"])
            
            # Add the sphere to the plotter
            self.plotter.add_mesh(
                sphere, 
                color=color, 
                smooth_shading=True,
                name=f"path_node_{node_id}"
            )
            
            # Add a label with the node name and sequence number
            name = node.metadata.get("name", f"Node {node_id[:8]}")
            self.plotter.add_point_labels(
                [center], 
                [f"{i+1}. {name}"], 
                font_size=12, 
                bold=True,
                point_color=color, 
                shape_opacity=0.7
            )
        
        # Create the path as a tube
        if len(points) >= 2:
            points = np.array(points)
            spline = pv.Spline(points, 100)
            tube = spline.tube(radius=0.3)
            self.plotter.add_mesh(tube, color="gold", smooth_shading=True, name="path_tube")
        
        # Add a title
        self.plotter.add_title(f"Path: {path.name}", font_size=18)
        
        # Show the visualization
        self.plotter.show()
    
    def interactive_explorer(self):
        """
        Launch an interactive explorer for the Mnemonic Data Architecture.
        This provides a more interactive experience with the ability to
        select nodes, view details, and navigate paths.
        """
        # Set up a callback for when a node is clicked
        def node_clicked(node_id):
            # Get the node
            node = self.mda.nodes[node_id]
            
            # Display node information
            info = (
                f"Node: {node.metadata.get('name', node_id)}\n"
                f"Type: {node.data_type}\n"
                f"Tags: {', '.join(node.metadata.get('tags', []))}\n"
                f"Connections: {len(node.connections)}\n"
            )
            
            self.plotter.add_text(info, position="upper_left", font_size=12)
            
            # Highlight this node and its connections
            self._highlight_node_and_connections(node_id)
        
        # Set up the plotter for interactive exploration
        self.plotter = pv.Plotter()
        self.plotter.set_background("white")
        self.plotter.add_axes()
        
        # Add all nodes and clusters
        self._add_clusters()
        self._add_nodes()
        self._add_connections()
        
        # Add instructions
        instructions = (
            "Left-click: Select node\n"
            "Right-click: Reset view\n"
            "Mouse wheel: Zoom\n"
            "Middle-click + drag: Pan\n"
        )
        self.plotter.add_text(instructions, position="lower_left", font_size=10)
        
        # Set up a picker to handle node selection
        self.plotter.enable_point_picking(
            callback=lambda point, actor: node_clicked(actor.name.split('_')[1]),
            show_message=False,
            font_size=12,
            color="black",
            point_size=10,
            use_mesh=True
        )
        
        # Add a title
        self.plotter.add_title("Mnemonic Data Architecture Explorer", font_size=18)
        
        # Show the interactive visualization
        self.plotter.show()
    
    def _highlight_node_and_connections(self, node_id):
        """
        Highlight a specific node and its connections.
        
        Args:
            node_id: ID of the node to highlight
        """
        # Reset any previous highlighting
        for actor_id, actor in self.nodes_actors.items():
            if actor_id == node_id:
                actor.GetProperty().SetOpacity(1.0)
            else:
                actor.GetProperty().SetOpacity(0.3)
        
        # Highlight connections
        node = self.mda.nodes[node_id]
        for conn_id in node.connections:
            if conn_id in self.nodes_actors:
                self.nodes_actors[conn_id].GetProperty().SetOpacity(1.0)
        
        self.plotter.update()


def main():
    """Main function to demonstrate the visual explorer."""
    # Check if a state file was provided
    if len(sys.argv) > 1:
        state_file = sys.argv[1]
        
        # Load the state
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Create a new MDA and import the state
        mda = MnemonicDataArchitecture()
        import_result = mda.import_state(state)
        
        if not import_result["success"]:
            print(f"Error importing state: {import_result.get('error', 'Unknown error')}")
            return
            
        print(f"Imported state with {import_result['node_count']} nodes, {import_result['cluster_count']} clusters, and {import_result['path_count']} paths")
    else:
        # Create a new MDA and populate with example data
        mda = MnemonicDataArchitecture()
        _populate_example_data(mda)
    
    # Create the visual explorer
    explorer = MnemonicVisualExplorer(mda)
    
    # Show the basic visualization
    print("Showing basic visualization...")
    explorer.visualize()
    
    # If we have paths, visualize the first one
    if mda.paths:
        path_id = next(iter(mda.paths.keys()))
        print(f"Visualizing path: {mda.paths[path_id].name}...")
        explorer.visualize_path(path_id)
    
    # Launch the interactive explorer
    print("Launching interactive explorer...")
    explorer.interactive_explorer()


def _populate_example_data(mda):
    """Populate the MDA with example data for visualization."""
    # Add some nodes
    node_ids = []
    for i in range(20):
        # Generate random position
        position = [np.random.uniform(0, 100) for _ in range(3)]
        
        # Select random type
        types = ["document", "image", "code", "spreadsheet", "email"]
        data_type = np.random.choice(types)
        
        # Generate some random tags
        all_tags = ["project", "planning", "technical", "design", "meeting", "client", 
                  "code", "architecture", "budget", "requirements", "feedback"]
        tags = np.random.choice(all_tags, size=np.random.randint(1, 4), replace=False).tolist()
        
        # Create node
        result = mda.add_node(
            data_type=data_type,
            content=f"Content for node {i}",
            position=position,
            metadata={
                "name": f"{data_type.capitalize()} {i+1}",
                "tags": tags
            }
        )
        
        node_ids.append(result["node_id"])
    
    # Create some connections (for simplicity, connect sequential nodes)
    for i in range(len(node_ids)-1):
        node_id = node_ids[i]
        # Connect to 1-3 random other nodes
        conn_count = np.random.randint(1, 4)
        connections = np.random.choice(
            [nid for nid in node_ids if nid != node_id], 
            size=min(conn_count, len(node_ids)-1), 
            replace=False
        ).tolist()
        
        mda.update_node(
            node_id=node_id,
            updates={"connections": connections}
        )
    
    # Create some clusters
    for i in range(3):
        # Generate random center
        center = [np.random.uniform(20, 80) for _ in range(3)]
        
        # Create cluster
        mda.create_cluster(
            name=f"Cluster {i+1}",
            center=center,
            radius=np.random.uniform(15, 25),
            theme=f"theme-{i+1}"
        )
    
    # Create a path
    path_nodes = np.random.choice(node_ids, size=min(5, len(node_ids)), replace=False).tolist()
    mda.create_path(
        name="Example Path",
        nodes=path_nodes,
        metadata={"description": "An example path through the data"}
    )


if __name__ == "__main__":
    main()

```