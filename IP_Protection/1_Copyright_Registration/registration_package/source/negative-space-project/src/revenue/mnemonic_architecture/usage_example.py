"""
Mnemonic Data Architecture Usage Example

This script demonstrates how to use the Mnemonic Data Architecture to organize and navigate data
using a spatial-mnemonic system.
"""

import json
import os
from pprint import pprint

# Adjust the import path based on your project structure
# This example assumes you're running from the project root
from src.revenue.mnemonic_architecture.mnemonic_data_architecture import MnemonicDataArchitecture


def main():
    # Initialize the Mnemonic Data Architecture
    print("Initializing Mnemonic Data Architecture...")
    mda = MnemonicDataArchitecture()
    
    # Add some example data nodes
    print("\nAdding sample data nodes...")
    
    # Add a document node
    document_result = mda.add_node(
        data_type="document",
        content={
            "title": "Project Proposal",
            "text": "This is a proposal for a new project that leverages spatial data organization...",
            "author": "Jane Smith"
        },
        metadata={
            "name": "Project Proposal",
            "tags": ["proposal", "project", "planning"],
            "creation_date": "2025-07-15"
        }
    )
    document_id = document_result["node_id"]
    print(f"Created document node: {document_id}")
    
    # Add an image node
    image_result = mda.add_node(
        data_type="image",
        content={
            "path": "/images/diagram.png",
            "dimensions": [1920, 1080],
            "format": "PNG"
        },
        metadata={
            "name": "System Architecture Diagram",
            "tags": ["diagram", "architecture", "technical"]
        }
    )
    image_id = image_result["node_id"]
    print(f"Created image node: {image_id}")
    
    # Add a code node
    code_result = mda.add_node(
        data_type="code",
        content={
            "language": "python",
            "code": "def process_data(data):\n    return data.transform()",
            "file_path": "/src/data_processor.py"
        },
        metadata={
            "name": "Data Processing Function",
            "tags": ["code", "python", "data-processing"]
        }
    )
    code_id = code_result["node_id"]
    print(f"Created code node: {code_id}")
    
    # Connect related nodes
    print("\nConnecting related nodes...")
    mda.update_node(
        node_id=document_id,
        updates={
            "connections": [image_id, code_id]
        }
    )
    print(f"Connected document to image and code nodes")
    
    # Create a cluster for project materials
    print("\nCreating a cluster for related nodes...")
    cluster_result = mda.create_cluster(
        name="Project Planning Materials",
        center=[50.0, 50.0, 50.0],
        radius=15.0,
        theme="corporate",
        metadata={
            "project_id": "PROJ-2025-007",
            "priority": "high"
        }
    )
    cluster_id = cluster_result["cluster_id"]
    print(f"Created cluster: {cluster_id} with {cluster_result['member_count']} members")
    
    # Create a path through the data
    print("\nCreating a guided path through the data...")
    path_result = mda.create_path(
        name="Project Overview Tour",
        nodes=[document_id, image_id, code_id],
        metadata={
            "description": "A tour of the key project planning materials",
            "duration": "5 minutes"
        }
    )
    path_id = path_result["path_id"]
    print(f"Created path: {path_id} with {path_result['node_count']} nodes")
    
    # Automatically organize a set of data items
    print("\nAutomatically organizing a larger dataset...")
    data_items = [
        {
            "type": "document",
            "name": "Meeting Minutes",
            "content": "Minutes from the project kickoff meeting...",
            "tags": ["meeting", "kickoff", "minutes"]
        },
        {
            "type": "spreadsheet",
            "name": "Budget Forecast",
            "content": {"cells": "...", "sheets": ["Q1", "Q2", "Q3", "Q4"]},
            "tags": ["budget", "finance", "forecast"]
        },
        {
            "type": "email",
            "name": "Client Feedback",
            "content": "Feedback from the client on the initial prototype...",
            "tags": ["feedback", "client", "communication"]
        },
        {
            "type": "document",
            "name": "Technical Specifications",
            "content": "Detailed technical specifications for the system...",
            "tags": ["technical", "specifications", "requirements"]
        },
        {
            "type": "image",
            "name": "User Interface Mockup",
            "content": {"path": "/images/ui_mockup.png"},
            "tags": ["ui", "design", "mockup"]
        }
    ]
    
    auto_result = mda.auto_organize_data(data_items)
    print(f"Auto-organized {auto_result['nodes_created']} nodes into {auto_result['clusters_created']} clusters")
    
    # Query the cognitive API for related information
    print("\nQuerying the cognitive API for related information...")
    cognitive_query = mda.query_cognitive_api({
        "type": "related_to",
        "node_id": document_id
    })
    
    print(f"Found {len(cognitive_query.get('direct_connections', []))} direct connections")
    print(f"Found {len(cognitive_query.get('cluster_connections', []))} cluster connections")
    print(f"Found {len(cognitive_query.get('spatial_connections', []))} spatial connections")
    
    # Export the entire state
    print("\nExporting the current state...")
    state = mda.export_state()
    print(f"Exported state with {len(state['nodes'])} nodes, {len(state['clusters'])} clusters, and {len(state['paths'])} paths")
    
    # Save to a file for demonstration
    with open("mnemonic_state.json", "w") as f:
        json.dump(state, f, indent=2)
    print(f"Saved state to mnemonic_state.json")
    
    # Search for content
    print("\nSearching for content...")
    search_results = mda.search_by_content("proposal")
    print(f"Found {search_results['count']} results for 'proposal'")
    
    # Print the first result
    if search_results['count'] > 0:
        first_result = search_results['results'][0]
        print(f"Top result: {first_result['metadata'].get('name', 'Unnamed')} (Score: {first_result['score']})")
    
    print("\nMnemonic Data Architecture demonstration complete!")


if __name__ == "__main__":
    main()
