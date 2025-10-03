# Negative Space Imaging Project - Phase 1 Implementation

## Project Overview

The Negative Space Imaging Project focuses on a novel approach to 3D scene analysis that emphasizes the empty spaces between objects rather than the objects themselves. This perspective shift allows us to extract unique spatial signatures and relationships that traditional object-focused analysis might miss.

## Phase 1: Acquisition and Reconstruction

Phase 1 implements the core functionality for acquiring images, preprocessing them, detecting features, generating point clouds, and analyzing the interstitial spaces between objects.

### Core Components

The project is structured as follows:

```
negative-space-project/
├── src/
│   ├── acquisition/
│   │   ├── camera_interface.py    # Camera connectivity and image capture
│   │   ├── image_preprocessor.py  # Image processing optimized for negative space
│   │   └── metadata_extractor.py  # EXIF and spatial metadata handling
│   ├── reconstruction/
│   │   ├── feature_detector.py       # Feature detection focused on negative space
│   │   ├── point_cloud_generator.py  # Point cloud generation with void space emphasis
│   │   ├── interstitial_analyzer.py  # Analysis of spaces between objects
│   │   └── model_assembler.py        # Assembly of complete 3D models
│   └── utils/
│       └── # Utility functions and classes
├── tests/
│   └── test_basic.py  # Basic unit tests
├── negative_space_demo.py        # Demo for negative space analysis
├── model_assembler_demo.py       # Demo for full pipeline
├── setup.py                      # Environment setup
└── README.md                     # Project documentation
```

## Key Innovations

### 1. Negative Space Focused Point Cloud Generation

Traditional point cloud generation focuses on object surfaces. Our implementation introduces specialized algorithms for detecting and characterizing the empty spaces between objects:

- **Void Space Detection**: Identifies empty regions within the scene and creates point representations
- **Boundary Classification**: Advanced algorithms to distinguish between object surfaces and void boundaries
- **Interstitial Space Mapping**: Creates relationships between empty spaces and adjacent objects

### 2. Interstitial Space Analysis

The `InterstitialAnalyzer` provides sophisticated analysis of the spaces between objects:

- **Region Identification**: Uses Voronoi tessellation, DBSCAN, or K-means clustering to identify distinct interstitial regions
- **Spatial Signatures**: Generates unique signatures for each interstitial region based on shape, volume, and adjacent objects
- **Adjacency Mapping**: Maps relationships between interstitial spaces and their neighboring objects

### 3. Complete Model Assembly

The `ModelAssembler` creates comprehensive 3D models that incorporate both objects and negative spaces:

- **Integrated Representation**: Combines object components and negative space components in a unified model
- **Volumetric Analysis**: Analyzes volumes and spatial relationships between components
- **Global Signatures**: Generates scene-level signatures that characterize the entire spatial arrangement

## Usage Examples

### Basic Negative Space Analysis

```python
# Create a point cloud with negative space optimization
generator = PointCloudGenerator(
    cloud_type=PointCloudType.NEGATIVE_SPACE_OPTIMIZED,
    params=PointCloudParams(
        point_density=2000,
        void_sampling_ratio=0.6
    )
)

# Generate a point cloud from input data
point_cloud = generator.generate_from_images(images)

# Classify points as objects, boundaries, or voids
point_cloud.classify_points()

# Compute spatial signature
signature = point_cloud.compute_spatial_signature()

# Visualize with negative space emphasis
point_cloud.visualize(show_classification=True)
```

### Full Pipeline with Interstitial Analysis

```python
# Generate or load a point cloud
point_cloud = load_point_cloud("input.ply")

# Analyze interstitial spaces
analyzer = InterstitialAnalyzer()
analyzer.set_object_points(point_cloud.object_points, point_cloud.object_labels)
analyzer.set_void_points(point_cloud.void_points)
regions = analyzer.analyze(method='voronoi')

# Assemble complete model
assembler = ModelAssembler()

# Add object components
for i, obj_points in enumerate(object_points):
    assembler.create_component_from_points(
        id=i, type=ComponentType.OBJECT, 
        points=obj_points, name=f"Object_{i}"
    )

# Add negative space components
for i, region in enumerate(analyzer.regions):
    ns_comp = assembler.create_component_from_points(
        id=i+100, type=ComponentType.NEGATIVE_SPACE, 
        points=region.points, name=f"NegativeSpace_{i}"
    )
    ns_comp.adjacent_objects = region.adjacent_objects

# Assemble and save the model
assembler.assemble()
assembler.save("output/assembled_model")
```

## Demo Applications

The project includes two comprehensive demos:

1. **negative_space_demo.py**: Demonstrates the core negative space analysis features:
   - Basic point cloud generation
   - Interstitial space analysis
   - Spatial signature generation
   - Void mesh visualization

2. **model_assembler_demo.py**: Demonstrates the complete pipeline:
   - Generates a test scene with multiple objects
   - Analyzes interstitial spaces between objects
   - Assembles a complete model with both objects and negative spaces
   - Visualizes spatial signatures and adjacency relationships

## Running the Demos

```bash
# Basic negative space demo
python negative_space_demo.py --demo_type interstitial --show_visualizations

# Complete pipeline demo
python model_assembler_demo.py --interstitial_method voronoi --show_vis
```

## Dependencies

- Python 3.8+
- OpenCV
- NumPy
- Open3D
- SciPy
- Matplotlib
- scikit-learn

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Next Steps

Phase 1 implementation lays the foundation for the subsequent phases:

- **Phase 2**: Advanced algorithms for temporal analysis, movement through negative spaces
- **Phase 3**: Application of negative space analysis to specific domains like architecture, urban planning, and virtual environments

## Contributors

- Project initiated with GitHub Copilot assistance
- Implemented by the Negative Space Imaging Team
