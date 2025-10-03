# Negative Space Imaging Project

## Project Overview
This project develops a novel system for analyzing and utilizing the negative space in 3D reconstructions from 2D images. The core concept involves extracting unique spatial signatures from the "empty" regions between objects, creating unforgeable data patterns that can be integrated with blockchain technology for various applications including encryption, authentication, and unique token generation.

## Project Phases

### Phase 1: Core Implementation ✅
- Point cloud generation with negative space detection
- Spatial signature extraction algorithms
- Interstitial space analysis between objects
- Basic visualizations and demos

### Phase 2: Temporal & Blockchain Integration ✅
- Temporal analysis of negative spaces over time
- Blockchain integration for secure signature storage
- Enhanced spatial signatures with cryptographic features
- Authentication token generation

### Phase 3: Advanced Applications 🔄
- Real-time negative space tracking ✅
- Advanced visualization with AR capabilities ✅
- Smart contract integration for blockchain verification ✅
- Multi-signature authentication ✅
- Mobile applications for field verification 🔄

#### Current Status ✅
- Real-time tracking framework implemented
- Webcam integration with point cloud generation
- Multiple analysis modes (continuous, interval, adaptive)
- Performance metrics and optimization
- Advanced visualization with AR overlays
- Smart contract integration for blockchain verification
- Authentication system with blockchain validation
- Multi-signature authentication with threshold and hierarchical verification

## Key Concepts

### Negative Space Analysis
Instead of focusing solely on the objects in an image, this project analyzes the empty spaces between them, extracting unique spatial signatures that are impossible to forge.

### Interstitial Negative Space
The project has evolved to incorporate multiple reference objects, creating "interstitial negative space" - the void between multiple known points - which generates exponentially more complex and unique spatial relationships.

### Temporal Variant Analysis
By tracking how negative space configurations change over time, the system can generate time-variant spatial signatures that add another dimension of uniqueness and security.

## Installation

### Prerequisites
- Python 3.8 or higher
- NumPy, Matplotlib for basic functionality
- OpenCV for image processing (optional)
- Open3D for 3D visualization and point cloud processing (optional)

### Setup
1. Clone the repository
2. Install the required packages:
   ```
   pip install numpy matplotlib
   ```
3. For full functionality (if available for your system):
   ```
   pip install opencv-python open3d scipy scikit-learn
   ```

## Running the Demos

The project includes several types of demos to showcase different aspects of the system:

### Core Demos (Phase 1)

#### Simplified Demo (No 3D dependencies)
This demo works without OpenCV or Open3D and provides 2D visualizations of the negative space concepts. It's perfect for getting started and understanding core concepts without installing all dependencies.

```
python simplified_demo.py --demo_type basic
```

Or to generate spatial signatures:

```
python simplified_demo.py --demo_type signature
```

#### Full Demos (Requires all dependencies)
If you have all the required packages including Open3D, you can run the more advanced demos:
```
python negative_space_demo.py --demo_type basic
python negative_space_demo.py --demo_type interstitial
python negative_space_demo.py --demo_type void_mesh
```

### Advanced Demos (Phase 2)

#### Temporal Analysis Demo
Demonstrates how negative spaces change over time:

```
python temporal_demo.py --num_frames 10
```

#### Blockchain Integration Demo
Shows how signatures can be securely stored and verified:

```
python blockchain_demo.py --num_signatures 5
```

#### Real-Time Analysis Demo (Phase 3)
Demonstrates real-time tracking of negative spaces using webcam input:

```
python realtime_demo.py --mode webcam
```

Or with a pre-recorded video:

```
python realtime_demo.py --mode video --video path/to/video.mp4
```

Or with synthetic data (no camera required):

```
python realtime_demo.py --mode synthetic
```

#### Enhanced Real-Time Visualization (Phase 3)
Advanced visualization with AR overlays and interactive controls:

```
python enhanced_realtime_demo.py --mode synthetic --viz advanced_2d
```

For 3D visualization (if Open3D is available):

```
python enhanced_realtime_demo.py --mode synthetic --viz basic_3d
```

For AR visualization mode:

```
python enhanced_realtime_demo.py --mode webcam --viz ar
```

## Installation and Setup

### System Requirements
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. Clone the repository:
   ```
   git clone [repository-url]
   cd negative-space-project
   ```

2. Run the setup script:
   ```
   python setup_project.py
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`

4. Verify basic installation:
   ```
   python -c "import numpy; import matplotlib; print('NumPy and Matplotlib are available')"
   ```

### Troubleshooting Package Installation

#### Open3D Installation Issues
If Open3D fails to install or import:

1. Try installing a specific version:
   ```
   pip install open3d==0.15.1
   ```

2. Check if your system meets the requirements:
   ```
   python check_open3d.py
   ```

3. Use the simplified demo instead:
   ```
   python simplified_demo.py
   ```

4. For Windows users, ensure you have:
   - Visual C++ Redistributable installed
   - Updated graphics drivers

#### Alternative Approaches
If you continue having issues with Open3D:

1. Use Docker:
   ```
   docker pull intel/open3d-ubuntu:latest
   ```

2. Use the fallback mechanism built into the project:
   ```
   # The utility in src/utils/open3d_support.py provides fallbacks
   from src.utils.open3d_support import o3d, np
   ```

3. Focus on the simplified demos that don't require Open3D

## Project Structure - Current Implementation

```
negative-space-project/
├── src/
│   ├── acquisition/                   # Image acquisition (webcam, file)
│   ├── reconstruction/                # 3D reconstruction core components
│   │   ├── point_cloud_generator.py   # Creates 3D point clouds
│   │   ├── interstitial_analyzer.py   # Analyzes spaces between objects
│   │   ├── model_assembler.py         # Assembles 3D models with voids
│   │   └── feature_detector.py        # Detects features in images
│   ├── temporal_variants/             # Phase 2: Temporal analysis
│   │   └── negative_space_tracker.py  # Tracks changes in negative spaces
│   ├── blockchain/                    # Phase 2: Blockchain integration
│   │   └── blockchain_integration.py  # Secure signature storage
│   ├── realtime/                      # Phase 3: Real-time analysis
│   │   ├── real_time_tracker.py       # Real-time tracking framework
│   │   └── webcam_integration.py      # Webcam and camera integration
│   └── utils/
│       ├── fallbacks.py               # Centralized fallback mechanisms
│       ├── open3d_support.py          # Support for Open3D with fallbacks
│       └── type_hints.py              # Custom type hints
├── demos/
│   ├── negative_space_demo.py         # Full demo with Open3D 
│   ├── simplified_demo.py             # 2D demo without Open3D dependencies
│   ├── temporal_demo.py               # Demo for temporal analysis
│   ├── blockchain_demo.py             # Demo for blockchain integration
│   └── realtime_demo.py               # Demo for real-time analysis
├── model_assembler_demo.py            # Demo for the complete pipeline
├── setup_project.py                   # Project setup utility
├── check_open3d.py                    # Test script for Open3D
├── requirements.txt                   # Package dependencies
├── PHASE1_README.md                   # Phase 1 detailed documentation
├── PHASE2_README.md                   # Phase 2 detailed documentation
├── PHASE3_ROADMAP.md                  # Phase 3 roadmap and features
└── README.md                          # This documentation
```

## Usage

### Basic Example

```python
from src.utils.open3d_support import o3d, np  # Safe import with fallbacks
from src.reconstruction.point_cloud_generator import PointCloudGenerator, PointCloudType

# Create a point cloud generator
generator = PointCloudGenerator(
    cloud_type=PointCloudType.NEGATIVE_SPACE_OPTIMIZED
)

# Generate and process a point cloud
point_cloud = generator.generate_from_image("data/sample.jpg")
point_cloud.classify_points()
point_cloud.generate_void_points()

# Get the spatial signature
signature = point_cloud.compute_spatial_signature()

# Visualize (works with or without Open3D)
point_cloud.visualize()
```

### Using Simplified Components

If you're having issues with Open3D, you can use the simplified components:

```python
from simplified_demo import SimplePointCloud, generate_test_scene

# Generate a test scene
point_cloud = generate_test_scene()

# Compute spatial signature
signature = point_cloud.compute_spatial_signature()

# Visualize in 2D
point_cloud.visualize("output/visualization.png")
```

## Demo Outputs and What to Expect

### Simplified Demo Output

When you run the simplified demo, you'll see:

1. **Basic Demo:**
   - Console output showing point cloud generation and classification
   - Three 2D projections of the scene (XY, XZ, YZ planes)
   - Color-coded points: blue (objects), red (voids), green (boundaries)
   - Output saved as PNG files in the output directory

2. **Signature Demo:**
   - Console output showing signature computation
   - 2D projections of the scene as in the basic demo
   - Bar chart visualization of the spatial signature (32 features)
   - Both visualizations and numerical data (CSV) saved to the output directory

### Full Demo Output (with Open3D)

When Open3D is available and you run the full demos:

1. **Basic Demo:**
   - Console output with detailed point statistics
   - Interactive 3D visualization window (if requested)
   - Color-coded 3D point cloud saved as PLY file

2. **Interstitial Demo:**
   - Advanced analysis of spaces between objects
   - Visualization of interstitial zones with heat map coloring
   - Metrics about void space distribution

3. **Void Mesh Demo:**
   - 3D mesh representation of the negative spaces
   - Interactive visualization with ability to rotate and zoom
   - STL/OBJ files for further processing or 3D printing

## Development Workflow

1. Start with detailed comments describing your algorithm's intent
2. Let Copilot generate implementation code based on your descriptions
3. Refine and customize the generated code to match your unique approach
4. Generate unit tests with Copilot to validate functionality
5. Use Copilot Chat within VS Code to explain and debug complex algorithms

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project builds upon concepts from computer vision, photogrammetry, and spatial analysis
- Special thanks to all contributors and supporters of the project
