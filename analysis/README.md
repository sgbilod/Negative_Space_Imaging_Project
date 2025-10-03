# Analysis System for Negative Space Imaging Project

Author: Stephen Bilodeau
Date: August 13, 2025

## Overview

This package provides comprehensive data analysis capabilities for the Negative Space Imaging Project. It includes tools for neural network analysis, data visualization, and report generation.

## Features

- **Neural Network Analysis**: Train and evaluate neural network models on imaging data
- **Data Visualization**: Generate visualizations including PCA, t-SNE, UMAP, correlation matrices, and heatmaps
- **Report Generation**: Create PDF reports summarizing analysis results
- **CLI Integration**: Seamlessly integrates with the main Negative Space Imaging CLI

## Directory Structure

```
analysis/
├── __init__.py                 # Package initialization
├── cli_integration.py          # CLI integration module
├── config/                     # Configuration files
│   └── analysis_config.json    # Analysis configuration
├── demo_analysis.py            # Demo script for the analysis system
├── models/                     # Neural network models
│   ├── __init__.py
│   └── neural_network.py       # Neural network analysis module
├── reporting/                  # Report generation
│   ├── __init__.py
│   └── report_generator.py     # PDF report generator
└── visualizations/             # Data visualization
    ├── __init__.py
    └── visualization.py        # Visualization module
```

## Installation

The analysis system is integrated with the main Negative Space Imaging Project. No separate installation is required.

## Requirements

- Python 3.9+
- NumPy
- Matplotlib
- scikit-learn
- TensorFlow/Keras
- ReportLab (for PDF generation)
- seaborn
- umap-learn

## Usage

### CLI Usage

```bash
# Run neural network analysis
python cli.py analyze neural --input-dir data/processed --output-dir analysis_results --model-type dense

# Generate visualizations
python cli.py analyze visualize --input-dir data/processed --output-dir analysis/visualizations --type pca

# Generate a report
python cli.py analyze report --input-dir analysis_results --output-file analysis_results/report.pdf
```

### Programmatic Usage

```python
from analysis.models.neural_network import NeuralNetworkAnalyzer
from analysis.visualizations.visualization import DataVisualizer
from analysis.reporting.report_generator import ReportGenerator

# Neural network analysis
analyzer = NeuralNetworkAnalyzer()
results = analyzer.analyze(
    input_data=data,
    output_dir='analysis_results',
    model_type='dense',
    epochs=50
)

# Data visualization
visualizer = DataVisualizer()
viz_path = visualizer.generate_visualization(
    input_dir='data/processed',
    output_dir='analysis/visualizations',
    viz_type='pca'
)

# Report generation
generator = ReportGenerator()
report_path = generator.generate_report(
    input_dir='analysis_results',
    output_file='analysis_results/report.pdf',
    template='standard'
)
```

## Demo

Run the demonstration script to see the analysis system in action:

```bash
python analysis/demo_analysis.py --generate-data --samples 1000 --features 20
```

This will:
1. Generate synthetic test data
2. Run neural network analysis
3. Create visualizations
4. Generate a comprehensive report

## Integration with Other Modules

The analysis system integrates with other components of the Negative Space Imaging Project:

- **Image Processing**: Analyze processed image data from the core processing pipeline
- **Multi-signature Verification**: Analyze verification patterns and security metrics
- **High-Performance Computing**: Leverage HPC capabilities for large-scale analysis

## License

Copyright © 2025 Stephen Bilodeau. All rights reserved.
