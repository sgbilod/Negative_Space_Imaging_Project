# Requirements

**Copyright © 2025 Stephen Bilodeau. All rights reserved.**

This document provides a comprehensive list of all requirements for the Negative Space Imaging System.

## Table of Contents

- [System Requirements](#system-requirements)
- [Python Dependencies](#python-dependencies)
- [Node.js Dependencies](#nodejs-dependencies)
- [Development vs Production](#development-vs-production)
- [Optional Dependencies](#optional-dependencies)

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows 10+, macOS 11+, Ubuntu 20.04+ |
| **CPU** | 4 cores (x86_64 or ARM64) |
| **RAM** | 16 GB |
| **Disk Space** | 100 GB SSD |
| **Python** | 3.10 or higher |
| **Node.js** | 18.x or higher |
| **npm** | 9.0.0 or higher |

### Recommended Requirements (Production)

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Ubuntu 22.04 LTS |
| **CPU** | 16+ cores |
| **RAM** | 64 GB |
| **Disk Space** | 500 GB NVMe SSD |
| **GPU** | NVIDIA RTX 3060 or better (CUDA 11.7+) |
| **Database** | PostgreSQL 15.x or higher |
| **Cache** | Redis 6.x or higher |

### Additional Services

| Service | Version | Purpose |
|---------|---------|---------|
| **PostgreSQL** | 15.x+ | Primary database |
| **Redis** | 6.x+ | Caching and sessions |
| **Docker** | 24.x+ | Containerization (optional) |
| **Kubernetes** | 1.27+ | Orchestration (optional) |

---

## Python Dependencies

### Core Dependencies

These packages are required for the core functionality:

```
numpy>=1.24.0              # Array processing and numerical operations
torch>=2.0.0               # Deep learning framework
torchvision>=0.15.0        # Computer vision models and utilities
matplotlib>=3.7.2          # Visualization and plotting
cryptography>=40.0.0       # Security and encryption
plotly>=5.13.0             # Interactive visualizations
scikit-learn>=1.3.0        # Machine learning utilities
```

### Neural Network & Transformers

```
transformers>=4.31.0       # Pre-trained transformer models
sentence-transformers>=2.2.2  # Sentence embedding models
timm>=0.9.2                # PyTorch Image Models
```

### Imaging & Video Processing

```
pillow>=10.0.0             # Image processing
opencv-python>=4.8.0       # Computer vision operations
scikit-image>=0.21.0       # Image processing algorithms
albumentations>=1.3.1      # Image augmentations
```

### Audio Processing (Optional)

```
librosa>=0.10.0            # Audio analysis
soundfile>=0.12.1          # Audio file I/O
```

### Data Processing & Analysis

```
pandas>=2.0.3              # Data manipulation
scipy>=1.11.0              # Scientific computing
networkx>=3.1.0            # Graph algorithms
h5py>=3.9.0                # HDF5 file support
```

### Visualization

```
seaborn>=0.12.2            # Statistical visualizations
```

### Utilities

```
tqdm>=4.65.0               # Progress bars
pyyaml>=6.0.1              # YAML configuration
```

### Database & Web

```
sqlalchemy>=2.0.0          # Database ORM
Flask>=2.3.0               # Web framework
```

### Security

```
pyjwt>=2.6.0               # JSON Web Tokens
python-jose>=3.3.0         # JOSE implementation
```

### Documentation

```
sphinx>=6.1.0              # Documentation generator
mkdocs>=1.4.0              # Markdown documentation
```

### Build Tools

```
setuptools>=68.0.0         # Package setup
wheel>=0.40.0              # Package building
pip>=23.2.1                # Package management
```

---

## Node.js Dependencies

### Production Dependencies

```json
{
  "bcrypt": "^5.1.1",           // Password hashing
  "compression": "^1.8.1",      // Response compression
  "cors": "^2.8.5",             // Cross-origin resource sharing
  "dotenv": "^16.6.1",          // Environment variables
  "express": "^4.21.2",         // Web framework
  "express-rate-limit": "^7.5.1", // Rate limiting
  "helmet": "^7.2.0",           // Security headers
  "hpp": "^0.2.3",              // HTTP parameter pollution protection
  "joi": "^17.11.0",            // Input validation
  "jsonwebtoken": "^9.0.2",     // JWT authentication
  "morgan": "^1.10.1",          // HTTP request logging
  "pg": "^8.11.3",              // PostgreSQL client
  "redis": "^5.8.2",            // Redis client
  "sequelize": "^6.37.7",       // Database ORM
  "sharp": "^0.33.0",           // Image processing
  "swagger-jsdoc": "^6.2.8",    // API documentation
  "swagger-ui-express": "^5.0.1", // Swagger UI
  "uuid": "^11.1.0",            // UUID generation
  "winston": "^3.17.0",         // Logging
  "xss-clean": "^0.1.4",        // XSS protection
  "zod": "^3.22.4",             // Schema validation
  "react": "^18.2.0",           // React UI library
  "react-dom": "^18.2.0",       // React DOM bindings
  "react-router-dom": "^6.20.0" // React routing
}
```

### Development Dependencies

```json
{
  "@playwright/test": "^1.40.0",    // End-to-end testing
  "@types/express": "^4.17.21",     // Express type definitions
  "@types/jest": "^29.5.14",        // Jest type definitions
  "@types/node": "^20.19.11",       // Node.js type definitions
  "@types/react": "^18.2.39",       // React type definitions
  "@typescript-eslint/eslint-plugin": "^6.13.0", // TypeScript ESLint
  "@typescript-eslint/parser": "^6.13.0",        // TypeScript parser
  "eslint": "^8.54.0",              // JavaScript linting
  "eslint-config-prettier": "^9.0.0", // Prettier integration
  "husky": "^8.0.3",                // Git hooks
  "jest": "^29.7.0",                // Testing framework
  "lint-staged": "^15.1.0",         // Pre-commit linting
  "prettier": "^3.1.0",             // Code formatting
  "snyk": "^1.1240.0",              // Security scanning
  "supertest": "^6.3.3",            // HTTP testing
  "ts-jest": "^29.1.1",             // TypeScript Jest support
  "ts-node": "^10.9.1",             // TypeScript execution
  "ts-node-dev": "^2.0.0",          // Development server
  "typedoc": "^0.25.3",             // API documentation
  "typescript": "^5.3.2"            // TypeScript compiler
}
```

---

## Development vs Production

### Development Requirements

For local development, install additional dependencies:

**Python:**
```bash
pip install -r requirements.dev.txt
```

Includes:
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting
- `black>=23.7.0` - Code formatting
- `flake8>=6.1.0` - Linting
- `mypy>=1.4.1` - Type checking
- `isort>=5.12.0` - Import sorting
- `tensorboard>=2.13.0` - Experiment tracking
- `wandb>=0.15.5` - Experiment monitoring

**Node.js:**
```bash
npm install --include=dev
```

### Production Requirements

For production deployment:

1. Use only production dependencies:
   ```bash
   npm install --omit=dev
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   NODE_ENV=production
   ```

3. Enable production optimizations in Python:
   ```bash
   PYTHONOPTIMIZE=2
   ```

---

## Optional Dependencies

### GPU Acceleration (CUDA)

For NVIDIA GPU support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Requirements:
- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA Toolkit 11.8+
- cuDNN 8.6+

### HPC/Distributed Computing

For high-performance computing:

```
ray>=2.6.1                 # Distributed computing
dask>=2023.6.0             # Parallel computing
mpi4py>=3.1.4              # MPI bindings
pytorch-lightning>=2.0.6   # Structured training
```

### Medical Imaging (DICOM)

For DICOM image support:

```
pydicom>=2.4.0             # DICOM file handling
```

### Astronomical Data (FITS)

For FITS file support:

```
astropy>=5.3.0             # Astronomy utilities
```

### Remote Acquisition

For remote image acquisition:

```
requests>=2.31.0           # HTTP client
paramiko>=3.3.0            # SFTP support
```

---

## Installation Quick Reference

### Full Development Setup

```bash
# Clone repository
git clone https://github.com/sgbilod/Negative_Space_Imaging_Project.git
cd Negative_Space_Imaging_Project

# Python environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install -r requirements.dev.txt

# Node.js dependencies
npm install

# Verify installation
python environment_verification.py
npm run test
```

### Production Setup

```bash
# Python dependencies only
pip install -r requirements.txt

# Node.js production dependencies only
npm install --omit=dev

# Initialize database
npm run db:init

# Start application
npm run start
```

---

## Troubleshooting

### Common Issues

1. **Python version mismatch:**
   ```bash
   python --version  # Should be 3.10+
   ```

2. **Node.js version mismatch:**
   ```bash
   node --version  # Should be 18.x+
   ```

3. **Missing build tools (Windows):**
   - Install Visual C++ Build Tools

4. **CUDA not detected:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

5. **PostgreSQL connection issues:**
   - Verify DATABASE_URL in `.env`
   - Ensure PostgreSQL service is running

---

Last Updated: November 2025
