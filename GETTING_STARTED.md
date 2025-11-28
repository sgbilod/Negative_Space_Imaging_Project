# Getting Started

**Copyright © 2025 Stephen Bilodeau. All rights reserved.**

This guide provides a step-by-step walkthrough for setting up and running the Negative Space Imaging System.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Local Development Setup](#local-development-setup)
- [Docker Development Setup](#docker-development-setup)
- [Running Tests](#running-tests)
- [First Project](#first-project)
- [Common Issues and Solutions](#common-issues-and-solutions)

---

## Prerequisites

Before you begin, ensure you have the following installed:

### Required Software

| Software | Version | Check Command |
|----------|---------|---------------|
| **Python** | 3.10+ | `python --version` |
| **Node.js** | 18.x+ | `node --version` |
| **npm** | 9.0.0+ | `npm --version` |
| **Git** | 2.x+ | `git --version` |

### Optional Software

| Software | Version | Purpose |
|----------|---------|---------|
| **PostgreSQL** | 15.x+ | Database (required for full features) |
| **Redis** | 6.x+ | Caching and sessions |
| **Docker** | 24.x+ | Containerized deployment |
| **NVIDIA CUDA** | 11.8+ | GPU acceleration |

### Hardware Recommendations

- **CPU**: 4+ cores
- **RAM**: 16 GB minimum (32 GB recommended)
- **Storage**: 100 GB SSD
- **GPU**: NVIDIA GTX 1060+ (optional, for acceleration)

---

## Quick Start

Get up and running in 5 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/sgbilod/Negative_Space_Imaging_Project.git
cd Negative_Space_Imaging_Project

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install Node.js dependencies
npm install

# 5. Copy environment configuration
cp .env.example .env

# 6. Verify installation
python environment_verification.py

# 7. Run the demo
python demo.py
```

---

## Local Development Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/sgbilod/Negative_Space_Imaging_Project.git
cd Negative_Space_Imaging_Project
```

### Step 2: Set Up Python Environment

#### Linux/macOS

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements.dev.txt
```

#### Windows (PowerShell)

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements.dev.txt
```

### Step 3: Set Up Node.js Environment

```bash
# Install all dependencies
npm install

# Or install only production dependencies
npm install --omit=dev
```

### Step 4: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
# Required variables:
# - DATABASE_URL: PostgreSQL connection string
# - JWT_SECRET: Secret key for authentication
# - ENCRYPTION_KEY: Key for data encryption
```

Example `.env` file:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/negative_space
REDIS_URL=redis://localhost:6379

# Security
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_EXPIRATION=24h
ENCRYPTION_KEY=your-32-byte-encryption-key-here

# Server
API_PORT=5000
FRONTEND_PORT=3000
NODE_ENV=development

# Processing
ENABLE_GPU=true
MAX_WORKERS=4
```

### Step 5: Initialize Database (Optional)

If using PostgreSQL:

```bash
# Option A: Using npm script
npm run db:init

# Option B: Using Python script
python setup_database.py --all

# Option C: Using shell scripts
./setup_database.sh --all  # Linux/macOS
setup_database.bat --all   # Windows
```

### Step 6: Verify Installation

```bash
# Run environment verification
python environment_verification.py

# This checks:
# - Python version and packages
# - Node.js and npm packages
# - Database connectivity
# - Redis connectivity (if configured)
# - GPU availability (if applicable)
```

### Step 7: Start Development Server

```bash
# Start both backend and frontend
npm run dev

# Or start them separately:
# Terminal 1 - Backend:
npm run server

# Terminal 2 - Frontend:
npm run client
```

Access the application:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **API Documentation**: http://localhost:5000/api-docs

---

## Docker Development Setup

### Prerequisites

- Docker 24.x or higher
- Docker Compose 2.x or higher

### Quick Docker Start

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps

# Stop all services
docker-compose down
```

### Available Docker Configurations

| File | Description |
|------|-------------|
| `docker-compose.yml` | Development environment (8 services) |
| `docker-compose.dev.yml` | Extended development with debugging |
| `docker-compose.prod.yml` | Production-ready configuration |
| `docker-compose.performance.yml` | Performance testing setup |

### Building Individual Images

```bash
# Build API image
docker build -f Dockerfile.api -t negative-space-api:latest .

# Build frontend image
docker build -f Dockerfile.frontend -t negative-space-frontend:latest .

# Build Python analyzer image
docker build -f Dockerfile.python -t negative-space-analyzer:latest .
```

### Using the Deploy Script

```bash
# Start all services
./scripts/docker-deploy.sh up

# Check health
./scripts/docker-deploy.sh health

# View logs
./scripts/docker-deploy.sh logs --follow

# Stop services
./scripts/docker-deploy.sh down
```

---

## Running Tests

### Python Tests

```bash
# Activate virtual environment first
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Run all Python tests
python test_suite.py --all

# Run specific test categories
python test_suite.py --unit
python test_suite.py --integration
python test_suite.py --security
python test_suite.py --performance

# Run with pytest
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py
```

### JavaScript/TypeScript Tests

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run in watch mode
npm run test:watch

# Run specific test file
npm test -- tests/controllers/auth.controller.test.ts
```

### End-to-End Tests

```bash
# Run Playwright tests
npm run test:e2e

# Run with UI
npm run test:e2e:ui

# Run headed (visible browser)
npm run test:e2e:headed

# Run specific browser
npm run test:e2e:chrome
npm run test:e2e:firefox
```

### Complete Test Suite

```bash
# Run all validation
npm run validate

# This runs:
# - Linting
# - Unit tests
# - E2E tests
# - Security scan
```

---

## First Project

### Example 1: Analyze a Local Image

```python
from negative_space_analysis import NegativeSpaceAnalyzer

# Initialize analyzer
analyzer = NegativeSpaceAnalyzer()

# Load image
image = analyzer.load_image("path/to/your/image.png")

# Detect negative space
negative_space_map = analyzer.detect_negative_space(image)

# Find anomalies
anomalies = analyzer.find_anomalies(negative_space_map)

# Generate report
report = analyzer.generate_report(anomalies)
print(report)
```

### Example 2: Acquire and Process Simulated Image

```python
from image_acquisition import ImageAcquisition, ImageFormat, AcquisitionMode

# Create acquisition system
acq = ImageAcquisition(
    format=ImageFormat.RAW,
    mode=AcquisitionMode.SIMULATION
)

# Acquire simulated image with negative space
image_data, metadata = acq.acquire(
    source="simulated_image",
    width=512,
    height=512,
    pattern="negative_space",
    negative_space_regions=5
)

# Save results
acq.save_image(image_data, "output/simulated.raw")
acq.save_metadata("output/metadata.json")

print(f"Image acquired: {metadata['size_bytes']} bytes")
print(f"Hash: {metadata['sha256_hash'][:16]}...")
```

### Example 3: Run Secure Workflow via CLI

```bash
# Complete secure imaging workflow
python cli.py workflow --mode threshold --signatures 5 --threshold 3

# Step by step:
# 1. Acquire image
python cli.py acquire --simulate --output image.raw

# 2. Process image
python cli.py process --input image.raw --output results.json

# 3. Verify with multi-signature
python cli.py verify --input results.json --mode threshold --signatures 5 --threshold 3

# 4. View audit logs
python cli.py audit --view --log security_audit.json
```

### Example 4: Run the Demo Scripts

```bash
# Main demonstration
python demo.py

# End-to-end workflow demo
python end_to_end_demo.py

# Image acquisition demo
python demo_acquisition.py

# Multi-signature verification demo
python multi_signature_demo.py --mode threshold --signatures 5 --threshold 3

# Hoag's Object analysis demo
python hoag_demo.py
```

---

## Common Issues and Solutions

### Issue 1: Python Version Mismatch

**Error**: `Python 3.x required, found 2.x`

**Solution**:
```bash
# Check Python version
python --version

# Use python3 explicitly if needed
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Issue 2: Node.js Version Too Old

**Error**: `Node.js 18.x required`

**Solution**:
```bash
# Using nvm (Node Version Manager)
nvm install 18
nvm use 18

# Verify
node --version
```

### Issue 3: Missing C++ Build Tools (Windows)

**Error**: `error: Microsoft Visual C++ 14.0 is required`

**Solution**:
1. Download Visual Studio Build Tools from Microsoft
2. Install "Desktop development with C++"
3. Restart terminal and retry `pip install`

### Issue 4: PostgreSQL Connection Failed

**Error**: `ECONNREFUSED` or `could not connect to server`

**Solution**:
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql  # Linux
brew services list                # macOS

# Start PostgreSQL
sudo systemctl start postgresql   # Linux
brew services start postgresql    # macOS

# Verify connection
psql -U postgres -h localhost
```

### Issue 5: CUDA Not Detected

**Error**: `torch.cuda.is_available() returns False`

**Solution**:
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 6: Permission Denied on Linux

**Error**: `Permission denied` when running scripts

**Solution**:
```bash
# Make scripts executable
chmod +x setup_database.sh
chmod +x scripts/*.sh

# For npm scripts
chmod +x node_modules/.bin/*
```

### Issue 7: Port Already in Use

**Error**: `EADDRINUSE: address already in use`

**Solution**:
```bash
# Find process using the port
lsof -i :3000  # macOS/Linux
netstat -ano | findstr :3000  # Windows

# Kill the process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or change port in .env
API_PORT=5001
FRONTEND_PORT=3001
```

### Issue 8: npm Install Fails

**Error**: Various npm errors

**Solution**:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json  # Linux/macOS
# Windows: rmdir /s /q node_modules & del package-lock.json
npm install

# Use legacy peer deps if needed
npm install --legacy-peer-deps
```

---

## Next Steps

Once you have the system running:

1. **Explore the Architecture**: See [ARCHITECTURE.md](./ARCHITECTURE.md)
2. **Learn the API**: See [API_REFERENCE.md](./API_REFERENCE.md)
3. **Understand Requirements**: See [REQUIREMENTS.md](./REQUIREMENTS.md)
4. **Contribute**: See [CONTRIBUTING.md](./CONTRIBUTING.md)
5. **Deploy to Production**: See [DOCKER_DEPLOYMENT_GUIDE.md](./DOCKER_DEPLOYMENT_GUIDE.md)

---

## Getting Help

- **Documentation**: Browse the `docs/` directory
- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Email**: support@negativespaceimaging.com

---

Last Updated: November 2025
