# Negative Space Imaging System

Copyright © 2025 Stephen Bilodeau. All rights reserved.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Build](https://img.shields.io/badge/build-passing-green.svg)
![Test Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)
![Security](https://img.shields.io/badge/security-HIPAA_compliant-brightgreen.svg)

## Overview

The Negative Space Imaging System is an advanced imaging platform designed for high-precision analysis of negative space in medical and astronomical imaging. This system leverages cutting-edge algorithms to detect patterns in what's not visibly present, providing unprecedented insights for medical diagnoses and astronomical discoveries.

## Key Features

- **High-Performance Image Processing**: Advanced algorithms for real-time negative space analysis
- **HIPAA Compliant Security**: End-to-end encryption and secure data handling
- **Multi-Signature Verification**: Ensures integrity and authenticity through cryptographic verification by multiple parties
- **AI-Powered Detection**: Machine learning models trained on extensive datasets
- **Scalable Architecture**: Handles everything from single images to massive datasets
- **Comprehensive Reporting**: Detailed analysis reports with visualization tools
- **Full Audit Trail**: Complete history of all system interactions

## Quick Start

### Prerequisites

- Node.js 18.x or higher
- PostgreSQL 15.x or higher
- TypeScript 5.x
- React 18.x
- Python 3.10 or higher
- Cryptography library for Python

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/negative-space-imaging.git

# Install JavaScript dependencies
npm install

# Setup environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
npm run db:init

# Or use our comprehensive database deployment system
python setup_database.py --all

# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Start development server
npm run dev
```

### Running Tests

```bash
# Run all tests
npm test

# Run with coverage report
npm run test:coverage

# Run Python-based tests
python test_suite.py --all

# Run specific test categories
python test_suite.py --unit
python test_suite.py --integration
python test_suite.py --security
python test_suite.py --performance

# Test database deployment
python deployment/test_database_deployment.py
```

### Database Setup and Management

The system includes a comprehensive database deployment and management system:

```bash
# Windows: Complete database setup
setup_database.bat --all

# Linux/macOS: Complete database setup
./setup_database.sh --all

# Deploy the database
python deployment/database_deploy.py --deploy

# Run database migrations
python deployment/database_deploy.py --migrate

# Backup the database
python deployment/database_deploy.py --backup

# Restore from a backup
python deployment/database_deploy.py --restore --backup-file deployment/database/backups/latest.sql
```

## Secure Imaging Workflow

The system provides a complete secure workflow for image acquisition, processing, and verification:

```bash
# Run the complete secure workflow
python cli.py workflow --mode threshold --signatures 5 --threshold 3

# Or run individual steps:
# 1. Acquire an image
python cli.py acquire --simulate --output image.raw

# 2. Process the image
python cli.py process --input image.raw --output results.json

# 3. Verify the processing results
python cli.py verify --input results.json --mode threshold --signatures 5 --threshold 3

# 4. View security audit logs
python cli.py audit --view --log security_audit.json
```

## Multi-Signature Verification

The system supports three different multi-signature verification modes:

1. **Threshold Mode**: Requires m-of-n signatures (e.g., 3 of 5 authorized signers)
2. **Sequential Mode**: Requires signatures in a specific sequence (e.g., analyst → supervisor → compliance officer)
3. **Role-Based Mode**: Requires signatures from specific roles (e.g., one from each: analyst, doctor, administrator)

Example usage:

```bash
# Threshold mode (3 of 5 signatures)
python multi_signature_demo.py --mode threshold --signatures 5 --threshold 3

# Sequential mode (3 signers in sequence)
python multi_signature_demo.py --mode sequential --signatures 3

# Role-based mode (requires specific roles)
python multi_signature_demo.py --mode role-based --roles analyst,physician,admin
```

## Architecture

The Negative Space Imaging System follows a modern microservices architecture:

- **Frontend**: React with TypeScript for type safety
- **Backend API**: Node.js with Express
- **Database**: PostgreSQL with advanced encryption
- **Image Processing**: Custom algorithms with GPU acceleration
- **Multi-Signature Verification**: Cryptographic verification system for processing integrity
- **Security Layer**: End-to-end encryption with access controls
- **Monitoring**: Comprehensive telemetry and alerting
- **Database Integration**: Scalable PostgreSQL database system for image and computation tracking
- **Multi-Signature Verification**: Cryptographic verification system for processing integrity
- **Security Layer**: End-to-end encryption with access controls
- **Monitoring**: Comprehensive telemetry and alerting

## Documentation

Detailed documentation is available in the `/docs` directory:

- [User Guide](./docs/user-guide.md)
- [API Reference](./docs/api-reference.md)
- [Security Model](./docs/security-model.md)
- [Multi-Signature Verification](./docs/multi-signature.md)
- [Development Guide](./docs/development-guide.md)
- [Deployment Guide](./docs/deployment-guide.md)
- [Database Deployment](./deployment/DATABASE_DEPLOYMENT.md)

## Security

This system is designed to be HIPAA compliant with:

- End-to-end encryption for all data
- Multi-signature verification for processing integrity
- Role-based access control
- Full audit logging
- Secure authentication with MFA
- Cryptographic verification of all processing results
- Regular security scans and penetration testing

For security issues, please see our [Security Policy](./SECURITY.md).

## Contributing

We welcome contributions! Please see our [Contributing Guide](./CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Support

For support, please open an issue or contact our support team at support@negativespacesystems.com.
