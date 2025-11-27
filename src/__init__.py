# =============================================================================
# Negative Space Imaging Project - Source Package
# =============================================================================
#
# Advanced medical/astronomical imaging platform with AI/ML integration.
#
# This package provides:
# - Agentic imaging pipeline orchestration (DeepAgent Supervisor)
# - Persistent memory with decay and LSH-based similarity search
# - JWT authentication with RS256 asymmetric signing
# - RESTful API with FastAPI
# - Edge and cloud deployment support
#
# Modules:
#   agents   - Agent orchestration and memory system
#   auth     - JWT authentication and authorization
#   api      - REST API routers
#   main     - Application entry point
#
# Quick Start:
#     from src.main import create_app
#     app = create_app()
#
# Or run from command line:
#     python -m src.main --mode dev --port 8080
#
# Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
# =============================================================================

__version__ = "1.0.0"
__author__ = "Stephen Bilodeau"
__license__ = "Proprietary"

__all__ = [
    "__version__",
    "__author__",
    "__license__",
]
