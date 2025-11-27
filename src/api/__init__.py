# =============================================================================
# Negative Space Imaging Project - API Package
# =============================================================================
#
# FastAPI routers for the NSIP REST API.
#
# Routers:
#   health_router  - Health, readiness, and liveness probes
#   imaging_router - Imaging pipeline and agent management
#
# Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
# =============================================================================

from .health import router as health_router
from .imaging import router as imaging_router

__all__ = [
    "health_router",
    "imaging_router",
]

__version__ = "1.0.0"
