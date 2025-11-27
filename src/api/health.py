# =============================================================================
# Negative Space Imaging Project - Health Check Endpoints
# =============================================================================
#
# Provides health, readiness, and liveness endpoints for:
# - Kubernetes probes
# - Load balancer health checks
# - Monitoring systems
#
# Endpoints:
#   GET /health   - Detailed health check with component status
#   GET /healthz  - Simple health check (Kubernetes style)
#   GET /ready    - Readiness probe
#   GET /live     - Liveness probe
#   GET /metrics  - Prometheus metrics endpoint
#
# Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
# =============================================================================

from __future__ import annotations

import logging
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

# Track startup time
_startup_time = time.time()


# =============================================================================
# Response Models
# =============================================================================

class ComponentStatus(BaseModel):
    """Status of a single component."""
    name: str
    status: str = Field(..., description="Status: up, down, degraded")
    latency_ms: Optional[float] = Field(None, description="Response latency in ms")
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Detailed health check response."""
    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    components: List[ComponentStatus] = Field(default_factory=list)
    system: Dict[str, Any] = Field(default_factory=dict)


class ReadinessResponse(BaseModel):
    """Readiness probe response."""
    ready: bool
    message: Optional[str] = None


class LivenessResponse(BaseModel):
    """Liveness probe response."""
    alive: bool
    timestamp: str


# =============================================================================
# Health Check Functions
# =============================================================================

async def check_memory_manager() -> ComponentStatus:
    """Check memory manager health."""
    try:
        from src.main import app_state

        if not app_state.memory_manager:
            return ComponentStatus(
                name="memory_manager",
                status="down",
                details={"reason": "Not initialized"}
            )

        start = time.time()
        # Quick health check - get stats
        stats = app_state.memory_manager.get_statistics()
        latency = (time.time() - start) * 1000

        return ComponentStatus(
            name="memory_manager",
            status="up",
            latency_ms=round(latency, 2),
            details={
                "total_entries": stats.get("total_entries", 0),
                "decay_enabled": stats.get("decay_enabled", False),
            }
        )
    except Exception as e:
        logger.warning(f"Memory manager health check failed: {e}")
        return ComponentStatus(
            name="memory_manager",
            status="down",
            details={"error": str(e)}
        )


async def check_supervisor() -> ComponentStatus:
    """Check agent supervisor health."""
    try:
        from src.main import app_state

        if not app_state.supervisor:
            return ComponentStatus(
                name="agent_supervisor",
                status="down",
                details={"reason": "Not initialized"}
            )

        start = time.time()
        agent_count = len(app_state.supervisor.agents)
        latency = (time.time() - start) * 1000

        # Check agent states
        healthy_agents = sum(
            1 for a in app_state.supervisor.agents.values()
            if hasattr(a, 'state') and a.state.name in ['IDLE', 'PROCESSING']
        )

        status = "up" if healthy_agents == agent_count else "degraded"

        return ComponentStatus(
            name="agent_supervisor",
            status=status,
            latency_ms=round(latency, 2),
            details={
                "total_agents": agent_count,
                "healthy_agents": healthy_agents,
            }
        )
    except Exception as e:
        logger.warning(f"Supervisor health check failed: {e}")
        return ComponentStatus(
            name="agent_supervisor",
            status="down",
            details={"error": str(e)}
        )


async def check_auth_manager() -> ComponentStatus:
    """Check authentication manager health."""
    try:
        from src.main import app_state

        if not app_state.auth_manager:
            return ComponentStatus(
                name="auth_manager",
                status="down",
                details={"reason": "Not initialized"}
            )

        start = time.time()
        # Quick verification - check if keys are loaded
        has_keys = (
            hasattr(app_state.auth_manager, '_private_key') and
            app_state.auth_manager._private_key is not None
        )
        latency = (time.time() - start) * 1000

        return ComponentStatus(
            name="auth_manager",
            status="up" if has_keys else "degraded",
            latency_ms=round(latency, 2),
            details={
                "keys_loaded": has_keys,
            }
        )
    except Exception as e:
        logger.warning(f"Auth manager health check failed: {e}")
        return ComponentStatus(
            name="auth_manager",
            status="down",
            details={"error": str(e)}
        )


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Detailed health check",
    description="Returns detailed health status of all system components"
)
async def health_check() -> HealthResponse:
    """
    Comprehensive health check endpoint.

    Returns status of all system components including:
    - Memory Manager
    - Agent Supervisor
    - Authentication Manager
    - System resources
    """
    # Check all components
    components = await asyncio.gather(
        check_memory_manager(),
        check_supervisor(),
        check_auth_manager(),
        return_exceptions=True
    )

    # Handle any exceptions
    component_list = []
    for c in components:
        if isinstance(c, Exception):
            component_list.append(ComponentStatus(
                name="unknown",
                status="error",
                details={"error": str(c)}
            ))
        else:
            component_list.append(c)

    # Determine overall status
    statuses = [c.status for c in component_list]
    if all(s == "up" for s in statuses):
        overall_status = "healthy"
    elif any(s == "down" for s in statuses):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    # Get version
    try:
        from src import __version__
        version = __version__
    except ImportError:
        version = "1.0.0"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=version,
        uptime_seconds=round(time.time() - _startup_time, 2),
        components=component_list,
        system={
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }
    )


# Need asyncio for gather
import asyncio


@router.get(
    "/healthz",
    summary="Simple health check",
    description="Kubernetes-style simple health check"
)
async def healthz():
    """Simple health check returning 200 if server is running."""
    return {"status": "ok"}


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Kubernetes readiness probe - checks if service is ready to accept traffic"
)
async def readiness_check() -> ReadinessResponse:
    """
    Readiness probe for Kubernetes.

    Returns ready=true when:
    - At least one critical component is initialized
    - Server is accepting connections
    """
    try:
        from src.main import app_state

        # Check if at least supervisor or memory manager is ready
        supervisor_ready = app_state.supervisor is not None
        memory_ready = app_state.memory_manager is not None

        if supervisor_ready or memory_ready:
            return ReadinessResponse(ready=True)
        else:
            return ReadinessResponse(
                ready=False,
                message="No components initialized yet"
            )
    except Exception as e:
        return ReadinessResponse(
            ready=False,
            message=f"Error checking readiness: {e}"
        )


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Kubernetes liveness probe - checks if process is alive"
)
async def liveness_check() -> LivenessResponse:
    """
    Liveness probe for Kubernetes.

    Always returns alive=true if the process is running.
    """
    return LivenessResponse(
        alive=True,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Prometheus-compatible metrics endpoint"
)
async def metrics(response: Response):
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    try:
        from src.main import app_state

        lines = []

        # Uptime metric
        uptime = time.time() - _startup_time
        lines.append(f"# HELP nsi_uptime_seconds Server uptime in seconds")
        lines.append(f"# TYPE nsi_uptime_seconds gauge")
        lines.append(f"nsi_uptime_seconds {uptime:.2f}")

        # Component status metrics
        lines.append(f"# HELP nsi_component_status Component status (1=up, 0=down)")
        lines.append(f"# TYPE nsi_component_status gauge")

        supervisor_status = 1 if app_state.supervisor else 0
        memory_status = 1 if app_state.memory_manager else 0
        auth_status = 1 if app_state.auth_manager else 0

        lines.append(f'nsi_component_status{{component="supervisor"}} {supervisor_status}')
        lines.append(f'nsi_component_status{{component="memory_manager"}} {memory_status}')
        lines.append(f'nsi_component_status{{component="auth_manager"}} {auth_status}')

        # Agent metrics
        if app_state.supervisor:
            agent_count = len(app_state.supervisor.agents)
            lines.append(f"# HELP nsi_agents_total Total number of agents")
            lines.append(f"# TYPE nsi_agents_total gauge")
            lines.append(f"nsi_agents_total {agent_count}")

        # Memory metrics
        if app_state.memory_manager:
            try:
                stats = app_state.memory_manager.get_statistics()
                entry_count = stats.get("total_entries", 0)
                lines.append(f"# HELP nsi_memory_entries_total Total memory entries")
                lines.append(f"# TYPE nsi_memory_entries_total gauge")
                lines.append(f"nsi_memory_entries_total {entry_count}")
            except Exception:
                pass

        response.headers["Content-Type"] = "text/plain; charset=utf-8"
        return "\n".join(lines) + "\n"

    except Exception as e:
        logger.error(f"Metrics generation failed: {e}")
        response.status_code = 500
        return f"# Error generating metrics: {e}\n"
