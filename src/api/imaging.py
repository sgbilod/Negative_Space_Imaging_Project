# =============================================================================
# Negative Space Imaging Project - Imaging Pipeline API
# =============================================================================
#
# REST API endpoints for imaging pipeline operations:
# - Pipeline submission and management
# - Job status tracking
# - Agent introspection
# - Memory system access
#
# Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
# =============================================================================

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/imaging", tags=["Imaging"])


# =============================================================================
# Enums and Constants
# =============================================================================

class PipelineType(str, Enum):
    """Available pipeline types."""
    FULL = "full_imaging_pipeline"
    ACQUISITION = "acquisition_only"
    RECONSTRUCTION = "reconstruction_only"
    ANALYSIS = "analysis_only"
    CUSTOM = "custom_pipeline"


class JobStatus(str, Enum):
    """Pipeline job statuses."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class Priority(str, Enum):
    """Job priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    STANDARD = "standard"
    BACKGROUND = "background"


# =============================================================================
# Request/Response Models
# =============================================================================

class AcquisitionConfig(BaseModel):
    """Configuration for image acquisition."""
    format: str = Field(default="DICOM", description="Image format")
    image_count: int = Field(default=1, ge=1, le=1000)
    resolution: Optional[str] = Field(default="1024x1024")
    bit_depth: int = Field(default=16, ge=8, le=32)
    compression: Optional[str] = Field(default=None)


class ReconstructionConfig(BaseModel):
    """Configuration for image reconstruction."""
    algorithm: str = Field(default="filtered_back_projection")
    iterations: int = Field(default=100, ge=1, le=10000)
    regularization: float = Field(default=0.01, ge=0.0, le=1.0)
    output_format: str = Field(default="numpy")


class AnalysisConfig(BaseModel):
    """Configuration for image analysis."""
    analysis_type: str = Field(default="anomaly_detection")
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    include_heatmap: bool = Field(default=True)
    include_segmentation: bool = Field(default=False)


class PipelineRequest(BaseModel):
    """Request model for pipeline submission."""
    pipeline_type: PipelineType = Field(
        default=PipelineType.FULL,
        description="Type of pipeline to execute"
    )
    acquisition: Optional[AcquisitionConfig] = Field(
        default_factory=AcquisitionConfig,
        description="Acquisition configuration"
    )
    reconstruction: Optional[ReconstructionConfig] = Field(
        default_factory=ReconstructionConfig,
        description="Reconstruction configuration"
    )
    analysis: Optional[AnalysisConfig] = Field(
        default_factory=AnalysisConfig,
        description="Analysis configuration"
    )
    priority: Priority = Field(
        default=Priority.STANDARD,
        description="Job priority"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to call when job completes"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata to attach to the job"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "pipeline_type": "full_imaging_pipeline",
                "acquisition": {"format": "DICOM", "image_count": 5},
                "reconstruction": {"algorithm": "filtered_back_projection"},
                "analysis": {"analysis_type": "anomaly_detection"},
                "priority": "standard"
            }
        }


class PipelineResponse(BaseModel):
    """Response model for pipeline submission."""
    job_id: str
    status: str
    message: str
    estimated_time_seconds: Optional[float] = None
    queue_position: Optional[int] = None


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    progress: float = Field(ge=0.0, le=100.0)
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentInfo(BaseModel):
    """Information about an agent."""
    agent_id: str
    capabilities: List[str]
    state: str
    performance_ms: Optional[float] = None
    tasks_completed: int = 0


class AgentListResponse(BaseModel):
    """Response model for agent listing."""
    agents: List[AgentInfo]
    total: int


class MemoryStatsResponse(BaseModel):
    """Response model for memory statistics."""
    total_entries: int
    entries_by_type: Dict[str, int]
    decay_enabled: bool
    average_relevance: float
    storage_path: str
    cache_hit_rate: Optional[float] = None


# =============================================================================
# Job Storage (In-Memory - Use Redis in Production)
# =============================================================================

class JobStore:
    """In-memory job store. Replace with Redis in production."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create(self, job_id: str, job_data: Dict[str, Any]) -> None:
        async with self._lock:
            self._jobs[job_id] = job_data

    async def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job_id: str, updates: Dict[str, Any]) -> None:
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(updates)

    async def delete(self, job_id: str) -> bool:
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

    async def list_all(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [
                {"job_id": k, **v}
                for k, v in self._jobs.items()
            ]


# Global job store
job_store = JobStore()


# =============================================================================
# Pipeline Execution
# =============================================================================

async def execute_pipeline(job_id: str, request: PipelineRequest) -> None:
    """
    Execute an imaging pipeline in the background.

    Args:
        job_id: Unique job identifier
        request: Pipeline configuration
    """
    try:
        from src.main import app_state

        await job_store.update(job_id, {
            "status": JobStatus.RUNNING.value,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })

        if not app_state.supervisor:
            raise RuntimeError("Agent supervisor not initialized")

        # Import task types
        from src.agents import AgentTask, TaskPriority

        # Map priority
        priority_map = {
            Priority.CRITICAL: TaskPriority.CRITICAL,
            Priority.HIGH: TaskPriority.HIGH,
            Priority.STANDARD: TaskPriority.STANDARD,
            Priority.BACKGROUND: TaskPriority.BACKGROUND,
        }

        # Create agent task
        task = AgentTask(
            task_id=job_id,
            task_type=request.pipeline_type.value,
            payload={
                "acquisition": request.acquisition.model_dump() if request.acquisition else {},
                "reconstruction": request.reconstruction.model_dump() if request.reconstruction else {},
                "analysis": request.analysis.model_dump() if request.analysis else {},
                "metadata": request.metadata or {},
            },
            priority=priority_map.get(request.priority, TaskPriority.STANDARD),
        )

        # Execute pipeline
        results = await app_state.supervisor.orchestrate_pipeline(task)

        # Format results
        formatted_results = {}
        for task_id, result in results.items():
            formatted_results[task_id] = {
                "success": result.success,
                "output": result.output,
                "execution_time_ms": getattr(result, 'execution_time_ms', None),
            }

        await job_store.update(job_id, {
            "status": JobStatus.COMPLETED.value,
            "progress": 100.0,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "results": formatted_results,
        })

        logger.info(f"Pipeline {job_id} completed successfully")

        # Call webhook if configured
        if request.callback_url:
            await notify_callback(request.callback_url, job_id, "completed", formatted_results)

    except asyncio.CancelledError:
        await job_store.update(job_id, {
            "status": JobStatus.CANCELLED.value,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(f"Pipeline {job_id} was cancelled")

    except Exception as e:
        error_msg = str(e)
        await job_store.update(job_id, {
            "status": JobStatus.FAILED.value,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "errors": [error_msg],
        })
        logger.error(f"Pipeline {job_id} failed: {e}")

        if request.callback_url:
            await notify_callback(request.callback_url, job_id, "failed", {"error": error_msg})


async def notify_callback(url: str, job_id: str, status: str, data: Dict[str, Any]) -> None:
    """Send webhook notification."""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            await client.post(
                url,
                json={
                    "job_id": job_id,
                    "status": status,
                    "data": data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                timeout=10.0,
            )
        logger.info(f"Callback sent to {url} for job {job_id}")
    except Exception as e:
        logger.warning(f"Failed to send callback to {url}: {e}")


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/pipeline",
    response_model=PipelineResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit imaging pipeline",
    description="Submit a new imaging pipeline job for background processing"
)
async def submit_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks,
) -> PipelineResponse:
    """
    Submit an imaging pipeline job.

    The job will be queued and executed asynchronously.
    Use the returned job_id to check status and retrieve results.
    """
    job_id = str(uuid.uuid4())

    # Create job record
    job_data = {
        "status": JobStatus.QUEUED.value,
        "progress": 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": request.model_dump(),
        "results": None,
        "errors": [],
    }

    await job_store.create(job_id, job_data)

    # Queue background execution
    background_tasks.add_task(execute_pipeline, job_id, request)

    logger.info(f"Pipeline job {job_id} submitted with type {request.pipeline_type.value}")

    return PipelineResponse(
        job_id=job_id,
        status="queued",
        message="Pipeline job submitted successfully",
        estimated_time_seconds=30.0,  # Placeholder
    )


@router.get(
    "/pipeline/{job_id}",
    response_model=JobStatusResponse,
    summary="Get pipeline status",
    description="Get the status and results of a pipeline job"
)
async def get_pipeline_status(job_id: str) -> JobStatusResponse:
    """Get the status of a pipeline job."""
    job = await job_store.get(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        results=job.get("results"),
        errors=job.get("errors") if job.get("errors") else None,
        metadata=job.get("request", {}).get("metadata"),
    )


@router.delete(
    "/pipeline/{job_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel pipeline job",
    description="Cancel a queued or running pipeline job"
)
async def cancel_pipeline(job_id: str) -> Dict[str, str]:
    """Cancel a pipeline job."""
    job = await job_store.get(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    current_status = job["status"]
    if current_status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in {current_status} state"
        )

    await job_store.update(job_id, {
        "status": JobStatus.CANCELLED.value,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })

    logger.info(f"Pipeline job {job_id} cancelled")

    return {"job_id": job_id, "status": "cancelled"}


@router.get(
    "/pipeline",
    summary="List pipeline jobs",
    description="List all pipeline jobs with optional filtering"
)
async def list_pipelines(
    status_filter: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """List all pipeline jobs."""
    jobs = await job_store.list_all()

    if status_filter:
        jobs = [j for j in jobs if j.get("status") == status_filter]

    return {
        "jobs": jobs[:limit],
        "total": len(jobs),
    }


@router.get(
    "/agents",
    response_model=AgentListResponse,
    summary="List agents",
    description="List all available imaging agents and their status"
)
async def list_agents() -> AgentListResponse:
    """List all available imaging agents."""
    try:
        from src.main import app_state

        if not app_state.supervisor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent supervisor not initialized"
            )

        agents = []
        for agent_id, agent in app_state.supervisor.agents.items():
            agent_info = AgentInfo(
                agent_id=agent_id,
                capabilities=getattr(agent, 'capabilities', []),
                state=agent.state.name if hasattr(agent, 'state') else "unknown",
                performance_ms=getattr(agent, 'get_average_performance', lambda: None)(),
                tasks_completed=getattr(agent, 'tasks_completed', 0),
            )
            agents.append(agent_info)

        return AgentListResponse(agents=agents, total=len(agents))

    except ImportError:
        return AgentListResponse(agents=[], total=0)


@router.get(
    "/agents/{agent_id}",
    response_model=AgentInfo,
    summary="Get agent details",
    description="Get detailed information about a specific agent"
)
async def get_agent(agent_id: str) -> AgentInfo:
    """Get details of a specific agent."""
    try:
        from src.main import app_state

        if not app_state.supervisor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent supervisor not initialized"
            )

        agent = app_state.supervisor.agents.get(agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )

        return AgentInfo(
            agent_id=agent_id,
            capabilities=getattr(agent, 'capabilities', []),
            state=agent.state.name if hasattr(agent, 'state') else "unknown",
            performance_ms=getattr(agent, 'get_average_performance', lambda: None)(),
            tasks_completed=getattr(agent, 'tasks_completed', 0),
        )

    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agents module not available"
        )


@router.get(
    "/memory/stats",
    response_model=MemoryStatsResponse,
    summary="Get memory statistics",
    description="Get statistics about the persistent memory system"
)
async def get_memory_stats() -> MemoryStatsResponse:
    """Get memory system statistics."""
    try:
        from src.main import app_state

        if not app_state.memory_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Memory manager not initialized"
            )

        stats = app_state.memory_manager.get_statistics()

        return MemoryStatsResponse(
            total_entries=stats.get("total_entries", 0),
            entries_by_type=stats.get("entries_by_type", {}),
            decay_enabled=stats.get("decay_enabled", False),
            average_relevance=stats.get("average_relevance", 0.0),
            storage_path=stats.get("storage_path", ""),
            cache_hit_rate=stats.get("cache_hit_rate"),
        )

    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory module not available"
        )


@router.post(
    "/memory/search",
    summary="Search memory",
    description="Search the memory system using similarity queries"
)
async def search_memory(
    query_vector: List[float],
    memory_type: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Search memory using vector similarity."""
    try:
        from src.main import app_state

        if not app_state.memory_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Memory manager not initialized"
            )

        results = await app_state.memory_manager.search_similar(
            query_vector=query_vector,
            memory_type=memory_type,
            limit=limit,
        )

        return {
            "results": [
                {
                    "entry_id": r.entry_id,
                    "memory_type": r.memory_type.value if hasattr(r.memory_type, 'value') else r.memory_type,
                    "relevance": r.relevance,
                    "similarity": getattr(r, 'similarity', None),
                }
                for r in results
            ],
            "total": len(results),
        }

    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory module not available"
        )
