"""
DeepAgent Supervisor - Hierarchical Task Orchestration System

Implements autonomous agent coordination for the Negative Space Imaging Project.
Provides task decomposition, dependency tracking, performance-based routing,
and parallel execution with retry logic.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# =============================================================================
# Enums
# =============================================================================


class AgentState(Enum):
    """Represents the current operational state of an agent."""
    IDLE = auto()
    PROCESSING = auto()
    AWAITING_SUBTASK = auto()
    COMPLETED = auto()
    FAILED = auto()
    RECOVERING = auto()


class TaskPriority(Enum):
    """Task priority levels for scheduling and resource allocation."""
    CRITICAL = 1
    HIGH = 2
    STANDARD = 3
    BACKGROUND = 4

    def __lt__(self, other: TaskPriority) -> bool:
        if isinstance(other, TaskPriority):
            return self.value < other.value
        return NotImplemented


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AgentTask:
    """
    Represents a unit of work for an agent.

    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task (e.g., 'dicom_import', 'ct_reconstruction')
        payload: Task-specific data and parameters
        priority: Scheduling priority
        parent_task_id: ID of parent task if this is a subtask
        created_at: Task creation timestamp
        deadline: Optional deadline for task completion
        retry_count: Current retry attempt number
        max_retries: Maximum allowed retry attempts
        metadata: Additional task metadata
    """
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.STANDARD
    parent_task_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.STANDARD,
        parent_task_id: Optional[str] = None,
        deadline: Optional[datetime] = None,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentTask:
        """Factory method to create a new task with auto-generated ID."""
        return cls(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            payload=payload,
            priority=priority,
            parent_task_id=parent_task_id,
            created_at=datetime.now(),
            deadline=deadline,
            max_retries=max_retries,
            metadata=metadata or {}
        )

    def is_expired(self) -> bool:
        """Check if task has passed its deadline."""
        if self.deadline is None:
            return False
        return datetime.now() > self.deadline

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries

    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.retry_count += 1


@dataclass
class AgentResult:
    """
    Represents the result of agent task execution.

    Attributes:
        task_id: ID of the completed task
        success: Whether execution succeeded
        output: Task output data
        execution_time_ms: Time taken in milliseconds
        agent_id: ID of the agent that executed the task
        errors: List of errors encountered
        warnings: List of warnings generated
        metrics: Performance and diagnostic metrics
    """
    task_id: str
    success: bool
    output: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0
    agent_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls,
        task_id: str,
        output: Dict[str, Any],
        execution_time_ms: int,
        agent_id: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """Create a successful result."""
        return cls(
            task_id=task_id,
            success=True,
            output=output,
            execution_time_ms=execution_time_ms,
            agent_id=agent_id,
            metrics=metrics or {}
        )

    @classmethod
    def failure_result(
        cls,
        task_id: str,
        errors: List[str],
        execution_time_ms: int,
        agent_id: str,
        warnings: Optional[List[str]] = None
    ) -> AgentResult:
        """Create a failure result."""
        return cls(
            task_id=task_id,
            success=False,
            errors=errors,
            execution_time_ms=execution_time_ms,
            agent_id=agent_id,
            warnings=warnings or []
        )


# =============================================================================
# Abstract Base Agent
# =============================================================================


class ImagingAgent(ABC):
    """
    Abstract base class for all imaging agents.

    Provides common interface for task execution, capability reporting,
    and performance tracking.
    """

    def __init__(self, agent_id: Optional[str] = None) -> None:
        """
        Initialize the imaging agent.

        Args:
            agent_id: Unique identifier for this agent instance
        """
        self._agent_id: str = agent_id or str(uuid.uuid4())
        self._capabilities: Set[str] = set()
        self._state: AgentState = AgentState.IDLE
        self._current_task: Optional[AgentTask] = None
        self._performance_history: List[Dict[str, Any]] = []
        self._max_history_size: int = 100

    @property
    def agent_id(self) -> str:
        """Get the agent's unique identifier."""
        return self._agent_id

    @property
    def capabilities(self) -> Set[str]:
        """Get the set of task types this agent can handle."""
        return self._capabilities

    @property
    def state(self) -> AgentState:
        """Get the current agent state."""
        return self._state

    @state.setter
    def state(self, new_state: AgentState) -> None:
        """Set the agent state with logging."""
        old_state = self._state
        self._state = new_state
        logger.debug(
            f"Agent {self._agent_id}: state transition "
            f"{old_state.name} -> {new_state.name}"
        )

    @property
    def current_task(self) -> Optional[AgentTask]:
        """Get the currently executing task."""
        return self._current_task

    @property
    def performance_history(self) -> List[Dict[str, Any]]:
        """Get the performance history."""
        return self._performance_history.copy()

    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute a task.

        Args:
            task: The task to execute

        Returns:
            AgentResult with execution outcome
        """
        pass

    @abstractmethod
    def can_handle(self, task_type: str) -> bool:
        """
        Check if this agent can handle a specific task type.

        Args:
            task_type: The type of task to check

        Returns:
            True if the agent can handle this task type
        """
        pass

    def get_average_performance(self, task_type: Optional[str] = None) -> float:
        """
        Calculate average execution time for completed tasks.

        Args:
            task_type: Optional filter by task type

        Returns:
            Average execution time in milliseconds, or -1 if no data
        """
        if not self._performance_history:
            return -1.0

        filtered = self._performance_history
        if task_type:
            filtered = [
                p for p in filtered
                if p.get("task_type") == task_type
            ]

        if not filtered:
            return -1.0

        times = [p.get("execution_time_ms", 0) for p in filtered]
        return sum(times) / len(times)

    def _record_performance(
        self,
        task: AgentTask,
        result: AgentResult
    ) -> None:
        """Record task performance for future routing decisions."""
        record = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "execution_time_ms": result.execution_time_ms,
            "success": result.success,
            "timestamp": datetime.now().isoformat()
        }
        self._performance_history.append(record)

        # Trim history if needed
        if len(self._performance_history) > self._max_history_size:
            self._performance_history = self._performance_history[-self._max_history_size:]

    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return self._state == AgentState.IDLE


# =============================================================================
# Specialized Agents
# =============================================================================


class AcquisitionAgent(ImagingAgent):
    """
    Agent specialized in data acquisition tasks.

    Handles:
    - DICOM file import
    - FITS file import
    - Raw sensor data capture
    - Batch acquisition workflows
    """

    def __init__(self, agent_id: Optional[str] = None) -> None:
        super().__init__(agent_id)
        self._capabilities = {
            "dicom_import",
            "fits_import",
            "raw_sensor_capture",
            "batch_acquisition"
        }
        logger.info(f"AcquisitionAgent {self._agent_id} initialized")

    def can_handle(self, task_type: str) -> bool:
        """Check if this agent can handle the task type."""
        return task_type in self._capabilities

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute an acquisition task."""
        start_time = time.time()
        self.state = AgentState.PROCESSING
        self._current_task = task

        logger.info(f"AcquisitionAgent executing: {task.task_type}")

        try:
            # Simulate task execution based on type
            if task.task_type == "dicom_import":
                output = await self._execute_dicom_import(task)
            elif task.task_type == "fits_import":
                output = await self._execute_fits_import(task)
            elif task.task_type == "raw_sensor_capture":
                output = await self._execute_raw_capture(task)
            elif task.task_type == "batch_acquisition":
                output = await self._execute_batch_acquisition(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            execution_time = int((time.time() - start_time) * 1000)

            result = AgentResult.success_result(
                task_id=task.task_id,
                output=output,
                execution_time_ms=execution_time,
                agent_id=self._agent_id,
                metrics={
                    "data_size_bytes": output.get("data_size", 0),
                    "items_processed": output.get("items_count", 1)
                }
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"AcquisitionAgent error: {e}")
            result = AgentResult.failure_result(
                task_id=task.task_id,
                errors=[str(e)],
                execution_time_ms=execution_time,
                agent_id=self._agent_id
            )

        finally:
            self._record_performance(task, result)
            self._current_task = None
            self.state = AgentState.IDLE

        return result

    async def _execute_dicom_import(self, task: AgentTask) -> Dict[str, Any]:
        """Execute DICOM import task."""
        # Simulate DICOM processing
        await asyncio.sleep(0.1)
        file_path = task.payload.get("file_path", "")
        return {
            "status": "imported",
            "file_path": file_path,
            "modality": task.payload.get("modality", "CT"),
            "slices": task.payload.get("slice_count", 128),
            "data_size": 134217728,  # 128MB simulated
            "items_count": 1,
            "image_data_ref": f"mem://{task.task_id}/dicom"
        }

    async def _execute_fits_import(self, task: AgentTask) -> Dict[str, Any]:
        """Execute FITS import task."""
        await asyncio.sleep(0.1)
        return {
            "status": "imported",
            "file_path": task.payload.get("file_path", ""),
            "hdu_count": task.payload.get("hdu_count", 3),
            "dimensions": task.payload.get("dimensions", [4096, 4096]),
            "data_size": 67108864,  # 64MB simulated
            "items_count": 1,
            "image_data_ref": f"mem://{task.task_id}/fits"
        }

    async def _execute_raw_capture(self, task: AgentTask) -> Dict[str, Any]:
        """Execute raw sensor capture task."""
        await asyncio.sleep(0.2)
        return {
            "status": "captured",
            "sensor_id": task.payload.get("sensor_id", "sensor_001"),
            "exposure_ms": task.payload.get("exposure_ms", 100),
            "resolution": task.payload.get("resolution", [2048, 2048]),
            "data_size": 16777216,  # 16MB
            "items_count": 1,
            "image_data_ref": f"mem://{task.task_id}/raw"
        }

    async def _execute_batch_acquisition(self, task: AgentTask) -> Dict[str, Any]:
        """Execute batch acquisition task."""
        items = task.payload.get("items", [])
        await asyncio.sleep(0.05 * len(items))
        return {
            "status": "batch_complete",
            "items_processed": len(items),
            "items_count": len(items),
            "data_size": 8388608 * len(items),
            "results": [f"mem://{task.task_id}/batch/{i}" for i in range(len(items))]
        }


class ReconstructionAgent(ImagingAgent):
    """
    Agent specialized in image reconstruction tasks.

    Handles:
    - CT reconstruction
    - MRI reconstruction
    - Tomographic synthesis
    - 3D rendering
    """

    def __init__(self, agent_id: Optional[str] = None) -> None:
        super().__init__(agent_id)
        self._capabilities = {
            "ct_reconstruction",
            "mri_reconstruction",
            "tomographic_synthesis",
            "3d_rendering"
        }
        logger.info(f"ReconstructionAgent {self._agent_id} initialized")

    def can_handle(self, task_type: str) -> bool:
        """Check if this agent can handle the task type."""
        return task_type in self._capabilities

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute a reconstruction task."""
        start_time = time.time()
        self.state = AgentState.PROCESSING
        self._current_task = task

        logger.info(f"ReconstructionAgent executing: {task.task_type}")

        try:
            if task.task_type == "ct_reconstruction":
                output = await self._execute_ct_reconstruction(task)
            elif task.task_type == "mri_reconstruction":
                output = await self._execute_mri_reconstruction(task)
            elif task.task_type == "tomographic_synthesis":
                output = await self._execute_tomographic_synthesis(task)
            elif task.task_type == "3d_rendering":
                output = await self._execute_3d_rendering(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            execution_time = int((time.time() - start_time) * 1000)

            result = AgentResult.success_result(
                task_id=task.task_id,
                output=output,
                execution_time_ms=execution_time,
                agent_id=self._agent_id,
                metrics={
                    "voxels_processed": output.get("voxel_count", 0),
                    "reconstruction_quality": output.get("quality_score", 0.95)
                }
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"ReconstructionAgent error: {e}")
            result = AgentResult.failure_result(
                task_id=task.task_id,
                errors=[str(e)],
                execution_time_ms=execution_time,
                agent_id=self._agent_id
            )

        finally:
            self._record_performance(task, result)
            self._current_task = None
            self.state = AgentState.IDLE

        return result

    async def _execute_ct_reconstruction(self, task: AgentTask) -> Dict[str, Any]:
        """Execute CT reconstruction."""
        await asyncio.sleep(0.3)  # Simulate computation
        input_ref = task.payload.get("input_data_ref", "")
        return {
            "status": "reconstructed",
            "algorithm": task.payload.get("algorithm", "filtered_backprojection"),
            "resolution": task.payload.get("resolution", [512, 512, 512]),
            "voxel_count": 512 ** 3,
            "quality_score": 0.97,
            "reconstruction_ref": f"mem://{task.task_id}/ct_volume",
            "input_ref": input_ref
        }

    async def _execute_mri_reconstruction(self, task: AgentTask) -> Dict[str, Any]:
        """Execute MRI reconstruction."""
        await asyncio.sleep(0.4)
        return {
            "status": "reconstructed",
            "sequence": task.payload.get("sequence", "T1"),
            "resolution": task.payload.get("resolution", [256, 256, 128]),
            "voxel_count": 256 * 256 * 128,
            "quality_score": 0.94,
            "reconstruction_ref": f"mem://{task.task_id}/mri_volume"
        }

    async def _execute_tomographic_synthesis(self, task: AgentTask) -> Dict[str, Any]:
        """Execute tomographic synthesis."""
        await asyncio.sleep(0.5)
        return {
            "status": "synthesized",
            "projection_count": task.payload.get("projections", 360),
            "resolution": [1024, 1024, 1024],
            "voxel_count": 1024 ** 3,
            "quality_score": 0.92,
            "synthesis_ref": f"mem://{task.task_id}/tomo_volume"
        }

    async def _execute_3d_rendering(self, task: AgentTask) -> Dict[str, Any]:
        """Execute 3D rendering."""
        await asyncio.sleep(0.2)
        return {
            "status": "rendered",
            "render_mode": task.payload.get("mode", "volume"),
            "output_resolution": task.payload.get("output_resolution", [1920, 1080]),
            "frame_count": task.payload.get("frames", 1),
            "quality_score": 0.98,
            "render_ref": f"mem://{task.task_id}/render"
        }


class AnalysisAgent(ImagingAgent):
    """
    Agent specialized in image analysis tasks.

    Handles:
    - Anomaly detection
    - Image segmentation
    - Feature extraction
    - Classification
    - Spatial signature analysis (negative space)
    """

    def __init__(self, agent_id: Optional[str] = None) -> None:
        super().__init__(agent_id)
        self._capabilities = {
            "anomaly_detection",
            "segmentation",
            "feature_extraction",
            "classification",
            "spatial_signature_analysis"
        }
        logger.info(f"AnalysisAgent {self._agent_id} initialized")

    def can_handle(self, task_type: str) -> bool:
        """Check if this agent can handle the task type."""
        return task_type in self._capabilities

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute an analysis task."""
        start_time = time.time()
        self.state = AgentState.PROCESSING
        self._current_task = task

        logger.info(f"AnalysisAgent executing: {task.task_type}")

        try:
            if task.task_type == "anomaly_detection":
                output = await self._execute_anomaly_detection(task)
            elif task.task_type == "segmentation":
                output = await self._execute_segmentation(task)
            elif task.task_type == "feature_extraction":
                output = await self._execute_feature_extraction(task)
            elif task.task_type == "classification":
                output = await self._execute_classification(task)
            elif task.task_type == "spatial_signature_analysis":
                output = await self._execute_spatial_signature_analysis(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            execution_time = int((time.time() - start_time) * 1000)

            result = AgentResult.success_result(
                task_id=task.task_id,
                output=output,
                execution_time_ms=execution_time,
                agent_id=self._agent_id,
                metrics={
                    "confidence": output.get("confidence", 0.0),
                    "regions_analyzed": output.get("region_count", 0)
                }
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"AnalysisAgent error: {e}")
            result = AgentResult.failure_result(
                task_id=task.task_id,
                errors=[str(e)],
                execution_time_ms=execution_time,
                agent_id=self._agent_id
            )

        finally:
            self._record_performance(task, result)
            self._current_task = None
            self.state = AgentState.IDLE

        return result

    async def _execute_anomaly_detection(self, task: AgentTask) -> Dict[str, Any]:
        """Execute anomaly detection."""
        await asyncio.sleep(0.2)
        return {
            "status": "analyzed",
            "anomalies_found": 3,
            "anomaly_regions": [
                {"id": 1, "location": [128, 256, 64], "severity": 0.8},
                {"id": 2, "location": [300, 150, 90], "severity": 0.6},
                {"id": 3, "location": [450, 400, 32], "severity": 0.4}
            ],
            "confidence": 0.93,
            "region_count": 3,
            "analysis_ref": f"mem://{task.task_id}/anomalies"
        }

    async def _execute_segmentation(self, task: AgentTask) -> Dict[str, Any]:
        """Execute image segmentation."""
        await asyncio.sleep(0.3)
        return {
            "status": "segmented",
            "segments_count": 7,
            "segments": [
                {"id": i, "label": f"region_{i}", "volume_ratio": 0.14}
                for i in range(7)
            ],
            "confidence": 0.91,
            "region_count": 7,
            "segmentation_ref": f"mem://{task.task_id}/segments"
        }

    async def _execute_feature_extraction(self, task: AgentTask) -> Dict[str, Any]:
        """Execute feature extraction."""
        await asyncio.sleep(0.15)
        return {
            "status": "extracted",
            "feature_count": 256,
            "feature_vector": [0.1] * 256,  # Placeholder
            "feature_type": task.payload.get("feature_type", "deep_features"),
            "confidence": 0.96,
            "region_count": 1,
            "features_ref": f"mem://{task.task_id}/features"
        }

    async def _execute_classification(self, task: AgentTask) -> Dict[str, Any]:
        """Execute classification."""
        await asyncio.sleep(0.1)
        return {
            "status": "classified",
            "primary_class": "normal",
            "class_probabilities": {
                "normal": 0.87,
                "benign": 0.10,
                "malignant": 0.03
            },
            "confidence": 0.87,
            "region_count": 1,
            "classification_ref": f"mem://{task.task_id}/classification"
        }

    async def _execute_spatial_signature_analysis(self, task: AgentTask) -> Dict[str, Any]:
        """Execute spatial signature (negative space) analysis."""
        await asyncio.sleep(0.25)
        return {
            "status": "analyzed",
            "negative_space_ratio": 0.42,
            "spatial_signature": [0.1] * 128,  # 128-dim signature
            "signature_hash": f"sig_{task.task_id[:8]}",
            "boundary_complexity": 0.73,
            "symmetry_score": 0.81,
            "confidence": 0.94,
            "region_count": 15,
            "signature_ref": f"mem://{task.task_id}/signature"
        }


# =============================================================================
# DeepAgent Supervisor
# =============================================================================


class DeepAgentSupervisor:
    """
    Hierarchical task orchestration supervisor for imaging agents.

    Provides:
    - Agent registration and management
    - Task decomposition and dependency tracking
    - Performance-based routing
    - Parallel execution with retry logic
    - Pipeline orchestration
    - Results caching and status tracking
    """

    def __init__(self) -> None:
        """Initialize the DeepAgent Supervisor."""
        self._agents: Dict[str, ImagingAgent] = {}
        self._agent_by_capability: Dict[str, List[str]] = {}
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._task_dependencies: Dict[str, Set[str]] = {}
        self._completed_tasks: Dict[str, AgentResult] = {}
        self._active_tasks: Dict[str, AgentTask] = {}
        self._results_cache: Dict[str, AgentResult] = {}
        self._lock = asyncio.Lock()

        logger.info("DeepAgentSupervisor initialized")

    # -------------------------------------------------------------------------
    # Agent Registration
    # -------------------------------------------------------------------------

    def register_agent(self, agent: ImagingAgent) -> None:
        """
        Register an agent with the supervisor.

        Args:
            agent: The agent to register
        """
        self._agents[agent.agent_id] = agent

        # Index by capability
        for capability in agent.capabilities:
            if capability not in self._agent_by_capability:
                self._agent_by_capability[capability] = []
            self._agent_by_capability[capability].append(agent.agent_id)

        logger.info(
            f"Registered agent {agent.agent_id} with capabilities: "
            f"{agent.capabilities}"
        )

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.

        Args:
            agent_id: ID of the agent to unregister

        Returns:
            True if agent was found and unregistered
        """
        if agent_id not in self._agents:
            return False

        agent = self._agents.pop(agent_id)

        for capability in agent.capabilities:
            if capability in self._agent_by_capability:
                if agent_id in self._agent_by_capability[capability]:
                    self._agent_by_capability[capability].remove(agent_id)

        logger.info(f"Unregistered agent {agent_id}")
        return True

    def get_agent(self, agent_id: str) -> Optional[ImagingAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_available_agents(self) -> List[ImagingAgent]:
        """Get all agents that are currently available."""
        return [a for a in self._agents.values() if a.is_available()]

    # -------------------------------------------------------------------------
    # Task Decomposition
    # -------------------------------------------------------------------------

    def decompose_task(self, task: AgentTask) -> List[AgentTask]:
        """
        Decompose a complex task into subtasks.

        For 'full_imaging_pipeline', creates:
        1. Acquisition subtask
        2. Reconstruction subtask (depends on acquisition)
        3. Analysis subtask (depends on reconstruction)

        Args:
            task: The task to decompose

        Returns:
            List of subtasks
        """
        if task.task_type != "full_imaging_pipeline":
            return [task]

        subtasks = []
        payload = task.payload

        # Subtask 1: Acquisition
        acquisition_task = AgentTask.create(
            task_type=payload.get("acquisition_type", "dicom_import"),
            payload={
                "file_path": payload.get("input_path", ""),
                "modality": payload.get("modality", "CT"),
                **payload.get("acquisition_params", {})
            },
            priority=task.priority,
            parent_task_id=task.task_id,
            metadata={"stage": "acquisition", "pipeline_id": task.task_id}
        )
        subtasks.append(acquisition_task)

        # Subtask 2: Reconstruction (depends on acquisition)
        reconstruction_task = AgentTask.create(
            task_type=payload.get("reconstruction_type", "ct_reconstruction"),
            payload={
                "input_data_ref": f"{{result:{acquisition_task.task_id}:output:image_data_ref}}",
                "algorithm": payload.get("algorithm", "filtered_backprojection"),
                **payload.get("reconstruction_params", {})
            },
            priority=task.priority,
            parent_task_id=task.task_id,
            metadata={"stage": "reconstruction", "pipeline_id": task.task_id}
        )
        subtasks.append(reconstruction_task)

        # Subtask 3: Analysis (depends on reconstruction)
        analysis_task = AgentTask.create(
            task_type=payload.get("analysis_type", "spatial_signature_analysis"),
            payload={
                "input_data_ref": f"{{result:{reconstruction_task.task_id}:output:reconstruction_ref}}",
                **payload.get("analysis_params", {})
            },
            priority=task.priority,
            parent_task_id=task.task_id,
            metadata={"stage": "analysis", "pipeline_id": task.task_id}
        )
        subtasks.append(analysis_task)

        # Set up dependencies
        self._task_dependencies[reconstruction_task.task_id] = {acquisition_task.task_id}
        self._task_dependencies[analysis_task.task_id] = {reconstruction_task.task_id}

        logger.info(
            f"Decomposed pipeline task {task.task_id} into "
            f"{len(subtasks)} subtasks"
        )

        return subtasks

    # -------------------------------------------------------------------------
    # Performance-Based Routing
    # -------------------------------------------------------------------------

    def route_task(self, task: AgentTask) -> Optional[ImagingAgent]:
        """
        Route a task to the best available agent.

        Uses performance history to select the fastest agent
        that can handle the task type.

        Args:
            task: The task to route

        Returns:
            The best agent, or None if no agent available
        """
        capability = task.task_type

        if capability not in self._agent_by_capability:
            logger.warning(f"No agents registered for capability: {capability}")
            return None

        # Get available agents with this capability
        candidate_ids = self._agent_by_capability[capability]
        available = [
            self._agents[aid]
            for aid in candidate_ids
            if self._agents[aid].is_available()
        ]

        if not available:
            logger.warning(f"No available agents for capability: {capability}")
            return None

        # Sort by average performance (prefer faster agents)
        def perf_key(agent: ImagingAgent) -> float:
            avg = agent.get_average_performance(task.task_type)
            return avg if avg >= 0 else float('inf')

        available.sort(key=perf_key)

        selected = available[0]
        logger.debug(
            f"Routed task {task.task_id} ({task.task_type}) to "
            f"agent {selected.agent_id}"
        )

        return selected

    # -------------------------------------------------------------------------
    # Task Execution
    # -------------------------------------------------------------------------

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """
        Execute a single task with retry logic and exponential backoff.

        Args:
            task: The task to execute

        Returns:
            The execution result
        """
        async with self._lock:
            self._active_tasks[task.task_id] = task

        while task.can_retry():
            agent = self.route_task(task)

            if agent is None:
                # Wait and retry routing
                await asyncio.sleep(0.5)
                task.increment_retry()
                continue

            try:
                result = await agent.execute(task)

                if result.success:
                    async with self._lock:
                        self._completed_tasks[task.task_id] = result
                        self._results_cache[task.task_id] = result
                        self._active_tasks.pop(task.task_id, None)
                    return result

                # Task failed, retry with backoff
                task.increment_retry()
                if task.can_retry():
                    backoff = (2 ** task.retry_count) * 0.1
                    logger.warning(
                        f"Task {task.task_id} failed, retrying in {backoff}s "
                        f"(attempt {task.retry_count}/{task.max_retries})"
                    )
                    await asyncio.sleep(backoff)

            except Exception as e:
                logger.error(f"Execution error for task {task.task_id}: {e}")
                task.increment_retry()
                if task.can_retry():
                    backoff = (2 ** task.retry_count) * 0.1
                    await asyncio.sleep(backoff)

        # Max retries exceeded
        result = AgentResult.failure_result(
            task_id=task.task_id,
            errors=["Max retries exceeded"],
            execution_time_ms=0,
            agent_id=""
        )

        async with self._lock:
            self._completed_tasks[task.task_id] = result
            self._active_tasks.pop(task.task_id, None)

        return result

    # -------------------------------------------------------------------------
    # Pipeline Orchestration
    # -------------------------------------------------------------------------

    async def orchestrate_pipeline(
        self,
        pipeline_task: AgentTask
    ) -> Dict[str, AgentResult]:
        """
        Orchestrate a full pipeline, respecting dependencies.

        Decomposes the task, tracks dependencies, and executes
        ready tasks in parallel when possible.

        Args:
            pipeline_task: The pipeline task to orchestrate

        Returns:
            Dict mapping task_id to result for all subtasks
        """
        subtasks = self.decompose_task(pipeline_task)
        results: Dict[str, AgentResult] = {}
        pending: Dict[str, AgentTask] = {t.task_id: t for t in subtasks}
        running: Dict[str, asyncio.Task] = {}

        logger.info(
            f"Starting pipeline orchestration with {len(subtasks)} subtasks"
        )

        while pending or running:
            # Find tasks ready to execute (dependencies satisfied)
            ready = []
            for task_id, task in list(pending.items()):
                deps = self._task_dependencies.get(task_id, set())
                if all(d in results and results[d].success for d in deps):
                    ready.append(task)
                    pending.pop(task_id)

            # Start ready tasks
            for task in ready:
                # Resolve any dependency references in payload
                resolved_task = self._resolve_payload_refs(task, results)
                coro = self.execute_task(resolved_task)
                running[task.task_id] = asyncio.create_task(coro)
                logger.debug(f"Started task {task.task_id}")

            # Wait for at least one task to complete
            if running:
                done, _ = await asyncio.wait(
                    running.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                for completed in done:
                    result = completed.result()
                    results[result.task_id] = result
                    running.pop(result.task_id, None)

                    if not result.success:
                        # Cancel dependent tasks
                        dependents = self._get_dependents(result.task_id)
                        for dep_id in dependents:
                            pending.pop(dep_id, None)
                            results[dep_id] = AgentResult.failure_result(
                                task_id=dep_id,
                                errors=["Dependency failed"],
                                execution_time_ms=0,
                                agent_id=""
                            )
            else:
                # No running tasks and nothing ready - check for deadlock
                if pending:
                    logger.error("Pipeline deadlock detected")
                    for task_id in pending:
                        results[task_id] = AgentResult.failure_result(
                            task_id=task_id,
                            errors=["Dependency deadlock"],
                            execution_time_ms=0,
                            agent_id=""
                        )
                    break

        logger.info(
            f"Pipeline orchestration complete: "
            f"{sum(1 for r in results.values() if r.success)}/{len(results)} "
            f"tasks succeeded"
        )

        return results

    def _resolve_payload_refs(
        self,
        task: AgentTask,
        results: Dict[str, AgentResult]
    ) -> AgentTask:
        """Resolve references to previous task results in payload."""
        import re

        def resolve_ref(match: re.Match) -> str:
            ref_parts = match.group(1).split(":")
            if len(ref_parts) >= 3 and ref_parts[0] == "result":
                task_id = ref_parts[1]
                path = ref_parts[2:]
                if task_id in results:
                    result = results[task_id]
                    value = result.output
                    for key in path:
                        if isinstance(value, dict) and key in value:
                            value = value[key]
                        else:
                            return match.group(0)
                    return str(value)
            return match.group(0)

        resolved_payload = {}
        for key, value in task.payload.items():
            if isinstance(value, str):
                resolved_payload[key] = re.sub(
                    r"\{([^}]+)\}",
                    resolve_ref,
                    value
                )
            else:
                resolved_payload[key] = value

        return AgentTask(
            task_id=task.task_id,
            task_type=task.task_type,
            payload=resolved_payload,
            priority=task.priority,
            parent_task_id=task.parent_task_id,
            created_at=task.created_at,
            deadline=task.deadline,
            retry_count=task.retry_count,
            max_retries=task.max_retries,
            metadata=task.metadata
        )

    def _get_dependents(self, task_id: str) -> Set[str]:
        """Get all tasks that depend on the given task."""
        dependents = set()
        for tid, deps in self._task_dependencies.items():
            if task_id in deps:
                dependents.add(tid)
                dependents.update(self._get_dependents(tid))
        return dependents

    # -------------------------------------------------------------------------
    # Status and Caching
    # -------------------------------------------------------------------------

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get the status of a task."""
        if task_id in self._completed_tasks:
            return "completed" if self._completed_tasks[task_id].success else "failed"
        if task_id in self._active_tasks:
            return "running"
        return None

    def get_cached_result(self, task_id: str) -> Optional[AgentResult]:
        """Get a cached result if available."""
        return self._results_cache.get(task_id)

    def clear_cache(self) -> None:
        """Clear the results cache."""
        self._results_cache.clear()
        logger.info("Results cache cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get supervisor statistics."""
        return {
            "registered_agents": len(self._agents),
            "capabilities": list(self._agent_by_capability.keys()),
            "active_tasks": len(self._active_tasks),
            "completed_tasks": len(self._completed_tasks),
            "cached_results": len(self._results_cache),
            "agents": {
                aid: {
                    "state": a.state.name,
                    "capabilities": list(a.capabilities),
                    "avg_performance_ms": a.get_average_performance()
                }
                for aid, a in self._agents.items()
            }
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_imaging_supervisor() -> DeepAgentSupervisor:
    """
    Create and configure a DeepAgentSupervisor with standard imaging agents.

    Returns:
        Configured supervisor with acquisition, reconstruction, and analysis agents
    """
    supervisor = DeepAgentSupervisor()

    # Register specialized agents
    supervisor.register_agent(AcquisitionAgent())
    supervisor.register_agent(ReconstructionAgent())
    supervisor.register_agent(AnalysisAgent())

    logger.info("Created imaging supervisor with standard agents")

    return supervisor


# =============================================================================
# Main Demonstration
# =============================================================================


async def main() -> None:
    """Demonstrate full pipeline execution."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 70)
    print("DeepAgent Supervisor - Pipeline Demonstration")
    print("=" * 70)

    # Create supervisor with agents
    supervisor = create_imaging_supervisor()

    # Display initial statistics
    stats = supervisor.get_statistics()
    print(f"\nRegistered Agents: {stats['registered_agents']}")
    print(f"Capabilities: {stats['capabilities']}")

    # Create a full imaging pipeline task
    pipeline_task = AgentTask.create(
        task_type="full_imaging_pipeline",
        payload={
            "input_path": "/data/patient_001/scan.dcm",
            "modality": "CT",
            "acquisition_type": "dicom_import",
            "reconstruction_type": "ct_reconstruction",
            "analysis_type": "spatial_signature_analysis",
            "acquisition_params": {"slice_count": 256},
            "reconstruction_params": {"algorithm": "iterative"},
            "analysis_params": {}
        },
        priority=TaskPriority.HIGH,
        metadata={"patient_id": "P001", "study_id": "S001"}
    )

    print(f"\nExecuting pipeline task: {pipeline_task.task_id}")
    print(f"Task type: {pipeline_task.task_type}")
    print(f"Priority: {pipeline_task.priority.name}")

    # Execute the pipeline
    start_time = time.time()
    results = await supervisor.orchestrate_pipeline(pipeline_task)
    total_time = time.time() - start_time

    # Display results
    print(f"\nPipeline completed in {total_time:.2f}s")
    print("-" * 50)

    for task_id, result in results.items():
        status = "✓" if result.success else "✗"
        print(f"{status} Task {task_id[:8]}...")
        print(f"  - Agent: {result.agent_id[:8] if result.agent_id else 'N/A'}...")
        print(f"  - Time: {result.execution_time_ms}ms")
        if result.output:
            print(f"  - Status: {result.output.get('status', 'N/A')}")
        if result.errors:
            print(f"  - Errors: {result.errors}")

    # Final statistics
    final_stats = supervisor.get_statistics()
    print("\nFinal Statistics:")
    print(f"  - Completed Tasks: {final_stats['completed_tasks']}")
    print(f"  - Cached Results: {final_stats['cached_results']}")

    for agent_id, agent_info in final_stats['agents'].items():
        print(f"\n  Agent {agent_id[:8]}...")
        print(f"    - State: {agent_info['state']}")
        avg_perf = agent_info['avg_performance_ms']
        if avg_perf >= 0:
            print(f"    - Avg Performance: {avg_perf:.1f}ms")

    print("\n" + "=" * 70)
    print("Pipeline demonstration complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
