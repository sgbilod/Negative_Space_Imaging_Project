#!/usr/bin/env python
"""
HPC Integration Orchestrator
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

High-level orchestration layer for coordinating HPC operations with
the Negative Space Imaging analysis pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from hpc_config import HPCConfig, load_config
from hpc_integration import HPCIntegration, AnalysisTask, AnalysisResult, JobStatus

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the HPC orchestrator."""
    max_concurrent_jobs: int = 100
    batch_size: int = 50
    poll_interval: float = 5.0
    auto_retry: bool = True
    max_retries: int = 3
    result_cache_size: int = 1000


@dataclass
class WorkflowStep:
    """A step in an analysis workflow."""
    step_id: str
    task: AnalysisTask
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[AnalysisResult] = None


@dataclass
class Workflow:
    """A complete analysis workflow."""
    workflow_id: str
    name: str
    steps: List[WorkflowStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "created"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "steps": [
                {
                    "step_id": s.step_id,
                    "task_id": s.task.task_id,
                    "dependencies": s.dependencies,
                    "status": s.status,
                }
                for s in self.steps
            ],
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "metadata": self.metadata,
        }


class HPCOrchestrator:
    """
    High-level orchestrator for HPC operations.

    Coordinates complex analysis workflows, handles dependencies,
    and manages job execution across the HPC cluster.

    Example:
        >>> orchestrator = HPCOrchestrator()
        >>> workflow = orchestrator.create_workflow("analysis_pipeline")
        >>> orchestrator.add_step(workflow, task1)
        >>> orchestrator.add_step(workflow, task2, depends_on=[task1.task_id])
        >>> await orchestrator.execute_workflow(workflow)
    """

    def __init__(
        self,
        hpc_config: Optional[HPCConfig] = None,
        orchestrator_config: Optional[OrchestratorConfig] = None
    ):
        """
        Initialize the orchestrator.

        Args:
            hpc_config: HPC configuration
            orchestrator_config: Orchestrator configuration
        """
        self.hpc_config = hpc_config or load_config()
        self.config = orchestrator_config or OrchestratorConfig()
        self.hpc = HPCIntegration(self.hpc_config)
        self.workflows: Dict[str, Workflow] = {}
        self._running = False

        logger.info("HPC Orchestrator initialized")

    def create_workflow(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Workflow:
        """
        Create a new analysis workflow.

        Args:
            name: Workflow name
            metadata: Optional workflow metadata

        Returns:
            Created workflow
        """
        import uuid
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"

        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            metadata=metadata or {},
        )

        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {workflow_id} ({name})")

        return workflow

    def add_step(
        self,
        workflow: Workflow,
        task: AnalysisTask,
        depends_on: Optional[List[str]] = None
    ) -> WorkflowStep:
        """
        Add a step to a workflow.

        Args:
            workflow: Target workflow
            task: Analysis task for this step
            depends_on: List of step IDs this step depends on

        Returns:
            Created workflow step
        """
        step = WorkflowStep(
            step_id=f"step_{len(workflow.steps)}",
            task=task,
            dependencies=depends_on or [],
        )

        workflow.steps.append(step)
        logger.debug(f"Added step {step.step_id} to workflow {workflow.workflow_id}")

        return step

    async def execute_workflow(
        self,
        workflow: Workflow,
        wait: bool = True
    ) -> Dict[str, AnalysisResult]:
        """
        Execute a workflow.

        Args:
            workflow: Workflow to execute
            wait: Whether to wait for completion

        Returns:
            Dictionary of step_id -> result
        """
        logger.info(f"Executing workflow: {workflow.workflow_id}")
        workflow.status = "running"

        results: Dict[str, AnalysisResult] = {}
        completed_steps = set()

        while len(completed_steps) < len(workflow.steps):
            # Find steps that are ready to run
            ready_steps = [
                step for step in workflow.steps
                if step.step_id not in completed_steps and
                all(dep in completed_steps for dep in step.dependencies)
            ]

            if not ready_steps:
                if len(completed_steps) < len(workflow.steps):
                    logger.error("Workflow stalled - no ready steps")
                    workflow.status = "failed"
                    break
                continue

            # Submit ready steps
            tasks_to_submit = [step.task for step in ready_steps]
            jobs = await self.hpc.submit_batch(
                tasks_to_submit,
                max_concurrent=self.config.max_concurrent_jobs
            )

            # Create mapping of task_id to step
            task_to_step = {
                step.task.task_id: step for step in ready_steps
            }

            # Wait for jobs and collect results
            for job in jobs:
                job_status = await self.hpc.wait_for_job(
                    job.job_id,
                    poll_interval=self.config.poll_interval
                )

                result = await self.hpc.collect_result(job.job_id)

                if result:
                    step = task_to_step.get(result.task_id)
                    if step:
                        step.status = "completed" if result.success else "failed"
                        step.result = result
                        results[step.step_id] = result
                        completed_steps.add(step.step_id)

                        if not result.success and self.config.auto_retry:
                            logger.warning(f"Step {step.step_id} failed, may retry")

        workflow.status = "completed" if len(completed_steps) == len(workflow.steps) else "failed"
        logger.info(f"Workflow {workflow.workflow_id} finished: {workflow.status}")

        return results

    async def execute_batch(
        self,
        tasks: List[AnalysisTask]
    ) -> List[AnalysisResult]:
        """
        Execute a batch of independent tasks.

        Args:
            tasks: List of analysis tasks

        Returns:
            List of results
        """
        logger.info(f"Executing batch of {len(tasks)} tasks")

        jobs = await self.hpc.submit_batch(
            tasks,
            max_concurrent=self.config.max_concurrent_jobs
        )

        results = await self.hpc.collect_all_results([job.job_id for job in jobs])

        return list(results.values())

    def get_workflow_status(
        self,
        workflow_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get status of a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Workflow status dictionary
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None

        return workflow.to_dict()

    def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a running workflow.

        Args:
            workflow_id: Workflow ID to cancel

        Returns:
            True if cancelled successfully
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return False

        workflow.status = "cancelled"
        logger.info(f"Cancelled workflow: {workflow_id}")

        return True

    def cleanup(self) -> None:
        """Clean up orchestrator resources."""
        self.hpc.cleanup()
        logger.info("Orchestrator cleanup completed")


async def run_pipeline(
    image_paths: List[str],
    config: Optional[Dict[str, Any]] = None
) -> List[AnalysisResult]:
    """
    Run a complete analysis pipeline on images.

    Args:
        image_paths: List of image file paths
        config: Optional analysis configuration

    Returns:
        List of analysis results
    """
    orchestrator = HPCOrchestrator()

    tasks = [
        AnalysisTask(
            task_id=f"analysis_{i}",
            image_path=path,
            config=config or {},
        )
        for i, path in enumerate(image_paths)
    ]

    try:
        results = await orchestrator.execute_batch(tasks)
        return results
    finally:
        orchestrator.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def main() -> None:
        orchestrator = HPCOrchestrator()

        # Create a simple workflow
        workflow = orchestrator.create_workflow("demo_workflow")

        # Add tasks
        task1 = AnalysisTask(
            task_id="preprocess",
            image_path="/data/input.fits",
        )
        step1 = orchestrator.add_step(workflow, task1)

        task2 = AnalysisTask(
            task_id="analyze",
            image_path="/data/input.fits",
        )
        step2 = orchestrator.add_step(workflow, task2, depends_on=[step1.step_id])

        print(f"Workflow created: {workflow.workflow_id}")
        print(f"Steps: {len(workflow.steps)}")

        orchestrator.cleanup()

    asyncio.run(main())
