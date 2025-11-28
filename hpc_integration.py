#!/usr/bin/env python
"""
HPC Integration Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module integrates HPC capabilities with the Negative Space Imaging
core system, providing job submission, monitoring, result collection,
and error handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from hpc_config import HPCBackend, HPCConfig, load_config

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """HPC job status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class HPCJob:
    """Represents an HPC job."""
    job_id: str
    name: str
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    submit_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    node_list: List[str] = field(default_factory=list)
    output_path: Optional[str] = None
    error_path: Optional[str] = None
    exit_code: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status.value,
            "priority": self.priority.value,
            "submit_time": self.submit_time.isoformat() if self.submit_time else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "node_list": self.node_list,
            "output_path": self.output_path,
            "error_path": self.error_path,
            "exit_code": self.exit_code,
            "metadata": self.metadata,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }

    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class AnalysisTask:
    """Task for negative space analysis."""
    task_id: str
    image_path: str
    config: Dict[str, Any] = field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "image_path": self.image_path,
            "config": self.config,
            "priority": self.priority.value,
            "dependencies": self.dependencies,
        }


@dataclass
class AnalysisResult:
    """Result from HPC analysis."""
    task_id: str
    job_id: str
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    node_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "job_id": self.job_id,
            "success": self.success,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "processing_time": self.processing_time,
            "node_id": self.node_id,
        }


class JobScheduler:
    """
    Interface for different HPC job schedulers.

    Provides a unified interface for SLURM, PBS, and LSF backends.
    """

    def __init__(self, backend: HPCBackend):
        """Initialize scheduler interface."""
        self.backend = backend

    def submit_job(self, script_path: str) -> Optional[str]:
        """
        Submit a job to the scheduler.

        Args:
            script_path: Path to the job script

        Returns:
            Job ID if successful, None otherwise
        """
        if self.backend == HPCBackend.SLURM:
            return self._submit_slurm(script_path)
        elif self.backend == HPCBackend.PBS:
            return self._submit_pbs(script_path)
        elif self.backend == HPCBackend.LSF:
            return self._submit_lsf(script_path)
        else:
            return self._submit_local(script_path)

    def _submit_slurm(self, script_path: str) -> Optional[str]:
        """Submit job to SLURM."""
        try:
            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                # Parse job ID from output like "Submitted batch job 123456"
                output = result.stdout.strip()
                job_id = output.split()[-1]
                logger.info(f"SLURM job submitted: {job_id}")
                return job_id
            else:
                logger.error(f"SLURM submit failed: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"SLURM submit error: {e}")
            return None

    def _submit_pbs(self, script_path: str) -> Optional[str]:
        """Submit job to PBS."""
        try:
            result = subprocess.run(
                ["qsub", script_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                job_id = result.stdout.strip()
                logger.info(f"PBS job submitted: {job_id}")
                return job_id
            else:
                logger.error(f"PBS submit failed: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"PBS submit error: {e}")
            return None

    def _submit_lsf(self, script_path: str) -> Optional[str]:
        """Submit job to LSF."""
        try:
            result = subprocess.run(
                ["bsub", "<", script_path],
                capture_output=True,
                text=True,
                shell=True,
                timeout=60,
            )
            if result.returncode == 0:
                # Parse job ID from output
                output = result.stdout.strip()
                import re
                match = re.search(r"<(\d+)>", output)
                if match:
                    job_id = match.group(1)
                    logger.info(f"LSF job submitted: {job_id}")
                    return job_id
            logger.error(f"LSF submit failed: {result.stderr}")
            return None
        except Exception as e:
            logger.error(f"LSF submit error: {e}")
            return None

    def _submit_local(self, script_path: str) -> Optional[str]:
        """Execute job locally."""
        job_id = f"local_{uuid.uuid4().hex[:8]}"
        try:
            subprocess.Popen(
                ["bash", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info(f"Local job started: {job_id}")
            return job_id
        except Exception as e:
            logger.error(f"Local execution error: {e}")
            return None

    def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get the status of a job.

        Args:
            job_id: The job ID to check

        Returns:
            Current job status
        """
        if self.backend == HPCBackend.SLURM:
            return self._get_slurm_status(job_id)
        elif self.backend == HPCBackend.PBS:
            return self._get_pbs_status(job_id)
        elif self.backend == HPCBackend.LSF:
            return self._get_lsf_status(job_id)
        else:
            return JobStatus.COMPLETED

    def _get_slurm_status(self, job_id: str) -> JobStatus:
        """Get SLURM job status."""
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                status = result.stdout.strip().upper()
                return self._parse_slurm_status(status)
            return JobStatus.COMPLETED
        except Exception as e:
            logger.error(f"SLURM status check error: {e}")
            return JobStatus.FAILED

    def _parse_slurm_status(self, status: str) -> JobStatus:
        """Parse SLURM status string."""
        status_map = {
            "PENDING": JobStatus.PENDING,
            "RUNNING": JobStatus.RUNNING,
            "COMPLETING": JobStatus.RUNNING,
            "COMPLETED": JobStatus.COMPLETED,
            "FAILED": JobStatus.FAILED,
            "CANCELLED": JobStatus.CANCELLED,
            "TIMEOUT": JobStatus.TIMEOUT,
        }
        return status_map.get(status, JobStatus.PENDING)

    def _get_pbs_status(self, job_id: str) -> JobStatus:
        """Get PBS job status."""
        try:
            result = subprocess.run(
                ["qstat", "-f", job_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "job_state" in line:
                        state = line.split("=")[-1].strip()
                        return self._parse_pbs_status(state)
            return JobStatus.COMPLETED
        except Exception:
            return JobStatus.COMPLETED

    def _parse_pbs_status(self, status: str) -> JobStatus:
        """Parse PBS status string."""
        status_map = {
            "Q": JobStatus.QUEUED,
            "R": JobStatus.RUNNING,
            "C": JobStatus.COMPLETED,
            "E": JobStatus.FAILED,
        }
        return status_map.get(status, JobStatus.PENDING)

    def _get_lsf_status(self, job_id: str) -> JobStatus:
        """Get LSF job status."""
        try:
            result = subprocess.run(
                ["bjobs", "-o", "stat", "-noheader", job_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                status = result.stdout.strip()
                return self._parse_lsf_status(status)
            return JobStatus.COMPLETED
        except Exception:
            return JobStatus.COMPLETED

    def _parse_lsf_status(self, status: str) -> JobStatus:
        """Parse LSF status string."""
        status_map = {
            "PEND": JobStatus.PENDING,
            "RUN": JobStatus.RUNNING,
            "DONE": JobStatus.COMPLETED,
            "EXIT": JobStatus.FAILED,
        }
        return status_map.get(status, JobStatus.PENDING)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancellation was successful
        """
        cmd = {
            HPCBackend.SLURM: ["scancel", job_id],
            HPCBackend.PBS: ["qdel", job_id],
            HPCBackend.LSF: ["bkill", job_id],
        }.get(self.backend)

        if cmd is None:
            return True

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Job cancellation error: {e}")
            return False


class HPCIntegration:
    """
    Main HPC integration class for Negative Space Imaging.

    Connects HPC capabilities with the NegativeSpaceAnalyzer,
    handles job submission, monitoring, and result collection.

    Example:
        >>> config = load_config()
        >>> hpc = HPCIntegration(config)
        >>> task = AnalysisTask(task_id="task1", image_path="/path/to/image.png")
        >>> result = await hpc.submit_analysis(task)
    """

    def __init__(
        self,
        config: Optional[HPCConfig] = None,
        work_dir: Optional[str] = None
    ):
        """
        Initialize HPC integration.

        Args:
            config: HPC configuration
            work_dir: Working directory for job files
        """
        self.config = config or load_config()
        self.work_dir = Path(work_dir or tempfile.mkdtemp(prefix="nsi_hpc_"))
        self.scheduler = JobScheduler(self.config.backend)
        self.jobs: Dict[str, HPCJob] = {}
        self.results: Dict[str, AnalysisResult] = {}
        self._running = False

        logger.info(f"HPC Integration initialized with backend: {self.config.backend.value}")

    async def submit_analysis(
        self,
        task: AnalysisTask,
        wait: bool = False
    ) -> HPCJob:
        """
        Submit an analysis task to the HPC cluster.

        Args:
            task: Analysis task to submit
            wait: Whether to wait for completion

        Returns:
            HPCJob instance
        """
        # Create job script
        script_path = self._create_job_script(task)

        # Submit job
        hpc_job_id = self.scheduler.submit_job(str(script_path))

        if hpc_job_id is None:
            raise RuntimeError(f"Failed to submit job for task {task.task_id}")

        # Create job record
        job = HPCJob(
            job_id=hpc_job_id,
            name=f"nsi_analysis_{task.task_id}",
            status=JobStatus.PENDING,
            priority=task.priority,
            submit_time=datetime.utcnow(),
            metadata={"task_id": task.task_id},
        )

        self.jobs[hpc_job_id] = job
        logger.info(f"Submitted job {hpc_job_id} for task {task.task_id}")

        if wait:
            await self.wait_for_job(hpc_job_id)

        return job

    async def submit_batch(
        self,
        tasks: List[AnalysisTask],
        max_concurrent: int = 10
    ) -> List[HPCJob]:
        """
        Submit multiple analysis tasks.

        Args:
            tasks: List of analysis tasks
            max_concurrent: Maximum concurrent jobs

        Returns:
            List of HPCJob instances
        """
        jobs = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def submit_with_limit(task: AnalysisTask) -> HPCJob:
            async with semaphore:
                return await self.submit_analysis(task)

        # Submit all tasks with concurrency limit
        results = await asyncio.gather(
            *[submit_with_limit(task) for task in tasks],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, HPCJob):
                jobs.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Batch submit error: {result}")

        return jobs

    async def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None
    ) -> JobStatus:
        """
        Wait for a job to complete.

        Args:
            job_id: Job ID to wait for
            poll_interval: Polling interval in seconds
            timeout: Maximum wait time in seconds

        Returns:
            Final job status
        """
        start_time = time.time()

        while True:
            status = self.scheduler.get_job_status(job_id)

            if job_id in self.jobs:
                self.jobs[job_id].status = status
                if status == JobStatus.RUNNING and self.jobs[job_id].start_time is None:
                    self.jobs[job_id].start_time = datetime.utcnow()

            if status in (JobStatus.COMPLETED, JobStatus.FAILED,
                         JobStatus.CANCELLED, JobStatus.TIMEOUT):
                if job_id in self.jobs:
                    self.jobs[job_id].end_time = datetime.utcnow()
                return status

            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Job {job_id} wait timeout")
                return JobStatus.TIMEOUT

            await asyncio.sleep(poll_interval)

    async def collect_result(self, job_id: str) -> Optional[AnalysisResult]:
        """
        Collect results from a completed job.

        Args:
            job_id: Job ID to collect results from

        Returns:
            AnalysisResult if available
        """
        if job_id not in self.jobs:
            logger.error(f"Unknown job: {job_id}")
            return None

        job = self.jobs[job_id]

        if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
            await self.wait_for_job(job_id)

        task_id = job.metadata.get("task_id", "unknown")

        # Read output file
        output_file = self.work_dir / f"{job_id}_output.json"
        result_data = None

        if output_file.exists():
            try:
                with open(output_file, "r") as f:
                    result_data = json.load(f)
            except Exception as e:
                logger.error(f"Error reading result file: {e}")

        result = AnalysisResult(
            task_id=task_id,
            job_id=job_id,
            success=job.status == JobStatus.COMPLETED,
            result_data=result_data,
            processing_time=job.duration or 0.0,
        )

        self.results[task_id] = result
        return result

    async def collect_all_results(
        self,
        job_ids: List[str]
    ) -> Dict[str, AnalysisResult]:
        """
        Collect results from multiple jobs.

        Args:
            job_ids: List of job IDs

        Returns:
            Dictionary of results by task ID
        """
        results = await asyncio.gather(
            *[self.collect_result(job_id) for job_id in job_ids],
            return_exceptions=True,
        )

        collected = {}
        for result in results:
            if isinstance(result, AnalysisResult):
                collected[result.task_id] = result

        return collected

    def _create_job_script(self, task: AnalysisTask) -> Path:
        """Create a job submission script."""
        script_path = self.work_dir / f"job_{task.task_id}.sh"
        output_path = self.work_dir / f"{task.task_id}_output.json"

        # Generate header based on backend
        header = self.config.get_job_script_header(f"nsi_{task.task_id}")

        # Create analysis script
        script_content = f"""{header}
# Negative Space Imaging Analysis Job
# Task ID: {task.task_id}

set -e

echo "Starting analysis for {task.task_id}"
echo "Image: {task.image_path}"
echo "Node: $(hostname)"

# Change to work directory
cd {self.work_dir}

# Run analysis (this would invoke the actual analyzer)
python -c "
import json
import sys
import time
import numpy as np

# Simulate analysis
start = time.time()

# This would be replaced with actual NegativeSpaceAnalyzer call
result = {{
    'task_id': '{task.task_id}',
    'image_path': '{task.image_path}',
    'regions_detected': np.random.randint(5, 20),
    'processing_time': time.time() - start,
    'status': 'completed'
}}

# Save results
with open('{output_path}', 'w') as f:
    json.dump(result, f, indent=2)

print('Analysis completed successfully')
"

echo "Job completed"
"""

        with open(script_path, "w") as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)
        return script_path

    async def retry_failed_job(
        self,
        job_id: str,
        max_retries: Optional[int] = None
    ) -> Optional[HPCJob]:
        """
        Retry a failed job.

        Args:
            job_id: Job ID to retry
            max_retries: Maximum retry attempts

        Returns:
            New HPCJob if retry was submitted
        """
        if job_id not in self.jobs:
            logger.error(f"Unknown job: {job_id}")
            return None

        job = self.jobs[job_id]
        max_retries = max_retries or job.max_retries

        if job.retry_count >= max_retries:
            logger.warning(f"Job {job_id} exceeded max retries")
            return None

        # Get original task
        task_id = job.metadata.get("task_id")
        if not task_id:
            logger.error("Cannot retry job without task_id")
            return None

        # Create new task
        new_task = AnalysisTask(
            task_id=f"{task_id}_retry{job.retry_count + 1}",
            image_path=job.metadata.get("image_path", ""),
            config=job.metadata.get("config", {}),
            priority=job.priority,
        )

        # Submit new job
        new_job = await self.submit_analysis(new_task)
        new_job.retry_count = job.retry_count + 1

        logger.info(f"Retried job {job_id} as {new_job.job_id}")
        return new_job

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancellation was successful
        """
        success = self.scheduler.cancel_job(job_id)

        if success and job_id in self.jobs:
            self.jobs[job_id].status = JobStatus.CANCELLED
            self.jobs[job_id].end_time = datetime.utcnow()

        return success

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get the status of a specific job."""
        if job_id in self.jobs:
            # Update status
            status = self.scheduler.get_job_status(job_id)
            self.jobs[job_id].status = status
            return status
        return None

    def get_all_jobs(self) -> Dict[str, HPCJob]:
        """Get all tracked jobs."""
        return self.jobs.copy()

    def get_pending_jobs(self) -> List[HPCJob]:
        """Get all pending jobs."""
        return [
            job for job in self.jobs.values()
            if job.status in (JobStatus.PENDING, JobStatus.QUEUED)
        ]

    def get_running_jobs(self) -> List[HPCJob]:
        """Get all running jobs."""
        return [
            job for job in self.jobs.values()
            if job.status == JobStatus.RUNNING
        ]

    def get_completed_jobs(self) -> List[HPCJob]:
        """Get all completed jobs."""
        return [
            job for job in self.jobs.values()
            if job.status == JobStatus.COMPLETED
        ]

    def get_failed_jobs(self) -> List[HPCJob]:
        """Get all failed jobs."""
        return [
            job for job in self.jobs.values()
            if job.status == JobStatus.FAILED
        ]

    def cleanup(self) -> None:
        """Clean up temporary files."""
        import shutil
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        logger.info("Cleaned up HPC work directory")


async def run_analysis(
    image_paths: List[str],
    config: Optional[HPCConfig] = None,
    wait: bool = True
) -> List[AnalysisResult]:
    """
    Convenience function to run HPC analysis on images.

    Args:
        image_paths: List of image file paths
        config: Optional HPC configuration
        wait: Whether to wait for completion

    Returns:
        List of analysis results
    """
    hpc = HPCIntegration(config)

    tasks = [
        AnalysisTask(
            task_id=f"analysis_{i}",
            image_path=path,
        )
        for i, path in enumerate(image_paths)
    ]

    jobs = await hpc.submit_batch(tasks)

    if wait:
        job_ids = [job.job_id for job in jobs]
        results = await hpc.collect_all_results(job_ids)
        return list(results.values())

    return []


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    async def main() -> None:
        config = load_config()
        hpc = HPCIntegration(config)

        # Example usage
        task = AnalysisTask(
            task_id="demo_task",
            image_path="/path/to/test/image.png",
        )

        print("Submitting analysis task...")
        job = await hpc.submit_analysis(task, wait=True)
        print(f"Job completed with status: {job.status.value}")

        result = await hpc.collect_result(job.job_id)
        if result:
            print(f"Result: {result.to_dict()}")

        hpc.cleanup()

    asyncio.run(main())
