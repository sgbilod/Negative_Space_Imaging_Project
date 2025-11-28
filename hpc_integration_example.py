#!/usr/bin/env python
"""
HPC Integration Examples
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides working examples demonstrating how to integrate
with the HPC cluster for distributed image processing workflows.

Examples include:
- Basic job submission
- Batch processing
- Distributed analysis pipeline
- GPU-accelerated workflows
- Best practices for parallel image processing
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Import HPC modules
from hpc_config import HPCConfig, HPCBackend, load_config
from hpc_integration import (
    HPCIntegration,
    AnalysisTask,
    AnalysisResult,
    JobPriority,
    run_analysis,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Basic Job Submission
# ============================================================================

async def basic_job_submission_example() -> None:
    """
    Demonstrates basic job submission to the HPC cluster.
    
    This example shows how to:
    1. Initialize HPC configuration
    2. Create an analysis task
    3. Submit the task to the cluster
    4. Wait for completion and collect results
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Job Submission")
    print("=" * 60)
    
    # Initialize HPC configuration
    # Uses environment variables or defaults
    config = load_config()
    print(f"Using HPC backend: {config.backend.value}")
    
    # Create HPC integration instance
    hpc = HPCIntegration(config)
    
    try:
        # Create a simple analysis task
        task = AnalysisTask(
            task_id="example_basic_001",
            image_path="/path/to/sample/image.fits",
            config={
                "analysis_type": "negative_space",
                "threshold": 0.5,
            },
            priority=JobPriority.NORMAL,
        )
        
        print(f"Submitting task: {task.task_id}")
        
        # Submit and wait for completion
        job = await hpc.submit_analysis(task, wait=True)
        
        print(f"Job completed with status: {job.status.value}")
        
        # Collect results
        result = await hpc.collect_result(job.job_id)
        if result and result.success:
            print(f"Analysis completed in {result.processing_time:.2f}s")
            print(f"Result data: {result.result_data}")
        else:
            print(f"Analysis failed: {result.error_message if result else 'Unknown error'}")
            
    finally:
        # Clean up temporary files
        hpc.cleanup()


# ============================================================================
# Example 2: Batch Processing
# ============================================================================

async def batch_processing_example(image_paths: Optional[List[str]] = None) -> List[AnalysisResult]:
    """
    Demonstrates batch processing of multiple images.
    
    This example shows how to:
    1. Submit multiple tasks efficiently
    2. Control concurrency
    3. Collect results from all jobs
    
    Args:
        image_paths: List of image file paths to process
        
    Returns:
        List of analysis results
    """
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)
    
    # Use sample paths if none provided
    if image_paths is None:
        image_paths = [
            f"/data/images/observation_{i:04d}.fits"
            for i in range(10)
        ]
    
    config = load_config()
    hpc = HPCIntegration(config)
    
    try:
        # Create tasks for all images
        tasks = [
            AnalysisTask(
                task_id=f"batch_{i:04d}",
                image_path=path,
                config={"batch_mode": True},
                priority=JobPriority.NORMAL,
            )
            for i, path in enumerate(image_paths)
        ]
        
        print(f"Submitting {len(tasks)} tasks...")
        
        # Submit batch with concurrency limit
        jobs = await hpc.submit_batch(tasks, max_concurrent=5)
        
        print(f"Submitted {len(jobs)} jobs")
        
        # Collect all results
        job_ids = [job.job_id for job in jobs]
        results = await hpc.collect_all_results(job_ids)
        
        # Summary
        successful = sum(1 for r in results.values() if r.success)
        print(f"Completed: {successful}/{len(results)} successful")
        
        return list(results.values())
        
    finally:
        hpc.cleanup()


# ============================================================================
# Example 3: Distributed Analysis Pipeline
# ============================================================================

async def distributed_pipeline_example() -> Dict[str, Any]:
    """
    Demonstrates a multi-stage distributed analysis pipeline.
    
    This example shows how to:
    1. Chain dependent tasks
    2. Handle task dependencies
    3. Aggregate results from multiple stages
    
    Pipeline stages:
    1. Preprocessing - Calibration and noise reduction
    2. Detection - Source detection and extraction
    3. Analysis - Negative space analysis
    4. Aggregation - Combine and summarize results
    """
    print("\n" + "=" * 60)
    print("Example 3: Distributed Analysis Pipeline")
    print("=" * 60)
    
    config = load_config()
    hpc = HPCIntegration(config)
    
    results: Dict[str, Any] = {
        "preprocessing": [],
        "detection": [],
        "analysis": [],
        "summary": {},
    }
    
    try:
        # Stage 1: Preprocessing
        print("\nStage 1: Preprocessing...")
        preprocess_tasks = [
            AnalysisTask(
                task_id=f"preprocess_{i}",
                image_path=f"/data/raw/image_{i}.fits",
                config={
                    "stage": "preprocessing",
                    "dark_subtraction": True,
                    "flat_correction": True,
                },
                priority=JobPriority.HIGH,
            )
            for i in range(4)
        ]
        
        preprocess_jobs = await hpc.submit_batch(preprocess_tasks)
        for job in preprocess_jobs:
            await hpc.wait_for_job(job.job_id)
            result = await hpc.collect_result(job.job_id)
            if result:
                results["preprocessing"].append(result.to_dict())
        
        print(f"  Completed {len(results['preprocessing'])} preprocessing tasks")
        
        # Stage 2: Detection (depends on preprocessing)
        print("\nStage 2: Source Detection...")
        detection_tasks = [
            AnalysisTask(
                task_id=f"detect_{i}",
                image_path=f"/data/preprocessed/image_{i}.fits",
                config={
                    "stage": "detection",
                    "detection_threshold": 5.0,
                    "min_area": 10,
                },
                priority=JobPriority.NORMAL,
                dependencies=[f"preprocess_{i}"],
            )
            for i in range(4)
        ]
        
        detection_jobs = await hpc.submit_batch(detection_tasks)
        for job in detection_jobs:
            await hpc.wait_for_job(job.job_id)
            result = await hpc.collect_result(job.job_id)
            if result:
                results["detection"].append(result.to_dict())
        
        print(f"  Completed {len(results['detection'])} detection tasks")
        
        # Stage 3: Negative Space Analysis
        print("\nStage 3: Negative Space Analysis...")
        analysis_tasks = [
            AnalysisTask(
                task_id=f"analyze_{i}",
                image_path=f"/data/detected/image_{i}.fits",
                config={
                    "stage": "analysis",
                    "negative_space_threshold": 0.3,
                    "region_min_size": 100,
                },
                priority=JobPriority.NORMAL,
                dependencies=[f"detect_{i}"],
            )
            for i in range(4)
        ]
        
        analysis_jobs = await hpc.submit_batch(analysis_tasks)
        for job in analysis_jobs:
            await hpc.wait_for_job(job.job_id)
            result = await hpc.collect_result(job.job_id)
            if result:
                results["analysis"].append(result.to_dict())
        
        print(f"  Completed {len(results['analysis'])} analysis tasks")
        
        # Stage 4: Aggregation
        print("\nStage 4: Aggregating Results...")
        results["summary"] = {
            "total_images": 4,
            "preprocessing_completed": len(results["preprocessing"]),
            "detection_completed": len(results["detection"]),
            "analysis_completed": len(results["analysis"]),
            "pipeline_success": all(
                len(results[stage]) == 4
                for stage in ["preprocessing", "detection", "analysis"]
            ),
        }
        
        print(f"  Pipeline complete: {results['summary']['pipeline_success']}")
        
        return results
        
    finally:
        hpc.cleanup()


# ============================================================================
# Example 4: GPU-Accelerated Workflow
# ============================================================================

async def gpu_accelerated_example() -> None:
    """
    Demonstrates GPU-accelerated image processing.
    
    This example shows how to:
    1. Configure GPU resources
    2. Submit GPU-specific tasks
    3. Optimize for GPU memory
    """
    print("\n" + "=" * 60)
    print("Example 4: GPU-Accelerated Workflow")
    print("=" * 60)
    
    # Create GPU-optimized configuration
    config = HPCConfig(
        backend=HPCBackend.LOCAL,
        cluster_name="gpu-cluster",
    )
    config.compute.gpus_per_node = 2
    config.scheduling.default_memory_gb = 32
    
    hpc = HPCIntegration(config)
    
    try:
        # Create GPU-intensive task
        task = AnalysisTask(
            task_id="gpu_analysis_001",
            image_path="/data/large_mosaic.fits",
            config={
                "use_gpu": True,
                "gpu_memory_fraction": 0.8,
                "batch_size": 64,
                "mixed_precision": True,
                "algorithm": "deep_learning_detection",
            },
            priority=JobPriority.HIGH,
        )
        
        print("Submitting GPU task...")
        job = await hpc.submit_analysis(task, wait=True)
        
        print(f"GPU job completed: {job.status.value}")
        
        result = await hpc.collect_result(job.job_id)
        if result:
            print(f"Processing time: {result.processing_time:.2f}s")
            
    finally:
        hpc.cleanup()


# ============================================================================
# Example 5: Best Practices for Parallel Image Processing
# ============================================================================

def best_practices_example() -> None:
    """
    Demonstrates best practices for parallel image processing.
    
    Key best practices:
    1. Proper resource estimation
    2. Efficient data staging
    3. Error handling and retries
    4. Progress monitoring
    """
    print("\n" + "=" * 60)
    print("Example 5: Best Practices")
    print("=" * 60)
    
    print("""
    Best Practices for HPC Image Processing:
    
    1. RESOURCE ESTIMATION
       - Profile your workload to understand CPU/GPU requirements
       - Request appropriate memory based on image size
       - Use job arrays for similar tasks
       
       Example:
       ```python
       # Estimate memory based on image size
       image_size_mb = os.path.getsize(image_path) / (1024 * 1024)
       memory_gb = max(4, int(image_size_mb * 3))  # 3x image size
       ```
    
    2. DATA STAGING
       - Stage data to fast scratch storage before processing
       - Use parallel I/O for large datasets
       - Clean up intermediate files
       
       Example:
       ```python
       # Stage data to scratch
       scratch_path = Path("/scratch") / job_id
       shutil.copy(source_path, scratch_path)
       # Process from scratch
       result = process_image(scratch_path / filename)
       # Clean up
       shutil.rmtree(scratch_path)
       ```
    
    3. ERROR HANDLING
       - Implement automatic retries for transient failures
       - Log errors comprehensively
       - Use checkpointing for long-running jobs
       
       Example:
       ```python
       async def process_with_retry(task, max_retries=3):
           for attempt in range(max_retries):
               try:
                   return await hpc.submit_analysis(task, wait=True)
               except Exception as e:
                   if attempt == max_retries - 1:
                       raise
                   await asyncio.sleep(2 ** attempt)
       ```
    
    4. PROGRESS MONITORING
       - Implement progress callbacks
       - Use structured logging
       - Monitor resource utilization
       
       Example:
       ```python
       # Progress tracking
       completed = 0
       for job in jobs:
           status = await hpc.wait_for_job(job.job_id)
           completed += 1
           print(f"Progress: {completed}/{len(jobs)} ({100*completed/len(jobs):.1f}%)")
       ```
    
    5. PARALLELIZATION STRATEGY
       - Use coarse-grained parallelism for I/O-bound tasks
       - Use fine-grained parallelism for compute-bound tasks
       - Balance load across nodes
       
    6. GPU OPTIMIZATION
       - Batch operations to maximize GPU utilization
       - Use mixed precision when possible
       - Monitor GPU memory usage
    """)


# ============================================================================
# Main Entry Point
# ============================================================================

async def run_all_examples() -> None:
    """Run all HPC integration examples."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    print("\n" + "=" * 60)
    print("HPC Integration Examples")
    print("Negative Space Imaging Project")
    print("=" * 60)
    
    # Example 1: Basic Job Submission
    await basic_job_submission_example()
    
    # Example 2: Batch Processing
    await batch_processing_example()
    
    # Example 3: Distributed Pipeline
    await distributed_pipeline_example()
    
    # Example 4: GPU-Accelerated Workflow
    await gpu_accelerated_example()
    
    # Example 5: Best Practices (synchronous)
    best_practices_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


def main() -> None:
    """Main entry point."""
    asyncio.run(run_all_examples())


if __name__ == "__main__":
    main()
