#!/usr/bin/env python
"""
HPC Demo Scripts
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

Demonstration scripts for HPC usage with the Negative Space Imaging system.
Provides quick-start examples and sample workflows.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_basic_configuration() -> None:
    """
    Demo: Basic HPC Configuration

    Shows how to create and configure HPC settings.
    """
    print("\n" + "=" * 60)
    print("Demo: Basic HPC Configuration")
    print("=" * 60 + "\n")

    from hpc_config import (
        HPCConfig,
        HPCBackend,
        ComputeNodeConfig,
        MemoryConfig,
        QueueConfig,
    )

    # Create configuration with defaults
    config = HPCConfig()
    print(f"Backend detected: {config.backend.value}")
    print(f"Cluster name: {config.cluster_name}")
    print(f"Max nodes: {config.compute.max_nodes}")
    print(f"CPUs per node: {config.compute.cpus_per_node}")

    # Create custom configuration
    custom_config = HPCConfig(
        backend=HPCBackend.LOCAL,
        cluster_name="demo-cluster",
        compute=ComputeNodeConfig(
            min_nodes=2,
            max_nodes=8,
            cpus_per_node=16,
            gpus_per_node=2,
        ),
        memory=MemoryConfig(
            per_node=128,
            per_task=16,
        ),
        queues=[
            QueueConfig(name="fast", priority=200, max_wall_time="01:00:00"),
            QueueConfig(name="batch", priority=100, max_wall_time="24:00:00"),
        ],
    )

    print("\nCustom configuration:")
    print(f"  Compute nodes: {custom_config.compute.min_nodes}-{custom_config.compute.max_nodes}")
    print(f"  GPUs per node: {custom_config.compute.gpus_per_node}")
    print(f"  Queues: {[q.name for q in custom_config.queues]}")

    # Validate configuration
    try:
        custom_config.validate()
        print("\n✓ Configuration is valid")
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")

    # Generate job script header
    print("\nSLURM job script header:")
    print("-" * 40)
    slurm_config = HPCConfig(backend=HPCBackend.SLURM)
    print(slurm_config.get_job_script_header("demo_job"))


def demo_benchmarking() -> None:
    """
    Demo: HPC Benchmarking

    Shows how to run and analyze benchmarks.
    """
    print("\n" + "=" * 60)
    print("Demo: HPC Benchmarking")
    print("=" * 60 + "\n")

    from hpc_benchmark import HPCBenchmark

    # Create benchmark suite
    benchmark = HPCBenchmark(
        warmup_iterations=2,
        benchmark_iterations=5,
        data_sizes=[1000, 5000],
    )

    # Print system info
    print("System Information:")
    print(f"  Hostname: {benchmark.system_info.hostname}")
    print(f"  Platform: {benchmark.system_info.platform}")
    print(f"  CPU: {benchmark.system_info.cpu_model}")
    print(f"  CPU Cores: {benchmark.system_info.cpu_count}")
    print(f"  GPU Available: {benchmark.system_info.gpu_available}")
    if benchmark.system_info.gpu_available:
        print(f"  GPUs: {benchmark.system_info.gpu_models}")

    # Run quick benchmarks
    print("\nRunning quick benchmarks...")
    report = benchmark.run_quick()

    # Print results
    print("\nBenchmark Results:")
    print("-" * 60)
    for result in report.results:
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.name}")
        print(f"      Duration: {result.duration_seconds:.4f}s")
        print(f"      Ops/sec: {result.operations_per_second:.2f}")

    # Print summary
    summary = report.get_summary()
    print("\nSummary:")
    print(f"  Total benchmarks: {summary['total_benchmarks']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Total duration: {report.total_duration_seconds:.2f}s")

    # Save report
    report_path = Path(tempfile.gettempdir()) / "hpc_benchmark_demo.json"
    report.save_json(report_path)
    print(f"\nReport saved to: {report_path}")


async def demo_job_submission() -> None:
    """
    Demo: HPC Job Submission

    Shows how to submit and monitor HPC jobs.
    """
    print("\n" + "=" * 60)
    print("Demo: HPC Job Submission")
    print("=" * 60 + "\n")

    from hpc_config import HPCConfig, HPCBackend
    from hpc_integration import HPCIntegration, AnalysisTask, JobPriority

    # Create configuration (using local backend for demo)
    config = HPCConfig(backend=HPCBackend.LOCAL)

    # Initialize HPC integration
    hpc = HPCIntegration(config)

    print(f"HPC Backend: {config.backend.value}")
    print(f"Work directory: {hpc.work_dir}")

    # Create analysis tasks
    tasks = [
        AnalysisTask(
            task_id=f"demo_task_{i}",
            image_path=f"/path/to/image_{i}.png",
            priority=JobPriority.NORMAL,
            config={"threshold": 0.5, "max_regions": 100},
        )
        for i in range(3)
    ]

    print(f"\nSubmitting {len(tasks)} analysis tasks...")

    # Submit batch
    jobs = await hpc.submit_batch(tasks, max_concurrent=2)

    print("\nSubmitted jobs:")
    for job in jobs:
        print(f"  - {job.job_id}: {job.name} (status: {job.status.value})")

    # Wait for completion
    print("\nWaiting for jobs to complete...")
    for job in jobs:
        status = await hpc.wait_for_job(job.job_id, timeout=30)
        print(f"  {job.job_id}: {status.value}")

    # Collect results
    print("\nCollecting results...")
    results = await hpc.collect_all_results([job.job_id for job in jobs])

    for task_id, result in results.items():
        status = "✓" if result.success else "✗"
        print(f"  {status} {task_id}: {result.processing_time:.2f}s")

    # Cleanup
    hpc.cleanup()
    print("\n✓ Demo completed")


def demo_multi_node_deployment() -> None:
    """
    Demo: Multi-Node Deployment

    Shows how to configure and deploy across multiple nodes.
    """
    print("\n" + "=" * 60)
    print("Demo: Multi-Node Deployment")
    print("=" * 60 + "\n")

    from hpc_multi_node_deploy import (
        MultiNodeDeployer,
        DeploymentConfig,
        NodeSpec,
    )

    # Create deployment configuration
    deploy_config = DeploymentConfig(
        cluster_name="demo-cluster",
        nodes=[
            NodeSpec(
                hostname="node1.cluster.local",
                cpus=32,
                memory_gb=128,
                gpus=2,
            ),
            NodeSpec(
                hostname="node2.cluster.local",
                cpus=32,
                memory_gb=128,
                gpus=2,
            ),
            NodeSpec(
                hostname="node3.cluster.local",
                cpus=64,
                memory_gb=256,
                gpus=4,
            ),
        ],
        enable_load_balancing=True,
        health_check_interval=30,
    )

    print(f"Cluster: {deploy_config.cluster_name}")
    print(f"Nodes: {len(deploy_config.nodes)}")

    total_cpus = sum(n.cpus for n in deploy_config.nodes)
    total_gpus = sum(n.gpus for n in deploy_config.nodes)
    total_memory = sum(n.memory_gb for n in deploy_config.nodes)

    print(f"\nTotal Resources:")
    print(f"  CPUs: {total_cpus}")
    print(f"  GPUs: {total_gpus}")
    print(f"  Memory: {total_memory} GB")

    # Create deployer
    deployer = MultiNodeDeployer(deploy_config)

    # Show node status
    print("\nNode Status:")
    for node in deployer.get_node_status():
        print(f"  {node['hostname']}: {node['status']}")
        print(f"    CPUs: {node['cpus']}, Memory: {node['memory_gb']} GB")

    # Calculate load distribution
    print("\nLoad Distribution (for 100 tasks):")
    distribution = deployer.calculate_load_distribution(100)
    for hostname, task_count in distribution.items():
        print(f"  {hostname}: {task_count} tasks")


def demo_performance_testing() -> None:
    """
    Demo: Performance Testing

    Shows how to run performance and scalability tests.
    """
    print("\n" + "=" * 60)
    print("Demo: Performance Testing")
    print("=" * 60 + "\n")

    from hpc_performance_testing import (
        PerformanceTestSuite,
        LoadTestConfig,
        ScalabilityTestConfig,
    )

    # Create test suite
    test_suite = PerformanceTestSuite()

    # Configure load test
    load_config = LoadTestConfig(
        target_operations=100,
        duration_seconds=5,
        concurrent_users=4,
    )

    print("Running load test...")
    print(f"  Target ops: {load_config.target_operations}")
    print(f"  Duration: {load_config.duration_seconds}s")
    print(f"  Concurrent users: {load_config.concurrent_users}")

    # Run load test
    load_results = test_suite.run_load_test(load_config)

    print("\nLoad Test Results:")
    print(f"  Total operations: {load_results['total_operations']}")
    print(f"  Successful: {load_results['successful']}")
    print(f"  Failed: {load_results['failed']}")
    print(f"  Avg latency: {load_results['avg_latency_ms']:.2f}ms")
    print(f"  P95 latency: {load_results['p95_latency_ms']:.2f}ms")
    print(f"  Throughput: {load_results['throughput_ops_sec']:.2f} ops/sec")

    # Configure scalability test
    scale_config = ScalabilityTestConfig(
        min_workers=1,
        max_workers=4,
        step=1,
        operations_per_step=50,
    )

    print("\nRunning scalability test...")
    scale_results = test_suite.run_scalability_test(scale_config)

    print("\nScalability Results:")
    for result in scale_results:
        efficiency = result['parallel_efficiency'] * 100
        print(f"  {result['workers']} workers: {result['throughput']:.2f} ops/sec (efficiency: {efficiency:.1f}%)")


def demo_extensions() -> None:
    """
    Demo: HPC Extensions

    Shows how to create and use HPC extensions.
    """
    print("\n" + "=" * 60)
    print("Demo: HPC Extensions")
    print("=" * 60 + "\n")

    from hpc_extensions import (
        ExtensionRegistry,
        BaseExtension,
        SchedulerExtension,
        MetricsExtension,
    )

    # Create extension registry
    registry = ExtensionRegistry()

    print("Available extension types:")
    print("  - SchedulerExtension: Custom job schedulers")
    print("  - MetricsExtension: Custom metrics collection")
    print("  - StorageExtension: Custom storage backends")

    # Register built-in extensions
    metrics_ext = MetricsExtension()
    registry.register(metrics_ext)

    print(f"\nRegistered extensions: {registry.list_extensions()}")

    # Demonstrate extension hooks
    print("\nExtension hooks:")
    print("  - on_job_submit: Called when job is submitted")
    print("  - on_job_complete: Called when job completes")
    print("  - on_error: Called on errors")
    print("  - collect_metrics: Collect custom metrics")

    # Show metrics
    print("\nCollected metrics:")
    metrics = metrics_ext.collect_metrics()
    for key, value in list(metrics.items())[:5]:
        print(f"  {key}: {value}")


def demo_imaging_workflow() -> None:
    """
    Demo: Complete Imaging Workflow

    Shows a complete negative space imaging workflow using HPC.
    """
    print("\n" + "=" * 60)
    print("Demo: Complete Imaging Workflow")
    print("=" * 60 + "\n")

    # Create sample image data
    print("Step 1: Creating sample image data")
    image_data = np.random.rand(1024, 1024, 3).astype(np.float32)
    print(f"  Image shape: {image_data.shape}")
    print(f"  Data type: {image_data.dtype}")

    # Simulate preprocessing
    print("\nStep 2: Preprocessing")
    gray = np.mean(image_data, axis=2)
    normalized = (gray - gray.min()) / (gray.max() - gray.min())
    print(f"  Converted to grayscale: {gray.shape}")
    print(f"  Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    # Simulate negative space detection
    print("\nStep 3: Negative Space Detection")
    threshold = 0.3
    negative_mask = normalized < threshold
    negative_ratio = negative_mask.sum() / negative_mask.size
    print(f"  Threshold: {threshold}")
    print(f"  Negative space ratio: {negative_ratio:.2%}")

    # Simulate feature extraction
    print("\nStep 4: Feature Extraction")
    from scipy import ndimage
    labeled, num_regions = ndimage.label(negative_mask)
    print(f"  Detected regions: {num_regions}")

    region_sizes = ndimage.sum(negative_mask, labeled, range(1, num_regions + 1))
    avg_size = np.mean(region_sizes) if len(region_sizes) > 0 else 0
    print(f"  Average region size: {avg_size:.0f} pixels")

    # Simulate HPC distribution
    print("\nStep 5: HPC Task Distribution")
    num_workers = 4
    tasks_per_worker = num_regions // num_workers
    print(f"  Workers: {num_workers}")
    print(f"  Tasks per worker: {tasks_per_worker}")
    print(f"  Estimated speedup: {min(num_workers, num_regions)}x")

    print("\n✓ Imaging workflow completed")


async def run_all_demos() -> None:
    """Run all demonstration scripts."""
    print("\n" + "#" * 60)
    print("#" + " " * 18 + "HPC DEMO SUITE" + " " * 18 + "#")
    print("#" * 60)

    demos = [
        ("Basic Configuration", demo_basic_configuration),
        ("Benchmarking", demo_benchmarking),
        ("Job Submission", demo_job_submission),
        ("Multi-Node Deployment", demo_multi_node_deployment),
        ("Performance Testing", demo_performance_testing),
        ("Extensions", demo_extensions),
        ("Imaging Workflow", demo_imaging_workflow),
    ]

    for name, demo_func in demos:
        try:
            if asyncio.iscoroutinefunction(demo_func):
                await demo_func()
            else:
                demo_func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            logger.exception(f"Demo error: {name}")

    print("\n" + "#" * 60)
    print("#" + " " * 15 + "DEMO SUITE COMPLETE" + " " * 14 + "#")
    print("#" * 60 + "\n")


def main() -> None:
    """Main entry point for demos."""
    import argparse

    parser = argparse.ArgumentParser(
        description="HPC Demo Scripts for Negative Space Imaging"
    )
    parser.add_argument(
        "--demo",
        choices=[
            "config",
            "benchmark",
            "job",
            "multinode",
            "performance",
            "extensions",
            "imaging",
            "all",
        ],
        default="all",
        help="Demo to run (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    demo_map = {
        "config": demo_basic_configuration,
        "benchmark": demo_benchmarking,
        "job": demo_job_submission,
        "multinode": demo_multi_node_deployment,
        "performance": demo_performance_testing,
        "extensions": demo_extensions,
        "imaging": demo_imaging_workflow,
    }

    if args.demo == "all":
        asyncio.run(run_all_demos())
    else:
        demo_func = demo_map[args.demo]
        if asyncio.iscoroutinefunction(demo_func):
            asyncio.run(demo_func())
        else:
            demo_func()


if __name__ == "__main__":
    main()
