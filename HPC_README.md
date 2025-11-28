# HPC Module Documentation

## Negative Space Imaging Project - High Performance Computing

Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [API Reference](#api-reference)
6. [Tutorials](#tutorials)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Overview

The HPC (High Performance Computing) module provides enterprise-grade distributed computing capabilities for the Negative Space Imaging Project. It enables efficient processing of large-scale image analysis tasks across multiple compute nodes.

### Key Features

- **Multi-Backend Support**: SLURM, PBS, and LSF job schedulers
- **Auto-Configuration**: Automatic environment detection and setup
- **Distributed Processing**: Scale analysis across multiple nodes
- **Benchmarking Suite**: Comprehensive performance testing tools
- **Load Balancing**: Intelligent task distribution
- **Health Monitoring**: Real-time cluster monitoring
- **Extension System**: Plugin architecture for customization

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HPC Integration                       │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Config     │  │  Benchmark   │  │  Extensions  │  │
│  │   Manager    │  │    Suite     │  │   Registry   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │    Job       │  │   Multi-Node │  │ Performance  │  │
│  │  Scheduler   │  │   Deployer   │  │   Testing    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────┤
│                   Backend Adapters                       │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐       │
│  │ SLURM  │  │  PBS   │  │  LSF   │  │ Local  │       │
│  └────────┘  └────────┘  └────────┘  └────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- NumPy >= 1.24.0
- PyYAML >= 6.0.1
- Optional: PyTorch (for GPU benchmarks)
- Optional: psutil (for system metrics)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Installation

```python
from hpc_config import HPCConfig, HPCBackend

config = HPCConfig()
print(f"Detected backend: {config.backend.value}")
print(f"Configuration valid: {config.validate()}")
```

---

## Quick Start

### 1. Basic Configuration

```python
from hpc_config import HPCConfig, HPCBackend, ComputeNodeConfig

# Auto-detect HPC environment
config = HPCConfig()

# Or configure manually
config = HPCConfig(
    backend=HPCBackend.SLURM,
    cluster_name="my-cluster",
    compute=ComputeNodeConfig(
        min_nodes=2,
        max_nodes=16,
        cpus_per_node=32,
        gpus_per_node=4,
    ),
)

# Validate configuration
config.validate()
```

### 2. Submit Analysis Job

```python
import asyncio
from hpc_integration import HPCIntegration, AnalysisTask

async def run_analysis():
    hpc = HPCIntegration()
    
    task = AnalysisTask(
        task_id="analysis_001",
        image_path="/data/images/galaxy.fits",
        config={"threshold": 0.3, "max_regions": 100},
    )
    
    job = await hpc.submit_analysis(task, wait=True)
    print(f"Job status: {job.status.value}")
    
    result = await hpc.collect_result(job.job_id)
    print(f"Regions detected: {result.result_data['regions_detected']}")

asyncio.run(run_analysis())
```

### 3. Run Benchmarks

```python
from hpc_benchmark import HPCBenchmark

benchmark = HPCBenchmark()
report = benchmark.run_quick()

print(f"CPU Performance: {report.results[0].operations_per_second:.2f} ops/sec")
report.save_json("benchmark_report.json")
```

---

## Configuration

### Configuration File Format (YAML)

```yaml
backend: slurm
cluster_name: negative-space-hpc
environment: production

compute:
  min_nodes: 1
  max_nodes: 16
  cpus_per_node: 32
  gpus_per_node: 4
  memory_gb: 128
  exclusive: false

memory:
  per_node: 128
  per_task: 16
  reserved: 4

scheduling:
  default_queue: batch
  default_wall_time: "04:00:00"
  default_cpus: 8
  default_memory_gb: 32
  retry_count: 3

queues:
  - name: fast
    priority: 200
    max_wall_time: "01:00:00"
  - name: batch
    priority: 100
    max_wall_time: "24:00:00"
  - name: gpu
    priority: 150
    partition: gpu-nodes

network:
  interconnect: infiniband
  bandwidth_gbps: 100
  mpi_enabled: true

storage:
  scratch_path: /scratch
  project_path: /project
  parallel_fs: true

monitoring:
  enabled: true
  metrics_interval: 30
  prometheus_enabled: true

modules:
  - python/3.11
  - cuda/12.0
  - openmpi/4.1

environment_variables:
  OMP_NUM_THREADS: "8"
  CUDA_VISIBLE_DEVICES: "0,1,2,3"
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HPC_BACKEND` | Job scheduler backend | auto |
| `HPC_CLUSTER_NAME` | Cluster name | negative-space-hpc |
| `HPC_MIN_NODES` | Minimum nodes | 1 |
| `HPC_MAX_NODES` | Maximum nodes | 16 |
| `HPC_CPUS_PER_NODE` | CPUs per node | 32 |
| `HPC_GPUS_PER_NODE` | GPUs per node | 0 |
| `HPC_MEMORY_PER_NODE` | Memory per node (GB) | 64 |

---

## API Reference

### hpc_config.py

#### HPCConfig

Main configuration class.

```python
class HPCConfig:
    """
    HPC configuration management.
    
    Attributes:
        backend: HPC scheduler backend (SLURM, PBS, LSF, LOCAL)
        cluster_name: Name of the cluster
        compute: Compute node configuration
        memory: Memory configuration
        scheduling: Job scheduling configuration
        queues: List of queue configurations
    """
    
    @classmethod
    def from_file(cls, path: str) -> HPCConfig:
        """Load configuration from YAML file."""
    
    @classmethod
    def from_environment(cls) -> HPCConfig:
        """Create configuration from environment variables."""
    
    def validate(self) -> bool:
        """Validate configuration settings."""
    
    def get_job_script_header(self, job_name: str) -> str:
        """Generate job script header for backend."""
```

### hpc_integration.py

#### HPCIntegration

Main integration class for job management.

```python
class HPCIntegration:
    """
    HPC integration with Negative Space Imaging.
    
    Methods:
        submit_analysis: Submit an analysis task
        submit_batch: Submit multiple tasks
        wait_for_job: Wait for job completion
        collect_result: Collect job results
        cancel_job: Cancel a running job
    """
    
    async def submit_analysis(
        self,
        task: AnalysisTask,
        wait: bool = False
    ) -> HPCJob:
        """Submit an analysis task to HPC cluster."""
    
    async def submit_batch(
        self,
        tasks: List[AnalysisTask],
        max_concurrent: int = 10
    ) -> List[HPCJob]:
        """Submit multiple analysis tasks."""
```

### hpc_benchmark.py

#### HPCBenchmark

Benchmarking suite.

```python
class HPCBenchmark:
    """
    HPC benchmarking suite.
    
    Methods:
        run_all: Run complete benchmark suite
        run_quick: Run quick benchmarks
        run_cpu_benchmarks: Run CPU-specific benchmarks
        run_gpu_benchmarks: Run GPU-specific benchmarks
    """
    
    def run_all(self) -> BenchmarkReport:
        """Run all benchmarks and generate report."""
    
    def run_quick(self) -> BenchmarkReport:
        """Run quick subset of benchmarks."""
```

### hpc_multi_node_deploy.py

#### MultiNodeDeployer

Multi-node cluster deployment.

```python
class MultiNodeDeployer:
    """
    Multi-node deployment manager.
    
    Methods:
        start: Start deployment
        stop: Stop deployment
        add_node: Add a node to cluster
        remove_node: Remove a node
        select_node: Select node for task
    """
```

### hpc_extensions.py

#### ExtensionRegistry

Extension management.

```python
class ExtensionRegistry:
    """
    Extension registry for custom plugins.
    
    Methods:
        register: Register an extension
        unregister: Unregister an extension
        get: Get extension by name
        get_by_type: Get extensions by type
    """
```

---

## Tutorials

### Tutorial 1: Basic Image Analysis Pipeline

```python
import asyncio
from hpc_config import HPCConfig
from hpc_integration import HPCIntegration, AnalysisTask

async def analyze_images(image_paths: list):
    # Initialize HPC
    config = HPCConfig()
    hpc = HPCIntegration(config)
    
    # Create tasks
    tasks = [
        AnalysisTask(
            task_id=f"task_{i}",
            image_path=path,
        )
        for i, path in enumerate(image_paths)
    ]
    
    # Submit batch
    jobs = await hpc.submit_batch(tasks)
    
    # Wait and collect results
    results = {}
    for job in jobs:
        await hpc.wait_for_job(job.job_id)
        result = await hpc.collect_result(job.job_id)
        results[job.metadata['task_id']] = result
    
    return results

# Run
images = ["/data/img1.fits", "/data/img2.fits", "/data/img3.fits"]
results = asyncio.run(analyze_images(images))
```

### Tutorial 2: Custom Extension

```python
from hpc_extensions import (
    BaseExtension,
    ExtensionMetadata,
    ExtensionType,
    HookType,
    HookContext,
    HookResult,
    register_extension,
)

class LoggingExtension(BaseExtension):
    """Custom extension that logs all job events."""
    
    def get_metadata(self) -> ExtensionMetadata:
        return ExtensionMetadata(
            name="logging-extension",
            version="1.0.0",
            author="My Team",
            description="Logs all job events",
            extension_type=ExtensionType.MONITORING,
        )
    
    def initialize(self, config: dict) -> None:
        self._config = config
        self._initialized = True
        
        # Register hooks
        self.register_hook(HookType.POST_SUBMIT, self._on_submit)
        self.register_hook(HookType.ON_COMPLETE, self._on_complete)
    
    def shutdown(self) -> None:
        self._initialized = False
    
    def _on_submit(self, context: HookContext) -> HookResult:
        print(f"Job submitted: {context.job_id}")
        return HookResult(success=True)
    
    def _on_complete(self, context: HookContext) -> HookResult:
        print(f"Job completed: {context.job_id}")
        return HookResult(success=True)

# Register extension
register_extension(LoggingExtension())
```

### Tutorial 3: Performance Testing

```python
from hpc_performance_testing import (
    PerformanceTestSuite,
    LoadTestConfig,
    ScalabilityTestConfig,
)

# Create test suite
suite = PerformanceTestSuite()

# Define custom workload
def my_workload():
    import numpy as np
    data = np.random.rand(1000, 1000)
    return np.fft.fft2(data)

suite.set_workload(my_workload)

# Run load test
load_config = LoadTestConfig(
    target_operations=1000,
    concurrent_users=10,
    duration_seconds=60,
)
load_results = suite.run_load_test(load_config)

print(f"Throughput: {load_results['throughput_ops_sec']:.2f} ops/sec")
print(f"P95 Latency: {load_results['p95_latency_ms']:.2f} ms")

# Run scalability test
scale_config = ScalabilityTestConfig(
    min_workers=1,
    max_workers=16,
    step=2,
)
scale_results = suite.run_scalability_test(scale_config)

for result in scale_results:
    print(f"{result['workers']} workers: "
          f"{result['throughput']:.2f} ops/sec, "
          f"efficiency: {result['parallel_efficiency']*100:.1f}%")
```

---

## Troubleshooting

### Common Issues

#### 1. Job Submission Fails

**Symptom**: Jobs fail to submit with scheduler errors.

**Solutions**:
- Verify scheduler is accessible: `which sbatch` (SLURM)
- Check queue availability: `sinfo` (SLURM)
- Verify resource requests don't exceed limits
- Check account/project configuration

#### 2. Jobs Stay Pending

**Symptom**: Jobs remain in pending state.

**Solutions**:
- Check queue status: `squeue` (SLURM)
- Verify resource availability
- Check for dependency issues
- Review job priority

#### 3. Memory Errors

**Symptom**: Jobs fail with out-of-memory errors.

**Solutions**:
- Increase memory allocation in config
- Enable memory optimization
- Process data in smaller chunks
- Use memory-efficient algorithms

#### 4. GPU Not Detected

**Symptom**: GPU benchmarks report no GPU available.

**Solutions**:
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA support: `torch.cuda.is_available()`
- Ensure GPU modules are loaded
- Verify GPU allocation in job script

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from hpc_config import HPCConfig
config = HPCConfig()  # Will show detailed debug output
```

### Log Files

Check job logs:
- Output: `<work_dir>/<job_id>_output.log`
- Error: `<work_dir>/<job_id>_error.log`
- Results: `<work_dir>/<job_id>_output.json`

---

## Best Practices

### 1. Resource Estimation

- Start with conservative resource requests
- Monitor actual usage and adjust
- Use job arrays for similar tasks
- Set appropriate wall time limits

### 2. Data Management

- Use scratch storage for temporary files
- Stage input data before job starts
- Archive results to permanent storage
- Clean up temporary files

### 3. Error Handling

- Implement retry logic for transient failures
- Set up email notifications for job status
- Monitor cluster health
- Keep detailed logs

### 4. Performance Optimization

- Profile code before scaling
- Use appropriate batch sizes
- Leverage GPU when available
- Minimize data transfer between nodes

### 5. Security

- Use secure communication channels
- Protect credentials and API keys
- Follow cluster security policies
- Audit access and usage

---

## Support

For issues and feature requests, please refer to the project documentation or contact the development team.

---

*Last updated: 2025*
