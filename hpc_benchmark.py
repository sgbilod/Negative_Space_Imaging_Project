#!/usr/bin/env python
"""
HPC Benchmarking Suite
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides comprehensive benchmarking capabilities for the
Negative Space Imaging HPC system, including CPU, GPU, memory, and
distributed computing benchmarks.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    category: str
    duration_seconds: float
    operations_per_second: float
    memory_used_mb: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "duration_seconds": self.duration_seconds,
            "operations_per_second": self.operations_per_second,
            "memory_used_mb": self.memory_used_mb,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class SystemInfo:
    """System information for benchmark context."""
    hostname: str
    platform: str
    python_version: str
    cpu_count: int
    cpu_model: str
    memory_total_gb: float
    gpu_available: bool
    gpu_count: int
    gpu_models: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hostname": self.hostname,
            "platform": self.platform,
            "python_version": self.python_version,
            "cpu_count": self.cpu_count,
            "cpu_model": self.cpu_model,
            "memory_total_gb": self.memory_total_gb,
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "gpu_models": self.gpu_models,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    system_info: SystemInfo
    results: List[BenchmarkResult]
    total_duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "system_info": self.system_info.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "total_duration_seconds": self.total_duration_seconds,
            "timestamp": self.timestamp,
            "summary": self.get_summary(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        categories: Dict[str, List[BenchmarkResult]] = {}
        for result in successful:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        return {
            "total_benchmarks": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "categories": {
                cat: {
                    "count": len(results),
                    "avg_duration": sum(r.duration_seconds for r in results) / len(results),
                    "total_ops_per_sec": sum(r.operations_per_second for r in results),
                }
                for cat, results in categories.items()
            },
        }

    def save_json(self, path: Union[str, Path]) -> None:
        """Save report to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Report saved to {path}")

    def save_html(self, path: Union[str, Path]) -> None:
        """Save report to HTML file."""
        path = Path(path)
        html = self._generate_html()
        with open(path, "w") as f:
            f.write(html)
        logger.info(f"HTML report saved to {path}")

    def _generate_html(self) -> str:
        """Generate HTML report."""
        summary = self.get_summary()
        rows = ""
        for result in self.results:
            status = "✓" if result.success else "✗"
            rows += f"""
            <tr class="{'success' if result.success else 'error'}">
                <td>{status}</td>
                <td>{result.name}</td>
                <td>{result.category}</td>
                <td>{result.duration_seconds:.4f}s</td>
                <td>{result.operations_per_second:.2f}</td>
                <td>{result.memory_used_mb:.2f} MB</td>
            </tr>
            """

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>HPC Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .success {{ background-color: #dff0d8; }}
        .error {{ background-color: #f2dede; }}
        .system-info {{ background: #e3f2fd; padding: 15px; margin: 20px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>HPC Benchmark Report</h1>
    <p>Generated: {self.timestamp}</p>

    <div class="system-info">
        <h2>System Information</h2>
        <p><strong>Hostname:</strong> {self.system_info.hostname}</p>
        <p><strong>Platform:</strong> {self.system_info.platform}</p>
        <p><strong>CPU:</strong> {self.system_info.cpu_model} ({self.system_info.cpu_count} cores)</p>
        <p><strong>Memory:</strong> {self.system_info.memory_total_gb:.2f} GB</p>
        <p><strong>GPU:</strong> {'Available' if self.system_info.gpu_available else 'Not Available'}
            {' - ' + ', '.join(self.system_info.gpu_models) if self.system_info.gpu_models else ''}</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Benchmarks:</strong> {summary['total_benchmarks']}</p>
        <p><strong>Successful:</strong> {summary['successful']}</p>
        <p><strong>Failed:</strong> {summary['failed']}</p>
        <p><strong>Total Duration:</strong> {self.total_duration_seconds:.2f}s</p>
    </div>

    <h2>Benchmark Results</h2>
    <table>
        <tr>
            <th>Status</th>
            <th>Name</th>
            <th>Category</th>
            <th>Duration</th>
            <th>Ops/sec</th>
            <th>Memory</th>
        </tr>
        {rows}
    </table>
</body>
</html>
"""


class HPCBenchmark:
    """
    HPC Benchmarking Suite.

    Provides comprehensive benchmarks for CPU, GPU, memory, and
    distributed computing performance.

    Example:
        >>> benchmark = HPCBenchmark()
        >>> report = benchmark.run_all()
        >>> report.save_json("benchmark_report.json")
    """

    def __init__(
        self,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
        data_sizes: Optional[List[int]] = None
    ):
        """
        Initialize the benchmarking suite.

        Args:
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
            data_sizes: List of data sizes to test
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.data_sizes = data_sizes or [1000, 10000, 100000]
        self.results: List[BenchmarkResult] = []
        self.system_info = self._collect_system_info()

    def _collect_system_info(self) -> SystemInfo:
        """Collect system information."""
        try:
            import psutil
            memory_total_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            memory_total_gb = 0.0

        gpu_available = False
        gpu_count = 0
        gpu_models: List[str] = []

        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                gpu_models = [
                    torch.cuda.get_device_name(i) for i in range(gpu_count)
                ]
        except ImportError:
            pass

        cpu_model = "Unknown"
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_model = line.split(":")[1].strip()
                            break
        except Exception:
            pass

        return SystemInfo(
            hostname=platform.node(),
            platform=f"{platform.system()} {platform.release()}",
            python_version=platform.python_version(),
            cpu_count=multiprocessing.cpu_count(),
            cpu_model=cpu_model,
            memory_total_gb=memory_total_gb,
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_models=gpu_models,
        )

    def _measure_memory(self) -> float:
        """Measure current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 ** 2)
        except ImportError:
            return 0.0

    def _run_benchmark(
        self,
        name: str,
        category: str,
        func: Callable[[], Any],
        iterations: Optional[int] = None
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        iterations = iterations or self.benchmark_iterations
        memory_before = self._measure_memory()

        # Warmup
        for _ in range(self.warmup_iterations):
            try:
                func()
            except Exception:
                pass

        # Benchmark
        start_time = time.perf_counter()
        for _ in range(iterations):
            try:
                func()
            except Exception as e:
                return BenchmarkResult(
                    name=name,
                    category=category,
                    duration_seconds=0,
                    operations_per_second=0,
                    memory_used_mb=0,
                    success=False,
                    error_message=str(e),
                )

        end_time = time.perf_counter()
        duration = end_time - start_time
        memory_after = self._measure_memory()

        return BenchmarkResult(
            name=name,
            category=category,
            duration_seconds=duration,
            operations_per_second=iterations / duration if duration > 0 else 0,
            memory_used_mb=max(0, memory_after - memory_before),
        )

    def run_cpu_benchmarks(self) -> List[BenchmarkResult]:
        """Run CPU benchmarks."""
        results = []

        # Matrix multiplication benchmark
        for size in self.data_sizes:
            def matrix_mult() -> None:
                a = np.random.rand(size // 10, size // 10)
                b = np.random.rand(size // 10, size // 10)
                _ = np.dot(a, b)

            result = self._run_benchmark(
                f"matrix_multiplication_{size}",
                "cpu",
                matrix_mult,
            )
            result.metadata["matrix_size"] = size // 10
            results.append(result)

        # FFT benchmark
        for size in self.data_sizes:
            def fft_benchmark() -> None:
                data = np.random.rand(size)
                _ = np.fft.fft(data)

            result = self._run_benchmark(
                f"fft_{size}",
                "cpu",
                fft_benchmark,
            )
            result.metadata["data_size"] = size
            results.append(result)

        # Sorting benchmark
        for size in self.data_sizes:
            def sort_benchmark() -> None:
                data = np.random.rand(size)
                _ = np.sort(data)

            result = self._run_benchmark(
                f"sort_{size}",
                "cpu",
                sort_benchmark,
            )
            result.metadata["data_size"] = size
            results.append(result)

        return results

    def run_gpu_benchmarks(self) -> List[BenchmarkResult]:
        """Run GPU benchmarks."""
        results = []

        if not self.system_info.gpu_available:
            logger.warning("GPU not available, skipping GPU benchmarks")
            return results

        try:
            import torch

            device = torch.device("cuda")

            # Matrix multiplication on GPU
            for size in self.data_sizes:
                def gpu_matmul() -> None:
                    a = torch.rand(size // 10, size // 10, device=device)
                    b = torch.rand(size // 10, size // 10, device=device)
                    _ = torch.mm(a, b)
                    torch.cuda.synchronize()

                result = self._run_benchmark(
                    f"gpu_matrix_multiplication_{size}",
                    "gpu",
                    gpu_matmul,
                )
                result.metadata["matrix_size"] = size // 10
                results.append(result)

            # Convolution benchmark
            for size in [64, 128, 256]:
                def conv_benchmark() -> None:
                    x = torch.rand(16, 3, size, size, device=device)
                    conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
                    _ = conv(x)
                    torch.cuda.synchronize()

                result = self._run_benchmark(
                    f"gpu_conv2d_{size}",
                    "gpu",
                    conv_benchmark,
                )
                result.metadata["input_size"] = size
                results.append(result)

        except ImportError:
            logger.warning("PyTorch not available for GPU benchmarks")
        except Exception as e:
            logger.error(f"GPU benchmark error: {e}")

        return results

    def run_memory_benchmarks(self) -> List[BenchmarkResult]:
        """Run memory benchmarks."""
        results = []

        # Memory allocation benchmark
        for size_mb in [10, 100, 500]:
            def memory_alloc() -> None:
                data = np.zeros((size_mb * 1024 * 1024 // 8,), dtype=np.float64)
                _ = data.sum()

            result = self._run_benchmark(
                f"memory_allocation_{size_mb}mb",
                "memory",
                memory_alloc,
                iterations=5,
            )
            result.metadata["allocation_size_mb"] = size_mb
            results.append(result)

        # Memory copy benchmark
        for size_mb in [10, 100]:
            def memory_copy() -> None:
                data = np.random.rand(size_mb * 1024 * 1024 // 8)
                _ = np.copy(data)

            result = self._run_benchmark(
                f"memory_copy_{size_mb}mb",
                "memory",
                memory_copy,
                iterations=5,
            )
            result.metadata["copy_size_mb"] = size_mb
            results.append(result)

        return results

    def run_distributed_benchmarks(self) -> List[BenchmarkResult]:
        """Run distributed computing benchmarks."""
        results = []

        # Multiprocessing benchmark
        def parallel_task(x: int) -> int:
            return sum(range(x))

        for num_workers in [2, 4, 8]:
            actual_workers = min(num_workers, multiprocessing.cpu_count())

            def multiprocessing_benchmark() -> None:
                with multiprocessing.Pool(actual_workers) as pool:
                    _ = pool.map(parallel_task, [10000] * 100)

            result = self._run_benchmark(
                f"multiprocessing_{actual_workers}_workers",
                "distributed",
                multiprocessing_benchmark,
                iterations=3,
            )
            result.metadata["num_workers"] = actual_workers
            results.append(result)

        return results

    def run_imaging_benchmarks(self) -> List[BenchmarkResult]:
        """Run imaging-specific benchmarks."""
        results = []

        # Image processing benchmark
        for size in [256, 512, 1024]:
            def image_processing() -> None:
                image = np.random.rand(size, size, 3).astype(np.float32)
                # Simulate negative space analysis operations
                gray = np.mean(image, axis=2)
                edges = np.gradient(gray)
                _ = np.fft.fft2(gray)

            result = self._run_benchmark(
                f"image_processing_{size}x{size}",
                "imaging",
                image_processing,
            )
            result.metadata["image_size"] = f"{size}x{size}"
            results.append(result)

        # Batch image processing
        for batch_size in [8, 16, 32]:
            def batch_processing() -> None:
                batch = np.random.rand(batch_size, 256, 256, 3).astype(np.float32)
                for i in range(batch_size):
                    gray = np.mean(batch[i], axis=2)
                    _ = np.gradient(gray)

            result = self._run_benchmark(
                f"batch_processing_{batch_size}",
                "imaging",
                batch_processing,
            )
            result.metadata["batch_size"] = batch_size
            results.append(result)

        return results

    def run_all(self) -> BenchmarkReport:
        """
        Run all benchmarks.

        Returns:
            Complete benchmark report
        """
        logger.info("Starting HPC benchmark suite")
        start_time = time.perf_counter()

        all_results = []

        logger.info("Running CPU benchmarks...")
        all_results.extend(self.run_cpu_benchmarks())

        logger.info("Running GPU benchmarks...")
        all_results.extend(self.run_gpu_benchmarks())

        logger.info("Running memory benchmarks...")
        all_results.extend(self.run_memory_benchmarks())

        logger.info("Running distributed benchmarks...")
        all_results.extend(self.run_distributed_benchmarks())

        logger.info("Running imaging benchmarks...")
        all_results.extend(self.run_imaging_benchmarks())

        end_time = time.perf_counter()
        total_duration = end_time - start_time

        logger.info(f"Benchmark suite completed in {total_duration:.2f}s")

        return BenchmarkReport(
            system_info=self.system_info,
            results=all_results,
            total_duration_seconds=total_duration,
        )

    def run_quick(self) -> BenchmarkReport:
        """
        Run a quick subset of benchmarks.

        Returns:
            Benchmark report with quick results
        """
        logger.info("Starting quick benchmark")
        start_time = time.perf_counter()

        # Save original settings
        orig_sizes = self.data_sizes
        orig_iterations = self.benchmark_iterations

        # Use smaller settings
        self.data_sizes = [1000]
        self.benchmark_iterations = 3

        results = []
        results.extend(self.run_cpu_benchmarks())
        results.extend(self.run_memory_benchmarks())

        # Restore settings
        self.data_sizes = orig_sizes
        self.benchmark_iterations = orig_iterations

        end_time = time.perf_counter()

        return BenchmarkReport(
            system_info=self.system_info,
            results=results,
            total_duration_seconds=end_time - start_time,
        )


def run_benchmark(
    output_path: Optional[str] = None,
    quick: bool = False
) -> BenchmarkReport:
    """
    Run HPC benchmarks and optionally save results.

    Args:
        output_path: Optional path to save results
        quick: Whether to run quick benchmarks only

    Returns:
        Benchmark report
    """
    benchmark = HPCBenchmark()

    if quick:
        report = benchmark.run_quick()
    else:
        report = benchmark.run_all()

    if output_path:
        path = Path(output_path)
        if path.suffix == ".html":
            report.save_html(path)
        else:
            report.save_json(path)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = run_benchmark("benchmark_report.json")
    print(f"Benchmarks completed: {len(report.results)} tests")
