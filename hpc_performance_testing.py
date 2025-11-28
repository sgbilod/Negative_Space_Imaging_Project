#!/usr/bin/env python
"""
HPC Performance Testing Suite
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

Comprehensive performance testing for HPC components including load testing,
stress testing, scalability testing, and performance regression detection.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    target_operations: int = 1000
    duration_seconds: int = 60
    concurrent_users: int = 10
    ramp_up_seconds: int = 10
    ramp_down_seconds: int = 5
    think_time_ms: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_operations": self.target_operations,
            "duration_seconds": self.duration_seconds,
            "concurrent_users": self.concurrent_users,
            "ramp_up_seconds": self.ramp_up_seconds,
            "ramp_down_seconds": self.ramp_down_seconds,
            "think_time_ms": self.think_time_ms,
        }


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    initial_load: int = 10
    max_load: int = 1000
    load_increment: int = 50
    increment_interval_seconds: int = 30
    failure_threshold: float = 0.05  # 5% failure rate
    memory_limit_mb: int = 4096
    cpu_limit_percent: int = 95

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initial_load": self.initial_load,
            "max_load": self.max_load,
            "load_increment": self.load_increment,
            "increment_interval_seconds": self.increment_interval_seconds,
            "failure_threshold": self.failure_threshold,
            "memory_limit_mb": self.memory_limit_mb,
            "cpu_limit_percent": self.cpu_limit_percent,
        }


@dataclass
class ScalabilityTestConfig:
    """Configuration for scalability testing."""
    min_workers: int = 1
    max_workers: int = 16
    step: int = 2
    operations_per_step: int = 1000
    warmup_operations: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "step": self.step,
            "operations_per_step": self.operations_per_step,
            "warmup_operations": self.warmup_operations,
        }


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    name: str
    throughput_ops_sec: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "throughput_ops_sec": self.throughput_ops_sec,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class TestResult:
    """Result of a performance test."""
    test_name: str
    success: bool
    total_operations: int
    successful: int
    failed: int
    duration_seconds: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_ops_sec: float
    error_messages: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "success": self.success,
            "total_operations": self.total_operations,
            "successful": self.successful,
            "failed": self.failed,
            "duration_seconds": self.duration_seconds,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_ops_sec": self.throughput_ops_sec,
            "error_messages": self.error_messages,
            "timestamp": self.timestamp,
        }


class PerformanceTestSuite:
    """
    Comprehensive performance testing suite for HPC components.

    Provides load testing, stress testing, scalability testing,
    and performance regression detection.

    Example:
        suite = PerformanceTestSuite()
        result = suite.run_load_test(LoadTestConfig())
        print(f"Throughput: {result['throughput_ops_sec']} ops/sec")
    """

    def __init__(self) -> None:
        """Initialize performance test suite."""
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self._workload_func: Optional[Callable] = None

    def set_workload(self, func: Callable[[], Any]) -> None:
        """
        Set the workload function to test.

        Args:
            func: Workload function
        """
        self._workload_func = func

    def _default_workload(self) -> None:
        """Default workload for testing."""
        # Simulate imaging operation
        data = np.random.rand(256, 256)
        _ = np.fft.fft2(data)
        _ = np.gradient(data)

    def _execute_operation(self) -> Tuple[bool, float, Optional[str]]:
        """Execute a single operation and measure latency."""
        workload = self._workload_func or self._default_workload

        start = time.perf_counter()
        try:
            workload()
            latency = (time.perf_counter() - start) * 1000  # ms
            return True, latency, None
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return False, latency, str(e)

    def run_load_test(
        self,
        config: LoadTestConfig
    ) -> Dict[str, Any]:
        """
        Run load test.

        Args:
            config: Load test configuration

        Returns:
            Test results dictionary
        """
        logger.info(f"Starting load test: {config.target_operations} ops, "
                   f"{config.concurrent_users} users")

        latencies: List[float] = []
        errors: List[str] = []
        successful = 0
        failed = 0

        start_time = time.perf_counter()

        def worker(num_ops: int) -> Tuple[List[float], int, int, List[str]]:
            local_latencies = []
            local_success = 0
            local_failed = 0
            local_errors = []

            for _ in range(num_ops):
                success, latency, error = self._execute_operation()
                local_latencies.append(latency)

                if success:
                    local_success += 1
                else:
                    local_failed += 1
                    if error:
                        local_errors.append(error)

                if config.think_time_ms > 0:
                    time.sleep(config.think_time_ms / 1000)

            return local_latencies, local_success, local_failed, local_errors

        # Distribute operations across workers
        ops_per_worker = config.target_operations // config.concurrent_users
        remaining = config.target_operations % config.concurrent_users

        with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            futures = []
            for i in range(config.concurrent_users):
                ops = ops_per_worker + (1 if i < remaining else 0)
                futures.append(executor.submit(worker, ops))

            for future in futures:
                result = future.result()
                latencies.extend(result[0])
                successful += result[1]
                failed += result[2]
                errors.extend(result[3])

        duration = time.perf_counter() - start_time

        return self._calculate_results(
            "load_test",
            latencies,
            successful,
            failed,
            duration,
            errors,
        )

    def run_stress_test(
        self,
        config: StressTestConfig
    ) -> Dict[str, Any]:
        """
        Run stress test.

        Args:
            config: Stress test configuration

        Returns:
            Test results dictionary
        """
        logger.info(f"Starting stress test: {config.initial_load} to {config.max_load} load")

        all_latencies: List[float] = []
        total_successful = 0
        total_failed = 0
        all_errors: List[str] = []

        current_load = config.initial_load
        start_time = time.perf_counter()
        breaking_point = None

        while current_load <= config.max_load:
            logger.info(f"Stress test load: {current_load}")

            # Run operations at current load
            interval_latencies = []
            interval_failed = 0

            for _ in range(current_load):
                success, latency, error = self._execute_operation()
                interval_latencies.append(latency)

                if success:
                    total_successful += 1
                else:
                    total_failed += 1
                    interval_failed += 1
                    if error:
                        all_errors.append(error)

            all_latencies.extend(interval_latencies)

            # Check failure rate
            failure_rate = interval_failed / current_load
            if failure_rate > config.failure_threshold:
                logger.warning(f"Breaking point reached at load {current_load}, "
                             f"failure rate: {failure_rate:.2%}")
                breaking_point = current_load
                break

            current_load += config.load_increment
            time.sleep(1)  # Brief pause between increments

        duration = time.perf_counter() - start_time

        result = self._calculate_results(
            "stress_test",
            all_latencies,
            total_successful,
            total_failed,
            duration,
            all_errors,
        )

        result["breaking_point"] = breaking_point
        result["max_stable_load"] = (
            breaking_point - config.load_increment if breaking_point else config.max_load
        )

        return result

    def run_scalability_test(
        self,
        config: ScalabilityTestConfig
    ) -> List[Dict[str, Any]]:
        """
        Run scalability test.

        Args:
            config: Scalability test configuration

        Returns:
            List of results for each worker count
        """
        logger.info(f"Starting scalability test: {config.min_workers} to {config.max_workers} workers")

        results = []
        baseline_throughput = None

        workers = config.min_workers
        while workers <= config.max_workers:
            logger.info(f"Scalability test: {workers} workers")

            # Warmup
            for _ in range(config.warmup_operations):
                self._execute_operation()

            # Run test
            latencies = []
            start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self._execute_operation)
                    for _ in range(config.operations_per_step)
                ]

                for future in futures:
                    success, latency, _ = future.result()
                    if success:
                        latencies.append(latency)

            duration = time.perf_counter() - start_time
            throughput = len(latencies) / duration if duration > 0 else 0

            if baseline_throughput is None:
                baseline_throughput = throughput

            efficiency = throughput / (baseline_throughput * workers) if baseline_throughput > 0 else 0

            results.append({
                "workers": workers,
                "operations": len(latencies),
                "duration": duration,
                "throughput": throughput,
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                "speedup": throughput / baseline_throughput if baseline_throughput > 0 else 1,
                "parallel_efficiency": efficiency,
            })

            workers += config.step

        return results

    def run_regression_test(
        self,
        test_name: str,
        regression_threshold: float = 0.10
    ) -> Dict[str, Any]:
        """
        Run performance regression test against baseline.

        Args:
            test_name: Name of the test
            regression_threshold: Threshold for regression detection (0.10 = 10%)

        Returns:
            Regression test results
        """
        if test_name not in self.baselines:
            logger.warning(f"No baseline found for {test_name}")
            return {"error": "No baseline found"}

        baseline = self.baselines[test_name]

        # Run current test
        load_config = LoadTestConfig(
            target_operations=100,
            concurrent_users=4,
            duration_seconds=30,
        )
        current = self.run_load_test(load_config)

        # Compare metrics
        throughput_change = (
            (current["throughput_ops_sec"] - baseline.throughput_ops_sec) /
            baseline.throughput_ops_sec
        )

        latency_change = (
            (current["avg_latency_ms"] - baseline.avg_latency_ms) /
            baseline.avg_latency_ms
        )

        # Negative throughput change or positive latency change indicates regression
        has_throughput_regression = throughput_change < -regression_threshold
        has_latency_regression = latency_change > regression_threshold

        return {
            "test_name": test_name,
            "baseline": baseline.to_dict(),
            "current": current,
            "throughput_change": throughput_change,
            "latency_change": latency_change,
            "has_regression": has_throughput_regression or has_latency_regression,
            "regression_details": {
                "throughput_regression": has_throughput_regression,
                "latency_regression": has_latency_regression,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    def set_baseline(
        self,
        test_name: str,
        result: Dict[str, Any]
    ) -> PerformanceBaseline:
        """
        Set performance baseline from test results.

        Args:
            test_name: Name of the test
            result: Test results dictionary

        Returns:
            Created baseline
        """
        baseline = PerformanceBaseline(
            name=test_name,
            throughput_ops_sec=result.get("throughput_ops_sec", 0),
            avg_latency_ms=result.get("avg_latency_ms", 0),
            p95_latency_ms=result.get("p95_latency_ms", 0),
            p99_latency_ms=result.get("p99_latency_ms", 0),
        )

        self.baselines[test_name] = baseline
        logger.info(f"Baseline set for {test_name}")
        return baseline

    def _calculate_results(
        self,
        test_name: str,
        latencies: List[float],
        successful: int,
        failed: int,
        duration: float,
        errors: List[str],
    ) -> Dict[str, Any]:
        """Calculate test result statistics."""
        total = successful + failed

        if not latencies:
            latencies = [0]

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return {
            "test_name": test_name,
            "success": failed == 0,
            "total_operations": total,
            "successful": successful,
            "failed": failed,
            "duration_seconds": duration,
            "avg_latency_ms": statistics.mean(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p50_latency_ms": sorted_latencies[int(n * 0.5)],
            "p95_latency_ms": sorted_latencies[int(n * 0.95)],
            "p99_latency_ms": sorted_latencies[min(int(n * 0.99), n - 1)],
            "throughput_ops_sec": total / duration if duration > 0 else 0,
            "error_rate": failed / total if total > 0 else 0,
            "error_messages": errors[:10],  # First 10 errors
            "timestamp": datetime.utcnow().isoformat(),
        }


class ImageProcessingBenchmark:
    """
    Specialized benchmark for image processing operations.

    Tests various image processing workloads typical in negative space analysis.
    """

    def __init__(self, image_sizes: Optional[List[int]] = None):
        """Initialize image processing benchmark."""
        self.image_sizes = image_sizes or [256, 512, 1024, 2048]

    def benchmark_operations(self) -> Dict[str, Dict[str, float]]:
        """
        Benchmark various image processing operations.

        Returns:
            Dictionary of operation -> size -> throughput
        """
        results = {}

        operations = {
            "grayscale": self._benchmark_grayscale,
            "gradient": self._benchmark_gradient,
            "fft": self._benchmark_fft,
            "threshold": self._benchmark_threshold,
            "morphology": self._benchmark_morphology,
        }

        for op_name, op_func in operations.items():
            results[op_name] = {}
            for size in self.image_sizes:
                throughput = op_func(size)
                results[op_name][str(size)] = throughput
                logger.info(f"{op_name} {size}x{size}: {throughput:.2f} ops/sec")

        return results

    def _benchmark_grayscale(self, size: int) -> float:
        """Benchmark grayscale conversion."""
        image = np.random.rand(size, size, 3).astype(np.float32)

        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = np.mean(image, axis=2)

        duration = time.perf_counter() - start
        return iterations / duration

    def _benchmark_gradient(self, size: int) -> float:
        """Benchmark gradient calculation."""
        image = np.random.rand(size, size).astype(np.float32)

        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = np.gradient(image)

        duration = time.perf_counter() - start
        return iterations / duration

    def _benchmark_fft(self, size: int) -> float:
        """Benchmark FFT operation."""
        image = np.random.rand(size, size).astype(np.float32)

        start = time.perf_counter()
        iterations = 50
        for _ in range(iterations):
            _ = np.fft.fft2(image)

        duration = time.perf_counter() - start
        return iterations / duration

    def _benchmark_threshold(self, size: int) -> float:
        """Benchmark thresholding."""
        image = np.random.rand(size, size).astype(np.float32)

        start = time.perf_counter()
        iterations = 500
        for _ in range(iterations):
            _ = image > 0.5

        duration = time.perf_counter() - start
        return iterations / duration

    def _benchmark_morphology(self, size: int) -> float:
        """Benchmark morphological operations."""
        from scipy import ndimage
        image = (np.random.rand(size, size) > 0.5).astype(np.uint8)

        start = time.perf_counter()
        iterations = 50
        for _ in range(iterations):
            _ = ndimage.binary_erosion(image)
            _ = ndimage.binary_dilation(image)

        duration = time.perf_counter() - start
        return iterations / duration


def run_performance_tests(
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run full performance test suite.

    Args:
        output_dir: Optional directory to save results

    Returns:
        Complete test results
    """
    import json

    suite = PerformanceTestSuite()
    results = {}

    # Load test
    logger.info("Running load test...")
    load_config = LoadTestConfig(
        target_operations=100,
        concurrent_users=4,
        duration_seconds=30,
    )
    results["load_test"] = suite.run_load_test(load_config)

    # Scalability test
    logger.info("Running scalability test...")
    scale_config = ScalabilityTestConfig(
        min_workers=1,
        max_workers=4,
        step=1,
        operations_per_step=50,
    )
    results["scalability_test"] = suite.run_scalability_test(scale_config)

    # Image processing benchmark
    logger.info("Running image processing benchmark...")
    img_benchmark = ImageProcessingBenchmark(image_sizes=[256, 512])
    results["image_processing"] = img_benchmark.benchmark_operations()

    # Save results
    if output_dir:
        output_path = Path(output_dir) / f"perf_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_performance_tests()

    print("\n" + "=" * 60)
    print("PERFORMANCE TEST RESULTS")
    print("=" * 60)

    if "load_test" in results:
        lt = results["load_test"]
        print(f"\nLoad Test:")
        print(f"  Throughput: {lt['throughput_ops_sec']:.2f} ops/sec")
        print(f"  Avg Latency: {lt['avg_latency_ms']:.2f} ms")
        print(f"  P95 Latency: {lt['p95_latency_ms']:.2f} ms")

    if "scalability_test" in results:
        print(f"\nScalability Test:")
        for r in results["scalability_test"]:
            print(f"  {r['workers']} workers: {r['throughput']:.2f} ops/sec "
                  f"(efficiency: {r['parallel_efficiency']*100:.1f}%)")
