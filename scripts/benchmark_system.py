#!/usr/bin/env python
"""
Performance Benchmarking System for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides comprehensive performance testing and monitoring:
1. Image processing speed
2. Memory usage analysis
3. GPU utilization tracking
4. Multi-threading efficiency
5. System resource monitoring
"""

import os
import time
import json
import psutil
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

@dataclass
class BenchmarkResult:
    component: str
    metric: str
    value: float
    unit: str
    timestamp: float
    context: Dict

class PerformanceMonitor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "benchmark_results"
        self.results_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("performance_monitor")
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(
            self.project_root / "logs" / "performance.log"
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def benchmark_image_processing(self, image_path: Path) -> List[BenchmarkResult]:
        """Benchmark image processing performance."""
        results = []

        # Load image
        start_time = time.time()
        img = cv2.imread(str(image_path))
        load_time = time.time() - start_time

        results.append(BenchmarkResult(
            component="image_processing",
            metric="load_time",
            value=load_time,
            unit="seconds",
            timestamp=time.time(),
            context={"image_size": os.path.getsize(image_path)}
        ))

        # Test various operations
        operations = [
            ("resize", lambda: cv2.resize(img, (800, 600))),
            ("blur", lambda: cv2.GaussianBlur(img, (5, 5), 0)),
            ("edge_detection", lambda: cv2.Canny(img, 100, 200))
        ]

        for op_name, operation in operations:
            start_time = time.time()
            operation()
            op_time = time.time() - start_time

            results.append(BenchmarkResult(
                component="image_processing",
                metric=f"{op_name}_time",
                value=op_time,
                unit="seconds",
                timestamp=time.time(),
                context={"image_shape": img.shape}
            ))

        return results

    def benchmark_memory_usage(self) -> BenchmarkResult:
        """Monitor memory usage during operations."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return BenchmarkResult(
            component="system",
            metric="memory_usage",
            value=memory_info.rss / 1024 / 1024,  # Convert to MB
            unit="MB",
            timestamp=time.time(),
            context={
                "virtual_memory": memory_info.vms / 1024 / 1024,
                "percent_used": process.memory_percent()
            }
        )

    def benchmark_multi_threading(self, n_threads: int = 4) -> BenchmarkResult:
        """Test multi-threading performance."""
        import threading
        import queue

        def worker(q, results):
            while True:
                try:
                    matrix_size = q.get_nowait()
                    start_time = time.time()

                    # Simulate work
                    matrix = np.random.rand(matrix_size, matrix_size)
                    np.dot(matrix, matrix.T)

                    results.append(time.time() - start_time)
                    q.task_done()
                except queue.Empty:
                    break

        work_queue = queue.Queue()
        for _ in range(n_threads * 2):
            work_queue.put(500)  # Matrix size

        thread_results = []
        threads = []

        start_time = time.time()

        for _ in range(n_threads):
            t = threading.Thread(
                target=worker,
                args=(work_queue, thread_results)
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        total_time = time.time() - start_time

        return BenchmarkResult(
            component="threading",
            metric="parallel_processing_time",
            value=total_time,
            unit="seconds",
            timestamp=time.time(),
            context={
                "n_threads": n_threads,
                "tasks_per_thread": 2,
                "avg_task_time": sum(thread_results) / len(thread_results)
            }
        )

    def run_benchmarks(self) -> Dict:
        """Run all benchmarks and collect results."""
        self.logger.info("Starting performance benchmarks")

        results = []

        # Image processing benchmarks
        test_image = self.project_root / "Hoag's_object.jpg"
        if test_image.exists():
            results.extend(self.benchmark_image_processing(test_image))

        # Memory usage benchmark
        results.append(self.benchmark_memory_usage())

        # Multi-threading benchmark
        results.append(self.benchmark_multi_threading())

        # Save results
        output = {
            "timestamp": time.time(),
            "results": [
                {
                    "component": r.component,
                    "metric": r.metric,
                    "value": r.value,
                    "unit": r.unit,
                    "context": r.context
                }
                for r in results
            ]
        }

        # Save to dated file
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"benchmark_{date_str}.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        return output

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    monitor = PerformanceMonitor(project_root)
    results = monitor.run_benchmarks()
    print(json.dumps(results, indent=2))
