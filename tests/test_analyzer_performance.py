"""
Performance Tests for Negative Space Imaging Analysis

Comprehensive performance testing including benchmarks, memory profiling,
concurrent load testing, and scalability analysis.

Test Categories:
- Image processing speed benchmarks
- Memory usage profiling
- Concurrent request handling
- Throughput measurement
- Scalability with image size
- Resource utilization optimization
"""

import pytest
import numpy as np
import torch
import time
import logging
import psutil
import gc
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import statistics

logger = logging.getLogger(__name__)


# =====================================================================
# SPEED & THROUGHPUT BENCHMARKS
# =====================================================================

class TestProcessingSpeed:
    """Tests for processing speed and throughput benchmarks."""

    @pytest.mark.performance
    def test_single_image_processing_speed(self, synthetic_image,
                                          mock_analyzer,
                                          benchmark_timer):
        """Benchmark single image processing speed."""
        # Warm up
        _ = synthetic_image.astype(np.float32) / 255.0

        # Benchmark
        for _ in range(5):
            benchmark_timer.start()

            preprocessed = synthetic_image.astype(np.float32) / 255.0
            _ = mock_analyzer._detect_negative_spaces(preprocessed)

            elapsed = benchmark_timer.stop()
            logger.info(f"Processing time: {elapsed*1000:.2f} ms")

        summary = benchmark_timer.summary()

        # Performance targets
        assert summary["mean_ms"] < 500, "Processing too slow"
        logger.info(f"Average processing time: {summary['mean_ms']:.2f} ms")

    @pytest.mark.performance
    def test_batch_processing_throughput(self, image_batch,
                                        mock_analyzer,
                                        benchmark_timer):
        """Benchmark batch processing throughput."""
        images = image_batch * 4  # 20 images

        benchmark_timer.start()

        for image in images:
            preprocessed = image.astype(np.float32) / 255.0
            _ = mock_analyzer._detect_negative_spaces(preprocessed)

        total_time = benchmark_timer.stop()

        throughput = len(images) / total_time
        logger.info(f"Throughput: {throughput:.2f} images/second")

        # Should process at least 2 images per second
        assert throughput > 2.0, f"Throughput too low: {throughput}"

    @pytest.mark.performance
    def test_large_image_processing_speed(self, mock_analyzer,
                                         benchmark_timer):
        """Benchmark processing of large images."""
        # Create large image (2048x2048)
        large_image = np.random.randint(0, 256, (2048, 2048),
                                       dtype=np.uint8)

        benchmark_timer.start()

        preprocessed = large_image.astype(np.float32) / 255.0
        _ = mock_analyzer._detect_negative_spaces(preprocessed)

        elapsed = benchmark_timer.stop()

        logger.info(f"Large image processing: {elapsed*1000:.2f} ms")

        # Large images should still process reasonably fast
        assert elapsed < 5.0, "Large image processing too slow"

    @pytest.mark.performance
    def test_various_image_sizes_speed(self, mock_analyzer,
                                      benchmark_timer):
        """Benchmark processing speed for various image sizes."""
        sizes = [(256, 256), (512, 512), (1024, 1024)]

        results = {}

        for width, height in sizes:
            image = np.random.randint(0, 256, (height, width),
                                     dtype=np.uint8)

            benchmark_timer.times = []  # Reset

            for _ in range(3):
                benchmark_timer.start()

                preprocessed = image.astype(np.float32) / 255.0
                _ = mock_analyzer._detect_negative_spaces(preprocessed)

                benchmark_timer.stop()

            avg_time = benchmark_timer.mean()
            results[f"{width}x{height}"] = avg_time
            logger.info(f"{width}x{height}: {avg_time*1000:.2f} ms avg")

        # Speed should scale reasonably with image size
        assert all(v < 5.0 for v in results.values())


# =====================================================================
# MEMORY USAGE PROFILING
# =====================================================================

class TestMemoryUsage:
    """Tests for memory usage and memory leak detection."""

    @pytest.mark.performance
    def test_single_image_memory_usage(self, synthetic_image,
                                      memory_profiler):
        """Profile memory usage for single image processing."""
        gc.collect()
        memory_profiler.take_snapshot("before")

        # Process image
        preprocessed = synthetic_image.astype(np.float32) / 255.0

        memory_profiler.take_snapshot("after_preprocess")

        # Check memory usage
        peak_memory = memory_profiler.get_peak_memory()
        logger.info(f"Peak memory: {peak_memory:.2f} MB")

        # Should be reasonable (< 200MB for single image)
        assert peak_memory < 200, "Memory usage too high"

    @pytest.mark.performance
    def test_batch_processing_memory_usage(self, image_batch,
                                          memory_profiler):
        """Profile memory usage during batch processing."""
        gc.collect()
        memory_profiler.take_snapshot("batch_start")

        # Process batch
        for image in image_batch:
            _ = image.astype(np.float32) / 255.0

        memory_profiler.take_snapshot("batch_end")

        peak_memory = memory_profiler.get_peak_memory()
        logger.info(f"Batch peak memory: {peak_memory:.2f} MB")

        # Batch of 5 256x256 images should be < 300MB
        assert peak_memory < 300

    @pytest.mark.performance
    def test_memory_leak_detection(self, synthetic_image,
                                  memory_profiler):
        """Detect potential memory leaks in repeated processing."""
        gc.collect()
        memory_profiler.take_snapshot("iteration_0")

        initial_memory = memory_profiler.snapshots[-1]["rss_mb"]

        # Repeat processing 20 times
        for i in range(20):
            image_copy = synthetic_image.copy()
            _ = image_copy.astype(np.float32) / 255.0

            if (i + 1) % 5 == 0:
                gc.collect()
                memory_profiler.take_snapshot(f"iteration_{i+1}")

        final_memory = memory_profiler.snapshots[-1]["rss_mb"]
        memory_growth = final_memory - initial_memory

        logger.info(f"Memory growth over 20 iterations: "
                   f"{memory_growth:.2f} MB")

        # Should not grow significantly (< 50MB growth)
        assert memory_growth < 50, "Potential memory leak detected"

    @pytest.mark.performance
    def test_memory_cleanup_after_processing(self, synthetic_image,
                                            memory_profiler):
        """Test that memory is properly released after processing."""
        gc.collect()
        memory_profiler.take_snapshot("cleanup_start")
        start_memory = memory_profiler.snapshots[-1]["rss_mb"]

        # Process image
        image = synthetic_image.copy()
        _ = image.astype(np.float32) / 255.0

        memory_profiler.take_snapshot("after_processing")

        # Cleanup
        del image
        gc.collect()

        memory_profiler.take_snapshot("after_cleanup")
        final_memory = memory_profiler.snapshots[-1]["rss_mb"]

        # Memory should be mostly recovered
        recovery_ratio = (start_memory - final_memory) / start_memory \
                        if start_memory > 0 else 0
        logger.info(f"Memory recovery ratio: {recovery_ratio:.2%}")

        assert final_memory < start_memory + 50


# =====================================================================
# CONCURRENT PROCESSING TESTS
# =====================================================================

class TestConcurrentProcessing:
    """Tests for concurrent request handling and thread safety."""

    @pytest.mark.performance
    @pytest.mark.concurrent
    def test_concurrent_image_processing(self, image_batch,
                                        mock_analyzer,
                                        thread_pool_executor):
        """Test concurrent processing of multiple images."""
        def process_image(image):
            preprocessed = image.astype(np.float32) / 255.0
            return mock_analyzer._detect_negative_spaces(preprocessed)

        # Submit tasks concurrently
        futures = [
            thread_pool_executor.submit(process_image, img)
            for img in image_batch * 2
        ]

        # Collect results
        results = [f.result() for f in futures]

        assert len(results) == len(image_batch) * 2
        assert all(isinstance(r, dict) for r in results)

    @pytest.mark.performance
    @pytest.mark.concurrent
    def test_concurrent_access_scaling(self, synthetic_image,
                                      concurrent_test_runner):
        """Test performance scaling with concurrent access."""
        def analyze(image_id):
            # Simulate processing
            time.sleep(0.01)
            return {"image_id": image_id, "status": "completed"}

        # Run with different worker counts
        for num_workers in [1, 2, 4, 8]:
            args_list = [(i,) for i in range(20)]

            start = time.time()
            results, errors = concurrent_test_runner.run(
                analyze, args_list, max_workers=num_workers
            )
            elapsed = time.time() - start

            throughput = len(results) / elapsed
            logger.info(f"Workers: {num_workers}, "
                       f"Throughput: {throughput:.2f} tasks/sec")

            assert len(errors) == 0, "Processing errors occurred"

    @pytest.mark.performance
    @pytest.mark.concurrent
    def test_thread_pool_efficiency(self, benchmark_timer):
        """Test thread pool efficiency and overhead."""
        def dummy_task():
            return sum(range(1000))

        num_tasks = 100

        # Single-threaded
        benchmark_timer.times = []
        benchmark_timer.start()

        for _ in range(num_tasks):
            dummy_task()

        single_threaded = benchmark_timer.stop()

        # Multi-threaded
        benchmark_timer.times = []
        benchmark_timer.start()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(dummy_task)
                      for _ in range(num_tasks)]
            [f.result() for f in futures]

        multi_threaded = benchmark_timer.stop()

        logger.info(f"Single-threaded: {single_threaded*1000:.2f} ms")
        logger.info(f"Multi-threaded: {multi_threaded*1000:.2f} ms")


# =====================================================================
# RESOURCE UTILIZATION TESTS
# =====================================================================

class TestResourceUtilization:
    """Tests for CPU and resource utilization."""

    @pytest.mark.performance
    def test_cpu_utilization_during_processing(self, synthetic_image):
        """Monitor CPU utilization during processing."""
        process = psutil.Process()

        cpu_before = process.cpu_percent(interval=0.1)

        # Simulate processing
        for _ in range(100):
            _ = synthetic_image.astype(np.float32) / 255.0
            _ = np.fft.fft2(synthetic_image)

        cpu_after = process.cpu_percent(interval=0.1)

        logger.info(f"CPU before: {cpu_before}%")
        logger.info(f"CPU after: {cpu_after}%")

    @pytest.mark.performance
    def test_disk_io_performance(self, synthetic_image, test_data_dir):
        """Test disk I/O performance."""
        filename = test_data_dir / "perf_test_image.npy"

        # Write benchmark
        write_timer = time.time()
        np.save(str(filename), synthetic_image)
        write_time = time.time() - write_timer

        file_size_mb = filename.stat().st_size / (1024 * 1024)
        write_speed = file_size_mb / write_time

        logger.info(f"Write speed: {write_speed:.2f} MB/s")

        # Read benchmark
        read_timer = time.time()
        _ = np.load(str(filename))
        read_time = time.time() - read_timer

        read_speed = file_size_mb / read_time
        logger.info(f"Read speed: {read_speed:.2f} MB/s")


# =====================================================================
# SCALABILITY TESTS
# =====================================================================

class TestScalability:
    """Tests for scalability and performance with increasing load."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_scalability_with_image_size(self, mock_analyzer,
                                        benchmark_timer):
        """Test scalability as image size increases."""
        sizes = [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
        ]

        results = {}

        for width, height in sizes:
            image = np.random.randint(0, 256, (height, width),
                                     dtype=np.uint8)

            benchmark_timer.times = []

            benchmark_timer.start()

            preprocessed = image.astype(np.float32) / 255.0
            _ = mock_analyzer._detect_negative_spaces(preprocessed)

            elapsed = benchmark_timer.stop()

            pixel_count = width * height
            ms_per_mpixel = (elapsed * 1000) / (pixel_count / 1_000_000)

            results[f"{width}x{height}"] = {
                "time_ms": elapsed * 1000,
                "ms_per_mpixel": ms_per_mpixel,
            }

            logger.info(f"{width}x{height}: "
                       f"{elapsed*1000:.2f} ms "
                       f"({ms_per_mpixel:.2f} ms/Mpixel)")

        # Performance should not degrade too much with size
        # (allowing for some increase due to overhead)
        ratio_1024_256 = (results["1024x1024"]["ms_per_mpixel"] /
                         results["256x256"]["ms_per_mpixel"])
        assert ratio_1024_256 < 2.0, "Performance degrades too much"

    @pytest.mark.performance
    def test_scalability_with_region_count(self, mock_analyzer):
        """Test performance with increasing number of regions."""
        # Create images with varying region density
        region_counts = [5, 10, 20, 50]

        results = {}

        for count in region_counts:
            image = np.zeros((256, 256), dtype=np.uint8)

            # Add regions
            for _ in range(count):
                x = np.random.randint(20, 236)
                y = np.random.randint(20, 236)
                cv2.circle(image, (x, y), 15, 255, -1)

            start = time.time()

            preprocessed = image.astype(np.float32) / 255.0
            detected = mock_analyzer._detect_negative_spaces(preprocessed)

            elapsed = time.time() - start

            results[count] = {
                "time_ms": elapsed * 1000,
                "detected_regions": len(detected),
            }

            logger.info(f"Regions: {count}, Time: {elapsed*1000:.2f} ms")

        # Processing time should scale reasonably
        time_increase = (results[50]["time_ms"] /
                        results[5]["time_ms"])
        assert time_increase < 10.0, "Time scales poorly with region count"


# =====================================================================
# OPTIMIZATION BENCHMARK TESTS
# =====================================================================

class TestOptimizationBenchmarks:
    """Tests for measuring optimization effectiveness."""

    @pytest.mark.performance
    def test_preprocessing_vs_no_preprocessing(self, synthetic_image,
                                              benchmark_timer):
        """Compare performance with and without preprocessing."""
        # Without preprocessing
        benchmark_timer.times = []
        for _ in range(10):
            benchmark_timer.start()
            _ = synthetic_image  # No processing
            benchmark_timer.stop()

        no_preprocess_time = benchmark_timer.mean()

        # With preprocessing
        benchmark_timer.times = []
        for _ in range(10):
            benchmark_timer.start()
            _ = synthetic_image.astype(np.float32) / 255.0
            benchmark_timer.stop()

        with_preprocess_time = benchmark_timer.mean()

        overhead = (with_preprocess_time / no_preprocess_time) \
                  if no_preprocess_time > 0 else 1

        logger.info(f"Preprocessing overhead: {overhead:.1f}x")

    @pytest.mark.performance
    @pytest.mark.gpu
    def test_cpu_vs_gpu_comparison(self):
        """Compare CPU vs GPU performance (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        data = torch.randn(256, 256)
        iterations = 100

        # CPU
        start = time.time()
        for _ in range(iterations):
            _ = data.to("cpu")
            _ = torch.fft.fft2(data)
        cpu_time = time.time() - start

        # GPU
        start = time.time()
        for _ in range(iterations):
            _ = data.to("cuda")
            _ = torch.fft.fft2(data.to("cuda"))
        gpu_time = time.time() - start

        speedup = cpu_time / gpu_time
        logger.info(f"GPU speedup vs CPU: {speedup:.2f}x")


# Helper for concurrent tests
try:
    import cv2
except ImportError:
    logger.warning("OpenCV not available for some tests")
