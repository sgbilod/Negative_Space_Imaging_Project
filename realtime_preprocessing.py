#!/usr/bin/env python
"""
Real-Time Preprocessing Pipeline
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module implements a real-time preprocessing pipeline for streaming images:
- Streaming image intake
- On-the-fly calibration
- Real-time quality assessment
- Pipeline stage management
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Processing pipeline stages."""
    INTAKE = "intake"
    CALIBRATION = "calibration"
    NOISE_REDUCTION = "noise_reduction"
    QUALITY_ASSESSMENT = "quality_assessment"
    OUTPUT = "output"


class QualityLevel(Enum):
    """Image quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    REJECTED = "rejected"


@dataclass
class ImageFrame:
    """Container for a single image frame in the pipeline."""
    frame_id: str
    data: np.ndarray
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage: PipelineStage = PipelineStage.INTAKE
    quality: Optional[QualityLevel] = None
    processing_history: List[Dict[str, Any]] = field(default_factory=list)

    def add_processing_step(self, stage: str, duration: float, details: Dict[str, Any] = None):
        """Record a processing step."""
        self.processing_history.append({
            "stage": stage,
            "timestamp": datetime.utcnow().isoformat(),
            "duration_ms": duration * 1000,
            "details": details or {},
        })


@dataclass
class CalibrationFrames:
    """Calibration frames for image correction."""
    dark_frame: Optional[np.ndarray] = None
    flat_frame: Optional[np.ndarray] = None
    bias_frame: Optional[np.ndarray] = None
    hot_pixel_map: Optional[np.ndarray] = None

    def is_complete(self) -> bool:
        """Check if all calibration frames are available."""
        return all([
            self.dark_frame is not None,
            self.flat_frame is not None,
        ])


class ProcessingStage(ABC):
    """Abstract base class for pipeline processing stages."""

    stage_name: str = "base"

    @abstractmethod
    def process(self, frame: ImageFrame) -> ImageFrame:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        pass

    def validate_input(self, frame: ImageFrame) -> bool:
        """Validate input frame."""
        return frame.data is not None and frame.data.size > 0


class IntakeStage(ProcessingStage):
    """Initial intake and validation stage."""

    stage_name = "intake"

    def __init__(self, min_size: int = 64, max_size: int = 16384):
        self.min_size = min_size
        self.max_size = max_size

    def process(self, frame: ImageFrame) -> ImageFrame:
        """Validate and prepare incoming frame."""
        start_time = time.time()

        # Validate dimensions
        height, width = frame.data.shape[:2]
        if width < self.min_size or height < self.min_size:
            raise ValueError(f"Image too small: {width}x{height}")
        if width > self.max_size or height > self.max_size:
            raise ValueError(f"Image too large: {width}x{height}")

        # Ensure correct data type
        if frame.data.dtype != np.float32:
            frame.data = frame.data.astype(np.float32)

        frame.stage = PipelineStage.INTAKE
        frame.add_processing_step(self.stage_name, time.time() - start_time, {
            "width": width,
            "height": height,
            "dtype": str(frame.data.dtype),
        })

        return frame


class CalibrationStage(ProcessingStage):
    """Calibration stage for dark subtraction and flat correction."""

    stage_name = "calibration"

    def __init__(self, calibration: Optional[CalibrationFrames] = None):
        self.calibration = calibration or CalibrationFrames()

    def set_dark_frame(self, dark: np.ndarray) -> None:
        """Set dark frame for subtraction."""
        self.calibration.dark_frame = dark.astype(np.float32)

    def set_flat_frame(self, flat: np.ndarray) -> None:
        """Set flat frame for correction."""
        # Normalize flat frame
        flat = flat.astype(np.float32)
        self.calibration.flat_frame = flat / np.mean(flat)

    def set_bias_frame(self, bias: np.ndarray) -> None:
        """Set bias frame for subtraction."""
        self.calibration.bias_frame = bias.astype(np.float32)

    def set_hot_pixel_map(self, hot_pixels: np.ndarray) -> None:
        """Set hot pixel map."""
        self.calibration.hot_pixel_map = hot_pixels.astype(bool)

    def process(self, frame: ImageFrame) -> ImageFrame:
        """Apply calibration corrections."""
        start_time = time.time()
        details: Dict[str, Any] = {}

        data = frame.data.copy()

        # Bias subtraction
        if self.calibration.bias_frame is not None:
            data = data - self.calibration.bias_frame
            details["bias_subtracted"] = True

        # Dark subtraction
        if self.calibration.dark_frame is not None:
            data = data - self.calibration.dark_frame
            details["dark_subtracted"] = True

        # Flat field correction
        if self.calibration.flat_frame is not None:
            # Avoid division by zero
            flat = np.where(
                self.calibration.flat_frame > 0.1,
                self.calibration.flat_frame,
                1.0
            )
            data = data / flat
            details["flat_corrected"] = True

        # Hot pixel removal
        if self.calibration.hot_pixel_map is not None:
            # Replace hot pixels with median of neighbors
            from scipy import ndimage
            data[self.calibration.hot_pixel_map] = ndimage.median_filter(data, 3)[
                self.calibration.hot_pixel_map
            ]
            details["hot_pixels_removed"] = True

        # Clip to valid range
        data = np.clip(data, 0, np.max(data))

        frame.data = data
        frame.stage = PipelineStage.CALIBRATION
        frame.add_processing_step(self.stage_name, time.time() - start_time, details)

        return frame


class NoiseReductionStage(ProcessingStage):
    """Noise reduction stage."""

    stage_name = "noise_reduction"

    def __init__(
        self,
        method: str = "gaussian",
        strength: float = 1.0,
    ):
        self.method = method
        self.strength = strength

    def process(self, frame: ImageFrame) -> ImageFrame:
        """Apply noise reduction."""
        start_time = time.time()

        if self.method == "gaussian":
            from scipy.ndimage import gaussian_filter
            frame.data = gaussian_filter(frame.data, sigma=self.strength)
        elif self.method == "median":
            from scipy.ndimage import median_filter
            size = int(self.strength * 2) + 1
            frame.data = median_filter(frame.data, size=size)
        elif self.method == "bilateral":
            # Simple bilateral-like filtering
            from scipy.ndimage import gaussian_filter
            # Preserve edges by combining with edge detection
            edges = np.abs(np.gradient(frame.data)[0]) + np.abs(np.gradient(frame.data)[1])
            weight = np.exp(-edges / (np.std(edges) + 1e-8))
            smoothed = gaussian_filter(frame.data, sigma=self.strength)
            frame.data = frame.data * (1 - weight) + smoothed * weight

        frame.stage = PipelineStage.NOISE_REDUCTION
        frame.add_processing_step(self.stage_name, time.time() - start_time, {
            "method": self.method,
            "strength": self.strength,
        })

        return frame


class QualityAssessmentStage(ProcessingStage):
    """Quality assessment stage."""

    stage_name = "quality_assessment"

    def __init__(
        self,
        min_snr: float = 10.0,
        min_sharpness: float = 0.3,
        max_noise: float = 0.1,
    ):
        self.min_snr = min_snr
        self.min_sharpness = min_sharpness
        self.max_noise = max_noise

    def process(self, frame: ImageFrame) -> ImageFrame:
        """Assess image quality."""
        start_time = time.time()

        metrics = self._compute_metrics(frame.data)

        # Determine quality level
        score = self._compute_score(metrics)

        if score >= 0.9:
            quality = QualityLevel.EXCELLENT
        elif score >= 0.7:
            quality = QualityLevel.GOOD
        elif score >= 0.5:
            quality = QualityLevel.ACCEPTABLE
        elif score >= 0.3:
            quality = QualityLevel.POOR
        else:
            quality = QualityLevel.REJECTED

        frame.quality = quality
        frame.metadata["quality_metrics"] = metrics
        frame.metadata["quality_score"] = score
        frame.stage = PipelineStage.QUALITY_ASSESSMENT
        frame.add_processing_step(self.stage_name, time.time() - start_time, {
            "quality": quality.value,
            "score": score,
            "metrics": metrics,
        })

        return frame

    def _compute_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics."""
        # Signal-to-noise ratio (estimated)
        signal = np.mean(data)
        noise = np.std(data)
        snr = signal / (noise + 1e-8)

        # Sharpness (Laplacian variance)
        laplacian = np.abs(
            np.gradient(np.gradient(data, axis=0), axis=0) +
            np.gradient(np.gradient(data, axis=1), axis=1)
        )
        sharpness = np.var(laplacian) / (np.mean(data)**2 + 1e-8)

        # Noise level (normalized std)
        noise_level = noise / (signal + 1e-8)

        # Dynamic range
        dynamic_range = (np.max(data) - np.min(data)) / (np.max(data) + 1e-8)

        return {
            "snr": float(snr),
            "sharpness": float(sharpness),
            "noise_level": float(noise_level),
            "dynamic_range": float(dynamic_range),
        }

    def _compute_score(self, metrics: Dict[str, float]) -> float:
        """Compute overall quality score (0-1)."""
        snr_score = min(1.0, metrics["snr"] / (self.min_snr * 2))
        sharpness_score = min(1.0, metrics["sharpness"] / (self.min_sharpness * 2))
        noise_score = max(0.0, 1.0 - metrics["noise_level"] / self.max_noise)
        range_score = metrics["dynamic_range"]

        # Weighted average
        score = (
            0.3 * snr_score +
            0.3 * sharpness_score +
            0.2 * noise_score +
            0.2 * range_score
        )

        return float(score)


class RealtimePreprocessingPipeline:
    """
    Real-time preprocessing pipeline for streaming images.
    
    Provides:
    - Asynchronous processing
    - Multiple processing stages
    - Quality filtering
    - Performance monitoring
    """

    def __init__(
        self,
        max_queue_size: int = 100,
        reject_poor_quality: bool = True,
    ):
        self.max_queue_size = max_queue_size
        self.reject_poor_quality = reject_poor_quality

        self._stages: List[ProcessingStage] = []
        self._input_queue: Queue = Queue(maxsize=max_queue_size)
        self._output_queue: Queue = Queue(maxsize=max_queue_size)
        self._running = False
        self._worker_thread: Optional[Thread] = None
        self._callbacks: List[Callable[[ImageFrame], None]] = []

        # Statistics
        self._stats = {
            "frames_processed": 0,
            "frames_rejected": 0,
            "total_processing_time": 0.0,
            "errors": 0,
        }

        # Add default stages
        self._add_default_stages()

    def _add_default_stages(self) -> None:
        """Add default processing stages."""
        self.add_stage(IntakeStage())
        self.add_stage(CalibrationStage())
        self.add_stage(NoiseReductionStage(method="gaussian", strength=0.5))
        self.add_stage(QualityAssessmentStage())

    def add_stage(self, stage: ProcessingStage) -> None:
        """Add a processing stage."""
        self._stages.append(stage)
        logger.info(f"Added stage: {stage.stage_name}")

    def set_calibration(self, calibration: CalibrationFrames) -> None:
        """Set calibration frames."""
        for stage in self._stages:
            if isinstance(stage, CalibrationStage):
                stage.calibration = calibration
                break

    def submit_frame(self, frame_id: str, data: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """
        Submit a frame for processing.
        
        Args:
            frame_id: Unique frame identifier
            data: Image data
            metadata: Optional metadata
            
        Returns:
            True if frame was queued
        """
        if self._input_queue.full():
            logger.warning("Input queue full, frame dropped")
            return False

        frame = ImageFrame(
            frame_id=frame_id,
            data=data,
            metadata=metadata or {},
        )

        self._input_queue.put(frame)
        return True

    def get_result(self, timeout: float = 1.0) -> Optional[ImageFrame]:
        """
        Get a processed frame from the output queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Processed frame or None
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except Exception:
            return None

    def register_callback(self, callback: Callable[[ImageFrame], None]) -> None:
        """Register a callback for processed frames."""
        self._callbacks.append(callback)

    def _process_frame(self, frame: ImageFrame) -> Optional[ImageFrame]:
        """Process a frame through all stages."""
        start_time = time.time()

        try:
            for stage in self._stages:
                if not stage.validate_input(frame):
                    raise ValueError(f"Invalid input for stage: {stage.stage_name}")
                frame = stage.process(frame)

            # Check quality
            if self.reject_poor_quality and frame.quality == QualityLevel.REJECTED:
                self._stats["frames_rejected"] += 1
                return None

            frame.stage = PipelineStage.OUTPUT
            self._stats["frames_processed"] += 1
            self._stats["total_processing_time"] += time.time() - start_time

            return frame

        except Exception as e:
            logger.error(f"Processing error: {e}")
            self._stats["errors"] += 1
            return None

    def _worker(self) -> None:
        """Worker thread for processing frames."""
        while self._running:
            try:
                frame = self._input_queue.get(timeout=0.1)
            except Exception:
                continue

            result = self._process_frame(frame)

            if result is not None:
                self._output_queue.put(result)

                # Call callbacks
                for callback in self._callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            self._input_queue.task_done()

    def start(self) -> None:
        """Start the pipeline."""
        if self._running:
            return

        self._running = True
        self._worker_thread = Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        logger.info("Pipeline started")

    def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        logger.info("Pipeline stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        avg_time = (
            self._stats["total_processing_time"] / self._stats["frames_processed"]
            if self._stats["frames_processed"] > 0 else 0.0
        )

        return {
            **self._stats,
            "average_processing_time_ms": avg_time * 1000,
            "input_queue_size": self._input_queue.qsize(),
            "output_queue_size": self._output_queue.qsize(),
            "stages": [s.stage_name for s in self._stages],
        }


def create_default_pipeline() -> RealtimePreprocessingPipeline:
    """Create a default preprocessing pipeline."""
    return RealtimePreprocessingPipeline()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Real-Time Preprocessing Pipeline Demo")
    print("=" * 40)

    # Create pipeline
    pipeline = create_default_pipeline()
    pipeline.start()

    try:
        # Process some test frames
        for i in range(5):
            # Generate test image
            test_image = np.random.randint(50, 200, (256, 256)).astype(np.float32)
            test_image += np.random.normal(0, 10, test_image.shape)

            # Submit frame
            pipeline.submit_frame(f"frame_{i}", test_image)
            print(f"Submitted frame {i}")

        # Collect results
        time.sleep(1)  # Wait for processing

        while True:
            result = pipeline.get_result(timeout=0.5)
            if result is None:
                break

            print(f"\nProcessed: {result.frame_id}")
            print(f"  Quality: {result.quality.value if result.quality else 'N/A'}")
            print(f"  Stages: {len(result.processing_history)}")

        # Show statistics
        stats = pipeline.get_statistics()
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    finally:
        pipeline.stop()
