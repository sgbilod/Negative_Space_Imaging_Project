"""
ML Pipeline Monitoring - Model Performance Monitoring

Monitors model performance, detects drift, and provides alerting for the pipeline.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy import stats

from ..core.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""

    model_name: str
    timestamp: float
    inference_time: float
    confidence_score: Optional[float] = None
    prediction_entropy: Optional[float] = None
    input_features: Optional[np.ndarray] = None
    prediction: Optional[Any] = None
    ground_truth: Optional[Any] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""

    model_name: str
    timestamp: float
    drift_detected: bool
    drift_score: float
    threshold: float
    method: str
    details: Dict[str, Any] = field(default_factory=dict)


class ModelMonitor:
    """
    Monitors model performance and detects concept drift.

    Features:
    - Real-time performance tracking
    - Statistical drift detection
    - Performance alerting
    - Model health monitoring
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Metrics storage
        self.metrics_buffer: Dict[str, deque] = {}
        self.max_buffer_size = 10000

        # Drift detection
        self.reference_distributions: Dict[str, Dict[str, Any]] = {}
        self.drift_detectors: Dict[str, DriftDetector] = {}

        # Alerting
        self.alerts: List[Dict[str, Any]] = []
        self.alert_callbacks: List[callable] = []

        # Monitoring settings
        self.monitoring_interval = config.metrics_interval_seconds
        self.is_monitoring = False

        logger.info("Initialized model monitor")

    async def start_monitoring(self) -> None:
        """Start the monitoring system."""
        self.is_monitoring = True

        # Start background monitoring task
        asyncio.create_task(self._monitoring_loop())

        logger.info("Model monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self.is_monitoring = False
        logger.info("Model monitoring stopped")

    def record_inference(
        self,
        model_name: str,
        inference_time: float,
        confidence_score: Optional[float] = None,
        prediction_entropy: Optional[float] = None,
        input_features: Optional[np.ndarray] = None,
        prediction: Optional[Any] = None,
        ground_truth: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record inference metrics for monitoring.

        Args:
            model_name: Name of the model
            inference_time: Time taken for inference
            confidence_score: Model confidence score
            prediction_entropy: Prediction entropy
            input_features: Input features for drift detection
            prediction: Model prediction
            ground_truth: Ground truth label (if available)
            metadata: Additional metadata
        """
        if not self.is_monitoring:
            return

        metrics = ModelMetrics(
            model_name=model_name,
            timestamp=time.time(),
            inference_time=inference_time,
            confidence_score=confidence_score,
            prediction_entropy=prediction_entropy,
            input_features=input_features,
            prediction=prediction,
            ground_truth=ground_truth,
            metadata=metadata or {},
        )

        # Store metrics
        if model_name not in self.metrics_buffer:
            self.metrics_buffer[model_name] = deque(maxlen=self.max_buffer_size)

        self.metrics_buffer[model_name].append(metrics)

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                await self._perform_monitoring_checks()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

    async def _perform_monitoring_checks(self) -> None:
        """Perform periodic monitoring checks."""
        for model_name in self.metrics_buffer.keys():
            await self._check_model_health(model_name)
            await self._detect_drift(model_name)

    async def _check_model_health(self, model_name: str) -> None:
        """Check model health metrics."""
        buffer = self.metrics_buffer.get(model_name, [])
        if len(buffer) < 10:  # Need minimum samples
            return

        recent_metrics = list(buffer)[-100:]  # Last 100 inferences

        # Calculate health metrics
        avg_inference_time = np.mean([m.inference_time for m in recent_metrics])
        avg_confidence = np.mean([m.confidence_score for m in recent_metrics if m.confidence_score is not None])

        # Check for performance degradation
        if avg_inference_time > 1.0:  # Threshold for slow inference
            await self._trigger_alert(
                model_name,
                "high_inference_time",
                f"Average inference time: {avg_inference_time:.3f}s",
                {"avg_inference_time": avg_inference_time}
            )

        if avg_confidence is not None and avg_confidence < 0.5:  # Low confidence threshold
            await self._trigger_alert(
                model_name,
                "low_confidence",
                f"Average confidence: {avg_confidence:.3f}",
                {"avg_confidence": avg_confidence}
            )

    async def _detect_drift(self, model_name: str) -> None:
        """Detect concept drift for a model."""
        buffer = self.metrics_buffer.get(model_name, [])
        if len(buffer) < 100:  # Need sufficient data for drift detection
            return

        # Get recent data
        recent_data = list(buffer)[-500:]  # Last 500 inferences

        # Extract features for drift detection
        features = []
        for metric in recent_data:
            if metric.input_features is not None:
                features.append(metric.input_features.flatten())

        if not features:
            return

        features = np.array(features)

        # Initialize or update drift detector
        if model_name not in self.drift_detectors:
            self.drift_detectors[model_name] = DriftDetector()

        detector = self.drift_detectors[model_name]

        # Detect drift
        drift_result = detector.detect_drift(features)

        if drift_result.drift_detected:
            await self._trigger_alert(
                model_name,
                "concept_drift",
                f"Concept drift detected (score: {drift_result.drift_score:.3f})",
                {
                    "drift_score": drift_result.drift_score,
                    "threshold": drift_result.threshold,
                    "method": drift_result.method,
                }
            )

    async def _trigger_alert(
        self,
        model_name: str,
        alert_type: str,
        message: str,
        details: Dict[str, Any]
    ) -> None:
        """Trigger an alert."""
        alert = {
            "model_name": model_name,
            "alert_type": alert_type,
            "message": message,
            "details": details,
            "timestamp": time.time(),
        }

        self.alerts.append(alert)

        # Keep only recent alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]

        logger.warning(f"Alert triggered: {model_name} - {alert_type}: {message}")

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def add_alert_callback(self, callback: callable) -> None:
        """Add a callback for alerts."""
        self.alert_callbacks.append(callback)

    def get_model_metrics(
        self,
        model_name: str,
        limit: Optional[int] = None
    ) -> List[ModelMetrics]:
        """Get metrics for a model."""
        buffer = self.metrics_buffer.get(model_name, [])
        metrics = list(buffer)

        if limit:
            metrics = metrics[-limit:]

        return metrics

    def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """Get statistical summary for a model."""
        buffer = self.metrics_buffer.get(model_name, [])
        if not buffer:
            return {}

        metrics = list(buffer)

        inference_times = [m.inference_time for m in metrics]
        confidence_scores = [m.confidence_score for m in metrics if m.confidence_score is not None]

        stats = {
            "total_inferences": len(metrics),
            "avg_inference_time": np.mean(inference_times),
            "min_inference_time": np.min(inference_times),
            "max_inference_time": np.max(inference_times),
            "inference_time_std": np.std(inference_times),
        }

        if confidence_scores:
            stats.update({
                "avg_confidence": np.mean(confidence_scores),
                "min_confidence": np.min(confidence_scores),
                "max_confidence": np.max(confidence_scores),
                "confidence_std": np.std(confidence_scores),
            })

        return stats

    def get_alerts(
        self,
        model_name: Optional[str] = None,
        alert_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        alerts = self.alerts

        if model_name:
            alerts = [a for a in alerts if a["model_name"] == model_name]

        if alert_type:
            alerts = [a for a in alerts if a["alert_type"] == alert_type]

        if limit:
            alerts = alerts[-limit:]

        return alerts

    def set_reference_distribution(self, model_name: str, features: np.ndarray) -> None:
        """
        Set reference distribution for drift detection.

        Args:
            model_name: Name of the model
            features: Reference feature distribution
        """
        self.reference_distributions[model_name] = {
            "mean": np.mean(features, axis=0),
            "std": np.std(features, axis=0),
            "timestamp": time.time(),
        }

        logger.info(f"Set reference distribution for {model_name}")

    async def cleanup(self) -> None:
        """Clean up monitoring resources."""
        await self.stop_monitoring()
        self.metrics_buffer.clear()
        self.reference_distributions.clear()
        self.drift_detectors.clear()
        self.alerts.clear()
        self.alert_callbacks.clear()


class DriftDetector:
    """
    Detects concept drift using statistical methods.

    Supports multiple drift detection algorithms:
    - Kolmogorov-Smirnov test
    - Population Stability Index (PSI)
    - Kullback-Leibler divergence
    """

    def __init__(self, method: str = "ks_test", threshold: float = 0.05):
        self.method = method
        self.threshold = threshold
        self.reference_data: Optional[np.ndarray] = None

    def set_reference(self, reference_data: np.ndarray) -> None:
        """Set reference data for drift detection."""
        self.reference_data = reference_data.copy()

    def detect_drift(self, current_data: np.ndarray) -> DriftDetectionResult:
        """
        Detect drift between reference and current data.

        Args:
            current_data: Current feature distribution

        Returns:
            Drift detection result
        """
        if self.reference_data is None:
            # Initialize with current data as reference
            self.reference_data = current_data.copy()
            return DriftDetectionResult(
                model_name="",  # Will be set by caller
                timestamp=time.time(),
                drift_detected=False,
                drift_score=0.0,
                threshold=self.threshold,
                method=self.method,
            )

        # Calculate drift score based on method
        if self.method == "ks_test":
            drift_score = self._ks_test_drift(current_data, self.reference_data)
        elif self.method == "psi":
            drift_score = self._psi_drift(current_data, self.reference_data)
        elif self.method == "kl_divergence":
            drift_score = self._kl_divergence_drift(current_data, self.reference_data)
        else:
            raise ValueError(f"Unsupported drift detection method: {self.method}")

        drift_detected = drift_score > self.threshold

        return DriftDetectionResult(
            model_name="",  # Will be set by caller
            timestamp=time.time(),
            drift_detected=drift_detected,
            drift_score=drift_score,
            threshold=self.threshold,
            method=self.method,
            details={"reference_samples": len(self.reference_data), "current_samples": len(current_data)},
        )

    def _ks_test_drift(self, current: np.ndarray, reference: np.ndarray) -> float:
        """Detect drift using Kolmogorov-Smirnov test."""
        # Flatten and sample for efficiency
        current_flat = current.flatten()
        reference_flat = reference.flatten()

        # Sample if too large
        max_samples = 10000
        if len(current_flat) > max_samples:
            current_flat = np.random.choice(current_flat, max_samples, replace=False)
        if len(reference_flat) > max_samples:
            reference_flat = np.random.choice(reference_flat, max_samples, replace=False)

        # Perform KS test on distributions
        statistic, p_value = stats.ks_2samp(current_flat, reference_flat)

        # Return 1 - p_value as drift score (higher = more drift)
        return 1.0 - p_value

    def _psi_drift(self, current: np.ndarray, reference: np.ndarray) -> float:
        """Detect drift using Population Stability Index."""
        # Calculate PSI for each feature
        psi_scores = []

        for i in range(current.shape[1]):
            current_feature = current[:, i]
            reference_feature = reference[:, i]

            # Create histograms
            bins = np.histogram_bin_edges(np.concatenate([current_feature, reference_feature]), bins=10)

            current_hist, _ = np.histogram(current_feature, bins=bins, density=True)
            reference_hist, _ = np.histogram(reference_feature, bins=bins, density=True)

            # Avoid division by zero
            reference_hist = np.where(reference_hist == 0, 1e-10, reference_hist)
            current_hist = np.where(current_hist == 0, 1e-10, current_hist)

            # Calculate PSI
            psi = np.sum((current_hist - reference_hist) * np.log(current_hist / reference_hist))
            psi_scores.append(psi)

        # Return average PSI
        return np.mean(psi_scores)

    def _kl_divergence_drift(self, current: np.ndarray, reference: np.ndarray) -> float:
        """Detect drift using KL divergence."""
        # Calculate KL divergence for each feature
        kl_scores = []

        for i in range(current.shape[1]):
            current_feature = current[:, i]
            reference_feature = reference[:, i]

            # Create histograms
            bins = np.histogram_bin_edges(np.concatenate([current_feature, reference_feature]), bins=10)

            current_hist, _ = np.histogram(current_feature, bins=bins, density=True)
            reference_hist, _ = np.histogram(reference_feature, bins=bins, density=True)

            # Avoid division by zero
            reference_hist = np.where(reference_hist == 0, 1e-10, reference_hist)

            # Calculate KL divergence
            kl = np.sum(current_hist * np.log(current_hist / reference_hist))
            kl_scores.append(kl)

        # Return average KL divergence
        return np.mean(kl_scores)
