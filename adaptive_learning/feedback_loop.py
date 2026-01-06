"""
Feedback Loop Module for Negative Space Imaging

Integrates real-world feedback and retraining triggers with:
- Feedback quality metrics computation
- Model retraining trigger detection
- Performance degradation monitoring
- User feedback integration
- Automatic retraining workflows
- Metric-based decision making

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeedbackMetrics:
    """Compute feedback quality metrics."""

    @staticmethod
    def compute_accuracy(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute prediction accuracy."""
        return float(np.mean(predictions == ground_truth))

    @staticmethod
    def compute_precision(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute precision (for binary classification)."""
        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        return float(tp / (tp + fp + 1e-6))

    @staticmethod
    def compute_recall(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute recall (for binary classification)."""
        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fn = np.sum((predictions == 0) & (ground_truth == 1))
        return float(tp / (tp + fn + 1e-6))

    @staticmethod
    def compute_f1_score(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute F1 score."""
        precision = FeedbackMetrics.compute_precision(predictions, ground_truth)
        recall = FeedbackMetrics.compute_recall(predictions, ground_truth)
        return float(2 * (precision * recall) / (precision + recall + 1e-6))

    @staticmethod
    def compute_confidence_interval(
        scores: np.ndarray, confidence: float = 0.95
    ) -> tuple:
        """Compute confidence interval for scores."""
        mean = np.mean(scores)
        std_error = np.std(scores) / np.sqrt(len(scores))
        z_score = 1.96 if confidence == 0.95 else 2.576  # 99%
        margin = z_score * std_error
        return (mean - margin, mean + margin)


class PerformanceMonitor:
    """Monitor model performance over time."""

    def __init__(self, window_size: int = 100, degradation_threshold: float = 0.05):
        """Initialize performance monitor."""
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.performance_history = deque(maxlen=window_size)
        self.baseline_performance = None

    def update(self, metric_value: float) -> bool:
        """Update with new metric and check for degradation."""
        self.performance_history.append(metric_value)

        if self.baseline_performance is None:
            self.baseline_performance = metric_value
            return False

        if len(self.performance_history) >= self.window_size // 2:
            current_avg = np.mean(list(self.performance_history))
            degradation = (self.baseline_performance - current_avg) / self.baseline_performance

            if degradation > self.degradation_threshold:
                logger.warning(f"Performance degradation detected: {degradation:.4f}")
                return True

        return False

    def get_trend(self) -> str:
        """Get performance trend."""
        if len(self.performance_history) < 10:
            return "insufficient_data"

        recent = list(self.performance_history)[-10:]
        older = list(self.performance_history)[-20:-10] if len(self.performance_history) >= 20 else recent

        if np.mean(recent) > np.mean(older):
            return "improving"
        elif np.mean(recent) < np.mean(older):
            return "degrading"
        else:
            return "stable"

    def get_statistics(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.performance_history:
            return {}

        values = np.array(list(self.performance_history))
        return {
            "current": float(values[-1]),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "trend": self.get_trend(),
        }


class RetrainingTrigger:
    """Determine when to trigger model retraining."""

    def __init__(
        self,
        performance_threshold: float = 0.1,
        feedback_accumulation_threshold: int = 100,
        time_interval_threshold: int = 3600,
    ):
        """Initialize retraining trigger."""
        self.performance_threshold = performance_threshold
        self.feedback_accumulation_threshold = feedback_accumulation_threshold
        self.time_interval_threshold = time_interval_threshold

        self.feedback_count = 0
        self.last_retraining_time = 0
        self.performance_monitor = PerformanceMonitor()

    def check_trigger(
        self,
        current_performance: float,
        current_time: float,
        new_feedback_count: int = 1,
    ) -> bool:
        """Check if retraining should be triggered."""
        self.feedback_count += new_feedback_count

        triggers = []

        # Trigger 1: Performance degradation
        degradation_detected = self.performance_monitor.update(current_performance)
        if degradation_detected:
            triggers.append("performance_degradation")

        # Trigger 2: Feedback accumulation
        if self.feedback_count >= self.feedback_accumulation_threshold:
            triggers.append("feedback_accumulation")
            self.feedback_count = 0

        # Trigger 3: Time interval
        if current_time - self.last_retraining_time > self.time_interval_threshold:
            triggers.append("time_interval")

        should_retrain = len(triggers) > 0

        if should_retrain:
            self.last_retraining_time = current_time
            logger.info(f"Retraining triggered by: {triggers}")

        return should_retrain

    def get_status(self) -> Dict[str, Any]:
        """Get trigger status."""
        return {
            "feedback_accumulated": self.feedback_count,
            "performance_stats": self.performance_monitor.get_statistics(),
        }


class FeedbackCollector:
    """Collect and manage user feedback."""

    def __init__(self, max_feedback_size: int = 10000):
        """Initialize feedback collector."""
        self.max_feedback_size = max_feedback_size
        self.feedback_data = deque(maxlen=max_feedback_size)
        self.feedback_count = 0

    def add_feedback(
        self,
        prediction: Any,
        ground_truth: Any,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a feedback entry."""
        feedback_entry = {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "confidence": confidence,
            "metadata": metadata or {},
            "correct": prediction == ground_truth,
        }

        self.feedback_data.append(feedback_entry)
        self.feedback_count += 1

        logger.debug(f"Feedback added: correct={feedback_entry['correct']}")

    def get_incorrect_predictions(self) -> List[Dict[str, Any]]:
        """Get all incorrect predictions."""
        return [fb for fb in self.feedback_data if not fb["correct"]]

    def get_low_confidence_predictions(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Get predictions with low confidence."""
        return [
            fb
            for fb in self.feedback_data
            if fb["confidence"] is not None and fb["confidence"] < threshold
        ]

    def get_batch_for_retraining(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Get batch of feedback for retraining."""
        # Prioritize incorrect predictions
        incorrect = self.get_incorrect_predictions()
        if len(incorrect) >= batch_size:
            return incorrect[:batch_size]

        # Fall back to all feedback
        all_feedback = list(self.feedback_data)
        return all_feedback[:batch_size]

    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        if not self.feedback_data:
            return {}

        incorrect = self.get_incorrect_predictions()
        low_confidence = self.get_low_confidence_predictions()

        return {
            "total_feedback": len(self.feedback_data),
            "incorrect_predictions": len(incorrect),
            "low_confidence_predictions": len(low_confidence),
            "accuracy": (len(self.feedback_data) - len(incorrect)) / len(self.feedback_data),
        }

    def clear(self) -> None:
        """Clear all feedback."""
        self.feedback_data.clear()
        logger.info("Feedback collector cleared")


class FeedbackLoop:
    """Main feedback loop for continuous improvement."""

    def __init__(
        self,
        model: nn.Module,
        performance_threshold: float = 0.1,
        feedback_threshold: int = 100,
    ):
        """Initialize feedback loop."""
        self.model = model
        self.collector = FeedbackCollector()
        self.trigger = RetrainingTrigger(
            performance_threshold=performance_threshold,
            feedback_accumulation_threshold=feedback_threshold,
        )
        self.metrics = FeedbackMetrics()
        self.retrain_callbacks: List[Callable] = []

        logger.info("Feedback loop initialized")

    def submit_feedback(
        self,
        prediction: Any,
        ground_truth: Any,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Submit user feedback."""
        self.collector.add_feedback(prediction, ground_truth, confidence, metadata)

    def evaluate_on_feedback(self) -> Dict[str, Any]:
        """Evaluate model performance on collected feedback."""
        feedback_list = list(self.collector.feedback_data)

        if not feedback_list:
            return {}

        predictions = np.array([fb["prediction"] for fb in feedback_list])
        ground_truths = np.array([fb["ground_truth"] for fb in feedback_list])

        return {
            "accuracy": self.metrics.compute_accuracy(predictions, ground_truths),
            "precision": self.metrics.compute_precision(predictions, ground_truths),
            "recall": self.metrics.compute_recall(predictions, ground_truths),
            "f1_score": self.metrics.compute_f1_score(predictions, ground_truths),
        }

    def check_and_trigger_retraining(self, current_time: float) -> bool:
        """Check if retraining should be triggered."""
        metrics = self.evaluate_on_feedback()

        if not metrics:
            return False

        accuracy = metrics.get("accuracy", 0.0)
        should_retrain = self.trigger.check_trigger(
            accuracy, current_time, new_feedback_count=len(self.collector.feedback_data)
        )

        if should_retrain:
            self._trigger_retraining()

        return should_retrain

    def _trigger_retraining(self) -> None:
        """Trigger retraining callbacks."""
        logger.info("Triggering retraining...")

        for callback in self.retrain_callbacks:
            try:
                callback(self.collector.get_batch_for_retraining())
            except Exception as e:
                logger.error(f"Retraining callback failed: {e}")

    def register_retrain_callback(self, callback: Callable) -> None:
        """Register retraining callback."""
        self.retrain_callbacks.append(callback)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics for debugging."""
        return {
            "feedback_stats": self.collector.get_statistics(),
            "trigger_status": self.trigger.get_status(),
            "performance_metrics": self.evaluate_on_feedback(),
        }
