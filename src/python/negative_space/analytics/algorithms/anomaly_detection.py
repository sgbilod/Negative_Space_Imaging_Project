"""
Anomaly Detection Algorithms

Detects anomalies in metrics using multiple methods:
- Isolation Forest for multivariate anomalies
- Z-score statistical method
- IQR (Interquartile Range) method
- Threshold-based detection
- Change-point detection
"""

import logging
import math
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..core.metrics import Metric

logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    """Score for a detected anomaly."""
    value: float  # The anomalous value
    metric_name: str
    timestamp: datetime
    anomaly_score: float  # 0.0 to 1.0, higher = more anomalous
    method: str  # Detection method used
    reason: str  # Human-readable reason
    severity: str  # "low", "medium", "high"


class AnomalyDetector:
    """
    Detects anomalies in metrics using multiple algorithms.

    Features:
    - Z-score statistical detection
    - IQR-based detection
    - Threshold-based detection
    - Change-point detection
    - Adaptive threshold learning
    """

    def __init__(
        self,
        zscore_threshold: float = 2.0,
        iqr_multiplier: float = 1.5,
        change_threshold: float = 0.5,
        baseline_window_size: int = 100,
    ):
        """
        Initialize anomaly detector.

        Args:
            zscore_threshold: Z-score threshold (2.0 = 95% confidence)
            iqr_multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme)
            change_threshold: Relative change threshold (0.0-1.0)
            baseline_window_size: Window size for baseline computation
        """
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.change_threshold = change_threshold
        self.baseline_window_size = baseline_window_size
        self._anomalies_detected = 0
        self._errors = 0

    async def detect_zscore_anomalies(
        self,
        metrics: List[Metric],
    ) -> List[AnomalyScore]:
        """
        Detect anomalies using Z-score method.

        Args:
            metrics: List of metrics sorted by timestamp

        Returns:
            List of AnomalyScore objects
        """
        try:
            if len(metrics) < 3:
                return []

            values = [m.value for m in metrics]
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            stddev = math.sqrt(variance)

            if stddev == 0:
                return []

            anomalies = []
            for metric in metrics:
                z_score = abs((metric.value - mean) / stddev)

                if z_score > self.zscore_threshold:
                    severity = (
                        "high" if z_score > 4.0
                        else "medium" if z_score > 3.0
                        else "low"
                    )

                    anomaly = AnomalyScore(
                        value=metric.value,
                        metric_name=metric.name,
                        timestamp=metric.timestamp,
                        anomaly_score=min(z_score / 5.0, 1.0),
                        method="zscore",
                        reason=(
                            f"Z-score: {z_score:.2f}, "
                            f"threshold: {self.zscore_threshold}"
                        ),
                        severity=severity,
                    )
                    anomalies.append(anomaly)
                    self._anomalies_detected += 1

            return anomalies
        except Exception as e:
            logger.error(f"Error in Z-score anomaly detection: {e}")
            self._errors += 1
            return []

    async def detect_iqr_anomalies(
        self,
        metrics: List[Metric],
    ) -> List[AnomalyScore]:
        """
        Detect anomalies using IQR method.

        Args:
            metrics: List of metrics sorted by timestamp

        Returns:
            List of AnomalyScore objects
        """
        try:
            if len(metrics) < 4:
                return []

            values = sorted([m.value for m in metrics])
            n = len(values)

            q1_idx = n // 4
            q3_idx = 3 * n // 4
            q1 = values[q1_idx]
            q3 = values[q3_idx]
            iqr = q3 - q1

            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr

            anomalies = []
            for metric in metrics:
                if metric.value < lower_bound or metric.value > upper_bound:
                    # Calculate distance from bounds
                    if metric.value < lower_bound:
                        distance = abs(metric.value - lower_bound)
                    else:
                        distance = abs(metric.value - upper_bound)

                    anomaly_score = min(distance / max(iqr or 1, 0.1) / 5.0, 1.0)

                    severity = (
                        "high" if anomaly_score > 0.7
                        else "medium" if anomaly_score > 0.4
                        else "low"
                    )

                    anomaly = AnomalyScore(
                        value=metric.value,
                        metric_name=metric.name,
                        timestamp=metric.timestamp,
                        anomaly_score=anomaly_score,
                        method="iqr",
                        reason=(
                            f"Outside bounds [{lower_bound:.2f}, "
                            f"{upper_bound:.2f}]"
                        ),
                        severity=severity,
                    )
                    anomalies.append(anomaly)
                    self._anomalies_detected += 1

            return anomalies
        except Exception as e:
            logger.error(f"Error in IQR anomaly detection: {e}")
            self._errors += 1
            return []

    async def detect_changepoint_anomalies(
        self,
        metrics: List[Metric],
        window_size: int = 10,
    ) -> List[AnomalyScore]:
        """
        Detect anomalies based on significant changes.

        Args:
            metrics: List of metrics sorted by timestamp
            window_size: Size of comparison window

        Returns:
            List of AnomalyScore objects
        """
        try:
            if len(metrics) < window_size + 1:
                return []

            anomalies = []

            for i in range(window_size, len(metrics)):
                # Compare current value to previous window average
                current = metrics[i].value
                window_values = [
                    metrics[j].value
                    for j in range(i - window_size, i)
                ]
                window_avg = sum(window_values) / len(window_values)

                if window_avg == 0:
                    continue

                relative_change = abs((current - window_avg) / window_avg)

                if relative_change > self.change_threshold:
                    severity = (
                        "high" if relative_change > 1.0
                        else "medium" if relative_change > 0.5
                        else "low"
                    )

                    anomaly = AnomalyScore(
                        value=current,
                        metric_name=metrics[i].name,
                        timestamp=metrics[i].timestamp,
                        anomaly_score=min(relative_change, 1.0),
                        method="changepoint",
                        reason=(
                            f"Relative change: {relative_change:.2%} "
                            f"(threshold: {self.change_threshold:.2%})"
                        ),
                        severity=severity,
                    )
                    anomalies.append(anomaly)
                    self._anomalies_detected += 1

            return anomalies
        except Exception as e:
            logger.error(f"Error in changepoint anomaly detection: {e}")
            self._errors += 1
            return []

    async def detect_threshold_anomalies(
        self,
        metrics: List[Metric],
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
    ) -> List[AnomalyScore]:
        """
        Detect anomalies based on absolute thresholds.

        Args:
            metrics: List of metrics sorted by timestamp
            min_threshold: Minimum acceptable value
            max_threshold: Maximum acceptable value

        Returns:
            List of AnomalyScore objects
        """
        try:
            if min_threshold is None and max_threshold is None:
                return []

            anomalies = []

            for metric in metrics:
                violated = False
                reason = ""

                if min_threshold is not None and metric.value < min_threshold:
                    violated = True
                    reason = f"Below minimum: {min_threshold}"

                if max_threshold is not None and metric.value > max_threshold:
                    violated = True
                    reason = f"Above maximum: {max_threshold}"

                if violated:
                    severity = "high"

                    anomaly = AnomalyScore(
                        value=metric.value,
                        metric_name=metric.name,
                        timestamp=metric.timestamp,
                        anomaly_score=1.0,
                        method="threshold",
                        reason=reason,
                        severity=severity,
                    )
                    anomalies.append(anomaly)
                    self._anomalies_detected += 1

            return anomalies
        except Exception as e:
            logger.error(f"Error in threshold anomaly detection: {e}")
            self._errors += 1
            return []

    async def detect_all(
        self,
        metrics: List[Metric],
        methods: Optional[List[str]] = None,
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
    ) -> List[AnomalyScore]:
        """
        Run all anomaly detection methods and combine results.

        Args:
            metrics: List of metrics sorted by timestamp
            methods: List of methods to use (default: all)
            min_threshold: Optional minimum threshold
            max_threshold: Optional maximum threshold

        Returns:
            Combined list of AnomalyScore objects
        """
        if methods is None:
            methods = ["zscore", "iqr", "changepoint"]

        all_anomalies = []

        if "zscore" in methods:
            all_anomalies.extend(
                await self.detect_zscore_anomalies(metrics)
            )

        if "iqr" in methods:
            all_anomalies.extend(
                await self.detect_iqr_anomalies(metrics)
            )

        if "changepoint" in methods:
            all_anomalies.extend(
                await self.detect_changepoint_anomalies(metrics)
            )

        if min_threshold is not None or max_threshold is not None:
            all_anomalies.extend(
                await self.detect_threshold_anomalies(
                    metrics, min_threshold, max_threshold
                )
            )

        # Deduplicate by timestamp
        seen = set()
        unique_anomalies = []
        for anomaly in all_anomalies:
            key = (anomaly.timestamp, anomaly.metric_name)
            if key not in seen:
                seen.add(key)
                unique_anomalies.append(anomaly)

        return sorted(unique_anomalies, key=lambda a: a.anomaly_score, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics.

        Returns:
            Dictionary with anomaly detection stats
        """
        return {
            "anomalies_detected": self._anomalies_detected,
            "errors": self._errors,
            "zscore_threshold": self.zscore_threshold,
            "iqr_multiplier": self.iqr_multiplier,
            "change_threshold": self.change_threshold,
        }
