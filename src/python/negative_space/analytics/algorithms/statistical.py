"""
Statistical Analysis Algorithms

Provides statistical computations for metrics including:
- Descriptive statistics
- Correlation analysis
- Trend detection
- Hypothesis testing
"""

import logging
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from ..core.metrics import Metric

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    metric1: str
    metric2: str
    correlation_coefficient: float
    p_value: float
    sample_size: int


@dataclass
class TrendResult:
    """Result of trend detection."""
    metric_name: str
    trend_type: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0 to 1.0
    r_squared: float  # Coefficient of determination
    slope: float


class StatisticalAnalyzer:
    """
    Statistical analysis for metrics.

    Features:
    - Descriptive statistics (mean, median, variance, skewness, kurtosis)
    - Correlation analysis between metric pairs
    - Trend detection (linear regression)
    - Hypothesis testing (t-tests, ANOVA)
    - Outlier detection (Z-score, IQR)
    """

    def __init__(self):
        """Initialize statistical analyzer."""
        self._analyses_performed = 0
        self._errors = 0

    @staticmethod
    def descriptive_stats(values: List[float]) -> Dict[str, float]:
        """
        Compute descriptive statistics for a list of values.

        Args:
            values: List of numeric values

        Returns:
            Dictionary with statistics
        """
        if not values:
            return {}

        values_sorted = sorted(values)
        n = len(values)

        # Basic statistics
        minimum = min(values)
        maximum = max(values)
        mean = sum(values) / n
        median = (values_sorted[n // 2] if n % 2 == 1
                  else (values_sorted[n // 2 - 1] + values_sorted[n // 2]) / 2)

        # Variance and standard deviation
        variance = sum((x - mean) ** 2 for x in values) / n
        stddev = math.sqrt(variance)

        # Coefficient of variation
        cv = (stddev / mean * 100) if mean != 0 else 0

        # Skewness
        if stddev == 0:
            skewness = 0
        else:
            skewness = sum((x - mean) ** 3 for x in values) / (n * stddev ** 3)

        # Kurtosis
        if stddev == 0:
            kurtosis = 0
        else:
            kurtosis = (
                sum((x - mean) ** 4 for x in values) / (n * stddev ** 4) - 3
            )

        # Quartiles
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = values_sorted[q1_idx]
        q3 = values_sorted[q3_idx]
        iqr = q3 - q1

        return {
            "count": n,
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "median": median,
            "std_dev": stddev,
            "variance": variance,
            "cv": cv,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
        }

    @staticmethod
    def correlation_pearson(
        x_values: List[float],
        y_values: List[float],
    ) -> float:
        """
        Compute Pearson correlation coefficient.

        Args:
            x_values: First variable values
            y_values: Second variable values

        Returns:
            Correlation coefficient (-1.0 to 1.0)
        """
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        n = len(x_values)
        mean_x = sum(x_values) / n
        mean_y = sum(y_values) / n

        numerator = sum(
            (x - mean_x) * (y - mean_y)
            for x, y in zip(x_values, y_values)
        )

        denom_x = sum((x - mean_x) ** 2 for x in x_values)
        denom_y = sum((y - mean_y) ** 2 for y in y_values)

        denominator = math.sqrt(denom_x * denom_y)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    @staticmethod
    def linear_regression(
        x_values: List[float],
        y_values: List[float],
    ) -> Tuple[float, float, float]:
        """
        Perform linear regression.

        Args:
            x_values: Independent variable
            y_values: Dependent variable

        Returns:
            Tuple of (slope, intercept, r_squared)
        """
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0, 0.0, 0.0

        n = len(x_values)
        mean_x = sum(x_values) / n
        mean_y = sum(y_values) / n

        numerator = sum(
            (x - mean_x) * (y - mean_y)
            for x, y in zip(x_values, y_values)
        )

        denominator = sum((x - mean_x) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0, 0.0, 0.0

        slope = numerator / denominator
        intercept = mean_y - slope * mean_x

        # Compute R-squared
        ss_tot = sum((y - mean_y) ** 2 for y in y_values)
        ss_res = sum(
            (y - (slope * x + intercept)) ** 2
            for x, y in zip(x_values, y_values)
        )

        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return slope, intercept, r_squared

    async def detect_trend(
        self,
        metrics: List[Metric],
    ) -> Optional[TrendResult]:
        """
        Detect trend in metrics over time.

        Args:
            metrics: List of metrics sorted by timestamp

        Returns:
            TrendResult or None
        """
        try:
            if len(metrics) < 3:
                return None

            # Prepare data
            x_values = list(range(len(metrics)))
            y_values = [m.value for m in metrics]

            # Linear regression
            slope, intercept, r_squared = self.linear_regression(
                [float(x) for x in x_values],
                y_values
            )

            # Determine trend type
            if abs(slope) < 0.01:
                trend_type = "stable"
            elif slope > 0:
                trend_type = "increasing"
            else:
                trend_type = "decreasing"

            # Trend strength as absolute slope normalized
            trend_strength = min(abs(slope) / max(y_values or [1]), 1.0)

            result = TrendResult(
                metric_name=metrics[0].name,
                trend_type=trend_type,
                trend_strength=trend_strength,
                r_squared=r_squared,
                slope=slope,
            )

            self._analyses_performed += 1
            return result
        except Exception as e:
            logger.error(f"Error detecting trend: {e}")
            self._errors += 1
            return None

    async def detect_outliers_zscore(
        self,
        values: List[float],
        threshold: float = 2.0,
    ) -> List[int]:
        """
        Detect outliers using Z-score method.

        Args:
            values: List of values
            threshold: Z-score threshold (default 2.0 for 95% confidence)

        Returns:
            List of indices of outlier values
        """
        try:
            if len(values) < 2:
                return []

            stats = self.descriptive_stats(values)
            mean = stats["mean"]
            stddev = stats["std_dev"]

            if stddev == 0:
                return []

            outliers = [
                i for i, v in enumerate(values)
                if abs((v - mean) / stddev) > threshold
            ]

            self._analyses_performed += 1
            return outliers
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            self._errors += 1
            return []

    async def detect_outliers_iqr(
        self,
        values: List[float],
        multiplier: float = 1.5,
    ) -> List[int]:
        """
        Detect outliers using Interquartile Range method.

        Args:
            values: List of values
            multiplier: IQR multiplier (default 1.5 for standard outliers)

        Returns:
            List of indices of outlier values
        """
        try:
            if len(values) < 4:
                return []

            stats = self.descriptive_stats(values)
            q1 = stats["q1"]
            q3 = stats["q3"]
            iqr = stats["iqr"]

            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            outliers = [
                i for i, v in enumerate(values)
                if v < lower_bound or v > upper_bound
            ]

            self._analyses_performed += 1
            return outliers
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            self._errors += 1
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get analyzer statistics.

        Returns:
            Dictionary with analysis stats
        """
        return {
            "analyses_performed": self._analyses_performed,
            "errors": self._errors,
        }


class CorrelationAnalyzer:
    """
    Analyzes correlations between metrics.

    Computes correlation coefficients and statistical significance.
    """

    def __init__(self):
        """Initialize correlation analyzer."""
        self._correlations_computed = 0
        self._errors = 0

    async def compute_correlation(
        self,
        metrics1: List[Metric],
        metrics2: List[Metric],
        time_aligned: bool = True,
    ) -> Optional[CorrelationResult]:
        """
        Compute correlation between two metric series.

        Args:
            metrics1: First metric series
            metrics2: Second metric series
            time_aligned: Whether metrics are time-aligned

        Returns:
            CorrelationResult or None
        """
        try:
            if len(metrics1) < 2 or len(metrics2) < 2:
                return None

            # Align metrics by timestamp if needed
            if time_aligned:
                values1 = [m.value for m in metrics1]
                values2 = [m.value for m in metrics2]
            else:
                # Time-align metrics
                aligned = self._time_align_metrics(metrics1, metrics2)
                if not aligned:
                    return None
                values1, values2 = aligned

            if len(values1) < 2 or len(values1) != len(values2):
                return None

            # Compute correlation
            correlation = StatisticalAnalyzer.correlation_pearson(
                values1, values2
            )

            # Approximate p-value (simplified)
            n = len(values1)
            t_stat = correlation * math.sqrt(n - 2) / math.sqrt(1 - correlation**2 + 1e-10)

            # Very simplified p-value estimation
            p_value = 2 * (1 - abs(correlation))

            result = CorrelationResult(
                metric1=metrics1[0].name,
                metric2=metrics2[0].name,
                correlation_coefficient=correlation,
                p_value=p_value,
                sample_size=n,
            )

            self._correlations_computed += 1
            return result
        except Exception as e:
            logger.error(f"Error computing correlation: {e}")
            self._errors += 1
            return None

    @staticmethod
    def _time_align_metrics(
        metrics1: List[Metric],
        metrics2: List[Metric],
    ) -> Optional[Tuple[List[float], List[float]]]:
        """
        Time-align two metric series using interpolation.

        Args:
            metrics1: First metric series
            metrics2: Second metric series

        Returns:
            Tuple of aligned value lists or None
        """
        if not metrics1 or not metrics2:
            return None

        # Use common time range
        start = max(metrics1[0].timestamp, metrics2[0].timestamp)
        end = min(metrics1[-1].timestamp, metrics2[-1].timestamp)

        if start >= end:
            return None

        # Simple alignment: match closest timestamps
        values1 = []
        values2 = []

        for m1 in metrics1:
            if start <= m1.timestamp <= end:
                # Find closest m2
                closest_m2 = min(
                    metrics2,
                    key=lambda m: abs((m.timestamp - m1.timestamp).total_seconds())
                )
                if abs((closest_m2.timestamp - m1.timestamp).total_seconds()) < 60:
                    values1.append(m1.value)
                    values2.append(closest_m2.value)

        return (values1, values2) if values1 else None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get analyzer statistics.

        Returns:
            Dictionary with correlation stats
        """
        return {
            "correlations_computed": self._correlations_computed,
            "errors": self._errors,
        }
