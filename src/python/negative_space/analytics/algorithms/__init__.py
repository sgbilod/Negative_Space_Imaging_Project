"""
Algorithms for advanced analytics computations.

Includes statistical analysis, correlation detection, trend analysis,
outlier detection, and anomaly detection algorithms.
"""

from .statistical import (
    StatisticalAnalyzer,
    CorrelationAnalyzer,
    CorrelationResult,
    TrendResult,
)
from .anomaly_detection import (
    AnomalyDetector,
    AnomalyScore,
)

__all__ = [
    "StatisticalAnalyzer",
    "CorrelationAnalyzer",
    "CorrelationResult",
    "TrendResult",
    "AnomalyDetector",
    "AnomalyScore",
]
