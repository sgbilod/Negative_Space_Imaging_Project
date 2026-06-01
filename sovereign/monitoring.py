"""Performance monitoring for Negative Space Imaging System."""

import time
import logging
import psutil
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    quantum_coherence: float
    operation_time: float
    success_rate: float

class PerformanceMonitor:
    """System-wide performance monitoring."""

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize performance monitor.

        Args:
            log_dir: Directory to store performance logs
        """
        self.start_time = time.time()
        self.metrics_history: List[PerformanceMetrics] = []
        self.process = psutil.Process()

        # Setup logging
        self.log_dir = log_dir or Path.cwd() / "logs" / "performance"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("PerformanceMonitor")
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure performance logging."""
        log_file = self.log_dir / f"performance_{datetime.now():%Y%m%d}.log"

        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            )
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def collect_metrics(
        self,
        quantum_coherence: float = 0.0,
        operation_time: float = 0.0,
        success_rate: float = 0.0
    ) -> PerformanceMetrics:
        """Collect current performance metrics.

        Args:
            quantum_coherence: Current quantum coherence value
            operation_time: Time taken for last operation
            success_rate: Current success rate

        Returns:
            PerformanceMetrics object with current values
        """
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=self.process.cpu_percent(),
            memory_usage=self.process.memory_info().rss / (1024 * 1024),  # MB
            quantum_coherence=quantum_coherence,
            operation_time=operation_time,
            success_rate=success_rate
        )

        self.metrics_history.append(metrics)
        self._log_metrics(metrics)

        return metrics

    def _log_metrics(self, metrics: PerformanceMetrics) -> None:
        """Log performance metrics.

        Args:
            metrics: Metrics to log
        """
        self.logger.info(
            f"Performance Metrics | "
            f"CPU: {metrics.cpu_usage:.1f}% | "
            f"Memory: {metrics.memory_usage:.1f}MB | "
            f"Coherence: {metrics.quantum_coherence:.3f} | "
            f"Op Time: {metrics.operation_time:.3f}s | "
            f"Success: {metrics.success_rate:.1%}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Generate performance summary.

        Returns:
            Dict containing performance statistics
        """
        if not self.metrics_history:
            return {}

        metrics_array = np.array([
            [m.cpu_usage for m in self.metrics_history],
            [m.memory_usage for m in self.metrics_history],
            [m.quantum_coherence for m in self.metrics_history],
            [m.operation_time for m in self.metrics_history],
            [m.success_rate for m in self.metrics_history]
        ])

        return {
            "runtime": time.time() - self.start_time,
            "total_operations": len(self.metrics_history),
            "avg_cpu_usage": np.mean(metrics_array[0]),
            "avg_memory_usage": np.mean(metrics_array[1]),
            "avg_coherence": np.mean(metrics_array[2]),
            "avg_operation_time": np.mean(metrics_array[3]),
            "success_rate": np.mean(metrics_array[4]),
            "peak_memory": np.max(metrics_array[1]),
            "peak_cpu": np.max(metrics_array[0])
        }

    def reset(self) -> None:
        """Reset performance monitor."""
        self.start_time = time.time()
        self.metrics_history.clear()

    def export_metrics(self, file_path: Path) -> None:
        """Export metrics to file.

        Args:
            file_path: Path to export metrics to
        """
        import json

        summary = self.get_summary()
        summary["export_time"] = datetime.now().isoformat()

        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Exported performance metrics to {file_path}")
