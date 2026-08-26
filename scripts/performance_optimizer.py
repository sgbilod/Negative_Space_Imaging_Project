#!/usr/bin/env python
"""
Performance Optimization Manager for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import json
import time
import psutil
import logging
import threading
from enum import Enum
from typing import Dict, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

@dataclass
class PerformanceMetric:
    name: str
    value: float
    unit: str
    timestamp: str
    threshold: float
    is_critical: bool

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    GPU = "gpu"
    NETWORK = "network"

class OptimizationAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    THROTTLE = "throttle"
    CACHE = "cache"
    OFFLOAD = "offload"

class PerformanceOptimizer:
    """Real-time performance optimization system."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = self._setup_logger()
        self.metrics_history = []
        self.optimization_rules = {}
        self.active_optimizations = {}
        self.monitor_thread = None
        self.should_run = threading.Event()

        # Load optimization configurations
        self.load_optimization_rules()

    def _setup_logger(self) -> logging.Logger:
        """Configure logging for performance optimization."""
        logger = logging.getLogger("performance_optimizer")
        logger.setLevel(logging.INFO)

        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(
            log_dir / "performance.log"
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def load_optimization_rules(self):
        """Load optimization rules from configuration."""
        config_path = self.project_root / "config" / "optimization.json"
        if not config_path.exists():
            self._create_default_rules()
            return

        try:
            with open(config_path) as f:
                self.optimization_rules = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load optimization rules: {e}")
            self._create_default_rules()

    def _create_default_rules(self):
        """Create default optimization rules."""
        self.optimization_rules = {
            ResourceType.CPU.value: {
                "high_threshold": 80,
                "critical_threshold": 90,
                "actions": [
                    {
                        "trigger": "high",
                        "action": OptimizationAction.THROTTLE.value,
                        "params": {"factor": 0.8}
                    },
                    {
                        "trigger": "critical",
                        "action": OptimizationAction.OFFLOAD.value,
                        "params": {"target": "gpu"}
                    }
                ]
            },
            ResourceType.MEMORY.value: {
                "high_threshold": 85,
                "critical_threshold": 95,
                "actions": [
                    {
                        "trigger": "high",
                        "action": OptimizationAction.CACHE.value,
                        "params": {"clear_factor": 0.3}
                    },
                    {
                        "trigger": "critical",
                        "action": OptimizationAction.SCALE_UP.value,
                        "params": {"increment": "2GB"}
                    }
                ]
            },
            ResourceType.GPU.value: {
                "high_threshold": 85,
                "critical_threshold": 95,
                "actions": [
                    {
                        "trigger": "high",
                        "action": OptimizationAction.THROTTLE.value,
                        "params": {"factor": 0.7}
                    },
                    {
                        "trigger": "critical",
                        "action": OptimizationAction.OFFLOAD.value,
                        "params": {"target": "cpu"}
                    }
                ]
            }
        }

        # Save default rules
        config_path = self.project_root / "config" / "optimization.json"
        with open(config_path, 'w') as f:
            json.dump(self.optimization_rules, f, indent=2)

    def collect_metrics(self) -> List[PerformanceMetric]:
        """Collect current performance metrics."""
        metrics = []
        timestamp = datetime.now().isoformat()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(PerformanceMetric(
            name=ResourceType.CPU.value,
            value=cpu_percent,
            unit="percent",
            timestamp=timestamp,
            threshold=self.optimization_rules[ResourceType.CPU.value]["high_threshold"],
            is_critical=cpu_percent >= self.optimization_rules[ResourceType.CPU.value]["critical_threshold"]
        ))

        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(PerformanceMetric(
            name=ResourceType.MEMORY.value,
            value=memory.percent,
            unit="percent",
            timestamp=timestamp,
            threshold=self.optimization_rules[ResourceType.MEMORY.value]["high_threshold"],
            is_critical=memory.percent >= self.optimization_rules[ResourceType.MEMORY.value]["critical_threshold"]
        ))

        # GPU metrics if available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

            metrics.append(PerformanceMetric(
                name=ResourceType.GPU.value,
                value=gpu_util,
                unit="percent",
                timestamp=timestamp,
                threshold=self.optimization_rules[ResourceType.GPU.value]["high_threshold"],
                is_critical=gpu_util >= self.optimization_rules[ResourceType.GPU.value]["critical_threshold"]
            ))
        except:
            pass

        return metrics

    def optimize_performance(self, metrics: List[PerformanceMetric]):
        """Apply performance optimizations based on metrics."""
        for metric in metrics:
            if metric.is_critical:
                self._apply_optimization(
                    metric,
                    trigger="critical"
                )
            elif metric.value >= metric.threshold:
                self._apply_optimization(
                    metric,
                    trigger="high"
                )

    def _apply_optimization(self, metric: PerformanceMetric, trigger: str):
        """Apply specific optimization action."""
        rules = self.optimization_rules[metric.name]
        actions = [a for a in rules["actions"] if a["trigger"] == trigger]

        for action in actions:
            try:
                self.logger.info(
                    f"Applying {action['action']} optimization for {metric.name}"
                )

                # Record optimization
                self.active_optimizations[metric.name] = {
                    "action": action["action"],
                    "params": action["params"],
                    "timestamp": datetime.now().isoformat()
                }

                # Apply optimization logic
                if action["action"] == OptimizationAction.THROTTLE.value:
                    self._apply_throttling(metric.name, action["params"])
                elif action["action"] == OptimizationAction.CACHE.value:
                    self._manage_cache(metric.name, action["params"])
                elif action["action"] == OptimizationAction.OFFLOAD.value:
                    self._offload_processing(metric.name, action["params"])
                elif action["action"] == OptimizationAction.SCALE_UP.value:
                    self._scale_resource(metric.name, action["params"])

            except Exception as e:
                self.logger.error(
                    f"Failed to apply optimization {action['action']}: {e}"
                )

    def _apply_throttling(self, resource: str, params: Dict):
        """Apply throttling to a resource."""
        self.logger.info(f"Throttling {resource} by factor {params['factor']}")
        # Implementation depends on resource type
        pass

    def _manage_cache(self, resource: str, params: Dict):
        """Manage cache for a resource."""
        self.logger.info(f"Managing cache for {resource}")
        # Implementation depends on resource type
        pass

    def _offload_processing(self, resource: str, params: Dict):
        """Offload processing to another resource."""
        self.logger.info(
            f"Offloading {resource} processing to {params['target']}"
        )
        # Implementation depends on resource type
        pass

    def _scale_resource(self, resource: str, params: Dict):
        """Scale a resource up or down."""
        self.logger.info(f"Scaling {resource} with {params}")
        # Implementation depends on resource type
        pass

    def start_monitoring(self):
        """Start performance monitoring."""
        self.should_run.set()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.should_run.clear()
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.should_run.is_set():
            try:
                # Collect and process metrics
                metrics = self.collect_metrics()
                self.metrics_history.extend(metrics)

                # Keep history limited
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]

                # Apply optimizations
                self.optimize_performance(metrics)

                # Sleep interval
                time.sleep(60)  # 1-minute interval

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(5)  # Short delay on error

    def get_performance_report(self) -> Dict:
        """Generate performance report."""
        if not self.metrics_history:
            return {}

        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics_summary": {},
            "active_optimizations": self.active_optimizations,
            "resource_status": {}
        }

        # Calculate metrics summary
        for resource_type in ResourceType:
            resource_metrics = [
                m for m in self.metrics_history
                if m.name == resource_type.value
            ]
            if resource_metrics:
                report["metrics_summary"][resource_type.value] = {
                    "current": resource_metrics[-1].value,
                    "average": sum(m.value for m in resource_metrics) / len(resource_metrics),
                    "peak": max(m.value for m in resource_metrics),
                    "critical_events": len([m for m in resource_metrics if m.is_critical])
                }

        return report

if __name__ == '__main__':
    # Test performance optimization
    project_root = Path(__file__).parent.parent
    optimizer = PerformanceOptimizer(project_root)

    # Start monitoring
    optimizer.start_monitoring()

    try:
        # Run for a test period
        time.sleep(300)  # 5 minutes

        # Get performance report
        report = optimizer.get_performance_report()
        print("\nPerformance Report:")
        print(json.dumps(report, indent=2))

    finally:
        optimizer.stop_monitoring()
