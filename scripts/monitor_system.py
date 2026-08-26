#!/usr/bin/env python
"""
System Monitor and Reporter for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides real-time monitoring and reporting capabilities:
1. System health monitoring
2. Performance metrics collection
3. Resource utilization tracking
4. Alert generation
5. Report generation
"""

import os
import json
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict
    process_count: int
    timestamp: str

class SystemMonitor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = self._setup_logger()
        self.config = self._load_config()
        self.metrics_history: List[SystemMetrics] = []

    def _setup_logger(self) -> logging.Logger:
        """Configure logging for monitoring."""
        logger = logging.getLogger("system_monitor")
        logger.setLevel(logging.INFO)

        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(
            log_dir / "monitoring.log"
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _load_config(self) -> Dict:
        """Load monitoring configuration."""
        config_path = self.project_root / "config" / "orchestrator.json"
        if not config_path.exists():
            return {}
        return json.loads(config_path.read_text())

    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        return SystemMetrics(
            cpu_usage=psutil.cpu_percent(interval=1),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            network_io=dict(psutil.net_io_counters()._asdict()),
            process_count=len(psutil.pids()),
            timestamp=datetime.now().isoformat()
        )

    def check_thresholds(self, metrics: SystemMetrics) -> List[str]:
        """Check metrics against defined thresholds."""
        alerts = []

        # CPU Usage Alert
        if metrics.cpu_usage > 90:
            alerts.append(f"High CPU usage: {metrics.cpu_usage}%")

        # Memory Usage Alert
        if metrics.memory_usage > 90:
            alerts.append(f"High memory usage: {metrics.memory_usage}%")

        # Disk Usage Alert
        if metrics.disk_usage > 90:
            alerts.append(f"High disk usage: {metrics.disk_usage}%")

        return alerts

    def generate_report(self) -> Dict:
        """Generate monitoring report."""
        if not self.metrics_history:
            return {}

        metrics = self.metrics_history[-1]
        alerts = self.check_thresholds(metrics)

        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "disk_usage": metrics.disk_usage,
                "network_io": metrics.network_io,
                "process_count": metrics.process_count
            },
            "alerts": alerts,
            "metrics_history": [
                {
                    "timestamp": m.timestamp,
                    "cpu_usage": m.cpu_usage,
                    "memory_usage": m.memory_usage
                }
                for m in self.metrics_history[-100:]  # Last 100 metrics
            ]
        }

    def save_report(self, report: Dict):
        """Save monitoring report to file."""
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"monitoring_report_{timestamp}.json"

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Saved monitoring report: {report_path}")

    def monitor(self, interval: int = 60):
        """Run continuous monitoring."""
        self.logger.info("Starting system monitoring")

        try:
            while True:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)

                # Keep history limited to last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]

                # Check for alerts
                alerts = self.check_thresholds(metrics)
                if alerts:
                    for alert in alerts:
                        self.logger.warning(f"Alert: {alert}")

                # Generate and save report every hour
                if len(self.metrics_history) % 60 == 0:
                    report = self.generate_report()
                    self.save_report(report)

                time.sleep(interval)

        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {str(e)}")
            raise

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    monitor = SystemMonitor(project_root)
    monitor.monitor()
