#!/usr/bin/env python
"""
Performance Monitoring System for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import psutil
import logging
import threading
import queue
import time
from typing import Dict, Any, Optional
from pathlib import Path


class PerformanceMonitor:
    """Real-time system performance monitoring."""

    def __init__(
        self,
        config: Dict[str, Any],
        metrics_queue: queue.Queue
    ):
        self.config = config
        self.metrics_queue = metrics_queue
        self.shutdown_flag = threading.Event()

        # Configure logging
        self.logger = logging.getLogger("PerformanceMonitor")

        # Initialize metrics history
        self.metrics_history = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "network": [],
            "gpu": []
        }

        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )

    def start(self):
        """Start performance monitoring."""
        self.logger.info("Starting performance monitoring")
        self.monitor_thread.start()

    def stop(self):
        """Stop performance monitoring."""
        self.logger.info("Stopping performance monitoring")
        self.shutdown_flag.set()
        self.monitor_thread.join(timeout=10)

    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.shutdown_flag.is_set():
            try:
                # Collect metrics
                metrics = {
                    "timestamp": time.time(),
                    "cpu": self._get_cpu_metrics(),
                    "memory": self._get_memory_metrics(),
                    "disk": self._get_disk_metrics(),
                    "network": self._get_network_metrics(),
                    "gpu": self._get_gpu_metrics()
                }

                # Update history
                self._update_history(metrics)

                # Send metrics to queue
                self.metrics_queue.put(metrics)

            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")

            finally:
                # Wait for next collection interval
                self.shutdown_flag.wait(
                    self.config.get("interval_seconds", 5)
                )

    def _get_cpu_metrics(self) -> Dict[str, float]:
        """Get CPU metrics."""
        try:
            return {
                "usage": psutil.cpu_percent(interval=1),
                "load_avg": psutil.getloadavg(),
                "cores_used": psutil.cpu_count(),
                "frequency": psutil.cpu_freq().current
            }
        except Exception as e:
            self.logger.error(f"Error getting CPU metrics: {e}")
            return {}

    def _get_memory_metrics(self) -> Dict[str, float]:
        """Get memory metrics."""
        try:
            mem = psutil.virtual_memory()
            return {
                "total": mem.total,
                "available": mem.available,
                "used": mem.used,
                "percent": mem.percent
            }
        except Exception as e:
            self.logger.error(f"Error getting memory metrics: {e}")
            return {}

    def _get_disk_metrics(self) -> Dict[str, float]:
        """Get disk metrics."""
        try:
            disk = psutil.disk_usage("/")
            return {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            }
        except Exception as e:
            self.logger.error(f"Error getting disk metrics: {e}")
            return {}

    def _get_network_metrics(self) -> Dict[str, float]:
        """Get network metrics."""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except Exception as e:
            self.logger.error(f"Error getting network metrics: {e}")
            return {}

    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics if available."""
        try:
            # Try to import GPU monitoring libraries
            import torch
            if not torch.cuda.is_available():
                return {}

            return {
                "gpu_count": torch.cuda.device_count(),
                "gpu_utilization": [
                    torch.cuda.get_device_properties(i).name
                    for i in range(torch.cuda.device_count())
                ]
            }
        except ImportError:
            return {}
        except Exception as e:
            self.logger.error(f"Error getting GPU metrics: {e}")
            return {}

    def _update_history(self, metrics: Dict[str, Any]):
        """Update metrics history."""
        try:
            max_history = self.config.get("history_size", 1000)

            for category in self.metrics_history:
                if category in metrics:
                    self.metrics_history[category].append(
                        metrics[category]
                    )

                    # Trim history if needed
                    if len(self.metrics_history[category]) > max_history:
                        self.metrics_history[category] = \
                            self.metrics_history[category][-max_history:]

        except Exception as e:
            self.logger.error(f"Error updating metrics history: {e}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            return {
                "cpu": self._get_cpu_metrics(),
                "memory": self._get_memory_metrics(),
                "disk": self._get_disk_metrics(),
                "network": self._get_network_metrics(),
                "gpu": self._get_gpu_metrics()
            }
        except Exception as e:
            self.logger.error(f"Error getting current metrics: {e}")
            return {}

    def get_metrics_history(
        self,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get metrics history."""
        try:
            if category:
                return {
                    category: self.metrics_history.get(category, [])
                }
            return self.metrics_history

        except Exception as e:
            self.logger.error(f"Error getting metrics history: {e}")
            return {}
