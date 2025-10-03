#!/usr/bin/env python
"""
Test Suite for System Management Components
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.monitor_system import SystemMonitor, SystemMetrics
from scripts.orchestrate_system import SystemOrchestrator, SystemState


class TestSystemMonitor(unittest.TestCase):
    """Test cases for SystemMonitor."""

    def setUp(self):
        self.project_root = Path(__file__).parent
        self.monitor = SystemMonitor(self.project_root)

    def test_collect_metrics(self):
        """Test metrics collection."""
        metrics = self.monitor.collect_metrics()
        self.assertIsInstance(metrics, SystemMetrics)
        self.assertGreaterEqual(metrics.cpu_usage, 0)
        self.assertLessEqual(metrics.cpu_usage, 100)
        self.assertGreaterEqual(metrics.memory_usage, 0)
        self.assertLessEqual(metrics.memory_usage, 100)

    def test_check_thresholds(self):
        """Test threshold checking."""
        metrics = SystemMetrics(
            cpu_usage=95,
            memory_usage=85,
            disk_usage=75,
            network_io={"bytes_sent": 1000, "bytes_recv": 1000},
            process_count=100,
            timestamp="2025-08-09T12:00:00"
        )
        alerts = self.monitor.check_thresholds(metrics)
        self.assertTrue(any("CPU" in alert for alert in alerts))

    def test_generate_report(self):
        """Test report generation."""
        metrics = SystemMetrics(
            cpu_usage=50,
            memory_usage=60,
            disk_usage=70,
            network_io={"bytes_sent": 1000, "bytes_recv": 1000},
            process_count=100,
            timestamp="2025-08-09T12:00:00"
        )
        self.monitor.metrics_history.append(metrics)
        report = self.monitor.generate_report()
        self.assertIn("current_metrics", report)
        self.assertIn("alerts", report)
        self.assertIn("metrics_history", report)


class TestSystemOrchestrator(unittest.TestCase):
    """Test cases for SystemOrchestrator."""

    def setUp(self):
        self.project_root = Path(__file__).parent
        self.orchestrator = SystemOrchestrator(self.project_root)

    @patch('scripts.validate_system.SystemValidator.run_validation')
    def test_initialize(self, mock_validation):
        """Test system initialization."""
        mock_validation.return_value = {
            "results": [{"success": True}]
        }
        result = self.orchestrator.initialize()
        self.assertTrue(result)
        self.assertEqual(self.orchestrator.state, SystemState.RUNNING)

    def test_shutdown(self):
        """Test system shutdown."""
        self.orchestrator.initialize()
        self.orchestrator.shutdown()
        self.assertEqual(self.orchestrator.state, SystemState.SHUTTING_DOWN)
        self.assertFalse(self.orchestrator.should_run.is_set())

    @patch('scripts.monitor_system.SystemMonitor.collect_metrics')
    def test_system_monitoring(self, mock_collect):
        """Test system monitoring."""
        mock_collect.return_value = SystemMetrics(
            cpu_usage=50,
            memory_usage=60,
            disk_usage=70,
            network_io={"bytes_sent": 1000, "bytes_recv": 1000},
            process_count=100,
            timestamp="2025-08-09T12:00:00"
        )
        self.orchestrator.initialize()
        self.assertTrue(self.orchestrator.system_monitor_thread.is_alive())
        self.orchestrator.shutdown()


if __name__ == '__main__':
    unittest.main()
