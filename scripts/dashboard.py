#!/usr/bin/env python
"""
System Dashboard for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides a real-time monitoring dashboard:
1. System metrics visualization
2. Performance analytics
3. Security status
4. Component health monitoring
"""

import json
import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import threading

from scripts.monitor_system import SystemMonitor
from scripts.validate_system import SystemValidator, ValidationLevel
from scripts.benchmark_system import PerformanceMonitor


class DashboardApp:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.monitor = SystemMonitor(project_root)
        self.validator = SystemValidator(project_root)
        self.performance_monitor = PerformanceMonitor(project_root)

        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Negative Space Imaging System Dashboard")
        self.root.geometry("1200x800")

        # Set up UI components
        self.setup_ui()

        # Start monitoring thread
        self.running = True
        self.update_thread = threading.Thread(target=self.update_metrics)
        self.update_thread.daemon = True
        self.update_thread.start()

    def setup_ui(self):
        """Set up the dashboard UI components."""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill='both', padx=10, pady=5)

        # System Overview Tab
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text='System Overview')

        # System metrics
        metrics_frame = ttk.LabelFrame(overview_frame, text='System Metrics')
        metrics_frame.pack(fill='x', padx=5, pady=5)

        self.cpu_label = ttk.Label(metrics_frame, text='CPU Usage: ---%')
        self.cpu_label.pack(anchor='w', padx=5, pady=2)

        self.memory_label = ttk.Label(metrics_frame, text='Memory Usage: ---%')
        self.memory_label.pack(anchor='w', padx=5, pady=2)

        self.disk_label = ttk.Label(metrics_frame, text='Disk Usage: ---%')
        self.disk_label.pack(anchor='w', padx=5, pady=2)

        # System Status
        status_frame = ttk.LabelFrame(overview_frame, text='System Status')
        status_frame.pack(fill='x', padx=5, pady=5)

        self.status_label = ttk.Label(status_frame, text='System Status: Unknown')
        self.status_label.pack(anchor='w', padx=5, pady=2)

        # Performance Tab
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text='Performance')

        # Performance metrics
        perf_metrics = ttk.LabelFrame(perf_frame, text='Performance Metrics')
        perf_metrics.pack(fill='x', padx=5, pady=5)

        self.throughput_label = ttk.Label(perf_metrics, text='Throughput: --- ops/s')
        self.throughput_label.pack(anchor='w', padx=5, pady=2)

        self.latency_label = ttk.Label(perf_metrics, text='Latency: --- ms')
        self.latency_label.pack(anchor='w', padx=5, pady=2)

        # Security Tab
        security_frame = ttk.Frame(notebook)
        notebook.add(security_frame, text='Security')

        # Security status
        security_status = ttk.LabelFrame(security_frame, text='Security Status')
        security_status.pack(fill='x', padx=5, pady=5)

        self.security_label = ttk.Label(security_status, text='Security Status: Unknown')
        self.security_label.pack(anchor='w', padx=5, pady=2)

        # Control buttons
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(control_frame, text='Run Validation',
                   command=self.run_validation).pack(side='left', padx=5)
        ttk.Button(control_frame, text='Run Benchmark',
                   command=self.run_benchmark).pack(side='left', padx=5)
        ttk.Button(control_frame, text='Generate Report',
                   command=self.generate_report).pack(side='left', padx=5)
        ttk.Button(control_frame, text='Exit',
                   command=self.shutdown).pack(side='right', padx=5)

    def update_metrics(self):
        """Update system metrics periodically."""
        while self.running:
            try:
                # Get system metrics
                metrics = self.monitor.collect_metrics()

                # Update UI
                self.cpu_label.config(text=f'CPU Usage: {metrics.cpu_usage:.1f}%')
                self.memory_label.config(text=f'Memory Usage: {metrics.memory_usage:.1f}%')
                self.disk_label.config(text=f'Disk Usage: {metrics.disk_usage:.1f}%')

                # Check thresholds and update status
                alerts = self.monitor.check_thresholds(metrics)
                if alerts:
                    status = "Warning: " + ", ".join(alerts)
                    self.status_label.config(text=f"System Status: {status}")
                else:
                    self.status_label.config(text="System Status: Healthy")

                time.sleep(1)  # Update every second

            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}")
                time.sleep(5)  # Wait before retry

    def run_validation(self):
        """Run system validation."""
        try:
            results = self.validator.run_validation(ValidationLevel.COMPLETE)
            success = all(r['success'] for r in results['results'])
            self.security_label.config(
                text=f"Security Status: {'Validated' if success else 'Failed'}"
            )
        except Exception as e:
            self.security_label.config(text=f"Validation Error: {str(e)}")

    def run_benchmark(self):
        """Run performance benchmark."""
        try:
            results = self.performance_monitor.run_benchmarks()
            self.throughput_label.config(
                text=f"Throughput: {results['throughput']:.1f} ops/s"
            )
            self.latency_label.config(
                text=f"Latency: {results['latency']:.1f} ms"
            )
        except Exception as e:
            self.throughput_label.config(text=f"Benchmark Error: {str(e)}")

    def generate_report(self):
        """Generate and save system report."""
        try:
            report = self.monitor.generate_report()
            self.monitor.save_report(report)
            self.status_label.config(text="Report generated successfully")
        except Exception as e:
            self.status_label.config(text=f"Report Error: {str(e)}")

    def shutdown(self):
        """Shutdown the dashboard gracefully."""
        self.running = False
        if self.update_thread.is_alive():
            self.update_thread.join()
        self.root.quit()

    def run(self):
        """Start the dashboard."""
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        self.root.mainloop()


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    dashboard = DashboardApp(project_root)
    dashboard.run()
