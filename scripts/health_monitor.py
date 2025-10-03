#!/usr/bin/env python
"""
System Health Monitoring Interface for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import json
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime
from typing import Dict
import threading
import queue


class SystemHealthGUI:
    """Real-time system health monitoring interface."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_queue = queue.Queue()

        # Initialize main window
        self.root = tk.Tk()
        self.root.title("System Health Monitor")
        self.root.geometry("1000x800")

        # Set up UI components
        self.setup_ui()

        # Start update thread
        self.update_thread = threading.Thread(target=self.update_data)
        self.update_thread.daemon = True
        self.update_thread.start()

    def setup_ui(self):
        """Set up the GUI components."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=10, pady=5)

        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="System Status")
        status_frame.pack(fill='x', padx=5, pady=5)

        self.status_label = ttk.Label(
            status_frame,
            text="Status: Initializing...",
            font=("Arial", 12, "bold")
        )
        self.status_label.pack(pady=5)

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(expand=True, fill='both')

        # Performance tab
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="Performance")

        # Performance metrics
        self.cpu_bar = self.create_metric_bar(
            perf_frame,
            "CPU Usage",
            0, 100
        )
        self.memory_bar = self.create_metric_bar(
            perf_frame,
            "Memory Usage",
            0, 100
        )
        self.disk_bar = self.create_metric_bar(
            perf_frame,
            "Disk Usage",
            0, 100
        )

        # Security tab
        security_frame = ttk.Frame(notebook)
        notebook.add(security_frame, text="Security")

        self.security_text = tk.Text(
            security_frame,
            height=10,
            wrap=tk.WORD
        )
        self.security_text.pack(expand=True, fill='both', padx=5, pady=5)

        # Error log tab
        error_frame = ttk.Frame(notebook)
        notebook.add(error_frame, text="Error Log")

        self.error_text = tk.Text(
            error_frame,
            height=10,
            wrap=tk.WORD
        )
        self.error_text.pack(expand=True, fill='both', padx=5, pady=5)

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(
            control_frame,
            text="Run Diagnostics",
            command=self.run_diagnostics
        ).pack(side='left', padx=5)

        ttk.Button(
            control_frame,
            text="Generate Report",
            command=self.generate_report
        ).pack(side='left', padx=5)

        ttk.Button(
            control_frame,
            text="Clear Log",
            command=self.clear_log
        ).pack(side='left', padx=5)

    def create_metric_bar(
        self,
        parent: ttk.Frame,
        label: str,
        min_val: int,
        max_val: int
    ) -> Dict:
        """Create a metric progress bar with label."""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=5, pady=5)

        label = ttk.Label(frame, text=f"{label}: 0%")
        label.pack(side='left', padx=5)

        bar = ttk.Progressbar(
            frame,
            orient='horizontal',
            length=200,
            mode='determinate',
            maximum=max_val,
            value=min_val
        )
        bar.pack(side='left', padx=5)

        return {"label": label, "bar": bar}

    def update_data(self):
        """Update data from the monitoring system."""
        while True:
            try:
                # Read system state
                state_file = self.project_root / "system_state.json"
                if state_file.exists():
                    state_data = json.loads(state_file.read_text())
                    self.data_queue.put(("state", state_data))

                # Read performance data
                perf_file = self.project_root / "reports" / "performance.json"
                if perf_file.exists():
                    perf_data = json.loads(perf_file.read_text())
                    self.data_queue.put(("performance", perf_data))

                # Read error log
                error_file = self.project_root / "logs" / "error_recovery.log"
                if error_file.exists():
                    errors = error_file.read_text().splitlines()[-100:]
                    self.data_queue.put(("errors", errors))

                time.sleep(1)  # Update every second

            except Exception as e:
                print(f"Update error: {e}")
                time.sleep(5)

    def update_ui(self):
        """Update UI with new data."""
        try:
            while True:
                try:
                    data_type, data = self.data_queue.get_nowait()

                    if data_type == "state":
                        self.status_label.config(
                            text=f"Status: {data['state']}"
                        )

                    elif data_type == "performance":
                        if "metrics_summary" in data:
                            summary = data["metrics_summary"]

                            if "cpu" in summary:
                                self.update_metric_bar(
                                    self.cpu_bar,
                                    "CPU Usage",
                                    summary["cpu"]["current"]
                                )

                            if "memory" in summary:
                                self.update_metric_bar(
                                    self.memory_bar,
                                    "Memory Usage",
                                    summary["memory"]["current"]
                                )

                            if "disk" in summary:
                                self.update_metric_bar(
                                    self.disk_bar,
                                    "Disk Usage",
                                    summary["disk"]["current"]
                                )

                    elif data_type == "errors":
                        self.error_text.delete(1.0, tk.END)
                        self.error_text.insert(tk.END, "\n".join(data))

                except queue.Empty:
                    break

        except Exception as e:
            messagebox.showerror("Error", f"UI update error: {e}")

        finally:
            # Schedule next update
            self.root.after(100, self.update_ui)

    def update_metric_bar(
        self,
        metric: Dict,
        label: str,
        value: float
    ):
        """Update a metric progress bar."""
        metric["bar"]["value"] = value
        metric["label"].config(text=f"{label}: {value:.1f}%")

        # Update color based on value
        if value >= 90:
            metric["bar"].configure(style="Critical.Horizontal.TProgressbar")
        elif value >= 75:
            metric["bar"].configure(style="Warning.Horizontal.TProgressbar")
        else:
            metric["bar"].configure(style="Normal.Horizontal.TProgressbar")

    def run_diagnostics(self):
        """Run system diagnostics."""
        try:
            # Add diagnostic logic here
            messagebox.showinfo(
                "Diagnostics",
                "Running system diagnostics..."
            )

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to run diagnostics: {e}"
            )

    def generate_report(self):
        """Generate system health report."""
        try:
            report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = (
                self.project_root / "reports" /
                f"health_report_{report_time}.txt"
            )

            with open(report_file, 'w') as f:
                f.write("System Health Report\n")
                f.write("===================\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")

                # Add report content

            messagebox.showinfo(
                "Report Generated",
                f"Report saved to {report_file}"
            )

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to generate report: {e}"
            )

    def clear_log(self):
        """Clear the error log display."""
        self.error_text.delete(1.0, tk.END)

    def run(self):
        """Start the monitoring interface."""
        # Configure progress bar styles
        style = ttk.Style()
        style.configure(
            "Critical.Horizontal.TProgressbar",
            troughcolor='gray',
            background='red'
        )
        style.configure(
            "Warning.Horizontal.TProgressbar",
            troughcolor='gray',
            background='orange'
        )
        style.configure(
            "Normal.Horizontal.TProgressbar",
            troughcolor='gray',
            background='green'
        )

        # Start UI updates
        self.update_ui()

        # Start main loop
        self.root.mainloop()

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    monitor = SystemHealthGUI(project_root)
    monitor.run()
