#!/usr/bin/env python
"""
Launch Script for Negative Space Imaging System
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This script launches all system components:
1. System Orchestrator
2. Monitoring Dashboard
3. Performance Monitoring
4. Security Services
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from typing import List, Dict
import threading

def start_component(cmd: List[str], name: str) -> subprocess.Popen:
    """Start a system component."""
    print(f"Starting {name}...")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

def monitor_process(process: subprocess.Popen, name: str):
    """Monitor a process and log its output."""
    while True:
        output = process.stdout.readline()
        if output:
            print(f"{name}: {output.strip()}")
        if process.poll() is not None:
            break

def launch_system():
    """Launch all system components."""
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"
    processes = []

    try:
        # Start System Orchestrator
        orchestrator = start_component(
            ["python", str(scripts_dir / "orchestrate_system.py")],
            "Orchestrator"
        )
        processes.append((orchestrator, "Orchestrator"))

        # Start Dashboard
        dashboard = start_component(
            ["python", str(scripts_dir / "dashboard.py")],
            "Dashboard"
        )
        processes.append((dashboard, "Dashboard"))

        # Start monitoring threads
        threads = []
        for proc, name in processes:
            thread = threading.Thread(
                target=monitor_process,
                args=(proc, name)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)

        # Wait for processes
        while all(proc.poll() is None for proc, _ in processes):
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
        for proc, name in processes:
            if proc.poll() is None:
                print(f"Stopping {name}...")
                proc.terminate()
                proc.wait()

    finally:
        # Ensure all processes are terminated
        for proc, name in processes:
            if proc.poll() is None:
                print(f"Force stopping {name}...")
                proc.kill()

def check_environment():
    """Verify the Python environment."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

    try:
        import tkinter
        import psutil
        import cryptography
    except ImportError as e:
        print(f"Error: Missing required package - {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    print("Negative Space Imaging System")
    print("============================")

    check_environment()
    launch_system()
