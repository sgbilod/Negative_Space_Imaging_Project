#!/usr/bin/env python3
"""
Performance Monitoring CLI for Negative Space Imaging Project
This script provides a command-line interface for interacting with the performance monitoring system.
"""

import argparse
import os
import sys
import json
import datetime
import subprocess
import yaml
import requests
import time
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.panel import Panel

console = Console()

def load_config():
    """Load the performance monitoring configuration."""
    config_path = os.path.join(os.path.dirname(__file__), "performance_monitoring_config.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        console.print(f"[bold red]Error loading configuration: {str(e)}[/bold red]")
        sys.exit(1)

def get_monitoring_status():
    """Check if the monitoring system is running."""
    try:
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.performance.yml", "ps", "-q", "monitoring"],
            capture_output=True,
            text=True,
            check=False
        )
        return bool(result.stdout.strip())
    except Exception:
        return False

def start_monitoring():
    """Start the monitoring system."""
    console.print("[bold blue]Starting performance monitoring system...[/bold blue]")
    try:
        subprocess.run(
            ["docker-compose", "-f", "docker-compose.performance.yml", "up", "-d", "monitoring"],
            check=True
        )
        console.print("[bold green]Monitoring system started successfully![/bold green]")
        console.print("Dashboards available at:")
        console.print("  - Grafana: [link]http://localhost:3001[/link]")
        console.print("  - Prometheus: [link]http://localhost:9090[/link]")
        return True
    except subprocess.CalledProcessError:
        console.print("[bold red]Failed to start monitoring system.[/bold red]")
        return False

def stop_monitoring():
    """Stop the monitoring system."""
    console.print("[bold yellow]Stopping performance monitoring system...[/bold yellow]")
    try:
        subprocess.run(
            ["docker-compose", "-f", "docker-compose.performance.yml", "stop", "monitoring"],
            check=True
        )
        console.print("[bold green]Monitoring system stopped successfully![/bold green]")
        return True
    except subprocess.CalledProcessError:
        console.print("[bold red]Failed to stop monitoring system.[/bold red]")
        return False

def restart_monitoring():
    """Restart the monitoring system."""
    stop_monitoring()
    time.sleep(2)
    return start_monitoring()

def get_metrics():
    """Get current performance metrics."""
    if not get_monitoring_status():
        console.print("[bold red]Monitoring system is not running.[/bold red]")
        return None

    # For demonstration purposes, we'll simulate fetching metrics
    # In a real implementation, you would query your metrics API or Prometheus

    metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "system": {
            "cpu_usage": 35.2,
            "memory_usage": 42.8,
            "disk_usage": 68.5,
            "network_rx": 1.2,
            "network_tx": 0.8
        },
        "application": {
            "requests_per_second": 12.5,
            "average_response_time": 120,
            "error_rate": 0.2,
            "active_users": 25
        },
        "database": {
            "queries_per_second": 45.6,
            "average_query_time": 15.8,
            "connection_count": 10,
            "cache_hit_ratio": 0.85
        },
        "hpc": {
            "active_jobs": 3,
            "gpu_utilization": 78.5,
            "node_count": 5,
            "completed_jobs_today": 12
        }
    }

    return metrics

def display_metrics(metrics):
    """Display metrics in a formatted table."""
    if not metrics:
        return

    console.print(Panel(f"[bold]Performance Metrics as of {metrics['timestamp']}[/bold]"))

    # System Metrics
    table = Table(title="System Resources")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")

    cpu_status = "✅" if metrics["system"]["cpu_usage"] < 70 else "⚠️"
    if metrics["system"]["cpu_usage"] > 85:
        cpu_status = "❌"

    memory_status = "✅" if metrics["system"]["memory_usage"] < 80 else "⚠️"
    if metrics["system"]["memory_usage"] > 90:
        memory_status = "❌"

    disk_status = "✅" if metrics["system"]["disk_usage"] < 80 else "⚠️"
    if metrics["system"]["disk_usage"] > 90:
        disk_status = "❌"

    table.add_row("CPU Usage", f"{metrics['system']['cpu_usage']}%", cpu_status)
    table.add_row("Memory Usage", f"{metrics['system']['memory_usage']}%", memory_status)
    table.add_row("Disk Usage", f"{metrics['system']['disk_usage']}%", disk_status)
    table.add_row("Network RX", f"{metrics['system']['network_rx']} MB/s", "✅")
    table.add_row("Network TX", f"{metrics['system']['network_tx']} MB/s", "✅")

    console.print(table)

    # Application Metrics
    table = Table(title="Application Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Requests/Second", f"{metrics['application']['requests_per_second']}")
    table.add_row("Avg Response Time", f"{metrics['application']['average_response_time']} ms")
    table.add_row("Error Rate", f"{metrics['application']['error_rate']}%")
    table.add_row("Active Users", f"{metrics['application']['active_users']}")

    console.print(table)

    # Database Metrics
    table = Table(title="Database Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Queries/Second", f"{metrics['database']['queries_per_second']}")
    table.add_row("Avg Query Time", f"{metrics['database']['average_query_time']} ms")
    table.add_row("Connections", f"{metrics['database']['connection_count']}")
    table.add_row("Cache Hit Ratio", f"{metrics['database']['cache_hit_ratio'] * 100}%")

    console.print(table)

    # HPC Metrics
    table = Table(title="HPC Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Active Jobs", f"{metrics['hpc']['active_jobs']}")
    table.add_row("GPU Utilization", f"{metrics['hpc']['gpu_utilization']}%")
    table.add_row("Node Count", f"{metrics['hpc']['node_count']}")
    table.add_row("Completed Jobs Today", f"{metrics['hpc']['completed_jobs_today']}")

    console.print(table)

def export_metrics(metrics, format_type="json", output_file=None):
    """Export metrics to a file in the specified format."""
    if not metrics:
        return False

    if not output_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"metrics_export_{timestamp}.{format_type}"

    try:
        if format_type == "json":
            with open(output_file, "w") as f:
                json.dump(metrics, f, indent=2)
        elif format_type == "csv":
            # Simplified CSV export for demonstration
            with open(output_file, "w") as f:
                # Write system metrics
                f.write("Metric,Value\n")
                for key, value in metrics["system"].items():
                    f.write(f"{key},{value}\n")

                # Write application metrics
                for key, value in metrics["application"].items():
                    f.write(f"{key},{value}\n")

                # Write database metrics
                for key, value in metrics["database"].items():
                    f.write(f"{key},{value}\n")

                # Write HPC metrics
                for key, value in metrics["hpc"].items():
                    f.write(f"{key},{value}\n")
        else:
            console.print(f"[bold red]Unsupported export format: {format_type}[/bold red]")
            return False

        console.print(f"[bold green]Metrics exported to {output_file}[/bold green]")
        return True
    except Exception as e:
        console.print(f"[bold red]Error exporting metrics: {str(e)}[/bold red]")
        return False

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Performance Monitoring CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check monitoring system status")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start monitoring system")

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop monitoring system")

    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart monitoring system")

    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Display current metrics")
    metrics_parser.add_argument("--export", choices=["json", "csv"], help="Export metrics to file")
    metrics_parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    if args.command == "status":
        is_running = get_monitoring_status()
        if is_running:
            console.print("[bold green]Monitoring system is running.[/bold green]")
        else:
            console.print("[bold yellow]Monitoring system is not running.[/bold yellow]")

    elif args.command == "start":
        start_monitoring()

    elif args.command == "stop":
        stop_monitoring()

    elif args.command == "restart":
        restart_monitoring()

    elif args.command == "metrics":
        metrics = get_metrics()
        display_metrics(metrics)

        if args.export:
            export_metrics(metrics, args.export, args.output)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
