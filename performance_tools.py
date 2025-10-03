#!/usr/bin/env python3
"""
Performance Optimization CLI Tool for Negative Space Imaging Project

This script provides a unified command-line interface for all performance
optimization tools, including benchmarking, profiling, and visualization.

Author: Stephen Bilodeau
Copyright: © 2025 Negative Space Imaging, Inc.
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("performance_tools.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("performance_tools")

# Set up console output
console = Console()


class PerformanceTools:
    """CLI for the performance optimization system."""

    def __init__(self):
        """Initialize the CLI."""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.base_dir, "performance_results")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run_benchmarks(self, args):
        """
        Run performance benchmarks.

        Args:
            args: Command-line arguments for benchmarks
        """
        console.print(Panel.fit(
            "[bold blue]Running Performance Benchmarks[/bold blue]",
            box=box.DOUBLE
        ))

        # Build benchmark command
        benchmark_script = os.path.join(self.base_dir, "optimization_benchmark.py")
        benchmark_cmd = [sys.executable, benchmark_script]

        # Add benchmark selection arguments
        if args.all:
            benchmark_cmd.append("--all")
        else:
            if args.memory:
                benchmark_cmd.append("--memory")
            if args.cpu:
                benchmark_cmd.append("--cpu")
            if args.io:
                benchmark_cmd.append("--io")
            if args.database:
                benchmark_cmd.append("--database")

        # Add output file
        output_file = os.path.join(
            self.output_dir, f"benchmark_results_{self.timestamp}.json"
        )
        benchmark_cmd.extend(["--output", output_file])

        # Add plot option if specified
        if args.plot:
            benchmark_cmd.append("--plot")

        # Execute benchmark command
        console.print(f"Executing: [cyan]{' '.join(benchmark_cmd)}[/cyan]")

        try:
            subprocess.run(benchmark_cmd, check=True)
            console.print(f"[bold green]Benchmarks completed successfully[/bold green]")
            console.print(f"Results saved to: [cyan]{output_file}[/cyan]")

            # Run visualization if requested
            if args.visualize:
                self.run_visualization(
                    argparse.Namespace(
                        benchmark=output_file,
                        profiler=None,
                        output=os.path.join(self.output_dir, "visualizations"),
                        prefix=f"benchmark_{self.timestamp}",
                        report=True
                    )
                )

            return output_file

        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error running benchmarks: {str(e)}[/bold red]")
            return None

    def run_profiler(self, args):
        """
        Run performance profiler.

        Args:
            args: Command-line arguments for profiler
        """
        console.print(Panel.fit(
            "[bold blue]Running Performance Profiler[/bold blue]",
            box=box.DOUBLE
        ))

        # Build profiler command
        profiler_script = os.path.join(self.base_dir, "performance_profiler.py")
        profiler_cmd = [sys.executable, profiler_script]

        # Add profiler arguments
        if args.module:
            profiler_cmd.extend(["--module", args.module])

        if args.script:
            profiler_cmd.extend(["--script", args.script])
            if args.script_args:
                profiler_cmd.extend(["--args"] + args.script_args)

        # Add output directory
        output_dir = os.path.join(self.output_dir, f"profiler_{self.timestamp}")
        profiler_cmd.extend(["--output", output_dir])

        # Add detailed flag if specified
        if args.detailed:
            profiler_cmd.append("--detailed")

        # Add report path
        report_path = os.path.join(output_dir, "profiler_report.html")
        profiler_cmd.extend(["--report", report_path])

        # Execute profiler command
        console.print(f"Executing: [cyan]{' '.join(profiler_cmd)}[/cyan]")

        try:
            subprocess.run(profiler_cmd, check=True)
            console.print(f"[bold green]Profiling completed successfully[/bold green]")
            console.print(f"Report saved to: [cyan]{report_path}[/cyan]")

            # Find JSON results for visualization
            json_files = []
            for file in os.listdir(output_dir):
                if file.endswith(".json"):
                    json_files.append(os.path.join(output_dir, file))

            profiler_results = json_files[0] if json_files else None

            # Run visualization if requested
            if args.visualize and profiler_results:
                self.run_visualization(
                    argparse.Namespace(
                        benchmark=None,
                        profiler=profiler_results,
                        output=os.path.join(self.output_dir, "visualizations"),
                        prefix=f"profiler_{self.timestamp}",
                        report=True
                    )
                )

            return profiler_results

        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error running profiler: {str(e)}[/bold red]")
            return None

    def run_visualization(self, args):
        """
        Run performance visualization.

        Args:
            args: Command-line arguments for visualization
        """
        console.print(Panel.fit(
            "[bold blue]Running Performance Visualization[/bold blue]",
            box=box.DOUBLE
        ))

        # Check if we have input files
        if not args.benchmark and not args.profiler:
            console.print("[bold yellow]No input files specified for visualization[/bold yellow]")
            return None

        # Build visualization command
        viz_script = os.path.join(self.base_dir, "performance_visualizer.py")
        viz_cmd = [sys.executable, viz_script]

        # Add input files
        if args.benchmark:
            viz_cmd.extend(["--benchmark", args.benchmark])

        if args.profiler:
            viz_cmd.extend(["--profiler", args.profiler])

        # Add output directory
        output_dir = args.output or os.path.join(
            self.output_dir, f"visualizations_{self.timestamp}"
        )
        viz_cmd.extend(["--output", output_dir])

        # Add prefix if specified
        if args.prefix:
            viz_cmd.extend(["--prefix", args.prefix])

        # Add report flag if specified
        if args.report:
            viz_cmd.append("--report")

        # Execute visualization command
        console.print(f"Executing: [cyan]{' '.join(viz_cmd)}[/cyan]")

        try:
            subprocess.run(viz_cmd, check=True)
            console.print(f"[bold green]Visualization completed successfully[/bold green]")
            console.print(f"Visualizations saved to: [cyan]{output_dir}[/cyan]")

            report_path = os.path.join(output_dir, "visualization_report.html")
            if os.path.exists(report_path):
                console.print(f"Report saved to: [cyan]{report_path}[/cyan]")

            return output_dir

        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error running visualization: {str(e)}[/bold red]")
            return None

    def show_config(self, args):
        """
        Show current performance optimization configuration.

        Args:
            args: Command-line arguments
        """
        console.print(Panel.fit(
            "[bold blue]Performance Optimization Configuration[/bold blue]",
            box=box.DOUBLE
        ))

        config_path = os.path.join(self.base_dir, "optimization_config.json")

        if not os.path.exists(config_path):
            console.print("[bold yellow]Configuration file not found[/bold yellow]")
            return

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Display configuration as a table
            table = Table(title="Optimization Configuration")
            table.add_column("Category", style="cyan")
            table.add_column("Setting", style="green")
            table.add_column("Value", style="yellow")

            for category, settings in config.items():
                if isinstance(settings, dict):
                    for setting, value in settings.items():
                        table.add_row(
                            category,
                            setting,
                            str(value)
                        )
                else:
                    table.add_row(
                        category,
                        "",
                        str(settings)
                    )

            console.print(table)

        except Exception as e:
            console.print(f"[bold red]Error reading configuration: {str(e)}[/bold red]")

    def update_config(self, args):
        """
        Update performance optimization configuration.

        Args:
            args: Command-line arguments containing configuration updates
        """
        console.print(Panel.fit(
            "[bold blue]Updating Performance Optimization Configuration[/bold blue]",
            box=box.DOUBLE
        ))

        config_path = os.path.join(self.base_dir, "optimization_config.json")

        try:
            # Load existing configuration
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}

            # Update configuration with provided values
            updated = False

            # Process simple enables/disables
            if args.enable_memory is not None:
                if "memory_optimizer" not in config:
                    config["memory_optimizer"] = {}
                config["memory_optimizer"]["enabled"] = args.enable_memory
                updated = True

            if args.enable_cpu is not None:
                if "cpu_optimizer" not in config:
                    config["cpu_optimizer"] = {}
                config["cpu_optimizer"]["enabled"] = args.enable_cpu
                updated = True

            if args.enable_io is not None:
                if "io_optimizer" not in config:
                    config["io_optimizer"] = {}
                config["io_optimizer"]["enabled"] = args.enable_io
                updated = True

            if args.enable_network is not None:
                if "network_optimizer" not in config:
                    config["network_optimizer"] = {}
                config["network_optimizer"]["enabled"] = args.enable_network
                updated = True

            if args.enable_database is not None:
                if "database_optimizer" not in config:
                    config["database_optimizer"] = {}
                config["database_optimizer"]["enabled"] = args.enable_database
                updated = True

            if args.enable_distributed is not None:
                if "distributed_optimizer" not in config:
                    config["distributed_optimizer"] = {}
                config["distributed_optimizer"]["enabled"] = args.enable_distributed
                updated = True

            # Process optimization levels
            if args.optimization_level:
                config["optimization_level"] = args.optimization_level
                updated = True

            # Process specific settings
            if args.settings:
                for setting in args.settings:
                    if "=" in setting:
                        key, value = setting.split("=", 1)

                        # Parse value as appropriate type
                        if value.lower() == "true":
                            parsed_value = True
                        elif value.lower() == "false":
                            parsed_value = False
                        elif value.isdigit():
                            parsed_value = int(value)
                        elif value.replace(".", "", 1).isdigit():
                            parsed_value = float(value)
                        else:
                            parsed_value = value

                        # Handle nested keys (e.g., "memory_optimizer.cache_size")
                        if "." in key:
                            parts = key.split(".")
                            category = parts[0]
                            setting_name = parts[1]

                            if category not in config:
                                config[category] = {}

                            config[category][setting_name] = parsed_value
                        else:
                            config[key] = parsed_value

                        updated = True

            # Save updated configuration
            if updated:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

                console.print(f"[bold green]Configuration updated successfully[/bold green]")
                self.show_config(args)
            else:
                console.print("[bold yellow]No configuration changes specified[/bold yellow]")

        except Exception as e:
            console.print(f"[bold red]Error updating configuration: {str(e)}[/bold red]")

    def show_help(self, args):
        """
        Show help information about the performance optimization system.

        Args:
            args: Command-line arguments
        """
        console.print(Panel.fit(
            "[bold blue]Performance Optimization System Help[/bold blue]",
            box=box.DOUBLE
        ))

        help_text = """
The Performance Optimization System is a comprehensive suite of tools designed to improve
the computational efficiency, resource utilization, and throughput of the Negative Space
Imaging Project.

[bold cyan]Main Components:[/bold cyan]

1. [bold]Performance Optimizer[/bold] (performance_optimizer.py)
   - Core optimization engine with specialized optimizers
   - Memory, CPU, I/O, Network, Database, and Distributed optimizations
   - Configurable through optimization_config.json

2. [bold]Benchmark Tool[/bold] (optimization_benchmark.py)
   - Measures and compares performance impact of optimizations
   - Supports various benchmark categories and data sizes
   - Generates detailed metrics and visualizations

3. [bold]Performance Profiler[/bold] (performance_profiler.py)
   - Identifies performance bottlenecks in existing code
   - Provides recommendations for optimization
   - Generates HTML reports with profiling insights

4. [bold]Visualization Tool[/bold] (performance_visualizer.py)
   - Creates data visualizations of benchmark and profiler results
   - Supports various chart types for performance analysis
   - Generates comprehensive HTML reports

[bold cyan]Common Usage Examples:[/bold cyan]

1. Run all benchmarks and visualize results:
   python performance_tools.py benchmark --all --visualize

2. Profile a specific module:
   python performance_tools.py profile --module my_module

3. Update optimization configuration:
   python performance_tools.py config --enable-memory true --optimization-level aggressive

4. Generate visualizations from existing results:
   python performance_tools.py visualize --benchmark results.json --profiler profile.json

For more detailed information, refer to PERFORMANCE_OPTIMIZATION.md
"""
        console.print(help_text)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Performance Optimization Tools for Negative Space Imaging Project"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to execute"
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run performance benchmarks"
    )
    benchmark_parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks"
    )
    benchmark_parser.add_argument(
        "--memory",
        action="store_true",
        help="Run memory optimization benchmarks"
    )
    benchmark_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run CPU optimization benchmarks"
    )
    benchmark_parser.add_argument(
        "--io",
        action="store_true",
        help="Run I/O optimization benchmarks"
    )
    benchmark_parser.add_argument(
        "--database",
        action="store_true",
        help="Run database optimization benchmarks"
    )
    benchmark_parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots of benchmark results"
    )
    benchmark_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Run visualization on benchmark results"
    )

    # Profile command
    profile_parser = subparsers.add_parser(
        "profile",
        help="Run performance profiler"
    )
    profile_parser.add_argument(
        "--module",
        help="Module to profile"
    )
    profile_parser.add_argument(
        "--script",
        help="Script to profile"
    )
    profile_parser.add_argument(
        "--script-args",
        nargs="+",
        help="Arguments to pass to the script"
    )
    profile_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed profiling information"
    )
    profile_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Run visualization on profiler results"
    )

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Generate performance visualizations"
    )
    visualize_parser.add_argument(
        "--benchmark",
        help="Path to benchmark results JSON file"
    )
    visualize_parser.add_argument(
        "--profiler",
        help="Path to profiler results JSON file"
    )
    visualize_parser.add_argument(
        "--output",
        help="Output directory for visualizations"
    )
    visualize_parser.add_argument(
        "--prefix",
        help="Prefix for saved visualization files"
    )
    visualize_parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report"
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="View or update performance optimization configuration"
    )
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration"
    )
    config_parser.add_argument(
        "--enable-memory",
        type=lambda x: x.lower() == "true",
        help="Enable/disable memory optimizer (true/false)"
    )
    config_parser.add_argument(
        "--enable-cpu",
        type=lambda x: x.lower() == "true",
        help="Enable/disable CPU optimizer (true/false)"
    )
    config_parser.add_argument(
        "--enable-io",
        type=lambda x: x.lower() == "true",
        help="Enable/disable I/O optimizer (true/false)"
    )
    config_parser.add_argument(
        "--enable-network",
        type=lambda x: x.lower() == "true",
        help="Enable/disable network optimizer (true/false)"
    )
    config_parser.add_argument(
        "--enable-database",
        type=lambda x: x.lower() == "true",
        help="Enable/disable database optimizer (true/false)"
    )
    config_parser.add_argument(
        "--enable-distributed",
        type=lambda x: x.lower() == "true",
        help="Enable/disable distributed optimizer (true/false)"
    )
    config_parser.add_argument(
        "--optimization-level",
        choices=["conservative", "balanced", "aggressive"],
        help="Set overall optimization level"
    )
    config_parser.add_argument(
        "--settings",
        nargs="+",
        help="Additional settings in key=value format"
    )

    # Help command
    help_parser = subparsers.add_parser(
        "help",
        help="Show help information"
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    tools = PerformanceTools()

    if args.command == "benchmark":
        tools.run_benchmarks(args)
    elif args.command == "profile":
        tools.run_profiler(args)
    elif args.command == "visualize":
        tools.run_visualization(args)
    elif args.command == "config":
        if args.show or (
            args.enable_memory is None and
            args.enable_cpu is None and
            args.enable_io is None and
            args.enable_network is None and
            args.enable_database is None and
            args.enable_distributed is None and
            not args.optimization_level and
            not args.settings
        ):
            tools.show_config(args)
        else:
            tools.update_config(args)
    elif args.command == "help" or args.command is None:
        tools.show_help(args)
    else:
        console.print("[bold red]Unknown command[/bold red]")
        tools.show_help(args)


if __name__ == "__main__":
    main()
