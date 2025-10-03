#!/usr/bin/env python3
"""
Performance Visualization Tool for Negative Space Imaging Project

This script visualizes performance metrics and optimization results
to provide insights into system performance characteristics.

Author: Stephen Bilodeau
Copyright: © 2025 Negative Space Imaging, Inc.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("visualization")

# Set up console output
console = Console()


class PerformanceVisualizer:
    """Visualizes performance data from benchmark and profiler results."""

    def __init__(self, output_dir="performance_visualizations"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to store visualization outputs
        """
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Set default style for plots
        plt.style.use('ggplot')
        sns.set_theme(style="whitegrid")

    def load_benchmark_data(self, file_path):
        """
        Load benchmark data from a JSON file.

        Args:
            file_path: Path to the benchmark results JSON file

        Returns:
            Loaded benchmark data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading benchmark data from {file_path}: {str(e)}")
            console.print(f"[red]Error loading benchmark data: {str(e)}[/red]")
            return None

    def load_profiler_data(self, file_path):
        """
        Load profiler data from a JSON file.

        Args:
            file_path: Path to the profiler results JSON file

        Returns:
            Loaded profiler data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading profiler data from {file_path}: {str(e)}")
            console.print(f"[red]Error loading profiler data: {str(e)}[/red]")
            return None

    def visualize_benchmark(self, benchmark_data, save_prefix=None):
        """
        Visualize benchmark data.

        Args:
            benchmark_data: Benchmark data to visualize
            save_prefix: Prefix for saved visualization files

        Returns:
            List of paths to generated visualization files
        """
        if not benchmark_data:
            return []

        generated_files = []

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Generating benchmark visualizations...",
                total=len(benchmark_data)
            )

            for category, data in benchmark_data.items():
                if "Memory Optimization" in category:
                    file_path = self._visualize_memory_optimization(data, save_prefix)
                    generated_files.append(file_path)

                elif "CPU Optimization" in category:
                    file_path = self._visualize_cpu_optimization(data, save_prefix)
                    generated_files.append(file_path)

                elif "I/O Optimization" in category:
                    file_path = self._visualize_io_optimization(data, save_prefix)
                    generated_files.append(file_path)

                elif "Database Optimization" in category:
                    file_path = self._visualize_db_optimization(data, save_prefix)
                    generated_files.append(file_path)

                progress.update(task, advance=1)

        # Generate summary comparison if we have multiple categories
        if len(benchmark_data) > 1:
            summary_path = self._visualize_optimization_summary(
                benchmark_data, save_prefix
            )
            generated_files.append(summary_path)

        return generated_files

    def _visualize_memory_optimization(self, data, save_prefix=None):
        """
        Visualize memory optimization benchmark results.

        Args:
            data: Memory optimization benchmark data
            save_prefix: Prefix for saved visualization file

        Returns:
            Path to generated visualization file
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Extract data
        sizes = []
        original_sizes = []
        optimized_sizes = []
        savings_pcts = []

        for key, result in data.items():
            if key.startswith("size_"):
                size = int(key.split("_")[1])
                sizes.append(size)
                original_sizes.append(result["original_size"])
                optimized_sizes.append(result["optimized_size"])
                savings_pcts.append(result["savings_percent"])

        # Sort by size
        indices = np.argsort(sizes)
        sizes = [sizes[i] for i in indices]
        original_sizes = [original_sizes[i] for i in indices]
        optimized_sizes = [optimized_sizes[i] for i in indices]
        savings_pcts = [savings_pcts[i] for i in indices]

        # Size comparison chart
        x = np.arange(len(sizes))
        width = 0.35

        # Convert to MB for readability if sizes are large
        scale_factor = 1
        size_unit = "bytes"
        if max(original_sizes) > 1024*1024:
            scale_factor = 1024*1024
            size_unit = "MB"
        elif max(original_sizes) > 1024:
            scale_factor = 1024
            size_unit = "KB"

        orig_scaled = [size/scale_factor for size in original_sizes]
        opt_scaled = [size/scale_factor for size in optimized_sizes]

        ax1.bar(x - width/2, orig_scaled, width, label='Original')
        ax1.bar(x + width/2, opt_scaled, width, label='Optimized')

        ax1.set_xlabel('Array Size')
        ax1.set_ylabel(f'Memory Usage ({size_unit})')
        ax1.set_title('Memory Usage Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{size:,}" for size in sizes])
        ax1.legend()

        # Rotate x-axis labels if needed
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

        # Memory savings percentage chart
        ax2.plot(sizes, savings_pcts, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Array Size')
        ax2.set_ylabel('Memory Savings (%)')
        ax2.set_title('Memory Optimization Effectiveness')
        ax2.set_xscale('log')
        ax2.grid(True)

        # Add data points with values
        for i, (size, pct) in enumerate(zip(sizes, savings_pcts)):
            ax2.annotate(
                f"{pct:.1f}%",
                (size, pct),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9
            )

        plt.suptitle("Memory Optimization Performance", fontsize=16)
        plt.tight_layout()

        # Save the figure
        filename = "memory_optimization_performance.png"
        if save_prefix:
            filename = f"{save_prefix}_{filename}"

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Memory optimization visualization saved to {filepath}")
        return filepath

    def _visualize_cpu_optimization(self, data, save_prefix=None):
        """
        Visualize CPU optimization benchmark results.

        Args:
            data: CPU optimization benchmark data
            save_prefix: Prefix for saved visualization file

        Returns:
            Path to generated visualization file
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Extract data
        sizes = []
        unoptimized_times = []
        optimized_times = []
        speedups = []

        for key, result in data.items():
            if key.startswith("size_"):
                size = int(key.split("_")[1])
                sizes.append(size)
                unoptimized_times.append(result["unoptimized_time"])
                optimized_times.append(result["optimized_time"])
                speedups.append(result["speedup"])

        # Sort by size
        indices = np.argsort(sizes)
        sizes = [sizes[i] for i in indices]
        unoptimized_times = [unoptimized_times[i] for i in indices]
        optimized_times = [optimized_times[i] for i in indices]
        speedups = [speedups[i] for i in indices]

        # Execution time comparison chart
        x = np.arange(len(sizes))
        width = 0.35

        ax1.bar(x - width/2, unoptimized_times, width, label='Unoptimized')
        ax1.bar(x + width/2, optimized_times, width, label='Optimized')

        ax1.set_xlabel('Data Size')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{size:,}" for size in sizes])
        ax1.legend()

        # Rotate x-axis labels if needed
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

        # Speedup chart
        ax2.plot(sizes, speedups, 'o-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Data Size')
        ax2.set_ylabel('Speedup Factor (x)')
        ax2.set_title('CPU Optimization Speedup')
        ax2.set_xscale('log')
        ax2.grid(True)

        # Add data points with values
        for i, (size, speedup) in enumerate(zip(sizes, speedups)):
            ax2.annotate(
                f"{speedup:.1f}x",
                (size, speedup),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9
            )

        plt.suptitle("CPU Optimization Performance", fontsize=16)
        plt.tight_layout()

        # Save the figure
        filename = "cpu_optimization_performance.png"
        if save_prefix:
            filename = f"{save_prefix}_{filename}"

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"CPU optimization visualization saved to {filepath}")
        return filepath

    def _visualize_io_optimization(self, data, save_prefix=None):
        """
        Visualize I/O optimization benchmark results.

        Args:
            data: I/O optimization benchmark data
            save_prefix: Prefix for saved visualization file

        Returns:
            Path to generated visualization file
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Extract data
        sizes = []
        unopt_write_times = []
        opt_write_times = []
        unopt_read_times = []
        opt_read_times = []
        write_speedups = []
        read_speedups = []

        for key, result in data.items():
            if key.startswith("size_"):
                size = int(key.split("_")[1])
                sizes.append(size)
                unopt_write_times.append(result["unoptimized_write_time"])
                opt_write_times.append(result["optimized_write_time"])
                unopt_read_times.append(result["unoptimized_read_time"])
                opt_read_times.append(result["optimized_read_time"])
                write_speedups.append(result["write_speedup"])
                read_speedups.append(result["read_speedup"])

        # Sort by size
        indices = np.argsort(sizes)
        sizes = [sizes[i] for i in indices]
        unopt_write_times = [unopt_write_times[i] for i in indices]
        opt_write_times = [opt_write_times[i] for i in indices]
        unopt_read_times = [unopt_read_times[i] for i in indices]
        opt_read_times = [opt_read_times[i] for i in indices]
        write_speedups = [write_speedups[i] for i in indices]
        read_speedups = [read_speedups[i] for i in indices]

        # Size unit for display
        size_unit = "bytes"
        if max(sizes) > 1024*1024:
            sizes_display = [f"{size/1024/1024:.1f} MB" for size in sizes]
            size_unit = "MB"
        elif max(sizes) > 1024:
            sizes_display = [f"{size/1024:.1f} KB" for size in sizes]
            size_unit = "KB"
        else:
            sizes_display = [f"{size} bytes" for size in sizes]

        # Execution time comparison chart (side by side grouped bars)
        x = np.arange(len(sizes))
        width = 0.2

        # Read operations
        ax1.bar(x - width*1.5, unopt_read_times, width, label='Unopt Read')
        ax1.bar(x - width/2, opt_read_times, width, label='Opt Read')

        # Write operations
        ax1.bar(x + width/2, unopt_write_times, width, label='Unopt Write')
        ax1.bar(x + width*1.5, opt_write_times, width, label='Opt Write')

        ax1.set_xlabel(f'File Size ({size_unit})')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('I/O Operation Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{size:,}" for size in sizes])
        ax1.legend()

        # Rotate x-axis labels if needed
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

        # Speedup chart
        ax2.plot(sizes, read_speedups, 'o-', linewidth=2, markersize=8,
                 label='Read Speedup')
        ax2.plot(sizes, write_speedups, 's-', linewidth=2, markersize=8,
                 label='Write Speedup')

        ax2.set_xlabel(f'File Size ({size_unit})')
        ax2.set_ylabel('Speedup Factor (x)')
        ax2.set_title('I/O Optimization Speedup')
        ax2.set_xscale('log')
        ax2.grid(True)
        ax2.legend()

        # Add data points with values
        for i, (size, speedup) in enumerate(zip(sizes, read_speedups)):
            ax2.annotate(
                f"{speedup:.1f}x",
                (size, speedup),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9
            )

        for i, (size, speedup) in enumerate(zip(sizes, write_speedups)):
            ax2.annotate(
                f"{speedup:.1f}x",
                (size, speedup),
                xytext=(5, -15),
                textcoords="offset points",
                fontsize=9
            )

        plt.suptitle("I/O Optimization Performance", fontsize=16)
        plt.tight_layout()

        # Save the figure
        filename = "io_optimization_performance.png"
        if save_prefix:
            filename = f"{save_prefix}_{filename}"

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"I/O optimization visualization saved to {filepath}")
        return filepath

    def _visualize_db_optimization(self, data, save_prefix=None):
        """
        Visualize database optimization benchmark results.

        Args:
            data: Database optimization benchmark data
            save_prefix: Prefix for saved visualization file

        Returns:
            Path to generated visualization file
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Extract data
        counts = []
        unopt_insert_times = []
        opt_insert_times = []
        unopt_query_times = []
        opt_query_times = []
        insert_speedups = []
        query_speedups = []

        for key, result in data.items():
            if key.startswith("count_"):
                count = int(key.split("_")[1])
                counts.append(count)
                unopt_insert_times.append(result["unoptimized_insert_time"])
                opt_insert_times.append(result["optimized_insert_time"])
                unopt_query_times.append(result["unoptimized_query_time"])
                opt_query_times.append(result["optimized_query_time"])
                insert_speedups.append(result["insert_speedup"])
                query_speedups.append(result["query_speedup"])

        # Sort by count
        indices = np.argsort(counts)
        counts = [counts[i] for i in indices]
        unopt_insert_times = [unopt_insert_times[i] for i in indices]
        opt_insert_times = [opt_insert_times[i] for i in indices]
        unopt_query_times = [unopt_query_times[i] for i in indices]
        opt_query_times = [opt_query_times[i] for i in indices]
        insert_speedups = [insert_speedups[i] for i in indices]
        query_speedups = [query_speedups[i] for i in indices]

        # Execution time comparison chart (side by side grouped bars)
        x = np.arange(len(counts))
        width = 0.2

        # Query operations
        ax1.bar(x - width*1.5, unopt_query_times, width, label='Unopt Query')
        ax1.bar(x - width/2, opt_query_times, width, label='Opt Query')

        # Insert operations
        ax1.bar(x + width/2, unopt_insert_times, width, label='Unopt Insert')
        ax1.bar(x + width*1.5, opt_insert_times, width, label='Opt Insert')

        ax1.set_xlabel('Record Count')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Database Operation Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{count:,}" for count in counts])
        ax1.legend()

        # Speedup chart
        ax2.plot(counts, query_speedups, 'o-', linewidth=2, markersize=8,
                 label='Query Speedup')
        ax2.plot(counts, insert_speedups, 's-', linewidth=2, markersize=8,
                 label='Insert Speedup')

        ax2.set_xlabel('Record Count')
        ax2.set_ylabel('Speedup Factor (x)')
        ax2.set_title('Database Optimization Speedup')
        ax2.set_xscale('log')
        ax2.grid(True)
        ax2.legend()

        # Add data points with values
        for i, (count, speedup) in enumerate(zip(counts, query_speedups)):
            ax2.annotate(
                f"{speedup:.1f}x",
                (count, speedup),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9
            )

        for i, (count, speedup) in enumerate(zip(counts, insert_speedups)):
            ax2.annotate(
                f"{speedup:.1f}x",
                (count, speedup),
                xytext=(5, -15),
                textcoords="offset points",
                fontsize=9
            )

        plt.suptitle("Database Optimization Performance", fontsize=16)
        plt.tight_layout()

        # Save the figure
        filename = "database_optimization_performance.png"
        if save_prefix:
            filename = f"{save_prefix}_{filename}"

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Database optimization visualization saved to {filepath}")
        return filepath

    def _visualize_optimization_summary(self, benchmark_data, save_prefix=None):
        """
        Visualize a summary of all optimization categories.

        Args:
            benchmark_data: Complete benchmark data
            save_prefix: Prefix for saved visualization file

        Returns:
            Path to generated visualization file
        """
        # Extract average improvements for each category
        categories = []
        improvements = []
        std_devs = []  # For error bars

        # Process memory optimization data
        if "Memory Optimization" in benchmark_data:
            memory_data = benchmark_data["Memory Optimization"]
            memory_savings = []

            for key, result in memory_data.items():
                if key.startswith("size_"):
                    memory_savings.append(result["savings_percent"] / 100)  # Convert to ratio

            if memory_savings:
                categories.append("Memory")
                improvements.append(np.mean(memory_savings))
                std_devs.append(np.std(memory_savings))

        # Process CPU optimization data
        if "CPU Optimization" in benchmark_data:
            cpu_data = benchmark_data["CPU Optimization"]
            cpu_speedups = []

            for key, result in cpu_data.items():
                if key.startswith("size_"):
                    cpu_speedups.append(result["speedup"])

            if cpu_speedups:
                categories.append("CPU")
                improvements.append(np.mean(cpu_speedups))
                std_devs.append(np.std(cpu_speedups))

        # Process I/O optimization data
        if "I/O Optimization" in benchmark_data:
            io_data = benchmark_data["I/O Optimization"]
            read_speedups = []
            write_speedups = []

            for key, result in io_data.items():
                if key.startswith("size_"):
                    read_speedups.append(result["read_speedup"])
                    write_speedups.append(result["write_speedup"])

            if read_speedups and write_speedups:
                # Combine read and write speedups
                combined_speedups = []
                for read, write in zip(read_speedups, write_speedups):
                    combined_speedups.append((read + write) / 2)

                categories.append("I/O")
                improvements.append(np.mean(combined_speedups))
                std_devs.append(np.std(combined_speedups))

        # Process database optimization data
        if "Database Optimization" in benchmark_data:
            db_data = benchmark_data["Database Optimization"]
            query_speedups = []
            insert_speedups = []

            for key, result in db_data.items():
                if key.startswith("count_"):
                    query_speedups.append(result["query_speedup"])
                    insert_speedups.append(result["insert_speedup"])

            if query_speedups and insert_speedups:
                # Combine query and insert speedups
                combined_speedups = []
                for query, insert in zip(query_speedups, insert_speedups):
                    combined_speedups.append((query + insert) / 2)

                categories.append("Database")
                improvements.append(np.mean(combined_speedups))
                std_devs.append(np.std(combined_speedups))

        # Create bar chart with error bars
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(categories))
        bars = ax.bar(x, improvements, yerr=std_devs, capsize=10,
                    color=plt.cm.viridis(np.linspace(0, 0.8, len(categories))))

        ax.set_xlabel('Optimization Category')
        ax.set_ylabel('Average Performance Improvement Factor')
        ax.set_title('Performance Optimization Summary')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.yaxis.grid(True)

        # Add values on top of bars
        for i, v in enumerate(improvements):
            ax.text(i, v + std_devs[i] + 0.1, f"{v:.2f}x",
                    ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # Save the figure
        filename = "optimization_summary.png"
        if save_prefix:
            filename = f"{save_prefix}_{filename}"

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Optimization summary visualization saved to {filepath}")
        return filepath

    def visualize_profiler_data(self, profiler_data, save_prefix=None):
        """
        Visualize profiler data.

        Args:
            profiler_data: Profiler data to visualize
            save_prefix: Prefix for saved visualization files

        Returns:
            List of paths to generated visualization files
        """
        if not profiler_data:
            return []

        generated_files = []

        # Process hotspots from the profiler data
        hotspots = self._extract_hotspots(profiler_data)

        if hotspots:
            # Create hotspot visualization
            filepath = self._visualize_hotspots(hotspots, save_prefix)
            generated_files.append(filepath)

        # Create function execution time visualization if available
        if self._has_function_stats(profiler_data):
            filepath = self._visualize_function_times(profiler_data, save_prefix)
            generated_files.append(filepath)

        return generated_files

    def _extract_hotspots(self, profiler_data):
        """
        Extract performance hotspots from profiler data.

        Args:
            profiler_data: Profiler data to analyze

        Returns:
            List of hotspots with time and function information
        """
        hotspots = []

        # Check if we have flat or nested profiler data
        if "function_stats" in profiler_data:
            # Process function stats directly
            for stat in profiler_data["function_stats"][:15]:  # Top 15 functions
                parts = stat.split()

                # Try to extract function name and time
                function_name = "Unknown"
                execution_time = 0.0

                for part in parts:
                    # Look for time value (usually format like "123.456:")
                    if ":" in part and part[0].isdigit():
                        try:
                            execution_time = float(part.split(":")[0])
                        except ValueError:
                            pass

                    # Look for function name (usually in format like "function_name()")
                    if "(" in part and ")" in part:
                        function_name = part

                if execution_time > 0:
                    hotspots.append({
                        "function": function_name,
                        "time": execution_time
                    })
        else:
            # Try to process nested structure
            for name, data in profiler_data.items():
                if isinstance(data, dict) and "function_stats" in data:
                    for stat in data["function_stats"][:5]:  # Top 5 functions for each subitem
                        parts = stat.split()

                        # Extract as above
                        function_name = "Unknown"
                        execution_time = 0.0

                        for part in parts:
                            if ":" in part and part[0].isdigit():
                                try:
                                    execution_time = float(part.split(":")[0])
                                except ValueError:
                                    pass

                            if "(" in part and ")" in part:
                                function_name = part

                        if execution_time > 0:
                            hotspots.append({
                                "function": function_name,
                                "time": execution_time,
                                "module": name
                            })

        # Sort hotspots by execution time (descending)
        hotspots.sort(key=lambda x: x["time"], reverse=True)

        return hotspots

    def _has_function_stats(self, profiler_data):
        """
        Check if profiler data contains function statistics.

        Args:
            profiler_data: Profiler data to check

        Returns:
            True if function statistics are available, False otherwise
        """
        if "function_stats" in profiler_data:
            return len(profiler_data["function_stats"]) > 0

        # Check nested structure
        for name, data in profiler_data.items():
            if isinstance(data, dict) and "function_stats" in data:
                if len(data["function_stats"]) > 0:
                    return True

        return False

    def _visualize_hotspots(self, hotspots, save_prefix=None):
        """
        Visualize performance hotspots.

        Args:
            hotspots: List of performance hotspots
            save_prefix: Prefix for saved visualization file

        Returns:
            Path to generated visualization file
        """
        # Take top 20 hotspots or all if less than 20
        top_hotspots = hotspots[:min(20, len(hotspots))]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Extract function names and times
        functions = []
        times = []
        colors = []

        color_map = plt.cm.viridis(np.linspace(0, 0.8, len(top_hotspots)))

        for i, hotspot in enumerate(top_hotspots):
            # Truncate long function names
            func_name = hotspot["function"]
            if len(func_name) > 40:
                func_name = func_name[:37] + "..."

            # Add module name if available
            if "module" in hotspot:
                func_name = f"{hotspot['module']}: {func_name}"

            functions.append(func_name)
            times.append(hotspot["time"])
            colors.append(color_map[i])

        # Create horizontal bar chart
        bars = ax.barh(range(len(functions)), times, color=colors)

        # Customize chart
        ax.set_yticks(range(len(functions)))
        ax.set_yticklabels(functions)
        ax.set_xlabel('Execution Time (seconds)')
        ax.set_title('Performance Hotspots')
        ax.grid(axis='x')

        # Add values to bars
        for i, (time, bar) in enumerate(zip(times, bars)):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height()/2,
                f"{time:.4f}s",
                va='center'
            )

        plt.tight_layout()

        # Save the figure
        filename = "performance_hotspots.png"
        if save_prefix:
            filename = f"{save_prefix}_{filename}"

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Performance hotspot visualization saved to {filepath}")
        return filepath

    def _visualize_function_times(self, profiler_data, save_prefix=None):
        """
        Visualize function execution times.

        Args:
            profiler_data: Profiler data containing function stats
            save_prefix: Prefix for saved visualization file

        Returns:
            Path to generated visualization file
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Process function stats
        if "function_stats" in profiler_data:
            # Extract function call counts and times
            functions = []
            times = []
            call_counts = []

            for stat in profiler_data["function_stats"][:15]:  # Top 15 functions
                parts = stat.split()

                # Extract function information
                function_name = "Unknown"
                execution_time = 0.0
                calls = 0

                # First part is usually ncalls
                if parts and "/" in parts[0]:
                    calls_part = parts[0].split("/")[0]
                    try:
                        calls = int(calls_part)
                    except ValueError:
                        pass

                for part in parts:
                    # Look for time value
                    if ":" in part and part[0].isdigit():
                        try:
                            execution_time = float(part.split(":")[0])
                        except ValueError:
                            pass

                    # Look for function name
                    if "(" in part and ")" in part:
                        function_name = part

                if execution_time > 0:
                    # Truncate long function names
                    if len(function_name) > 30:
                        function_name = function_name[:27] + "..."

                    functions.append(function_name)
                    times.append(execution_time)
                    call_counts.append(calls)

            # Create scatter plot of execution time vs. call count
            scatter = ax.scatter(
                call_counts,
                times,
                s=[min(t*500, 1000) for t in times],  # Size based on time
                c=times,  # Color based on time
                cmap='viridis',
                alpha=0.7
            )

            # Add function labels
            for i, func in enumerate(functions):
                ax.annotate(
                    func,
                    (call_counts[i], times[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8
                )

            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Execution Time (seconds)')

            # Customize chart
            ax.set_xlabel('Number of Calls')
            ax.set_ylabel('Execution Time (seconds)')
            ax.set_title('Function Execution Time vs. Call Count')
            ax.set_xscale('log')
            ax.grid(True)

            # Save the figure
            filename = "function_execution_times.png"
            if save_prefix:
                filename = f"{save_prefix}_{filename}"

            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Function execution time visualization saved to {filepath}")
            return filepath

        return None

    def generate_html_report(self, visualization_files, title="Performance Visualization Report", output_file=None):
        """
        Generate an HTML report containing all visualizations.

        Args:
            visualization_files: List of visualization file paths
            title: Title for the HTML report
            output_file: Path to save the HTML report

        Returns:
            Path to the generated HTML report
        """
        if not output_file:
            output_file = os.path.join(self.output_dir, "visualization_report.html")

        # Start HTML content
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .visualization {{ margin-bottom: 40px; }}
        img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
        .timestamp {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
"""

        # Add each visualization
        for i, file_path in enumerate(visualization_files):
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                title = ' '.join(word.capitalize() for word in file_name.split('_')[:-1])

                html_content += f"""
        <div class="visualization">
            <h2>{title}</h2>
            <img src="{file_path}" alt="{title}">
        </div>
"""

        # Close HTML content
        html_content += """
    </div>
</body>
</html>
"""

        # Write HTML to file
        with open(output_file, 'w') as f:
            f.write(html_content)

        console.print(f"[bold green]HTML report generated: {output_file}[/bold green]")
        return output_file


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Performance Visualization Tool for Negative Space Imaging Project"
    )

    # Add command line arguments
    parser.add_argument(
        "--benchmark", "-b",
        help="Path to benchmark results JSON file"
    )
    parser.add_argument(
        "--profiler", "-p",
        help="Path to profiler results JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for visualizations",
        default="performance_visualizations"
    )
    parser.add_argument(
        "--prefix",
        help="Prefix for saved visualization files"
    )
    parser.add_argument(
        "--report", "-r",
        help="Generate HTML report and save to specified path",
        action="store_true"
    )

    args = parser.parse_args()

    # Check if at least one input file is provided
    if not args.benchmark and not args.profiler:
        console.print("[bold red]Error: At least one input file (benchmark or profiler results) is required[/bold red]")
        parser.print_help()
        return

    # Initialize visualizer
    visualizer = PerformanceVisualizer(output_dir=args.output)

    all_visualizations = []

    # Process benchmark data if provided
    if args.benchmark:
        console.print(f"[bold]Processing benchmark data from: {args.benchmark}[/bold]")
        benchmark_data = visualizer.load_benchmark_data(args.benchmark)

        if benchmark_data:
            vis_files = visualizer.visualize_benchmark(benchmark_data, args.prefix)
            all_visualizations.extend(vis_files)

    # Process profiler data if provided
    if args.profiler:
        console.print(f"[bold]Processing profiler data from: {args.profiler}[/bold]")
        profiler_data = visualizer.load_profiler_data(args.profiler)

        if profiler_data:
            vis_files = visualizer.visualize_profiler_data(profiler_data, args.prefix)
            all_visualizations.extend(vis_files)

    # Generate HTML report if requested
    if args.report and all_visualizations:
        report_path = os.path.join(args.output, "visualization_report.html")
        visualizer.generate_html_report(all_visualizations, output_file=report_path)

    # Show summary
    if all_visualizations:
        console.print(f"[bold green]Successfully generated {len(all_visualizations)} visualizations[/bold green]")
        for viz in all_visualizations:
            console.print(f"  - {viz}")
    else:
        console.print("[bold yellow]No visualizations were generated[/bold yellow]")


if __name__ == "__main__":
    main()
