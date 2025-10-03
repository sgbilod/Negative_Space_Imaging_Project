#!/usr/bin/env python3
"""
Performance Optimization Dashboard Generator

This script creates a comprehensive HTML dashboard summarizing the
performance optimization system, its components, and sample results.

Author: Stephen Bilodeau
Copyright: © 2025 Negative Space Imaging, Inc.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("dashboard_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dashboard_generator")

# Set up console output
console = Console()


class DashboardGenerator:
    """Generates a performance optimization dashboard."""

    def __init__(self, output_dir="performance_dashboard"):
        """
        Initialize the dashboard generator.

        Args:
            output_dir: Directory to store the dashboard
        """
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create assets directory
        assets_dir = os.path.join(output_dir, "assets")
        if not os.path.exists(assets_dir):
            os.makedirs(assets_dir)

    def generate_sample_data(self):
        """
        Generate sample optimization data for demonstration.

        Returns:
            Dictionary containing sample data
        """
        # Sample memory optimization data
        memory_sizes = [100, 1000, 10000, 100000, 1000000]
        memory_original = [size * 4 for size in memory_sizes]  # 4 bytes per int32
        memory_optimized = memory_sizes.copy()  # 1 byte per uint8
        memory_savings = [orig - opt for orig, opt in zip(memory_original, memory_optimized)]
        memory_savings_pct = [100 * (orig - opt) / orig for orig, opt in zip(memory_original, memory_optimized)]

        # Sample CPU optimization data
        cpu_sizes = [1000, 10000, 100000, 1000000]
        cpu_unoptimized = [0.05, 0.5, 5.0, 50.0]  # Seconds
        cpu_speedups = [1.2, 2.5, 3.8, 4.2]
        cpu_optimized = [t / s for t, s in zip(cpu_unoptimized, cpu_speedups)]

        # Sample I/O optimization data
        io_sizes = [1024, 10240, 102400, 1024000]
        io_read_speedups = [1.3, 1.8, 2.2, 2.7]
        io_write_speedups = [1.1, 1.5, 2.0, 2.4]

        # Sample database optimization data
        db_record_counts = [100, 1000, 10000]
        db_query_speedups = [1.5, 2.2, 3.0]
        db_insert_speedups = [1.2, 1.8, 2.5]

        # Combine all data
        return {
            "memory": {
                "sizes": memory_sizes,
                "original": memory_original,
                "optimized": memory_optimized,
                "savings": memory_savings,
                "savings_pct": memory_savings_pct
            },
            "cpu": {
                "sizes": cpu_sizes,
                "unoptimized": cpu_unoptimized,
                "optimized": cpu_optimized,
                "speedups": cpu_speedups
            },
            "io": {
                "sizes": io_sizes,
                "read_speedups": io_read_speedups,
                "write_speedups": io_write_speedups
            },
            "database": {
                "record_counts": db_record_counts,
                "query_speedups": db_query_speedups,
                "insert_speedups": db_insert_speedups
            }
        }

    def generate_charts(self, data):
        """
        Generate charts for the dashboard.

        Args:
            data: Performance data

        Returns:
            Dictionary of chart file paths
        """
        charts = {}

        # Generate memory optimization chart
        memory_chart = os.path.join(self.output_dir, "assets", "memory_optimization.png")
        self._generate_memory_chart(data["memory"], memory_chart)
        charts["memory"] = "assets/memory_optimization.png"

        # Generate CPU optimization chart
        cpu_chart = os.path.join(self.output_dir, "assets", "cpu_optimization.png")
        self._generate_cpu_chart(data["cpu"], cpu_chart)
        charts["cpu"] = "assets/cpu_optimization.png"

        # Generate I/O optimization chart
        io_chart = os.path.join(self.output_dir, "assets", "io_optimization.png")
        self._generate_io_chart(data["io"], io_chart)
        charts["io"] = "assets/io_optimization.png"

        # Generate database optimization chart
        db_chart = os.path.join(self.output_dir, "assets", "database_optimization.png")
        self._generate_db_chart(data["database"], db_chart)
        charts["database"] = "assets/database_optimization.png"

        # Generate summary chart
        summary_chart = os.path.join(self.output_dir, "assets", "optimization_summary.png")
        self._generate_summary_chart(data, summary_chart)
        charts["summary"] = "assets/optimization_summary.png"

        return charts

    def _generate_memory_chart(self, memory_data, output_file):
        """
        Generate memory optimization chart.

        Args:
            memory_data: Memory optimization data
            output_file: Output file path
        """
        plt.figure(figsize=(10, 6))

        # Extract data
        sizes = memory_data["sizes"]
        original = memory_data["original"]
        optimized = memory_data["optimized"]
        savings_pct = memory_data["savings_pct"]

        # Create bar chart
        x = np.arange(len(sizes))
        width = 0.35

        plt.bar(x - width/2, original, width, label='Original Size')
        plt.bar(x + width/2, optimized, width, label='Optimized Size')

        # Add second y-axis for percentage
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(x, savings_pct, 'ro-', linewidth=2, label='Memory Savings (%)')
        ax2.set_ylabel('Memory Savings (%)')
        ax2.set_ylim(0, 100)

        # Customize chart
        plt.xlabel('Array Size')
        plt.ylabel('Memory Usage (bytes)')
        plt.title('Memory Optimization Performance')
        plt.xticks(x, [f"{size:,}" for size in sizes])

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def _generate_cpu_chart(self, cpu_data, output_file):
        """
        Generate CPU optimization chart.

        Args:
            cpu_data: CPU optimization data
            output_file: Output file path
        """
        plt.figure(figsize=(10, 6))

        # Extract data
        sizes = cpu_data["sizes"]
        unoptimized = cpu_data["unoptimized"]
        optimized = cpu_data["optimized"]
        speedups = cpu_data["speedups"]

        # Create bar chart
        x = np.arange(len(sizes))
        width = 0.35

        plt.bar(x - width/2, unoptimized, width, label='Unoptimized Time')
        plt.bar(x + width/2, optimized, width, label='Optimized Time')

        # Add second y-axis for speedup
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(x, speedups, 'ro-', linewidth=2, label='Speedup Factor')
        ax2.set_ylabel('Speedup Factor (x)')

        # Customize chart
        plt.xlabel('Data Size')
        plt.ylabel('Execution Time (seconds)')
        plt.title('CPU Optimization Performance')
        plt.xticks(x, [f"{size:,}" for size in sizes])

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def _generate_io_chart(self, io_data, output_file):
        """
        Generate I/O optimization chart.

        Args:
            io_data: I/O optimization data
            output_file: Output file path
        """
        plt.figure(figsize=(10, 6))

        # Extract data
        sizes = io_data["sizes"]
        read_speedups = io_data["read_speedups"]
        write_speedups = io_data["write_speedups"]

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

        # Create line chart
        plt.plot(sizes, read_speedups, 'bo-', linewidth=2, markersize=8,
                 label='Read Operations')
        plt.plot(sizes, write_speedups, 'go-', linewidth=2, markersize=8,
                 label='Write Operations')

        # Add data labels
        for i, (size, speedup) in enumerate(zip(sizes, read_speedups)):
            plt.text(size, speedup, f"{speedup:.1f}x",
                     ha='center', va='bottom')

        for i, (size, speedup) in enumerate(zip(sizes, write_speedups)):
            plt.text(size, speedup, f"{speedup:.1f}x",
                     ha='center', va='bottom')

        # Customize chart
        plt.xlabel(f'File Size ({size_unit})')
        plt.ylabel('Speedup Factor (x)')
        plt.title('I/O Optimization Performance')
        plt.xscale('log')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def _generate_db_chart(self, db_data, output_file):
        """
        Generate database optimization chart.

        Args:
            db_data: Database optimization data
            output_file: Output file path
        """
        plt.figure(figsize=(10, 6))

        # Extract data
        record_counts = db_data["record_counts"]
        query_speedups = db_data["query_speedups"]
        insert_speedups = db_data["insert_speedups"]

        # Create grouped bar chart
        x = np.arange(len(record_counts))
        width = 0.35

        plt.bar(x - width/2, query_speedups, width, label='Query Operations')
        plt.bar(x + width/2, insert_speedups, width, label='Insert Operations')

        # Add data labels
        for i, speedup in enumerate(query_speedups):
            plt.text(i - width/2, speedup, f"{speedup:.1f}x",
                     ha='center', va='bottom')

        for i, speedup in enumerate(insert_speedups):
            plt.text(i + width/2, speedup, f"{speedup:.1f}x",
                     ha='center', va='bottom')

        # Customize chart
        plt.xlabel('Record Count')
        plt.ylabel('Speedup Factor (x)')
        plt.title('Database Optimization Performance')
        plt.xticks(x, [f"{count:,}" for count in record_counts])
        plt.grid(axis='y')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def _generate_summary_chart(self, all_data, output_file):
        """
        Generate summary chart of all optimizations.

        Args:
            all_data: All optimization data
            output_file: Output file path
        """
        plt.figure(figsize=(12, 7))

        # Calculate average improvements for each category
        memory_avg = np.mean(all_data["memory"]["savings_pct"]) / 100
        cpu_avg = np.mean(all_data["cpu"]["speedups"])
        io_avg = np.mean(all_data["io"]["read_speedups"] + all_data["io"]["write_speedups"]) / 2
        db_avg = np.mean(all_data["database"]["query_speedups"] + all_data["database"]["insert_speedups"]) / 2

        # Prepare data
        categories = ['Memory', 'CPU', 'I/O', 'Database']
        values = [memory_avg, cpu_avg, io_avg, db_avg]

        # Create bar chart
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
        bars = plt.bar(categories, values, color=colors)

        # Add data labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f"{value:.2f}x",
                    ha='center', va='bottom', fontweight='bold')

        # Customize chart
        plt.ylabel('Average Performance Improvement Factor')
        plt.title('Performance Optimization Summary')
        plt.ylim(0, max(values) * 1.2)  # Add some space for labels
        plt.grid(axis='y')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    def generate_dashboard(self, data=None, title="Performance Optimization Dashboard"):
        """
        Generate the HTML dashboard.

        Args:
            data: Performance data (if None, sample data will be used)
            title: Dashboard title

        Returns:
            Path to the generated dashboard HTML file
        """
        # Use sample data if none provided
        if data is None:
            data = self.generate_sample_data()

        # Generate charts
        charts = self.generate_charts(data)

        # Generate HTML
        dashboard_path = os.path.join(self.output_dir, "index.html")

        with open(dashboard_path, 'w') as f:
            f.write(self._generate_html(data, charts, title))

        logger.info(f"Dashboard generated at: {dashboard_path}")
        return dashboard_path

    def _generate_html(self, data, charts, title):
        """
        Generate HTML content for the dashboard.

        Args:
            data: Performance data
            charts: Chart file paths
            title: Dashboard title

        Returns:
            HTML content
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px 0;
            text-align: center;
        }}
        .dashboard-title {{
            margin: 0;
            font-size: 28px;
        }}
        .timestamp {{
            font-size: 14px;
            color: #ccc;
            margin-top: 5px;
        }}
        .summary-section {{
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            margin: 20px 0;
        }}
        .card-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }}
        .card {{
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            flex: 1;
            min-width: 300px;
            padding: 20px;
        }}
        .card h2 {{
            color: #2c3e50;
            font-size: 20px;
            margin-top: 0;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        .chart-container {{
            width: 100%;
            max-width: 800px;
            margin: 20px auto;
        }}
        .chart {{
            width: 100%;
            height: auto;
            border: 1px solid #eee;
            border-radius: 5px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-item {{
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 15px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .system-components {{
            margin: 30px 0;
        }}
        .component {{
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }}
        .component h3 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        .component p {{
            line-height: 1.6;
        }}
        footer {{
            background-color: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px 0;
            margin-top: 40px;
        }}
        .code-example {{
            background-color: #f4f4f4;
            border-left: 4px solid #2980b9;
            padding: 15px;
            margin: 20px 0;
            overflow-x: auto;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1 class="dashboard-title">{title}</h1>
            <div class="timestamp">Generated on: {timestamp}</div>
        </div>
    </header>

    <div class="container">
        <div class="summary-section">
            <h2>Performance Optimization Summary</h2>
            <p>
                The Performance Optimization System provides comprehensive tools for optimizing application performance
                across multiple dimensions including memory usage, CPU utilization, I/O operations, and database access.
            </p>

            <div class="chart-container">
                <img src="{charts['summary']}" alt="Optimization Summary" class="chart">
            </div>

            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-label">Memory Efficiency</div>
                    <div class="metric-value">{data['memory']['savings_pct'][0]:.1f}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">CPU Speedup</div>
                    <div class="metric-value">{data['cpu']['speedups'][-1]:.1f}x</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">I/O Improvement</div>
                    <div class="metric-value">{max(data['io']['read_speedups']):.1f}x</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Database Speedup</div>
                    <div class="metric-value">{max(data['database']['query_speedups']):.1f}x</div>
                </div>
            </div>
        </div>

        <div class="card-container">
            <div class="card">
                <h2>Memory Optimization</h2>
                <p>Optimizing memory usage through efficient data structures, compression, and pooling.</p>
                <div class="chart-container">
                    <img src="{charts['memory']}" alt="Memory Optimization" class="chart">
                </div>
                <table>
                    <tr>
                        <th>Array Size</th>
                        <th>Original (bytes)</th>
                        <th>Optimized (bytes)</th>
                        <th>Savings</th>
                    </tr>
"""

        # Add memory optimization table rows
        for i, size in enumerate(data["memory"]["sizes"]):
            html += f"""                    <tr>
                        <td>{size:,}</td>
                        <td>{data["memory"]["original"][i]:,}</td>
                        <td>{data["memory"]["optimized"][i]:,}</td>
                        <td>{data["memory"]["savings_pct"][i]:.1f}%</td>
                    </tr>
"""

        html += """                </table>
            </div>

            <div class="card">
                <h2>CPU Optimization</h2>
                <p>Improving processing speed with parallelization, vectorization, and algorithm optimization.</p>
                <div class="chart-container">
                    <img src="{0}" alt="CPU Optimization" class="chart">
                </div>
                <table>
                    <tr>
                        <th>Data Size</th>
                        <th>Unoptimized (s)</th>
                        <th>Optimized (s)</th>
                        <th>Speedup</th>
                    </tr>
""".format(charts['cpu'])

        # Add CPU optimization table rows
        for i, size in enumerate(data["cpu"]["sizes"]):
            html += f"""                    <tr>
                        <td>{size:,}</td>
                        <td>{data["cpu"]["unoptimized"][i]:.4f}</td>
                        <td>{data["cpu"]["optimized"][i]:.4f}</td>
                        <td>{data["cpu"]["speedups"][i]:.1f}x</td>
                    </tr>
"""

        html += """                </table>
            </div>
        </div>

        <div class="card-container">
            <div class="card">
                <h2>I/O Optimization</h2>
                <p>Enhancing file operations with buffering, memory mapping, and asynchronous techniques.</p>
                <div class="chart-container">
                    <img src="{0}" alt="I/O Optimization" class="chart">
                </div>
                <table>
                    <tr>
                        <th>File Size</th>
                        <th>Read Speedup</th>
                        <th>Write Speedup</th>
                    </tr>
""".format(charts['io'])

        # Add I/O optimization table rows
        for i, size in enumerate(data["io"]["sizes"]):
            # Format size with appropriate unit
            if size >= 1024*1024:
                size_str = f"{size/1024/1024:.1f} MB"
            elif size >= 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} bytes"

            html += f"""                    <tr>
                        <td>{size_str}</td>
                        <td>{data["io"]["read_speedups"][i]:.1f}x</td>
                        <td>{data["io"]["write_speedups"][i]:.1f}x</td>
                    </tr>
"""

        html += """                </table>
            </div>

            <div class="card">
                <h2>Database Optimization</h2>
                <p>Improving database operations with connection pooling, query optimization, and caching.</p>
                <div class="chart-container">
                    <img src="{0}" alt="Database Optimization" class="chart">
                </div>
                <table>
                    <tr>
                        <th>Record Count</th>
                        <th>Query Speedup</th>
                        <th>Insert Speedup</th>
                    </tr>
""".format(charts['database'])

        # Add database optimization table rows
        for i, count in enumerate(data["database"]["record_counts"]):
            html += f"""                    <tr>
                        <td>{count:,}</td>
                        <td>{data["database"]["query_speedups"][i]:.1f}x</td>
                        <td>{data["database"]["insert_speedups"][i]:.1f}x</td>
                    </tr>
"""

        html += """                </table>
            </div>
        </div>

        <div class="system-components">
            <h2>Performance Optimization System Components</h2>

            <div class="component">
                <h3>Core Performance Optimizer (performance_optimizer.py)</h3>
                <p>
                    The central optimization engine with specialized optimizers for different performance aspects:
                    memory usage, CPU utilization, I/O operations, network communication, database operations,
                    and distributed computing.
                </p>
                <div class="code-example">
<pre>
# Import the performance optimizer
from performance_optimizer import PerformanceOptimizer

# Create an optimizer instance
optimizer = PerformanceOptimizer()

# Use optimization decorators for functions
@optimizer.timed_function
def my_function():
    # Your code here
    pass

# Use optimization context managers
with optimizer.measure_time("operation_name"):
    # Your code here
    pass
</pre>
                </div>
            </div>

            <div class="component">
                <h3>Benchmark Tool (optimization_benchmark.py)</h3>
                <p>
                    A comprehensive benchmarking utility that measures and compares the performance impact
                    of various optimization strategies across different data sizes and workloads.
                </p>
                <div class="code-example">
<pre>
# Run all benchmarks
python optimization_benchmark.py --all

# Run specific benchmark categories
python optimization_benchmark.py --memory --cpu

# Generate visualization plots
python optimization_benchmark.py --all --plot
</pre>
                </div>
            </div>

            <div class="component">
                <h3>Performance Profiler (performance_profiler.py)</h3>
                <p>
                    A profiling tool to identify performance bottlenecks in existing code, analyze
                    function execution times, and provide recommendations for optimization.
                </p>
                <div class="code-example">
<pre>
# Profile a specific module
python performance_profiler.py --module my_module

# Profile a specific script
python performance_profiler.py --script my_script.py --args arg1 arg2

# Generate detailed HTML report
python performance_profiler.py --script my_script.py --report
</pre>
                </div>
            </div>

            <div class="component">
                <h3>Visualization Tool (performance_visualizer.py)</h3>
                <p>
                    A visualization utility that generates insightful charts and reports from benchmark
                    and profiler results for easy interpretation and analysis.
                </p>
                <div class="code-example">
<pre>
# Visualize benchmark results
python performance_visualizer.py --benchmark benchmark_results.json

# Visualize profiler results
python performance_visualizer.py --profiler profiler_results.json

# Generate HTML report with all visualizations
python performance_visualizer.py --benchmark results.json --profiler profile.json --report
</pre>
                </div>
            </div>

            <div class="component">
                <h3>Command-Line Interface (performance_tools.py)</h3>
                <p>
                    A unified command-line interface for all performance tools, providing commands for
                    benchmarking, profiling, visualization, and configuration management.
                </p>
                <div class="code-example">
<pre>
# Run all benchmarks and visualize results
python performance_tools.py benchmark --all --visualize

# Profile a specific module
python performance_tools.py profile --module my_module

# Update optimization configuration
python performance_tools.py config --enable-memory true --optimization-level aggressive
</pre>
                </div>
            </div>
        </div>
    </div>

    <footer>
        <div class="container">
            <p>Performance Optimization System for Negative Space Imaging Project</p>
            <p>© 2025 Negative Space Imaging, Inc. | Author: Stephen Bilodeau</p>
        </div>
    </footer>
</body>
</html>
"""

        return html


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Performance Optimization Dashboard Generator"
    )

    parser.add_argument(
        "--output",
        help="Output directory for the dashboard",
        default="performance_dashboard"
    )
    parser.add_argument(
        "--title",
        help="Dashboard title",
        default="Performance Optimization Dashboard"
    )

    args = parser.parse_args()

    # Generate dashboard
    generator = DashboardGenerator(output_dir=args.output)
    dashboard_path = generator.generate_dashboard(title=args.title)

    console.print(f"[bold green]Dashboard generated at: {dashboard_path}[/bold green]")
    console.print("[bold]Open the HTML file in a web browser to view the dashboard[/bold]")


if __name__ == "__main__":
    main()
