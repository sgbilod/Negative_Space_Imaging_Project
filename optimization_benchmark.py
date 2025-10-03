#!/usr/bin/env python3
"""
Performance Optimization Benchmark Tool for Negative Space Imaging Project

This script runs a series of benchmarks to test and demonstrate the performance
optimizations implemented in the performance_optimizer module.

Author: Stephen Bilodeau
Copyright: © 2025 Negative Space Imaging, Inc.
"""

import os
import sys
import time
import json
import random
import argparse
import logging
import numpy as np
from typing import Dict, List, Any, Callable
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import matplotlib.pyplot as plt

# Import the performance optimizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from performance_optimizer import (
    PerformanceOptimizer, MemoryOptimizer, CPUOptimizer,
    IOOptimizer, NetworkOptimizer, DatabaseOptimizer,
    DistributedOptimizer, measure_time, timed_function
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("benchmark")

# Set up console output
console = Console()


class Benchmark:
    """Base class for benchmarks."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results = {}

    def setup(self):
        """Set up the benchmark."""
        pass

    def run(self):
        """Run the benchmark."""
        raise NotImplementedError("Subclasses must implement run method")

    def cleanup(self):
        """Clean up after the benchmark."""
        pass


class MemoryBenchmark(Benchmark):
    """Benchmark for memory optimizations."""

    def __init__(self):
        super().__init__("Memory Optimization", "Tests memory optimization techniques")
        self.data_sizes = [100, 1000, 10000, 100000, 1000000]
        self.memory_optimizer = MemoryOptimizer()

    def run(self):
        """Run memory optimization benchmarks."""
        console.print("[bold blue]Running Memory Optimization Benchmarks[/bold blue]")

        for size in self.data_sizes:
            # Create a large array
            console.print(f"Testing with array size: {size}")

            # Test with optimizations disabled
            self.memory_optimizer.enabled = False
            with measure_time(f"memory_unoptimized_{size}"):
                array = np.random.randint(0, 100, size=size)
                result = self._process_array(array)

            # Test with optimizations enabled
            self.memory_optimizer.enabled = True
            with measure_time(f"memory_optimized_{size}"):
                array = np.random.randint(0, 100, size=size)
                optimized_array = self.memory_optimizer.optimize_array(array)
                result = self._process_array(optimized_array)

            # Calculate memory savings
            original_size = array.nbytes
            optimized_size = optimized_array.nbytes
            savings = original_size - optimized_size
            savings_percent = (savings / original_size) * 100 if original_size > 0 else 0

            console.print(f"  Original size: {original_size} bytes")
            console.print(f"  Optimized size: {optimized_size} bytes")
            console.print(f"  Memory savings: {savings} bytes ({savings_percent:.2f}%)")

            self.results[f"size_{size}"] = {
                "original_size": original_size,
                "optimized_size": optimized_size,
                "savings": savings,
                "savings_percent": savings_percent
            }

    def _process_array(self, array):
        """Process an array with some operations."""
        # Simple array operations
        result = array + 10
        result = result * 2
        result = np.sqrt(result)
        return result


class CPUBenchmark(Benchmark):
    """Benchmark for CPU optimizations."""

    def __init__(self):
        super().__init__("CPU Optimization", "Tests CPU optimization techniques")
        self.data_sizes = [1000, 10000, 100000, 1000000]
        self.cpu_optimizer = CPUOptimizer()

    def run(self):
        """Run CPU optimization benchmarks."""
        console.print("[bold blue]Running CPU Optimization Benchmarks[/bold blue]")

        for size in self.data_sizes:
            # Generate test data
            console.print(f"Testing with data size: {size}")
            data = [random.random() for _ in range(size)]

            # Test without parallelization
            self.cpu_optimizer.enabled = False
            start_time = time.time()
            result1 = list(map(self._compute_value, data))
            unoptimized_time = time.time() - start_time

            # Test with parallelization
            self.cpu_optimizer.enabled = True
            start_time = time.time()
            result2 = self.cpu_optimizer.parallel_map(self._compute_value, data)
            optimized_time = time.time() - start_time

            # Calculate speedup
            speedup = unoptimized_time / optimized_time if optimized_time > 0 else 0

            console.print(f"  Unoptimized time: {unoptimized_time:.6f} seconds")
            console.print(f"  Optimized time: {optimized_time:.6f} seconds")
            console.print(f"  Speedup: {speedup:.2f}x")

            self.results[f"size_{size}"] = {
                "unoptimized_time": unoptimized_time,
                "optimized_time": optimized_time,
                "speedup": speedup
            }

            # Verify results are the same
            assert len(result1) == len(result2), "Results have different lengths"
            for a, b in zip(result1[:100], result2[:100]):  # Check first 100 items
                assert abs(a - b) < 1e-10, "Results are different"

    def _compute_value(self, x):
        """Compute a value (CPU-intensive operation)."""
        # Simulate a complex computation
        result = 0
        for _ in range(1000):
            result += np.sin(x) * np.cos(x) / (1 + np.abs(x))
        return result


class IOBenchmark(Benchmark):
    """Benchmark for I/O optimizations."""

    def __init__(self):
        super().__init__("I/O Optimization", "Tests I/O optimization techniques")
        self.file_sizes = [1024, 10240, 102400, 1024000]
        self.io_optimizer = IOOptimizer()
        self.test_dir = "benchmark_data"

    def setup(self):
        """Set up the benchmark."""
        # Create test directory
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def run(self):
        """Run I/O optimization benchmarks."""
        console.print("[bold blue]Running I/O Optimization Benchmarks[/bold blue]")

        for size in self.file_sizes:
            # Generate test data
            console.print(f"Testing with file size: {size} bytes")
            test_data = os.urandom(size)

            # Test file writing without optimization
            self.io_optimizer.enabled = False
            unoptimized_write_file = os.path.join(self.test_dir, f"unoptimized_{size}.dat")
            start_time = time.time()
            self.io_optimizer.write_file(unoptimized_write_file, test_data, binary=True)
            unoptimized_write_time = time.time() - start_time

            # Test file writing with optimization
            self.io_optimizer.enabled = True
            optimized_write_file = os.path.join(self.test_dir, f"optimized_{size}.dat")
            start_time = time.time()
            self.io_optimizer.write_file(optimized_write_file, test_data, binary=True)
            optimized_write_time = time.time() - start_time

            # Test file reading without optimization
            self.io_optimizer.enabled = False
            start_time = time.time()
            data1 = self.io_optimizer.read_file(unoptimized_write_file, binary=True)
            unoptimized_read_time = time.time() - start_time

            # Test file reading with optimization
            self.io_optimizer.enabled = True
            start_time = time.time()
            data2 = self.io_optimizer.read_file(optimized_write_file, binary=True)
            optimized_read_time = time.time() - start_time

            # Calculate speedups
            write_speedup = unoptimized_write_time / optimized_write_time if optimized_write_time > 0 else 0
            read_speedup = unoptimized_read_time / optimized_read_time if optimized_read_time > 0 else 0

            console.print(f"  Write: Unoptimized: {unoptimized_write_time:.6f}s, Optimized: {optimized_write_time:.6f}s, Speedup: {write_speedup:.2f}x")
            console.print(f"  Read: Unoptimized: {unoptimized_read_time:.6f}s, Optimized: {optimized_read_time:.6f}s, Speedup: {read_speedup:.2f}x")

            self.results[f"size_{size}"] = {
                "unoptimized_write_time": unoptimized_write_time,
                "optimized_write_time": optimized_write_time,
                "write_speedup": write_speedup,
                "unoptimized_read_time": unoptimized_read_time,
                "optimized_read_time": optimized_read_time,
                "read_speedup": read_speedup
            }

            # Verify data integrity
            assert data1 == data2, "Data integrity check failed"

    def cleanup(self):
        """Clean up after the benchmark."""
        # Remove test files
        for size in self.file_sizes:
            unoptimized_file = os.path.join(self.test_dir, f"unoptimized_{size}.dat")
            optimized_file = os.path.join(self.test_dir, f"optimized_{size}.dat")

            if os.path.exists(unoptimized_file):
                os.remove(unoptimized_file)

            if os.path.exists(optimized_file):
                os.remove(optimized_file)


class DatabaseBenchmark(Benchmark):
    """Benchmark for database optimizations."""

    def __init__(self):
        super().__init__("Database Optimization", "Tests database optimization techniques")
        self.db_optimizer = DatabaseOptimizer()
        self.record_counts = [100, 1000, 10000]
        self.db_file = os.path.join("benchmark_data", "test.db")

    def setup(self):
        """Set up the benchmark."""
        # Create test directory if it doesn't exist
        if not os.path.exists(os.path.dirname(self.db_file)):
            os.makedirs(os.path.dirname(self.db_file))

        # Set up SQLite database
        import sqlite3
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Create test table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL,
                data BLOB
            )
        ''')

        conn.commit()
        conn.close()

    def run(self):
        """Run database optimization benchmarks."""
        console.print("[bold blue]Running Database Optimization Benchmarks[/bold blue]")

        try:
            import sqlite3

            for count in self.record_counts:
                console.print(f"Testing with {count} records")

                # Generate test data
                test_data = []
                for i in range(count):
                    test_data.append((
                        i,
                        f"name_{i}",
                        random.random() * 1000,
                        os.urandom(100)
                    ))

                # Insert data without optimization
                self.db_optimizer.enabled = False
                conn = sqlite3.connect(self.db_file)
                start_time = time.time()
                self._insert_records(conn, test_data, use_optimization=False)
                unoptimized_insert_time = time.time() - start_time
                conn.close()

                # Clear database
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM test_table")
                conn.commit()
                conn.close()

                # Insert data with optimization
                self.db_optimizer.enabled = True
                conn_str = f"sqlite://{self.db_file}"
                conn = self.db_optimizer.get_connection(conn_str)
                start_time = time.time()
                self._insert_records(conn, test_data, use_optimization=True)
                optimized_insert_time = time.time() - start_time
                self.db_optimizer.release_connection(conn_str, conn)

                # Query data without optimization
                self.db_optimizer.enabled = False
                conn = sqlite3.connect(self.db_file)
                start_time = time.time()
                result1 = self._query_records(conn, use_optimization=False)
                unoptimized_query_time = time.time() - start_time
                conn.close()

                # Query data with optimization
                self.db_optimizer.enabled = True
                conn = self.db_optimizer.get_connection(conn_str)
                start_time = time.time()
                result2 = self._query_records(conn, use_optimization=True)
                optimized_query_time = time.time() - start_time
                self.db_optimizer.release_connection(conn_str, conn)

                # Calculate speedups
                insert_speedup = unoptimized_insert_time / optimized_insert_time if optimized_insert_time > 0 else 0
                query_speedup = unoptimized_query_time / optimized_query_time if optimized_query_time > 0 else 0

                console.print(f"  Insert: Unoptimized: {unoptimized_insert_time:.6f}s, Optimized: {optimized_insert_time:.6f}s, Speedup: {insert_speedup:.2f}x")
                console.print(f"  Query: Unoptimized: {unoptimized_query_time:.6f}s, Optimized: {optimized_query_time:.6f}s, Speedup: {query_speedup:.2f}x")

                self.results[f"count_{count}"] = {
                    "unoptimized_insert_time": unoptimized_insert_time,
                    "optimized_insert_time": optimized_insert_time,
                    "insert_speedup": insert_speedup,
                    "unoptimized_query_time": unoptimized_query_time,
                    "optimized_query_time": optimized_query_time,
                    "query_speedup": query_speedup
                }

                # Verify result integrity
                assert len(result1) == len(result2), "Result count mismatch"

                # Clear database for next test
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM test_table")
                conn.commit()
                conn.close()

        except ImportError:
            console.print("[bold red]SQLite3 module not available, skipping database benchmarks[/bold red]")

    def _insert_records(self, conn, records, use_optimization=False):
        """Insert records into the database."""
        if use_optimization:
            # Use optimized query execution
            self.db_optimizer.execute_query(
                conn,
                "INSERT INTO test_table (id, name, value, data) VALUES (?, ?, ?, ?)",
                records[0]  # For prepared statement
            )

            # Batch insert remaining records
            for record in records[1:]:
                self.db_optimizer.execute_query(
                    conn,
                    "INSERT INTO test_table (id, name, value, data) VALUES (?, ?, ?, ?)",
                    record
                )
        else:
            cursor = conn.cursor()
            for record in records:
                cursor.execute(
                    "INSERT INTO test_table (id, name, value, data) VALUES (?, ?, ?, ?)",
                    record
                )
            conn.commit()

    def _query_records(self, conn, use_optimization=False):
        """Query records from the database."""
        if use_optimization:
            # Use optimized query execution
            result, _ = self.db_optimizer.execute_query(
                conn,
                "SELECT id, name, value FROM test_table WHERE value > ? ORDER BY value DESC",
                [0.0]
            )
            return result
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, value FROM test_table WHERE value > ? ORDER BY value DESC", (0.0,))
            return cursor.fetchall()

    def cleanup(self):
        """Clean up after the benchmark."""
        # Remove test database
        if os.path.exists(self.db_file):
            os.remove(self.db_file)


def run_benchmarks(args):
    """Run all selected benchmarks."""
    # Create benchmark instances
    benchmarks = {
        "memory": MemoryBenchmark(),
        "cpu": CPUBenchmark(),
        "io": IOBenchmark(),
        "database": DatabaseBenchmark()
    }

    # Filter benchmarks based on arguments
    selected_benchmarks = []
    if args.all:
        selected_benchmarks = list(benchmarks.values())
    else:
        if args.memory:
            selected_benchmarks.append(benchmarks["memory"])
        if args.cpu:
            selected_benchmarks.append(benchmarks["cpu"])
        if args.io:
            selected_benchmarks.append(benchmarks["io"])
        if args.database:
            selected_benchmarks.append(benchmarks["database"])

    if not selected_benchmarks:
        console.print("[bold yellow]No benchmarks selected. Use --all or specify individual benchmarks.[/bold yellow]")
        return

    # Run benchmarks
    results = {}

    with Progress() as progress:
        overall_task = progress.add_task("[green]Running benchmarks...", total=len(selected_benchmarks))

        for benchmark in selected_benchmarks:
            benchmark_task = progress.add_task(f"[cyan]{benchmark.name}...", total=3)

            # Setup
            progress.update(benchmark_task, description=f"[cyan]{benchmark.name} (setup)...", advance=0)
            benchmark.setup()
            progress.update(benchmark_task, advance=1)

            # Run
            progress.update(benchmark_task, description=f"[cyan]{benchmark.name} (running)...", advance=0)
            benchmark.run()
            progress.update(benchmark_task, advance=1)

            # Cleanup
            progress.update(benchmark_task, description=f"[cyan]{benchmark.name} (cleanup)...", advance=0)
            benchmark.cleanup()
            progress.update(benchmark_task, advance=1)

            # Store results
            results[benchmark.name] = benchmark.results

            progress.update(overall_task, advance=1)

    # Display summary table
    display_summary(results)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[bold green]Results saved to {args.output}[/bold green]")

    # Generate plots
    if args.plot:
        generate_plots(results)


def display_summary(results: Dict[str, Any]):
    """Display a summary of benchmark results."""
    console.print("\n[bold green]Benchmark Summary[/bold green]")

    for benchmark_name, benchmark_results in results.items():
        table = Table(title=benchmark_name)

        # Customize table based on benchmark type
        if "Memory Optimization" in benchmark_name:
            table.add_column("Data Size", style="cyan")
            table.add_column("Original Size (bytes)", style="magenta")
            table.add_column("Optimized Size (bytes)", style="magenta")
            table.add_column("Memory Savings", style="green")

            for key, data in benchmark_results.items():
                size = int(key.split("_")[1])
                savings_percent = data["savings_percent"]
                savings_text = f"{data['savings']} bytes ({savings_percent:.2f}%)"

                table.add_row(
                    f"{size:,}",
                    f"{data['original_size']:,}",
                    f"{data['optimized_size']:,}",
                    savings_text
                )

        elif "CPU Optimization" in benchmark_name:
            table.add_column("Data Size", style="cyan")
            table.add_column("Unoptimized Time (s)", style="magenta")
            table.add_column("Optimized Time (s)", style="magenta")
            table.add_column("Speedup", style="green")

            for key, data in benchmark_results.items():
                size = int(key.split("_")[1])

                table.add_row(
                    f"{size:,}",
                    f"{data['unoptimized_time']:.6f}",
                    f"{data['optimized_time']:.6f}",
                    f"{data['speedup']:.2f}x"
                )

        elif "I/O Optimization" in benchmark_name:
            table.add_column("File Size (bytes)", style="cyan")
            table.add_column("Operation", style="blue")
            table.add_column("Unoptimized Time (s)", style="magenta")
            table.add_column("Optimized Time (s)", style="magenta")
            table.add_column("Speedup", style="green")

            for key, data in benchmark_results.items():
                size = int(key.split("_")[1])

                table.add_row(
                    f"{size:,}",
                    "Write",
                    f"{data['unoptimized_write_time']:.6f}",
                    f"{data['optimized_write_time']:.6f}",
                    f"{data['write_speedup']:.2f}x"
                )

                table.add_row(
                    f"{size:,}",
                    "Read",
                    f"{data['unoptimized_read_time']:.6f}",
                    f"{data['optimized_read_time']:.6f}",
                    f"{data['read_speedup']:.2f}x"
                )

        elif "Database Optimization" in benchmark_name:
            table.add_column("Record Count", style="cyan")
            table.add_column("Operation", style="blue")
            table.add_column("Unoptimized Time (s)", style="magenta")
            table.add_column("Optimized Time (s)", style="magenta")
            table.add_column("Speedup", style="green")

            for key, data in benchmark_results.items():
                count = int(key.split("_")[1])

                table.add_row(
                    f"{count:,}",
                    "Insert",
                    f"{data['unoptimized_insert_time']:.6f}",
                    f"{data['optimized_insert_time']:.6f}",
                    f"{data['insert_speedup']:.2f}x"
                )

                table.add_row(
                    f"{count:,}",
                    "Query",
                    f"{data['unoptimized_query_time']:.6f}",
                    f"{data['optimized_query_time']:.6f}",
                    f"{data['query_speedup']:.2f}x"
                )

        console.print(table)
        console.print("")


def generate_plots(results: Dict[str, Any]):
    """Generate plots for benchmark results."""
    plot_dir = "benchmark_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for benchmark_name, benchmark_results in results.items():
        if "Memory Optimization" in benchmark_name:
            # Memory optimization plot
            sizes = []
            savings_pcts = []

            for key, data in benchmark_results.items():
                size = int(key.split("_")[1])
                sizes.append(size)
                savings_pcts.append(data["savings_percent"])

            plt.figure(figsize=(10, 6))
            plt.bar(range(len(sizes)), savings_pcts)
            plt.xticks(range(len(sizes)), [f"{size:,}" for size in sizes])
            plt.xlabel("Array Size")
            plt.ylabel("Memory Savings (%)")
            plt.title("Memory Optimization: Savings by Array Size")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "memory_optimization.png"))
            plt.close()

        elif "CPU Optimization" in benchmark_name:
            # CPU optimization plot
            sizes = []
            speedups = []

            for key, data in benchmark_results.items():
                size = int(key.split("_")[1])
                sizes.append(size)
                speedups.append(data["speedup"])

            plt.figure(figsize=(10, 6))
            plt.bar(range(len(sizes)), speedups)
            plt.xticks(range(len(sizes)), [f"{size:,}" for size in sizes])
            plt.xlabel("Data Size")
            plt.ylabel("Speedup Factor (x)")
            plt.title("CPU Optimization: Speedup by Data Size")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "cpu_optimization.png"))
            plt.close()

        elif "I/O Optimization" in benchmark_name:
            # I/O optimization plot
            sizes = []
            write_speedups = []
            read_speedups = []

            for key, data in benchmark_results.items():
                size = int(key.split("_")[1])
                sizes.append(size)
                write_speedups.append(data["write_speedup"])
                read_speedups.append(data["read_speedup"])

            plt.figure(figsize=(10, 6))
            x = range(len(sizes))
            width = 0.35

            plt.bar([i - width/2 for i in x], write_speedups, width, label='Write')
            plt.bar([i + width/2 for i in x], read_speedups, width, label='Read')

            plt.xlabel('File Size (bytes)')
            plt.ylabel('Speedup Factor (x)')
            plt.title('I/O Optimization: Speedup by File Size')
            plt.xticks(x, [f"{size:,}" for size in sizes])
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "io_optimization.png"))
            plt.close()

        elif "Database Optimization" in benchmark_name:
            # Database optimization plot
            counts = []
            insert_speedups = []
            query_speedups = []

            for key, data in benchmark_results.items():
                count = int(key.split("_")[1])
                counts.append(count)
                insert_speedups.append(data["insert_speedup"])
                query_speedups.append(data["query_speedup"])

            plt.figure(figsize=(10, 6))
            x = range(len(counts))
            width = 0.35

            plt.bar([i - width/2 for i in x], insert_speedups, width, label='Insert')
            plt.bar([i + width/2 for i in x], query_speedups, width, label='Query')

            plt.xlabel('Record Count')
            plt.ylabel('Speedup Factor (x)')
            plt.title('Database Optimization: Speedup by Record Count')
            plt.xticks(x, [f"{count:,}" for count in counts])
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "database_optimization.png"))
            plt.close()

    # Combined results plot
    plt.figure(figsize=(12, 8))

    # Get average speedups for each benchmark
    avg_speedups = {}

    for benchmark_name, benchmark_results in results.items():
        if "Memory Optimization" in benchmark_name:
            # Use memory savings percentage as "speedup"
            avg_savings = sum(data["savings_percent"] for data in benchmark_results.values()) / len(benchmark_results)
            avg_speedups["Memory"] = avg_savings / 100  # Convert to ratio for comparison

        elif "CPU Optimization" in benchmark_name:
            avg_speedup = sum(data["speedup"] for data in benchmark_results.values()) / len(benchmark_results)
            avg_speedups["CPU"] = avg_speedup

        elif "I/O Optimization" in benchmark_name:
            # Average of read and write speedups
            avg_speedup = sum((data["read_speedup"] + data["write_speedup"]) / 2 for data in benchmark_results.values()) / len(benchmark_results)
            avg_speedups["I/O"] = avg_speedup

        elif "Database Optimization" in benchmark_name:
            # Average of insert and query speedups
            avg_speedup = sum((data["insert_speedup"] + data["query_speedup"]) / 2 for data in benchmark_results.values()) / len(benchmark_results)
            avg_speedups["Database"] = avg_speedup

    # Plot average speedups
    categories = list(avg_speedups.keys())
    values = list(avg_speedups.values())

    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
    plt.bar(range(len(categories)), values, color=colors)
    plt.xticks(range(len(categories)), categories)
    plt.ylabel('Performance Improvement Factor')
    plt.title('Average Performance Improvement by Category')

    # Add values on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, f"{v:.2f}x", ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "overall_performance.png"))
    plt.close()

    console.print(f"[bold green]Plots saved to {plot_dir} directory[/bold green]")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Performance Optimization Benchmark")

    # Benchmark selection
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--memory", action="store_true", help="Run memory optimization benchmarks")
    parser.add_argument("--cpu", action="store_true", help="Run CPU optimization benchmarks")
    parser.add_argument("--io", action="store_true", help="Run I/O optimization benchmarks")
    parser.add_argument("--database", action="store_true", help="Run database optimization benchmarks")

    # Output options
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--plot", action="store_true", help="Generate plots of benchmark results")

    return parser.parse_args()


def main():
    """Main entry point."""
    console.print("[bold]Performance Optimization Benchmark Tool[/bold]")
    console.print("=======================================\n")

    args = parse_args()
    run_benchmarks(args)


if __name__ == "__main__":
    main()
