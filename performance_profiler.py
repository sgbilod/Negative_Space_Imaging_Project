#!/usr/bin/env python3
"""
Performance Profiler for Negative Space Imaging Project

This script provides a comprehensive profiling utility for identifying
performance bottlenecks in the codebase and suggesting optimizations.

Author: Stephen Bilodeau
Copyright: © 2025 Negative Space Imaging, Inc.
"""

import os
import sys
import time
import json
import cProfile
import pstats
import io
import logging
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("profiler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("profiler")

# Set up console output
console = Console()


class PerformanceProfiler:
    """Performance profiler for Python code in the Negative Space Imaging Project."""

    def __init__(self, target_module=None, output_dir="profiler_results"):
        """
        Initialize the profiler.

        Args:
            target_module: The module to profile (either module name or file path)
            output_dir: Directory to store profiling results
        """
        self.target_module = target_module
        self.output_dir = output_dir
        self.results = {}

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def profile_function(self, func, *args, **kwargs):
        """
        Profile a single function.

        Args:
            func: The function to profile
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Profiling statistics and result of the function
        """
        # Set up profiler
        profiler = cProfile.Profile()

        # Profile function execution
        profiler.enable()
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        profiler.disable()

        # Process profiling results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Print top 20 functions by cumulative time

        # Extract stats
        stats_text = s.getvalue()

        # Get function call statistics
        function_stats = []
        for line in stats_text.split('\n'):
            if line and not line.startswith('ncalls') and not line.startswith('Ordered by'):
                if ':' in line and '/' in line:  # Only process lines with file info
                    function_stats.append(line.strip())

        # Return profiling information and function result
        return {
            "execution_time": execution_time,
            "profiler_stats": stats_text,
            "function_stats": function_stats,
            "timestamp": datetime.now().isoformat()
        }, result

    def profile_module(self, module_name=None):
        """
        Profile an entire module.

        Args:
            module_name: The name of the module to profile (if different from self.target_module)

        Returns:
            Profiling statistics for the module
        """
        target = module_name or self.target_module

        if target is None:
            raise ValueError("No target module specified for profiling")

        # Handle module name or file path
        if target.endswith('.py'):
            # It's a file path
            module_path = os.path.abspath(target)
            module_name = os.path.basename(target).replace('.py', '')

            # Add directory to Python path if needed
            module_dir = os.path.dirname(module_path)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
        else:
            # It's a module name
            module_name = target

        try:
            # Import the module
            module = __import__(module_name)

            # Get all functions and methods from the module
            functions = []
            for name in dir(module):
                attr = getattr(module, name)
                if callable(attr) and not name.startswith('_'):
                    functions.append((name, attr))

            # Profile each function
            module_results = {}
            for name, func in functions:
                try:
                    console.print(f"Profiling function: [cyan]{name}[/cyan]")
                    stats, _ = self.profile_function(func)
                    module_results[name] = stats
                except Exception as e:
                    console.print(f"[red]Error profiling {name}: {str(e)}[/red]")
                    module_results[name] = {"error": str(e)}

            # Save results
            self.results[module_name] = module_results

            # Write results to file
            results_file = os.path.join(self.output_dir, f"{module_name}_profile.json")
            with open(results_file, 'w') as f:
                json.dump(module_results, f, indent=2)

            return module_results

        except ImportError as e:
            console.print(f"[red]Error importing module {module_name}: {str(e)}[/red]")
            return {"error": str(e)}

    def profile_script(self, script_path, args=None):
        """
        Profile a Python script by running it directly.

        Args:
            script_path: Path to the Python script
            args: List of command-line arguments to pass to the script

        Returns:
            Profiling statistics for the script
        """
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Set up profiler
        profiler = cProfile.Profile()

        # Prepare script arguments
        script_args = [script_path]
        if args:
            script_args.extend(args)

        # Save original sys.argv
        original_argv = sys.argv.copy()

        try:
            # Replace sys.argv with our args
            sys.argv = script_args

            # Set up profiling context
            script_locals = {'__name__': '__main__'}
            script_globals = globals().copy()

            # Read script content
            with open(script_path, 'r') as f:
                script_content = f.read()

            # Profile script execution
            profiler.enable()
            start_time = time.time()

            # Execute the script in the profiling context
            exec(compile(script_content, script_path, 'exec'), script_globals, script_locals)

            execution_time = time.time() - start_time
            profiler.disable()

            # Process profiling results
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(30)  # Print top 30 functions by cumulative time

            # Extract stats
            stats_text = s.getvalue()

            # Get function call statistics
            function_stats = []
            for line in stats_text.split('\n'):
                if line and not line.startswith('ncalls') and not line.startswith('Ordered by'):
                    if ':' in line and '/' in line:  # Only process lines with file info
                        function_stats.append(line.strip())

            # Generate results
            script_name = os.path.basename(script_path)
            results = {
                "execution_time": execution_time,
                "profiler_stats": stats_text,
                "function_stats": function_stats,
                "timestamp": datetime.now().isoformat(),
                "args": args
            }

            # Save results
            self.results[script_name] = results

            # Write results to file
            results_file = os.path.join(self.output_dir, f"{script_name}_profile.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            return results

        finally:
            # Restore original sys.argv
            sys.argv = original_argv

    def analyze_hotspots(self, profile_data):
        """
        Analyze profiling data to identify performance hotspots.

        Args:
            profile_data: Profiling data to analyze

        Returns:
            List of identified hotspots with recommendations
        """
        hotspots = []

        # Parse the profiler stats
        if "function_stats" not in profile_data:
            return []

        for stat in profile_data["function_stats"][:10]:  # Top 10 functions by time
            parts = stat.split()

            # Try to extract information from the line
            try:
                # This is a heuristic to parse the profiler output lines
                # Format can vary, but we try to extract the essentials
                time_info = [p for p in parts if ":" in p and p[0].isdigit()]
                time_value = float(time_info[0].split(":")[0]) if time_info else 0

                function_info = [p for p in parts if "(" in p and ")" in p]
                function_name = function_info[0] if function_info else "Unknown"

                file_info = [p for p in parts if "/" in p or "\\" in p]
                file_name = file_info[0] if file_info else "Unknown"

                # If time is significant, add to hotspots
                if time_value > 0.1:  # Consider functions taking more than 0.1s
                    hotspot = {
                        "function": function_name,
                        "file": file_name,
                        "time": time_value,
                        "recommendations": self._generate_recommendations(function_name, time_value)
                    }
                    hotspots.append(hotspot)
            except Exception as e:
                logger.warning(f"Error parsing profiler stat line: {str(e)}")
                continue

        return hotspots

    def _generate_recommendations(self, function_name, execution_time):
        """
        Generate optimization recommendations based on function name and execution time.

        Args:
            function_name: Name of the function
            execution_time: Execution time in seconds

        Returns:
            List of recommendations
        """
        recommendations = []

        # Generic recommendations based on function name patterns
        if "load" in function_name.lower() or "read" in function_name.lower():
            recommendations.append("Consider caching results to avoid repeated I/O operations")
            recommendations.append("Use memory-mapped files for large datasets")

        if "process" in function_name.lower() or "calculate" in function_name.lower():
            recommendations.append("Vectorize calculations using NumPy where possible")
            recommendations.append("Consider parallel processing for CPU-intensive operations")

        if "query" in function_name.lower() or "database" in function_name.lower():
            recommendations.append("Optimize database queries and consider indexing")
            recommendations.append("Use connection pooling for database operations")

        if "render" in function_name.lower() or "display" in function_name.lower():
            recommendations.append("Cache rendered results where appropriate")
            recommendations.append("Consider using GPU acceleration for rendering operations")

        # Recommendations based on execution time
        if execution_time > 1.0:
            recommendations.append("This function is a critical performance bottleneck")
            recommendations.append("Consider rewriting in Cython or using C extensions")
        elif execution_time > 0.5:
            recommendations.append("Function is relatively slow and should be optimized")
            recommendations.append("Profile this function in isolation to identify specific bottlenecks")

        # Add a fallback recommendation if none were generated
        if not recommendations:
            recommendations.append("Review function implementation for optimization opportunities")

        return recommendations

    def display_profile_results(self, profile_data, detailed=False):
        """
        Display profiling results in a readable format.

        Args:
            profile_data: Profiling data to display
            detailed: Whether to show detailed statistics
        """
        # Display execution time
        console.print(Panel(
            f"Execution Time: [bold]{profile_data.get('execution_time', 'N/A'):.4f}[/bold] seconds",
            title="Performance Profile",
            expand=False
        ))

        # Display hotspots
        hotspots = self.analyze_hotspots(profile_data)

        if hotspots:
            hotspot_table = Table(title="Performance Hotspots")
            hotspot_table.add_column("Function", style="cyan")
            hotspot_table.add_column("File", style="green")
            hotspot_table.add_column("Time (s)", style="yellow")
            hotspot_table.add_column("Recommendations", style="magenta")

            for hotspot in hotspots:
                recommendations = "\n".join([f"• {rec}" for rec in hotspot["recommendations"]])
                hotspot_table.add_row(
                    hotspot["function"],
                    hotspot["file"],
                    f"{hotspot['time']:.4f}",
                    recommendations
                )

            console.print(hotspot_table)

        # Display detailed profiler output if requested
        if detailed and "profiler_stats" in profile_data:
            console.print("\n[bold]Detailed Profile:[/bold]")
            syntax = Syntax(
                profile_data["profiler_stats"],
                "python",
                theme="monokai",
                line_numbers=True,
                word_wrap=True
            )
            console.print(syntax)

    def generate_report(self, output_file=None):
        """
        Generate a comprehensive HTML report of all profiling results.

        Args:
            output_file: Path to save the HTML report (defaults to profiler_report.html in output_dir)
        """
        if not output_file:
            output_file = os.path.join(self.output_dir, "profiler_report.html")

        # Generate HTML report
        with open(output_file, 'w') as f:
            # Write HTML header
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Performance Profiler Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2, h3 { color: #333; }
        .container { max-width: 1200px; margin: 0 auto; }
        .section { margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
        .hotspot { background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 5px solid #ff6b6b; }
        .recommendation { color: #0066cc; margin: 5px 0 5px 20px; }
        .time { font-weight: bold; color: #e74c3c; }
        .function { font-weight: bold; color: #2980b9; }
        .file { color: #27ae60; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Performance Profiler Report</h1>
        <p>Generated on: %s</p>
""" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # Write summary section
            f.write("""
        <div class="section">
            <h2>Summary</h2>
            <p>Total profiled items: %d</p>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Execution Time (s)</th>
                    <th>Hotspots</th>
                </tr>
""" % len(self.results))

            # Add summary rows
            for name, data in self.results.items():
                execution_time = data.get("execution_time", "N/A")
                if isinstance(execution_time, (int, float)):
                    execution_time = f"{execution_time:.4f}"

                hotspots = len(self.analyze_hotspots(data)) if "function_stats" in data else "N/A"

                f.write(f"""
                <tr>
                    <td>{name}</td>
                    <td>{execution_time}</td>
                    <td>{hotspots}</td>
                </tr>""")

            f.write("""
            </table>
        </div>
""")

            # Write detailed sections for each profiled item
            for name, data in self.results.items():
                f.write(f"""
        <div class="section">
            <h2>Profile: {name}</h2>
""")

                if "error" in data:
                    f.write(f"""
            <div style="color: red; font-weight: bold;">Error: {data["error"]}</div>
""")
                    continue

                execution_time = data.get("execution_time", "N/A")
                if isinstance(execution_time, (int, float)):
                    execution_time = f"{execution_time:.4f}"

                f.write(f"""
            <p>Execution Time: <span class="time">{execution_time} seconds</span></p>
            <p>Timestamp: {data.get("timestamp", "N/A")}</p>
""")

                # Write hotspots
                hotspots = self.analyze_hotspots(data)
                if hotspots:
                    f.write("""
            <h3>Performance Hotspots</h3>
""")

                    for hotspot in hotspots:
                        f.write(f"""
            <div class="hotspot">
                <div><span class="function">{hotspot["function"]}</span> in <span class="file">{hotspot["file"]}</span></div>
                <div>Time: <span class="time">{hotspot["time"]:.4f} seconds</span></div>
                <div>Recommendations:</div>
""")

                        for rec in hotspot["recommendations"]:
                            f.write(f"""
                <div class="recommendation">• {rec}</div>
""")

                        f.write("""
            </div>
""")

                # Write detailed profile if available
                if "profiler_stats" in data:
                    f.write("""
            <h3>Detailed Profile</h3>
            <pre>%s</pre>
""" % data["profiler_stats"])

                f.write("""
        </div>
""")

            # Write HTML footer
            f.write("""
    </div>
</body>
</html>
""")

        console.print(f"[bold green]Report generated: {output_file}[/bold green]")
        return output_file


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Performance Profiler for Negative Space Imaging Project")

    # Add command line arguments
    parser.add_argument("--module", "-m", help="Module to profile")
    parser.add_argument("--script", "-s", help="Script to profile")
    parser.add_argument("--args", "-a", nargs="+", help="Arguments to pass to the script")
    parser.add_argument("--output", "-o", help="Output directory for profiling results")
    parser.add_argument("--report", "-r", help="Generate HTML report and save to specified path")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed profiling information")

    args = parser.parse_args()

    # Initialize profiler
    output_dir = args.output or "profiler_results"
    profiler = PerformanceProfiler(output_dir=output_dir)

    # Perform profiling
    if args.module:
        console.print(f"[bold]Profiling module: {args.module}[/bold]")
        results = profiler.profile_module(args.module)

        # Display results
        for func_name, func_results in results.items():
            console.print(f"\n[bold cyan]Function: {func_name}[/bold cyan]")
            profiler.display_profile_results(func_results, detailed=args.detailed)

    elif args.script:
        console.print(f"[bold]Profiling script: {args.script}[/bold]")
        script_args = args.args or []
        results = profiler.profile_script(args.script, script_args)

        # Display results
        profiler.display_profile_results(results, detailed=args.detailed)

    else:
        console.print("[yellow]No module or script specified. Use --module or --script to specify a target.[/yellow]")
        return

    # Generate report if requested
    if args.report or (not args.report and (args.module or args.script)):
        report_path = args.report or os.path.join(output_dir, "profiler_report.html")
        profiler.generate_report(report_path)


if __name__ == "__main__":
    main()
