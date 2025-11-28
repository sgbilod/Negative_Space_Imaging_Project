#!/usr/bin/env python
"""
Comprehensive Test Runner for Negative Space Imaging Project.

This script provides a unified interface for discovering, executing,
and reporting on all tests across the project.

Features:
- Discovers all tests across the project (tests/ directory and root test files)
- Runs tests in parallel where possible
- Generates coverage reports
- Outputs results in multiple formats (console, HTML, XML)
- Supports filtering by markers, keywords, and paths

Usage:
    python test_runner.py                    # Run all tests
    python test_runner.py --unit             # Run only unit tests
    python test_runner.py --integration      # Run only integration tests
    python test_runner.py --coverage         # Run with coverage report
    python test_runner.py --parallel         # Run tests in parallel
    python test_runner.py --html             # Generate HTML report
    python test_runner.py --xml              # Generate XML report (JUnit format)
"""

import argparse
import subprocess
import sys
import os
import time
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime


class TestRunner:
    """Comprehensive test runner for the Negative Space Imaging Project."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the test runner.

        Args:
            project_root: Root directory of the project. Defaults to current directory.
        """
        self.project_root = project_root or Path.cwd()
        self.tests_dir = self.project_root / "tests"
        self.results: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def discover_tests(self) -> List[Path]:
        """Discover all test files in the project.

        Returns:
            List of paths to test files.
        """
        test_files = []

        # Find tests in tests/ directory
        if self.tests_dir.exists():
            test_files.extend(self.tests_dir.glob("test_*.py"))

        # Find test files in root directory (excluding empty files)
        for test_file in self.project_root.glob("test_*.py"):
            if test_file.stat().st_size > 0 and test_file.name != "test_runner.py":
                test_files.append(test_file)

        return sorted(test_files)

    def build_pytest_command(
        self,
        markers: Optional[List[str]] = None,
        keywords: Optional[str] = None,
        coverage: bool = False,
        parallel: bool = False,
        html_report: bool = False,
        xml_report: bool = False,
        verbose: bool = True,
        paths: Optional[List[str]] = None,
        extra_args: Optional[List[str]] = None
    ) -> List[str]:
        """Build the pytest command with specified options.

        Args:
            markers: List of pytest markers to filter tests.
            keywords: Keyword expression to filter tests.
            coverage: Enable coverage reporting.
            parallel: Run tests in parallel using pytest-xdist.
            html_report: Generate HTML coverage report.
            xml_report: Generate XML test report (JUnit format).
            verbose: Enable verbose output.
            paths: Specific test paths to run.
            extra_args: Additional arguments to pass to pytest.

        Returns:
            List of command arguments.
        """
        cmd = [sys.executable, "-m", "pytest"]

        # Add test paths
        if paths:
            cmd.extend(paths)
        else:
            # Include both tests directory and root test files
            cmd.append(str(self.tests_dir))
            for test_file in self.project_root.glob("test_*.py"):
                if test_file.stat().st_size > 0 and test_file.name != "test_runner.py":
                    cmd.append(str(test_file))

        # Verbose output
        if verbose:
            cmd.append("-v")

        # Add markers
        if markers:
            marker_expr = " or ".join(markers)
            cmd.extend(["-m", marker_expr])

        # Add keywords
        if keywords:
            cmd.extend(["-k", keywords])

        # Coverage options
        if coverage:
            cmd.extend([
                "--cov=sovereign",
                "--cov=quantum",
                "--cov=negative_space_analysis",
                "--cov-report=term-missing"
            ])
            if html_report:
                cmd.append("--cov-report=html:htmlcov")
            cmd.append("--cov-report=json:coverage.json")

        # Parallel execution
        if parallel:
            cmd.extend(["-n", "auto"])

        # XML report for CI
        if xml_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cmd.append(f"--junitxml=test-results-{timestamp}.xml")

        # Duration reporting
        cmd.append("--durations=10")

        # Extra arguments
        if extra_args:
            cmd.extend(extra_args)

        return cmd

    def run_tests(
        self,
        markers: Optional[List[str]] = None,
        keywords: Optional[str] = None,
        coverage: bool = False,
        parallel: bool = False,
        html_report: bool = False,
        xml_report: bool = False,
        verbose: bool = True,
        paths: Optional[List[str]] = None,
        extra_args: Optional[List[str]] = None
    ) -> int:
        """Run tests with specified options.

        Args:
            markers: List of pytest markers to filter tests.
            keywords: Keyword expression to filter tests.
            coverage: Enable coverage reporting.
            parallel: Run tests in parallel.
            html_report: Generate HTML coverage report.
            xml_report: Generate XML test report.
            verbose: Enable verbose output.
            paths: Specific test paths to run.
            extra_args: Additional arguments to pass to pytest.

        Returns:
            Exit code from pytest.
        """
        self.start_time = time.time()

        cmd = self.build_pytest_command(
            markers=markers,
            keywords=keywords,
            coverage=coverage,
            parallel=parallel,
            html_report=html_report,
            xml_report=xml_report,
            verbose=verbose,
            paths=paths,
            extra_args=extra_args
        )

        print("=" * 60)
        print("NEGATIVE SPACE IMAGING PROJECT - TEST RUNNER")
        print("=" * 60)
        print(f"Running: {' '.join(cmd)}")
        print("=" * 60)

        # Run pytest
        result = subprocess.run(
            cmd,
            cwd=str(self.project_root),
            env={**os.environ, "PYTHONPATH": str(self.project_root)}
        )

        self.end_time = time.time()
        self._record_results(result.returncode)

        return result.returncode

    def _record_results(self, return_code: int) -> None:
        """Record test results.

        Args:
            return_code: Exit code from pytest.
        """
        duration = (self.end_time - self.start_time) if self.start_time and self.end_time else 0

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration, 2),
            "return_code": return_code,
            "success": return_code == 0,
            "test_files_discovered": len(self.discover_tests())
        }

        # Save results to JSON
        results_file = self.project_root / "test_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def print_summary(self) -> None:
        """Print a summary of test execution."""
        print("\n" + "=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60)

        if self.results:
            print(f"Timestamp: {self.results.get('timestamp', 'N/A')}")
            print(f"Duration: {self.results.get('duration_seconds', 0):.2f} seconds")
            print(f"Test Files: {self.results.get('test_files_discovered', 0)}")
            print(f"Status: {'PASSED' if self.results.get('success') else 'FAILED'}")
        else:
            print("No results available.")

        print("=" * 60)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Runner for Negative Space Imaging Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_runner.py                     # Run all tests
    python test_runner.py --unit              # Run only unit tests
    python test_runner.py --integration       # Run only integration tests
    python test_runner.py --coverage --html   # Run with HTML coverage report
    python test_runner.py --parallel          # Run tests in parallel
    python test_runner.py -k "test_image"     # Run tests matching keyword
        """
    )

    # Test selection options
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests (marked with @pytest.mark.unit)"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests (marked with @pytest.mark.integration)"
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run only performance tests (marked with @pytest.mark.performance)"
    )
    parser.add_argument(
        "--security",
        action="store_true",
        help="Run only security tests (marked with @pytest.mark.security)"
    )
    parser.add_argument(
        "-m", "--markers",
        type=str,
        help="Custom marker expression (e.g., 'unit or integration')"
    )
    parser.add_argument(
        "-k", "--keyword",
        type=str,
        help="Only run tests matching the given keyword expression"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Specific test paths to run"
    )

    # Output options
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage reporting"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        help="Generate XML test report (JUnit format)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Disable verbose output"
    )

    # Execution options
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel using pytest-xdist"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all discovered test files without running them"
    )

    args = parser.parse_args()

    # Initialize runner
    runner = TestRunner()

    # List mode
    if args.list:
        print("Discovered test files:")
        for test_file in runner.discover_tests():
            print(f"  - {test_file}")
        return 0

    # Build markers list
    markers = []
    if args.unit:
        markers.append("unit")
    if args.integration:
        markers.append("integration")
    if args.performance:
        markers.append("performance")
    if args.security:
        markers.append("security")
    if args.markers:
        markers.extend(args.markers.split())

    # Run tests
    exit_code = runner.run_tests(
        markers=markers if markers else None,
        keywords=args.keyword,
        coverage=args.coverage,
        parallel=args.parallel,
        html_report=args.html,
        xml_report=args.xml,
        verbose=not args.quiet,
        paths=args.paths if args.paths else None
    )

    # Print summary
    runner.print_summary()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
