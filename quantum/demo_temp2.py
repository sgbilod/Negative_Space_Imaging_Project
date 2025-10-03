"""
Sovereign Quantum System Demo
Copyright (c) 2025 Stephen Bilodeau

This module provides a comprehensive demo of the Sovereign Quantum System,
including stability tests, visualization, and performance metrics.

Features:
    - Comprehensive stability testing
    - Real-time visualization
    - Performance metrics and statistics
    - Progress tracking and reporting
    - Configurable test parameters
    - Detailed error handling and reporting
    - Clean shutdown handling
"""

import argparse
import json
import logging
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
    MEMORY_TRACKING = True
except ImportError:
    MEMORY_TRACKING = False

import numpy as np
from quantum.sovereign import SovereignQuantumSystem

# ANSI color codes for terminal output
COLORS = {
    'RED': '\033[91m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'BLUE': '\033[94m',
    'CYAN': '\033[96m',
    'END': '\033[0m'
}


def color_text(text: str, color: str) -> str:
    """Wrap text in ANSI color codes."""
    return f"{COLORS[color]}{text}{COLORS['END']}"


def make_progress_bar(progress: float, width: int = 40) -> str:
    """Create a progress bar string."""
    filled = int(width * progress)
    bar = '=' * filled + '-' * (width - filled)
    return f"[{bar}] {progress*100:3.0f}%"


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    if not MEMORY_TRACKING:
        return {}

    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        'rss': mem_info.rss / (1024 * 1024),  # MB
        'vms': mem_info.vms / (1024 * 1024),  # MB
        'percent': process.memory_percent()
    }


def format_metric_box(metrics: Dict[str, Any], width: int = 48) -> None:
    """Format and print metrics in a boxed display."""
    print("─" * width)
    for name, value in metrics.items():
        if isinstance(value, bool):
            value_str = f"{str(value):>8}"
        elif isinstance(value, float):
            value_str = f"{value:>8.3f}"
        elif isinstance(value, (int, str)):
            value_str = f"{value:>8}"
        else:
            value_str = f"{str(value):>8}"
        print(f"│ {name:<16} {value_str}           │")
    print("─" * width)


def setup_logging(log_path: Optional[Path] = None) -> None:
    """Configure logging for the demo.

    Args:
        log_path: Path to write log file. If None, only console output used.
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_path is not None:
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )


def setup_signal_handlers(
    sovereign: Optional[SovereignQuantumSystem] = None
) -> None:
    """Set up signal handlers for clean program termination."""
    def signal_handler(signum: int, frame: Any) -> None:
        print("\nReceived termination signal. Cleaning up...")
        if sovereign is not None:
            try:
                sovereign.visualizer.cleanup()
            except Exception as e:
                print(f"Cleanup error: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the demo."""
    parser = argparse.ArgumentParser(
        description="Sovereign Quantum System Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    test_group = parser.add_argument_group("Test Execution")
    test_group.add_argument(
        "--pause",
        type=float,
        default=2.0,
        help="Pause duration between test cases (seconds)"
    )
    test_group.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running tests after errors"
    )

    display_group = parser.add_argument_group("Display Options")
    display_group.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization initialization"
    )
    display_group.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--log",
        type=Path,
        help="Path to write log file"
    )
    output_group.add_argument(
        "--save-results",
        type=Path,
        help="Save test results to JSON file"
    )

    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument(
        "--skip-memory",
        action="store_true",
        help="Skip memory usage tracking"
    )

    args = parser.parse_args()

    if args.log is not None:
        setup_logging(args.log)

    if args.skip_memory:
        global MEMORY_TRACKING
        MEMORY_TRACKING = False

    if args.no_color:
        for color in COLORS:
            COLORS[color] = ''

    return args


def execute_test_cases(
        test_cases: List[Tuple[str, np.ndarray]],
        args: argparse.Namespace,
        sovereign: SovereignQuantumSystem
) -> Dict[str, Any]:
    """Execute a series of quantum test cases."""
    results = []
    total_time = 0.0
    total_coherence = 0.0
    successful_tests = 0
    failed_tests = 0

    total_cases = len(test_cases)
    for idx, (case_name, test_region) in enumerate(test_cases, 1):
        progress = idx / total_cases
        print(f"\n{make_progress_bar(progress)}")
        case_status = f"Test Case {idx}/{total_cases}: {case_name}"
        print(color_text(case_status, "YELLOW"))

        try:
            # Track memory if enabled
            mem_before = get_memory_usage()

            # Execute quantum operation with timing
            start_time = time.time()
            result = sovereign.execute_quantum_operation(
                test_region,
                operation_type='standard'
            )
            exec_time = time.time() - start_time

            # Validate and get metrics
            validation = sovereign.validate_quantum_operation(result)
            coherence = result.get('coherence', {}).get('coherence', 0.0)
            state_valid = validation.get('state_valid', False)
            max_val = float(np.max(np.abs(test_region)))

            # Get final memory usage
            mem_after = get_memory_usage()

            # Create test result
            test_result = {
                'name': case_name,
                'coherence': coherence,
                'execution_time': exec_time,
                'state_valid': state_valid,
                'max_value': max_val,
                'success': state_valid
            }

            # Display metrics with clean formatting
            metrics = {
                "Coherence": color_text(f"{coherence:.3f}", "BLUE"),
                "Execution Time": f"{exec_time:.3f}s",
                "State Valid": color_text(
                    "Yes" if state_valid else "No",
                    "GREEN" if state_valid else "RED"
                ),
                "Max Value": f"{max_val:.2e}"
            }

            # Add memory metrics if tracking enabled
            if MEMORY_TRACKING:
                delta_mb = mem_after['rss'] - mem_before['rss']
                metrics["Memory Δ"] = f"{delta_mb:+.1f}MB"
                test_result['memory_delta'] = delta_mb

            format_metric_box(metrics)

            # Update visualization
            if not args.skip_viz:
                try:
                    sovereign.visualizer.update_quantum_state(
                        result.get('quantum_state', np.zeros((10, 10, 10, 4))),
                        metrics=test_result
                    )
                except Exception as viz_err:
                    print(f"Warning: Visualization failed: {viz_err}")

            # Store result and update statistics
            results.append(test_result)
            total_time += exec_time
            total_coherence += coherence
            successful_tests += 1

            # Pause between tests if requested
            time.sleep(args.pause)

        except Exception as e:
            print(f"Error in test case {case_name}: {str(e)}")
            failed_tests += 1
            if not args.continue_on_error:
                raise
            continue

    return {
        'results': results,
        'stats': {
            'total_time': total_time,
            'total_coherence': total_coherence,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'total_cases': total_cases
        }
    }


def run_sovereign_demo() -> None:
    """Run sovereign quantum system demo with comprehensive stability tests."""
    args = parse_arguments()
    sovereign = None
    start_total = time.time()

    try:
        print("Initializing Sovereign Quantum System...")
        sovereign = SovereignQuantumSystem()
        setup_signal_handlers(sovereign)

        if not args.skip_viz:
            print("Initializing visualization system...")
            try:
                sovereign.visualizer.initialize_real_time_display()
            except Exception as viz_err:
                print("Warning: Visualization initialization failed:")
                print(f"  {viz_err}")
                print("Continuing without visualization...")

        # Define test cases
        test_cases = [
            ("Random Values", np.random.random((10, 10, 10)) * 0.1),
            ("Near-Zero", np.full((10, 10, 10), 1e-10)),
            ("Large Values", np.full((10, 10, 10), 1e10)),
            ("Mixed Scale", np.logspace(-10, 10, 1000).reshape(10, 10, 10)),
            ("Alternating", np.indices((10, 10, 10))[0] % 2)
        ]

        # Run test cases
        results = execute_test_cases(test_cases, args, sovereign)

        # Calculate and display final statistics
        total_runtime = time.time() - start_total
        stats = results['stats']

        successful = stats['successful_tests']
        total = stats['total_cases']
        failed = stats['failed_tests']

        if successful:
            avg_exec = stats['total_time'] / successful
            avg_coherence = stats['total_coherence'] / successful
        else:
            avg_exec = 0
            avg_coherence = 0

        success_rate = successful / total
        status_color = "GREEN" if success_rate > 0.8 else "YELLOW"
        if success_rate < 0.5:
            status_color = "RED"

        title = color_text("Test Suite Complete", status_color)
        logging.info(f"\n{title}")

        final_metrics = {
            "Total Runtime": f"{total_runtime:.2f}s",
            "Avg Exec Time": f"{avg_exec:.3f}s",
            "Avg Coherence": color_text(f"{avg_coherence:.3f}", "BLUE"),
            "Success Rate": color_text(f"{successful}/{total}", status_color),
            "Failed Tests": (
                color_text(str(failed), "RED") if failed
                else color_text("0", "GREEN")
            )
        }
        format_metric_box(final_metrics)

        # Save results if requested
        if args.save_results is not None:
            args.save_results.parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)

        # Log final status
        log_msg = (
            f"Demo completed with {successful} successful tests "
            f"and {failed} failures"
        )
        if failed:
            logging.warning(log_msg)
        else:
            logging.info(log_msg)

        if not args.skip_viz:
            input("\nPress Enter to close visualization...")

    except Exception as e:
        print(f"Fatal error: {e}")
        if not args.continue_on_error:
            raise
    finally:
        if sovereign is not None and not args.skip_viz:
            try:
                sovereign.visualizer.cleanup()
            except Exception as cleanup_err:
                print(f"Warning: Visualization cleanup failed: {cleanup_err}")
        print("Demo completed.")


if __name__ == "__main__":
    run_sovereign_demo()
