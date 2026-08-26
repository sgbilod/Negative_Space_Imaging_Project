#!/usr/bin/env python
"""
Performance Optimization CLI
© 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

Command-line interface for the Sovereign Performance Optimization System.
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import logging

from sovereign.performance import (
    PerformanceManager,
    OptimizationLevel,
    OptimizationTarget,
    get_performance_manager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('performance_cli.log')
    ]
)
logger = logging.getLogger('sovereign.performance_cli')


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Sovereign Performance Optimization System'
    )

    # Main command
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Optimize command
    optimize_parser = subparsers.add_parser(
        'optimize',
        help='Optimize system performance'
    )
    optimize_parser.add_argument(
        '--target',
        choices=[t.value for t in OptimizationTarget],
        default='ALL',
        help='Optimization target'
    )
    optimize_parser.add_argument(
        '--level',
        choices=[l.value for l in OptimizationLevel],
        default=None,
        help='Optimization level'
    )
    optimize_parser.add_argument(
        '--component',
        help='Specific component to optimize'
    )

    # Monitor command
    monitor_parser = subparsers.add_parser(
        'monitor',
        help='Monitor system performance'
    )
    monitor_parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Monitoring interval in seconds'
    )
    monitor_parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Monitoring duration in seconds'
    )
    monitor_parser.add_argument(
        '--output',
        help='Output file for monitoring data'
    )

    # Profile command
    profile_parser = subparsers.add_parser(
        'profile',
        help='Profile system performance'
    )
    profile_parser.add_argument(
        'module',
        help='Module to profile'
    )
    profile_parser.add_argument(
        '--function',
        help='Function to profile'
    )
    profile_parser.add_argument(
        '--output',
        help='Output directory for profile data'
    )

    # Report command
    report_parser = subparsers.add_parser(
        'report',
        help='Generate performance report'
    )
    report_parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Report format'
    )
    report_parser.add_argument(
        '--output',
        help='Output file for report'
    )

    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show current performance status'
    )
    status_parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Status format'
    )

    # Cache command
    cache_parser = subparsers.add_parser(
        'cache',
        help='Manage performance cache'
    )
    cache_subparsers = cache_parser.add_subparsers(
        dest='cache_command',
        help='Cache command'
    )

    # Cache stats command
    cache_stats_parser = cache_subparsers.add_parser(
        'stats',
        help='Show cache statistics'
    )

    # Cache clear command
    cache_clear_parser = cache_subparsers.add_parser(
        'clear',
        help='Clear the cache'
    )

    # Cache resize command
    cache_resize_parser = cache_subparsers.add_parser(
        'resize',
        help='Resize the cache'
    )
    cache_resize_parser.add_argument(
        'size',
        type=int,
        help='New cache size'
    )

    # Memory command
    memory_parser = subparsers.add_parser(
        'memory',
        help='Manage memory optimization'
    )
    memory_subparsers = memory_parser.add_subparsers(
        dest='memory_command',
        help='Memory command'
    )

    # Memory stats command
    memory_stats_parser = memory_subparsers.add_parser(
        'stats',
        help='Show memory statistics'
    )

    # Memory gc command
    memory_gc_parser = memory_subparsers.add_parser(
        'gc',
        help='Run garbage collection'
    )

    # Memory track command
    memory_track_parser = memory_subparsers.add_parser(
        'track',
        help='Track memory usage'
    )
    memory_track_parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Tracking interval in seconds'
    )
    memory_track_parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Tracking duration in seconds'
    )
    memory_track_parser.add_argument(
        '--output',
        help='Output file for tracking data'
    )

    return parser.parse_args()


def format_size(size_bytes):
    """Format size in bytes to human-readable string"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def print_memory_stats(stats):
    """Print memory statistics in human-readable format"""
    print(f"RSS: {format_size(stats['rss'] * 1024 * 1024)}")
    print(f"VMS: {format_size(stats['vms'] * 1024 * 1024)}")
    print(f"Percent: {stats['percent']:.2f}%")
    print(f"Timestamp: {datetime.fromtimestamp(stats['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")


def print_cache_stats(stats):
    """Print cache statistics in human-readable format"""
    print(f"Size: {stats['size']} / {stats['max_size']}")
    print(f"Hits: {stats['hits']}")
    print(f"Misses: {stats['misses']}")
    print(f"Hit ratio: {stats['hit_ratio'] * 100:.2f}%")
    print(f"TTL: {stats['ttl']} seconds")


def print_performance_report(report):
    """Print performance report in human-readable format"""
    print("=== Sovereign Performance Report ===")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Optimization Level: {report['optimization_level']}")
    print()

    print("--- Current Metrics ---")
    print(f"CPU Usage: {report['current_metrics']['cpu_usage']:.2f}%")
    print(f"Memory Usage: {report['current_metrics']['memory_usage']:.2f} MB")
    print(f"Thread Count: {report['thread_count']}")
    print(f"Process Count: {report['process_count']}")
    print()

    print("--- Cache Stats ---")
    print_cache_stats(report['cache_stats'])
    print()

    print("--- Memory Stats ---")
    print_memory_stats(report['memory_stats'])
    print()

    print("--- Garbage Collection Stats ---")
    print(f"Collections: {report['gc_stats']['collections']}")
    print(f"Objects Collected: {report['gc_stats']['collected']}")
    print(f"Uncollectable Objects: {report['gc_stats']['uncollectable']}")
    print(f"Collection Time: {report['gc_stats']['collection_time']:.4f} seconds")
    print()

    if report['recommendations']:
        print("--- Optimization Recommendations ---")
        for rec in report['recommendations']:
            print(f"* [{rec['level'].upper()}] {rec['message']}")
        print()


def command_optimize(args):
    """Execute the optimize command"""
    pm = get_performance_manager()

    # Set optimization level if specified
    if args.level:
        pm.set_optimization_level(OptimizationLevel(args.level))

    # Run optimization
    target = OptimizationTarget(args.target)
    component = args.component

    print(f"Optimizing {'component ' + component if component else 'system'} for target: {target.value}")

    if component:
        # Optimize specific component
        profile = pm.get_component_profile(component)
        if not profile:
            pm.register_component(component, targets=[target])
            profile = pm.get_component_profile(component)

        profile.enabled = True
        print(f"Component {component} optimization profile updated")
    else:
        # Optimize entire system
        results = pm.optimize_system(target)

        print(f"Applied {len(results['optimizations_applied'])} optimizations:")
        for opt in results['optimizations_applied']:
            print(f"* {opt['type']}")


def command_monitor(args):
    """Execute the monitor command"""
    pm = get_performance_manager()
    interval = args.interval
    duration = args.duration
    output_file = args.output

    iterations = duration // interval
    results = []

    print(f"Monitoring system performance for {duration} seconds (interval: {interval}s)")

    try:
        for i in range(iterations):
            metrics = pm.collect_metrics()

            # Format for display
            print(f"\rIteration {i+1}/{iterations} - "
                  f"CPU: {metrics.cpu_usage:.2f}% | "
                  f"Memory: {metrics.memory_usage:.2f} MB | "
                  f"Threads: {metrics.thread_count}", end="")

            # Save metrics
            results.append(metrics.to_dict())

            # Wait for next interval
            if i < iterations - 1:
                time.sleep(interval)

        print("\nMonitoring complete")

        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'interval': interval,
                    'duration': duration,
                    'metrics': results
                }, f, indent=2)
            print(f"Results saved to {output_file}")

    except KeyboardInterrupt:
        print("\nMonitoring interrupted")


def command_profile(args):
    """Execute the profile command"""
    pm = get_performance_manager()
    module_name = args.module
    function_name = args.function
    output_dir = args.output

    if output_dir:
        pm.profiler.output_dir = Path(output_dir)

    print(f"Profiling module: {module_name}")

    try:
        # Import the module
        module = __import__(module_name, fromlist=['*'])

        if function_name:
            # Profile specific function
            if not hasattr(module, function_name):
                print(f"Error: Function '{function_name}' not found in module '{module_name}'")
                return

            func = getattr(module, function_name)
            print(f"Profiling function: {function_name}")

            # Run profiling
            _, profile_data = pm.profiler.profile_function(func)

            # Print profile summary
            print("\nProfile summary:")
            print(profile_data['stats'])

        else:
            # Profile entire module
            print("Module profiling not implemented yet")

    except ImportError:
        print(f"Error: Could not import module '{module_name}'")


def command_report(args):
    """Execute the report command"""
    pm = get_performance_manager()
    format_type = args.format
    output_file = args.output

    # Generate report
    report = pm.get_performance_report()

    # Output report
    if format_type == 'json':
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_file}")
        else:
            print(json.dumps(report, indent=2))
    else:
        print_performance_report(report)

        if output_file:
            with open(output_file, 'w') as f:
                # Write text report
                f.write("=== Sovereign Performance Report ===\n")
                f.write(f"Timestamp: {report['timestamp']}\n")
                f.write(f"Optimization Level: {report['optimization_level']}\n\n")

                f.write("--- Current Metrics ---\n")
                f.write(f"CPU Usage: {report['current_metrics']['cpu_usage']:.2f}%\n")
                f.write(f"Memory Usage: {report['current_metrics']['memory_usage']:.2f} MB\n")
                f.write(f"Thread Count: {report['thread_count']}\n")
                f.write(f"Process Count: {report['process_count']}\n\n")

                f.write("--- Recommendations ---\n")
                for rec in report['recommendations']:
                    f.write(f"* [{rec['level'].upper()}] {rec['message']}\n")

            print(f"Report saved to {output_file}")


def command_status(args):
    """Execute the status command"""
    pm = get_performance_manager()
    format_type = args.format

    # Collect metrics
    metrics = pm.collect_metrics()

    # Format status
    status = {
        'timestamp': datetime.now().isoformat(),
        'optimization_level': pm.optimization_level.value,
        'metrics': metrics.to_dict(),
        'components': len(pm.optimization_profiles),
        'recommendations': pm.get_optimization_recommendations()
    }

    # Output status
    if format_type == 'json':
        print(json.dumps(status, indent=2))
    else:
        print("=== Sovereign Performance Status ===")
        print(f"Timestamp: {status['timestamp']}")
        print(f"Optimization Level: {status['optimization_level']}")
        print(f"Registered Components: {status['components']}")
        print()

        print("--- Current Metrics ---")
        print(f"CPU Usage: {metrics.cpu_usage:.2f}%")
        print(f"Memory Usage: {metrics.memory_usage:.2f} MB")
        print(f"Execution Time: {metrics.execution_time:.4f} seconds")
        print(f"Thread Count: {metrics.thread_count}")
        print(f"Process Count: {metrics.process_count}")
        print(f"Cache Hits/Misses: {metrics.cache_hits}/{metrics.cache_misses}")
        print()

        if status['recommendations']:
            print("--- Recommendations ---")
            for rec in status['recommendations']:
                print(f"* [{rec['level'].upper()}] {rec['message']}")


def command_cache(args):
    """Execute the cache command"""
    pm = get_performance_manager()

    if args.cache_command == 'stats':
        # Show cache statistics
        stats = pm.cache.get_stats()
        print("=== Cache Statistics ===")
        print_cache_stats(stats)

    elif args.cache_command == 'clear':
        # Clear the cache
        pm.cache.clear()
        print("Cache cleared")

    elif args.cache_command == 'resize':
        # Resize the cache
        new_size = args.size
        pm.cache.resize(new_size)
        print(f"Cache resized to {new_size}")


def command_memory(args):
    """Execute the memory command"""
    pm = get_performance_manager()

    if args.memory_command == 'stats':
        # Show memory statistics
        stats = pm.memory_optimizer.get_current_memory_usage()
        print("=== Memory Statistics ===")
        print_memory_stats(stats)

        # Also show GC stats
        gc_stats = pm.memory_optimizer.get_gc_stats()
        print("\n=== Garbage Collection Statistics ===")
        print(f"Collections: {gc_stats['collections']}")
        print(f"Objects Collected: {gc_stats['collected']}")
        print(f"Uncollectable Objects: {gc_stats['uncollectable']}")
        print(f"Collection Time: {gc_stats['collection_time']:.4f} seconds")

    elif args.memory_command == 'gc':
        # Run garbage collection
        print("Running garbage collection...")
        stats = pm.memory_optimizer.force_garbage_collection()
        print(f"Collected {stats['collected']} objects in {stats['collection_time']:.4f} seconds")
        print(f"Uncollectable objects: {stats['uncollectable']}")

    elif args.memory_command == 'track':
        # Track memory usage
        interval = args.interval
        duration = args.duration
        output_file = args.output

        iterations = duration // interval
        results = []

        print(f"Tracking memory usage for {duration} seconds (interval: {interval}s)")

        try:
            for i in range(iterations):
                stats = pm.memory_optimizer.get_current_memory_usage()

                # Format for display
                print(f"\rIteration {i+1}/{iterations} - "
                      f"RSS: {stats['rss']:.2f} MB | "
                      f"VMS: {stats['vms']:.2f} MB | "
                      f"Percent: {stats['percent']:.2f}%", end="")

                # Save stats
                results.append(stats)

                # Wait for next interval
                if i < iterations - 1:
                    time.sleep(interval)

            print("\nTracking complete")

            # Save results if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'interval': interval,
                        'duration': duration,
                        'stats': results
                    }, f, indent=2)
                print(f"Results saved to {output_file}")

        except KeyboardInterrupt:
            print("\nTracking interrupted")


def main():
    """Main function"""
    args = parse_args()

    # Initialize performance manager
    pm = get_performance_manager()

    # Execute command
    if args.command == 'optimize':
        command_optimize(args)
    elif args.command == 'monitor':
        command_monitor(args)
    elif args.command == 'profile':
        command_profile(args)
    elif args.command == 'report':
        command_report(args)
    elif args.command == 'status':
        command_status(args)
    elif args.command == 'cache':
        command_cache(args)
    elif args.command == 'memory':
        command_memory(args)
    else:
        print("Error: No command specified")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
