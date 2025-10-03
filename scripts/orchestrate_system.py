#!/usr/bin/env python
"""
System Orchestrator for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module serves as the central orchestrator for the entire system:
1. System initialization and startup
2. Component coordination
3. Resource management
4. Monitoring and logging
5. Shutdown and cleanup
"""

import sys
import time
import json
import signal
import logging
import threading
from pathlib import Path
from datetime import datetime

from scripts.validate_system import SystemValidator, ValidationLevel
from scripts.benchmark_system import PerformanceMonitor
from scripts.manage_config import ConfigurationManager
from scripts.monitor_system import SystemMonitor
from scripts.error_recovery import ErrorRecoveryManager, ErrorContext, ErrorSeverity
from scripts.performance_optimizer import PerformanceOptimizer


class SystemState:
    INITIALIZING = "initializing"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"

class SystemOrchestrator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.state = SystemState.INITIALIZING
        self.logger = self._setup_logger()

        # Initialize core components
        self.config_manager = ConfigurationManager(project_root)
        self.validator = SystemValidator(project_root)
        self.performance_monitor = PerformanceMonitor(project_root)
        self.system_monitor = SystemMonitor(project_root)
        self.error_manager = ErrorRecoveryManager(project_root)
        self.perf_optimizer = PerformanceOptimizer(project_root)

        # Thread management
        self.monitoring_thread = None
        self.system_monitor_thread = None
        self.optimization_thread = None
        self.should_run = threading.Event()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("system_orchestrator")
        logger.setLevel(logging.INFO)

        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_dir / "system.log")
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _handle_shutdown(self, signum, frame):
        """Handle system shutdown gracefully."""
        self.logger.info(f"Received shutdown signal {signum}")
        self.shutdown()

    def _monitor_system(self):
        """Continuous system monitoring."""
        while self.should_run.is_set():
            try:
                # Run validation checks
                validation_results = self.validator.run_validation(
                    ValidationLevel.BASIC
                )

                # Run performance checks
                performance_results = self.performance_monitor.run_benchmarks()

                # Check configurations
                config_validation = self.config_manager.validate_configs()

                # Log results
                self.logger.info("System Status:")
                self.logger.info(f"Validation: {validation_results}")
                self.logger.info(f"Performance: {performance_results}")
                self.logger.info(f"Config: {config_validation}")

                # Sleep for monitoring interval
                time.sleep(300)  # 5 minutes

            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")
                self.state = SystemState.ERROR

    def initialize(self) -> bool:
        """Initialize the system."""
        try:
            self.logger.info("Initializing system...")

            # Validate core configurations
            config_validation = self.config_manager.validate_configs()
            if not all(config_validation.values()):
                error_ctx = ErrorContext(
                    component="configuration",
                    error_type="ValidationError",
                    message="Configuration validation failed",
                    timestamp=datetime.now().isoformat(),
                    severity=ErrorSeverity.HIGH.value,
                    retry_count=0,
                    context={"validation": config_validation}
                )
                if not self.error_manager.handle_error(error_ctx):
                    raise ValueError("Configuration validation failed")

            # Run system validation
            validation_results = self.validator.run_validation(
                ValidationLevel.COMPLETE
            )
            if not all(r['success'] for r in validation_results['results']):
                error_ctx = ErrorContext(
                    component="system_validation",
                    error_type="ValidationError",
                    message="System validation failed",
                    timestamp=datetime.now().isoformat(),
                    severity=ErrorSeverity.HIGH.value,
                    retry_count=0,
                    context={"validation": validation_results}
                )
                if not self.error_manager.handle_error(error_ctx):
                    raise ValueError("System validation failed")

            # Start monitoring threads
            self.should_run.set()

            # Start validation monitoring
            self.monitoring_thread = threading.Thread(
                target=self._monitor_system
            )
            self.monitoring_thread.start()

            # Start system monitoring
            self.system_monitor_thread = threading.Thread(
                target=self._run_system_monitor
            )
            self.system_monitor_thread.start()

            # Start performance optimization
            self.optimization_thread = threading.Thread(
                target=self._run_performance_optimization
            )
            self.optimization_thread.start()

            self.state = SystemState.RUNNING
            self.logger.info("System initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            self.state = SystemState.ERROR
            return False

    def _run_system_monitor(self):
        """Run system monitoring in background."""
        try:
            while self.should_run.is_set():
                self.system_monitor.collect_metrics()
                time.sleep(60)  # Collect metrics every minute
        except Exception as e:
            self.logger.error(f"System monitoring error: {str(e)}")
            self.state = SystemState.ERROR

    def _run_performance_optimization(self):
        """Run performance optimization in background."""
        try:
            self.perf_optimizer.start_monitoring()
            while self.should_run.is_set():
                time.sleep(60)  # Check every minute
        except Exception as e:
            error_ctx = ErrorContext(
                component="performance_optimization",
                error_type="OptimizationError",
                message=str(e),
                timestamp=datetime.now().isoformat(),
                severity=ErrorSeverity.MEDIUM.value,
                retry_count=0,
                context={}
            )
            self.error_manager.handle_error(error_ctx)
            self.state = SystemState.ERROR
        finally:
            self.perf_optimizer.stop_monitoring()

    def shutdown(self):
        """Shutdown the system gracefully."""
        try:
            self.logger.info("Initiating system shutdown...")
            self.state = SystemState.SHUTTING_DOWN

            # Stop monitoring threads
            self.should_run.clear()
            if self.monitoring_thread:
                self.monitoring_thread.join()
            if self.system_monitor_thread:
                self.system_monitor_thread.join()
            if self.optimization_thread:
                self.optimization_thread.join()

            # Generate final reports
            final_report = self.system_monitor.generate_report()
            self.system_monitor.save_report(final_report)

            # Get performance optimization report
            perf_report = self.perf_optimizer.get_performance_report()

            # Get error analysis
            error_analysis = self.error_manager.analyze_error_patterns()

            # Run final performance check
            self.performance_monitor.run_benchmarks()

            # Save final system state
            final_state = {
                "timestamp": datetime.now().isoformat(),
                "state": self.state,
                "last_validation": self.validator.run_validation(
                    ValidationLevel.BASIC
                ),
                "final_metrics": final_report,
                "performance_optimization": perf_report,
                "error_analysis": error_analysis
            }

            state_file = self.project_root / "system_state.json"
            with open(state_file, 'w') as f:
                json.dump(final_state, f, indent=2)

            self.logger.info("System shutdown completed")

        except Exception as e:
            self.logger.error(f"Shutdown error: {str(e)}")
            self.state = SystemState.ERROR
            raise

def main():
    """Main entry point for system orchestration."""
    project_root = Path(__file__).parent.parent
    orchestrator = SystemOrchestrator(project_root)

    try:
        if orchestrator.initialize():
            # Keep the main thread alive
            while orchestrator.state == SystemState.RUNNING:
                time.sleep(1)
    except KeyboardInterrupt:
        orchestrator.shutdown()
    except Exception as e:
        print(f"System error: {str(e)}")
        orchestrator.shutdown()
        sys.exit(1)

if __name__ == '__main__':
    main()
