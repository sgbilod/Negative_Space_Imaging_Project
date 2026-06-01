#!/usr/bin/env python
"""
Error Recovery and Resilience Module for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import time
import json
import logging
from enum import Enum
from typing import Any, Dict, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ErrorContext:
    component: str
    error_type: str
    message: str
    timestamp: str
    severity: str
    retry_count: int
    context: Dict[str, Any]

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorRecoveryManager:
    """Manages error recovery and system resilience."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = self._setup_logger()
        self.error_history = []
        self.recovery_strategies = {}
        self.max_retries = 3

        # Load error recovery configurations
        self.load_recovery_strategies()

    def _setup_logger(self) -> logging.Logger:
        """Configure logging for error recovery."""
        logger = logging.getLogger("error_recovery")
        logger.setLevel(logging.INFO)

        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(
            log_dir / "error_recovery.log"
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def load_recovery_strategies(self):
        """Load error recovery strategies from configuration."""
        config_path = self.project_root / "config" / "error_recovery.json"
        if not config_path.exists():
            self._create_default_strategies()
            return

        try:
            with open(config_path) as f:
                strategies = json.load(f)

            for component, strategy in strategies.items():
                self.recovery_strategies[component] = {
                    "max_retries": strategy.get("max_retries", 3),
                    "retry_delay": strategy.get("retry_delay", 5),
                    "fallback": strategy.get("fallback", None),
                    "severity_threshold": strategy.get(
                        "severity_threshold",
                        ErrorSeverity.HIGH.value
                    )
                }
        except Exception as e:
            self.logger.error(f"Failed to load recovery strategies: {e}")
            self._create_default_strategies()

    def _create_default_strategies(self):
        """Create default recovery strategies."""
        self.recovery_strategies = {
            "security": {
                "max_retries": 3,
                "retry_delay": 5,
                "fallback": "safe_mode",
                "severity_threshold": ErrorSeverity.HIGH.value
            },
            "image_processing": {
                "max_retries": 5,
                "retry_delay": 2,
                "fallback": "basic_processing",
                "severity_threshold": ErrorSeverity.MEDIUM.value
            },
            "storage": {
                "max_retries": 3,
                "retry_delay": 10,
                "fallback": "local_cache",
                "severity_threshold": ErrorSeverity.HIGH.value
            }
        }

        # Save default strategies
        config_path = self.project_root / "config" / "error_recovery.json"
        with open(config_path, 'w') as f:
            json.dump(self.recovery_strategies, f, indent=2)

    def handle_error(self, error_context: ErrorContext) -> bool:
        """Handle an error and attempt recovery."""
        self.error_history.append(error_context)
        self.logger.error(
            f"Error in {error_context.component}: {error_context.message}"
        )

        # Get recovery strategy
        strategy = self.recovery_strategies.get(
            error_context.component,
            self.recovery_strategies.get("default", {})
        )

        if not strategy:
            self.logger.error(f"No recovery strategy for {error_context.component}")
            return False

        # Check severity threshold
        if (error_context.severity == ErrorSeverity.CRITICAL.value or
            error_context.retry_count >= strategy["max_retries"]):
            return self._handle_critical_error(error_context, strategy)

        # Attempt recovery
        return self._attempt_recovery(error_context, strategy)

    def _attempt_recovery(
        self,
        error_context: ErrorContext,
        strategy: Dict
    ) -> bool:
        """Attempt to recover from an error."""
        try:
            # Log recovery attempt
            self.logger.info(
                f"Attempting recovery for {error_context.component} "
                f"(Attempt {error_context.retry_count + 1})"
            )

            # Wait before retry
            time.sleep(strategy["retry_delay"])

            # Update error context
            error_context.retry_count += 1

            return True

        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return False

    def _handle_critical_error(
        self,
        error_context: ErrorContext,
        strategy: Dict
    ) -> bool:
        """Handle a critical error or exhausted retries."""
        try:
            fallback = strategy.get("fallback")
            if not fallback:
                self.logger.error("No fallback available for critical error")
                return False

            self.logger.warning(
                f"Activating fallback '{fallback}' for {error_context.component}"
            )

            # Record the fallback activation
            self._record_fallback_activation(error_context, fallback)

            return True

        except Exception as e:
            self.logger.error(f"Fallback activation failed: {e}")
            return False

    def _record_fallback_activation(
        self,
        error_context: ErrorContext,
        fallback: str
    ):
        """Record fallback activation for analysis."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "component": error_context.component,
            "error_type": error_context.error_type,
            "error_message": error_context.message,
            "retry_count": error_context.retry_count,
            "fallback": fallback
        }

        # Save fallback record
        fallback_dir = self.project_root / "logs" / "fallbacks"
        fallback_dir.mkdir(exist_ok=True)

        fallback_file = fallback_dir / f"fallback_{int(time.time())}.json"
        with open(fallback_file, 'w') as f:
            json.dump(record, f, indent=2)

    def analyze_error_patterns(self) -> Dict:
        """Analyze error patterns for system improvement."""
        if not self.error_history:
            return {}

        analysis = {
            "total_errors": len(self.error_history),
            "errors_by_component": {},
            "errors_by_type": {},
            "error_timeline": [],
            "critical_errors": []
        }

        for error in self.error_history:
            # Component analysis
            if error.component not in analysis["errors_by_component"]:
                analysis["errors_by_component"][error.component] = 0
            analysis["errors_by_component"][error.component] += 1

            # Error type analysis
            if error.error_type not in analysis["errors_by_type"]:
                analysis["errors_by_type"][error.error_type] = 0
            analysis["errors_by_type"][error.error_type] += 1

            # Timeline
            analysis["error_timeline"].append({
                "timestamp": error.timestamp,
                "component": error.component,
                "error_type": error.error_type
            })

            # Critical errors
            if error.severity == ErrorSeverity.CRITICAL.value:
                analysis["critical_errors"].append({
                    "timestamp": error.timestamp,
                    "component": error.component,
                    "message": error.message
                })

        return analysis

    def get_error_history(self) -> list:
        """Get the error history."""
        return self.error_history

    def clear_error_history(self):
        """Clear the error history."""
        self.error_history = []

if __name__ == '__main__':
    # Test error recovery
    project_root = Path(__file__).parent.parent
    recovery_manager = ErrorRecoveryManager(project_root)

    # Test error handling
    test_error = ErrorContext(
        component="image_processing",
        error_type="ProcessingError",
        message="Failed to process image",
        timestamp=datetime.now().isoformat(),
        severity=ErrorSeverity.MEDIUM.value,
        retry_count=0,
        context={"file": "test.jpg"}
    )

    recovery_success = recovery_manager.handle_error(test_error)
    print(f"Recovery success: {recovery_success}")

    # Analyze patterns
    analysis = recovery_manager.analyze_error_patterns()
    print("\nError Analysis:")
    print(json.dumps(analysis, indent=2))
