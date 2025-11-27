"""
Feature Flag System for Advanced Tool Use.

Provides runtime control over feature availability and gradual rollout.
"""

from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import os
import json
import logging


logger = logging.getLogger(__name__)


class FlagState(Enum):
    """State of a feature flag."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    PERCENTAGE = "percentage"  # Gradual rollout
    CONDITIONAL = "conditional"  # Based on conditions


@dataclass
class FeatureFlag:
    """Definition of a feature flag."""
    name: str
    description: str
    state: FlagState
    default_value: bool = False
    percentage: float = 0.0  # For gradual rollout (0-100)
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    owner: str = "system"


class FeatureFlagManager:
    """
    Manages feature flags for the Advanced Tool Use system.

    Flags:
    - TOOL_SEARCH_ENABLED: Enable/disable Tool Search Tool
    - PTC_ENABLED: Enable/disable Programmatic Tool Calling
    - EXAMPLES_ENABLED: Enable/disable Tool Use Examples
    - DEFERRED_LOADING: Enable/disable deferred tool loading
    - BM25_SEARCH: Use BM25 vs simple regex search
    - PARALLEL_EXECUTION: Enable parallel tool execution in PTC
    """

    def __init__(self) -> None:
        self._flags: Dict[str, FeatureFlag] = {}
        self._overrides: Dict[str, bool] = {}
        self._listeners: List[Callable[[str, bool], None]] = []
        self._initialize_default_flags()

    def _initialize_default_flags(self) -> None:
        """Initialize default feature flags."""
        defaults = [
            FeatureFlag(
                name="TOOL_SEARCH_ENABLED",
                description="Enable Tool Search Tool for dynamic tool discovery",
                state=FlagState.ENABLED,
                default_value=True,
                owner="advanced_tool_use"
            ),
            FeatureFlag(
                name="PTC_ENABLED",
                description="Enable Programmatic Tool Calling for code-based orchestration",
                state=FlagState.ENABLED,
                default_value=True,
                owner="advanced_tool_use"
            ),
            FeatureFlag(
                name="EXAMPLES_ENABLED",
                description="Enable Tool Use Examples for improved parameter accuracy",
                state=FlagState.ENABLED,
                default_value=True,
                owner="advanced_tool_use"
            ),
            FeatureFlag(
                name="DEFERRED_LOADING",
                description="Enable deferred loading to reduce initial context",
                state=FlagState.ENABLED,
                default_value=True,
                owner="advanced_tool_use"
            ),
            FeatureFlag(
                name="BM25_SEARCH",
                description="Use BM25 algorithm for tool search (vs simple regex)",
                state=FlagState.ENABLED,
                default_value=True,
                owner="tool_search"
            ),
            FeatureFlag(
                name="PARALLEL_EXECUTION",
                description="Enable parallel tool execution in PTC sandbox",
                state=FlagState.ENABLED,
                default_value=True,
                owner="ptc"
            ),
            FeatureFlag(
                name="STRICT_VALIDATION",
                description="Enable strict input validation for all tools",
                state=FlagState.ENABLED,
                default_value=True,
                owner="validation"
            ),
            FeatureFlag(
                name="METRICS_COLLECTION",
                description="Collect usage metrics for tool optimization",
                state=FlagState.ENABLED,
                default_value=True,
                owner="observability"
            ),
            FeatureFlag(
                name="SANDBOX_RESTRICTIONS",
                description="Apply strict sandbox restrictions for PTC",
                state=FlagState.ENABLED,
                default_value=True,
                owner="security"
            ),
            FeatureFlag(
                name="EXPERIMENTAL_FEATURES",
                description="Enable experimental features (use with caution)",
                state=FlagState.DISABLED,
                default_value=False,
                owner="system"
            ),
        ]

        for flag in defaults:
            self._flags[flag.name] = flag

    def is_enabled(self, flag_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            flag_name: Name of the flag
            context: Optional context for conditional flags

        Returns:
            True if enabled, False otherwise
        """
        # Check override first
        if flag_name in self._overrides:
            return self._overrides[flag_name]

        # Check environment variable
        env_var = f"NSIP_{flag_name}"
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value.lower() in ("true", "1", "yes", "on")

        # Check flag definition
        flag = self._flags.get(flag_name)
        if not flag:
            logger.warning(f"Unknown feature flag: {flag_name}")
            return False

        if flag.state == FlagState.ENABLED:
            return True
        elif flag.state == FlagState.DISABLED:
            return False
        elif flag.state == FlagState.PERCENTAGE:
            import random
            return random.random() * 100 < flag.percentage
        elif flag.state == FlagState.CONDITIONAL:
            return self._evaluate_conditions(flag.conditions, context)

        return flag.default_value

    def _evaluate_conditions(
        self,
        conditions: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Evaluate conditional flag rules."""
        if not context:
            return False

        for key, expected in conditions.items():
            actual = context.get(key)
            if actual != expected:
                return False

        return True

    def set_override(self, flag_name: str, value: bool) -> None:
        """Set a runtime override for a flag."""
        self._overrides[flag_name] = value
        logger.info(f"Flag override set: {flag_name} = {value}")

        # Notify listeners
        for listener in self._listeners:
            listener(flag_name, value)

    def clear_override(self, flag_name: str) -> None:
        """Clear a runtime override."""
        if flag_name in self._overrides:
            del self._overrides[flag_name]
            logger.info(f"Flag override cleared: {flag_name}")

    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get flag definition."""
        return self._flags.get(flag_name)

    def list_flags(self) -> List[FeatureFlag]:
        """List all feature flags."""
        return list(self._flags.values())

    def get_enabled_flags(self) -> Set[str]:
        """Get all currently enabled flags."""
        return {name for name in self._flags if self.is_enabled(name)}

    def register(self, flag: FeatureFlag) -> None:
        """Register a new feature flag."""
        self._flags[flag.name] = flag
        logger.info(f"Registered feature flag: {flag.name}")

    def update(
        self,
        flag_name: str,
        state: Optional[FlagState] = None,
        percentage: Optional[float] = None,
        conditions: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a feature flag."""
        if flag_name not in self._flags:
            return False

        flag = self._flags[flag_name]

        if state is not None:
            flag.state = state
        if percentage is not None:
            flag.percentage = percentage
        if conditions is not None:
            flag.conditions = conditions

        flag.updated_at = datetime.now()
        logger.info(f"Updated feature flag: {flag_name}")
        return True

    def on_change(self, callback: Callable[[str, bool], None]) -> None:
        """Register a callback for flag changes."""
        self._listeners.append(callback)

    def export_config(self) -> Dict[str, Any]:
        """Export current flag configuration."""
        return {
            name: {
                "state": flag.state.value,
                "default": flag.default_value,
                "percentage": flag.percentage,
                "conditions": flag.conditions,
                "enabled": self.is_enabled(name)
            }
            for name, flag in self._flags.items()
        }

    def import_config(self, config: Dict[str, Any]) -> None:
        """Import flag configuration."""
        for name, settings in config.items():
            if name in self._flags:
                self.update(
                    name,
                    state=FlagState(settings.get("state", "disabled")),
                    percentage=settings.get("percentage", 0.0),
                    conditions=settings.get("conditions", {})
                )


# Global feature flag manager
feature_flags = FeatureFlagManager()


# Convenience functions
def is_enabled(flag_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """Check if a feature flag is enabled."""
    return feature_flags.is_enabled(flag_name, context)


def require_flag(flag_name: str) -> Callable:
    """Decorator to require a feature flag for a function."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_enabled(flag_name):
                raise RuntimeError(f"Feature {flag_name} is not enabled")
            return func(*args, **kwargs)
        return wrapper
    return decorator
