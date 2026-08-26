#!/usr/bin/env python
"""
HPC Extensions Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

Plugin architecture for custom HPC extensions, including support for
custom schedulers, integration hooks, and third-party tools.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ExtensionType(Enum):
    """Types of HPC extensions."""
    SCHEDULER = "scheduler"
    STORAGE = "storage"
    METRICS = "metrics"
    MONITORING = "monitoring"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    CUSTOM = "custom"


class HookType(Enum):
    """Types of extension hooks."""
    PRE_SUBMIT = "pre_submit"
    POST_SUBMIT = "post_submit"
    PRE_EXECUTE = "pre_execute"
    POST_EXECUTE = "post_execute"
    ON_ERROR = "on_error"
    ON_CANCEL = "on_cancel"
    ON_COMPLETE = "on_complete"
    ON_TIMEOUT = "on_timeout"


@dataclass
class ExtensionMetadata:
    """Metadata for an extension."""
    name: str
    version: str
    author: str
    description: str
    extension_type: ExtensionType
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "extension_type": self.extension_type.value,
            "dependencies": self.dependencies,
            "config_schema": self.config_schema,
        }


@dataclass
class HookContext:
    """Context passed to hook functions."""
    hook_type: HookType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    job_id: Optional[str] = None
    task_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None


@dataclass
class HookResult:
    """Result from a hook function."""
    success: bool
    modified_data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    stop_propagation: bool = False


class BaseExtension(ABC):
    """
    Base class for all HPC extensions.

    All custom extensions should inherit from this class and implement
    the required abstract methods.

    Example:
        class MyCustomExtension(BaseExtension):
            def get_metadata(self) -> ExtensionMetadata:
                return ExtensionMetadata(
                    name="my-extension",
                    version="1.0.0",
                    author="Developer",
                    description="A custom extension",
                    extension_type=ExtensionType.CUSTOM,
                )

            def initialize(self, config: Dict[str, Any]) -> None:
                self.config = config

            def shutdown(self) -> None:
                pass
    """

    def __init__(self) -> None:
        """Initialize the extension."""
        self._initialized = False
        self._config: Dict[str, Any] = {}
        self._hooks: Dict[HookType, List[Callable]] = {}

    @abstractmethod
    def get_metadata(self) -> ExtensionMetadata:
        """
        Get extension metadata.

        Returns:
            ExtensionMetadata describing the extension
        """
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the extension with configuration.

        Args:
            config: Extension configuration dictionary
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up extension resources."""
        pass

    def register_hook(
        self,
        hook_type: HookType,
        callback: Callable[[HookContext], HookResult]
    ) -> None:
        """
        Register a callback for a specific hook.

        Args:
            hook_type: Type of hook to register for
            callback: Callback function
        """
        if hook_type not in self._hooks:
            self._hooks[hook_type] = []
        self._hooks[hook_type].append(callback)
        logger.debug(f"Registered hook {hook_type.value} for {self.get_metadata().name}")

    def execute_hooks(self, context: HookContext) -> List[HookResult]:
        """
        Execute all registered hooks for a context.

        Args:
            context: Hook context

        Returns:
            List of hook results
        """
        results = []
        hooks = self._hooks.get(context.hook_type, [])

        for hook in hooks:
            try:
                result = hook(context)
                results.append(result)
                if result.stop_propagation:
                    break
            except Exception as e:
                logger.error(f"Hook execution error: {e}")
                results.append(HookResult(success=False, message=str(e)))

        return results

    @property
    def is_initialized(self) -> bool:
        """Check if extension is initialized."""
        return self._initialized


class SchedulerExtension(BaseExtension):
    """
    Base class for custom scheduler extensions.

    Allows integration with custom job scheduling systems.
    """

    @abstractmethod
    def submit_job(self, script_path: str, job_config: Dict[str, Any]) -> Optional[str]:
        """
        Submit a job to the custom scheduler.

        Args:
            script_path: Path to the job script
            job_config: Job configuration

        Returns:
            Job ID if successful
        """
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> str:
        """
        Get the status of a job.

        Args:
            job_id: Job ID

        Returns:
            Status string
        """
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job ID

        Returns:
            True if cancellation was successful
        """
        pass

    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all jobs.

        Returns:
            List of job information dictionaries
        """
        return []


class StorageExtension(BaseExtension):
    """
    Base class for custom storage extensions.

    Allows integration with custom storage backends.
    """

    @abstractmethod
    def store(self, key: str, data: bytes) -> bool:
        """
        Store data.

        Args:
            key: Storage key
            data: Data to store

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Optional[bytes]:
        """
        Retrieve data.

        Args:
            key: Storage key

        Returns:
            Stored data or None
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete data.

        Args:
            key: Storage key

        Returns:
            True if successful
        """
        pass

    def exists(self, key: str) -> bool:
        """
        Check if key exists.

        Args:
            key: Storage key

        Returns:
            True if exists
        """
        return self.retrieve(key) is not None

    def list_keys(self, prefix: str = "") -> List[str]:
        """
        List keys with prefix.

        Args:
            prefix: Key prefix

        Returns:
            List of keys
        """
        return []


class MetricsExtension(BaseExtension):
    """
    Extension for collecting and exporting metrics.

    Provides hooks for metrics collection and export to various backends.
    """

    def __init__(self) -> None:
        """Initialize metrics extension."""
        super().__init__()
        self._metrics: Dict[str, float] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}

    def get_metadata(self) -> ExtensionMetadata:
        """Get extension metadata."""
        return ExtensionMetadata(
            name="metrics-extension",
            version="1.0.0",
            author="NSI Team",
            description="Built-in metrics collection extension",
            extension_type=ExtensionType.METRICS,
        )

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the extension."""
        self._config = config
        self._initialized = True
        logger.info("Metrics extension initialized")

    def shutdown(self) -> None:
        """Shutdown the extension."""
        self._metrics.clear()
        self._counters.clear()
        self._gauges.clear()
        self._initialized = False

    def record_metric(self, name: str, value: float) -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
        """
        self._metrics[name] = value

    def increment_counter(self, name: str, value: int = 1) -> None:
        """
        Increment a counter.

        Args:
            name: Counter name
            value: Increment value
        """
        self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float) -> None:
        """
        Set a gauge value.

        Args:
            name: Gauge name
            value: Gauge value
        """
        self._gauges[name] = value

    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect all metrics.

        Returns:
            Dictionary of all metrics
        """
        return {
            "metrics": self._metrics.copy(),
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        for name, value in self._metrics.items():
            lines.append(f"nsi_{name} {value}")

        for name, value in self._counters.items():
            lines.append(f"nsi_{name}_total {value}")

        for name, value in self._gauges.items():
            lines.append(f"nsi_{name}_gauge {value}")

        return "\n".join(lines) + "\n"


class MonitoringExtension(BaseExtension):
    """
    Extension for job monitoring and alerting.

    Provides real-time monitoring of HPC jobs and alert capabilities.
    """

    def __init__(self) -> None:
        """Initialize monitoring extension."""
        super().__init__()
        self._alert_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self._thresholds: Dict[str, float] = {}

    def get_metadata(self) -> ExtensionMetadata:
        """Get extension metadata."""
        return ExtensionMetadata(
            name="monitoring-extension",
            version="1.0.0",
            author="NSI Team",
            description="Built-in monitoring and alerting extension",
            extension_type=ExtensionType.MONITORING,
        )

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the extension."""
        self._config = config
        self._thresholds = config.get("thresholds", {})
        self._initialized = True
        logger.info("Monitoring extension initialized")

    def shutdown(self) -> None:
        """Shutdown the extension."""
        self._alert_handlers.clear()
        self._initialized = False

    def register_alert_handler(
        self,
        handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Register an alert handler.

        Args:
            handler: Alert handler function
        """
        self._alert_handlers.append(handler)

    def check_threshold(self, metric_name: str, value: float) -> bool:
        """
        Check if a value exceeds its threshold.

        Args:
            metric_name: Metric name
            value: Current value

        Returns:
            True if threshold exceeded
        """
        threshold = self._thresholds.get(metric_name)
        if threshold is not None and value > threshold:
            self._trigger_alert({
                "type": "threshold_exceeded",
                "metric": metric_name,
                "value": value,
                "threshold": threshold,
            })
            return True
        return False

    def _trigger_alert(self, alert_data: Dict[str, Any]) -> None:
        """Trigger an alert."""
        alert_data["timestamp"] = datetime.utcnow().isoformat()
        for handler in self._alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")


class ExtensionRegistry:
    """
    Registry for managing HPC extensions.

    Provides registration, discovery, and lifecycle management for extensions.

    Example:
        registry = ExtensionRegistry()
        registry.register(MyExtension())
        registry.initialize_all(config)
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._extensions: Dict[str, BaseExtension] = {}
        self._extension_types: Dict[ExtensionType, List[str]] = {}

    def register(
        self,
        extension: BaseExtension,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register an extension.

        Args:
            extension: Extension instance
            config: Optional configuration
        """
        metadata = extension.get_metadata()
        name = metadata.name

        if name in self._extensions:
            logger.warning(f"Extension {name} already registered, overwriting")

        self._extensions[name] = extension

        # Track by type
        ext_type = metadata.extension_type
        if ext_type not in self._extension_types:
            self._extension_types[ext_type] = []
        if name not in self._extension_types[ext_type]:
            self._extension_types[ext_type].append(name)

        # Initialize if config provided
        if config:
            extension.initialize(config)

        logger.info(f"Registered extension: {name} v{metadata.version}")

    def unregister(self, name: str) -> bool:
        """
        Unregister an extension.

        Args:
            name: Extension name

        Returns:
            True if successfully unregistered
        """
        if name not in self._extensions:
            return False

        extension = self._extensions[name]
        if extension.is_initialized:
            extension.shutdown()

        metadata = extension.get_metadata()
        ext_type = metadata.extension_type
        if ext_type in self._extension_types:
            self._extension_types[ext_type].remove(name)

        del self._extensions[name]
        logger.info(f"Unregistered extension: {name}")
        return True

    def get(self, name: str) -> Optional[BaseExtension]:
        """
        Get an extension by name.

        Args:
            name: Extension name

        Returns:
            Extension instance or None
        """
        return self._extensions.get(name)

    def get_by_type(self, ext_type: ExtensionType) -> List[BaseExtension]:
        """
        Get all extensions of a specific type.

        Args:
            ext_type: Extension type

        Returns:
            List of extensions
        """
        names = self._extension_types.get(ext_type, [])
        return [self._extensions[name] for name in names]

    def list_extensions(self) -> List[str]:
        """List all registered extension names."""
        return list(self._extensions.keys())

    def initialize_all(self, config: Dict[str, Any]) -> None:
        """
        Initialize all extensions.

        Args:
            config: Configuration dictionary with extension-specific configs
        """
        for name, extension in self._extensions.items():
            ext_config = config.get(name, {})
            try:
                extension.initialize(ext_config)
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")

    def shutdown_all(self) -> None:
        """Shutdown all extensions."""
        for name, extension in self._extensions.items():
            try:
                extension.shutdown()
            except Exception as e:
                logger.error(f"Failed to shutdown {name}: {e}")

    def execute_hooks(
        self,
        hook_type: HookType,
        context_data: Optional[Dict[str, Any]] = None
    ) -> List[HookResult]:
        """
        Execute hooks across all extensions.

        Args:
            hook_type: Type of hook
            context_data: Hook context data

        Returns:
            List of results from all hooks
        """
        context = HookContext(
            hook_type=hook_type,
            data=context_data or {},
        )

        all_results = []
        for extension in self._extensions.values():
            results = extension.execute_hooks(context)
            all_results.extend(results)

        return all_results


class ExtensionLoader:
    """
    Utility class for loading extensions from modules.

    Supports loading extensions from Python modules or packages.
    """

    @staticmethod
    def load_from_module(module_path: str) -> Optional[BaseExtension]:
        """
        Load an extension from a Python module.

        Args:
            module_path: Python module path (e.g., 'my_package.my_extension')

        Returns:
            Extension instance or None
        """
        try:
            import importlib
            module = importlib.import_module(module_path)

            # Look for extension class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type) and
                    issubclass(attr, BaseExtension) and
                    attr is not BaseExtension
                ):
                    return attr()

            logger.warning(f"No extension found in {module_path}")
            return None

        except Exception as e:
            logger.error(f"Failed to load extension from {module_path}: {e}")
            return None

    @staticmethod
    def load_from_file(file_path: str) -> Optional[BaseExtension]:
        """
        Load an extension from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Extension instance or None
        """
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("extension", file_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type) and
                    issubclass(attr, BaseExtension) and
                    attr is not BaseExtension
                ):
                    return attr()

            return None

        except Exception as e:
            logger.error(f"Failed to load extension from {file_path}: {e}")
            return None


# Global registry instance
_global_registry: Optional[ExtensionRegistry] = None


def get_registry() -> ExtensionRegistry:
    """
    Get the global extension registry.

    Returns:
        Global ExtensionRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ExtensionRegistry()
    return _global_registry


def register_extension(extension: BaseExtension) -> None:
    """
    Register an extension globally.

    Args:
        extension: Extension to register
    """
    get_registry().register(extension)


def get_extension(name: str) -> Optional[BaseExtension]:
    """
    Get an extension by name from global registry.

    Args:
        name: Extension name

    Returns:
        Extension instance or None
    """
    return get_registry().get(name)
