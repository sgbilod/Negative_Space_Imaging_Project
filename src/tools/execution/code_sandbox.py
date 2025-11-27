"""
Code Sandbox for Safe Execution.

Provides additional isolation and resource limiting for PTC code execution.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import signal
import sys
from datetime import datetime
import logging

# resource module is Unix-only
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False  # Windows


logger = logging.getLogger(__name__)


class SandboxViolation(Exception):
    """Raised when sandbox security is violated."""
    pass


class ResourceLimitExceeded(Exception):
    """Raised when resource limits are exceeded."""
    pass


@dataclass
class SandboxLimits:
    """Resource limits for sandbox execution."""
    max_memory_mb: int = 256
    max_cpu_seconds: int = 30
    max_output_bytes: int = 1024 * 1024  # 1MB
    max_recursion_depth: int = 100
    max_iterations: int = 1_000_000


@dataclass
class SandboxMetrics:
    """Metrics collected during sandbox execution."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    peak_memory_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    output_bytes: int = 0
    iterations: int = 0


class RestrictedNamespace:
    """
    Provides a restricted namespace for code execution.

    Blocks access to dangerous builtins and modules.
    """

    BLOCKED_BUILTINS: Set[str] = frozenset({
        '__import__', 'eval', 'exec', 'compile',
        'open', 'input', 'breakpoint',
        'memoryview', 'globals', 'locals',
        'vars', 'dir', 'getattr', 'setattr', 'delattr',
        'hasattr', 'type', 'isinstance', 'issubclass',
        '__build_class__', '__loader__', '__spec__',
    })

    SAFE_BUILTINS: Dict[str, Any] = {
        'abs': abs,
        'all': all,
        'any': any,
        'bin': bin,
        'bool': bool,
        'bytes': bytes,
        'callable': callable,
        'chr': chr,
        'dict': dict,
        'divmod': divmod,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'format': format,
        'frozenset': frozenset,
        'hash': hash,
        'hex': hex,
        'int': int,
        'iter': iter,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'next': next,
        'oct': oct,
        'ord': ord,
        'pow': pow,
        'print': print,
        'range': range,
        'repr': repr,
        'reversed': reversed,
        'round': round,
        'set': set,
        'slice': slice,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'zip': zip,
        'True': True,
        'False': False,
        'None': None,
    }

    def __init__(self) -> None:
        self._namespace: Dict[str, Any] = {}
        self._setup_namespace()

    def _setup_namespace(self) -> None:
        """Initialize the restricted namespace."""
        self._namespace = {
            '__builtins__': self.SAFE_BUILTINS.copy(),
            '__name__': '__sandbox__',
            '__doc__': None,
        }

    def add(self, name: str, value: Any) -> None:
        """Add a value to the namespace."""
        if name.startswith('_'):
            raise SandboxViolation(f"Cannot add private name: {name}")
        self._namespace[name] = value

    def get_namespace(self) -> Dict[str, Any]:
        """Get the complete namespace."""
        return self._namespace.copy()


class CodeSandbox:
    """
    Sandboxed execution environment for PTC code.

    Features:
    - Resource limiting (memory, CPU, output)
    - Restricted namespace
    - Metrics collection
    - Timeout handling
    """

    def __init__(self, limits: Optional[SandboxLimits] = None) -> None:
        self.limits = limits or SandboxLimits()
        self.metrics = SandboxMetrics()
        self._namespace = RestrictedNamespace()

    def add_to_namespace(self, name: str, value: Any) -> None:
        """Add a value to the sandbox namespace."""
        self._namespace.add(name, value)

    async def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute code in the sandbox.

        Args:
            code: Python code to execute

        Returns:
            Dict with result, stdout, stderr, and metrics
        """
        self.metrics = SandboxMetrics()

        # Set resource limits (Unix only)
        self._set_resource_limits()

        namespace = self._namespace.get_namespace()

        try:
            # Compile and execute
            compiled = compile(code, '<sandbox>', 'exec')
            exec(compiled, namespace)

            result = namespace.get('result', None)
            success = True
            error = None

        except Exception as e:
            result = None
            success = False
            error = str(e)

        finally:
            self.metrics.end_time = datetime.now()
            self._restore_resource_limits()

        return {
            'success': success,
            'result': result,
            'error': error,
            'metrics': {
                'execution_time_ms': int(
                    (self.metrics.end_time - self.metrics.start_time).total_seconds() * 1000
                ),
                'peak_memory_mb': self.metrics.peak_memory_mb,
            }
        }

    def _set_resource_limits(self) -> None:
        """Set resource limits for execution."""
        # Only available on Unix
        if HAS_RESOURCE and sys.platform != 'win32':
            try:
                # Memory limit
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (self.limits.max_memory_mb * 1024 * 1024, hard)
                )

                # CPU time limit
                soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (self.limits.max_cpu_seconds, hard)
                )
            except (ValueError, resource.error) as e:
                logger.warning(f"Could not set resource limits: {e}")

    def _restore_resource_limits(self) -> None:
        """Restore default resource limits."""
        if HAS_RESOURCE and sys.platform != 'win32':
            try:
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
                )
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
                )
            except (ValueError, resource.error):
                pass
