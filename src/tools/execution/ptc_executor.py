"""
Programmatic Tool Calling (PTC) Executor.

Enables Claude to orchestrate tools via code instead of individual API calls.
Reduces context pollution by 37% and enables parallel execution.

Reference: Anthropic Advanced Tool Use - Programmatic Tool Calling
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import traceback
from datetime import datetime
import logging
import ast
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from ..registry.tool_registry import registry


logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of PTC execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class PTCExecutionResult:
    """Result of PTC execution."""
    status: ExecutionStatus
    stdout: str
    stderr: str
    final_result: Any
    tool_calls_made: int
    execution_time_ms: int
    context_tokens_saved: int


class PTCCodeValidator:
    """
    Validates PTC code before execution.

    Ensures:
    - No dangerous operations
    - Only allowed imports
    - Safe execution patterns
    """

    ALLOWED_IMPORTS = frozenset({
        'json', 'math', 'datetime', 'collections',
        'itertools', 'functools', 'typing', 're',
        'asyncio', 'statistics'
    })

    FORBIDDEN_PATTERNS = (
        'import os', 'import sys', 'import subprocess',
        '__import__', 'eval(', 'exec(', 'open(',
        'compile(', 'globals(', 'locals(', 'getattr(',
        'setattr(', 'delattr(', '__builtins__'
    )

    def validate(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Validate code for safe execution.

        Returns:
            (is_valid, error_message)
        """
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in code:
                return False, f"Forbidden pattern detected: {pattern}"

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module not in self.ALLOWED_IMPORTS:
                        return False, f"Import not allowed: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    if module not in self.ALLOWED_IMPORTS:
                        return False, f"Import not allowed: {node.module}"

        return True, None


class PTCExecutor:
    """
    Executes Claude's orchestration code in a sandboxed environment.

    Key features:
    - Tool calls don't enter Claude's context
    - Parallel execution via asyncio
    - Only final output returned to model
    """

    def __init__(self, timeout_seconds: int = 60) -> None:
        self.timeout = timeout_seconds
        self.validator = PTCCodeValidator()
        self._request_counter = 0

    async def execute(
        self,
        code: str,
        tool_results: Optional[Dict[str, Any]] = None
    ) -> PTCExecutionResult:
        """
        Execute PTC code.

        Args:
            code: Python code to execute
            tool_results: Results from previous tool requests

        Returns:
            PTCExecutionResult with final output
        """
        start_time = datetime.now()

        is_valid, error = self.validator.validate(code)
        if not is_valid:
            return PTCExecutionResult(
                status=ExecutionStatus.FAILED,
                stdout="",
                stderr=f"Validation failed: {error}",
                final_result=None,
                tool_calls_made=0,
                execution_time_ms=0,
                context_tokens_saved=0
            )

        stdout_capture = StringIO()
        stderr_capture = StringIO()
        tool_calls: List[Dict] = []

        async def make_tool_call(tool_name: str, **kwargs: Any) -> Any:
            """Wrapper for tool calls from PTC code."""
            self._request_counter += 1

            if not registry.validate_ptc_caller(tool_name, "code_execution_20250825"):
                raise ValueError(f"Tool {tool_name} does not support PTC")

            tool = registry.get_tool(tool_name)
            if not tool:
                raise ValueError(f"Tool not found: {tool_name}")

            result = await tool.execute(**kwargs)

            tool_calls.append({
                "tool": tool_name,
                "params_size": len(json.dumps(kwargs)),
                "result_size": len(json.dumps(result))
            })

            return result

        namespace = self._build_namespace(make_tool_call)

        try:
            async def run_code() -> Any:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    if 'await' in code:
                        wrapped = self._wrap_async(code)
                        exec(compile(wrapped, '<ptc>', 'exec'), namespace)
                        return await namespace.get('__ptc_main__', lambda: None)()
                    else:
                        exec(compile(code, '<ptc>', 'exec'), namespace)
                        return namespace.get('result')

            final_result = await asyncio.wait_for(
                run_code(),
                timeout=self.timeout
            )
            status = ExecutionStatus.COMPLETED

        except asyncio.TimeoutError:
            status = ExecutionStatus.TIMEOUT
            final_result = None
            stderr_capture.write("Execution timeout exceeded")
        except Exception as e:
            status = ExecutionStatus.FAILED
            final_result = None
            stderr_capture.write(f"Execution error: {e}\n{traceback.format_exc()}")

        execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
        total_result_bytes = sum(tc["result_size"] for tc in tool_calls)
        tokens_saved = total_result_bytes // 4

        return PTCExecutionResult(
            status=status,
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            final_result=final_result or stdout_capture.getvalue(),
            tool_calls_made=len(tool_calls),
            execution_time_ms=execution_time,
            context_tokens_saved=tokens_saved
        )

    def _build_namespace(self, make_call: Callable) -> Dict[str, Any]:
        """Build execution namespace with safe builtins and tool functions."""
        import math
        import datetime as dt
        import collections
        import statistics

        namespace = {
            'asyncio': asyncio,
            'json': json,
            'math': math,
            'datetime': dt,
            'collections': collections,
            'statistics': statistics,
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sum': sum,
            'min': min,
            'max': max,
            'sorted': sorted,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'True': True,
            'False': False,
            'None': None,
        }

        for tool in registry.get_ptc_enabled_tools():
            tool_name = tool.metadata.name
            func_name = tool_name.replace(".", "_").replace("-", "_")

            def make_func(name: str) -> Callable:
                async def tool_func(**kwargs: Any) -> Any:
                    return await make_call(name, **kwargs)
                return tool_func

            namespace[func_name] = make_func(tool_name)

        return namespace

    def _wrap_async(self, code: str) -> str:
        """Wrap code in async function."""
        indented = "\n".join("    " + line for line in code.split("\n"))
        return f"async def __ptc_main__():\n{indented}"

    def get_tool_definition(self) -> Dict[str, Any]:
        """Get the API definition for Code Execution tool."""
        return {
            "type": "code_execution_20250825",
            "name": "code_execution",
            "description": """Execute Python code to orchestrate multiple tool calls.

Use this for:
- Batch processing (loop through items, call tools in parallel)
- Data aggregation (sum, filter, transform tool results)
- Complex workflows (conditional logic, error handling)
- Reducing context usage (intermediate results stay in sandbox)

Available in sandbox:
- asyncio, json, math, datetime, collections, statistics
- All PTC-enabled tools as async functions

Tool results are processed in the sandbox - only your final
print() output or return value enters the model context.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
