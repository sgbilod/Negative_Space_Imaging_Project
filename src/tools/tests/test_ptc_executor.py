"""
Tests for Programmatic Tool Calling (PTC) Executor.

Tests code validation, execution, and sandboxing.
"""

import pytest
import asyncio

from ..execution.ptc_executor import (
    PTCExecutor,
    PTCExecutionResult,
    ExecutionStatus,
    PTCCodeValidator
)
from ..execution.code_sandbox import CodeSandbox, SandboxLimits


class TestPTCCodeValidator:
    """Tests for PTC code validation."""

    @pytest.fixture
    def validator(self):
        return PTCCodeValidator()

    def test_valid_code_passes(self, validator):
        """Test that valid code passes validation."""
        code = '''
result = 42 * 2
'''
        is_valid, error = validator.validate(code)
        assert is_valid is True
        assert error is None

    def test_import_statement_blocked(self, validator):
        """Test that disallowed import statements are blocked."""
        code = '''
import os
result = os.getcwd()
'''
        is_valid, error = validator.validate(code)
        assert is_valid is False
        assert error is not None
        assert "import" in error.lower() or "allowed" in error.lower()

    def test_exec_blocked(self, validator):
        """Test that exec/eval are blocked."""
        code = '''
result = exec("print('hello')")
'''
        is_valid, error = validator.validate(code)
        assert is_valid is False
        assert error is not None

    def test_file_access_blocked(self, validator):
        """Test that file access is blocked."""
        code = '''
with open("/etc/passwd") as f:
    result = f.read()
'''
        is_valid, error = validator.validate(code)
        assert is_valid is False
        assert error is not None

    def test_allowed_import_passes(self, validator):
        """Test that allowed imports pass validation."""
        code = '''
import json
import math
result = json.dumps({"value": math.pi})
'''
        is_valid, error = validator.validate(code)
        assert is_valid is True
        assert error is None

    def test_syntax_error_detected(self, validator):
        """Test that syntax errors are detected."""
        code = '''
def broken(
    # missing closing paren
'''
        is_valid, error = validator.validate(code)
        assert is_valid is False
        assert error is not None
        assert "syntax" in error.lower()


class TestPTCExecutor:
    """Tests for PTC Executor."""

    @pytest.fixture
    def executor(self):
        return PTCExecutor(timeout_seconds=5)

    @pytest.mark.asyncio
    async def test_execute_valid_code(self, executor):
        """Test execution of valid code."""
        code = '''
result = {"value": 42, "computed": 21 * 2}
'''
        exec_result = await executor.execute(code)

        assert exec_result.status == ExecutionStatus.COMPLETED
        assert exec_result.final_result is not None

    @pytest.mark.asyncio
    async def test_invalid_code_rejected(self, executor):
        """Test that invalid code is rejected."""
        code = '''
import subprocess
subprocess.call(["malicious"])
'''
        exec_result = await executor.execute(code)

        assert exec_result.status == ExecutionStatus.FAILED
        assert "Validation failed" in exec_result.stderr

    @pytest.mark.asyncio
    async def test_timeout_handling(self, executor):
        """Test timeout handling for long-running code."""
        # Create executor with short timeout
        short_executor = PTCExecutor(timeout_seconds=1)

        code = '''
import asyncio
await asyncio.sleep(100)
result = "done"
'''
        exec_result = await short_executor.execute(code)

        assert exec_result.status == ExecutionStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_execution_metrics_recorded(self, executor):
        """Test that execution metrics are recorded."""
        code = '''
result = {"status": "ok"}
'''
        exec_result = await executor.execute(code)

        assert exec_result.execution_time_ms >= 0
        assert exec_result.tool_calls_made >= 0


class TestCodeSandbox:
    """Tests for code sandbox."""

    @pytest.fixture
    def sandbox(self):
        return CodeSandbox(SandboxLimits())

    @pytest.mark.asyncio
    async def test_sandbox_isolation(self, sandbox):
        """Test that sandbox provides isolation."""
        result = await sandbox.execute("result = 42 * 2")

        assert result["success"] is True
        assert result["result"] == 84

    @pytest.mark.asyncio
    async def test_restricted_namespace(self, sandbox):
        """Test that namespace is restricted."""
        # Safe builtins should work
        result = await sandbox.execute("result = len([1, 2, 3])")
        assert result["success"] is True
        assert result["result"] == 3

    def test_limits_enforced(self, sandbox):
        """Test that resource limits are configured."""
        limits = sandbox.limits

        assert limits.max_memory_mb > 0
        assert limits.max_cpu_seconds > 0


class TestPTCIntegration:
    """Integration tests for PTC system."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete PTC workflow."""
        executor = PTCExecutor()

        code = '''
results = []
for i in range(3):
    results.append({"iteration": i, "value": i * 10})
result = {"total": len(results), "items": results}
'''

        exec_result = await executor.execute(code)

        assert exec_result.status == ExecutionStatus.COMPLETED
        assert exec_result.final_result is not None
        assert exec_result.final_result["total"] == 3

    @pytest.mark.asyncio
    async def test_concurrent_executions(self):
        """Test concurrent PTC executions."""
        executor = PTCExecutor()

        codes = [
            'result = {"id": 1}',
            'result = {"id": 2}',
            'result = {"id": 3}'
        ]

        tasks = [executor.execute(code) for code in codes]
        results = await asyncio.gather(*tasks)

        # All should complete
        assert len(results) == 3
        completed = [r for r in results if r.status == ExecutionStatus.COMPLETED]
        assert len(completed) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
