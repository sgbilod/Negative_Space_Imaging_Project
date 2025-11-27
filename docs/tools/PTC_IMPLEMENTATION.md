# Programmatic Tool Calling (PTC) Implementation

## Negative Space Imaging Project - Code-Based Tool Orchestration

**Version:** 1.0.0
**Reference:** [Anthropic Engineering - Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)

---

## Overview

Programmatic Tool Calling (PTC) enables Claude to orchestrate multiple tool calls via Python code executed in a sandboxed environment. This reduces context pollution by 37% and enables sophisticated workflows.

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Context Reduction** | Intermediate results stay in sandbox, not in Claude's context |
| **Parallel Execution** | Execute multiple tools simultaneously via asyncio |
| **Complex Logic** | Loops, conditionals, aggregation in code |
| **Error Handling** | Try/except for graceful failure recovery |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                 PTC Execution Flow                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Claude generates Python code                               │
│     ┌─────────────────────────────────────────┐               │
│     │  images = await get_batch(batch_id)     │               │
│     │  results = await asyncio.gather(*[      │               │
│     │      analyze(img["id"]) for img         │               │
│     │  ])                                     │               │
│     │  print(json.dumps(summary))             │               │
│     └─────────────────────────────────────────┘               │
│                          │                                     │
│                          ▼                                     │
│  2. Code validated by PTCCodeValidator                         │
│     • Check for forbidden patterns                            │
│     • Verify allowed imports only                             │
│     • Parse AST for security                                  │
│                          │                                     │
│                          ▼                                     │
│  3. Execute in sandboxed environment                           │
│     • Restricted builtins                                     │
│     • Tool functions available                                │
│     • Asyncio event loop                                      │
│                          │                                     │
│                          ▼                                     │
│  4. Tool calls made (results NOT in context)                   │
│     ┌─────────────────────────────────────────┐               │
│     │  Tool 1 → Result 1 (in sandbox only)    │               │
│     │  Tool 2 → Result 2 (in sandbox only)    │               │
│     │  Tool 3 → Result 3 (in sandbox only)    │               │
│     └─────────────────────────────────────────┘               │
│                          │                                     │
│                          ▼                                     │
│  5. Only final print() output returns to Claude                │
│     ┌─────────────────────────────────────────┐               │
│     │  {"total": 50, "avg_ratio": 0.34, ...}  │               │
│     └─────────────────────────────────────────┘               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## API Definition

The Code Execution tool is automatically included in the initial context:

```python
{
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
```

---

## Tool Configuration for PTC

### Enabling PTC for a Tool

```python
@register_tool
class MyTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="my_tool",
            allowed_callers=[CallerType.BOTH],  # Enable PTC
            # ...
        )
```

### Caller Types

```python
class CallerType(Enum):
    DIRECT = "direct"                           # Only via API
    CODE_EXECUTION = "code_execution_20250825"  # Only via PTC
    BOTH = "both"                               # Either method
```

### Output Schema (Critical for PTC)

Claude needs the output schema to write correct parsing code:

```python
@property
def output_schema(self) -> OutputSchema:
    return OutputSchema(
        description="Analysis results with detected regions",
        properties={
            "success": {
                "type": "boolean",
                "description": "Whether analysis completed"
            },
            "ratio": {
                "type": "number",
                "description": "Negative space ratio (0.0-1.0)"
            },
            "regions": {
                "type": "array",
                "description": "Detected regions"
            }
        }
    )
```

---

## Code Validation

The `PTCCodeValidator` ensures safe execution:

### Forbidden Patterns

```python
FORBIDDEN_PATTERNS = (
    'import os', 'import sys', 'import subprocess',
    '__import__', 'eval(', 'exec(', 'open(',
    'compile(', 'globals(', 'locals(', 'getattr(',
    'setattr(', 'delattr(', '__builtins__'
)
```

### Allowed Imports

```python
ALLOWED_IMPORTS = frozenset({
    'json', 'math', 'datetime', 'collections',
    'itertools', 'functools', 'typing', 're',
    'asyncio', 'statistics'
})
```

---

## Orchestration Templates

Common patterns are available as templates:

### Batch Processing

```python
# Process images in batches with parallel execution
images = await get_batch_images(batch_id="batch_123")
results = []
errors = []

for i in range(0, len(images), 10):  # Batch size 10
    batch = images[i:i+10]
    batch_results = await asyncio.gather(*[
        analyze_negative_space(image_id=img["id"])
        for img in batch
    ], return_exceptions=True)

    for img, result in zip(batch, batch_results):
        if isinstance(result, Exception):
            errors.append({"id": img["id"], "error": str(result)})
        else:
            results.append(result)

summary = {
    "total": len(images),
    "successful": len(results),
    "failed": len(errors),
    "avg_ratio": sum(r["ratio"] for r in results) / len(results) if results else 0
}
print(json.dumps(summary))
```

### Map-Reduce

```python
# Map-reduce for aggregation
images = await get_batch_images(batch_id="batch_123")

# Map phase - parallel analysis
analyses = await asyncio.gather(*[
    analyze_negative_space(image_id=img["id"], mode="advanced")
    for img in images
])

# Reduce phase - aggregate results
summary = {
    "total_images": len(analyses),
    "avg_ratio": sum(a["ratio"] for a in analyses) / len(analyses),
    "max_ratio": max(a["ratio"] for a in analyses),
    "min_ratio": min(a["ratio"] for a in analyses),
    "anomalies": [a for a in analyses if a["anomaly_score"] > 0.8]
}
print(json.dumps(summary))
```

### Pipeline Processing

```python
# Sequential pipeline
image_id = "img_123"

# Step 1: Analyze
analysis = await analyze_negative_space(
    image_id=image_id,
    mode="ml_enhanced",
    include_visualization=True
)

# Step 2: Generate report
report = await export_report(
    analysis_id=analysis["id"],
    format="pdf"
)

# Step 3: Store
storage = await store_artifact(
    artifact_type="report",
    data=report
)

result = {
    "analysis_id": analysis["id"],
    "report_url": report["download_url"],
    "storage_id": storage["id"]
}
print(json.dumps(result))
```

---

## Execution Results

```python
@dataclass
class PTCExecutionResult:
    status: ExecutionStatus      # COMPLETED, FAILED, TIMEOUT
    stdout: str                  # Captured print output
    stderr: str                  # Error output
    final_result: Any            # Return value or stdout
    tool_calls_made: int         # Number of tools called
    execution_time_ms: int       # Total execution time
    context_tokens_saved: int    # Estimated tokens not added to context
```

---

## Security Features

### Code Sandbox

```python
class CodeSandbox:
    """
    Additional isolation for PTC execution:
    - Resource limits (memory, CPU)
    - Restricted builtins
    - Iteration limits
    - Output size limits
    """

    @dataclass
    class SandboxLimits:
        max_memory_mb: int = 256
        max_cpu_seconds: int = 30
        max_output_bytes: int = 1_000_000  # 1MB
        max_recursion_depth: int = 100
        max_iterations: int = 1_000_000
```

### Restricted Namespace

```python
# Safe builtins available
SAFE_BUILTINS = {
    'abs', 'all', 'any', 'bin', 'bool', 'bytes',
    'chr', 'dict', 'divmod', 'enumerate', 'filter',
    'float', 'format', 'frozenset', 'hash', 'hex',
    'int', 'iter', 'len', 'list', 'map', 'max', 'min',
    'next', 'oct', 'ord', 'pow', 'print', 'range',
    'reversed', 'round', 'set', 'slice', 'sorted',
    'str', 'sum', 'tuple', 'zip'
}
```

---

## Context Token Savings

### Traditional Approach

```
User: Analyze 50 images
Claude: [50 separate tool calls]
Context: +50 tool results = ~25K tokens added
```

### PTC Approach

```
User: Analyze 50 images
Claude: [1 code execution with 50 parallel calls]
Context: +1 summary result = ~500 tokens added
Savings: ~24.5K tokens (98%)
```

---

## Usage Example

```python
from src.tools.execution.ptc_executor import PTCExecutor

executor = PTCExecutor(timeout_seconds=60)

code = """
images = await get_batch_images(batch_id="batch_001")
results = await asyncio.gather(*[
    analyze_negative_space(image_id=img["id"])
    for img in images
])
avg = sum(r["ratio"] for r in results) / len(results)
print(json.dumps({"count": len(results), "avg_ratio": avg}))
"""

result = await executor.execute(code)

print(result.status)              # ExecutionStatus.COMPLETED
print(result.final_result)        # {"count": 50, "avg_ratio": 0.34}
print(result.tool_calls_made)     # 50
print(result.context_tokens_saved)  # ~12500
```

---

## Best Practices

### 1. Always Print JSON Output

```python
# ✅ Good - structured output
print(json.dumps({"status": "complete", "count": 50}))

# ❌ Bad - unstructured output
print("Done! Processed 50 images")
```

### 2. Handle Exceptions

```python
# ✅ Good - error handling
results = await asyncio.gather(*tasks, return_exceptions=True)
for result in results:
    if isinstance(result, Exception):
        errors.append(str(result))
```

### 3. Use Batching for Large Datasets

```python
# ✅ Good - process in batches
for i in range(0, len(items), 10):
    batch = items[i:i+10]
    await asyncio.gather(*[process(x) for x in batch])
```

### 4. Return Summary, Not All Data

```python
# ✅ Good - summary only
print(json.dumps({"total": 1000, "avg": 0.5, "anomalies": 3}))

# ❌ Bad - full data dump
print(json.dumps(all_1000_results))
```

---

## Related Documentation

- [Tool Architecture](./TOOL_ARCHITECTURE.md)
- [Tool Search Guide](./TOOL_SEARCH_GUIDE.md)
- [Tool Examples Reference](./TOOL_EXAMPLES_REFERENCE.md)
