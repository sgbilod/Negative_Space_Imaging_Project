# AI Agent Integration Guide

## Negative Space Imaging Project - Connecting to Claude

**Version:** 1.0.0
**Reference:** [Anthropic Engineering - Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)

---

## Overview

This guide explains how to integrate the NSIP Advanced Tool Use infrastructure with Claude via the Anthropic API. The implementation supports all three advanced features:

1. **Tool Search Tool** - Dynamic tool discovery
2. **Programmatic Tool Calling** - Code-based orchestration
3. **Tool Use Examples** - Improved parameter accuracy

---

## Quick Start

### 1. Get API Configuration

```python
from src.tools import get_api_configuration

config = get_api_configuration()

# config contains:
# {
#     "beta": "advanced-tool-use-2025-11-20",
#     "model": "claude-sonnet-4-5-20250929",
#     "tools": [...],  # Tool definitions
#     "headers": {"anthropic-beta": "advanced-tool-use-2025-11-20"}
# }
```

### 2. Make API Request

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model=config["model"],
    max_tokens=4096,
    tools=config["tools"],
    messages=[
        {"role": "user", "content": "Analyze the negative space in image img_abc123"}
    ],
    extra_headers=config["headers"]  # Required beta header
)
```

### 3. Handle Tool Calls

```python
from src.tools import registry

for content_block in response.content:
    if content_block.type == "tool_use":
        tool_name = content_block.name
        tool_input = content_block.input

        # Get and execute tool
        tool = registry.get_tool(tool_name)
        result = await tool.execute(**tool_input)

        # Continue conversation with result
        messages.append({
            "role": "assistant",
            "content": response.content
        })
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": content_block.id,
                "content": json.dumps(result)
            }]
        })
```

---

## Full Integration Example

```python
import asyncio
import json
import anthropic
from src.tools import (
    get_api_configuration,
    registry,
    PTCExecutor
)
from src.tools.registry.tool_search import ToolSearchTool


class NSIPAgent:
    """AI agent with NSIP tool integration."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.config = get_api_configuration()
        self.search_tool = ToolSearchTool(registry)
        self.ptc_executor = PTCExecutor()
        self.messages = []

    async def chat(self, user_message: str) -> str:
        """Process a user message and return response."""
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        while True:
            response = self.client.messages.create(
                model=self.config["model"],
                max_tokens=4096,
                tools=self.config["tools"],
                messages=self.messages,
                extra_headers=self.config["headers"]
            )

            # Check if done (no tool calls)
            has_tool_call = any(
                block.type == "tool_use"
                for block in response.content
            )

            if not has_tool_call:
                # Extract text response
                text = "".join(
                    block.text for block in response.content
                    if block.type == "text"
                )
                return text

            # Process tool calls
            self.messages.append({
                "role": "assistant",
                "content": response.content
            })

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await self._execute_tool(
                        block.name,
                        block.input,
                        block.id
                    )
                    tool_results.append(result)

            self.messages.append({
                "role": "user",
                "content": tool_results
            })

    async def _execute_tool(
        self,
        name: str,
        params: dict,
        tool_id: str
    ) -> dict:
        """Execute a tool and return result block."""
        try:
            if name == "tool_search":
                # Handle Tool Search Tool
                result = await self.search_tool.search(**params)
                # Load discovered tools into context
                for tool_result in result.results:
                    tool_def = registry.get_tool_definition(tool_result.tool_name)
                    if tool_def:
                        self.config["tools"].append(tool_def)
                content = json.dumps(result.to_dict())

            elif name == "code_execution":
                # Handle PTC
                result = await self.ptc_executor.execute(params["code"])
                content = result.final_result or result.stderr

            else:
                # Regular tool execution
                tool = registry.get_tool(name)
                if not tool:
                    content = json.dumps({"error": f"Tool not found: {name}"})
                else:
                    result = await tool.execute(**params)
                    content = json.dumps(result)

        except Exception as e:
            content = json.dumps({"error": str(e)})

        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": content
        }


# Usage
async def main():
    agent = NSIPAgent()

    # User asks for help
    response = await agent.chat(
        "I have 50 images to analyze for negative space patterns. "
        "Can you process them efficiently and summarize the results?"
    )

    print(response)

asyncio.run(main())
```

---

## Tool Search Integration

When Claude discovers it needs a specific capability:

```python
# Claude calls tool_search
response = await search_tool.search(
    query="export PDF report",
    limit=5
)

# Returns matching tools
# {
#     "query": "export PDF report",
#     "results": [
#         {"name": "export_report", "category": "EXPORT", "score": 12.5},
#         {"name": "generate_pdf", "category": "EXPORT", "score": 10.2}
#     ]
# }

# Load tool definitions into context
for result in response.results:
    tool_def = registry.get_tool_definition(result.tool_name)
    tools.append(tool_def)
```

---

## PTC Integration

When Claude generates orchestration code:

```python
# Claude generates code
code = """
images = await get_batch_images(batch_id="batch_001")
results = await asyncio.gather(*[
    analyze_negative_space(image_id=img["id"])
    for img in images
])
summary = {
    "total": len(results),
    "avg_ratio": sum(r["ratio"] for r in results) / len(results)
}
print(json.dumps(summary))
"""

# Execute in sandbox
result = await ptc_executor.execute(code)

# Only summary returned to Claude (not 50 individual results)
# result.final_result = {"total": 50, "avg_ratio": 0.34}
```

---

## Configuration Options

### Environment Variables

```bash
# Enable/disable features
NSIP_TOOL_SEARCH_ENABLED=true
NSIP_PTC_ENABLED=true
NSIP_PTC_TIMEOUT=60

# Environment
NSIP_ENVIRONMENT=production
```

### Feature Flags

```python
from src.tools.config.feature_flags import is_enabled

# Check feature availability
if is_enabled("TOOL_SEARCH_ENABLED"):
    config["tools"].append(search_tool.get_tool_definition())

if is_enabled("PTC_ENABLED"):
    config["tools"].append(ptc_executor.get_tool_definition())
```

---

## Error Handling

### Tool Not Found

```python
tool = registry.get_tool(name)
if not tool:
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": json.dumps({
            "error": f"Tool '{name}' not found",
            "suggestion": "Use tool_search to find available tools"
        }),
        "is_error": True
    }
```

### Validation Errors

```python
try:
    await tool.validate_input(**params)
except ValueError as e:
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": json.dumps({
            "error": f"Validation error: {str(e)}"
        }),
        "is_error": True
    }
```

### PTC Execution Errors

```python
result = await ptc_executor.execute(code)

if result.status == ExecutionStatus.FAILED:
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": json.dumps({
            "error": "Code execution failed",
            "stderr": result.stderr
        }),
        "is_error": True
    }
```

---

## Streaming Support

For streaming responses with tool use:

```python
with client.messages.stream(
    model=config["model"],
    max_tokens=4096,
    tools=config["tools"],
    messages=messages,
    extra_headers=config["headers"]
) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                print(event.delta.text, end="")
        elif event.type == "message_stop":
            # Check for tool calls
            message = stream.get_final_message()
            # Process tool calls...
```

---

## Best Practices

### 1. Cache Tool Definitions

```python
# Don't reload definitions for every request
_cached_config = None

def get_cached_config():
    global _cached_config
    if _cached_config is None:
        _cached_config = get_api_configuration()
    return _cached_config
```

### 2. Parallel Tool Execution

```python
# Execute multiple tool calls in parallel
async def execute_tools(tool_calls):
    tasks = [
        execute_tool(tc.name, tc.input, tc.id)
        for tc in tool_calls
    ]
    return await asyncio.gather(*tasks)
```

### 3. Log Tool Usage

```python
import logging
logger = logging.getLogger(__name__)

async def execute_tool(name, params, tool_id):
    logger.info(f"Executing tool: {name}")
    start = time.time()

    result = await tool.execute(**params)

    duration = time.time() - start
    logger.info(f"Tool {name} completed in {duration:.2f}s")

    return result
```

### 4. Handle Rate Limits

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def chat_with_retry(self, message):
    return await self.chat(message)
```

---

## Monitoring and Metrics

### Track Token Usage

```python
@dataclass
class UsageMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    context_tokens_saved: int = 0

def track_usage(response, ptc_result=None):
    metrics = UsageMetrics(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens
    )
    if ptc_result:
        metrics.context_tokens_saved = ptc_result.context_tokens_saved
    return metrics
```

### Registry Stats

```python
stats = registry.get_stats()
print(f"Total tools: {stats.total_tools}")
print(f"Always loaded: {stats.always_loaded}")
print(f"PTC enabled: {stats.ptc_enabled}")
print(f"Initial context tokens: {stats.estimated_always_loaded_tokens}")
```

---

## Related Documentation

- [Tool Architecture](./TOOL_ARCHITECTURE.md)
- [Tool Search Guide](./TOOL_SEARCH_GUIDE.md)
- [PTC Implementation](./PTC_IMPLEMENTATION.md)
- [Tool Examples Reference](./TOOL_EXAMPLES_REFERENCE.md)
