"""
NSIP Advanced Tool Use Infrastructure.

Implements three cutting-edge capabilities:
- Tool Search Tool: 85% token reduction
- Programmatic Tool Calling (PTC): 37% context reduction
- Tool Use Examples: 72% → 90% parameter accuracy

Beta Header: advanced-tool-use-2025-11-20
"""

from typing import Dict, Any, List, Optional

from .registry.tool_registry import registry, register_tool
from .registry.tool_search import ToolSearchTool
from .config.tool_config import config, AdvancedToolUseConfig

# Import all tool definitions to trigger registration
from .definitions import (
    imaging_tools,
    database_tools,
    security_tools,
    export_tools,
    ml_tools,
    specialized_tools,
    admin_tools
)


def get_api_configuration() -> Dict[str, Any]:
    """
    Get the complete API configuration for Anthropic API calls.

    Returns:
        Dict containing:
        - headers: Required beta headers
        - tools: Tool definitions (always-loaded only)
        - system_prompt_addition: Tool search tool docs
    """
    always_loaded = registry.get_always_loaded_definitions()

    tool_search = ToolSearchTool(registry)
    tool_search_def = tool_search.get_tool_definition()

    return {
        "headers": config.to_api_headers(),
        "tools": [tool_search_def] + always_loaded,
        "system_prompt_addition": _build_system_prompt_addition(),
        "config": {
            "tool_search_enabled": config.tool_search.enabled,
            "ptc_enabled": config.ptc.enabled,
            "examples_enabled": config.examples.enabled
        }
    }


def _build_system_prompt_addition() -> str:
    """Build system prompt addition for tool search."""
    stats = registry.get_stats()
    categories = list(stats.by_category.keys())

    return f"""
## Tool Search

You have access to a Tool Search Tool that can find relevant tools from a library of {stats.total_tools} tools.

**Available Categories:**
{chr(10).join(f'- {cat}' for cat in categories)}

**How to Use:**
1. Use the `tool_search_tool_regex_20251119` tool to search for relevant tools
2. Search by keywords, category, or natural language query
3. Only request tool definitions you need for the current task
4. This reduces context and improves performance

**Programmatic Tool Calling (PTC):**
When you need to call multiple tools in sequence, use the PTC executor with `code_execution_20250825`.
"""


def get_tool(name: str) -> Any:
    """Get a tool instance by name."""
    return registry.get_tool(name)


def search_tools(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for tools matching a query."""
    return registry.search(query, limit=limit)


__all__ = [
    "registry",
    "register_tool",
    "config",
    "get_api_configuration",
    "get_tool",
    "search_tools"
]
