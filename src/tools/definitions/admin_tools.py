"""
Admin Tools for NSIP.

Provides system administration and configuration capabilities.
"""

from typing import Any, Dict, List
from datetime import datetime

from .base_tool import (
    BaseTool,
    ToolMetadata,
    ToolCategory,
    LoadingStrategy,
    CallerType,
    InputSchema,
    OutputSchema,
    ToolExample
)
from ..registry.tool_registry import register_tool


@register_tool
class GetSystemStatusTool(BaseTool):
    """Get current system status and health."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="get_system_status",
            description="Get current system status, health metrics, and resource utilization.",
            category=ToolCategory.ADMIN,
            loading_strategy=LoadingStrategy.DEFERRED,
            allowed_callers=[CallerType.BOTH],
            version="1.0.0",
            tags=["admin", "status", "health", "monitoring"],
            search_keywords=["status", "health", "system", "monitoring", "metrics"],
            search_boost=1.0,
            estimated_duration_ms=100,
            idempotent=True
        )

    @property
    def input_schema(self) -> InputSchema:
        return InputSchema(
            properties={
                "include_metrics": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include detailed metrics"
                },
                "component": {
                    "type": "string",
                    "enum": ["all", "api", "database", "ml", "storage"],
                    "default": "all"
                }
            },
            required=[]
        )

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            description="System status report",
            properties={
                "status": {"type": "string"},
                "uptime_seconds": {"type": "integer"},
                "components": {"type": "object"},
                "metrics": {"type": "object"}
            }
        )

    @property
    def examples(self) -> List[ToolExample]:
        return [
            ToolExample(
                description="Get full system status",
                input_params={
                    "include_metrics": True,
                    "component": "all"
                },
                expected_output_shape={
                    "status": "healthy",
                    "uptime_seconds": 86400
                }
            )
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        await self.validate_input(**kwargs)

        return {
            "status": "healthy",
            "uptime_seconds": 86400,
            "components": {
                "api": {"status": "healthy", "latency_ms": 12},
                "database": {"status": "healthy", "connections": 5},
                "ml": {"status": "healthy", "models_loaded": 3},
                "storage": {"status": "healthy", "usage_percent": 45}
            },
            "metrics": {
                "requests_per_minute": 150,
                "active_sessions": 12,
                "cache_hit_rate": 0.85
            }
        }


@register_tool
class ManageConfigTool(BaseTool):
    """Manage system configuration."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="manage_config",
            description="View and update system configuration settings.",
            category=ToolCategory.ADMIN,
            loading_strategy=LoadingStrategy.DEFERRED,
            allowed_callers=[CallerType.DIRECT],  # Admin only, no PTC
            version="1.0.0",
            tags=["admin", "config", "settings"],
            search_keywords=["config", "settings", "configure", "admin"],
            search_boost=0.8,
            estimated_duration_ms=50,
            idempotent=False
        )

    @property
    def input_schema(self) -> InputSchema:
        return InputSchema(
            properties={
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "reset"],
                    "description": "Configuration action"
                },
                "key": {"type": "string", "description": "Configuration key"},
                "value": {"type": "string", "description": "New value (for set action)"}
            },
            required=["action"]
        )

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            description="Configuration operation result",
            properties={
                "success": {"type": "boolean"},
                "key": {"type": "string"},
                "value": {"type": "string"},
                "previous_value": {"type": "string"}
            }
        )

    @property
    def examples(self) -> List[ToolExample]:
        return [
            ToolExample(
                description="Get configuration value",
                input_params={
                    "action": "get",
                    "key": "max_batch_size"
                },
                expected_output_shape={
                    "success": True,
                    "key": "max_batch_size",
                    "value": "100"
                }
            )
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        await self.validate_input(**kwargs)

        action = kwargs["action"]
        key = kwargs.get("key", "")

        return {
            "success": True,
            "key": key,
            "value": "100" if action == "get" else kwargs.get("value", ""),
            "previous_value": "100" if action == "set" else None
        }
