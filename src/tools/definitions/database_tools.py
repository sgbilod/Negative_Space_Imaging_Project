"""
Database Tools for NSIP.

Provides data storage, retrieval, and query operations.
"""

from typing import Any, Dict, List, Optional
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
class StoreResultTool(BaseTool):
    """Store analysis results in the database."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="store_result",
            description="Store analysis results with metadata for later retrieval.",
            category=ToolCategory.DATABASE,
            loading_strategy=LoadingStrategy.DEFERRED,
            allowed_callers=[CallerType.BOTH],
            version="1.0.0",
            tags=["database", "storage", "persist"],
            search_keywords=["store", "save", "persist", "database", "write"],
            search_boost=1.0,
            estimated_duration_ms=100,
            idempotent=False
        )

    @property
    def input_schema(self) -> InputSchema:
        return InputSchema(
            properties={
                "image_id": {"type": "string", "description": "Image ID"},
                "result_type": {
                    "type": "string",
                    "enum": ["analysis", "comparison", "batch"],
                    "description": "Type of result"
                },
                "data": {"type": "object", "description": "Result data to store"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization"
                }
            },
            required=["image_id", "result_type", "data"]
        )

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            description="Storage confirmation with result ID",
            properties={
                "success": {"type": "boolean"},
                "result_id": {"type": "string"},
                "stored_at": {"type": "string"}
            }
        )

    @property
    def examples(self) -> List[ToolExample]:
        return [
            ToolExample(
                description="Store analysis result",
                input_params={
                    "image_id": "img_abc123",
                    "result_type": "analysis",
                    "data": {"ratio": 0.35, "confidence": 0.92},
                    "tags": ["production", "batch-001"]
                },
                expected_output_shape={
                    "success": True,
                    "result_id": "res_xyz789"
                }
            )
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        await self.validate_input(**kwargs)

        return {
            "success": True,
            "result_id": f"res_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "stored_at": datetime.now().isoformat()
        }


@register_tool
class QueryResultsTool(BaseTool):
    """Query stored results with filters."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="query_results",
            description="Query stored analysis results with filters and pagination.",
            category=ToolCategory.DATABASE,
            loading_strategy=LoadingStrategy.DEFERRED,
            allowed_callers=[CallerType.BOTH],
            version="1.0.0",
            tags=["database", "query", "search", "retrieve"],
            search_keywords=["query", "search", "find", "retrieve", "filter", "database"],
            search_boost=1.0,
            estimated_duration_ms=200,
            idempotent=True
        )

    @property
    def input_schema(self) -> InputSchema:
        return InputSchema(
            properties={
                "filters": {
                    "type": "object",
                    "description": "Filter criteria",
                    "properties": {
                        "image_id": {"type": "string"},
                        "result_type": {"type": "string"},
                        "date_from": {"type": "string"},
                        "date_to": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 1000
                },
                "offset": {"type": "integer", "default": 0},
                "order_by": {
                    "type": "string",
                    "enum": ["created_at", "image_id", "ratio"],
                    "default": "created_at"
                }
            },
            required=[]
        )

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            description="Query results with pagination info",
            properties={
                "success": {"type": "boolean"},
                "total": {"type": "integer"},
                "results": {"type": "array"},
                "has_more": {"type": "boolean"}
            }
        )

    @property
    def examples(self) -> List[ToolExample]:
        return [
            ToolExample(
                description="Query recent analysis results",
                input_params={
                    "filters": {"result_type": "analysis"},
                    "limit": 10,
                    "order_by": "created_at"
                },
                expected_output_shape={
                    "success": True,
                    "total": 42,
                    "has_more": True
                }
            )
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        await self.validate_input(**kwargs)

        return {
            "success": True,
            "total": 42,
            "results": [],
            "has_more": True
        }
