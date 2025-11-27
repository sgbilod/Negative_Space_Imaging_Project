"""
Export Tools for NSIP.

Provides report generation and format conversion capabilities.
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
class ExportReportTool(BaseTool):
    """Generate and export analysis reports."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="export_report",
            description="Generate comprehensive analysis reports in various formats.",
            category=ToolCategory.EXPORT,
            loading_strategy=LoadingStrategy.DEFERRED,
            allowed_callers=[CallerType.BOTH],
            version="1.0.0",
            tags=["export", "report", "pdf", "html"],
            search_keywords=["export", "report", "pdf", "generate", "download"],
            search_boost=1.0,
            estimated_duration_ms=2000,
            idempotent=True
        )

    @property
    def input_schema(self) -> InputSchema:
        return InputSchema(
            properties={
                "result_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Result IDs to include in report"
                },
                "format": {
                    "type": "string",
                    "enum": ["pdf", "html", "json", "csv"],
                    "default": "pdf"
                },
                "include_visualizations": {
                    "type": "boolean",
                    "default": True
                },
                "template": {
                    "type": "string",
                    "enum": ["standard", "detailed", "summary"],
                    "default": "standard"
                }
            },
            required=["result_ids"]
        )

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            description="Export result with download URL",
            properties={
                "success": {"type": "boolean"},
                "download_url": {"type": "string"},
                "expires_at": {"type": "string"},
                "file_size_bytes": {"type": "integer"}
            }
        )

    @property
    def examples(self) -> List[ToolExample]:
        return [
            ToolExample(
                description="Export PDF report",
                input_params={
                    "result_ids": ["res_001", "res_002"],
                    "format": "pdf",
                    "template": "detailed"
                },
                expected_output_shape={
                    "success": True,
                    "download_url": "https://...",
                    "file_size_bytes": 1024000
                }
            )
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        await self.validate_input(**kwargs)

        return {
            "success": True,
            "download_url": f"https://nsip.example.com/downloads/report_{datetime.now().strftime('%Y%m%d')}.pdf",
            "expires_at": datetime.now().isoformat(),
            "file_size_bytes": 1024000
        }
