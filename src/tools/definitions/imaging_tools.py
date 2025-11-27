"""
Core Imaging Tools with Advanced Tool Use Features.

Implements all three features:
- Tool Search Tool: Proper metadata and keywords
- PTC: Output schemas and allowed_callers
- Tool Use Examples: Concrete invocation examples
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
class AnalyzeNegativeSpaceTool(BaseTool):
    """Primary negative space analysis tool."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="analyze_negative_space",
            description="""Analyze an image to detect and quantify negative space regions.

Performs multi-scale analysis to identify areas of visual absence or
background that create compositional balance. Supports basic edge detection,
advanced pattern recognition, and ML-enhanced deep analysis modes.""",
            category=ToolCategory.IMAGING_CORE,
            loading_strategy=LoadingStrategy.ALWAYS_LOADED,
            allowed_callers=[CallerType.BOTH],
            version="2.0.0",
            tags=["imaging", "analysis", "negative-space", "composition"],
            search_keywords=[
                "negative space", "image analysis", "composition",
                "visual balance", "background detection"
            ],
            search_boost=2.0,
            estimated_duration_ms=500,
            idempotent=True,
            supports_batch=True,
            max_batch_size=50
        )

    @property
    def input_schema(self) -> InputSchema:
        return InputSchema(
            properties={
                "image_id": {
                    "type": "string",
                    "description": "UUID of the uploaded image",
                    "pattern": "^img_[a-zA-Z0-9]{8,}$"
                },
                "mode": {
                    "type": "string",
                    "enum": ["basic", "advanced", "ml_enhanced"],
                    "default": "basic",
                    "description": "Analysis depth"
                },
                "threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Detection sensitivity"
                },
                "include_visualization": {
                    "type": "boolean",
                    "default": False,
                    "description": "Generate annotated output image"
                },
                "roi": {
                    "type": "object",
                    "description": "Optional region of interest",
                    "properties": {
                        "x": {"type": "integer", "minimum": 0},
                        "y": {"type": "integer", "minimum": 0},
                        "width": {"type": "integer", "minimum": 1},
                        "height": {"type": "integer", "minimum": 1}
                    }
                }
            },
            required=["image_id"]
        )

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            description="Analysis results with detected regions and metrics",
            properties={
                "success": {"type": "boolean", "description": "Analysis completed"},
                "image_id": {"type": "string", "description": "ID of analyzed image"},
                "ratio": {"type": "number", "description": "Negative space ratio (0.0-1.0)"},
                "confidence": {"type": "number", "description": "Confidence score"},
                "regions": {"type": "array", "description": "Detected regions"},
                "anomaly_score": {"type": "number", "description": "Anomaly score"},
                "time_ms": {"type": "integer", "description": "Processing time"},
                "visualization_url": {"type": "string", "description": "Annotated image URL"}
            }
        )

    @property
    def examples(self) -> List[ToolExample]:
        return [
            ToolExample(
                description="Basic analysis with defaults",
                input_params={"image_id": "img_abc12345"},
                expected_output_shape={"success": True, "ratio": 0.34, "confidence": 0.92},
                notes="Fastest mode for quick checks"
            ),
            ToolExample(
                description="Advanced analysis with custom threshold",
                input_params={
                    "image_id": "img_def67890",
                    "mode": "advanced",
                    "threshold": 0.3
                },
                expected_output_shape={
                    "success": True,
                    "ratio": 0.45,
                    "regions": [{"id": "r1", "area_percent": 0.23}]
                },
                notes="Good balance of speed and accuracy"
            ),
            ToolExample(
                description="ML-enhanced with visualization",
                input_params={
                    "image_id": "img_ghi11111",
                    "mode": "ml_enhanced",
                    "threshold": 0.7,
                    "include_visualization": True,
                    "roi": {"x": 100, "y": 100, "width": 500, "height": 400}
                },
                expected_output_shape={
                    "success": True,
                    "ratio": 0.28,
                    "confidence": 0.98,
                    "visualization_url": "https://..."
                },
                notes="Most accurate for detailed analysis"
            )
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute negative space analysis."""
        await self.validate_input(**kwargs)

        image_id = kwargs["image_id"]
        mode = kwargs.get("mode", "basic")
        include_viz = kwargs.get("include_visualization", False)

        start_time = datetime.now()

        # Placeholder implementation
        result = {
            "success": True,
            "image_id": image_id,
            "ratio": 0.34,
            "confidence": 0.92,
            "regions": [],
            "anomaly_score": 0.12,
            "time_ms": int((datetime.now() - start_time).total_seconds() * 1000)
        }

        if include_viz:
            result["visualization_url"] = f"https://nsip.example.com/viz/{image_id}"

        return result


@register_tool
class BatchAnalyzeTool(BaseTool):
    """Batch processing for multiple images."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="batch_analyze",
            description="Process multiple images in a single request with parallel execution.",
            category=ToolCategory.IMAGING_CORE,
            loading_strategy=LoadingStrategy.ALWAYS_LOADED,
            allowed_callers=[CallerType.BOTH],
            version="1.0.0",
            tags=["batch", "parallel", "imaging"],
            search_keywords=["batch", "multiple images", "parallel processing"],
            search_boost=1.5,
            supports_batch=True,
            max_batch_size=100
        )

    @property
    def input_schema(self) -> InputSchema:
        return InputSchema(
            properties={
                "image_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of image IDs to process",
                    "minItems": 1,
                    "maxItems": 100
                },
                "mode": {
                    "type": "string",
                    "enum": ["basic", "advanced", "ml_enhanced"],
                    "default": "basic"
                },
                "parallel": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable parallel processing"
                }
            },
            required=["image_ids"]
        )

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            description="Batch processing results",
            properties={
                "success": {"type": "boolean"},
                "total": {"type": "integer"},
                "completed": {"type": "integer"},
                "failed": {"type": "integer"},
                "results": {"type": "array"},
                "summary": {"type": "object"}
            }
        )

    @property
    def examples(self) -> List[ToolExample]:
        return [
            ToolExample(
                description="Process batch of images",
                input_params={
                    "image_ids": ["img_001", "img_002", "img_003"],
                    "mode": "basic",
                    "parallel": True
                },
                expected_output_shape={
                    "success": True,
                    "total": 3,
                    "completed": 3,
                    "failed": 0
                }
            )
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        await self.validate_input(**kwargs)
        image_ids = kwargs["image_ids"]

        return {
            "success": True,
            "total": len(image_ids),
            "completed": len(image_ids),
            "failed": 0,
            "results": [{"image_id": img, "ratio": 0.35} for img in image_ids],
            "summary": {"avg_ratio": 0.35}
        }


@register_tool
class CompareImagesTool(BaseTool):
    """Compare negative space between two images."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="compare_images",
            description="Compare negative space patterns between two images to identify similarities and differences.",
            category=ToolCategory.IMAGING_CORE,
            loading_strategy=LoadingStrategy.ALWAYS_LOADED,
            allowed_callers=[CallerType.BOTH],
            version="1.0.0",
            tags=["comparison", "imaging", "analysis"],
            search_keywords=["compare", "difference", "similarity", "images"],
            search_boost=1.3,
            estimated_duration_ms=800,
            idempotent=True
        )

    @property
    def input_schema(self) -> InputSchema:
        return InputSchema(
            properties={
                "image_id_1": {
                    "type": "string",
                    "description": "First image ID"
                },
                "image_id_2": {
                    "type": "string",
                    "description": "Second image ID"
                },
                "comparison_mode": {
                    "type": "string",
                    "enum": ["structural", "statistical", "perceptual"],
                    "default": "structural",
                    "description": "Type of comparison"
                }
            },
            required=["image_id_1", "image_id_2"]
        )

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            description="Comparison results with similarity metrics",
            properties={
                "success": {"type": "boolean"},
                "similarity_score": {"type": "number", "description": "0.0-1.0 similarity"},
                "difference_regions": {"type": "array"},
                "structural_similarity": {"type": "number"},
                "perceptual_hash_distance": {"type": "integer"}
            }
        )

    @property
    def examples(self) -> List[ToolExample]:
        return [
            ToolExample(
                description="Compare two images structurally",
                input_params={
                    "image_id_1": "img_aaa111",
                    "image_id_2": "img_bbb222",
                    "comparison_mode": "structural"
                },
                expected_output_shape={
                    "success": True,
                    "similarity_score": 0.78,
                    "structural_similarity": 0.82
                }
            )
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        await self.validate_input(**kwargs)

        return {
            "success": True,
            "similarity_score": 0.78,
            "difference_regions": [],
            "structural_similarity": 0.82,
            "perceptual_hash_distance": 12
        }


@register_tool
class ExtractRegionsTool(BaseTool):
    """Extract and export detected negative space regions."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="extract_regions",
            description="Extract individual negative space regions as separate images or masks.",
            category=ToolCategory.IMAGING_ADVANCED,
            loading_strategy=LoadingStrategy.DEFERRED,
            allowed_callers=[CallerType.BOTH],
            version="1.0.0",
            tags=["extraction", "regions", "masks", "imaging"],
            search_keywords=["extract", "regions", "mask", "segment", "crop"],
            search_boost=1.2,
            estimated_duration_ms=1200,
            idempotent=True
        )

    @property
    def input_schema(self) -> InputSchema:
        return InputSchema(
            properties={
                "image_id": {
                    "type": "string",
                    "description": "Source image ID"
                },
                "region_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional specific region IDs to extract"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["png", "svg", "mask"],
                    "default": "png"
                },
                "include_metadata": {
                    "type": "boolean",
                    "default": True
                }
            },
            required=["image_id"]
        )

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            description="Extracted regions with URLs and metadata",
            properties={
                "success": {"type": "boolean"},
                "regions_extracted": {"type": "integer"},
                "regions": {"type": "array"},
                "total_area_percent": {"type": "number"}
            }
        )

    @property
    def examples(self) -> List[ToolExample]:
        return [
            ToolExample(
                description="Extract all regions as PNG",
                input_params={
                    "image_id": "img_xyz789",
                    "output_format": "png",
                    "include_metadata": True
                },
                expected_output_shape={
                    "success": True,
                    "regions_extracted": 5,
                    "total_area_percent": 0.42
                }
            )
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        await self.validate_input(**kwargs)

        return {
            "success": True,
            "regions_extracted": 5,
            "regions": [
                {"id": "r1", "url": "https://...", "area_percent": 0.15},
                {"id": "r2", "url": "https://...", "area_percent": 0.12}
            ],
            "total_area_percent": 0.42
        }
