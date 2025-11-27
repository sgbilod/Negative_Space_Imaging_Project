"""
Tool Example Library.

Provides a centralized collection of tool usage examples
for improving parameter accuracy.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ExampleSet:
    """Collection of examples for a specific use case."""
    use_case: str
    description: str
    examples: List[Dict[str, Any]]


# Common workflow examples
WORKFLOW_EXAMPLES = [
    ExampleSet(
        use_case="single_image_analysis",
        description="Analyze a single image for negative space",
        examples=[
            {
                "step": 1,
                "tool": "analyze_negative_space",
                "params": {"image_id": "img_example", "mode": "basic"},
                "expected": {"success": True, "ratio": 0.35}
            }
        ]
    ),
    ExampleSet(
        use_case="batch_analysis_with_export",
        description="Analyze multiple images and export a report",
        examples=[
            {
                "step": 1,
                "tool": "batch_analyze",
                "params": {"image_ids": ["img_001", "img_002"], "parallel": True},
                "expected": {"success": True, "completed": 2}
            },
            {
                "step": 2,
                "tool": "export_report",
                "params": {"result_ids": ["res_001", "res_002"], "format": "pdf"},
                "expected": {"success": True, "download_url": "https://..."}
            }
        ]
    ),
    ExampleSet(
        use_case="ml_enhanced_detection",
        description="Use ML for anomaly detection in images",
        examples=[
            {
                "step": 1,
                "tool": "analyze_negative_space",
                "params": {"image_id": "img_sample", "mode": "ml_enhanced"},
                "expected": {"success": True, "confidence": 0.95}
            },
            {
                "step": 2,
                "tool": "predict_anomaly",
                "params": {"image_id": "img_sample", "confidence_threshold": 0.7},
                "expected": {"success": True, "is_anomaly": False}
            }
        ]
    )
]


def get_examples_for_tool(tool_name: str) -> List[Dict[str, Any]]:
    """Get all examples that use a specific tool."""
    examples = []
    for example_set in WORKFLOW_EXAMPLES:
        for example in example_set.examples:
            if example.get("tool") == tool_name:
                examples.append({
                    "use_case": example_set.use_case,
                    "params": example["params"],
                    "expected": example["expected"]
                })
    return examples


def get_workflow_examples() -> List[ExampleSet]:
    """Get all workflow examples."""
    return WORKFLOW_EXAMPLES
