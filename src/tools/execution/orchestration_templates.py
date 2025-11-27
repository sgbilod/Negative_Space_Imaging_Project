"""
Orchestration Templates for Programmatic Tool Calling.

Provides common patterns for tool orchestration that Claude can use
via the PTC executor. These templates reduce boilerplate and ensure
best practices for common workflows.

Reference: Anthropic Advanced Tool Use - Programmatic Tool Calling
"""

from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
import asyncio
import json


class TemplateType(Enum):
    """Types of orchestration templates."""
    BATCH_PROCESS = "batch_process"
    MAP_REDUCE = "map_reduce"
    PIPELINE = "pipeline"
    PARALLEL_AGGREGATE = "parallel_aggregate"
    CONDITIONAL_BRANCH = "conditional_branch"
    RETRY_WITH_FALLBACK = "retry_with_fallback"


@dataclass
class TemplateResult:
    """Result from template execution."""
    success: bool
    results: List[Any]
    errors: List[str]
    total_items: int
    successful_items: int
    execution_time_ms: int


class OrchestrationTemplates:
    """
    Collection of orchestration templates for PTC.

    These templates provide optimized patterns that:
    - Reduce context pollution
    - Enable parallel execution
    - Handle errors gracefully
    - Return aggregated results only
    """

    @staticmethod
    async def batch_process(
        items: List[Any],
        process_func: Callable[[Any], Awaitable[Any]],
        batch_size: int = 10,
        continue_on_error: bool = True
    ) -> TemplateResult:
        """
        Process items in batches with parallelism.

        Args:
            items: List of items to process
            process_func: Async function to apply to each item
            batch_size: Number of items per batch
            continue_on_error: Whether to continue if an item fails

        Example PTC code:
            results = await batch_process(
                items=image_ids,
                process_func=lambda img: analyze_negative_space(image_id=img),
                batch_size=10
            )
        """
        import time
        start = time.time()
        results = []
        errors = []
        successful = 0

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            tasks = [process_func(item) for item in batch]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for item, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    errors.append(f"Item {item}: {str(result)}")
                    if not continue_on_error:
                        break
                else:
                    results.append(result)
                    successful += 1

        return TemplateResult(
            success=len(errors) == 0,
            results=results,
            errors=errors,
            total_items=len(items),
            successful_items=successful,
            execution_time_ms=int((time.time() - start) * 1000)
        )

    @staticmethod
    async def map_reduce(
        items: List[Any],
        map_func: Callable[[Any], Awaitable[Any]],
        reduce_func: Callable[[List[Any]], Any]
    ) -> Any:
        """
        Apply map-reduce pattern for aggregation.

        Args:
            items: List of items to process
            map_func: Async function to apply to each item
            reduce_func: Function to aggregate results

        Example PTC code:
            total_ratio = await map_reduce(
                items=image_ids,
                map_func=lambda img: analyze_negative_space(image_id=img),
                reduce_func=lambda results: sum(r['ratio'] for r in results) / len(results)
            )
        """
        tasks = [map_func(item) for item in items]
        mapped = await asyncio.gather(*tasks)
        return reduce_func(mapped)

    @staticmethod
    async def pipeline(
        initial_data: Any,
        steps: List[Callable[[Any], Awaitable[Any]]]
    ) -> Any:
        """
        Execute a sequence of processing steps.

        Args:
            initial_data: Starting data for the pipeline
            steps: List of async functions to apply in sequence

        Example PTC code:
            result = await pipeline(
                initial_data="img_123",
                steps=[
                    lambda img_id: analyze_negative_space(image_id=img_id),
                    lambda analysis: export_report(data=analysis, format="json"),
                    lambda report: store_result(report_id=report['id'])
                ]
            )
        """
        data = initial_data
        for step in steps:
            data = await step(data)
        return data

    @staticmethod
    async def parallel_aggregate(
        queries: Dict[str, Callable[[], Awaitable[Any]]],
        aggregator: Optional[Callable[[Dict[str, Any]], Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute multiple queries in parallel and aggregate.

        Args:
            queries: Dict of name -> async function
            aggregator: Optional function to combine results

        Example PTC code:
            stats = await parallel_aggregate(
                queries={
                    'total_images': lambda: count_images(filter='all'),
                    'anomalies': lambda: count_images(filter='anomaly'),
                    'avg_ratio': lambda: get_avg_ratio()
                },
                aggregator=lambda r: {**r, 'anomaly_rate': r['anomalies'] / r['total_images']}
            )
        """
        tasks = {name: func() for name, func in queries.items()}
        results = {}

        for name, task in tasks.items():
            results[name] = await task

        if aggregator:
            return aggregator(results)
        return results

    @staticmethod
    async def conditional_branch(
        condition_func: Callable[[], Awaitable[bool]],
        if_true: Callable[[], Awaitable[Any]],
        if_false: Callable[[], Awaitable[Any]]
    ) -> Any:
        """
        Execute conditional logic based on a check.

        Example PTC code:
            result = await conditional_branch(
                condition_func=lambda: check_image_exists(image_id="img_123"),
                if_true=lambda: analyze_negative_space(image_id="img_123"),
                if_false=lambda: {'error': 'Image not found'}
            )
        """
        condition = await condition_func()
        if condition:
            return await if_true()
        else:
            return await if_false()

    @staticmethod
    async def retry_with_fallback(
        primary_func: Callable[[], Awaitable[Any]],
        fallback_func: Callable[[], Awaitable[Any]],
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0
    ) -> Any:
        """
        Retry an operation with fallback on failure.

        Example PTC code:
            result = await retry_with_fallback(
                primary_func=lambda: analyze_with_ml(image_id="img_123"),
                fallback_func=lambda: analyze_basic(image_id="img_123"),
                max_retries=2
            )
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return await primary_func()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay_seconds)

        # Try fallback
        try:
            return await fallback_func()
        except Exception as fallback_error:
            raise RuntimeError(
                f"Primary failed after {max_retries} attempts ({last_error}), "
                f"fallback also failed: {fallback_error}"
            )


# Template code snippets for PTC code generation
TEMPLATE_SNIPPETS = {
    TemplateType.BATCH_PROCESS: '''
# Batch process images with parallel execution
images = await get_batch_images(batch_id="{batch_id}")
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
            errors.append({{"id": img["id"], "error": str(result)}})
        else:
            results.append(result)

summary = {{
    "total": len(images),
    "successful": len(results),
    "failed": len(errors),
    "avg_ratio": sum(r["ratio"] for r in results) / len(results) if results else 0
}}
print(json.dumps(summary))
''',

    TemplateType.MAP_REDUCE: '''
# Map-reduce for aggregation
images = await get_batch_images(batch_id="{batch_id}")

# Map phase - parallel analysis
analyses = await asyncio.gather(*[
    analyze_negative_space(image_id=img["id"], mode="advanced")
    for img in images
])

# Reduce phase - aggregate results
summary = {{
    "total_images": len(analyses),
    "total_ratio": sum(a["ratio"] for a in analyses),
    "avg_ratio": sum(a["ratio"] for a in analyses) / len(analyses),
    "max_ratio": max(a["ratio"] for a in analyses),
    "min_ratio": min(a["ratio"] for a in analyses),
    "anomalies": [a for a in analyses if a["anomaly_score"] > 0.8]
}}
print(json.dumps(summary))
''',

    TemplateType.PIPELINE: '''
# Pipeline processing
image_id = "{image_id}"

# Step 1: Analyze
analysis = await analyze_negative_space(
    image_id=image_id,
    mode="ml_enhanced",
    include_visualization=True
)

# Step 2: Generate report
report = await export_report(
    analysis_id=analysis["id"],
    format="pdf",
    include_charts=True
)

# Step 3: Store and notify
storage = await store_artifact(
    artifact_type="report",
    data=report
)

result = {{
    "analysis_id": analysis["id"],
    "report_url": report["download_url"],
    "storage_id": storage["id"]
}}
print(json.dumps(result))
'''
}


def get_template_snippet(template_type: TemplateType, **kwargs: Any) -> str:
    """
    Get a code snippet for a template type.

    Args:
        template_type: Type of template
        **kwargs: Template variables

    Returns:
        Python code string ready for PTC execution
    """
    template = TEMPLATE_SNIPPETS.get(template_type, "")
    return template.format(**kwargs)


def list_available_templates() -> List[Dict[str, str]]:
    """List all available orchestration templates."""
    return [
        {
            "type": t.value,
            "description": getattr(OrchestrationTemplates, t.value.lower()).__doc__
        }
        for t in TemplateType
        if hasattr(OrchestrationTemplates, t.value.lower())
    ]
