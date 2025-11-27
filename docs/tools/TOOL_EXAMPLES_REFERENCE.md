# Tool Use Examples Reference

## Negative Space Imaging Project - Improving Parameter Accuracy

**Version:** 1.0.0
**Reference:** [Anthropic Engineering - Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)

---

## Overview

Tool Use Examples improve Claude's parameter accuracy from 72% to 90% by providing concrete invocation examples. Each tool includes examples showing minimal, typical, and full usage patterns.

### Impact

| Metric | Without Examples | With Examples | Improvement |
|--------|-----------------|---------------|-------------|
| Parameter Accuracy | 72% | 90% | +18% |
| Correct Types | 85% | 98% | +13% |
| Valid Enums | 80% | 99% | +19% |
| Complete Calls | 70% | 95% | +25% |

---

## Example Structure

### ToolExample Class

```python
@dataclass
class ToolExample:
    description: str                              # What this example demonstrates
    input_params: Dict[str, Any]                  # Actual parameters to use
    expected_output_shape: Optional[Dict[str, Any]] = None  # Sample output
    notes: Optional[str] = None                   # Usage tips

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "input": self.input_params,
            "expected_output": self.expected_output_shape,
            "notes": self.notes
        }
```

---

## Example Categories

### 1. Minimal Example

Shows required parameters only - the simplest valid invocation.

```python
ToolExample(
    description="Basic analysis with defaults",
    input_params={
        "image_id": "img_abc12345"  # Only required param
    },
    expected_output_shape={
        "success": True,
        "ratio": 0.34,
        "confidence": 0.92
    },
    notes="Fastest mode, suitable for quick checks"
)
```

### 2. Typical Example

Shows common use case with frequently-used optional parameters.

```python
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
        "confidence": 0.95,
        "regions": [{"id": "r1", "area_percent": 0.23}]
    },
    notes="Good balance of speed and accuracy"
)
```

### 3. Full Example

Shows all parameters for comprehensive usage.

```python
ToolExample(
    description="ML-enhanced analysis with visualization",
    input_params={
        "image_id": "img_ghi11111",
        "mode": "ml_enhanced",
        "threshold": 0.7,
        "include_visualization": True,
        "roi": {
            "x": 100,
            "y": 100,
            "width": 500,
            "height": 400
        }
    },
    expected_output_shape={
        "success": True,
        "ratio": 0.28,
        "confidence": 0.98,
        "regions": [],
        "anomaly_score": 0.15,
        "visualization_url": "https://..."
    },
    notes="Most accurate, use for detailed analysis"
)
```

---

## Adding Examples to Tools

```python
@register_tool
class AnalyzeNegativeSpaceTool(BaseTool):
    @property
    def examples(self) -> List[ToolExample]:
        return [
            # Example 1: Minimal
            ToolExample(
                description="Basic analysis with defaults",
                input_params={"image_id": "img_abc12345"},
                expected_output_shape={"success": True, "ratio": 0.34},
                notes="Fastest mode for quick checks"
            ),

            # Example 2: Typical
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

            # Example 3: Full
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
```

---

## API Integration

Examples are automatically included in tool definitions:

```python
def to_api_definition(self) -> Dict[str, Any]:
    return {
        "name": self.metadata.name,
        "description": self._build_full_description(),
        "input_schema": self.input_schema.to_json_schema(),
        # Examples included here
        "input_examples": [ex.input_params for ex in self.examples]
    }
```

---

## Example Generator

The `ExampleGenerator` can auto-generate examples:

```python
from src.tools.examples.example_generator import ExampleGenerator

generator = ExampleGenerator()

# Generate examples for a tool
examples = generator.generate_for_tool(
    tool=my_tool,
    count=3,
    strategies=["minimal", "typical", "full"]
)

for generated in examples:
    print(f"Status: {generated.validation_status}")
    print(f"Params: {generated.example.input_params}")
```

### Generation Strategies

| Strategy | Description |
|----------|-------------|
| `minimal` | Required parameters only |
| `typical` | Required + ~50% optional |
| `full` | All parameters |
| `edge` | Boundary/edge case values |

---

## Example Enhancer

The `ExampleEnhancer` improves existing examples:

```python
from src.tools.examples.example_generator import ExampleEnhancer

enhancer = ExampleEnhancer()
enhanced_examples = enhancer.enhance(my_tool)

# Adds:
# - Missing expected outputs
# - Helpful notes
# - Validation status
```

---

## Example Library

Pre-defined workflow examples:

```python
from src.tools.examples.example_library import (
    get_examples_for_tool,
    get_workflow_examples,
    WORKFLOW_EXAMPLES
)

# Get all examples for a specific tool
examples = get_examples_for_tool("analyze_negative_space")

# Get all workflow examples
workflows = get_workflow_examples()
```

### Available Workflows

```python
WORKFLOW_EXAMPLES = [
    ExampleSet(
        use_case="single_image_analysis",
        description="Analyze a single image for negative space",
        examples=[...]
    ),
    ExampleSet(
        use_case="batch_analysis_with_export",
        description="Analyze multiple images and export a report",
        examples=[...]
    ),
    ExampleSet(
        use_case="ml_enhanced_detection",
        description="Use ML for anomaly detection in images",
        examples=[...]
    )
]
```

---

## Best Practices

### 1. Cover All Parameter Types

```python
# ✅ Good - shows different value types
examples = [
    ToolExample(
        description="String and number params",
        input_params={
            "name": "sample",      # string
            "count": 10,           # integer
            "ratio": 0.5,          # float
            "enabled": True,       # boolean
            "tags": ["a", "b"],    # array
            "config": {"x": 1}     # object
        }
    )
]
```

### 2. Show Enum Values

```python
# ✅ Good - demonstrates valid enum options
examples = [
    ToolExample(
        description="Using 'basic' mode",
        input_params={"mode": "basic"}
    ),
    ToolExample(
        description="Using 'advanced' mode",
        input_params={"mode": "advanced"}
    ),
    ToolExample(
        description="Using 'ml_enhanced' mode",
        input_params={"mode": "ml_enhanced"}
    )
]
```

### 3. Include Expected Output

```python
# ✅ Good - shows what to expect
ToolExample(
    input_params={"image_id": "img_123"},
    expected_output_shape={
        "success": True,
        "ratio": 0.34,
        "regions": [{"id": "...", "area": 0.1}]
    }
)
```

### 4. Add Helpful Notes

```python
# ✅ Good - explains when to use
ToolExample(
    description="ML-enhanced analysis",
    input_params={"mode": "ml_enhanced"},
    notes="Use this mode for best accuracy. Requires GPU. Takes ~2x longer."
)
```

---

## Validation

Examples are validated against schemas:

```python
def _validate_example(
    self,
    tool: BaseTool,
    example: ToolExample
) -> tuple[bool, Optional[str]]:
    """Validate an example against tool schema."""
    schema = tool.input_schema
    params = example.input_params

    # Check required fields
    for field in schema.required:
        if field not in params:
            return False, f"Missing required field: {field}"

    # Check types
    for field, value in params.items():
        if field not in schema.properties:
            continue
        expected_type = schema.properties[field].get("type")
        # Type validation...

    return True, None
```

---

## Testing

```python
import pytest

class TestToolExamples:
    def test_tools_have_examples(self):
        """Test that registered tools have examples."""
        for entry in registry._search_index:
            tool = registry.get_tool(entry["name"])
            if tool:
                assert len(tool.examples) >= 1

    def test_examples_have_required_fields(self):
        """Test that examples have all required fields."""
        for entry in registry._search_index:
            tool = registry.get_tool(entry["name"])
            if tool:
                for example in tool.examples:
                    assert example.description
                    assert example.input_params

    def test_examples_match_schema(self):
        """Test that example params match input schema."""
        for entry in registry._search_index:
            tool = registry.get_tool(entry["name"])
            if tool:
                for example in tool.examples:
                    # Validate required fields present
                    for field in tool.input_schema.required:
                        assert field in example.input_params
```

---

## Related Documentation

- [Tool Architecture](./TOOL_ARCHITECTURE.md)
- [Tool Search Guide](./TOOL_SEARCH_GUIDE.md)
- [PTC Implementation](./PTC_IMPLEMENTATION.md)
