"""
Tool Example Generator.

Automatically generates and validates tool usage examples
to improve parameter accuracy from 72% to 90%.

Reference: Anthropic Advanced Tool Use - Tool Use Examples
"""

from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass
import json
import random
import string
from datetime import datetime

from ..definitions.base_tool import BaseTool, ToolExample, InputSchema


@dataclass
class GeneratedExample:
    """An auto-generated example with metadata."""
    example: ToolExample
    generated_at: datetime
    validation_status: str
    validation_message: Optional[str] = None


class ExampleGenerator:
    """
    Generates tool usage examples automatically.

    Strategies:
    - Minimal: Required parameters only
    - Typical: Common use case with some optional params
    - Full: All parameters specified
    - Edge case: Boundary values and special cases
    """

    def __init__(self) -> None:
        self._type_generators = {
            "string": self._generate_string,
            "integer": self._generate_integer,
            "number": self._generate_number,
            "boolean": self._generate_boolean,
            "array": self._generate_array,
            "object": self._generate_object,
        }

    def generate_for_tool(
        self,
        tool: BaseTool,
        count: int = 3,
        strategies: Optional[List[str]] = None
    ) -> List[GeneratedExample]:
        """
        Generate examples for a tool.

        Args:
            tool: The tool to generate examples for
            count: Number of examples to generate
            strategies: List of strategies to use (minimal, typical, full, edge)

        Returns:
            List of generated examples
        """
        strategies = strategies or ["minimal", "typical", "full"]
        examples = []

        for i, strategy in enumerate(strategies[:count]):
            params = self._generate_params(tool.input_schema, strategy)
            expected = self._estimate_output(tool, params)

            example = ToolExample(
                description=f"{strategy.capitalize()} invocation example",
                input_params=params,
                expected_output_shape=expected,
                notes=f"Auto-generated using {strategy} strategy"
            )

            validation = self._validate_example(tool, example)

            examples.append(GeneratedExample(
                example=example,
                generated_at=datetime.now(),
                validation_status="valid" if validation[0] else "invalid",
                validation_message=validation[1]
            ))

        return examples

    def _generate_params(
        self,
        schema: InputSchema,
        strategy: str
    ) -> Dict[str, Any]:
        """Generate parameters based on strategy."""
        params = {}

        if strategy == "minimal":
            # Only required params
            for field in schema.required:
                if field in schema.properties:
                    params[field] = self._generate_value(
                        schema.properties[field]
                    )

        elif strategy == "typical":
            # Required + some optional
            for field in schema.required:
                if field in schema.properties:
                    params[field] = self._generate_value(
                        schema.properties[field]
                    )
            # Add ~50% of optional params
            optional = [f for f in schema.properties if f not in schema.required]
            for field in optional[:len(optional) // 2]:
                params[field] = self._generate_value(
                    schema.properties[field]
                )

        elif strategy == "full":
            # All params
            for field, prop in schema.properties.items():
                params[field] = self._generate_value(prop)

        elif strategy == "edge":
            # Edge cases - min/max values
            for field in schema.required:
                if field in schema.properties:
                    params[field] = self._generate_edge_value(
                        schema.properties[field]
                    )

        return params

    def _generate_value(self, prop: Dict[str, Any]) -> Any:
        """Generate a value based on property schema."""
        prop_type = prop.get("type", "string")
        generator = self._type_generators.get(prop_type, self._generate_string)
        return generator(prop)

    def _generate_edge_value(self, prop: Dict[str, Any]) -> Any:
        """Generate an edge case value."""
        prop_type = prop.get("type", "string")

        if prop_type == "integer":
            if "minimum" in prop:
                return prop["minimum"]
            if "maximum" in prop:
                return prop["maximum"]
            return 0

        if prop_type == "number":
            if "minimum" in prop:
                return prop["minimum"]
            if "maximum" in prop:
                return prop["maximum"]
            return 0.0

        if prop_type == "string":
            if prop.get("minLength"):
                return "a" * prop["minLength"]
            return ""

        return self._generate_value(prop)

    def _generate_string(self, prop: Dict[str, Any]) -> str:
        """Generate a string value."""
        if "enum" in prop:
            return random.choice(prop["enum"])

        if "pattern" in prop:
            pattern = prop["pattern"]
            if pattern.startswith("^img_"):
                return f"img_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}"

        default = prop.get("default")
        if default:
            return default

        return f"sample_{''.join(random.choices(string.ascii_lowercase, k=6))}"

    def _generate_integer(self, prop: Dict[str, Any]) -> int:
        """Generate an integer value."""
        minimum = prop.get("minimum", 0)
        maximum = prop.get("maximum", 100)
        default = prop.get("default")

        if default is not None:
            return default

        return random.randint(minimum, maximum)

    def _generate_number(self, prop: Dict[str, Any]) -> float:
        """Generate a float value."""
        minimum = prop.get("minimum", 0.0)
        maximum = prop.get("maximum", 1.0)
        default = prop.get("default")

        if default is not None:
            return default

        return round(random.uniform(minimum, maximum), 2)

    def _generate_boolean(self, prop: Dict[str, Any]) -> bool:
        """Generate a boolean value."""
        default = prop.get("default")
        if default is not None:
            return default
        return random.choice([True, False])

    def _generate_array(self, prop: Dict[str, Any]) -> List[Any]:
        """Generate an array value."""
        items_schema = prop.get("items", {"type": "string"})
        count = random.randint(1, 3)
        return [self._generate_value(items_schema) for _ in range(count)]

    def _generate_object(self, prop: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an object value."""
        properties = prop.get("properties", {})
        return {
            key: self._generate_value(val_schema)
            for key, val_schema in properties.items()
        }

    def _estimate_output(
        self,
        tool: BaseTool,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate expected output based on tool schema."""
        output_schema = tool.output_schema
        estimated = {}

        for name, prop in output_schema.properties.items():
            prop_type = prop.get("type", "any")

            if prop_type == "boolean":
                estimated[name] = True
            elif prop_type == "number":
                estimated[name] = 0.5
            elif prop_type == "integer":
                estimated[name] = 100
            elif prop_type == "string":
                estimated[name] = f"<{name}>"
            elif prop_type == "array":
                estimated[name] = []
            elif prop_type == "object":
                estimated[name] = {}

        return estimated

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

            prop = schema.properties[field]
            expected_type = prop.get("type")

            if expected_type == "string" and not isinstance(value, str):
                return False, f"Field {field} should be string"
            if expected_type == "integer" and not isinstance(value, int):
                return False, f"Field {field} should be integer"
            if expected_type == "number" and not isinstance(value, (int, float)):
                return False, f"Field {field} should be number"
            if expected_type == "boolean" and not isinstance(value, bool):
                return False, f"Field {field} should be boolean"

        return True, None


class ExampleEnhancer:
    """
    Enhances existing examples for better parameter accuracy.

    Improvements:
    - Adds missing expected outputs
    - Adds explanatory notes
    - Validates against schema
    """

    def __init__(self, generator: Optional[ExampleGenerator] = None) -> None:
        self.generator = generator or ExampleGenerator()

    def enhance(self, tool: BaseTool) -> List[ToolExample]:
        """
        Enhance a tool's examples.

        Returns improved examples with complete information.
        """
        existing = tool.examples
        enhanced = []

        for example in existing:
            new_example = ToolExample(
                description=example.description,
                input_params=example.input_params,
                expected_output_shape=(
                    example.expected_output_shape or
                    self.generator._estimate_output(tool, example.input_params)
                ),
                notes=example.notes or self._generate_notes(tool, example)
            )
            enhanced.append(new_example)

        return enhanced

    def _generate_notes(self, tool: BaseTool, example: ToolExample) -> str:
        """Generate helpful notes for an example."""
        params = example.input_params
        notes = []

        # Check if using defaults
        schema = tool.input_schema
        optional_used = [
            f for f in params
            if f not in schema.required
        ]

        if not optional_used:
            notes.append("Uses only required parameters.")
        else:
            notes.append(f"Optional parameters: {', '.join(optional_used)}.")

        # Check for special values
        for field, value in params.items():
            if field in schema.properties:
                prop = schema.properties[field]
                if "enum" in prop:
                    notes.append(f"{field} can be one of: {', '.join(prop['enum'])}")

        return " ".join(notes) if notes else "Standard invocation."


def generate_all_examples(
    tools: List[BaseTool],
    examples_per_tool: int = 3
) -> Dict[str, List[ToolExample]]:
    """
    Generate examples for multiple tools.

    Returns dict of tool_name -> list of examples.
    """
    generator = ExampleGenerator()
    result = {}

    for tool in tools:
        generated = generator.generate_for_tool(tool, count=examples_per_tool)
        result[tool.metadata.name] = [g.example for g in generated]

    return result
