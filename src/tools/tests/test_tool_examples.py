"""
Tests for Tool Use Examples functionality.

Verifies that examples improve parameter accuracy as specified
in the Anthropic Advanced Tool Use documentation.

Reference: Anthropic Advanced Tool Use - Tool Use Examples
"""

import pytest
from typing import Dict, Any, List

from ..definitions.base_tool import (
    BaseTool,
    ToolMetadata,
    ToolCategory,
    LoadingStrategy,
    CallerType,
    InputSchema,
    OutputSchema,
    ToolExample
)
from ..examples.example_generator import (
    ExampleGenerator,
    ExampleEnhancer,
    GeneratedExample,
    generate_all_examples
)
from ..examples.example_library import (
    get_examples_for_tool,
    get_workflow_examples,
    WORKFLOW_EXAMPLES
)
from ..registry.tool_registry import registry


class MockToolWithExamples(BaseTool):
    """Mock tool for testing examples functionality."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="mock_example_tool",
            description="A mock tool for testing examples",
            category=ToolCategory.UTILITY,
            loading_strategy=LoadingStrategy.DEFERRED,
            allowed_callers=[CallerType.BOTH],
            tags=["test", "mock"],
            search_keywords=["test", "example"]
        )

    @property
    def input_schema(self) -> InputSchema:
        return InputSchema(
            properties={
                "required_string": {
                    "type": "string",
                    "description": "A required string parameter"
                },
                "optional_int": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "default": 50,
                    "description": "An optional integer"
                },
                "optional_float": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "An optional float"
                },
                "optional_bool": {
                    "type": "boolean",
                    "default": False,
                    "description": "An optional boolean"
                },
                "optional_enum": {
                    "type": "string",
                    "enum": ["option_a", "option_b", "option_c"],
                    "default": "option_a",
                    "description": "An optional enum"
                }
            },
            required=["required_string"]
        )

    @property
    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            description="Mock output for testing",
            properties={
                "success": {"type": "boolean", "description": "Success status"},
                "value": {"type": "number", "description": "Result value"}
            }
        )

    @property
    def examples(self) -> List[ToolExample]:
        return [
            ToolExample(
                description="Minimal example",
                input_params={"required_string": "test"},
                expected_output_shape={"success": True, "value": 1.0},
                notes="Uses only required parameters"
            ),
            ToolExample(
                description="Full example",
                input_params={
                    "required_string": "test",
                    "optional_int": 75,
                    "optional_float": 0.8,
                    "optional_bool": True,
                    "optional_enum": "option_b"
                },
                expected_output_shape={"success": True, "value": 2.0},
                notes="Uses all parameters"
            )
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        return {"success": True, "value": 1.0}


class TestExampleGenerator:
    """Tests for the ExampleGenerator class."""

    @pytest.fixture
    def generator(self) -> ExampleGenerator:
        return ExampleGenerator()

    @pytest.fixture
    def mock_tool(self) -> MockToolWithExamples:
        return MockToolWithExamples()

    def test_generate_minimal_example(self, generator: ExampleGenerator, mock_tool: MockToolWithExamples):
        """Test generating minimal example with only required params."""
        examples = generator.generate_for_tool(mock_tool, count=1, strategies=["minimal"])

        assert len(examples) == 1
        example = examples[0]
        assert example.validation_status == "valid"
        assert "required_string" in example.example.input_params

    def test_generate_typical_example(self, generator: ExampleGenerator, mock_tool: MockToolWithExamples):
        """Test generating typical example with some optional params."""
        examples = generator.generate_for_tool(mock_tool, count=1, strategies=["typical"])

        assert len(examples) == 1
        example = examples[0]
        assert example.validation_status == "valid"
        # Should have required + some optional
        assert "required_string" in example.example.input_params

    def test_generate_full_example(self, generator: ExampleGenerator, mock_tool: MockToolWithExamples):
        """Test generating full example with all params."""
        examples = generator.generate_for_tool(mock_tool, count=1, strategies=["full"])

        assert len(examples) == 1
        example = examples[0]
        assert example.validation_status == "valid"
        # Should have all parameters
        params = example.example.input_params
        assert "required_string" in params
        assert "optional_int" in params
        assert "optional_float" in params

    def test_generate_multiple_examples(self, generator: ExampleGenerator, mock_tool: MockToolWithExamples):
        """Test generating multiple examples with different strategies."""
        examples = generator.generate_for_tool(
            mock_tool,
            count=3,
            strategies=["minimal", "typical", "full"]
        )

        assert len(examples) == 3
        # Each should have valid status
        assert all(e.validation_status == "valid" for e in examples)

    def test_example_has_expected_output(self, generator: ExampleGenerator, mock_tool: MockToolWithExamples):
        """Test that generated examples include expected output shape."""
        examples = generator.generate_for_tool(mock_tool, count=1)

        assert examples[0].example.expected_output_shape is not None
        output = examples[0].example.expected_output_shape
        assert "success" in output
        assert "value" in output

    def test_enum_value_generation(self, generator: ExampleGenerator, mock_tool: MockToolWithExamples):
        """Test that enum values are properly generated."""
        examples = generator.generate_for_tool(mock_tool, count=1, strategies=["full"])

        params = examples[0].example.input_params
        if "optional_enum" in params:
            assert params["optional_enum"] in ["option_a", "option_b", "option_c"]

    def test_number_range_generation(self, generator: ExampleGenerator, mock_tool: MockToolWithExamples):
        """Test that numbers respect min/max constraints."""
        examples = generator.generate_for_tool(mock_tool, count=1, strategies=["full"])

        params = examples[0].example.input_params
        if "optional_int" in params:
            assert 0 <= params["optional_int"] <= 100
        if "optional_float" in params:
            assert 0.0 <= params["optional_float"] <= 1.0


class TestExampleEnhancer:
    """Tests for the ExampleEnhancer class."""

    @pytest.fixture
    def enhancer(self) -> ExampleEnhancer:
        return ExampleEnhancer()

    @pytest.fixture
    def mock_tool(self) -> MockToolWithExamples:
        return MockToolWithExamples()

    def test_enhance_preserves_existing_examples(self, enhancer: ExampleEnhancer, mock_tool: MockToolWithExamples):
        """Test that enhancement preserves existing example data."""
        enhanced = enhancer.enhance(mock_tool)

        assert len(enhanced) == len(mock_tool.examples)
        for original, enhanced_ex in zip(mock_tool.examples, enhanced):
            assert enhanced_ex.description == original.description
            assert enhanced_ex.input_params == original.input_params

    def test_enhance_adds_expected_output_if_missing(self, enhancer: ExampleEnhancer, mock_tool: MockToolWithExamples):
        """Test that enhancement adds expected output when missing."""
        enhanced = enhancer.enhance(mock_tool)

        for example in enhanced:
            assert example.expected_output_shape is not None

    def test_enhance_adds_notes_if_missing(self, enhancer: ExampleEnhancer, mock_tool: MockToolWithExamples):
        """Test that enhancement adds notes when missing."""
        enhanced = enhancer.enhance(mock_tool)

        for example in enhanced:
            assert example.notes is not None
            assert len(example.notes) > 0


class TestExampleLibrary:
    """Tests for the example library."""

    def test_workflow_examples_not_empty(self):
        """Test that workflow examples are defined."""
        examples = get_workflow_examples()
        assert len(examples) > 0

    def test_workflow_examples_have_steps(self):
        """Test that workflow examples have defined steps."""
        for example_set in WORKFLOW_EXAMPLES:
            assert example_set.use_case
            assert example_set.description
            assert len(example_set.examples) > 0

    def test_get_examples_for_tool(self):
        """Test retrieving examples for a specific tool."""
        examples = get_examples_for_tool("analyze_negative_space")
        # May be empty if no examples reference this tool, but should not error
        assert isinstance(examples, list)

    def test_examples_have_required_fields(self):
        """Test that all examples have required fields."""
        for example_set in WORKFLOW_EXAMPLES:
            for example in example_set.examples:
                assert "tool" in example
                assert "params" in example
                assert "expected" in example


class TestToolExamplesIntegration:
    """Integration tests for tool examples with the registry."""

    def test_registered_tools_have_examples(self):
        """Test that registered tools have defined examples."""
        stats = registry.get_stats()
        if stats.total_tools > 0:
            # Get a sample of tools
            for entry in registry._search_index[:5]:
                tool = registry.get_tool(entry["name"])
                if tool:
                    examples = tool.examples
                    assert isinstance(examples, list)

    def test_examples_serializable(self):
        """Test that examples can be serialized to dict."""
        for entry in registry._search_index[:5]:
            tool = registry.get_tool(entry["name"])
            if tool:
                for example in tool.examples:
                    as_dict = example.to_dict()
                    assert "description" in as_dict
                    assert "input" in as_dict

    def test_api_definition_includes_examples(self):
        """Test that API definitions include examples."""
        for entry in registry._search_index[:3]:
            tool = registry.get_tool(entry["name"])
            if tool:
                api_def = tool.to_api_definition()
                assert "input_examples" in api_def
                assert isinstance(api_def["input_examples"], list)


class TestExampleValidation:
    """Tests for example validation against tool schemas."""

    @pytest.fixture
    def mock_tool(self) -> MockToolWithExamples:
        return MockToolWithExamples()

    def test_examples_match_input_schema(self, mock_tool: MockToolWithExamples):
        """Test that examples match the tool's input schema."""
        schema = mock_tool.input_schema

        for example in mock_tool.examples:
            params = example.input_params

            # Check required fields
            for required_field in schema.required:
                assert required_field in params, f"Missing required: {required_field}"

            # Check types
            for field, value in params.items():
                if field in schema.properties:
                    expected_type = schema.properties[field].get("type")
                    # Basic type checking
                    if expected_type == "string":
                        assert isinstance(value, str)
                    elif expected_type == "integer":
                        assert isinstance(value, int)
                    elif expected_type == "number":
                        assert isinstance(value, (int, float))
                    elif expected_type == "boolean":
                        assert isinstance(value, bool)

    def test_examples_respect_enum_constraints(self, mock_tool: MockToolWithExamples):
        """Test that examples respect enum constraints."""
        schema = mock_tool.input_schema

        for example in mock_tool.examples:
            params = example.input_params

            for field, value in params.items():
                if field in schema.properties:
                    prop = schema.properties[field]
                    if "enum" in prop:
                        assert value in prop["enum"], f"Invalid enum value: {value}"

    def test_examples_respect_range_constraints(self, mock_tool: MockToolWithExamples):
        """Test that examples respect min/max constraints."""
        schema = mock_tool.input_schema

        for example in mock_tool.examples:
            params = example.input_params

            for field, value in params.items():
                if field in schema.properties:
                    prop = schema.properties[field]

                    if "minimum" in prop and isinstance(value, (int, float)):
                        assert value >= prop["minimum"]

                    if "maximum" in prop and isinstance(value, (int, float)):
                        assert value <= prop["maximum"]


class TestGenerateAllExamples:
    """Tests for bulk example generation."""

    def test_generate_for_multiple_tools(self):
        """Test generating examples for multiple tools."""
        tools = []
        for entry in registry._search_index[:3]:
            tool = registry.get_tool(entry["name"])
            if tool:
                tools.append(tool)

        if tools:
            result = generate_all_examples(tools, examples_per_tool=2)
            assert len(result) == len(tools)
            for tool_name, examples in result.items():
                assert len(examples) == 2
