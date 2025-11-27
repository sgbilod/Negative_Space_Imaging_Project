"""
Integration Tests for Advanced Tool Use Infrastructure.

Tests the complete system including all three features:
- Tool Search Tool
- Programmatic Tool Calling
- Tool Use Examples
"""

import pytest
import asyncio

from .. import (
    get_api_configuration,
    get_tool,
    search_tools,
    registry,
    config
)
from ..definitions.base_tool import ToolCategory, CallerType


class TestAPIConfiguration:
    """Tests for API configuration generation."""

    def test_get_api_configuration_returns_required_fields(self):
        """Test that API configuration includes all required fields."""
        api_config = get_api_configuration()

        assert "headers" in api_config
        assert "tools" in api_config
        assert "system_prompt_addition" in api_config
        assert "config" in api_config

    def test_headers_include_beta_flag(self):
        """Test that headers include the beta flag."""
        api_config = get_api_configuration()

        headers = api_config["headers"]
        assert "anthropic-beta" in headers
        assert "advanced-tool-use" in headers["anthropic-beta"]

    def test_tools_list_not_empty(self):
        """Test that tools list is populated."""
        api_config = get_api_configuration()

        # Should at least have tool_search
        assert len(api_config["tools"]) >= 1

    def test_config_flags_present(self):
        """Test that feature flags are present."""
        api_config = get_api_configuration()

        config_section = api_config["config"]
        assert "tool_search_enabled" in config_section
        assert "ptc_enabled" in config_section
        assert "examples_enabled" in config_section


class TestToolRegistry:
    """Tests for tool registry functionality."""

    def test_registry_singleton(self):
        """Test that registry is a singleton."""
        from ..registry.tool_registry import registry as reg1
        from ..registry.tool_registry import registry as reg2

        assert reg1 is reg2

    def test_registered_tools_have_metadata(self):
        """Test all registered tools have proper metadata."""
        for name, tool in registry._tools.items():
            assert tool.metadata is not None
            assert tool.metadata.name == name
            assert tool.metadata.category is not None
            assert tool.input_schema is not None

    def test_get_tool_returns_correct_tool(self):
        """Test getting a tool by name."""
        tool = get_tool("analyze_negative_space")

        if tool:
            assert tool.metadata.name == "analyze_negative_space"

    def test_search_tools_returns_results(self):
        """Test searching for tools."""
        results = search_tools("analyze image", limit=5)

        assert isinstance(results, list)
        # Results should be limited
        assert len(results) <= 5


class TestToolExamples:
    """Tests for tool use examples."""

    def test_tools_have_examples(self):
        """Test that key tools have examples."""
        key_tools = ["analyze_negative_space", "batch_analyze"]

        for tool_name in key_tools:
            tool = registry.get_tool(tool_name)
            if tool:
                examples = tool.examples
                assert len(examples) >= 1, f"{tool_name} should have examples"

    def test_examples_have_required_fields(self):
        """Test that examples have required fields."""
        for name, tool in registry._tools.items():
            for example in tool.examples:
                assert example.description, f"{name} example missing description"
                assert example.input_params is not None, f"{name} example missing input_params"


class TestToolCategories:
    """Tests for tool categorization."""

    def test_all_categories_represented(self):
        """Test that multiple categories are represented."""
        categories = registry.get_categories()

        assert len(categories) >= 3  # Should have at least a few categories

    def test_category_filtering_works(self):
        """Test filtering tools by category."""
        for category in [ToolCategory.IMAGING_CORE, ToolCategory.DATABASE]:
            tools = registry.get_by_category(category)
            for tool in tools:
                assert tool.metadata.category == category


class TestCallerTypeRestrictions:
    """Tests for caller type restrictions."""

    def test_ptc_enabled_tools(self):
        """Test getting PTC-enabled tools."""
        ptc_tools = registry.get_ptc_enabled_tools()

        for tool in ptc_tools:
            assert CallerType.CODE_EXECUTION in tool.metadata.allowed_callers or \
                   CallerType.BOTH in tool.metadata.allowed_callers

    def test_direct_only_tools_excluded_from_ptc(self):
        """Test that direct-only tools are excluded from PTC."""
        ptc_tool_names = [t.metadata.name for t in registry.get_ptc_enabled_tools()]

        for name, tool in registry._tools.items():
            if CallerType.DIRECT in tool.metadata.allowed_callers and \
               CallerType.BOTH not in tool.metadata.allowed_callers and \
               CallerType.CODE_EXECUTION not in tool.metadata.allowed_callers:
                assert name not in ptc_tool_names


class TestToolExecution:
    """Tests for tool execution."""

    @pytest.mark.asyncio
    async def test_execute_imaging_tool(self):
        """Test executing an imaging tool."""
        tool = registry.get_tool("analyze_negative_space")

        if tool:
            result = await tool.execute(image_id="img_test123")

            assert "success" in result
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_batch_tool_execution(self):
        """Test batch tool execution."""
        tool = registry.get_tool("batch_analyze")

        if tool:
            result = await tool.execute(
                image_ids=["img_001", "img_002", "img_003"],
                parallel=True
            )

            assert result["success"] is True
            assert result["total"] == 3

    @pytest.mark.asyncio
    async def test_invalid_input_rejected(self):
        """Test that invalid input is rejected."""
        tool = registry.get_tool("analyze_negative_space")

        if tool:
            with pytest.raises(ValueError):
                await tool.execute()  # Missing required image_id


class TestSystemIntegration:
    """Full system integration tests."""

    @pytest.mark.asyncio
    async def test_search_and_execute_workflow(self):
        """Test searching for a tool and executing it."""
        # Search for analysis tools
        results = search_tools("analyze negative space")

        if results:
            tool_name = results[0].get("name") if isinstance(results[0], dict) else results[0]
            tool = get_tool(tool_name)

            if tool:
                result = await tool.execute(image_id="img_workflow_test")
                assert "success" in result

    def test_api_configuration_complete(self):
        """Test that API configuration is complete and usable."""
        api_config = get_api_configuration()

        # Verify structure matches Anthropic API expectations
        assert "headers" in api_config
        assert "tools" in api_config

        # Each tool should have required fields
        for tool_def in api_config["tools"]:
            assert "name" in tool_def or "type" in tool_def
            assert "description" in tool_def


class TestConfiguration:
    """Tests for configuration management."""

    def test_config_from_environment(self):
        """Test configuration from environment variables."""
        from ..config.tool_config import AdvancedToolUseConfig

        test_config = AdvancedToolUseConfig.from_environment()

        assert test_config is not None
        assert test_config.tool_search is not None
        assert test_config.ptc is not None

    def test_default_config_values(self):
        """Test default configuration values."""
        assert config.tool_search.enabled is True
        assert config.ptc.enabled is True
        assert config.ptc.max_parallel_calls > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
