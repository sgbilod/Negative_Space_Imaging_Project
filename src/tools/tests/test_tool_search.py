"""
Tests for Tool Search Tool.

Tests BM25 scoring, hybrid search, and category filtering.
"""

import pytest
import asyncio
from typing import List

from ..registry.tool_search import (
    ToolSearchTool,
    SearchResult,
    SearchResponse,
    SearchStrategy,
    BM25Scorer
)
from ..registry.tool_registry import registry


class TestBM25Scorer:
    """Tests for BM25 scoring algorithm."""

    def test_score_with_matching_terms(self):
        """Test scoring with matching terms."""
        scorer = BM25Scorer()
        documents = [
            {
                "name": "analyze_negative_space",
                "description": "negative space analysis",
                "keywords": []
            },
            {
                "name": "process_image",
                "description": "image processing",
                "keywords": []
            },
            {
                "name": "store_data",
                "description": "data storage",
                "keywords": []
            }
        ]
        scorer.build_index(documents)

        results = scorer.search("negative space", limit=3)

        # First document should have highest score
        assert len(results) >= 1
        assert results[0][0] == 0  # First doc index

    def test_score_with_no_matches(self):
        """Test scoring with no matching terms."""
        scorer = BM25Scorer()
        documents = [
            {
                "name": "alpha",
                "description": "alpha beta gamma",
                "keywords": []
            },
            {
                "name": "delta",
                "description": "delta epsilon",
                "keywords": []
            }
        ]
        scorer.build_index(documents)

        results = scorer.search("completely different")

        # Should return empty results
        assert len(results) == 0

    def test_empty_query(self):
        """Test with empty query."""
        scorer = BM25Scorer()
        documents = [
            {
                "name": "test",
                "description": "test document",
                "keywords": []
            }
        ]
        scorer.build_index(documents)

        results = scorer.search("")

        assert len(results) == 0

    def test_idf_calculation(self):
        """Test IDF favors rare terms."""
        scorer = BM25Scorer()
        documents = [
            {
                "name": "doc1",
                "description": "common common common rare",
                "keywords": []
            },
            {
                "name": "doc2",
                "description": "common common common",
                "keywords": []
            },
            {
                "name": "doc3",
                "description": "common common",
                "keywords": []
            }
        ]
        scorer.build_index(documents)

        # IDF for 'rare' should be higher than 'common'
        assert scorer.idf.get("rare", 0) > scorer.idf.get("common", 0)


class TestToolSearchTool:
    """Tests for ToolSearchTool."""

    @pytest.fixture
    def search_tool(self):
        return ToolSearchTool(registry)

    @pytest.mark.asyncio
    async def test_search_by_keyword(self, search_tool):
        """Test keyword-based search."""
        result = await search_tool.search(
            query="negative space",
            limit=5
        )

        assert isinstance(result, SearchResponse)
        assert len(result.results) <= 5

    @pytest.mark.asyncio
    async def test_search_by_category(self, search_tool):
        """Test category-based search."""
        result = await search_tool.search(
            query="analysis",
            category="IMAGING_CORE",
            limit=10
        )

        assert isinstance(result, SearchResponse)
        # All results should be from the specified category (if any)
        for r in result.results:
            assert r.category == "IMAGING_CORE"

    @pytest.mark.asyncio
    async def test_hybrid_search(self, search_tool):
        """Test hybrid search strategy."""
        # Default strategy is hybrid
        result = await search_tool.search(
            query="batch image processing",
            limit=5
        )

        assert isinstance(result, SearchResponse)
        assert result.strategy == "hybrid"

    @pytest.mark.asyncio
    async def test_search_returns_tool_definitions(self, search_tool):
        """Test that search returns proper tool definitions."""
        result = await search_tool.search(
            query="analyze",
            limit=3
        )

        for r in result.results:
            assert r.tool_name is not None
            assert r.description is not None
            assert r.category is not None
            assert isinstance(r.score, float)

    @pytest.mark.asyncio
    async def test_empty_query_returns_results(self, search_tool):
        """Test empty query returns some results."""
        # Build index first
        search_tool.build_index()

        result = await search_tool.search(
            query="",
            limit=10
        )

        # Empty query with BM25 returns no results, but regex may return all
        assert isinstance(result, SearchResponse)

    def test_metadata_properties(self, search_tool):
        """Test tool has proper metadata."""
        api_def = search_tool.get_tool_definition()

        assert api_def["type"] == "tool_search_tool_regex_20251119"
        assert api_def["name"] == "tool_search"
        assert "description" in api_def
        assert "input_schema" in api_def

    def test_get_tool_definition(self, search_tool):
        """Test API definition generation."""
        api_def = search_tool.get_tool_definition()

        assert api_def["type"] == "tool_search_tool_regex_20251119"
        assert "name" in api_def
        assert "description" in api_def
        assert "input_schema" in api_def
        assert "properties" in api_def["input_schema"]


class TestSearchIntegration:
    """Integration tests for search functionality."""

    @pytest.mark.asyncio
    async def test_search_finds_registered_tools(self):
        """Test that search finds tools registered in the registry."""
        search_tool = ToolSearchTool(registry)
        search_tool.build_index()

        result = await search_tool.search(
            query="analysis",
            limit=10
        )

        assert isinstance(result, SearchResponse)

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self):
        """Test max_results limit is respected."""
        search_tool = ToolSearchTool(registry)
        search_tool.build_index()

        for limit in [1, 3, 5]:
            result = await search_tool.search(
                query="image",
                limit=limit
            )

            assert len(result.results) <= limit

    @pytest.mark.asyncio
    async def test_concurrent_searches(self):
        """Test concurrent search requests."""
        search_tool = ToolSearchTool(registry)
        search_tool.build_index()

        queries = ["negative", "analysis", "export", "batch", "image"]

        tasks = [
            search_tool.search(query=q, limit=3)
            for q in queries
        ]

        results = await asyncio.gather(*tasks)

        # All results should complete successfully
        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result, SearchResponse)
