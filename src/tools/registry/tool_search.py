"""
Tool Search Tool Implementation.

Enables dynamic tool discovery without loading all definitions upfront.
Reduces context usage from ~55K tokens to ~3K tokens.

Reference: Anthropic Advanced Tool Use - Tool Search Tool
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re
import math
from collections import Counter

from .tool_registry import registry, ToolRegistry
from ..definitions.base_tool import ToolCategory


class SearchStrategy(Enum):
    """Search algorithm selection."""
    REGEX = "regex"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """Individual search result."""
    tool_name: str
    description: str
    category: str
    score: float
    matched_terms: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.tool_name,
            "description": self.description,
            "category": self.category,
            "relevance_score": round(self.score, 3),
            "matched_terms": self.matched_terms
        }


@dataclass
class SearchResponse:
    """Complete search response."""
    query: str
    strategy: str
    total_results: int
    results: List[SearchResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "total_results": self.total_results,
            "tools": [r.to_dict() for r in self.results]
        }


class BM25Scorer:
    """
    BM25 ranking algorithm for tool search.

    Better for natural language queries than simple regex.
    Uses Okapi BM25 with configurable k1 and b parameters.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: List[Dict] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freqs: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self._built = False

    def build_index(self, documents: List[Dict]) -> None:
        """Build BM25 index from tool documents."""
        self.documents = documents
        self.doc_lengths = []
        self.doc_freqs = Counter()

        for doc in documents:
            text = self._get_searchable_text(doc)
            tokens = self._tokenize(text)
            self.doc_lengths.append(len(tokens))

            for token in set(tokens):
                self.doc_freqs[token] += 1

        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        n_docs = len(documents)
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

        self._built = True

    def search(self, query: str, limit: int = 10) -> List[tuple]:
        """Search using BM25 scoring."""
        if not self._built:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc in enumerate(self.documents):
            text = self._get_searchable_text(doc)
            doc_tokens = self._tokenize(text)
            doc_length = self.doc_lengths[idx]

            score = 0.0
            matched_terms = []

            for term in query_tokens:
                if term not in self.idf:
                    continue

                tf = doc_tokens.count(term)
                if tf == 0:
                    continue

                matched_terms.append(term)
                idf = self.idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / self.avg_doc_length
                )
                score += idf * (numerator / denominator)

            score *= doc.get("boost", 1.0)

            if score > 0:
                scores.append((idx, score, matched_terms))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    def _get_searchable_text(self, doc: Dict) -> str:
        """Extract searchable text from document."""
        parts = [
            doc.get("name", ""),
            doc.get("description", ""),
            " ".join(doc.get("keywords", [])),
            " ".join(doc.get("tags", []))
        ]
        return " ".join(parts)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation tokenization."""
        return re.findall(r'\b\w+\b', text.lower())


class ToolSearchTool:
    """
    Tool Search Tool - enables dynamic tool discovery.

    API Definition:
    {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search"
    }
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        strategy: SearchStrategy = SearchStrategy.HYBRID
    ) -> None:
        self.registry = tool_registry
        self.strategy = strategy
        self.bm25 = BM25Scorer()
        self._index_built = False

    def build_index(self) -> None:
        """Build search index from registry."""
        documents = list(self.registry._search_index)
        self.bm25.build_index(documents)
        self._index_built = True

    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> SearchResponse:
        """
        Search for tools matching the query.

        Args:
            query: Natural language search query
            category: Optional category filter
            limit: Maximum results to return

        Returns:
            SearchResponse with matching tools
        """
        if not self._index_built:
            self.build_index()

        if self.strategy == SearchStrategy.REGEX:
            results = self._regex_search(query, category, limit)
        elif self.strategy == SearchStrategy.BM25:
            results = self._bm25_search(query, category, limit)
        else:
            regex_results = self._regex_search(query, category, limit * 2)
            bm25_results = self._bm25_search(query, category, limit * 2)

            seen: set = set()
            combined = []
            for r in regex_results + bm25_results:
                if r.tool_name not in seen:
                    seen.add(r.tool_name)
                    combined.append(r)

            combined.sort(key=lambda x: x.score, reverse=True)
            results = combined[:limit]

        return SearchResponse(
            query=query,
            strategy=self.strategy.value,
            total_results=len(results),
            results=results
        )

    def _regex_search(
        self,
        query: str,
        category: Optional[str],
        limit: int
    ) -> List[SearchResult]:
        """Simple regex-based search."""
        cat_enum = ToolCategory[category] if category else None
        raw_results = self.registry.search(query, category=cat_enum, limit=limit)

        return [
            SearchResult(
                tool_name=r["name"],
                description=r["description"][:200],
                category=r["category"],
                score=r["score"],
                matched_terms=[]
            )
            for r in raw_results
        ]

    def _bm25_search(
        self,
        query: str,
        category: Optional[str],
        limit: int
    ) -> List[SearchResult]:
        """BM25-based search."""
        raw_results = self.bm25.search(query, limit=limit * 2)
        results = []

        for idx, score, matched in raw_results:
            doc = self.registry._search_index[idx]
            if category and doc["category"] != category:
                continue

            results.append(SearchResult(
                tool_name=doc["name"],
                description=doc["description"][:200],
                category=doc["category"],
                score=score,
                matched_terms=matched
            ))

        return results[:limit]

    def get_tool_definition(self) -> Dict[str, Any]:
        """Get the API definition for Tool Search Tool itself."""
        return {
            "type": "tool_search_tool_regex_20251119",
            "name": "tool_search",
            "description": """Search for available tools by capability or name.

Use this to discover tools for:
- Image analysis and negative space detection
- Database operations (CRUD, queries)
- Security (authentication, encryption, audit)
- Export (reports, format conversion)
- ML inference (model predictions)
- Specialized formats (DICOM, FITS)
- Administrative functions

Returns matching tools with descriptions and categories.
After finding relevant tools, they will be loaded into context.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "category": {
                        "type": "string",
                        "enum": [c.name for c in ToolCategory],
                        "description": "Optional category filter"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum number of results"
                    }
                },
                "required": ["query"]
            }
        }
