# Tool Search Tool Guide

## Negative Space Imaging Project - Dynamic Tool Discovery

**Version:** 1.0.0
**Reference:** [Anthropic Engineering - Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)

---

## Overview

The Tool Search Tool enables Claude to dynamically discover relevant tools without loading all 50+ tool definitions upfront. This reduces initial context from ~55K tokens to ~3K tokens (an 85% reduction).

### How It Works

```
┌────────────────────────────────────────────────────────────────┐
│                     Tool Search Flow                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Initial Context (3K tokens)                                │
│     ┌─────────────────────────────────────────┐               │
│     │  • Tool Search Tool definition          │               │
│     │  • Code Execution Tool definition       │               │
│     │  • 5-10 core tool definitions           │               │
│     └─────────────────────────────────────────┘               │
│                          │                                     │
│                          ▼                                     │
│  2. User asks: "I need to analyze DICOM files"                │
│                          │                                     │
│                          ▼                                     │
│  3. Claude calls: tool_search(query="DICOM analysis")         │
│                          │                                     │
│                          ▼                                     │
│  4. Search returns matching tools                              │
│     ┌─────────────────────────────────────────┐               │
│     │  • parse_dicom (score: 9.5)             │               │
│     │  • extract_dicom_metadata (score: 8.2)  │               │
│     │  • dicom_to_png (score: 7.8)            │               │
│     └─────────────────────────────────────────┘               │
│                          │                                     │
│                          ▼                                     │
│  5. Tool definitions loaded into context                       │
│                          │                                     │
│                          ▼                                     │
│  6. Claude uses the discovered tools                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## API Definition

The Tool Search Tool is automatically included in the initial context:

```python
{
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

Returns matching tools with their descriptions and categories.
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
                "enum": ["IMAGING_CORE", "DATABASE", "SECURITY", ...],
                "description": "Optional category filter"
            },
            "limit": {
                "type": "integer",
                "default": 5,
                "description": "Maximum results"
            }
        },
        "required": ["query"]
    }
}
```

---

## Search Strategies

### 1. Regex Search (Default)

Simple pattern matching on tool names, descriptions, and keywords.

```python
from src.tools.registry.tool_search import ToolSearchTool, SearchStrategy

search = ToolSearchTool(registry, strategy=SearchStrategy.REGEX)
results = await search.search("negative space")
```

### 2. BM25 Search

Okapi BM25 algorithm for better natural language queries.

```python
search = ToolSearchTool(registry, strategy=SearchStrategy.BM25)
results = await search.search("how do I analyze medical images?")
```

### 3. Hybrid Search (Recommended)

Combines both strategies for best results.

```python
search = ToolSearchTool(registry, strategy=SearchStrategy.HYBRID)
results = await search.search("export analysis as PDF report")
```

---

## Search Response Format

```python
@dataclass
class SearchResponse:
    query: str
    strategy: str
    total_results: int
    results: List[SearchResult]

@dataclass
class SearchResult:
    tool_name: str
    description: str
    category: str
    score: float
    matched_terms: List[str]
```

Example response:

```json
{
    "query": "negative space analysis",
    "strategy": "hybrid",
    "total_results": 3,
    "tools": [
        {
            "name": "analyze_negative_space",
            "description": "Analyze an image to detect and quantify negative space...",
            "category": "IMAGING_CORE",
            "relevance_score": 15.234,
            "matched_terms": ["negative", "space", "analysis"]
        },
        {
            "name": "batch_analyze",
            "description": "Process multiple images...",
            "category": "IMAGING_CORE",
            "relevance_score": 8.123,
            "matched_terms": ["analyze"]
        }
    ]
}
```

---

## Tool Registration for Search

### Adding Search Keywords

```python
@register_tool
class MyTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="my_tool",
            description="Tool description",
            category=ToolCategory.IMAGING_CORE,
            # Search optimization
            search_keywords=[
                "keyword1", "keyword2", "synonym"
            ],
            search_boost=1.5,  # Higher = more likely to appear
            tags=["category1", "category2"]
        )
```

### Loading Strategies

```python
class LoadingStrategy(Enum):
    ALWAYS_LOADED = "always"     # In initial context (~5-10 tools)
    DEFERRED = "deferred"        # Discovered via search
    LAZY = "lazy"                # Loaded on first call
```

**Guidelines:**
- `ALWAYS_LOADED`: Only the most common tools (analyze_negative_space, batch_analyze)
- `DEFERRED`: Most tools - loaded via Tool Search Tool
- `LAZY`: Rarely used admin tools

---

## Category Filtering

Filter search by category for more precise results:

```python
# Only search imaging tools
results = await search.search(
    query="detect patterns",
    category="IMAGING_ADVANCED"
)

# Only search security tools
results = await search.search(
    query="user authentication",
    category="SECURITY"
)
```

Available categories:
- `IMAGING_CORE` - Core negative space analysis
- `IMAGING_ADVANCED` - Advanced imaging features
- `DATABASE` - CRUD operations
- `SECURITY` - Auth, encryption, audit
- `EXPORT` - Format conversion, reports
- `ML_INFERENCE` - Model inference
- `SPECIALIZED_MEDICAL` - DICOM, medical imaging
- `SPECIALIZED_ASTRO` - FITS, astronomical
- `HPC` - High-performance computing
- `ADMIN` - Administrative functions
- `UTILITY` - Helper utilities

---

## Integration Example

```python
from src.tools import registry, get_api_configuration
from src.tools.registry.tool_search import ToolSearchTool

# Create search tool
search_tool = ToolSearchTool(registry)

# Build API configuration with search tool
config = get_api_configuration()

# Claude receives tools including:
# - tool_search (always loaded)
# - code_execution (always loaded)
# - Core tools (always loaded)

# When Claude needs a specific capability, it calls:
# tool_search(query="export to PDF")

# This returns matching tools, which are then loaded
```

---

## Best Practices

### 1. Craft Good Search Keywords

```python
# ❌ Poor keywords
search_keywords=["tool", "function"]

# ✅ Good keywords
search_keywords=[
    "dicom", "medical imaging", "radiology",
    "pixel data", "patient metadata"
]
```

### 2. Use Appropriate Boost Values

```python
# Core functionality - higher boost
search_boost=2.0

# Standard tools - default boost
search_boost=1.0

# Niche tools - lower boost
search_boost=0.5
```

### 3. Categorize Correctly

Place tools in the most specific category:
- Use `SPECIALIZED_MEDICAL` for DICOM, not `IMAGING_CORE`
- Use `SECURITY` for auth tools, not `UTILITY`

### 4. Write Searchable Descriptions

```python
# ❌ Poor description
description="Parses files"

# ✅ Good description
description="""Parse DICOM medical imaging files.
Extracts patient metadata, pixel data, and imaging parameters.
Supports DICOM 3.0 format from CT, MRI, and X-ray modalities."""
```

---

## Token Savings Analysis

| Scenario | Without Search | With Search | Savings |
|----------|----------------|-------------|---------|
| Initial load | 55K tokens | 3.8K tokens | 93% |
| After 1 search | 55K tokens | 6K tokens | 89% |
| After 5 searches | 55K tokens | 15K tokens | 73% |
| Typical session | 55K tokens | 10K tokens | 82% |

The Tool Search Tool enables conversations with more history while using fewer tokens.

---

## Related Documentation

- [Tool Architecture](./TOOL_ARCHITECTURE.md)
- [PTC Implementation](./PTC_IMPLEMENTATION.md)
- [Tool Examples Reference](./TOOL_EXAMPLES_REFERENCE.md)
