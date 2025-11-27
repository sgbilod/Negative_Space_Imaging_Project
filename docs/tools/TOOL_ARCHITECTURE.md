# Tool Architecture Guide

## Negative Space Imaging Project - Advanced Tool Use Implementation

**Version:** 1.0.0
**Reference:** [Anthropic Engineering - Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)

---

## Overview

The NSIP tool infrastructure implements Anthropic's three Advanced Tool Use features:

1. **Tool Search Tool** - Dynamic tool discovery without loading all definitions upfront
2. **Programmatic Tool Calling (PTC)** - Code-based tool orchestration
3. **Tool Use Examples** - Improved parameter accuracy through concrete examples

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NSIP Advanced Tool Use Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   Tool Search   │     │  PTC Executor   │     │  Tool Examples  │       │
│  │      Tool       │     │                 │     │                 │       │
│  │                 │     │  ┌───────────┐  │     │  ┌───────────┐  │       │
│  │  ┌───────────┐  │     │  │  Sandbox  │  │     │  │ Generator │  │       │
│  │  │  BM25     │  │     │  │ Validator │  │     │  │ Enhancer  │  │       │
│  │  │  Scorer   │  │     │  └───────────┘  │     │  └───────────┘  │       │
│  │  └───────────┘  │     │                 │     │                 │       │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘       │
│           │                       │                       │                 │
│           └───────────────────────┼───────────────────────┘                 │
│                                   │                                         │
│                         ┌─────────▼─────────┐                               │
│                         │   Tool Registry   │                               │
│                         │                   │                               │
│                         │  ┌─────────────┐  │                               │
│                         │  │  Deferred   │  │                               │
│                         │  │   Loader    │  │                               │
│                         │  └─────────────┘  │                               │
│                         └─────────┬─────────┘                               │
│                                   │                                         │
│     ┌─────────────────────────────┼─────────────────────────────────┐       │
│     │                             │                                 │       │
│     ▼                             ▼                                 ▼       │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐      │
│  │ Imaging      │          │ Database     │          │ Security     │      │
│  │ Tools        │          │ Tools        │          │ Tools        │      │
│  └──────────────┘          └──────────────┘          └──────────────┘      │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐      │
│  │ Export       │          │ ML           │          │ Specialized  │      │
│  │ Tools        │          │ Tools        │          │ Tools        │      │
│  └──────────────┘          └──────────────┘          └──────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
/src/tools/
├── __init__.py                      # Package exports and get_api_configuration()
├── registry/
│   ├── __init__.py
│   ├── tool_registry.py             # Central registration system
│   ├── tool_search.py               # Tool Search Tool with BM25
│   ├── tool_categories.py           # Category definitions
│   └── deferred_loader.py           # Lazy loading mechanism
├── definitions/
│   ├── __init__.py
│   ├── base_tool.py                 # BaseTool with all features
│   ├── imaging_tools.py             # Core imaging analysis
│   ├── database_tools.py            # CRUD operations
│   ├── security_tools.py            # Auth/encryption
│   ├── export_tools.py              # Format conversion
│   ├── ml_tools.py                  # ML inference
│   ├── specialized_tools.py         # DICOM, FITS, HPC
│   └── admin_tools.py               # Administrative
├── execution/
│   ├── __init__.py
│   ├── ptc_executor.py              # PTC sandbox executor
│   ├── code_sandbox.py              # Security isolation
│   ├── orchestration_templates.py   # Common patterns
│   └── result_processor.py          # Output filtering
├── examples/
│   ├── __init__.py
│   ├── example_generator.py         # Auto-generation
│   └── example_library.py           # Pre-defined examples
├── config/
│   ├── __init__.py
│   ├── tool_config.py               # Configuration
│   └── feature_flags.py             # Runtime toggles
└── tests/
    ├── test_tool_search.py
    ├── test_ptc_executor.py
    ├── test_tool_examples.py
    └── test_integration.py
```

---

## Core Components

### 1. BaseTool Class

All tools inherit from `BaseTool`, which provides:

```python
class BaseTool(ABC):
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Tool metadata for registration and search"""

    @property
    @abstractmethod
    def input_schema(self) -> InputSchema:
        """JSON Schema for inputs"""

    @property
    @abstractmethod
    def output_schema(self) -> OutputSchema:
        """Output schema for PTC code generation"""

    @property
    @abstractmethod
    def examples(self) -> List[ToolExample]:
        """Concrete usage examples"""

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool"""
```

### 2. Tool Categories

```python
class ToolCategory(Enum):
    IMAGING_CORE = auto()        # Core negative space analysis
    IMAGING_ADVANCED = auto()    # Advanced imaging features
    DATABASE = auto()            # CRUD operations
    SECURITY = auto()            # Auth, encryption, audit
    EXPORT = auto()              # Format conversion, reports
    ML_INFERENCE = auto()        # Model inference
    SPECIALIZED_MEDICAL = auto() # DICOM, medical imaging
    SPECIALIZED_ASTRO = auto()   # FITS, astronomical
    HPC = auto()                 # High-performance computing
    ADMIN = auto()               # Administrative functions
    UTILITY = auto()             # Helper utilities
```

### 3. Loading Strategies

```python
class LoadingStrategy(Enum):
    ALWAYS_LOADED = "always"     # Core tools, always in context
    DEFERRED = "deferred"        # Loaded on-demand via search
    LAZY = "lazy"                # Loaded only when explicitly called
```

---

## Token Optimization

### Before Advanced Tool Use

- All 50+ tools loaded in context: ~55K tokens
- Limited room for conversation history
- Slower response times

### After Advanced Tool Use

| Component | Tokens | Notes |
|-----------|--------|-------|
| Tool Search Tool | ~500 | Always loaded |
| Code Execution Tool | ~800 | Always loaded |
| Core Tools (5-10) | ~2,500 | Always loaded |
| **Total Initial Context** | **~3,800** | 93% reduction |
| Deferred Tools | On-demand | Loaded when searched |

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Tools Registered | 50+ | ✅ Check `registry.get_stats().total_tools` |
| Always-Loaded | ≤10 | ✅ Check `registry.get_stats().always_loaded` |
| PTC-Enabled | 30+ | ✅ Check `len(registry.get_ptc_enabled_tools())` |
| Test Coverage | 90%+ | ✅ Run `pytest --cov` |
| Token Savings | 85%+ | ✅ Compare initial vs. full load |

---

## Configuration

### Environment Variables

```bash
NSIP_ENVIRONMENT=development|staging|production
NSIP_TOOL_SEARCH_ENABLED=true|false
NSIP_PTC_ENABLED=true|false
NSIP_PTC_TIMEOUT=60
```

### Feature Flags

```python
from src.tools.config.feature_flags import is_enabled

if is_enabled("TOOL_SEARCH_ENABLED"):
    # Tool search is available

if is_enabled("PTC_ENABLED"):
    # Programmatic Tool Calling is available
```

---

## Integration with Claude

### API Configuration

```python
from src.tools import get_api_configuration

config = get_api_configuration()
# Returns:
# {
#     "beta": "advanced-tool-use-2025-11-20",
#     "model": "claude-sonnet-4-5-20250929",
#     "tools": [...],  # Tool definitions
#     "headers": {"anthropic-beta": "advanced-tool-use-2025-11-20"}
# }
```

### Required Beta Header

```python
headers = {
    "anthropic-beta": "advanced-tool-use-2025-11-20"
}
```

---

## Related Documentation

- [Tool Search Guide](./TOOL_SEARCH_GUIDE.md)
- [PTC Implementation](./PTC_IMPLEMENTATION.md)
- [Tool Examples Reference](./TOOL_EXAMPLES_REFERENCE.md)
- [AI Agent Integration](./AI_AGENT_INTEGRATION.md)
