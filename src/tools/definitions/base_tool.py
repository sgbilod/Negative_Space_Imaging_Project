"""
Base Tool Definition with Advanced Tool Use Features.

Implements:
- Tool Search Tool metadata and search indexing
- Programmatic Tool Calling via allowed_callers and output schemas
- Tool Use Examples for parameter accuracy improvement

Reference: https://www.anthropic.com/engineering/advanced-tool-use
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from datetime import datetime
import json


class ToolCategory(Enum):
    """Tool categorization for search optimization."""
    IMAGING_CORE = auto()
    IMAGING_ADVANCED = auto()
    DATABASE = auto()
    SECURITY = auto()
    EXPORT = auto()
    ML_INFERENCE = auto()
    SPECIALIZED_MEDICAL = auto()
    SPECIALIZED_ASTRO = auto()
    HPC = auto()
    ADMIN = auto()
    UTILITY = auto()


class LoadingStrategy(Enum):
    """Tool loading strategy for context optimization."""
    ALWAYS_LOADED = "always"
    DEFERRED = "deferred"
    LAZY = "lazy"


class CallerType(Enum):
    """Permitted invocation methods."""
    DIRECT = "direct"
    CODE_EXECUTION = "code_execution_20250825"
    BOTH = "both"


@dataclass(frozen=True)
class ToolExample:
    """
    Concrete example of tool invocation.

    Improves parameter accuracy from 72% to 90% by providing
    the model with explicit input/output patterns.
    """
    description: str
    input_params: Dict[str, Any]
    expected_output_shape: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API transmission."""
        return {
            "description": self.description,
            "input": self.input_params,
            "expected_output": self.expected_output_shape,
            "notes": self.notes
        }


@dataclass
class ToolMetadata:
    """Complete tool metadata for registration and discovery."""
    name: str
    description: str
    category: ToolCategory
    loading_strategy: LoadingStrategy
    allowed_callers: List[CallerType]
    version: str = "1.0.0"
    deprecated: bool = False
    deprecation_message: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    search_keywords: List[str] = field(default_factory=list)
    search_boost: float = 1.0
    estimated_duration_ms: Optional[int] = None
    idempotent: bool = True
    supports_batch: bool = False
    max_batch_size: Optional[int] = None


@dataclass
class InputSchema:
    """JSON Schema definition for tool inputs."""
    properties: Dict[str, Dict[str, Any]]
    required: List[str] = field(default_factory=list)
    additional_properties: bool = False

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        return {
            "type": "object",
            "properties": self.properties,
            "required": self.required,
            "additionalProperties": self.additional_properties
        }


@dataclass
class OutputSchema:
    """
    Schema for tool outputs - critical for PTC.

    Claude uses this to generate correct parsing code
    in the sandbox environment.
    """
    description: str
    properties: Dict[str, Dict[str, Any]]

    def to_documentation(self) -> str:
        """Generate human-readable output documentation."""
        lines = [f"Returns: {self.description}", ""]
        for name, spec in self.properties.items():
            type_str = spec.get("type", "any")
            desc = spec.get("description", "")
            lines.append(f"  - {name} ({type_str}): {desc}")
        return "\n".join(lines)


class BaseTool(ABC):
    """
    Abstract base class for all NSIP tools.

    Implements all three Advanced Tool Use features:
    1. Tool Search Tool - via metadata and search keywords
    2. Programmatic Tool Calling - via allowed_callers and output schema
    3. Tool Use Examples - via examples property

    Subclasses must implement:
    - metadata: Tool registration information
    - input_schema: Parameter definitions
    - output_schema: Return value definitions
    - examples: Concrete usage examples
    - execute: Async execution logic
    """

    def __init__(self) -> None:
        self._registered_at: Optional[datetime] = None
        self._invocation_count: int = 0
        self._last_invoked: Optional[datetime] = None

    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Tool metadata for registration and search."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> InputSchema:
        """Input parameter schema."""
        ...

    @property
    @abstractmethod
    def output_schema(self) -> OutputSchema:
        """Output schema for PTC code generation."""
        ...

    @property
    @abstractmethod
    def examples(self) -> List[ToolExample]:
        """
        Concrete usage examples.

        Must include:
        - Minimal invocation (required params only)
        - Typical invocation (common use case)
        - Full invocation (all parameters)
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        ...

    async def validate_input(self, **kwargs: Any) -> bool:
        """Validate input against schema."""
        for field_name in self.input_schema.required:
            if field_name not in kwargs:
                raise ValueError(f"Missing required field: {field_name}")
        return True

    def to_api_definition(self) -> Dict[str, Any]:
        """
        Generate API-compatible tool definition.

        Includes all Advanced Tool Use features:
        - defer_loading for Tool Search Tool
        - allowed_callers for PTC
        - input_examples for accuracy
        """
        description = self.metadata.description
        if self.output_schema:
            description += f"\n\n{self.output_schema.to_documentation()}"
        if self.metadata.search_keywords:
            description += f"\n\nKeywords: {', '.join(self.metadata.search_keywords)}"

        return {
            "name": self.metadata.name,
            "description": description,
            "input_schema": self.input_schema.to_json_schema(),
            "defer_loading": self.metadata.loading_strategy != LoadingStrategy.ALWAYS_LOADED,
            "allowed_callers": [
                c.value for c in self.metadata.allowed_callers
                if c != CallerType.DIRECT
            ],
            "input_examples": [ex.input_params for ex in self.examples]
        }

    def get_search_index_entry(self) -> Dict[str, Any]:
        """Generate entry for tool search index."""
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "category": self.metadata.category.name,
            "keywords": self.metadata.search_keywords,
            "tags": self.metadata.tags,
            "boost": self.metadata.search_boost
        }

    def __repr__(self) -> str:
        return f"<Tool: {self.metadata.name} ({self.metadata.category.name})>"
