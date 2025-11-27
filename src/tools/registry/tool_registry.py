"""
Central Tool Registry with Advanced Tool Use Features.

Manages:
- Tool registration and discovery
- Deferred loading for context optimization
- Search indexing for Tool Search Tool
- Caller validation for PTC
"""

from typing import Dict, List, Optional, Type, Set
from dataclasses import dataclass, field
import logging
import re

from ..definitions.base_tool import (
    BaseTool,
    ToolCategory,
    LoadingStrategy,
    CallerType
)


logger = logging.getLogger(__name__)


@dataclass
class RegistryStats:
    """Statistics about registered tools."""
    total_tools: int = 0
    always_loaded: int = 0
    deferred: int = 0
    ptc_enabled: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    estimated_always_loaded_tokens: int = 0


class ToolRegistry:
    """
    Central registry for all NSIP tools.

    Implements Tool Search Tool pattern:
    - Core tools always loaded (~3K tokens)
    - Specialized tools deferred (~50K+ tokens saved)
    - Dynamic discovery via search
    """

    _instance: Optional['ToolRegistry'] = None

    def __new__(cls) -> 'ToolRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._tools: Dict[str, BaseTool] = {}
        self._deferred_tools: Dict[str, Type[BaseTool]] = {}
        self._search_index: List[Dict] = []
        self._category_index: Dict[ToolCategory, Set[str]] = {
            cat: set() for cat in ToolCategory
        }
        self._ptc_enabled: Set[str] = set()
        self._initialized = True
        logger.info("ToolRegistry initialized")

    def register(self, tool_class: Type[BaseTool]) -> None:
        """
        Register a tool with the registry.

        Handles loading strategy:
        - ALWAYS_LOADED: Instantiate immediately
        - DEFERRED/LAZY: Store class for lazy instantiation
        """
        temp_instance = tool_class()
        metadata = temp_instance.metadata

        if metadata.loading_strategy == LoadingStrategy.ALWAYS_LOADED:
            self._tools[metadata.name] = temp_instance
            logger.info(f"Registered (always loaded): {metadata.name}")
        else:
            self._deferred_tools[metadata.name] = tool_class
            logger.info(f"Registered (deferred): {metadata.name}")

        self._category_index[metadata.category].add(metadata.name)
        self._search_index.append(temp_instance.get_search_index_entry())

        if CallerType.CODE_EXECUTION in metadata.allowed_callers or \
           CallerType.BOTH in metadata.allowed_callers:
            self._ptc_enabled.add(metadata.name)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name, instantiating if deferred."""
        if name in self._tools:
            return self._tools[name]

        if name in self._deferred_tools:
            tool_class = self._deferred_tools[name]
            instance = tool_class()
            self._tools[name] = instance
            logger.info(f"Lazily instantiated: {name}")
            return instance

        return None

    def search(
        self,
        query: str,
        category: Optional[ToolCategory] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for tools matching query.

        Core of Tool Search Tool functionality using
        regex matching on name, description, and keywords.
        """
        results = []
        query_lower = query.lower()
        query_pattern = re.compile(re.escape(query_lower), re.IGNORECASE)

        for entry in self._search_index:
            if category and entry["category"] != category.name:
                continue

            score = 0.0
            boost = entry.get("boost", 1.0)

            if query_lower in entry["name"].lower():
                score += 10.0 * boost

            if query_pattern.search(entry["description"]):
                score += 5.0 * boost

            for keyword in entry.get("keywords", []):
                if query_lower in keyword.lower():
                    score += 3.0 * boost

            for tag in entry.get("tags", []):
                if query_lower in tag.lower():
                    score += 2.0 * boost

            if score > 0:
                results.append({**entry, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def get_always_loaded_definitions(self) -> List[Dict]:
        """Get API definitions for always-loaded tools."""
        return [
            tool.to_api_definition()
            for tool in self._tools.values()
            if tool.metadata.loading_strategy == LoadingStrategy.ALWAYS_LOADED
        ]

    def get_tool_definition(self, name: str) -> Optional[Dict]:
        """Get API definition for a specific tool."""
        tool = self.get_tool(name)
        return tool.to_api_definition() if tool else None

    def get_ptc_enabled_tools(self) -> List[BaseTool]:
        """Get list of tools that support Programmatic Tool Calling."""
        return [self.get_tool(name) for name in self._ptc_enabled if self.get_tool(name)]

    def get_categories(self) -> List[ToolCategory]:
        """Get list of categories that have registered tools."""
        return [
            cat for cat, names in self._category_index.items()
            if len(names) > 0
        ]

    def get_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get all tools in a specific category."""
        names = self._category_index.get(category, set())
        return [self.get_tool(name) for name in names if self.get_tool(name)]

    def get_stats(self) -> RegistryStats:
        """Get registry statistics."""
        stats = RegistryStats(
            total_tools=len(self._search_index),
            always_loaded=len([
                t for t in self._tools.values()
                if t.metadata.loading_strategy == LoadingStrategy.ALWAYS_LOADED
            ]),
            deferred=len(self._deferred_tools),
            ptc_enabled=len(self._ptc_enabled)
        )

        for entry in self._search_index:
            cat = entry["category"]
            stats.by_category[cat] = stats.by_category.get(cat, 0) + 1

        stats.estimated_always_loaded_tokens = stats.always_loaded * 500
        return stats

    def validate_ptc_caller(self, tool_name: str, caller: str) -> bool:
        """Validate that a caller can invoke tool via PTC."""
        tool = self.get_tool(tool_name)
        if not tool:
            return False

        allowed = tool.metadata.allowed_callers
        if CallerType.BOTH in allowed:
            return True

        try:
            caller_type = CallerType(caller)
            return caller_type in allowed
        except ValueError:
            return False


# Global registry singleton
registry = ToolRegistry()


def register_tool(tool_class: Type[BaseTool]) -> Type[BaseTool]:
    """Decorator for automatic tool registration."""
    registry.register(tool_class)
    return tool_class
