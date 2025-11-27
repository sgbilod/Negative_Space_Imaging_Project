"""
Deferred Tool Loader.

Implements lazy loading mechanism for tools to optimize context usage.
Tools are only instantiated when explicitly requested or discovered via search.

Reference: Anthropic Advanced Tool Use - Tool Search Tool
"""

from typing import Dict, List, Optional, Set, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from datetime import datetime

from ..definitions.base_tool import (
    BaseTool,
    ToolMetadata,
    LoadingStrategy,
    ToolCategory
)


logger = logging.getLogger(__name__)


class LoadState(Enum):
    """Loading state for a tool."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"


@dataclass
class DeferredToolEntry:
    """Entry for a deferred tool."""
    tool_class: Type[BaseTool]
    metadata: ToolMetadata
    state: LoadState = LoadState.NOT_LOADED
    instance: Optional[BaseTool] = None
    load_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class LoaderStats:
    """Statistics for the deferred loader."""
    total_deferred: int = 0
    currently_loaded: int = 0
    load_failures: int = 0
    average_load_time_ms: float = 0.0
    most_accessed: List[str] = field(default_factory=list)


class DeferredToolLoader:
    """
    Manages lazy loading of tools.

    Benefits:
    - Reduces initial context from ~55K to ~3K tokens
    - Tools loaded only when needed
    - Supports preloading frequently used tools
    - Tracks usage patterns for optimization
    """

    def __init__(self, max_cached: int = 50) -> None:
        self._entries: Dict[str, DeferredToolEntry] = {}
        self._max_cached = max_cached
        self._load_order: List[str] = []
        self._load_callbacks: List[Callable[[str, BaseTool], None]] = []

    def register(self, tool_class: Type[BaseTool]) -> None:
        """
        Register a tool class for deferred loading.

        The tool is not instantiated until explicitly requested.
        """
        temp_instance = tool_class()
        metadata = temp_instance.metadata

        if metadata.loading_strategy == LoadingStrategy.ALWAYS_LOADED:
            logger.debug(f"Skipping deferred registration for always-loaded: {metadata.name}")
            return

        self._entries[metadata.name] = DeferredToolEntry(
            tool_class=tool_class,
            metadata=metadata
        )
        logger.info(f"Registered for deferred loading: {metadata.name}")

    def get(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool, loading it if necessary.

        Implements LRU-style eviction if cache is full.
        """
        if name not in self._entries:
            return None

        entry = self._entries[name]
        entry.access_count += 1
        entry.last_accessed = datetime.now()

        if entry.state == LoadState.LOADED and entry.instance:
            return entry.instance

        if entry.state == LoadState.FAILED:
            logger.warning(f"Skipping failed tool: {name}")
            return None

        return self._load_tool(name)

    def _load_tool(self, name: str) -> Optional[BaseTool]:
        """Load a single tool."""
        entry = self._entries[name]
        entry.state = LoadState.LOADING
        start_time = datetime.now()

        try:
            entry.instance = entry.tool_class()
            entry.state = LoadState.LOADED
            entry.load_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            self._load_order.append(name)
            self._evict_if_needed()

            for callback in self._load_callbacks:
                callback(name, entry.instance)

            logger.info(f"Loaded tool: {name} in {entry.load_time_ms}ms")
            return entry.instance

        except Exception as e:
            entry.state = LoadState.FAILED
            entry.error_message = str(e)
            logger.error(f"Failed to load tool {name}: {e}")
            return None

    def _evict_if_needed(self) -> None:
        """Evict least recently used tools if cache is full."""
        while len(self._load_order) > self._max_cached:
            oldest = self._load_order.pop(0)
            if oldest in self._entries:
                entry = self._entries[oldest]
                entry.instance = None
                entry.state = LoadState.NOT_LOADED
                logger.debug(f"Evicted tool: {oldest}")

    async def preload(self, names: List[str]) -> Dict[str, bool]:
        """
        Preload multiple tools in parallel.

        Returns dict of tool name -> success status.
        """
        results = {}

        async def load_one(name: str) -> tuple[str, bool]:
            tool = self.get(name)
            return name, tool is not None

        tasks = [load_one(name) for name in names if name in self._entries]
        completed = await asyncio.gather(*tasks)

        for name, success in completed:
            results[name] = success

        return results

    def preload_by_category(self, category: ToolCategory) -> Dict[str, bool]:
        """Preload all tools in a category."""
        names = [
            name for name, entry in self._entries.items()
            if entry.metadata.category == category
        ]
        return asyncio.run(self.preload(names))

    def get_stats(self) -> LoaderStats:
        """Get loader statistics."""
        loaded_entries = [e for e in self._entries.values() if e.state == LoadState.LOADED]
        failed_entries = [e for e in self._entries.values() if e.state == LoadState.FAILED]

        load_times = [e.load_time_ms for e in loaded_entries if e.load_time_ms]
        avg_load_time = sum(load_times) / len(load_times) if load_times else 0.0

        most_accessed = sorted(
            self._entries.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )[:5]

        return LoaderStats(
            total_deferred=len(self._entries),
            currently_loaded=len(loaded_entries),
            load_failures=len(failed_entries),
            average_load_time_ms=avg_load_time,
            most_accessed=[name for name, _ in most_accessed]
        )

    def is_loaded(self, name: str) -> bool:
        """Check if a tool is currently loaded."""
        return (
            name in self._entries and
            self._entries[name].state == LoadState.LOADED
        )

    def unload(self, name: str) -> bool:
        """Unload a tool to free memory."""
        if name not in self._entries:
            return False

        entry = self._entries[name]
        if entry.state != LoadState.LOADED:
            return False

        entry.instance = None
        entry.state = LoadState.NOT_LOADED

        if name in self._load_order:
            self._load_order.remove(name)

        logger.info(f"Unloaded tool: {name}")
        return True

    def on_load(self, callback: Callable[[str, BaseTool], None]) -> None:
        """Register a callback for when tools are loaded."""
        self._load_callbacks.append(callback)

    def get_loaded_names(self) -> Set[str]:
        """Get names of all currently loaded tools."""
        return {
            name for name, entry in self._entries.items()
            if entry.state == LoadState.LOADED
        }

    def get_deferred_names(self) -> Set[str]:
        """Get names of all deferred (not loaded) tools."""
        return {
            name for name, entry in self._entries.items()
            if entry.state == LoadState.NOT_LOADED
        }


# Global loader instance
deferred_loader = DeferredToolLoader()
