"""
Result Processing for Tool Execution.

Handles result transformation, caching, and aggregation.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import logging


logger = logging.getLogger(__name__)


@dataclass
class ProcessedResult:
    """A processed tool result with metadata."""
    tool_name: str
    raw_result: Any
    processed_result: Any
    execution_time_ms: int
    cache_key: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResultCache:
    """
    In-memory cache for tool results.

    Implements LRU eviction and TTL expiration.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._access_order: List[str] = []

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache if not expired."""
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]

        # Check TTL
        if (datetime.now() - timestamp).total_seconds() > self.ttl_seconds:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return None

        # Update access order (LRU)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        return value

    def set(self, key: str, value: Any) -> None:
        """Set a value in cache."""
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]

        self._cache[key] = (value, datetime.now())
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._access_order.clear()

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


class ResultProcessor:
    """
    Processes and transforms tool execution results.

    Features:
    - Result normalization
    - Caching with key generation
    - Aggregation for batch results
    - Format conversion
    """

    def __init__(self, enable_cache: bool = True) -> None:
        self.enable_cache = enable_cache
        self.cache = ResultCache() if enable_cache else None

    def process(
        self,
        tool_name: str,
        raw_result: Any,
        execution_time_ms: int,
        params: Optional[Dict[str, Any]] = None
    ) -> ProcessedResult:
        """
        Process a raw tool result.

        Args:
            tool_name: Name of the tool that produced the result
            raw_result: Raw result from tool execution
            execution_time_ms: Time taken for execution
            params: Original parameters (for cache key)

        Returns:
            ProcessedResult with normalized data
        """
        cache_key = None
        if params and self.enable_cache:
            cache_key = self._generate_cache_key(tool_name, params)

            # Check cache
            cached = self.cache.get(cache_key)
            if cached is not None:
                return ProcessedResult(
                    tool_name=tool_name,
                    raw_result=cached,
                    processed_result=cached,
                    execution_time_ms=0,
                    cache_key=cache_key,
                    metadata={"from_cache": True}
                )

        # Normalize result
        processed = self._normalize_result(raw_result)

        # Cache if enabled
        if cache_key and self.cache:
            self.cache.set(cache_key, processed)

        return ProcessedResult(
            tool_name=tool_name,
            raw_result=raw_result,
            processed_result=processed,
            execution_time_ms=execution_time_ms,
            cache_key=cache_key,
            metadata={"from_cache": False}
        )

    def aggregate(self, results: List[ProcessedResult]) -> Dict[str, Any]:
        """
        Aggregate multiple results into a summary.

        Args:
            results: List of processed results to aggregate

        Returns:
            Aggregated summary
        """
        if not results:
            return {
                "count": 0,
                "success_rate": 0.0,
                "avg_execution_time_ms": 0,
                "results": []
            }

        successful = [r for r in results if r.processed_result is not None]

        return {
            "count": len(results),
            "success_rate": len(successful) / len(results),
            "avg_execution_time_ms": sum(r.execution_time_ms for r in results) / len(results),
            "total_execution_time_ms": sum(r.execution_time_ms for r in results),
            "cache_hit_rate": sum(1 for r in results if r.metadata.get("from_cache")) / len(results),
            "results": [r.processed_result for r in results]
        }

    def to_json(self, result: ProcessedResult) -> str:
        """Convert result to JSON string."""
        return json.dumps({
            "tool": result.tool_name,
            "result": result.processed_result,
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp.isoformat(),
            "metadata": result.metadata
        }, default=str)

    def _generate_cache_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Generate a cache key from tool name and parameters."""
        key_data = f"{tool_name}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _normalize_result(self, result: Any) -> Any:
        """Normalize a result to a consistent format."""
        if result is None:
            return None

        if isinstance(result, dict):
            return {k: self._normalize_result(v) for k, v in result.items()}

        if isinstance(result, list):
            return [self._normalize_result(item) for item in result]

        if isinstance(result, (str, int, float, bool)):
            return result

        if isinstance(result, datetime):
            return result.isoformat()

        # Convert other types to string
        return str(result)
