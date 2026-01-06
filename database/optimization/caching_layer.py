"""
Database Caching Layer with Redis Integration
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from functools import wraps
import redis
from redis.exceptions import RedisError, ConnectionError

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[int, Union[int, bytes]] = field(default_factory=dict)
    health_check_interval: int = 30
    max_connections: int = 20

    # Cache settings
    default_ttl: int = 3600  # 1 hour
    max_memory: str = '512mb'
    eviction_policy: str = 'allkeys-lru'


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    ttl: int
    created_at: float
    hits: int = 0
    last_accessed: float = field(default_factory=time.time)


class CacheMetrics:
    """Cache performance metrics."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.errors = 0
        self.total_requests = 0

    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'errors': self.errors,
            'total_requests': self.total_requests,
            'hit_rate': self.hit_rate()
        }


class RedisCache:
    """Redis-based caching layer for database operations."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.metrics = CacheMetrics()

        # Initialize Redis connection pool
        self.redis_pool = redis.ConnectionPool(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            socket_timeout=config.socket_timeout,
            socket_connect_timeout=config.socket_connect_timeout,
            socket_keepalive=config.socket_keepalive,
            socket_keepalive_options=config.socket_keepalive_options,
            max_connections=config.max_connections,
            health_check_interval=config.health_check_interval
        )

        # Test connection
        self._test_connection()

        # Configure Redis settings
        self._configure_redis()

        logger.info("Redis cache initialized successfully")

    def _test_connection(self):
        """Test Redis connection."""
        try:
            with redis.Redis(connection_pool=self.redis_pool) as client:
                client.ping()
        except ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def _configure_redis(self):
        """Configure Redis settings."""
        try:
            with redis.Redis(connection_pool=self.redis_pool) as client:
                # Set max memory
                client.config_set('maxmemory', self.config.max_memory)
                client.config_set('maxmemory-policy', self.config.eviction_policy)

                logger.info(f"Redis configured: maxmemory={self.config.max_memory}, policy={self.config.eviction_policy}")

        except RedisError as e:
            logger.warning(f"Failed to configure Redis: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self.metrics.total_requests += 1

        try:
            with redis.Redis(connection_pool=self.redis_pool) as client:
                value = client.get(key)
                if value is not None:
                    self.metrics.hits += 1
                    # Update last accessed time
                    client.hset(f"meta:{key}", "last_accessed", time.time())
                    client.hincrby(f"meta:{key}", "hits", 1)

                    return json.loads(value)
                else:
                    self.metrics.misses += 1
                    return None

        except RedisError as e:
            self.metrics.errors += 1
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        try:
            serialized_value = json.dumps(value)
            ttl_value = ttl or self.config.default_ttl

            with redis.Redis(connection_pool=self.redis_pool) as client:
                result = client.setex(key, ttl_value, serialized_value)

                # Store metadata
                client.hset(f"meta:{key}", mapping={
                    "created_at": time.time(),
                    "ttl": ttl_value,
                    "hits": 0,
                    "last_accessed": time.time()
                })

                return result

        except (RedisError, TypeError) as e:
            self.metrics.errors += 1
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            with redis.Redis(connection_pool=self.redis_pool) as client:
                # Delete both data and metadata
                client.delete(key)
                client.delete(f"meta:{key}")
                return True

        except RedisError as e:
            self.metrics.errors += 1
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            with redis.Redis(connection_pool=self.redis_pool) as client:
                return client.exists(key) > 0

        except RedisError as e:
            self.metrics.errors += 1
            logger.error(f"Cache exists error for key {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            with redis.Redis(connection_pool=self.redis_pool) as client:
                client.flushdb()
                return True

        except RedisError as e:
            self.metrics.errors += 1
            logger.error(f"Cache clear error: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        try:
            with redis.Redis(connection_pool=self.redis_pool) as client:
                info = client.info()
                memory_info = client.memory_stats()

                return {
                    'cache_metrics': self.metrics.to_dict(),
                    'redis_info': {
                        'connected_clients': info.get('connected_clients', 0),
                        'used_memory': info.get('used_memory_human', '0B'),
                        'total_connections_received': info.get('total_connections_received', 0),
                        'evicted_keys': info.get('evicted_keys', 0),
                        'keyspace_hits': info.get('keyspace_hits', 0),
                        'keyspace_misses': info.get('keyspace_misses', 0),
                    },
                    'memory_stats': memory_info
                }

        except RedisError as e:
            logger.error(f"Failed to get cache metrics: {e}")
            return {'error': str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        try:
            with redis.Redis(connection_pool=self.redis_pool) as client:
                start_time = time.time()
                client.ping()
                response_time = time.time() - start_time

                return {
                    'status': 'healthy',
                    'response_time': response_time,
                    'pool_size': self.redis_pool._available_connections.qsize() if hasattr(self.redis_pool, '_available_connections') else 0
                }

        except RedisError as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


class QueryResultCache:
    """Cache for database query results."""

    def __init__(self, cache: RedisCache):
        self.cache = cache

    def generate_cache_key(self, query: str, params: Dict[str, Any] = None) -> str:
        """Generate a unique cache key for a query."""
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        params_str = json.dumps(params or {}, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]

        return f"query:{query_hash}:{params_hash}"

    def get_query_result(self, query: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """Get cached query result."""
        cache_key = self.generate_cache_key(query, params)
        return self.cache.get(cache_key)

    def set_query_result(self, query: str, result: Any, params: Dict[str, Any] = None, ttl: Optional[int] = None):
        """Cache query result."""
        cache_key = self.generate_cache_key(query, params)
        self.cache.set(cache_key, result, ttl)

    def invalidate_query_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern."""
        try:
            with redis.Redis(connection_pool=self.cache.redis_pool) as client:
                keys = client.keys(f"query:{pattern}*")
                if keys:
                    client.delete(*keys)
                    logger.info(f"Invalidated {len(keys)} cache entries matching pattern: {pattern}")

        except RedisError as e:
            logger.error(f"Cache invalidation error: {e}")


class SessionCache:
    """Cache for user sessions and authentication."""

    def __init__(self, cache: RedisCache):
        self.cache = cache

    def store_session(self, session_id: str, user_data: Dict[str, Any], ttl: int = 3600):
        """Store user session data."""
        cache_key = f"session:{session_id}"
        self.cache.set(cache_key, user_data, ttl)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user session data."""
        cache_key = f"session:{session_id}"
        return self.cache.get(cache_key)

    def delete_session(self, session_id: str):
        """Delete user session."""
        cache_key = f"session:{session_id}"
        self.cache.delete(cache_key)

    def extend_session(self, session_id: str, ttl: int = 3600):
        """Extend session TTL."""
        cache_key = f"session:{session_id}"
        session_data = self.cache.get(cache_key)
        if session_data:
            self.cache.set(cache_key, session_data, ttl)


def cached_query(ttl: Optional[int] = None):
    """Decorator for caching database query results."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'query_cache'):
                return func(self, *args, **kwargs)

            # Generate cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            cache_key = hashlib.sha256(':'.join(key_parts).encode()).hexdigest()

            # Try to get from cache
            cached_result = self.query_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute query and cache result
            result = func(self, *args, **kwargs)
            self.query_cache.set(cache_key, result, ttl)
            return result

        return wrapper
    return decorator


class CacheManager:
    """Unified cache management system."""

    def __init__(self, cache_config: CacheConfig):
        self.cache = RedisCache(cache_config)
        self.query_cache = QueryResultCache(self.cache)
        self.session_cache = SessionCache(self.cache)

    def warmup_cache(self, queries: List[Dict[str, Any]]):
        """Warm up cache with frequently used queries."""
        logger.info(f"Warming up cache with {len(queries)} queries")

        for query_info in queries:
            try:
                query = query_info['query']
                params = query_info.get('params', {})
                ttl = query_info.get('ttl', self.cache.config.default_ttl)

                # Execute query and cache result
                # This would typically be done through the database connection
                # For now, we'll just log the intent
                logger.debug(f"Would cache query: {query[:50]}...")

            except Exception as e:
                logger.error(f"Cache warmup failed for query: {e}")

    def invalidate_table_cache(self, table_name: str):
        """Invalidate all cache entries related to a table."""
        pattern = f"*{table_name}*"
        self.query_cache.invalidate_query_pattern(pattern)
        logger.info(f"Invalidated cache for table: {table_name}")

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics."""
        return {
            'cache_health': self.cache.health_check(),
            'cache_performance': self.cache.get_metrics(),
            'query_cache_stats': {
                'enabled': True,  # Would track actual usage
                'hit_rate': 0.0   # Would calculate from usage
            },
            'session_cache_stats': {
                'active_sessions': 0  # Would count active sessions
            }
        }
