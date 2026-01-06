"""
Advanced Database Connection Pooling with Auto-Scaling
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from queue import Queue, Empty
import psycopg2.pool
from psycopg2.extensions import connection as pg_connection
from psycopg2 import OperationalError, InterfaceError

logger = logging.getLogger(__name__)


@dataclass
class PoolMetrics:
    """Connection pool performance metrics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    waiting_clients: int = 0
    connection_errors: int = 0
    avg_wait_time: float = 0.0
    peak_connections: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class PoolConfig:
    """Connection pool configuration."""
    min_connections: int = 5
    max_connections: int = 50
    max_idle_time: int = 300  # seconds
    max_lifetime: int = 3600  # seconds
    connection_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 0.1
    health_check_interval: int = 60
    auto_scale_enabled: bool = True
    scale_up_threshold: float = 0.8  # Scale up when 80% of connections are active
    scale_down_threshold: float = 0.2  # Scale down when 20% of connections are active


class ConnectionHealthChecker:
    """Health checking for database connections."""

    def __init__(self, test_query: str = "SELECT 1"):
        self.test_query = test_query

    def is_connection_healthy(self, conn: pg_connection) -> bool:
        """Check if a connection is healthy."""
        try:
            with conn.cursor() as cursor:
                cursor.execute(self.test_query)
                cursor.fetchone()
            return True
        except (OperationalError, InterfaceError):
            return False

    def validate_pool_connections(self, pool) -> Dict[str, Any]:
        """Validate all connections in the pool."""
        healthy = 0
        unhealthy = 0

        try:
            # This is a simplified check - actual implementation would depend on pool type
            # For demonstration, we'll assume pool has a way to iterate connections
            return {
                'healthy_connections': healthy,
                'unhealthy_connections': unhealthy,
                'total_checked': healthy + unhealthy
            }
        except Exception as e:
            logger.error(f"Pool validation failed: {e}")
            return {'error': str(e)}


class AdvancedConnectionPool:
    """Advanced connection pool with auto-scaling and monitoring."""

    def __init__(self, config: PoolConfig, connection_params: Dict[str, Any]):
        self.config = config
        self.connection_params = connection_params
        self.health_checker = ConnectionHealthChecker()

        # Initialize the underlying PostgreSQL connection pool
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=config.min_connections,
            maxconn=config.max_connections,
            **connection_params
        )

        # Metrics and monitoring
        self.metrics = PoolMetrics()
        self.metrics_history: List[PoolMetrics] = []
        self.wait_times: List[float] = []

        # Auto-scaling
        self.scaling_lock = threading.Lock()
        self.last_scale_time = time.time()

        # Health monitoring
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self.health_monitor_thread.start()

        logger.info(f"Advanced connection pool initialized with {config.min_connections}-{config.max_connections} connections")

    def get_connection(self) -> pg_connection:
        """Get a connection from the pool with monitoring."""
        start_time = time.time()

        try:
            conn = self.pool.getconn()
            wait_time = time.time() - start_time

            # Update metrics
            with self.scaling_lock:
                self.metrics.active_connections += 1
                self.metrics.pool_hits += 1
                self.wait_times.append(wait_time)
                if len(self.wait_times) > 1000:  # Keep last 1000 wait times
                    self.wait_times.pop(0)

            # Check connection health
            if not self.health_checker.is_connection_healthy(conn):
                logger.warning("Unhealthy connection detected, attempting to replace")
                self.pool.putconn(conn, close=True)
                conn = self.pool.getconn()

            return conn

        except psycopg2.pool.PoolError as e:
            wait_time = time.time() - start_time
            with self.scaling_lock:
                self.metrics.pool_misses += 1
                self.metrics.waiting_clients += 1
                self.wait_times.append(wait_time)

            logger.error(f"Connection pool exhausted: {e}")
            raise

    def return_connection(self, conn: pg_connection):
        """Return a connection to the pool."""
        try:
            # Quick health check before returning
            if self.health_checker.is_connection_healthy(conn):
                self.pool.putconn(conn)
            else:
                logger.warning("Returning unhealthy connection to pool")
                self.pool.putconn(conn, close=True)

            with self.scaling_lock:
                self.metrics.active_connections -= 1
                if self.metrics.active_connections < 0:
                    self.metrics.active_connections = 0

        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")

    @contextmanager
    def connection(self):
        """Context manager for getting and returning connections."""
        conn = None
        try:
            conn = self.get_connection()
            yield conn
        finally:
            if conn:
                self.return_connection(conn)

    def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while True:
            try:
                time.sleep(self.config.health_check_interval)
                self._perform_health_check()
                self._auto_scale_if_needed()
                self._record_metrics()

            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    def _perform_health_check(self):
        """Perform periodic health checks."""
        try:
            # Get pool statistics
            stats = self.pool._get_stats()

            with self.scaling_lock:
                self.metrics.total_connections = stats.get('total_connections', 0)
                self.metrics.idle_connections = stats.get('idle_connections', 0)
                self.metrics.peak_connections = max(
                    self.metrics.peak_connections,
                    self.metrics.active_connections
                )

            # Validate a sample of connections
            health_stats = self.health_checker.validate_pool_connections(self.pool)

            if health_stats.get('unhealthy_connections', 0) > 0:
                logger.warning(f"Found {health_stats['unhealthy_connections']} unhealthy connections")

        except Exception as e:
            logger.error(f"Health check failed: {e}")

    def _auto_scale_if_needed(self):
        """Auto-scale the connection pool based on usage patterns."""
        if not self.config.auto_scale_enabled:
            return

        current_time = time.time()
        if current_time - self.last_scale_time < 60:  # Scale at most once per minute
            return

        with self.scaling_lock:
            active_ratio = self.metrics.active_connections / max(self.metrics.total_connections, 1)

            if active_ratio >= self.config.scale_up_threshold:
                self._scale_up()
            elif active_ratio <= self.config.scale_down_threshold and self.metrics.total_connections > self.config.min_connections:
                self._scale_down()

    def _scale_up(self):
        """Increase pool size."""
        try:
            current_max = self.pool.maxconn
            new_max = min(current_max + 5, 100)  # Cap at 100 connections

            if new_max > current_max:
                self.pool.maxconn = new_max
                self.last_scale_time = time.time()
                logger.info(f"Scaled up connection pool to max {new_max} connections")

        except Exception as e:
            logger.error(f"Scale up failed: {e}")

    def _scale_down(self):
        """Decrease pool size."""
        try:
            current_max = self.pool.maxconn
            new_max = max(current_max - 2, self.config.min_connections)

            if new_max < current_max:
                self.pool.maxconn = new_max
                self.last_scale_time = time.time()
                logger.info(f"Scaled down connection pool to max {new_max} connections")

        except Exception as e:
            logger.error(f"Scale down failed: {e}")

    def _record_metrics(self):
        """Record current metrics for historical analysis."""
        with self.scaling_lock:
            current_metrics = PoolMetrics(
                total_connections=self.metrics.total_connections,
                active_connections=self.metrics.active_connections,
                idle_connections=self.metrics.idle_connections,
                waiting_clients=self.metrics.waiting_clients,
                connection_errors=self.metrics.connection_errors,
                avg_wait_time=sum(self.wait_times[-100:]) / max(len(self.wait_times[-100:]), 1),
                peak_connections=self.metrics.peak_connections,
                pool_hits=self.metrics.pool_hits,
                pool_misses=self.metrics.pool_misses
            )

            self.metrics_history.append(current_metrics)
            if len(self.metrics_history) > 1000:  # Keep last 1000 records
                self.metrics_history.pop(0)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current pool metrics."""
        with self.scaling_lock:
            hit_rate = (
                self.metrics.pool_hits /
                max(self.metrics.pool_hits + self.metrics.pool_misses, 1)
            ) * 100

            return {
                'current': {
                    'total_connections': self.metrics.total_connections,
                    'active_connections': self.metrics.active_connections,
                    'idle_connections': self.metrics.idle_connections,
                    'waiting_clients': self.metrics.waiting_clients,
                    'connection_errors': self.metrics.connection_errors,
                    'avg_wait_time': self.metrics.avg_wait_time,
                    'peak_connections': self.metrics.peak_connections,
                    'pool_hit_rate': hit_rate
                },
                'config': {
                    'min_connections': self.config.min_connections,
                    'max_connections': self.config.max_connections,
                    'auto_scale_enabled': self.config.auto_scale_enabled
                },
                'history_size': len(self.metrics_history)
            }

    def close(self):
        """Close the connection pool."""
        try:
            self.pool.closeall()
            logger.info("Connection pool closed")
        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")


class PoolManager:
    """Manager for multiple connection pools."""

    def __init__(self):
        self.pools: Dict[str, AdvancedConnectionPool] = {}
        self.lock = threading.Lock()

    def create_pool(self, name: str, config: PoolConfig, connection_params: Dict[str, Any]) -> AdvancedConnectionPool:
        """Create a new connection pool."""
        with self.lock:
            if name in self.pools:
                logger.warning(f"Pool {name} already exists, returning existing")
                return self.pools[name]

            pool = AdvancedConnectionPool(config, connection_params)
            self.pools[name] = pool
            logger.info(f"Created connection pool: {name}")
            return pool

    def get_pool(self, name: str) -> Optional[AdvancedConnectionPool]:
        """Get a connection pool by name."""
        return self.pools.get(name)

    def close_all_pools(self):
        """Close all managed pools."""
        with self.lock:
            for name, pool in self.pools.items():
                try:
                    pool.close()
                    logger.info(f"Closed pool: {name}")
                except Exception as e:
                    logger.error(f"Error closing pool {name}: {e}")

            self.pools.clear()

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all pools."""
        return {name: pool.get_metrics() for name, pool in self.pools.items()}
