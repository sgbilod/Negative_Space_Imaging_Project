"""
Time Series Database Storage Layer

Handles persistent storage of metrics and analytics data in PostgreSQL with:
- Partitioned table management
- Batch operations for performance
- Efficient time-range queries
- Connection pooling
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
import asyncpg

from ..core.metrics import Metric, AggregatedMetric

logger = logging.getLogger(__name__)


class TimeSeriesDatabase:
    """
    PostgreSQL-based time series database for metrics storage.

    Features:
    - Connection pooling with asyncpg
    - Automatic partition management
    - Batch insert optimization
    - Time-range query support
    - Configurable retention policies
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "analytics",
        user: str = "postgres",
        password: str = "",
        min_connections: int = 5,
        max_connections: int = 20,
        retention_days: int = 90,
    ):
        """
        Initialize time series database connection.

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
            min_connections: Minimum pool size
            max_connections: Maximum pool size
            retention_days: Days to retain metrics
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.retention_days = retention_days
        self.pool: Optional[asyncpg.pool.Pool] = None
        self._metrics_inserted = 0
        self._batches_processed = 0
        self._errors = 0

    async def connect(self) -> None:
        """
        Create connection pool.

        Raises:
            Exception: If connection fails
        """
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_connections,
                max_size=self.max_connections,
            )
            logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Disconnected from PostgreSQL")

    @asynccontextmanager
    async def _get_connection(self):
        """
        Context manager for getting a database connection from pool.

        Yields:
            asyncpg.Connection: Database connection
        """
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        connection = await self.pool.acquire()
        try:
            yield connection
        finally:
            await self.pool.release(connection)

    async def ensure_partition(self, table: str, timestamp: datetime) -> None:
        """
        Ensure partition exists for given timestamp.

        Creates partition if it doesn't exist.

        Args:
            table: Parent table name (e.g., 'metrics_timeseries')
            timestamp: Timestamp for partition
        """
        partition_date = timestamp.date()
        partition_name = f"{table}_p{partition_date.strftime('%Y%m%d')}"

        async with self._get_connection() as conn:
            # Check if partition exists
            exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = $1
                )
                """,
                partition_name
            )

            if exists:
                return

            # Create partition
            try:
                await conn.execute(f"""
                    CREATE TABLE {partition_name} PARTITION OF {table}
                    FOR VALUES FROM ('{partition_date}'::timestamp)
                    TO ('{partition_date + timedelta(days=1)}'::timestamp)
                """)
                logger.info(f"Created partition {partition_name}")
            except Exception as e:
                if "already exists" not in str(e):
                    logger.error(
                        f"Failed to create partition {partition_name}: {e}"
                    )

    async def insert_metric(self, metric: Metric) -> bool:
        """
        Insert single metric into time series database.

        Args:
            metric: Metric to insert

        Returns:
            bool: True if successful
        """
        try:
            await self.ensure_partition("metrics_timeseries", metric.timestamp)

            async with self._get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO metrics_timeseries
                    (name, value, metric_type, tags, timestamp, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    metric.name,
                    metric.value,
                    metric.metric_type,
                    metric.tags,
                    metric.timestamp,
                    metric.created_at,
                )
                self._metrics_inserted += 1
                return True
        except Exception as e:
            logger.error(f"Failed to insert metric {metric.name}: {e}")
            self._errors += 1
            return False

    async def insert_metrics_batch(self, metrics: List[Metric]) -> int:
        """
        Batch insert metrics for better performance.

        Args:
            metrics: List of metrics to insert

        Returns:
            int: Number of metrics successfully inserted
        """
        if not metrics:
            return 0

        try:
            # Ensure partitions exist
            for metric in metrics:
                await self.ensure_partition(
                    "metrics_timeseries", metric.timestamp
                )

            # Prepare batch data
            batch_data = [
                (
                    m.name,
                    m.value,
                    m.metric_type,
                    m.tags,
                    m.timestamp,
                    m.created_at,
                )
                for m in metrics
            ]

            async with self._get_connection() as conn:
                await conn.executemany(
                    """
                    INSERT INTO metrics_timeseries
                    (name, value, metric_type, tags, timestamp, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    batch_data,
                )

            self._metrics_inserted += len(metrics)
            self._batches_processed += 1
            logger.debug(f"Inserted batch of {len(metrics)} metrics")
            return len(metrics)
        except Exception as e:
            logger.error(f"Failed to insert metrics batch: {e}")
            self._errors += 1
            return 0

    async def query_metrics(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        Query metrics for given time range and name.

        Args:
            name: Metric name
            start_time: Start of time range
            end_time: End of time range
            tags: Optional tags to filter by
            limit: Maximum results to return

        Returns:
            List of metric records
        """
        try:
            query = """
                SELECT name, value, metric_type, tags, timestamp, created_at
                FROM metrics_timeseries
                WHERE name = $1
                AND timestamp >= $2
                AND timestamp <= $3
            """
            params: List[Any] = [name, start_time, end_time]

            # Add tag filtering if provided
            if tags:
                conditions = []
                param_idx = 4
                for key, value in tags.items():
                    conditions.append(f"tags->>'{key}' = ${param_idx}")
                    params.append(value)
                    param_idx += 1
                query += " AND " + " AND ".join(conditions)

            query += f" ORDER BY timestamp DESC LIMIT {limit}"

            async with self._get_connection() as conn:
                rows = await conn.fetch(query, *params)

            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to query metrics {name}: {e}")
            return []

    async def query_aggregated(
        self,
        name: str,
        aggregation_level: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query pre-aggregated metrics.

        Args:
            name: Metric name
            aggregation_level: 'minute', 'hour', 'day'
            start_time: Start of time range
            end_time: End of time range
            tags: Optional tags to filter by

        Returns:
            List of aggregated metric records
        """
        try:
            query = """
                SELECT name, aggregation_level, period_start, period_end,
                       count, min_value, max_value, avg_value, median_value,
                       stddev_value, p95_value, p99_value, created_at
                FROM aggregated_metrics
                WHERE name = $1
                AND aggregation_level = $2
                AND period_start >= $3
                AND period_end <= $4
            """
            params: List[Any] = [name, aggregation_level, start_time, end_time]

            # Add tag filtering if provided
            if tags:
                conditions = []
                param_idx = 5
                for key, value in tags.items():
                    conditions.append(f"tags->>'{key}' = ${param_idx}")
                    params.append(value)
                    param_idx += 1
                query += " AND " + " AND ".join(conditions)

            query += " ORDER BY period_start DESC"

            async with self._get_connection() as conn:
                rows = await conn.fetch(query, *params)

            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to query aggregated metrics {name}: {e}")
            return []

    async def insert_aggregated(self, aggregated: AggregatedMetric) -> bool:
        """
        Insert aggregated metric record.

        Args:
            aggregated: Aggregated metric to insert

        Returns:
            bool: True if successful
        """
        try:
            async with self._get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO aggregated_metrics
                    (name, aggregation_level, period_start, period_end,
                     count, min_value, max_value, avg_value, median_value,
                     stddev_value, p95_value, p99_value, tags, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (name, aggregation_level, period_start)
                    DO UPDATE SET
                        count = EXCLUDED.count,
                        min_value = EXCLUDED.min_value,
                        max_value = EXCLUDED.max_value,
                        avg_value = EXCLUDED.avg_value,
                        median_value = EXCLUDED.median_value,
                        stddev_value = EXCLUDED.stddev_value,
                        p95_value = EXCLUDED.p95_value,
                        p99_value = EXCLUDED.p99_value
                    """,
                    aggregated.name,
                    aggregated.aggregation_level,
                    aggregated.period_start,
                    aggregated.period_end,
                    aggregated.count,
                    aggregated.min_value,
                    aggregated.max_value,
                    aggregated.avg_value,
                    aggregated.median_value,
                    aggregated.stddev_value,
                    aggregated.p95_value,
                    aggregated.p99_value,
                    aggregated.tags,
                    aggregated.created_at,
                )
                return True
        except Exception as e:
            logger.error(f"Failed to insert aggregated metric: {e}")
            self._errors += 1
            return False

    async def cleanup_old_metrics(self) -> int:
        """
        Delete metrics older than retention period.

        Returns:
            int: Number of metrics deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

            async with self._get_connection() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM metrics_timeseries
                    WHERE created_at < $1
                    """,
                    cutoff_date
                )

            # Parse result string like "DELETE X"
            count = int(result.split()[-1]) if result else 0
            logger.info(f"Deleted {count} old metrics")
            return count
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
            self._errors += 1
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage layer statistics.

        Returns:
            Dictionary with metrics_inserted, batches_processed, errors
        """
        return {
            "metrics_inserted": self._metrics_inserted,
            "batches_processed": self._batches_processed,
            "errors": self._errors,
        }
