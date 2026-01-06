"""
Database Query Optimization and Indexing System
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

from sqlalchemy import create_engine, text, Index, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for query performance analysis."""
    query: str
    execution_time: float
    rows_affected: int
    timestamp: float
    slow_query: bool = False


@dataclass
class IndexRecommendation:
    """Index optimization recommendation."""
    table: str
    columns: List[str]
    index_type: str  # 'btree', 'hash', 'gin', 'gist'
    estimated_improvement: float
    current_selectivity: float


class QueryOptimizer:
    """Database query optimization and performance monitoring."""

    def __init__(self, engine):
        self.engine = engine
        self.metrics: List[QueryMetrics] = []
        self.slow_query_threshold = 1.0  # seconds

    @contextmanager
    def monitored_query(self, query_name: str = None):
        """Context manager for monitoring query performance."""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            if query_name:
                metrics = QueryMetrics(
                    query=query_name,
                    execution_time=execution_time,
                    rows_affected=0,  # Would need to be set by caller
                    timestamp=start_time,
                    slow_query=execution_time > self.slow_query_threshold
                )
                self.metrics.append(metrics)

                if metrics.slow_query:
                    logger.warning(f"Slow query detected: {query_name} ({execution_time:.2f}s)")

    def analyze_query_performance(self, query: str, params: Dict = None) -> Dict[str, Any]:
        """Analyze query performance with EXPLAIN."""
        try:
            with self.engine.connect() as conn:
                # Get execution plan
                explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
                result = conn.execute(text(explain_query), params or {})

                plan = result.fetchone()[0] if result.rowcount > 0 else None

                return {
                    'execution_plan': plan,
                    'query_cost': self._extract_cost_from_plan(plan),
                    'estimated_rows': self._extract_rows_from_plan(plan),
                    'actual_time': self._extract_actual_time_from_plan(plan)
                }

        except SQLAlchemyError as e:
            logger.error(f"Query analysis failed: {e}")
            return {}

    def _extract_cost_from_plan(self, plan) -> Optional[float]:
        """Extract cost from execution plan."""
        if not plan or not isinstance(plan, list):
            return None
        try:
            return plan[0].get('Plan', {}).get('Total Cost')
        except (KeyError, IndexError):
            return None

    def _extract_rows_from_plan(self, plan) -> Optional[int]:
        """Extract estimated rows from execution plan."""
        if not plan or not isinstance(plan, list):
            return None
        try:
            return plan[0].get('Plan', {}).get('Plan Rows')
        except (KeyError, IndexError):
            return None

    def _extract_actual_time_from_plan(self, plan) -> Optional[float]:
        """Extract actual execution time from plan."""
        if not plan or not isinstance(plan, list):
            return None
        try:
            return plan[0].get('Plan', {}).get('Actual Total Time')
        except (KeyError, IndexError):
            return None

    def recommend_indexes(self) -> List[IndexRecommendation]:
        """Analyze database and recommend index optimizations."""
        recommendations = []

        try:
            with self.engine.connect() as conn:
                # Get table statistics
                tables_result = conn.execute(text("""
                    SELECT schemaname, tablename
                    FROM pg_tables
                    WHERE schemaname = 'public'
                """))

                for row in tables_result:
                    table_name = row[1]
                    table_recs = self._analyze_table_indexes(table_name, conn)
                    recommendations.extend(table_recs)

        except SQLAlchemyError as e:
            logger.error(f"Index analysis failed: {e}")

        return recommendations

    def _analyze_table_indexes(self, table_name: str, conn) -> List[IndexRecommendation]:
        """Analyze indexes for a specific table."""
        recommendations = []

        try:
            # Get current indexes
            indexes_result = conn.execute(text("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = :table
            """), {'table': table_name})

            current_indexes = {row[0]: row[1] for row in indexes_result}

            # Analyze query patterns (simplified - would need query log analysis)
            # This is a basic implementation - production would need more sophisticated analysis

            # Check for missing indexes on foreign keys
            fk_result = conn.execute(text("""
                SELECT conname, conkey, confrelid
                FROM pg_constraint
                WHERE contype = 'f' AND conrelid = (
                    SELECT oid FROM pg_class WHERE relname = :table
                )
            """), {'table': table_name})

            for fk_row in fk_result:
                constraint_name = fk_row[0]
                # Check if foreign key has index
                if not any(f"({constraint_name})" in idx_def for idx_def in current_indexes.values()):
                    recommendations.append(IndexRecommendation(
                        table=table_name,
                        columns=[constraint_name],  # Simplified
                        index_type='btree',
                        estimated_improvement=0.5,  # Placeholder
                        current_selectivity=0.1     # Placeholder
                    ))

        except SQLAlchemyError as e:
            logger.error(f"Table analysis failed for {table_name}: {e}")

        return recommendations

    def create_optimized_indexes(self, recommendations: List[IndexRecommendation]):
        """Create recommended indexes."""
        with self.engine.connect() as conn:
            for rec in recommendations:
                try:
                    index_name = f"idx_{rec.table}_{'_'.join(rec.columns)}"
                    columns_str = ', '.join(rec.columns)

                    create_stmt = f"CREATE INDEX {index_name} ON {rec.table} ({columns_str})"

                    conn.execute(text(create_stmt))
                    conn.commit()

                    logger.info(f"Created index {index_name} on {rec.table}")

                except SQLAlchemyError as e:
                    logger.error(f"Failed to create index on {rec.table}: {e}")
                    conn.rollback()

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        if not self.metrics:
            return {}

        total_queries = len(self.metrics)
        slow_queries = len([m for m in self.metrics if m.slow_query])
        avg_execution_time = sum(m.execution_time for m in self.metrics) / total_queries
        max_execution_time = max(m.execution_time for m in self.metrics)

        return {
            'total_queries': total_queries,
            'slow_queries': slow_queries,
            'slow_query_percentage': (slow_queries / total_queries) * 100,
            'avg_execution_time': avg_execution_time,
            'max_execution_time': max_execution_time,
            'recent_queries': self.metrics[-10:]  # Last 10 queries
        }


class IndexManager:
    """Database index management and optimization."""

    def __init__(self, engine):
        self.engine = engine

    def create_performance_indexes(self):
        """Create indexes optimized for common query patterns."""
        indexes = [
            # User indexes
            Index('idx_users_email', 'users.email'),
            Index('idx_users_role', 'users.role'),
            Index('idx_users_active', 'users.is_active'),
            Index('idx_users_created', 'users.created_at'),

            # Image indexes
            Index('idx_images_owner', 'images.owner_id'),
            Index('idx_images_hash', 'images.file_hash'),
            Index('idx_images_processed', 'images.processed'),
            Index('idx_images_created', 'images.created_at'),

            # Processing job indexes
            Index('idx_jobs_image', 'processing_jobs.image_id'),
            Index('idx_jobs_status', 'processing_jobs.status'),
            Index('idx_jobs_started', 'processing_jobs.started_at'),

            # Processing result indexes
            Index('idx_results_job', 'processing_results.job_id'),
            Index('idx_results_type', 'processing_results.result_type'),

            # Signature indexes
            Index('idx_signatures_image', 'signatures.image_id'),
            Index('idx_signatures_signer', 'signatures.signer_id'),
            Index('idx_signatures_valid', 'signatures.is_valid'),

            # Audit log indexes
            Index('idx_audit_user', 'audit_logs.user_id'),
            Index('idx_audit_action', 'audit_logs.action'),
            Index('idx_audit_timestamp', 'audit_logs.timestamp'),

            # Composite indexes for common queries
            Index('idx_users_role_active', 'users.role', 'users.is_active'),
            Index('idx_images_owner_created', 'images.owner_id', 'images.created_at'),
            Index('idx_jobs_status_started', 'processing_jobs.status', 'processing_jobs.started_at'),
        ]

        try:
            with self.engine.connect() as conn:
                for index in indexes:
                    try:
                        index.create(conn)
                        logger.info(f"Created index: {index.name}")
                    except SQLAlchemyError as e:
                        logger.warning(f"Index {index.name} may already exist: {e}")

                conn.commit()
                logger.info("Performance indexes created successfully")

        except SQLAlchemyError as e:
            logger.error(f"Failed to create performance indexes: {e}")

    def analyze_index_usage(self) -> Dict[str, Any]:
        """Analyze index usage statistics."""
        try:
            with self.engine.connect() as conn:
                # Get index usage statistics
                result = conn.execute(text("""
                    SELECT
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes
                    ORDER BY idx_scan DESC
                """))

                usage_stats = []
                for row in result:
                    usage_stats.append({
                        'schema': row[0],
                        'table': row[1],
                        'index': row[2],
                        'scans': row[3],
                        'tuples_read': row[4],
                        'tuples_fetched': row[5]
                    })

                return {
                    'index_usage': usage_stats,
                    'unused_indexes': [stat for stat in usage_stats if stat['scans'] == 0]
                }

        except SQLAlchemyError as e:
            logger.error(f"Index usage analysis failed: {e}")
            return {}

    def cleanup_unused_indexes(self, dry_run: bool = True):
        """Remove unused indexes."""
        usage_stats = self.analyze_index_usage()
        unused = usage_stats.get('unused_indexes', [])

        if dry_run:
            logger.info(f"Dry run: Would remove {len(unused)} unused indexes")
            for idx in unused:
                logger.info(f"Would drop: {idx['index']} on {idx['table']}")
            return

        try:
            with self.engine.connect() as conn:
                for idx in unused:
                    try:
                        drop_stmt = f"DROP INDEX {idx['index']}"
                        conn.execute(text(drop_stmt))
                        logger.info(f"Dropped unused index: {idx['index']}")
                    except SQLAlchemyError as e:
                        logger.error(f"Failed to drop index {idx['index']}: {e}")

                conn.commit()
                logger.info(f"Cleaned up {len(unused)} unused indexes")

        except SQLAlchemyError as e:
            logger.error(f"Index cleanup failed: {e}")
