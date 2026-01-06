"""
Database Optimization Integration Module
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import sql

# Import optimization components
from .query_optimizer import QueryOptimizer
from .advanced_pool import AdvancedConnectionPool
from .caching_layer import CachingLayer
from .migration_system import MigrationManager
from .performance_monitor import PerformanceMonitor
from .backup_recovery import BackupManager
from .high_availability import HAManager, ReplicationManager, ReplicationNode
from .security_hardening import SecurityManager

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for database optimization components."""
    enable_query_optimization: bool = True
    enable_advanced_pooling: bool = True
    enable_caching: bool = True
    enable_migrations: bool = True
    enable_monitoring: bool = True
    enable_backup_recovery: bool = True
    enable_high_availability: bool = False  # Requires replication setup
    enable_security_hardening: bool = True

    # Component-specific configs
    pool_config: Dict[str, Any] = field(default_factory=dict)
    cache_config: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    backup_config: Dict[str, Any] = field(default_factory=dict)
    ha_config: Dict[str, Any] = field(default_factory=dict)


class DatabaseOptimizer:
    """Integrated database optimization system."""

    def __init__(self, db_config: Dict[str, Any], optimization_config: OptimizationConfig):
        self.db_config = db_config
        self.optimization_config = optimization_config

        # Initialize components
        self.components: Dict[str, Any] = {}
        self._initialize_components()

        # Integration state
        self.optimization_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.health_check_interval = 60  # seconds

    def _initialize_components(self):
        """Initialize optimization components based on configuration."""
        logger.info("Initializing database optimization components...")

        # Query Optimizer
        if self.optimization_config.enable_query_optimization:
            self.components['query_optimizer'] = QueryOptimizer(self.db_config)
            logger.info("Query optimizer initialized")

        # Advanced Connection Pool
        if self.optimization_config.enable_advanced_pooling:
            pool_config = self.optimization_config.pool_config
            self.components['connection_pool'] = AdvancedConnectionPool(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('database', 'postgres'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', ''),
                min_connections=pool_config.get('min_connections', 5),
                max_connections=pool_config.get('max_connections', 20),
                health_check_interval=pool_config.get('health_check_interval', 30)
            )
            logger.info("Advanced connection pool initialized")

        # Caching Layer
        if self.optimization_config.enable_caching:
            cache_config = self.optimization_config.cache_config
            self.components['caching_layer'] = CachingLayer(
                redis_host=cache_config.get('redis_host', 'localhost'),
                redis_port=cache_config.get('redis_port', 6379),
                default_ttl=cache_config.get('default_ttl', 3600),
                max_memory=cache_config.get('max_memory', '512mb')
            )
            logger.info("Caching layer initialized")

        # Migration Manager
        if self.optimization_config.enable_migrations:
            self.components['migration_manager'] = MigrationManager(
                db_config=self.db_config,
                migrations_path="./database/migrations"
            )
            logger.info("Migration manager initialized")

        # Performance Monitor
        if self.optimization_config.enable_monitoring:
            monitoring_config = self.optimization_config.monitoring_config
            self.components['performance_monitor'] = PerformanceMonitor(
                db_config=self.db_config,
                alert_thresholds=monitoring_config.get('alert_thresholds', {}),
                metrics_retention_days=monitoring_config.get('metrics_retention_days', 30)
            )
            logger.info("Performance monitor initialized")

        # Backup Manager
        if self.optimization_config.enable_backup_recovery:
            backup_config = self.optimization_config.backup_config
            self.components['backup_manager'] = BackupManager(
                db_config=self.db_config,
                backup_path=backup_config.get('backup_path', './backups'),
                retention_days=backup_config.get('retention_days', 30),
                compression_enabled=backup_config.get('compression_enabled', True)
            )
            logger.info("Backup manager initialized")

        # High Availability Manager
        if self.optimization_config.enable_high_availability:
            ha_config = self.optimization_config.ha_config
            replication_nodes = [
                ReplicationNode(
                    host=node['host'],
                    port=node.get('port', 5432),
                    database=self.db_config.get('database', 'postgres'),
                    user=self.db_config.get('user', 'postgres'),
                    password=self.db_config.get('password', ''),
                    role=node.get('role', 'standby'),
                    priority=node.get('priority', 100)
                )
                for node in ha_config.get('nodes', [])
            ]

            replication_manager = ReplicationManager(replication_nodes)
            self.components['ha_manager'] = HAManager(replication_manager)
            logger.info("High availability manager initialized")

        # Security Manager
        if self.optimization_config.enable_security_hardening:
            self.components['security_manager'] = SecurityManager()
            logger.info("Security manager initialized")

    def start_optimization(self):
        """Start all optimization components."""
        if self.optimization_active:
            logger.warning("Optimization already active")
            return

        logger.info("Starting database optimization...")

        # Start individual components
        for name, component in self.components.items():
            try:
                if hasattr(component, 'start'):
                    component.start()
                    logger.info(f"Started {name}")
                elif hasattr(component, 'connect'):
                    component.connect()
                    logger.info(f"Connected {name}")
            except Exception as e:
                logger.error(f"Failed to start {name}: {e}")

        # Start monitoring thread
        self.optimization_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()

        logger.info("Database optimization started successfully")

    def stop_optimization(self):
        """Stop all optimization components."""
        if not self.optimization_active:
            logger.warning("Optimization not active")
            return

        logger.info("Stopping database optimization...")

        self.optimization_active = False

        # Stop monitoring thread
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)

        # Stop individual components
        for name, component in self.components.items():
            try:
                if hasattr(component, 'stop'):
                    component.stop()
                    logger.info(f"Stopped {name}")
                elif hasattr(component, 'disconnect'):
                    component.disconnect()
                    logger.info(f"Disconnected {name}")
            except Exception as e:
                logger.error(f"Failed to stop {name}: {e}")

        logger.info("Database optimization stopped")

    def _monitoring_loop(self):
        """Main monitoring and health check loop."""
        while self.optimization_active:
            try:
                self._perform_health_checks()
                self._optimize_performance()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.health_check_interval)

    def _perform_health_checks(self):
        """Perform health checks on all components."""
        health_status = {}

        for name, component in self.components.items():
            try:
                if hasattr(component, 'health_check'):
                    health_status[name] = component.health_check()
                elif hasattr(component, 'is_healthy'):
                    health_status[name] = component.is_healthy()
                else:
                    health_status[name] = {'status': 'unknown'}
            except Exception as e:
                health_status[name] = {'status': 'error', 'error': str(e)}
                logger.error(f"Health check failed for {name}: {e}")

        # Log overall health
        unhealthy_components = [
            name for name, status in health_status.items()
            if status.get('status') not in ['healthy', 'ok']
        ]

        if unhealthy_components:
            logger.warning(f"Unhealthy components: {unhealthy_components}")
        else:
            logger.debug("All components healthy")

    def _optimize_performance(self):
        """Perform ongoing performance optimizations."""
        try:
            # Query optimization
            if 'query_optimizer' in self.components:
                optimizer = self.components['query_optimizer']
                optimizer.optimize_slow_queries()

            # Cache maintenance
            if 'caching_layer' in self.components:
                cache = self.components['caching_layer']
                cache.cleanup_expired_keys()

            # Pool optimization
            if 'connection_pool' in self.components:
                pool = self.components['connection_pool']
                pool.optimize_pool_size()

            # Performance monitoring
            if 'performance_monitor' in self.components:
                monitor = self.components['performance_monitor']
                monitor.collect_metrics()
                monitor.check_alerts()

        except Exception as e:
            logger.error(f"Performance optimization error: {e}")

    def execute_optimized_query(self, query: str, params: Optional[Dict[str, Any]] = None,
                               use_cache: bool = True) -> List[Dict[str, Any]]:
        """Execute a query with all optimizations applied."""
        start_time = time.time()

        try:
            # Security check
            if 'security_manager' in self.components:
                security = self.components['security_manager']
                user = self.db_config.get('user', 'unknown')
                database = self.db_config.get('database', 'unknown')

                is_safe, violations = security.validate_query(query, user, database)
                if not is_safe:
                    raise ValueError(f"Query security violation: {violations}")

            # Check cache first
            cache_key = None
            if use_cache and 'caching_layer' in self.components:
                cache = self.components['caching_layer']
                cache_key = cache.generate_cache_key(query, params or {})
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    logger.debug("Query served from cache")
                    return cached_result

            # Get connection from pool
            if 'connection_pool' in self.components:
                pool = self.components['connection_pool']
                connection = pool.get_connection()
            else:
                # Fallback to direct connection
                connection = psycopg2.connect(**self.db_config)

            try:
                with connection.cursor() as cursor:
                    # Analyze query if optimizer available
                    if 'query_optimizer' in self.components:
                        optimizer = self.components['query_optimizer']
                        analysis = optimizer.analyze_query(query)
                        if analysis['needs_optimization']:
                            logger.info(f"Query optimization recommended: {analysis['recommendations']}")

                    # Execute query
                    cursor.execute(query, params or ())

                    if cursor.description:
                        columns = [desc[0] for desc in cursor.description]
                        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    else:
                        results = []

                    # Cache result
                    if cache_key and use_cache and 'caching_layer' in self.components:
                        cache.set(cache_key, results)

                    return results

            finally:
                if 'connection_pool' in self.components:
                    pool.return_connection(connection)
                else:
                    connection.close()

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed after {execution_time:.2f}s: {e}")

            # Record performance metrics
            if 'performance_monitor' in self.components:
                monitor = self.components['performance_monitor']
                monitor.record_query_metrics(query, execution_time, success=False, error=str(e))

            raise

        finally:
            execution_time = time.time() - start_time

            # Record successful query metrics
            if 'performance_monitor' in self.components:
                monitor = self.components['performance_monitor']
                monitor.record_query_metrics(query, execution_time, success=True)

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        status = {
            'optimization_active': self.optimization_active,
            'components': {},
            'overall_health': 'healthy',
            'timestamp': datetime.now().isoformat()
        }

        unhealthy_count = 0

        for name, component in self.components.items():
            try:
                if hasattr(component, 'get_status'):
                    component_status = component.get_status()
                elif hasattr(component, 'health_check'):
                    component_status = component.health_check()
                else:
                    component_status = {'status': 'unknown'}

                status['components'][name] = component_status

                if component_status.get('status') not in ['healthy', 'ok']:
                    unhealthy_count += 1

            except Exception as e:
                status['components'][name] = {'status': 'error', 'error': str(e)}
                unhealthy_count += 1

        # Determine overall health
        if unhealthy_count > 0:
            status['overall_health'] = 'degraded' if unhealthy_count < len(self.components) else 'unhealthy'

        return status

    def run_maintenance_tasks(self):
        """Run maintenance tasks across all components."""
        logger.info("Running maintenance tasks...")

        tasks_completed = 0
        tasks_failed = 0

        # Migration check
        if 'migration_manager' in self.components:
            try:
                migration_mgr = self.components['migration_manager']
                pending = migration_mgr.get_pending_migrations()
                if pending:
                    logger.info(f"Pending migrations: {len(pending)}")
                    # Optionally auto-apply migrations
                    # migration_mgr.apply_pending_migrations()
                tasks_completed += 1
            except Exception as e:
                logger.error(f"Migration maintenance failed: {e}")
                tasks_failed += 1

        # Backup verification
        if 'backup_manager' in self.components:
            try:
                backup_mgr = self.components['backup_manager']
                last_backup = backup_mgr.get_last_backup_info()
                if last_backup:
                    age = datetime.now() - last_backup['timestamp']
                    if age > timedelta(days=1):
                        logger.warning(f"Last backup is {age.days} days old")
                tasks_completed += 1
            except Exception as e:
                logger.error(f"Backup maintenance failed: {e}")
                tasks_failed += 1

        # Cache cleanup
        if 'caching_layer' in self.components:
            try:
                cache = self.components['caching_layer']
                cleaned = cache.cleanup_expired_keys()
                logger.info(f"Cleaned {cleaned} expired cache keys")
                tasks_completed += 1
            except Exception as e:
                logger.error(f"Cache maintenance failed: {e}")
                tasks_failed += 1

        # Index optimization
        if 'query_optimizer' in self.components:
            try:
                optimizer = self.components['query_optimizer']
                recommendations = optimizer.get_index_recommendations()
                if recommendations:
                    logger.info(f"Index optimization recommendations: {len(recommendations)}")
                tasks_completed += 1
            except Exception as e:
                logger.error(f"Index maintenance failed: {e}")
                tasks_failed += 1

        # Security audit
        if 'security_manager' in self.components:
            try:
                security = self.components['security_manager']
                report = security.generate_security_report(days=1)
                critical_events = report['audit_report']['critical_events']
                if critical_events:
                    logger.warning(f"Critical security events in last 24h: {len(critical_events)}")
                tasks_completed += 1
            except Exception as e:
                logger.error(f"Security maintenance failed: {e}")
                tasks_failed += 1

        logger.info(f"Maintenance tasks completed: {tasks_completed}, failed: {tasks_failed}")

        return {
            'tasks_completed': tasks_completed,
            'tasks_failed': tasks_failed,
            'timestamp': datetime.now().isoformat()
        }

    def create_backup(self, backup_type: str = 'full') -> Dict[str, Any]:
        """Create a database backup."""
        if 'backup_manager' not in self.components:
            raise ValueError("Backup manager not enabled")

        backup_mgr = self.components['backup_manager']
        return backup_mgr.create_backup(backup_type)

    def restore_from_backup(self, backup_id: str) -> bool:
        """Restore database from backup."""
        if 'backup_manager' not in self.components:
            raise ValueError("Backup manager not enabled")

        backup_mgr = self.components['backup_manager']
        return backup_mgr.restore_backup(backup_id)

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance report for the specified time period."""
        if 'performance_monitor' not in self.components:
            raise ValueError("Performance monitor not enabled")

        monitor = self.components['performance_monitor']
        return monitor.generate_report(hours=hours)

    def get_security_report(self, days: int = 7) -> Dict[str, Any]:
        """Get security report for the specified time period."""
        if 'security_manager' not in self.components:
            raise ValueError("Security manager not enabled")

        security = self.components['security_manager']
        return security.generate_security_report(days=days)
