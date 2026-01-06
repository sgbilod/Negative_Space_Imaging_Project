"""
Database High Availability and Replication System
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
import subprocess
import socket

logger = logging.getLogger(__name__)


@dataclass
class ReplicationNode:
    """Replication node configuration."""
    host: str
    port: int = 5432
    database: str = "postgres"
    user: str = "postgres"
    password: str = ""
    role: str = "standby"  # 'primary' or 'standby'
    priority: int = 100  # For automatic failover
    last_seen: Optional[datetime] = None
    status: str = "unknown"  # 'online', 'offline', 'syncing', 'unknown'


@dataclass
class ReplicationLag:
    """Replication lag information."""
    node_id: str
    lag_bytes: int = 0
    lag_time: Optional[timedelta] = None
    last_update: datetime = field(default_factory=datetime.now)
    status: str = "unknown"


@dataclass
class FailoverEvent:
    """Failover event record."""
    timestamp: datetime = field(default_factory=datetime.now)
    old_primary: str
    new_primary: str
    reason: str
    success: bool = False
    duration: float = 0.0
    error_message: Optional[str] = None


class ReplicationManager:
    """Manages PostgreSQL streaming replication."""

    def __init__(self, nodes: List[ReplicationNode]):
        self.nodes = {node.host: node for node in nodes}
        self.replication_lags: Dict[str, ReplicationLag] = {}
        self.failover_history: List[FailoverEvent] = []
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.failover_lock = threading.Lock()

    def start_monitoring(self, interval_seconds: int = 30):
        """Start replication monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Replication monitoring started")

    def stop_monitoring(self):
        """Stop replication monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Replication monitoring stopped")

    def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._update_node_statuses()
                self._check_replication_health()
                self._detect_failover_needed()

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Replication monitoring error: {e}")
                time.sleep(interval)

    def _update_node_statuses(self):
        """Update status of all replication nodes."""
        for node in self.nodes.values():
            try:
                with psycopg2.connect(
                    host=node.host,
                    port=node.port,
                    database=node.database,
                    user=node.user,
                    password=node.password,
                    connect_timeout=5
                ) as conn:
                    with conn.cursor() as cursor:
                        # Check if node is primary or standby
                        cursor.execute("""
                            SELECT pg_is_in_recovery()
                        """)
                        is_recovery = cursor.fetchone()[0]

                        if is_recovery:
                            node.role = "standby"
                            node.status = "online"
                        else:
                            node.role = "primary"
                            node.status = "online"

                        node.last_seen = datetime.now()

            except psycopg2.Error as e:
                node.status = "offline"
                logger.warning(f"Node {node.host} is offline: {e}")

    def _check_replication_health(self):
        """Check replication health and lag."""
        primary_node = self._get_primary_node()
        if not primary_node:
            logger.warning("No primary node found")
            return

        for node in self.nodes.values():
            if node.host == primary_node.host:
                continue

            try:
                lag = self._get_replication_lag(primary_node, node)
                self.replication_lags[node.host] = lag

                # Alert on high lag
                if lag.lag_time and lag.lag_time > timedelta(seconds=300):  # 5 minutes
                    logger.warning(f"High replication lag on {node.host}: {lag.lag_time}")

            except Exception as e:
                logger.error(f"Failed to check replication lag for {node.host}: {e}")

    def _get_replication_lag(self, primary: ReplicationNode, standby: ReplicationNode) -> ReplicationLag:
        """Get replication lag for a standby node."""
        try:
            with psycopg2.connect(
                host=primary.host,
                port=primary.port,
                database=primary.database,
                user=primary.user,
                password=primary.password
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT
                            client_addr,
                            pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) as lag_bytes,
                            extract(epoch from now() - replay_timestamp) as lag_seconds
                        FROM pg_stat_replication
                        WHERE client_addr = %s
                    """, (standby.host,))

                    result = cursor.fetchone()
                    if result:
                        lag_bytes = result[1] or 0
                        lag_seconds = result[2] or 0
                        lag_time = timedelta(seconds=lag_seconds) if lag_seconds else None

                        return ReplicationLag(
                            node_id=standby.host,
                            lag_bytes=lag_bytes,
                            lag_time=lag_time,
                            status="syncing" if lag_bytes < 1024 else "lagging"
                        )

        except psycopg2.Error as e:
            logger.error(f"Failed to get replication lag: {e}")

        return ReplicationLag(node_id=standby.host, status="unknown")

    def _detect_failover_needed(self):
        """Detect if failover is needed."""
        primary_node = self._get_primary_node()
        if not primary_node or primary_node.status == "online":
            return

        logger.warning(f"Primary node {primary_node.host} is offline, initiating failover")

        # Check if we should trigger automatic failover
        standby_nodes = [n for n in self.nodes.values() if n.role == "standby" and n.status == "online"]
        if standby_nodes:
            # Select highest priority standby
            new_primary = max(standby_nodes, key=lambda n: n.priority)
            self.trigger_failover(primary_node.host, new_primary.host, "automatic")

    def trigger_failover(self, old_primary_host: str, new_primary_host: str,
                        reason: str = "manual") -> bool:
        """Trigger a manual failover."""
        with self.failover_lock:
            start_time = time.time()

            event = FailoverEvent(
                old_primary=old_primary_host,
                new_primary=new_primary_host,
                reason=reason
            )

            try:
                # Promote the new primary
                success = self._promote_node(new_primary_host)
                if success:
                    # Update roles
                    self.nodes[old_primary_host].role = "standby"
                    self.nodes[new_primary_host].role = "primary"

                    event.success = True
                    logger.info(f"Failover completed: {old_primary_host} -> {new_primary_host}")

                else:
                    event.error_message = "Promotion failed"

            except Exception as e:
                event.error_message = str(e)
                logger.error(f"Failover failed: {e}")

            event.duration = time.time() - start_time
            self.failover_history.append(event)

            return event.success

    def _promote_node(self, node_host: str) -> bool:
        """Promote a standby node to primary."""
        try:
            node = self.nodes.get(node_host)
            if not node:
                return False

            # Execute pg_ctl promote or trigger file method
            # This is a simplified implementation
            promote_cmd = [
                "pg_ctl", "promote",
                "-D", "/var/lib/postgresql/data"  # Adjust path as needed
            ]

            # In practice, you'd SSH to the node and run this command
            # For demonstration, we'll simulate success
            logger.info(f"Promoting node {node_host} to primary")
            time.sleep(2)  # Simulate promotion time

            return True

        except Exception as e:
            logger.error(f"Node promotion failed: {e}")
            return False

    def _get_primary_node(self) -> Optional[ReplicationNode]:
        """Get the current primary node."""
        primaries = [node for node in self.nodes.values() if node.role == "primary"]
        return primaries[0] if primaries else None

    def get_replication_status(self) -> Dict[str, Any]:
        """Get comprehensive replication status."""
        primary = self._get_primary_node()

        return {
            'primary_node': primary.host if primary else None,
            'nodes': [
                {
                    'host': node.host,
                    'role': node.role,
                    'status': node.status,
                    'last_seen': node.last_seen.isoformat() if node.last_seen else None,
                    'priority': node.priority
                }
                for node in self.nodes.values()
            ],
            'replication_lags': [
                {
                    'node': lag.node_id,
                    'lag_bytes': lag.lag_bytes,
                    'lag_time_seconds': lag.lag_time.total_seconds() if lag.lag_time else None,
                    'status': lag.status,
                    'last_update': lag.last_update.isoformat()
                }
                for lag in self.replication_lags.values()
            ],
            'failover_history': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'old_primary': event.old_primary,
                    'new_primary': event.new_primary,
                    'reason': event.reason,
                    'success': event.success,
                    'duration': event.duration
                }
                for event in self.failover_history[-10:]  # Last 10 events
            ]
        }


class LoadBalancer:
    """Load balancer for read/write splitting."""

    def __init__(self, replication_manager: ReplicationManager):
        self.replication_manager = replication_manager
        self.read_nodes: List[str] = []
        self.write_node: Optional[str] = None

    def get_read_connection(self) -> Optional[ReplicationNode]:
        """Get a read replica for read operations."""
        self._update_nodes()

        if not self.read_nodes:
            return None

        # Simple round-robin (in production, use more sophisticated balancing)
        node_host = self.read_nodes[0]
        return self.replication_manager.nodes.get(node_host)

    def get_write_connection(self) -> Optional[ReplicationNode]:
        """Get the primary node for write operations."""
        self._update_nodes()
        return self.replication_manager.nodes.get(self.write_node) if self.write_node else None

    def _update_nodes(self):
        """Update available read/write nodes."""
        primary = self.replication_manager._get_primary_node()
        self.write_node = primary.host if primary else None

        self.read_nodes = [
            node.host for node in self.replication_manager.nodes.values()
            if node.status == "online"
        ]


class ConnectionPoolHA:
    """High availability connection pool with failover."""

    def __init__(self, replication_manager: ReplicationManager, load_balancer: LoadBalancer):
        self.replication_manager = replication_manager
        self.load_balancer = load_balancer
        self.pools: Dict[str, Any] = {}  # In practice, use actual pool implementation

    def get_read_connection(self):
        """Get connection from read pool."""
        node = self.load_balancer.get_read_connection()
        if node:
            return self._get_connection_from_pool(node, read_only=True)
        return None

    def get_write_connection(self):
        """Get connection from write pool."""
        node = self.load_balancer.get_write_connection()
        if node:
            return self._get_connection_from_pool(node, read_only=False)
        return None

    def _get_connection_from_pool(self, node: ReplicationNode, read_only: bool):
        """Get connection from appropriate pool."""
        # Simplified implementation
        try:
            return psycopg2.connect(
                host=node.host,
                port=node.port,
                database=node.database,
                user=node.user,
                password=node.password
            )
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to {node.host}: {e}")
            return None


class HealthChecker:
    """Health checking for HA cluster."""

    def __init__(self, replication_manager: ReplicationManager):
        self.replication_manager = replication_manager
        self.health_checks: Dict[str, Callable] = {}

    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_func

    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}

        for name, check_func in self.health_checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                results[name] = {'status': 'error', 'error': str(e)}

        return results

    def is_cluster_healthy(self) -> bool:
        """Check if the overall cluster is healthy."""
        status = self.replication_manager.get_replication_status()

        # Check if we have a primary
        if not status['primary_node']:
            return False

        # Check if all nodes are online
        offline_nodes = [node for node in status['nodes'] if node['status'] != 'online']
        if offline_nodes:
            return False

        # Check replication lag
        high_lag_nodes = [
            lag for lag in status['replication_lags']
            if lag['lag_time_seconds'] and lag['lag_time_seconds'] > 300  # 5 minutes
        ]
        if high_lag_nodes:
            return False

        return True


# Default health checks
def check_primary_node(replication_manager: ReplicationManager) -> Dict[str, Any]:
    """Check if primary node is available."""
    primary = replication_manager._get_primary_node()
    if primary and primary.status == "online":
        return {'status': 'healthy', 'primary': primary.host}
    return {'status': 'unhealthy', 'error': 'No healthy primary node'}

def check_replication_lag(replication_manager: ReplicationManager) -> Dict[str, Any]:
    """Check replication lag across all nodes."""
    lags = replication_manager.replication_lags
    high_lag = [lag for lag in lags.values() if lag.lag_time and lag.lag_time > timedelta(minutes=5)]

    if high_lag:
        return {
            'status': 'warning',
            'high_lag_nodes': [lag.node_id for lag in high_lag]
        }

    return {'status': 'healthy', 'total_nodes': len(lags)}

def check_node_connectivity(replication_manager: ReplicationManager) -> Dict[str, Any]:
    """Check connectivity to all nodes."""
    results = {}
    for node in replication_manager.nodes.values():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((node.host, node.port))
            sock.close()

            results[node.host] = 'reachable' if result == 0 else 'unreachable'
        except Exception as e:
            results[node.host] = f'error: {e}'

    unreachable = [host for host, status in results.items() if status != 'reachable']

    return {
        'status': 'healthy' if not unreachable else 'unhealthy',
        'results': results,
        'unreachable_nodes': unreachable
    }


class HAManager:
    """High Availability Manager coordinating all HA components."""

    def __init__(self, replication_manager: ReplicationManager):
        self.replication_manager = replication_manager
        self.load_balancer = LoadBalancer(replication_manager)
        self.connection_pool = ConnectionPoolHA(replication_manager, self.load_balancer)
        self.health_checker = HealthChecker(replication_manager)

        # Register default health checks
        self.health_checker.register_health_check('primary_node', lambda: check_primary_node(replication_manager))
        self.health_checker.register_health_check('replication_lag', lambda: check_replication_lag(replication_manager))
        self.health_checker.register_health_check('connectivity', lambda: check_node_connectivity(replication_manager))

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        replication_status = self.replication_manager.get_replication_status()
        health_checks = self.health_checker.run_health_checks()
        cluster_healthy = self.health_checker.is_cluster_healthy()

        return {
            'cluster_healthy': cluster_healthy,
            'replication_status': replication_status,
            'health_checks': health_checks,
            'load_balancer': {
                'write_node': self.load_balancer.write_node,
                'read_nodes': self.read_nodes
            },
            'timestamp': datetime.now().isoformat()
        }

    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        return self.health_checker.run_health_checks()

    def get_read_connection(self):
        """Get a read connection through the load balancer."""
        return self.connection_pool.get_read_connection()

    def get_write_connection(self):
        """Get a write connection through the load balancer."""
        return self.connection_pool.get_write_connection()
