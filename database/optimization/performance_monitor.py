"""
Database Performance Monitoring and Alerting System
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    metric: str
    condition: str  # '>', '<', '>=', '<=', '==', '!='
    threshold: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    enabled: bool = True
    cooldown_minutes: int = 5


@dataclass
class Alert:
    """Alert instance."""
    rule_name: str
    severity: str
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class DatabaseMetrics:
    """Comprehensive database performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)

    # Connection metrics
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    waiting_clients: int = 0

    # Query performance
    slow_queries: int = 0
    avg_query_time: float = 0.0
    total_queries: int = 0

    # Cache metrics
    cache_hit_ratio: float = 0.0
    cache_size: int = 0

    # Storage metrics
    db_size_bytes: int = 0
    table_sizes: Dict[str, int] = field(default_factory=dict)
    index_sizes: Dict[str, int] = field(default_factory=dict)

    # Lock metrics
    active_locks: int = 0
    waiting_locks: int = 0

    # Replication metrics (if applicable)
    replication_lag: Optional[float] = None
    replication_status: str = "unknown"


class AlertManager:
    """Manages database alerts and notifications."""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.rules: Dict[str, AlertRule] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}

    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def evaluate_metric(self, metric_name: str, value: float):
        """Evaluate a metric against all rules."""
        for rule in self.rules.values():
            if not rule.enabled or rule.metric != metric_name:
                continue

            # Check cooldown
            if rule.name in self.alert_cooldowns:
                if datetime.now() < self.alert_cooldowns[rule.name]:
                    continue

            # Evaluate condition
            triggered = self._evaluate_condition(value, rule.condition, rule.threshold)

            if triggered:
                self._trigger_alert(rule, value)
                self.alert_cooldowns[rule.name] = datetime.now() + timedelta(minutes=rule.cooldown_minutes)

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == '>':
            return value > threshold
        elif condition == '<':
            return value < threshold
        elif condition == '>=':
            return value >= threshold
        elif condition == '<=':
            return value <= threshold
        elif condition == '==':
            return abs(value - threshold) < 0.001
        elif condition == '!=':
            return abs(value - threshold) >= 0.001
        else:
            logger.error(f"Unknown condition: {condition}")
            return False

    def _trigger_alert(self, rule: AlertRule, value: float):
        """Trigger an alert."""
        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            message=f"{rule.description} (Value: {value:.2f}, Threshold: {rule.threshold:.2f})",
            value=value,
            threshold=rule.threshold
        )

        self.alerts.append(alert)
        self.active_alerts[rule.name] = alert

        logger.warning(f"ALERT [{rule.severity.upper()}]: {alert.message}")

        # Here you would integrate with notification systems
        # self._send_notification(alert)

    def resolve_alert(self, rule_name: str):
        """Resolve an active alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved = True
            alert.resolved_at = datetime.now()

            logger.info(f"RESOLVED: {alert.message}")
            del self.active_alerts[rule_name]

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff]


class DatabaseMonitor:
    """Database performance monitoring system."""

    def __init__(self, connection_string: str, alert_manager: AlertManager):
        self.connection_string = connection_string
        self.alert_manager = alert_manager
        self.metrics_history: List[DatabaseMetrics] = []
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self, interval_seconds: int = 60):
        """Start the monitoring loop."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Database monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Database monitoring stopped")

    def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)

                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history.pop(0)

                # Evaluate alerts
                self._evaluate_alerts(metrics)

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)

    def _collect_metrics(self) -> DatabaseMetrics:
        """Collect comprehensive database metrics."""
        metrics = DatabaseMetrics()

        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    # Connection metrics
                    cursor.execute("""
                        SELECT count(*) as active_connections
                        FROM pg_stat_activity
                        WHERE state = 'active'
                    """)
                    metrics.active_connections = cursor.fetchone()[0]

                    cursor.execute("""
                        SELECT count(*) as idle_connections
                        FROM pg_stat_activity
                        WHERE state = 'idle'
                    """)
                    metrics.idle_connections = cursor.fetchone()[0]

                    cursor.execute("""
                        SELECT count(*) as total_connections
                        FROM pg_stat_activity
                    """)
                    metrics.total_connections = cursor.fetchone()[0]

                    # Query performance (simplified)
                    cursor.execute("""
                        SELECT count(*) as slow_queries
                        FROM pg_stat_activity
                        WHERE state = 'active'
                        AND now() - query_start > interval '1 second'
                    """)
                    metrics.slow_queries = cursor.fetchone()[0]

                    # Database size
                    cursor.execute("""
                        SELECT pg_database_size(current_database())
                    """)
                    metrics.db_size_bytes = cursor.fetchone()[0]

                    # Table sizes
                    cursor.execute("""
                        SELECT schemaname || '.' || tablename as table_name,
                               pg_total_relation_size(schemaname || '.' || tablename) as size_bytes
                        FROM pg_tables
                        WHERE schemaname = 'public'
                        ORDER BY size_bytes DESC
                        LIMIT 10
                    """)
                    metrics.table_sizes = {row[0]: row[1] for row in cursor.fetchall()}

                    # Lock metrics
                    cursor.execute("""
                        SELECT count(*) as active_locks
                        FROM pg_locks
                        WHERE granted = true
                    """)
                    metrics.active_locks = cursor.fetchone()[0]

                    cursor.execute("""
                        SELECT count(*) as waiting_locks
                        FROM pg_locks
                        WHERE granted = false
                    """)
                    metrics.waiting_locks = cursor.fetchone()[0]

        except psycopg2.Error as e:
            logger.error(f"Failed to collect database metrics: {e}")

        return metrics

    def _evaluate_alerts(self, metrics: DatabaseMetrics):
        """Evaluate metrics against alert rules."""
        self.alert_manager.evaluate_metric('active_connections', metrics.active_connections)
        self.alert_manager.evaluate_metric('slow_queries', metrics.slow_queries)
        self.alert_manager.evaluate_metric('active_locks', metrics.active_locks)
        self.alert_manager.evaluate_metric('waiting_locks', metrics.waiting_locks)

        # Calculate and alert on connection utilization
        if metrics.total_connections > 0:
            connection_utilization = (metrics.active_connections / metrics.total_connections) * 100
            self.alert_manager.evaluate_metric('connection_utilization', connection_utilization)

    def get_current_metrics(self) -> Optional[DatabaseMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_history(self, hours: int = 1) -> List[DatabaseMetrics]:
        """Get metrics history for the specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff]

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        if not self.metrics_history:
            return {}

        recent_metrics = self.get_metrics_history(hours=1)

        if not recent_metrics:
            return {}

        # Calculate averages
        avg_active_connections = sum(m.active_connections for m in recent_metrics) / len(recent_metrics)
        avg_slow_queries = sum(m.slow_queries for m in recent_metrics) / len(recent_metrics)
        max_active_connections = max(m.active_connections for m in recent_metrics)

        # Get current alerts
        active_alerts = self.alert_manager.get_active_alerts()

        return {
            'summary': {
                'monitoring_period_hours': 1,
                'total_measurements': len(recent_metrics),
                'avg_active_connections': avg_active_connections,
                'avg_slow_queries': avg_slow_queries,
                'max_active_connections': max_active_connections,
                'active_alerts_count': len(active_alerts)
            },
            'current_metrics': self.get_current_metrics().__dict__ if self.get_current_metrics() else {},
            'active_alerts': [alert.__dict__ for alert in active_alerts],
            'recommendations': self._generate_recommendations(recent_metrics)
        }

    def _generate_recommendations(self, metrics: List[DatabaseMetrics]) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []

        if metrics:
            avg_active = sum(m.active_connections for m in metrics) / len(metrics)
            max_active = max(m.active_connections for m in metrics)

            if max_active > 80:  # Assuming max connections = 100
                recommendations.append("Consider increasing max_connections in postgresql.conf")

            if avg_active > 50:
                recommendations.append("High connection utilization - consider connection pooling")

            slow_queries_avg = sum(m.slow_queries for m in metrics) / len(metrics)
            if slow_queries_avg > 5:
                recommendations.append("High number of slow queries - review query performance and add indexes")

            locks_avg = sum(m.active_locks for m in metrics) / len(metrics)
            if locks_avg > 20:
                recommendations.append("High lock contention - review transaction isolation and locking strategy")

        return recommendations


class EmailNotifier:
    """Email notification system for alerts."""

    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    def send_alert(self, alert: Alert, recipients: List[str]):
        """Send alert notification via email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.upper()}] Database Alert: {alert.rule_name}"

            body = f"""
Database Alert Details:

Severity: {alert.severity.upper()}
Rule: {alert.rule_name}
Message: {alert.message}
Value: {alert.value}
Threshold: {alert.threshold}
Time: {alert.timestamp}

This is an automated message from the Database Monitoring System.
"""
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Alert email sent to {len(recipients)} recipients")

        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")


# Default alert rules
DEFAULT_ALERT_RULES = [
    AlertRule(
        name="high_connection_utilization",
        metric="connection_utilization",
        condition=">",
        threshold=80.0,
        severity="high",
        description="Connection utilization is above 80%"
    ),
    AlertRule(
        name="excessive_slow_queries",
        metric="slow_queries",
        condition=">",
        threshold=10,
        severity="medium",
        description="More than 10 slow queries detected"
    ),
    AlertRule(
        name="high_lock_contention",
        metric="active_locks",
        condition=">",
        threshold=50,
        severity="high",
        description="High number of active locks indicating contention"
    ),
    AlertRule(
        name="connection_pool_exhausted",
        metric="active_connections",
        condition=">",
        threshold=90,
        severity="critical",
        description="Connection pool nearly exhausted"
    )
]
