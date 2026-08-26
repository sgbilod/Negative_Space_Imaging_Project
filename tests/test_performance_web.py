"""
Performance Web Interface Test Suite
© 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

Test suite for the Sovereign Performance Optimization Web Interface.
"""

import os
import json
import unittest
from unittest import mock
from datetime import datetime, timedelta

import pytest
from flask import url_for, session

from sovereign.app import create_app
from sovereign.performance import (
    PerformanceManager,
    OptimizationLevel,
    OptimizationTarget
)
from sovereign.security import SecurityManager, SecurityLevel
from sovereign.web_interface import init_app, security_manager
from sovereign.performance_web import performance_bp, init_performance_web

class PerformanceWebTestCase(unittest.TestCase):
    """Test case for performance web interface"""

    def setUp(self):
        """Set up test environment"""
        # Create test app
        self.app = create_app(testing=True)
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.app.config['SERVER_NAME'] = 'localhost'

        # Register blueprint
        init_performance_web(self.app)

        # Create test client
        self.client = self.app.test_client()

        # Mock security manager
        self.security_manager_patcher = mock.patch('sovereign.web_interface.security_manager')
        self.mock_security_manager = self.security_manager_patcher.start()

        # Mock performance manager
        self.performance_manager_patcher = mock.patch('sovereign.performance_web.performance_manager')
        self.mock_performance_manager = self.performance_manager_patcher.start()

        # Create application context
        self.app_context = self.app.app_context()
        self.app_context.push()

        # Set up mock security validation
        self.mock_security_manager.validate_session_token.return_value = (True, 'admin')
        self.mock_security_manager.check_authorization.return_value = True

        # Set up mock performance data
        self._setup_mock_performance_data()

    def tearDown(self):
        """Tear down test environment"""
        # Remove app context
        self.app_context.pop()

        # Stop patches
        self.security_manager_patcher.stop()
        self.performance_manager_patcher.stop()

    def _setup_mock_performance_data(self):
        """Set up mock performance data for tests"""
        # Mock metrics
        mock_metrics = mock.MagicMock()
        mock_metrics.to_dict.return_value = {
            'cpu_usage': 45.5,
            'memory_usage': 512.0,
            'thread_count': 8,
            'process_count': 4,
            'cache_hits': 1000,
            'cache_misses': 200
        }
        self.mock_performance_manager.current_metrics = mock_metrics
        self.mock_performance_manager.collect_metrics.return_value = mock_metrics

        # Mock memory history
        timestamp = datetime.now()
        memory_history = []
        for i in range(20):
            entry_time = timestamp - timedelta(seconds=i*30)
            memory_history.append({
                'timestamp': entry_time.timestamp(),
                'rss': 500.0 - i * 2,
                'vms': 1000.0 - i * 4
            })
        self.mock_performance_manager.memory_optimizer.get_memory_usage_history.return_value = memory_history

        # Mock cache stats
        cache_stats = {
            'hits': 1000,
            'misses': 200,
            'hit_ratio': 0.83,
            'size': 500,
            'max_size': 1000,
            'types': {
                'query': {'hits': 500, 'misses': 100},
                'image': {'hits': 300, 'misses': 50},
                'user': {'hits': 200, 'misses': 50}
            }
        }
        self.mock_performance_manager.cache.get_stats.return_value = cache_stats

        # Mock database query stats
        query_stats = {
            'query1': {
                'query_text': 'SELECT * FROM images',
                'execution_count': 100,
                'total_time': 5.0,
                'avg_time': 0.05,
                'min_time': 0.01,
                'max_time': 0.2,
                'last_executed': datetime.now()
            },
            'query2': {
                'query_text': 'SELECT * FROM users',
                'execution_count': 50,
                'total_time': 10.0,
                'avg_time': 0.2,
                'min_time': 0.05,
                'max_time': 0.5,
                'last_executed': datetime.now()
            }
        }
        self.mock_performance_manager.database_optimizer.get_query_stats.return_value = query_stats

        # Mock slow queries
        slow_queries = [
            {
                'query_id': 'query2',
                'query_text': 'SELECT * FROM users',
                'avg_time': 0.2
            }
        ]
        self.mock_performance_manager.database_optimizer.get_slow_queries.return_value = slow_queries

        # Mock metrics history
        metrics_history = []
        for i in range(60):
            entry_time = timestamp - timedelta(seconds=i*60)
            metrics_history.append({
                'timestamp': entry_time.isoformat(),
                'cpu_usage': 40.0 + i % 20,
                'memory_usage': 500.0 - i % 50,
                'thread_count': 8 + i % 4
            })
        self.mock_performance_manager.metrics_history = metrics_history

        # Mock optimization level
        self.mock_performance_manager.optimization_level = OptimizationLevel.STANDARD

        # Mock optimization profiles
        self.mock_performance_manager.optimization_profiles = {
            'component1': mock.MagicMock(),
            'component2': mock.MagicMock()
        }

    def _login(self):
        """Helper to simulate login"""
        with self.client.session_transaction() as sess:
            sess['token'] = 'test_token'
            sess['username'] = 'admin'

    def test_dashboard_requires_login(self):
        """Test that dashboard route requires login"""
        response = self.client.get(url_for('performance.dashboard'))
        assert response.status_code == 302  # Redirect to login

    def test_dashboard_with_login(self):
        """Test dashboard route with login"""
        self._login()
        response = self.client.get(url_for('performance.dashboard'))
        assert response.status_code == 200

        # Check that collect_metrics was called
        self.mock_performance_manager.collect_metrics.assert_called_once()

        # Check that memory history was fetched
        self.mock_performance_manager.memory_optimizer.get_memory_usage_history.assert_called_once()

        # Check that cache stats were fetched
        self.mock_performance_manager.cache.get_stats.assert_called_once()

    def test_monitor_requires_admin(self):
        """Test that monitor route requires admin privileges"""
        self._login()

        # Mock check_authorization to return False
        self.mock_security_manager.check_authorization.return_value = False

        response = self.client.get(url_for('performance.monitor'))
        assert response.status_code == 302  # Redirect to index

    def test_monitor_with_admin(self):
        """Test monitor route with admin privileges"""
        self._login()

        # Mock check_authorization to return True
        self.mock_security_manager.check_authorization.return_value = True

        response = self.client.get(url_for('performance.monitor'))
        assert response.status_code == 200

    def test_optimize_system(self):
        """Test optimize system route"""
        self._login()

        # Mock optimize_system
        self.mock_performance_manager.optimize_system.return_value = {
            'optimizations_applied': ['opt1', 'opt2', 'opt3'],
            'time_taken': 0.5
        }

        response = self.client.post(
            url_for('performance.optimize'),
            data={'target': 'MEMORY', 'level': 'ENHANCED'}
        )

        # Check redirect
        assert response.status_code == 302

        # Check that set_optimization_level was called
        self.mock_performance_manager.set_optimization_level.assert_called_once_with(
            OptimizationLevel('ENHANCED')
        )

        # Check that optimize_system was called
        self.mock_performance_manager.optimize_system.assert_called_once_with(
            OptimizationTarget('MEMORY')
        )

    def test_cache_clear(self):
        """Test cache clear route"""
        self._login()

        response = self.client.post(url_for('performance.cache_clear'))

        # Check redirect
        assert response.status_code == 302

        # Check that cache.clear was called
        self.mock_performance_manager.cache.clear.assert_called_once()

    def test_memory_gc(self):
        """Test memory garbage collection route"""
        self._login()

        response = self.client.post(url_for('performance.memory_gc'))

        # Check redirect
        assert response.status_code == 302

        # Check that memory_optimizer.run_garbage_collection was called
        self.mock_performance_manager.memory_optimizer.run_garbage_collection.assert_called_once()

    def test_api_metrics(self):
        """Test API metrics endpoint"""
        self._login()

        response = self.client.get(url_for('performance.api_metrics'))

        # Check response
        assert response.status_code == 200

        # Check JSON data
        data = json.loads(response.data)
        assert 'cpu_usage' in data
        assert 'memory_usage' in data
        assert 'thread_count' in data

    def test_api_memory(self):
        """Test API memory endpoint"""
        self._login()

        # Mock get_current_memory_usage
        self.mock_performance_manager.memory_optimizer.get_current_memory_usage.return_value = {
            'rss': 512.0,
            'vms': 1024.0,
            'percent': 25.6
        }

        response = self.client.get(url_for('performance.api_memory'))

        # Check response
        assert response.status_code == 200

        # Check JSON data
        data = json.loads(response.data)
        assert 'rss' in data
        assert 'vms' in data
        assert 'percent' in data

    def test_database_optimization(self):
        """Test database optimization route"""
        self._login()

        response = self.client.get(url_for('performance.database_optimization'))

        # Check response
        assert response.status_code == 200

        # Check that database_optimizer.get_query_stats was called
        self.mock_performance_manager.database_optimizer.get_query_stats.assert_called_once()

        # Check that database_optimizer.get_slow_queries was called
        self.mock_performance_manager.database_optimizer.get_slow_queries.assert_called_once()

if __name__ == '__main__':
    unittest.main()
