#!/usr/bin/env python3
"""
Test suite for TaskExecutor stub method implementations.
Tests the full implementation of _handle_failure, _cleanup_resources,
_setup_resource, _execute_action, and _validate_success_criteria.
"""

import unittest
import logging
import tempfile
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, Any
from pathlib import Path

# Import using importlib to avoid package initialization issues
import importlib.util

# Locate the executor module relative to this test file
_test_dir = Path(__file__).parent
_executor_path = _test_dir.parent / 'sovereign' / 'executor.py'

spec = importlib.util.spec_from_file_location('executor', str(_executor_path))
executor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor_module)
TaskExecutor = executor_module.TaskExecutor


class TestTaskExecutorInit(unittest.TestCase):
    """Test TaskExecutor initialization with new attributes."""

    def setUp(self):
        """Set up test environment."""
        self.executor = TaskExecutor()

    def test_resources_dict_exists(self):
        """Test that resources tracking dict is initialized."""
        self.assertIsInstance(self.executor.resources, dict)
        self.assertEqual(len(self.executor.resources), 0)

    def test_action_handlers_registered(self):
        """Test that action handlers are registered."""
        self.assertIsInstance(self.executor.action_handlers, dict)
        expected_handlers = ['process', 'transform', 'analyze', 'store', 'notify', 'validate']
        for handler_name in expected_handlers:
            self.assertIn(handler_name, self.executor.action_handlers)
            self.assertTrue(callable(self.executor.action_handlers[handler_name]))

    def test_retry_config_exists(self):
        """Test that retry configuration is initialized."""
        self.assertIsInstance(self.executor.retry_config, dict)
        self.assertIn('max_retries', self.executor.retry_config)
        self.assertIn('retry_delay', self.executor.retry_config)
        self.assertIn('exponential_backoff', self.executor.retry_config)
        self.assertIn('retryable_errors', self.executor.retry_config)

    def test_metrics_collection_exists(self):
        """Test that metrics collection is initialized."""
        self.assertIsInstance(self.executor.metrics, dict)
        self.assertIn('total_executions', self.executor.metrics)
        self.assertIn('successful_executions', self.executor.metrics)
        self.assertIn('failed_executions', self.executor.metrics)


class TestSetupResource(unittest.TestCase):
    """Test _setup_resource implementation."""

    def setUp(self):
        """Set up test environment."""
        self.executor = TaskExecutor()
        # Set up current_execution for phase tracking
        self.executor.current_execution = {'current_phase': 'test_phase'}

    def test_setup_memory_resource(self):
        """Test memory resource setup."""
        config = {'type': 'memory', 'capacity': 2048, 'unit': 'MB'}
        self.executor._setup_resource('test_memory', config)

        self.assertIn('test_memory', self.executor.resources)
        resource = self.executor.resources['test_memory']
        self.assertEqual(resource['type'], 'memory')
        self.assertEqual(resource['state'], 'ready')
        self.assertEqual(resource['capacity'], 2048)
        self.assertEqual(resource['unit'], 'MB')

    def test_setup_compute_resource(self):
        """Test compute resource setup."""
        config = {'type': 'compute', 'cores': 4, 'gpu': True, 'priority': 'high'}
        self.executor._setup_resource('test_compute', config)

        self.assertIn('test_compute', self.executor.resources)
        resource = self.executor.resources['test_compute']
        self.assertEqual(resource['type'], 'compute')
        self.assertEqual(resource['cores'], 4)
        self.assertTrue(resource['gpu'])
        self.assertEqual(resource['priority'], 'high')

    def test_setup_storage_resource(self):
        """Test storage resource setup."""
        config = {'type': 'storage', 'path': '/tmp/test', 'capacity': 1000}
        self.executor._setup_resource('test_storage', config)

        self.assertIn('test_storage', self.executor.resources)
        resource = self.executor.resources['test_storage']
        self.assertEqual(resource['type'], 'storage')
        self.assertEqual(resource['path'], '/tmp/test')

    def test_setup_network_resource(self):
        """Test network resource setup."""
        config = {'type': 'network', 'host': 'localhost', 'port': 8080, 'protocol': 'http'}
        self.executor._setup_resource('test_network', config)

        self.assertIn('test_network', self.executor.resources)
        resource = self.executor.resources['test_network']
        self.assertEqual(resource['type'], 'network')
        self.assertEqual(resource['host'], 'localhost')
        self.assertEqual(resource['port'], 8080)

    def test_setup_quantum_resource(self):
        """Test quantum resource setup."""
        config = {'type': 'quantum', 'qubits': 16, 'backend': 'simulator', 'fidelity': 0.95}
        self.executor._setup_resource('test_quantum', config)

        self.assertIn('test_quantum', self.executor.resources)
        resource = self.executor.resources['test_quantum']
        self.assertEqual(resource['type'], 'quantum')
        self.assertEqual(resource['qubits'], 16)
        self.assertEqual(resource['backend'], 'simulator')

    def test_setup_unknown_resource_type(self):
        """Test handling of unknown resource type."""
        config = {'type': 'unknown_type', 'custom_field': 'value'}
        self.executor._setup_resource('test_unknown', config)

        self.assertIn('test_unknown', self.executor.resources)
        resource = self.executor.resources['test_unknown']
        self.assertEqual(resource['state'], 'ready')
        self.assertIn('warning', resource)

    def test_invalid_config_raises_error(self):
        """Test that invalid config raises ValueError."""
        with self.assertRaises(ValueError):
            self.executor._setup_resource('test', 'invalid_config')

    def test_resource_limits_and_quotas(self):
        """Test that limits and quotas are set."""
        config = {
            'type': 'memory',
            'capacity': 1024,
            'limits': {'max_usage': 512},
            'quota': {'daily_limit': 1000}
        }
        self.executor._setup_resource('test_limited', config)

        resource = self.executor.resources['test_limited']
        self.assertEqual(resource['limits'], {'max_usage': 512})
        self.assertEqual(resource['quota'], {'daily_limit': 1000})

    def test_metrics_updated_on_setup(self):
        """Test that metrics are updated when resources are set up."""
        initial_count = self.executor.metrics.get('resource_allocations', 0)
        config = {'type': 'memory', 'capacity': 1024}
        self.executor._setup_resource('test_metrics', config)

        self.assertEqual(
            self.executor.metrics['resource_allocations'],
            initial_count + 1
        )


class TestExecuteAction(unittest.TestCase):
    """Test _execute_action implementation."""

    def setUp(self):
        """Set up test environment."""
        self.executor = TaskExecutor()

    def test_process_action(self):
        """Test process action execution."""
        action = {
            'type': 'process',
            'params': {'data': [1, 2, 3], 'operation': 'sum'}
        }
        result = self.executor._execute_action(action)

        self.assertTrue(result['success'])
        self.assertEqual(result['action_type'], 'process')
        self.assertIn('output', result)
        self.assertIn('metrics', result)
        self.assertIn('duration_ms', result['metrics'])

    def test_transform_action(self):
        """Test transform action execution."""
        action = {
            'type': 'transform',
            'params': {'input': {'key': 'value'}, 'transformation': 'uppercase'}
        }
        result = self.executor._execute_action(action)

        self.assertTrue(result['success'])
        self.assertEqual(result['action_type'], 'transform')
        self.assertIn('transformed_data', result['output'])

    def test_analyze_action(self):
        """Test analyze action execution."""
        action = {
            'type': 'analyze',
            'params': {'data': [1, 2, 3], 'analysis_type': 'statistics'}
        }
        result = self.executor._execute_action(action)

        self.assertTrue(result['success'])
        self.assertEqual(result['action_type'], 'analyze')
        self.assertIn('analysis_results', result['output'])

    def test_store_action(self):
        """Test store action execution."""
        action = {
            'type': 'store',
            'params': {'data': {'key': 'value'}, 'path': '/tmp/test.json'}
        }
        result = self.executor._execute_action(action)

        self.assertTrue(result['success'])
        self.assertEqual(result['action_type'], 'store')
        self.assertTrue(result['output']['stored'])

    def test_notify_action(self):
        """Test notify action execution."""
        action = {
            'type': 'notify',
            'params': {'message': 'Test notification', 'target': 'admin'}
        }
        result = self.executor._execute_action(action)

        self.assertTrue(result['success'])
        self.assertEqual(result['action_type'], 'notify')
        self.assertTrue(result['output']['sent'])

    def test_validate_action(self):
        """Test validate action execution."""
        action = {
            'type': 'validate',
            'params': {'data': {'name': 'test'}, 'schema': {'name': 'string'}}
        }
        result = self.executor._execute_action(action)

        self.assertTrue(result['success'])
        self.assertEqual(result['action_type'], 'validate')
        self.assertIn('valid', result['output'])

    def test_unknown_action_raises_error(self):
        """Test that unknown action type raises ValueError."""
        action = {'type': 'nonexistent_action', 'params': {}}
        with self.assertRaises(ValueError):
            self.executor._execute_action(action)

    def test_action_metrics_updated(self):
        """Test that action metrics are updated."""
        initial_count = self.executor.metrics.get('total_actions', 0)
        action = {'type': 'process', 'params': {}}
        self.executor._execute_action(action)

        self.assertEqual(self.executor.metrics['total_actions'], initial_count + 1)

    def test_action_result_contains_timestamps(self):
        """Test that action result contains start and end timestamps."""
        action = {'type': 'process', 'params': {}}
        result = self.executor._execute_action(action)

        self.assertIn('start_time', result)
        self.assertIn('end_time', result)
        # Verify timestamps are valid ISO format
        datetime.fromisoformat(result['start_time'])
        datetime.fromisoformat(result['end_time'])


class TestValidateSuccessCriteria(unittest.TestCase):
    """Test _validate_success_criteria implementation."""

    def setUp(self):
        """Set up test environment."""
        self.executor = TaskExecutor()
        self.executor.current_execution = {}

    def test_empty_criteria_returns_true(self):
        """Test that empty criteria returns True."""
        steps = []
        criteria = []
        result = self.executor._validate_success_criteria(steps, criteria)
        self.assertTrue(result)

    def test_none_criteria_returns_true(self):
        """Test that None criteria returns True."""
        steps = []
        result = self.executor._validate_success_criteria(steps, None)
        self.assertTrue(result)

    def test_threshold_criterion_pass(self):
        """Test threshold criterion that passes."""
        steps = [{'metrics': {'score': 95}}]
        criteria = [{'type': 'threshold', 'metric': 'score', 'threshold': 90, 'operator': '>='}]
        result = self.executor._validate_success_criteria(steps, criteria)
        self.assertTrue(result)

    def test_threshold_criterion_fail(self):
        """Test threshold criterion that fails."""
        steps = [{'metrics': {'score': 85}}]
        criteria = [{'type': 'threshold', 'metric': 'score', 'threshold': 90, 'operator': '>='}]
        result = self.executor._validate_success_criteria(steps, criteria)
        self.assertFalse(result)

    def test_completion_criterion_all_complete(self):
        """Test completion criterion with all steps complete."""
        steps = [
            {'success': True, 'step': {'name': 'step1'}},
            {'success': True, 'step': {'name': 'step2'}}
        ]
        criteria = [{'type': 'completion', 'min_completion_rate': 1.0}]
        result = self.executor._validate_success_criteria(steps, criteria)
        self.assertTrue(result)

    def test_completion_criterion_partial_complete(self):
        """Test completion criterion with partial completion."""
        steps = [
            {'success': True, 'step': {'name': 'step1'}},
            {'success': False, 'step': {'name': 'step2'}}
        ]
        criteria = [{'type': 'completion', 'min_completion_rate': 0.5}]
        result = self.executor._validate_success_criteria(steps, criteria)
        self.assertTrue(result)

    def test_required_steps_completion(self):
        """Test completion criterion with required steps."""
        steps = [
            {'success': True, 'step': {'name': 'critical_step'}},
            {'success': False, 'step': {'name': 'optional_step'}}
        ]
        criteria = [{'type': 'completion', 'required_steps': ['critical_step']}]
        result = self.executor._validate_success_criteria(steps, criteria)
        self.assertTrue(result)

    def test_time_criterion_max_duration(self):
        """Test time criterion with max duration."""
        steps = [{'metrics': {'duration_ms': 100}}, {'metrics': {'duration_ms': 200}}]
        criteria = [{'type': 'time', 'max_duration_ms': 500}]
        result = self.executor._validate_success_criteria(steps, criteria)
        self.assertTrue(result)

    def test_time_criterion_exceeds_max(self):
        """Test time criterion that exceeds max duration."""
        steps = [{'metrics': {'duration_ms': 300}}, {'metrics': {'duration_ms': 400}}]
        criteria = [{'type': 'time', 'max_duration_ms': 500}]
        result = self.executor._validate_success_criteria(steps, criteria)
        self.assertFalse(result)

    def test_quality_criterion_pass(self):
        """Test quality criterion that passes."""
        steps = [{'output': {'quality_score': 0.9}}, {'output': {'quality_score': 0.8}}]
        criteria = [{'type': 'quality', 'min_quality_score': 0.8}]
        result = self.executor._validate_success_criteria(steps, criteria)
        self.assertTrue(result)

    def test_resource_criterion_pass(self):
        """Test resource criterion that passes."""
        self.executor.current_execution = {'resource_usage': {'memory_mb': 100}}
        steps = []
        criteria = [{'type': 'resource', 'max_memory_mb': 200}]
        result = self.executor._validate_success_criteria(steps, criteria)
        self.assertTrue(result)

    def test_unknown_criterion_type(self):
        """Test that unknown criterion type doesn't fail validation."""
        steps = []
        criteria = [{'type': 'unknown_criterion'}]
        result = self.executor._validate_success_criteria(steps, criteria)
        self.assertTrue(result)

    def test_multiple_criteria(self):
        """Test multiple criteria validation."""
        steps = [
            {'success': True, 'step': {'name': 'step1'}, 'metrics': {'score': 95}}
        ]
        criteria = [
            {'type': 'threshold', 'metric': 'score', 'threshold': 90, 'operator': '>='},
            {'type': 'completion', 'min_completion_rate': 1.0}
        ]
        result = self.executor._validate_success_criteria(steps, criteria)
        self.assertTrue(result)


class TestHandleFailure(unittest.TestCase):
    """Test _handle_failure implementation."""

    def setUp(self):
        """Set up test environment."""
        self.executor = TaskExecutor()
        self.executor.current_execution = {
            'plan': {'notifications': {}},
            'status': 'in_progress'
        }
        # Disable retry delay for faster tests
        self.executor.retry_config['retry_delay'] = 0.01
        self.executor.retry_config['max_retries'] = 1

    def test_failure_details_recorded(self):
        """Test that failure details are recorded."""
        result = {
            'error': 'Test error',
            'phase': {'name': 'test_phase'},
            'steps': []
        }
        self.executor._handle_failure(result)

        self.assertIn('failure_details', self.executor.current_execution)
        failure_details = self.executor.current_execution['failure_details']
        self.assertEqual(failure_details['error'], 'Test error')
        self.assertEqual(failure_details['phase'], 'test_phase')

    def test_failure_metrics_updated(self):
        """Test that failure metrics are updated."""
        initial_count = self.executor.metrics.get('failed_executions', 0)
        result = {'error': 'Test error', 'phase': {}, 'steps': []}
        self.executor._handle_failure(result)

        self.assertEqual(
            self.executor.metrics['failed_executions'],
            initial_count + 1
        )

    def test_classify_timeout_error(self):
        """Test error classification for timeout."""
        error_type = self.executor._classify_error("Connection timeout occurred")
        self.assertEqual(error_type, 'timeout')

    def test_classify_connection_error(self):
        """Test error classification for connection error."""
        error_type = self.executor._classify_error("Connection error: failed to connect")
        self.assertEqual(error_type, 'connection_error')

    def test_classify_resource_unavailable(self):
        """Test error classification for resource unavailable."""
        error_type = self.executor._classify_error("Resource is unavailable")
        self.assertEqual(error_type, 'resource_unavailable')

    def test_classify_unknown_error(self):
        """Test error classification for unknown error."""
        error_type = self.executor._classify_error("Some random error")
        self.assertEqual(error_type, 'unknown')


class TestCleanupResources(unittest.TestCase):
    """Test _cleanup_resources implementation."""

    def setUp(self):
        """Set up test environment."""
        self.executor = TaskExecutor()

    def test_cleanup_releases_all_resources(self):
        """Test that cleanup releases all tracked resources."""
        # Add some test resources
        self.executor.resources = {
            'resource1': {'type': 'memory', 'state': 'ready'},
            'resource2': {'type': 'compute', 'state': 'ready'}
        }
        self.executor._cleanup_resources()

        self.assertEqual(len(self.executor.resources), 0)

    def test_cleanup_updates_metrics(self):
        """Test that cleanup updates metrics."""
        initial_count = self.executor.metrics.get('resource_cleanup_count', 0)
        self.executor._cleanup_resources()

        self.assertEqual(
            self.executor.metrics['resource_cleanup_count'],
            initial_count + 1
        )

    def test_cleanup_temp_files(self):
        """Test cleanup of temporary files."""
        # Create a real temp file
        fd, temp_path = tempfile.mkstemp()
        os.close(fd)
        self.executor._temp_files.append(temp_path)

        self.assertTrue(os.path.exists(temp_path))
        self.executor._cleanup_resources()
        self.assertFalse(os.path.exists(temp_path))

    def test_cleanup_file_handles(self):
        """Test cleanup of file handles."""
        # Create a file handle
        fd, temp_path = tempfile.mkstemp()
        file_handle = os.fdopen(fd, 'w')
        self.executor._file_handles.append(file_handle)

        self.assertFalse(file_handle.closed)
        self.executor._cleanup_resources()
        self.assertTrue(file_handle.closed)

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_cleanup_handles_errors_gracefully(self):
        """Test that cleanup handles errors gracefully."""
        # Add a non-existent temp file
        self.executor._temp_files.append('/nonexistent/path/file.txt')
        # Should not raise an exception
        self.executor._cleanup_resources()


class TestEndToEndExecution(unittest.TestCase):
    """Test end-to-end plan execution with new implementations."""

    def setUp(self):
        """Set up test environment."""
        self.executor = TaskExecutor()

    def test_simple_plan_execution(self):
        """Test execution of a simple plan."""
        plan = {
            'objectives': ['test'],
            'resources': {
                'test_memory': {'type': 'memory', 'capacity': 1024}
            },
            'timeline': {
                'phases': [{
                    'name': 'test_phase',
                    'steps': [{
                        'name': 'test_step',
                        'action': {'type': 'process', 'params': {'data': [1, 2, 3]}}
                    }]
                }]
            },
            'success_criteria': []
        }

        result = self.executor.execute_plan(plan)

        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result.get('completed', False))

    def test_plan_with_multiple_action_types(self):
        """Test execution with multiple action types."""
        plan = {
            'objectives': ['test'],
            'resources': {
                'test_resource': {'type': 'compute', 'cores': 2}
            },
            'timeline': {
                'phases': [{
                    'name': 'multi_action_phase',
                    'steps': [
                        {'name': 'process_step', 'action': {'type': 'process', 'params': {}}},
                        {'name': 'transform_step', 'action': {'type': 'transform', 'params': {}}},
                        {'name': 'analyze_step', 'action': {'type': 'analyze', 'params': {}}}
                    ]
                }]
            },
            'success_criteria': []
        }

        result = self.executor.execute_plan(plan)

        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(result['steps']), 1)  # One phase
        self.assertEqual(len(result['steps'][0]['steps']), 3)  # Three steps

    def test_plan_with_success_criteria(self):
        """Test execution with success criteria validation."""
        plan = {
            'objectives': ['test'],
            'resources': {},
            'timeline': {
                'phases': [{
                    'name': 'test_phase',
                    'steps': [
                        {'name': 'step1', 'action': {'type': 'process', 'params': {}}}
                    ]
                }]
            },
            'success_criteria': [
                {'type': 'completion', 'min_completion_rate': 1.0}
            ]
        }

        result = self.executor.execute_plan(plan)

        self.assertEqual(result['status'], 'completed')


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    unittest.main(verbosity=2)
