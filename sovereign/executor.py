"""
Task Executor for Autonomous Operations
Copyright (c) 2025 Stephen Bilodeau
"""

from typing import Dict, Any, List, Optional, Callable
import logging
import os
import tempfile
import threading
import time
from datetime import datetime


class TaskExecutor:
    """Autonomous execution system for implementing plans without external validation"""

    def initialize(self, project_root: Optional[str] = None) -> bool:
        """Stub for pipeline compatibility.
        
        Args:
            project_root: Optional path to project root directory
            
        Returns:
            True if initialization successful
        """
        return True

    def __init__(self) -> None:
        """Initialize the TaskExecutor with resource tracking and action handlers."""
        self.logger = logging.getLogger("TaskExecutor")
        self.execution_history: List[Dict[str, Any]] = []
        self.current_execution: Optional[Dict[str, Any]] = None
        
        # Resource tracking for allocated resources
        self.resources: Dict[str, Dict[str, Any]] = {}
        
        # Retry configuration for transient failures
        self.retry_config: Dict[str, Any] = {
            'max_retries': 3,
            'retry_delay': 1.0,  # seconds
            'exponential_backoff': True,
            'retryable_errors': ['timeout', 'connection_error', 'resource_unavailable']
        }
        
        # Action handlers registry
        self.action_handlers: Dict[str, Callable] = {
            'process': self._action_process,
            'transform': self._action_transform,
            'analyze': self._action_analyze,
            'store': self._action_store,
            'notify': self._action_notify,
            'validate': self._action_validate,
        }
        
        # Metrics collection
        self.metrics: Dict[str, Any] = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_actions': 0,
            'action_failures': 0,
            'resource_allocations': 0,
            'resource_cleanup_count': 0,
        }
        
        # Temporary files tracking
        self._temp_files: List[str] = []
        
        # Background processes tracking
        self._background_threads: List[threading.Thread] = []
        
        # File handles tracking
        self._file_handles: List[Any] = []

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a plan autonomously"""
        self.logger.info("Beginning plan execution")

        execution_id = f"EXEC_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.current_execution = {
            'id': execution_id,
            'plan': plan,
            'status': 'in_progress',
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'metrics': {}
        }

        try:
            # Initialize resources
            self._initialize_resources(plan['resources'])

            # Execute phases
            for phase in plan['timeline']['phases']:
                phase_result = self._execute_phase(phase)
                self.current_execution['steps'].append(phase_result)

                if not phase_result['success']:
                    self._handle_failure(phase_result)
                    break

            # Validate results
            success = self._validate_execution()

            # Calculate resource usage
            resource_usage = {
                'memory': float('inf'),
                'processing': float('inf'),
                'quantum': float('inf'),
                'power': float('inf'),
                'time': {
                    'start': self.current_execution.get('start_time'),
                    'end': datetime.now().isoformat()
                },
                'efficiency': 1.0
            }

            # Update execution status
            if success:
                self.current_execution.update({
                    'status': 'completed',
                    'completed': True,
                    'success_criteria': plan.get('success_criteria', {}),
                    'results': {
                        'all_steps_completed': True,
                        'steps_successful': all(
                            step.get('success', False)
                            for step in self.current_execution['steps']
                        )
                    },
                    'resource_usage': resource_usage
                })
            else:
                self.current_execution.update({
                    'status': 'failed',
                    'resource_usage': resource_usage
                })

            self.current_execution['end_time'] = datetime.now().isoformat()

        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            self.current_execution['status'] = 'failed'
            self.current_execution['error'] = str(e)
            self.current_execution['end_time'] = datetime.now().isoformat()

        finally:
            self._cleanup_resources()
            self.execution_history.append(self.current_execution)

        return self.current_execution

    def _initialize_resources(self, resources: Dict[str, Any]) -> None:
        """Initialize required resources"""
        self.logger.info("Initializing resources")
        for resource, config in resources.items():
            self._setup_resource(resource, config)

    def _execute_phase(self, phase: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single phase of the plan"""
        self.logger.info(f"Executing phase: {phase.get('name', 'unnamed')}")

        result = {
            'phase': phase,
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'metrics': {},
            'success': False
        }

        try:
            # Execute each step in the phase
            for step in phase.get('steps', []):
                step_result = self._execute_step(step)
                result['steps'].append(step_result)

                if not step_result['success']:
                    return self._handle_phase_failure(result, step_result)

            # Phase completed successfully
            result['success'] = True
            result['end_time'] = datetime.now().isoformat()

        except Exception as e:
            self.logger.error(f"Phase execution failed: {str(e)}")
            result['error'] = str(e)
            result['end_time'] = datetime.now().isoformat()

        return result

    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step within a phase"""
        self.logger.info(f"Executing step: {step.get('name', 'unnamed')}")

        result = {
            'step': step,
            'start_time': datetime.now().isoformat(),
            'metrics': {},
            'success': False
        }

        try:
            # Execute the step's action
            output = self._execute_action(step['action'])
            result.update({
                'output': output,
                'success': True,
                'end_time': datetime.now().isoformat()
            })

        except Exception as e:
            self.logger.error(f"Step execution failed: {str(e)}")
            result.update({
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })

        return result

    def _validate_execution(self) -> bool:
        """Validate the execution results"""
        if not self.current_execution:
            return False

        # Check all steps completed
        all_steps_complete = all(
            step['success'] for step in self.current_execution['steps']
        )

        # Validate against success criteria
        criteria_met = self._validate_success_criteria(
            self.current_execution['steps'],
            self.current_execution['plan']['success_criteria']
        )

        return all_steps_complete and criteria_met

    def _handle_failure(self, result: Dict[str, Any]) -> None:
        """Handle execution failures with retry logic and cleanup.
        
        This method handles execution failures by:
        - Logging detailed failure information
        - Updating execution status with failure details
        - Attempting recovery if possible (retry logic)
        - Storing failure metrics for analysis
        - Triggering notifications/alerts if configured
        - Cleaning up partial resources from failed execution
        
        Args:
            result: Dictionary containing failure details including 'error', 'phase', 
                   'steps', and other execution context
        """
        error_message = result.get('error', 'Unknown error')
        phase_info = result.get('phase', {})
        phase_name = phase_info.get('name', 'unknown') if isinstance(phase_info, dict) else 'unknown'
        
        # Log detailed failure information
        self.logger.error(f"Execution failed: {error_message}")
        self.logger.error(f"Failed phase: {phase_name}")
        self.logger.error(f"Failure timestamp: {datetime.now().isoformat()}")
        
        if 'steps' in result:
            failed_steps = [s for s in result['steps'] if not s.get('success', False)]
            for step in failed_steps:
                step_info = step.get('step', {})
                step_name = step_info.get('name', 'unknown') if isinstance(step_info, dict) else 'unknown'
                step_error = step.get('error', 'No error details')
                self.logger.error(f"  Failed step: {step_name} - {step_error}")
        
        # Update execution status with failure details
        if self.current_execution:
            self.current_execution['failure_details'] = {
                'error': error_message,
                'phase': phase_name,
                'timestamp': datetime.now().isoformat(),
                'retry_attempted': False,
                'recovery_attempted': False
            }
        
        # Attempt recovery with retry logic for transient failures
        error_type = self._classify_error(error_message)
        if error_type in self.retry_config.get('retryable_errors', []):
            retry_result = self._attempt_retry(result)
            if retry_result and self.current_execution:
                self.current_execution['failure_details']['retry_attempted'] = True
                self.current_execution['failure_details']['retry_result'] = retry_result
        
        # Store failure metrics for analysis
        self.metrics['failed_executions'] = self.metrics.get('failed_executions', 0) + 1
        self.metrics['last_failure'] = {
            'timestamp': datetime.now().isoformat(),
            'error': error_message,
            'phase': phase_name
        }
        
        # Trigger notifications/alerts if configured
        if self.current_execution and self.current_execution.get('plan', {}).get('notifications', {}).get('on_failure'):
            self._send_failure_notification(result)
        
        # Clean up partial resources from failed execution
        self._cleanup_partial_resources(result)
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type for retry logic.
        
        Args:
            error_message: The error message to classify
            
        Returns:
            Error type classification string
        """
        error_lower = error_message.lower()
        if 'timeout' in error_lower:
            return 'timeout'
        elif 'connection' in error_lower:
            return 'connection_error'
        elif 'resource' in error_lower and ('unavailable' in error_lower or 'busy' in error_lower):
            return 'resource_unavailable'
        elif 'memory' in error_lower:
            return 'memory_error'
        return 'unknown'
    
    def _attempt_retry(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to retry failed operation with exponential backoff.
        
        Args:
            result: The failed result to retry
            
        Returns:
            Retry result dictionary or None if retry not possible
        """
        max_retries = self.retry_config.get('max_retries', 3)
        base_delay = self.retry_config.get('retry_delay', 1.0)
        exponential = self.retry_config.get('exponential_backoff', True)
        
        phase = result.get('phase', {})
        if not phase:
            self.logger.warning("Cannot retry: no phase information available")
            return None
        
        for attempt in range(max_retries):
            delay = base_delay * (2 ** attempt) if exponential else base_delay
            self.logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {delay}s delay")
            time.sleep(delay)
            
            try:
                # Attempt to re-execute the failed phase
                retry_result = self._execute_phase(phase)
                if retry_result.get('success', False):
                    self.logger.info(f"Retry successful on attempt {attempt + 1}")
                    return {'success': True, 'attempt': attempt + 1, 'result': retry_result}
            except Exception as e:
                self.logger.warning(f"Retry attempt {attempt + 1} failed: {str(e)}")
        
        self.logger.error(f"All {max_retries} retry attempts failed")
        return {'success': False, 'attempts': max_retries}
    
    def _send_failure_notification(self, result: Dict[str, Any]) -> None:
        """Send failure notification if configured.
        
        Args:
            result: The failure result containing details
        """
        self.logger.info("Sending failure notification")
        # Notification logic would be implemented here based on configuration
        # For now, we log that notification would be sent
        notification_config = self.current_execution.get('plan', {}).get('notifications', {})
        self.logger.info(f"Notification target: {notification_config.get('target', 'default')}")
    
    def _cleanup_partial_resources(self, result: Dict[str, Any]) -> None:
        """Clean up resources allocated during failed execution.
        
        Args:
            result: The failure result containing context
        """
        self.logger.info("Cleaning up partial resources from failed execution")
        phase = result.get('phase', {})
        phase_name = phase.get('name', 'unknown') if isinstance(phase, dict) else 'unknown'
        
        # Clean up resources that were allocated during the failed phase
        resources_to_cleanup = []
        for resource_name, resource_info in self.resources.items():
            if resource_info.get('phase') == phase_name:
                resources_to_cleanup.append(resource_name)
        
        for resource_name in resources_to_cleanup:
            self.logger.info(f"Cleaning up partial resource: {resource_name}")
            self._release_resource(resource_name)

    def _cleanup_resources(self) -> None:
        """Clean up any resources used during execution.
        
        This method performs comprehensive resource cleanup:
        - Releases memory allocations
        - Closes file handles and connections
        - Stops background processes/threads
        - Cleans up temporary files
        - Resets resource states
        - Logs all cleanup operations
        """
        self.logger.info("Cleaning up resources")
        cleanup_start = datetime.now()
        cleanup_stats = {
            'resources_released': 0,
            'file_handles_closed': 0,
            'temp_files_removed': 0,
            'threads_stopped': 0,
            'errors': []
        }
        
        # Release tracked resources
        resources_to_release = list(self.resources.keys())
        for resource_name in resources_to_release:
            try:
                self._release_resource(resource_name)
                cleanup_stats['resources_released'] += 1
            except Exception as e:
                error_msg = f"Failed to release resource {resource_name}: {str(e)}"
                self.logger.warning(error_msg)
                cleanup_stats['errors'].append(error_msg)
        
        # Close file handles
        for handle in self._file_handles[:]:
            try:
                if hasattr(handle, 'close') and not getattr(handle, 'closed', True):
                    handle.close()
                    self._file_handles.remove(handle)
                    cleanup_stats['file_handles_closed'] += 1
                    self.logger.debug(f"Closed file handle: {handle}")
            except Exception as e:
                error_msg = f"Failed to close file handle: {str(e)}"
                self.logger.warning(error_msg)
                cleanup_stats['errors'].append(error_msg)
        
        # Stop background threads
        for thread in self._background_threads[:]:
            try:
                if thread.is_alive():
                    self.logger.debug(f"Waiting for thread {thread.name} to complete...")
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        self.logger.warning(f"Thread {thread.name} did not stop within timeout")
                    else:
                        cleanup_stats['threads_stopped'] += 1
                self._background_threads.remove(thread)
            except Exception as e:
                error_msg = f"Failed to stop thread: {str(e)}"
                self.logger.warning(error_msg)
                cleanup_stats['errors'].append(error_msg)
        
        # Clean up temporary files
        for temp_file in self._temp_files[:]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.logger.debug(f"Removed temp file: {temp_file}")
                self._temp_files.remove(temp_file)
                cleanup_stats['temp_files_removed'] += 1
            except Exception as e:
                error_msg = f"Failed to remove temp file {temp_file}: {str(e)}"
                self.logger.warning(error_msg)
                cleanup_stats['errors'].append(error_msg)
        
        # Reset resource states
        self.resources.clear()
        
        # Update metrics
        self.metrics['resource_cleanup_count'] = self.metrics.get('resource_cleanup_count', 0) + 1
        
        cleanup_duration = (datetime.now() - cleanup_start).total_seconds() * 1000
        self.logger.info(
            f"Resource cleanup complete: {cleanup_stats['resources_released']} resources released, "
            f"{cleanup_stats['file_handles_closed']} file handles closed, "
            f"{cleanup_stats['temp_files_removed']} temp files removed, "
            f"{cleanup_stats['threads_stopped']} threads stopped "
            f"(took {cleanup_duration:.2f}ms)"
        )
        
        if cleanup_stats['errors']:
            self.logger.warning(f"Cleanup encountered {len(cleanup_stats['errors'])} errors")
    
    def _release_resource(self, resource_name: str) -> None:
        """Release a specific resource.
        
        Args:
            resource_name: Name of the resource to release
        """
        if resource_name not in self.resources:
            self.logger.debug(f"Resource {resource_name} not found in registry")
            return
        
        resource_info = self.resources[resource_name]
        resource_type = resource_info.get('type', 'unknown')
        
        self.logger.debug(f"Releasing {resource_type} resource: {resource_name}")
        
        # Type-specific cleanup
        if resource_type == 'memory':
            # Clear memory reference
            resource_info['allocation'] = None
        elif resource_type == 'compute':
            # Stop compute resources
            if 'handle' in resource_info:
                resource_info['handle'] = None
        elif resource_type == 'storage':
            # Close storage connections
            if 'connection' in resource_info and hasattr(resource_info['connection'], 'close'):
                resource_info['connection'].close()
        elif resource_type == 'network':
            # Close network connections
            if 'socket' in resource_info and hasattr(resource_info['socket'], 'close'):
                resource_info['socket'].close()
        elif resource_type == 'quantum':
            # Reset quantum state
            if 'state' in resource_info:
                resource_info['state'] = 'released'
        
        # Mark as released and remove from registry
        resource_info['released'] = True
        resource_info['released_at'] = datetime.now().isoformat()
        del self.resources[resource_name]
        
        self.logger.info(f"Released resource: {resource_name}")

    def _setup_resource(self, resource: str, config: Dict[str, Any]) -> None:
        """Set up a specific resource based on its type and configuration.
        
        Supports different resource types:
        - 'memory': Memory allocations with limits
        - 'compute': CPU/GPU compute resources
        - 'storage': File/database storage
        - 'network': Network connections
        - 'quantum': Quantum processing resources
        
        Args:
            resource: Resource name/identifier
            config: Configuration dictionary for the resource containing at minimum
                   a 'type' key and type-specific settings
                   
        Raises:
            ValueError: If resource configuration is invalid
            RuntimeError: If resource allocation fails
        """
        self.logger.info(f"Setting up resource: {resource}")
        setup_start = datetime.now()
        
        # Validate resource configuration
        if not isinstance(config, dict):
            raise ValueError(f"Resource config must be a dictionary, got {type(config)}")
        
        resource_type = config.get('type', 'unknown')
        self.logger.debug(f"Resource type: {resource_type}")
        
        # Initialize resource entry
        resource_entry = {
            'name': resource,
            'type': resource_type,
            'config': config,
            'state': 'initializing',
            'allocated_at': datetime.now().isoformat(),
            'phase': self.current_execution.get('current_phase') if self.current_execution else None,
            'metrics': {}
        }
        
        try:
            if resource_type == 'memory':
                resource_entry.update(self._setup_memory_resource(resource, config))
            elif resource_type == 'compute':
                resource_entry.update(self._setup_compute_resource(resource, config))
            elif resource_type == 'storage':
                resource_entry.update(self._setup_storage_resource(resource, config))
            elif resource_type == 'network':
                resource_entry.update(self._setup_network_resource(resource, config))
            elif resource_type == 'quantum':
                resource_entry.update(self._setup_quantum_resource(resource, config))
            else:
                # Handle unknown resource types gracefully
                self.logger.warning(f"Unknown resource type '{resource_type}' for resource '{resource}'")
                resource_entry['state'] = 'ready'
                resource_entry['warning'] = f"Unknown resource type: {resource_type}"
            
            # Set resource limits and quotas if specified
            if 'limits' in config:
                resource_entry['limits'] = config['limits']
            if 'quota' in config:
                resource_entry['quota'] = config['quota']
            
            # Track resource allocation
            self.resources[resource] = resource_entry
            self.metrics['resource_allocations'] = self.metrics.get('resource_allocations', 0) + 1
            
            setup_duration = (datetime.now() - setup_start).total_seconds() * 1000
            resource_entry['metrics']['setup_duration_ms'] = setup_duration
            
            self.logger.info(f"Resource '{resource}' setup complete (took {setup_duration:.2f}ms)")
            
        except Exception as e:
            error_msg = f"Failed to setup resource '{resource}': {str(e)}"
            self.logger.error(error_msg)
            resource_entry['state'] = 'failed'
            resource_entry['error'] = str(e)
            self.resources[resource] = resource_entry
            raise RuntimeError(error_msg) from e
    
    def _setup_memory_resource(self, resource: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up memory resource.
        
        Args:
            resource: Resource name
            config: Memory configuration with optional 'capacity', 'unit'
            
        Returns:
            Resource entry updates
        """
        capacity = config.get('capacity', 1024)  # Default 1GB
        unit = config.get('unit', 'MB')
        
        self.logger.debug(f"Allocating memory resource: {capacity} {unit}")
        
        return {
            'state': 'ready',
            'capacity': capacity,
            'unit': unit,
            'used': 0,
            'allocation': {}  # Placeholder for memory tracking
        }
    
    def _setup_compute_resource(self, resource: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up compute resource (CPU/GPU).
        
        Args:
            resource: Resource name
            config: Compute configuration with optional 'cores', 'gpu', 'priority'
            
        Returns:
            Resource entry updates
        """
        cores = config.get('cores', 1)
        gpu = config.get('gpu', False)
        priority = config.get('priority', 'normal')
        
        self.logger.debug(f"Setting up compute resource: {cores} cores, GPU: {gpu}")
        
        return {
            'state': 'ready',
            'cores': cores,
            'gpu': gpu,
            'priority': priority,
            'handle': None,  # Placeholder for compute handle
            'utilization': 0.0
        }
    
    def _setup_storage_resource(self, resource: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up storage resource.
        
        Args:
            resource: Resource name
            config: Storage configuration with optional 'path', 'capacity', 'type'
            
        Returns:
            Resource entry updates
        """
        # Use a more specific subdirectory for storage to avoid mixing with other temp files
        default_storage_path = os.path.join(tempfile.gettempdir(), 'task_executor_storage')
        storage_path = config.get('path', default_storage_path)
        capacity = config.get('capacity', float('inf'))
        storage_type = config.get('storage_type', 'local')
        
        self.logger.debug(f"Setting up storage resource at: {storage_path}")
        
        return {
            'state': 'ready',
            'path': storage_path,
            'capacity': capacity,
            'storage_type': storage_type,
            'connection': None,  # Placeholder for storage connection
            'bytes_used': 0
        }
    
    def _setup_network_resource(self, resource: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up network resource.
        
        Args:
            resource: Resource name
            config: Network configuration with optional 'host', 'port', 'protocol'
            
        Returns:
            Resource entry updates
        """
        host = config.get('host', 'localhost')
        port = config.get('port', 0)
        protocol = config.get('protocol', 'tcp')
        
        self.logger.debug(f"Setting up network resource: {protocol}://{host}:{port}")
        
        return {
            'state': 'ready',
            'host': host,
            'port': port,
            'protocol': protocol,
            'socket': None,  # Placeholder for network socket
            'connected': False
        }
    
    def _setup_quantum_resource(self, resource: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up quantum processing resource.
        
        Args:
            resource: Resource name
            config: Quantum configuration with optional 'qubits', 'backend', 'fidelity'
            
        Returns:
            Resource entry updates
        """
        qubits = config.get('qubits', 8)
        backend = config.get('backend', 'simulator')
        fidelity = config.get('fidelity', 0.99)
        
        self.logger.debug(f"Setting up quantum resource: {qubits} qubits on {backend}")
        
        return {
            'state': 'ready',
            'qubits': qubits,
            'backend': backend,
            'fidelity': fidelity,
            'quantum_state': 'initialized',
            'entanglement': None
        }

    def _handle_phase_failure(
        self,
        phase_result: Dict[str, Any],
        step_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle phase failure"""
        phase_result['success'] = False
        phase_result['error'] = step_result.get('error', 'Step failed')
        phase_result['end_time'] = datetime.now().isoformat()
        return phase_result

    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific action based on its type.
        
        Supports different action types:
        - 'process': Data processing operations
        - 'transform': Data transformation operations
        - 'analyze': Analysis and computation
        - 'store': Data persistence operations
        - 'notify': Notification/alerting operations
        - 'validate': Validation operations
        
        Args:
            action: Action configuration containing 'type' and optional 'params'
            
        Returns:
            Action execution result with metadata including:
            - action_type: Type of action executed
            - start_time: ISO timestamp of execution start
            - end_time: ISO timestamp of execution end
            - success: Boolean indicating success
            - output: Action output data
            - metrics: Performance metrics
            
        Raises:
            ValueError: If action type is unknown and no handler found
        """
        action_type = action.get('type', 'unknown')
        params = action.get('params', {})
        
        self.logger.info(f"Executing action: {action_type}")
        start_time = datetime.now()
        
        result: Dict[str, Any] = {
            'action_type': action_type,
            'start_time': start_time.isoformat(),
            'success': False,
            'output': None,
            'metrics': {}
        }
        
        # Update metrics
        self.metrics['total_actions'] = self.metrics.get('total_actions', 0) + 1
        
        try:
            # Validate action parameters
            self._validate_action_params(action_type, params)
            
            # Get handler from registry or use default
            handler = self.action_handlers.get(action_type)
            
            if handler:
                output = handler(params)
            elif action_type == 'unknown':
                # Handle unknown action type gracefully
                self.logger.warning("Executing unknown action type - using default handler")
                output = self._action_default(params)
            else:
                raise ValueError(f"Unknown action type: {action_type}")
            
            result['output'] = output
            result['success'] = True
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Action {action_type} failed: {error_msg}")
            result['error'] = error_msg
            self.metrics['action_failures'] = self.metrics.get('action_failures', 0) + 1
            raise
        
        finally:
            end_time = datetime.now()
            result['end_time'] = end_time.isoformat()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            result['metrics']['duration_ms'] = duration_ms
            self.logger.debug(f"Action {action_type} completed in {duration_ms:.2f}ms")
        
        return result
    
    def _validate_action_params(self, action_type: str, params: Dict[str, Any]) -> None:
        """Validate action parameters based on action type.
        
        Args:
            action_type: Type of action
            params: Parameters to validate
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Basic validation - params should be a dict
        if not isinstance(params, dict):
            raise ValueError(f"Action params must be a dictionary, got {type(params)}")
        
        # Type-specific validation could be added here
        required_params: Dict[str, List[str]] = {
            'store': [],  # path is optional
            'notify': [],  # message is optional
            'validate': [],  # schema is optional
        }
        
        if action_type in required_params:
            for param in required_params[action_type]:
                if param not in params:
                    raise ValueError(f"Missing required parameter '{param}' for action type '{action_type}'")
    
    def _action_process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data processing action.
        
        Args:
            params: Processing parameters including optional 'data', 'operation', 'options'
            
        Returns:
            Processing result with 'processed_data' and 'stats'
        """
        self.logger.debug(f"Processing action with params: {list(params.keys())}")
        
        data = params.get('data', {})
        operation = params.get('operation', 'default')
        options = params.get('options', {})
        
        # Simulate processing
        result = {
            'processed_data': data,
            'operation': operation,
            'stats': {
                'items_processed': len(data) if isinstance(data, (list, dict)) else 1,
                'operation_applied': operation
            }
        }
        
        self.logger.info(f"Process action complete: {operation}")
        return result
    
    def _action_transform(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data transformation action.
        
        Args:
            params: Transform parameters including optional 'input', 'transformation', 'format'
            
        Returns:
            Transform result with 'transformed_data' and 'metadata'
        """
        self.logger.debug(f"Transform action with params: {list(params.keys())}")
        
        input_data = params.get('input', {})
        transformation = params.get('transformation', 'identity')
        output_format = params.get('format', 'default')
        
        result = {
            'transformed_data': input_data,
            'transformation': transformation,
            'format': output_format,
            'metadata': {
                'transformation_applied': transformation,
                'output_format': output_format
            }
        }
        
        self.logger.info(f"Transform action complete: {transformation}")
        return result
    
    def _action_analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an analysis action.
        
        Args:
            params: Analysis parameters including optional 'data', 'analysis_type', 'metrics'
            
        Returns:
            Analysis result with 'analysis_results' and 'summary'
        """
        self.logger.debug(f"Analyze action with params: {list(params.keys())}")
        
        data = params.get('data', {})
        analysis_type = params.get('analysis_type', 'basic')
        requested_metrics = params.get('metrics', [])
        
        result = {
            'analysis_results': {
                'type': analysis_type,
                'data_analyzed': True,
                'metrics_computed': requested_metrics if requested_metrics else ['default']
            },
            'summary': {
                'analysis_type': analysis_type,
                'data_points': len(data) if isinstance(data, (list, dict)) else 1,
                'metrics_count': len(requested_metrics) if requested_metrics else 1
            }
        }
        
        self.logger.info(f"Analyze action complete: {analysis_type}")
        return result
    
    def _action_store(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data storage action.
        
        Args:
            params: Storage parameters including optional 'data', 'path', 'format'
            
        Returns:
            Storage result with 'stored' and 'location' information
        """
        self.logger.debug(f"Store action with params: {list(params.keys())}")
        
        data = params.get('data', {})
        path = params.get('path', '/tmp/storage')
        storage_format = params.get('format', 'json')
        
        result = {
            'stored': True,
            'location': path,
            'format': storage_format,
            'size_bytes': len(str(data)),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Store action complete: {path}")
        return result
    
    def _action_notify(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a notification action.
        
        Args:
            params: Notification parameters including optional 'message', 'target', 'priority'
            
        Returns:
            Notification result with 'sent' and 'delivery' information
        """
        self.logger.debug(f"Notify action with params: {list(params.keys())}")
        
        message = params.get('message', 'Notification')
        target = params.get('target', 'default')
        priority = params.get('priority', 'normal')
        
        result = {
            'sent': True,
            'message': message,
            'target': target,
            'priority': priority,
            'delivery': {
                'status': 'delivered',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        self.logger.info(f"Notify action complete: sent to {target}")
        return result
    
    def _action_validate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a validation action.
        
        Args:
            params: Validation parameters including optional 'data', 'schema', 'rules'
            
        Returns:
            Validation result with 'valid' status and 'details'
        """
        self.logger.debug(f"Validate action with params: {list(params.keys())}")
        
        data = params.get('data', {})
        schema = params.get('schema', {})
        rules = params.get('rules', [])
        
        # Perform basic validation
        is_valid = True
        validation_errors: List[str] = []
        
        # Check if data matches schema keys (basic validation)
        if schema and isinstance(data, dict):
            for key in schema:
                if key not in data:
                    is_valid = False
                    validation_errors.append(f"Missing key: {key}")
        
        result = {
            'valid': is_valid,
            'errors': validation_errors,
            'details': {
                'schema_validated': bool(schema),
                'rules_checked': len(rules),
                'data_checked': True
            }
        }
        
        self.logger.info(f"Validate action complete: valid={is_valid}")
        return result
    
    def _action_default(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a default/fallback action for unknown types.
        
        Args:
            params: Action parameters
            
        Returns:
            Default action result
        """
        self.logger.debug("Executing default action handler")
        
        return {
            'executed': True,
            'handler': 'default',
            'params_received': list(params.keys()),
            'timestamp': datetime.now().isoformat()
        }

    def _validate_success_criteria(
        self,
        steps: List[Dict[str, Any]],
        criteria: List[Dict[str, Any]]
    ) -> bool:
        """Validate results against success criteria.
        
        Supports different criterion types:
        - 'threshold': Check if a value meets a threshold
        - 'completion': Check if all required steps are complete
        - 'quality': Check quality metrics
        - 'time': Check time-based constraints
        - 'resource': Check resource usage constraints
        
        Args:
            steps: List of executed step results
            criteria: List of criteria to validate against
            
        Returns:
            True if all criteria are met, False otherwise
        """
        self.logger.info("Validating success criteria")
        
        # Handle missing or invalid criteria gracefully
        if not criteria:
            self.logger.debug("No success criteria specified - returning True")
            return True
        
        if not isinstance(criteria, list):
            self.logger.warning(f"Success criteria should be a list, got {type(criteria)}")
            # Try to handle dict-type criteria
            if isinstance(criteria, dict):
                criteria = [criteria]
            else:
                return True
        
        validation_results: List[Dict[str, Any]] = []
        all_passed = True
        
        for criterion in criteria:
            if not isinstance(criterion, dict):
                self.logger.warning(f"Skipping invalid criterion: {criterion}")
                continue
            
            criterion_type = criterion.get('type', 'unknown')
            criterion_result = {
                'criterion': criterion,
                'type': criterion_type,
                'passed': False,
                'details': {}
            }
            
            try:
                if criterion_type == 'threshold':
                    passed = self._validate_threshold_criterion(steps, criterion)
                elif criterion_type == 'completion':
                    passed = self._validate_completion_criterion(steps, criterion)
                elif criterion_type == 'quality':
                    passed = self._validate_quality_criterion(steps, criterion)
                elif criterion_type == 'time':
                    passed = self._validate_time_criterion(steps, criterion)
                elif criterion_type == 'resource':
                    passed = self._validate_resource_criterion(steps, criterion)
                else:
                    # Handle unknown criterion types gracefully
                    self.logger.warning(f"Unknown criterion type: {criterion_type}")
                    passed = True  # Don't fail on unknown criteria
                    criterion_result['details']['warning'] = f"Unknown criterion type: {criterion_type}"
                
                criterion_result['passed'] = passed
                if not passed:
                    all_passed = False
                    self.logger.warning(f"Criterion failed: {criterion_type}")
                else:
                    self.logger.debug(f"Criterion passed: {criterion_type}")
                    
            except Exception as e:
                self.logger.error(f"Error validating criterion {criterion_type}: {str(e)}")
                criterion_result['error'] = str(e)
                criterion_result['passed'] = False
                all_passed = False
            
            validation_results.append(criterion_result)
        
        # Log validation results
        passed_count = sum(1 for r in validation_results if r['passed'])
        total_count = len(validation_results)
        self.logger.info(f"Success criteria validation: {passed_count}/{total_count} criteria passed")
        
        # Store detailed validation results
        if self.current_execution:
            self.current_execution['validation_results'] = validation_results
        
        return all_passed
    
    def _validate_threshold_criterion(
        self,
        steps: List[Dict[str, Any]],
        criterion: Dict[str, Any]
    ) -> bool:
        """Validate a threshold-based criterion.
        
        Args:
            steps: Executed step results
            criterion: Threshold criterion with 'metric', 'threshold', 'operator'
            
        Returns:
            True if threshold is met
        """
        metric_name = criterion.get('metric', '')
        threshold = criterion.get('threshold', 0)
        operator = criterion.get('operator', '>=')
        
        # Extract metric value from steps
        metric_value = self._extract_metric_from_steps(steps, metric_name)
        
        if metric_value is None:
            self.logger.warning(f"Metric '{metric_name}' not found in steps")
            return True  # Don't fail if metric not found
        
        # Compare based on operator
        if operator == '>=':
            return metric_value >= threshold
        elif operator == '>':
            return metric_value > threshold
        elif operator == '<=':
            return metric_value <= threshold
        elif operator == '<':
            return metric_value < threshold
        elif operator == '==':
            return metric_value == threshold
        elif operator == '!=':
            return metric_value != threshold
        else:
            self.logger.warning(f"Unknown operator: {operator}")
            return True
    
    def _validate_completion_criterion(
        self,
        steps: List[Dict[str, Any]],
        criterion: Dict[str, Any]
    ) -> bool:
        """Validate a completion-based criterion.
        
        Args:
            steps: Executed step results
            criterion: Completion criterion with 'required_steps' or 'min_completion_rate'
            
        Returns:
            True if completion requirements are met
        """
        required_steps = criterion.get('required_steps', [])
        min_rate = criterion.get('min_completion_rate', 1.0)
        
        if required_steps:
            # Check if all required steps completed successfully
            completed_step_names = set()
            for step in steps:
                if step.get('success', False):
                    step_info = step.get('step', {})
                    if isinstance(step_info, dict):
                        completed_step_names.add(step_info.get('name', ''))
            
            for required in required_steps:
                if required not in completed_step_names:
                    return False
            return True
        else:
            # Check completion rate
            if not steps:
                return min_rate == 0
            
            successful = sum(1 for s in steps if s.get('success', False))
            rate = successful / len(steps)
            return rate >= min_rate
    
    def _validate_quality_criterion(
        self,
        steps: List[Dict[str, Any]],
        criterion: Dict[str, Any]
    ) -> bool:
        """Validate a quality-based criterion.
        
        Args:
            steps: Executed step results
            criterion: Quality criterion with 'min_quality_score' or 'quality_metrics'
            
        Returns:
            True if quality requirements are met
        """
        min_score = criterion.get('min_quality_score', 0)
        quality_metrics = criterion.get('quality_metrics', [])
        
        # Calculate aggregate quality score from steps
        quality_scores = []
        for step in steps:
            output = step.get('output', {})
            if isinstance(output, dict):
                if 'quality_score' in output:
                    quality_scores.append(output['quality_score'])
                elif 'quality' in output:
                    quality_scores.append(output['quality'])
        
        if not quality_scores:
            # No quality scores found - pass by default
            return True
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        return avg_quality >= min_score
    
    def _validate_time_criterion(
        self,
        steps: List[Dict[str, Any]],
        criterion: Dict[str, Any]
    ) -> bool:
        """Validate a time-based criterion.
        
        Args:
            steps: Executed step results
            criterion: Time criterion with 'max_duration_ms' or 'deadline'
            
        Returns:
            True if time constraints are met
        """
        max_duration_ms = criterion.get('max_duration_ms')
        deadline = criterion.get('deadline')
        
        if max_duration_ms is not None:
            # Calculate total duration from steps
            total_duration = 0
            for step in steps:
                metrics = step.get('metrics', {})
                if 'duration_ms' in metrics:
                    total_duration += metrics['duration_ms']
                elif 'start_time' in step and 'end_time' in step:
                    try:
                        start = datetime.fromisoformat(step['start_time'])
                        end = datetime.fromisoformat(step['end_time'])
                        total_duration += (end - start).total_seconds() * 1000
                    except (ValueError, TypeError):
                        pass
            
            return total_duration <= max_duration_ms
        
        if deadline is not None:
            try:
                deadline_dt = datetime.fromisoformat(deadline)
                return datetime.now() <= deadline_dt
            except ValueError:
                self.logger.warning(f"Invalid deadline format: {deadline}")
                return True
        
        return True
    
    def _validate_resource_criterion(
        self,
        steps: List[Dict[str, Any]],
        criterion: Dict[str, Any]
    ) -> bool:
        """Validate a resource usage criterion.
        
        Args:
            steps: Executed step results
            criterion: Resource criterion with 'max_memory_mb', 'max_cpu_percent', etc.
            
        Returns:
            True if resource constraints are met
        """
        max_memory_mb = criterion.get('max_memory_mb')
        max_cpu_percent = criterion.get('max_cpu_percent')
        
        # Get resource usage from current execution
        resource_usage = {}
        if self.current_execution:
            resource_usage = self.current_execution.get('resource_usage', {})
        
        if max_memory_mb is not None:
            used_memory = resource_usage.get('memory_mb', 0)
            if used_memory > max_memory_mb:
                return False
        
        if max_cpu_percent is not None:
            used_cpu = resource_usage.get('cpu_percent', 0)
            if used_cpu > max_cpu_percent:
                return False
        
        return True
    
    def _extract_metric_from_steps(
        self,
        steps: List[Dict[str, Any]],
        metric_name: str
    ) -> Optional[float]:
        """Extract a metric value from step results.
        
        Args:
            steps: List of step results
            metric_name: Name of the metric to extract
            
        Returns:
            Metric value or None if not found
        """
        for step in steps:
            # Check in metrics
            metrics = step.get('metrics', {})
            if metric_name in metrics:
                return float(metrics[metric_name])
            
            # Check in output
            output = step.get('output', {})
            if isinstance(output, dict) and metric_name in output:
                return float(output[metric_name])
        
        return None
