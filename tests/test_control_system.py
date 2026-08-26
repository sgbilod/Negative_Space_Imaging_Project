"""Tests for the Sovereign Control System"""

import unittest
from sovereign.control_system import (
    SovereignControlSystem,
    IntegrationMode,
    SystemState,
    SystemMetrics,
    ExecutionContext
)


class TestSovereignControlSystem(unittest.TestCase):
    """Test cases for SovereignControlSystem"""

    def setUp(self):
        """Set up test cases"""
        self.scs = SovereignControlSystem()
        self.test_control = "TEST_CONTROL"

    def test_initialization(self):
        """Test system initialization"""
        context = self.scs.initialize_sovereign_control(
            self.test_control
        )

        self.assertIsInstance(context, ExecutionContext)
        self.assertEqual(
            context.mode,
            IntegrationMode.AUTONOMOUS
        )
        self.assertEqual(
            self.scs.current_state,
            SystemState.INTEGRATING
        )

    def test_directive_execution(self):
        """Test directive execution"""
        # Initialize first
        self.scs.initialize_sovereign_control(self.test_control)

        # Execute directive
        result = self.scs.execute_sovereign_directive(
            self.test_control,
            "TEST_DIRECTIVE"
        )

        self.assertEqual(
            self.scs.current_state,
            SystemState.EXECUTING
        )
        self.assertEqual(
            result['integration_factor'],
            float('inf')
        )

    def test_execution_optimization(self):
        """Test execution optimization"""
        # Initialize first
        self.scs.initialize_sovereign_control(self.test_control)

        # Execute something
        self.scs.execute_sovereign_directive(
            self.test_control,
            "TEST_DIRECTIVE"
        )

        # Optimize
        metrics = self.scs.optimize_sovereign_execution(
            self.test_control
        )

        self.assertEqual(
            self.scs.current_state,
            SystemState.OPTIMIZING
        )
        self.assertEqual(
            metrics.quantum_coherence,
            float('inf')
        )

    def test_system_state(self):
        """Test system state reporting"""
        # Initialize first
        self.scs.initialize_sovereign_control(self.test_control)

        # Get state
        state = self.scs.get_sovereign_state(self.test_control)

        self.assertEqual(state['control_id'], self.test_control)
        self.assertIn('metrics', state)
        self.assertEqual(
            state['metrics']['autonomy_level'],
            float('inf')
        )

    def test_invalid_control(self):
        """Test handling of invalid control ID"""
        with self.assertRaises(ValueError):
            self.scs.get_sovereign_state("NONEXISTENT_CONTROL")

    def test_metrics_optimization(self):
        """Test metrics optimization"""
        # Initialize first
        context = self.scs.initialize_sovereign_control(
            self.test_control
        )

        # Create target metrics
        target_metrics = SystemMetrics(
            quantum_coherence=float('inf'),
            hypercognition_factor=float('inf'),
            acceleration_rate=float('inf'),
            integration_stability=float('inf'),
            autonomy_level=float('inf'),
            execution_efficiency=float('inf')
        )

        # Optimize
        optimized = self.scs._optimize_system_metrics(
            context.metrics,
            target_metrics,
            {},
            {},
            None
        )

        self.assertEqual(
            optimized.quantum_coherence,
            float('inf')
        )
        self.assertEqual(
            optimized.execution_efficiency,
            float('inf')
        )


if __name__ == '__main__':
    unittest.main()
