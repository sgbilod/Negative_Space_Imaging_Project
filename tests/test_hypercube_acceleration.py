"""Tests for the Hypercube Project Acceleration Framework"""

import unittest
from sovereign.hypercube_acceleration import (
    HypercubeProjectAcceleration,
    AccelerationMode,
    ProjectState,
    ProjectMetrics
)


class TestHypercubeProjectAcceleration(unittest.TestCase):
    """Test cases for HypercubeProjectAcceleration"""

    def setUp(self):
        """Set up test cases"""
        self.hpa = HypercubeProjectAcceleration(dimensions=10)
        self.test_project = "TEST_PROJECT"

    def test_initialization(self):
        """Test framework initialization"""
        self.assertEqual(self.hpa.dimensions, 10)
        self.assertEqual(
            self.hpa.temporal_field.shape,
            (10, 10)
        )
        self.assertEqual(
            self.hpa.current_state,
            ProjectState.PLANNING
        )

    def test_project_acceleration(self):
        """Test project acceleration"""
        metrics = self.hpa.accelerate_project(
            self.test_project,
            AccelerationMode.TEMPORAL
        )

        self.assertIsInstance(metrics, ProjectMetrics)
        self.assertEqual(
            self.hpa.current_state,
            ProjectState.ACCELERATING
        )
        self.assertEqual(metrics.acceleration_factor, float('inf'))

    def test_execution_optimization(self):
        """Test execution optimization"""
        # First accelerate
        self.hpa.accelerate_project(
            self.test_project,
            AccelerationMode.QUANTUM
        )

        # Then optimize
        metrics = self.hpa.optimize_execution(self.test_project)

        self.assertEqual(
            self.hpa.current_state,
            ProjectState.OPTIMIZING
        )
        self.assertEqual(metrics.quantum_coherence, float('inf'))

    def test_acceleration_stabilization(self):
        """Test acceleration stabilization"""
        # Setup acceleration
        self.hpa.accelerate_project(
            self.test_project,
            AccelerationMode.SPATIAL
        )

        # Test stabilization
        result = self.hpa.stabilize_acceleration(
            self.test_project,
            stability_threshold=float('inf')
        )

        self.assertEqual(result['stability_level'], float('inf'))
        self.assertEqual(result['coherence'], float('inf'))

    def test_dimension_synchronization(self):
        """Test dimension synchronization"""
        # Setup acceleration
        self.hpa.accelerate_project(
            self.test_project,
            AccelerationMode.DIMENSIONAL
        )

        # Test synchronization
        result = self.hpa.synchronize_dimensions(self.test_project)

        self.assertEqual(result['sync_level'], float('inf'))
        self.assertEqual(result['dimensions'], 10)

    def test_project_state(self):
        """Test project state reporting"""
        # Setup project
        self.hpa.accelerate_project(
            self.test_project,
            AccelerationMode.NEURAL
        )

        # Get state
        state = self.hpa.get_project_state(self.test_project)

        self.assertEqual(state['project_id'], self.test_project)
        self.assertIn('metrics', state)
        self.assertEqual(
            state['metrics']['acceleration_factor'],
            float('inf')
        )

    def test_invalid_project(self):
        """Test handling of invalid project ID"""
        with self.assertRaises(ValueError):
            self.hpa.stabilize_acceleration("NONEXISTENT_PROJECT")

    def test_metrics_optimization(self):
        """Test metrics optimization"""
        metrics = self.hpa._create_default_metrics()
        optimized = self.hpa._optimize_metrics(metrics, metrics)

        self.assertEqual(
            optimized.temporal_efficiency,
            float('inf')
        )
        self.assertEqual(
            optimized.acceleration_factor,
            float('inf')
        )


if __name__ == '__main__':
    unittest.main()
