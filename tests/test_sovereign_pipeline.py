"""Test suite for Sovereign Pipeline components."""

import unittest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from sovereign.quantum_state import QuantumState
from sovereign.quantum_engine import QuantumEngine
from sovereign.pipeline.implementation import SovereignImplementationPipeline

class TestQuantumState(unittest.TestCase):
    """Test cases for QuantumState."""

    def setUp(self):
        """Set up test cases."""
        self.state = QuantumState(dimensions=1024)

    def test_initialization(self):
        """Test quantum state initialization."""
        self.assertEqual(self.state.dimensions, 1024)
        self.assertEqual(self.state.wave_factor, 1.0)
        self.assertEqual(self.state.state_matrix.shape, (1000, 1000))
        self.assertFalse(self.state.initialized)

    def test_invalid_dimensions(self):
        """Test initialization with invalid dimensions."""
        with self.assertRaises(ValueError):
            QuantumState(dimensions=-1)
        with self.assertRaises(ValueError):
            QuantumState(dimensions=2e6)

    def test_state_verification(self):
        """Test state verification."""
        self.assertFalse(self.state.verify_state())  # Not initialized
        self.state.establish_sovereign_state()
        self.assertTrue(self.state.verify_state())

    def test_reset_to_baseline(self):
        """Test reset functionality."""
        self.state.establish_sovereign_state()
        self.state.reset_to_baseline()
        self.assertFalse(self.state.initialized)
        self.assertEqual(self.state.wave_factor, 1.0)
        np.testing.assert_array_equal(self.state.state_matrix, np.zeros((1000, 1000)))

class TestQuantumEngine(unittest.TestCase):
    """Test cases for QuantumEngine."""

    def setUp(self):
        """Set up test cases."""
        self.engine = QuantumEngine(dimensions=1024)

    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.dimensions, 1024)
        self.assertFalse(self.engine.running)
        self.assertEqual(self.engine.quantum_field['field_strength'], 0.0)

    def test_invalid_dimensions(self):
        """Test initialization with invalid dimensions."""
        with self.assertRaises(ValueError):
            QuantumEngine(dimensions=-1)
        with self.assertRaises(ValueError):
            QuantumEngine(dimensions=2e6)

    def test_start_stop(self):
        """Test engine start/stop cycle."""
        self.assertTrue(self.engine.start())
        self.assertTrue(self.engine.running)
        self.assertEqual(self.engine.quantum_field['field_strength'], 1.0)

        self.assertTrue(self.engine.stop())
        self.assertFalse(self.engine.running)
        self.assertEqual(self.engine.quantum_field['field_strength'], 0.0)

    def test_emergency_reset(self):
        """Test emergency reset functionality."""
        self.engine.start()
        self.engine.emergency_reset()
        self.assertFalse(self.engine.running)
        self.assertEqual(self.engine.quantum_field['field_strength'], 0.0)

class TestSovereignPipeline(unittest.TestCase):
    """Test cases for SovereignImplementationPipeline."""

    @patch('sovereign.pipeline.implementation.QuantumProcessor')
    @patch('sovereign.pipeline.implementation.RealityManipulator')
    @patch('sovereign.pipeline.implementation.MasterController')
    def setUp(self, mock_controller, mock_manipulator, mock_processor):
        """Set up test cases with mocked components."""
        self.pipeline = SovereignImplementationPipeline()
        self.mock_processor = mock_processor
        self.mock_manipulator = mock_manipulator
        self.mock_controller = mock_controller

    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertIsInstance(self.pipeline.project_root, Path)
        self.assertTrue(hasattr(self.pipeline, 'logger'))

    def test_activation(self):
        """Test pipeline activation sequence."""
        # Configure mocks
        self.pipeline.quantum_processor.verify_state.return_value = True
        self.pipeline.reality_manipulator.verify_state.return_value = True
        self.pipeline.master_controller.verify_state.return_value = True

        # Test activation
        self.pipeline.activate()

        # Verify component initialization
        self.pipeline.quantum_processor.initialize.assert_called_once()
        self.pipeline.reality_manipulator.configure.assert_called_once()
        self.pipeline.master_controller.start.assert_called_once()

    def test_activation_failure(self):
        """Test pipeline activation failure handling."""
        # Configure mock to fail
        self.pipeline.quantum_processor.verify_state.return_value = False

        # Test activation failure
        with self.assertRaises(RuntimeError):
            self.pipeline.activate()

        # Verify emergency shutdown was called
        self.pipeline.quantum_processor.emergency_reset.assert_called_once()

    def test_emergency_shutdown(self):
        """Test emergency shutdown procedure."""
        self.pipeline._emergency_shutdown()

        # Verify all components were properly stopped
        self.pipeline.master_controller.emergency_stop.assert_called_once()
        self.pipeline.quantum_processor.emergency_reset.assert_called_once()
        self.pipeline.reality_manipulator.reset.assert_called_once()

if __name__ == '__main__':
    unittest.main()
