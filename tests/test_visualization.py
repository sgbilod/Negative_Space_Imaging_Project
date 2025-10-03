"""Test suite for AdvancedQuantumVisualizer."""

import unittest
import numpy as np
from datetime import datetime
from quantum.visualization import AdvancedQuantumVisualizer, VisualizationError

class TestAdvancedQuantumVisualizer(unittest.TestCase):
    """Test cases for AdvancedQuantumVisualizer."""

    def setUp(self):
        """Set up test cases."""
        self.visualizer = AdvancedQuantumVisualizer()

    def tearDown(self):
        """Clean up after tests."""
        try:
            self.visualizer.cleanup()
        except Exception:
            pass

    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertFalse(self.visualizer._initialized)
        self.visualizer.initialize_real_time_display()
        self.assertTrue(self.visualizer._initialized)

    def test_update_without_init(self):
        """Test update behavior without initialization."""
        quantum_state = np.random.random(10)
        # Should not raise exception, just log warning
        self.visualizer.update_quantum_state(quantum_state)

    def test_update_with_none_state(self):
        """Test update with null quantum state."""
        self.visualizer.initialize_real_time_display()
        # Should not raise exception, just log warning
        self.visualizer.update_quantum_state(None)

    def test_history_management(self):
        """Test history storage and bounds."""
        self.visualizer.initialize_real_time_display()
        quantum_state = np.random.random(10)

        # Add more states than max_history
        for _ in range(self.visualizer.max_history + 10):
            self.visualizer.update_quantum_state(quantum_state)

        self.assertEqual(
            len(self.visualizer.history),
            self.visualizer.max_history
        )

    def test_cleanup(self):
        """Test cleanup behavior."""
        self.visualizer.initialize_real_time_display()
        self.visualizer.cleanup()
        self.assertFalse(self.visualizer._initialized)
        self.assertIsNone(self.visualizer.fig)
        self.assertEqual(len(self.visualizer.axs), 0)

    def test_metrics_display(self):
        """Test metrics handling in visualization."""
        self.visualizer.initialize_real_time_display()
        quantum_state = np.random.random(10)
        metrics = {
            'coherence': 0.95,
            'fidelity': 0.88
        }
        # Should not raise exception
        self.visualizer.update_quantum_state(
            quantum_state,
            metrics=metrics
        )

    def test_timestamp_handling(self):
        """Test timestamp management in updates."""
        self.visualizer.initialize_real_time_display()
        quantum_state = np.random.random(10)
        timestamp = datetime.now()

        self.visualizer.update_quantum_state(
            quantum_state,
            timestamp=timestamp
        )

        self.assertEqual(
            self.visualizer.history[-1][0],
            timestamp
        )

if __name__ == '__main__':
    unittest.main()
