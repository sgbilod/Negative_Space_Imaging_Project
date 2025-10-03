"""Tests for the Hypercognition Directive System"""

import unittest
import numpy as np
from sovereign.hypercognition import (
    HypercognitionDirectiveSystem,
    CognitionMode,
    DirectiveState,
    DirectiveContext
)


class TestHypercognitionDirectiveSystem(unittest.TestCase):
    """Test cases for HypercognitionDirectiveSystem"""

    def setUp(self):
        """Set up test cases"""
        self.system = HypercognitionDirectiveSystem()

    def test_initialization(self):
        """Test system initialization"""
        self.assertEqual(self.system.dimensions, 1000)
        self.assertEqual(self.system.current_state, DirectiveState.PROCESSING)
        self.assertEqual(self.system.mode, CognitionMode.QUANTUM)

    def test_context_initialization(self):
        """Test context map initialization"""
        context = self.system._initialize_context()
        self.assertIsInstance(context, DirectiveContext)
        self.assertEqual(context.temporal_coordinates['now'], float('inf'))
        self.assertEqual(context.spatial_dimensions['quantum'], float('inf'))
        self.assertTrue(np.array_equal(
            context.quantum_states['primary'],
            self.system.quantum_field
        ))

    def test_directive_processing(self):
        """Test directive processing workflow"""
        test_directive = "TEST_DIRECTIVE"
        result = self.system.process_directive(test_directive)

        self.assertIsInstance(result, dict)
        self.assertEqual(self.system.current_state, DirectiveState.COMPLETED)

    def test_quantum_enhancement(self):
        """Test quantum enhancement capabilities"""
        test_data = "TEST_DATA"
        enhanced = self.system._apply_quantum_enhancement(test_data)

        self.assertEqual(enhanced['data'], test_data)
        self.assertEqual(enhanced['quantum_factor'], float('inf'))
        self.assertEqual(enhanced['coherence'], float('inf'))
        self.assertEqual(enhanced['entanglement'], float('inf'))

    def test_system_state(self):
        """Test system state reporting"""
        state = self.system.get_system_state()

        expected_state = DirectiveState.PROCESSING.value
        self.assertEqual(state['current_state'], expected_state)
        self.assertEqual(state['cognition_mode'], CognitionMode.QUANTUM.value)
        self.assertEqual(state['quantum_coherence'], float('inf'))
        self.assertEqual(state['neural_coherence'], float('inf'))
        self.assertEqual(state['temporal_coherence'], float('inf'))


if __name__ == '__main__':
    unittest.main()
