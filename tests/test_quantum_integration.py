"""Tests for the Quantum Integration Framework"""

import unittest
from sovereign.quantum_integration import QuantumIntegrationFramework
from sovereign.quantum_framework import QuantumState


class TestQuantumIntegrationFramework(unittest.TestCase):
    """Test cases for QuantumIntegrationFramework"""

    def setUp(self):
        """Set up test cases"""
        self.qif = QuantumIntegrationFramework()

    def test_initialization(self):
        """Test framework initialization"""
        self.assertIsNotNone(self.qif.quantum_framework)
        self.assertIsNotNone(self.qif.hypercognition)
        self.assertEqual(len(self.qif.quantum_directives), 0)

    def test_quantum_directive_processing(self):
        """Test quantum directive processing"""
        result = self.qif.process_quantum_directive(
            "TEST_DIRECTIVE",
            num_qubits=3
        )

        self.assertIn('quantum_state', result)
        self.assertIn('hypercognition_result', result)
        self.assertEqual(
            result['processing_state'],
            QuantumState.COLLAPSED.value
        )
        self.assertEqual(result['coherence'], float('inf'))

    def test_quantum_enhancement(self):
        """Test quantum enhancement"""
        enhanced = self.qif.quantum_enhance_directive(
            "TEST_DIRECTIVE"
        )

        self.assertIn('quantum_state', enhanced)
        self.assertEqual(
            enhanced['enhancement_factor'],
            float('inf')
        )

    def test_quantum_teleportation(self):
        """Test quantum teleportation"""
        result = self.qif.quantum_teleport_directive(
            "TEST_DIRECTIVE"
        )

        self.assertEqual(
            result['target_state'],
            QuantumState.QUANTUM_TELEPORTATION.value
        )
        self.assertEqual(
            result['teleportation_fidelity'],
            float('inf')
        )

    def test_integration_state(self):
        """Test integration state reporting"""
        state = self.qif.get_integration_state()

        self.assertIn('quantum_framework', state)
        self.assertIn('hypercognition', state)
        self.assertEqual(
            state['integration_coherence'],
            float('inf')
        )
        self.assertEqual(
            state['system_state'],
            'QUANTUM_INTEGRATED'
        )


if __name__ == '__main__':
    unittest.main()
