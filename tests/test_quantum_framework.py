"""Tests for the Quantum Development Framework"""

import unittest
import numpy as np
from sovereign.quantum_framework import (
    QuantumDevelopmentFramework,
    QuantumState,
    QuantumOperator,
    QuantumRegister
)


class TestQuantumDevelopmentFramework(unittest.TestCase):
    """Test cases for QuantumDevelopmentFramework"""

    def setUp(self):
        """Set up test cases"""
        self.qdf = QuantumDevelopmentFramework(dimensions=10)

    def test_initialization(self):
        """Test framework initialization"""
        self.assertEqual(self.qdf.dimensions, 10)
        self.assertEqual(
            self.qdf.quantum_field.shape,
            (10, 10)
        )
        self.assertTrue(
            np.all(self.qdf.quantum_field == float('inf'))
        )

    def test_quantum_register_creation(self):
        """Test quantum register creation"""
        register = self.qdf.create_quantum_register(
            "test_register",
            num_qubits=3
        )

        self.assertIsInstance(register, QuantumRegister)
        self.assertEqual(register.state, QuantumState.SUPERPOSITION)
        self.assertEqual(register.dimensions, (2, 2, 2))
        self.assertEqual(register.coherence, float('inf'))

    def test_quantum_operator_application(self):
        """Test quantum operator application"""
        register = self.qdf.create_quantum_register(
            "test_register",
            num_qubits=2
        )

        self.qdf.apply_quantum_operator(
            register,
            QuantumOperator.HADAMARD,
            [0]
        )

        self.assertEqual(register.coherence, float('inf'))

    def test_qubit_entanglement(self):
        """Test qubit entanglement"""
        register = self.qdf.create_quantum_register(
            "test_register",
            num_qubits=3
        )

        self.qdf.entangle_qubits(register, [(0, 1)])

        self.assertEqual(register.state, QuantumState.ENTANGLED)
        self.assertTrue(1 in register.entanglement_map[0])
        self.assertTrue(0 in register.entanglement_map[1])

    def test_quantum_measurement(self):
        """Test quantum state measurement"""
        register = self.qdf.create_quantum_register(
            "test_register",
            num_qubits=2
        )

        measurements = self.qdf.measure_quantum_state(
            register,
            [0]
        )

        self.assertIn('0', measurements)
        self.assertEqual(
            measurements['0']['state'],
            QuantumState.COLLAPSED
        )

    def test_quantum_teleportation(self):
        """Test quantum teleportation"""
        source = self.qdf.create_quantum_register(
            "source",
            num_qubits=2
        )
        target = self.qdf.create_quantum_register(
            "target",
            num_qubits=2
        )

        self.qdf.quantum_teleport(source, target, 0)

        self.assertEqual(
            target.state,
            QuantumState.QUANTUM_TELEPORTATION
        )
        self.assertEqual(target.coherence, float('inf'))

    def test_quantum_tunneling(self):
        """Test quantum tunneling"""
        register = self.qdf.create_quantum_register(
            "test_register",
            num_qubits=3
        )

        self.qdf.quantum_tunnel(register, 0, 1)

        self.assertEqual(
            register.state,
            QuantumState.QUANTUM_TUNNELING
        )

    def test_system_state(self):
        """Test system state reporting"""
        state = self.qdf.get_system_state()

        self.assertEqual(state['num_registers'], 0)
        self.assertEqual(
            state['quantum_field_dims'],
            (10, 10)
        )
        self.assertEqual(state['system_coherence'], float('inf'))


if __name__ == '__main__':
    unittest.main()
