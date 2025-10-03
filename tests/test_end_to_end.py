"""End-to-end test suite for the Negative Space Imaging System."""

import unittest
import tempfile
from pathlib import Path
import logging
import numpy as np
import time

from sovereign.pipeline.implementation import SovereignImplementationPipeline
from sovereign.quantum_state import QuantumState
from sovereign.quantum_engine import QuantumEngine
from quantum.demo import execute_test_cases
from sovereign.control_mode import ControlMode

class EndToEndTests(unittest.TestCase):
    """End-to-end integration tests."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger("EndToEndTest")

        # Create temp directory for test artifacts
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.logger.info(f"Test artifacts directory: {cls.test_dir}")

    def setUp(self):
        """Set up individual test cases."""
        self.pipeline = SovereignImplementationPipeline()

    def test_full_pipeline_execution(self):
        """Test complete pipeline execution."""
        # 1. Initialize pipeline
        self.pipeline.activate()

        # 2. Create test objectives
        objectives = [
            "Initialize quantum state",
            "Apply sovereign transformations",
            "Verify state integrity"
        ]

        # 3. Set up resources
        resources = {
            "quantum_state": QuantumState(dimensions=1024),
            "quantum_engine": QuantumEngine(dimensions=1024),
            "output_path": self.test_dir / "results.json"
        }

        # 4. Execute pipeline
        result = self.pipeline.execute_task(
            objectives=objectives,
            resources=resources
        )

        # 5. Verify results
        self.assertTrue(result.get("success", False))
        self.assertGreaterEqual(result.get("completion_percentage", 0), 95)

    def test_quantum_state_integration(self):
        """Test quantum state integration with engine."""
        state = QuantumState(dimensions=1024)
        engine = QuantumEngine(dimensions=1024)

        # 1. Initialize components
        engine.start()
        state.initialize()

        # 2. Apply transformations
        test_matrix = np.random.random((1000, 1000))
        state.apply_stability_matrix(test_matrix)

        # 3. Verify states
        self.assertTrue(state.verify_state())
        self.assertTrue(engine.verify_engine_state())

    def test_visualization_integration(self):
        """Test visualization system integration."""
        from quantum.visualization import AdvancedQuantumVisualizer

        # 1. Create test data
        quantum_state = np.random.random((1000, 1000))
        metrics = {
            "coherence": 0.95,
            "stability": 0.98
        }

        # 2. Initialize visualizer
        viz = AdvancedQuantumVisualizer()
        viz.initialize_real_time_display()

        # 3. Update and verify
        try:
            viz.update_quantum_state(quantum_state, metrics=metrics)
            time.sleep(1)  # Allow visualization to render
        finally:
            viz.cleanup()

    def test_sovereign_mode_execution(self):
        """Test execution in SOVEREIGN mode."""
        # 1. Set up controller
        from sovereign.master_controller import MasterController

        controller = MasterController(
            mode=ControlMode.SOVEREIGN,
            project_root=Path(__file__).parent.parent
        )

        # 2. Start controller
        controller.start()

        # 3. Execute autonomous operation
        result = controller.execute_autonomous_sequence([
            "quantum_state_initialization",
            "reality_manipulation",
            "sovereign_transformation"
        ])

        # 4. Verify autonomous execution
        self.assertTrue(result.get("autonomous_execution_success", False))
        self.assertGreaterEqual(
            result.get("autonomous_completion_percentage", 0),
            95
        )

    def test_error_handling_and_recovery(self):
        """Test system error handling and recovery."""
        # 1. Create invalid state
        state = QuantumState(dimensions=1024)
        state.wave_factor = -1  # Invalid value

        # 2. Attempt recovery
        engine = QuantumEngine(dimensions=1024)
        engine.start()

        try:
            # This should fail but recover
            state.establish_sovereign_state()
            self.fail("Should have raised ValueError")
        except ValueError:
            # 3. Verify recovery
            state.reset_to_baseline()
            state.initialize()
            self.assertTrue(state.verify_state())

    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'pipeline'):
            try:
                self._emergency_shutdown()
            except Exception as e:
                self.logger.warning(f"Teardown warning: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Clean up temp directory
        try:
            import shutil
            shutil.rmtree(cls.test_dir)
        except Exception as e:
            cls.logger.warning(f"Cleanup warning: {e}")

if __name__ == '__main__':
    unittest.main()
