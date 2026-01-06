"""
Qiskit Quantum Integration Module for Negative Space Imaging

Comprehensive Qiskit integration providing:
- IBM Quantum account configuration and authentication
- Circuit construction and transpilation utilities
- Job submission to simulators and real hardware
- Result retrieval and parsing
- Advanced error handling for queue, timeout, and connectivity
- Support for multiple backends (QASM Simulator, real hardware)
- Circuit optimization with configurable transpilation levels

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit import (
    IBMQ,
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile,
    execute,
)
from qiskit.providers import Backend, Job
from qiskit.providers.ibmq import IBMQBackend, IBMQJob
from qiskit.providers.fake import FakeSydney, FakeHanoi, FakeJakarta
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Estimator
from qiskit.tools import job_monitor
from qiskit.result import Result
from qiskit.exceptions import QiskitError

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class QiskitEnvironmentManager:
    """Manages Qiskit environment configuration and IBM Quantum authentication."""

    def __init__(self) -> None:
        """Initialize Qiskit environment manager."""
        self.ibm_token: Optional[str] = None
        self.service: Optional[QiskitRuntimeService] = None
        self.authenticated: bool = False
        self._initialize_environment()

    def _initialize_environment(self) -> None:
        """Initialize and configure Qiskit environment."""
        logger.info("Initializing Qiskit environment manager...")

        # Get IBM Quantum token from environment
        self.ibm_token = os.getenv("IBM_QUANTUM_TOKEN")

        if not self.ibm_token:
            logger.warning(
                "IBM_QUANTUM_TOKEN not found in environment. "
                "Simulator-only mode will be available."
            )
        else:
            self._authenticate_ibm()

    def _authenticate_ibm(self) -> None:
        """Authenticate with IBM Quantum platform."""
        try:
            logger.info("Authenticating with IBM Quantum platform...")
            # Save token and initialize service
            QiskitRuntimeService.save_account(
                channel="ibm_quantum",
                token=self.ibm_token,
                overwrite=True
            )
            self.service = QiskitRuntimeService(channel="ibm_quantum")
            self.authenticated = True
            logger.info("Successfully authenticated with IBM Quantum")
        except Exception as e:
            logger.error(f"Failed to authenticate with IBM Quantum: {e}")
            self.authenticated = False

    def get_service(self) -> Optional[QiskitRuntimeService]:
        """Get QiskitRuntimeService instance."""
        return self.service

    def is_authenticated(self) -> bool:
        """Check if authenticated with IBM Quantum."""
        return self.authenticated


class CircuitBuilder:
    """Utility class for building and optimizing quantum circuits."""

    def __init__(self, num_qubits: int, num_cbits: int = 0) -> None:
        """
        Initialize circuit builder.

        Args:
            num_qubits: Number of qubits in circuit
            num_cbits: Number of classical bits (defaults to num_qubits)
        """
        self.num_qubits = num_qubits
        self.num_cbits = num_cbits or num_qubits
        self.circuit = self._create_circuit()

    def _create_circuit(self) -> QuantumCircuit:
        """Create base quantum circuit."""
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(self.num_cbits, "c")
        return QuantumCircuit(qr, cr)

    def add_barrier(self) -> None:
        """Add barrier to circuit for visualization."""
        self.circuit.barrier()

    def measure_all(self) -> None:
        """Add measurement of all qubits to classical bits."""
        self.circuit.measure(range(self.num_qubits), range(self.num_cbits))

    def get_circuit(self) -> QuantumCircuit:
        """Get the constructed circuit."""
        return self.circuit

    def circuit_depth(self) -> int:
        """Get circuit depth."""
        return self.circuit.depth()

    def circuit_size(self) -> int:
        """Get circuit size (number of gates)."""
        return self.circuit.size()

    def circuit_width(self) -> int:
        """Get circuit width."""
        return self.circuit.width()

    def to_string(self) -> str:
        """Convert circuit to string representation."""
        return str(self.circuit)


class TranspilerConfig:
    """Configuration for circuit transpilation."""

    def __init__(
        self,
        optimization_level: int = 3,
        layout_method: str = "sabre",
        routing_method: str = "sabre",
        seed_transpiler: Optional[int] = None,
    ) -> None:
        """
        Initialize transpiler configuration.

        Args:
            optimization_level: 0-3, higher = more aggressive optimization
            layout_method: Circuit layout method (sabre, dense, trivial)
            routing_method: Routing method (sabre, basic, stochastic)
            seed_transpiler: Random seed for reproducibility
        """
        self.optimization_level = optimization_level
        self.layout_method = layout_method
        self.routing_method = routing_method
        self.seed_transpiler = seed_transpiler
        logger.debug(f"TranspilerConfig created: {self}")

    def __str__(self) -> str:
        """String representation."""
        return (
            f"TranspilerConfig(opt_level={self.optimization_level}, "
            f"layout={self.layout_method}, routing={self.routing_method})"
        )


class CircuitTranspiler:
    """Transpiles circuits for target backends."""

    def __init__(self, config: Optional[TranspilerConfig] = None) -> None:
        """
        Initialize transpiler.

        Args:
            config: TranspilerConfig instance
        """
        self.config = config or TranspilerConfig()

    def transpile_for_backend(
        self,
        circuit: QuantumCircuit,
        backend: Optional[Backend] = None,
    ) -> QuantumCircuit:
        """
        Transpile circuit for target backend.

        Args:
            circuit: Input circuit
            backend: Target backend (optional)

        Returns:
            Transpiled circuit
        """
        try:
            logger.info(
                f"Transpiling circuit with {self.config.optimization_level} "
                f"optimization level..."
            )

            transpiled = transpile(
                circuit,
                backend=backend,
                optimization_level=self.config.optimization_level,
                layout_method=self.config.layout_method,
                routing_method=self.config.routing_method,
                seed_transpiler=self.config.seed_transpiler,
            )

            logger.debug(
                f"Transpilation complete. "
                f"Depth: {transpiled.depth()}, Size: {transpiled.size()}"
            )

            return transpiled
        except Exception as e:
            logger.error(f"Circuit transpilation failed: {e}")
            raise

    def measure_transpilation_impact(
        self, original: QuantumCircuit, transpiled: QuantumCircuit
    ) -> Dict[str, int]:
        """
        Measure transpilation impact on circuit.

        Args:
            original: Original circuit
            transpiled: Transpiled circuit

        Returns:
            Dictionary with depth and size changes
        """
        return {
            "original_depth": original.depth(),
            "transpiled_depth": transpiled.depth(),
            "original_size": original.size(),
            "transpiled_size": transpiled.size(),
            "depth_reduction": original.depth() - transpiled.depth(),
            "size_reduction": original.size() - transpiled.size(),
        }


class BackendManager:
    """Manages quantum backends (simulators and real hardware)."""

    def __init__(self, env_manager: QiskitEnvironmentManager) -> None:
        """
        Initialize backend manager.

        Args:
            env_manager: QiskitEnvironmentManager instance
        """
        self.env_manager = env_manager
        self.simulators: Dict[str, Backend] = {}
        self.real_backends: Dict[str, IBMQBackend] = {}
        self._initialize_simulators()
        self._initialize_real_backends()

    def _initialize_simulators(self) -> None:
        """Initialize local simulators."""
        try:
            logger.info("Initializing simulators...")

            # Aer QASM Simulator (default, noiseless)
            self.simulators["qasm_simulator"] = AerSimulator(
                method="statevector",
                max_qubits=30
            )

            # Aer Simulator with noise model support
            self.simulators["aer_simulator_with_noise"] = AerSimulator(
                method="density_matrix",
                max_qubits=20
            )

            # Fake backends for circuit compilation testing
            self.simulators["fake_sydney"] = FakeSydney()
            self.simulators["fake_hanoi"] = FakeHanoi()
            self.simulators["fake_jakarta"] = FakeJakarta()

            logger.info(
                f"Initialized {len(self.simulators)} simulators/fake backends"
            )
        except Exception as e:
            logger.error(f"Failed to initialize simulators: {e}")

    def _initialize_real_backends(self) -> None:
        """Initialize real IBM Quantum backends."""
        if not self.env_manager.is_authenticated():
            logger.warning("Not authenticated - real backends unavailable")
            return

        try:
            logger.info("Loading available IBM Quantum backends...")
            service = self.env_manager.get_service()
            if service:
                # Get available backends
                backends = service.backends()
                for backend in backends[:5]:  # Limit to first 5
                    self.real_backends[backend.name] = backend
                    logger.info(f"  Available: {backend.name}")
        except Exception as e:
            logger.error(f"Failed to load real backends: {e}")

    def get_backend(self, backend_name: str) -> Optional[Backend]:
        """
        Get backend by name.

        Args:
            backend_name: Backend identifier

        Returns:
            Backend instance or None
        """
        if backend_name in self.simulators:
            return self.simulators[backend_name]
        elif backend_name in self.real_backends:
            return self.real_backends[backend_name]
        else:
            logger.warning(f"Backend '{backend_name}' not found")
            return None

    def list_available_backends(self) -> Dict[str, List[str]]:
        """List all available backends."""
        return {
            "simulators": list(self.simulators.keys()),
            "real_backends": list(self.real_backends.keys()),
        }


class JobSubmissionManager:
    """Manages job submission and result retrieval."""

    def __init__(
        self,
        backend: Backend,
        transpiler: Optional[CircuitTranspiler] = None,
    ) -> None:
        """
        Initialize job submission manager.

        Args:
            backend: Target backend
            transpiler: CircuitTranspiler instance
        """
        self.backend = backend
        self.transpiler = transpiler or CircuitTranspiler()
        self.jobs: Dict[str, Job] = {}

    def submit_job(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        job_name: Optional[str] = None,
        max_credits: int = 10,
    ) -> Tuple[str, Job]:
        """
        Submit circuit as job to backend.

        Args:
            circuit: Quantum circuit
            shots: Number of measurement shots
            job_name: Optional job name
            max_credits: Maximum credits for real hardware

        Returns:
            Tuple of (job_id, Job instance)
        """
        try:
            logger.info(f"Submitting job to {self.backend.name}...")

            # Transpile for backend
            transpiled = self.transpiler.transpile_for_backend(circuit, self.backend)

            # Submit job
            job = execute(
                transpiled,
                self.backend,
                shots=shots,
                job_name=job_name or f"nsip_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                max_credits=max_credits,
            )

            job_id = job.job_id()
            self.jobs[job_id] = job

            logger.info(f"Job submitted. ID: {job_id}")
            return job_id, job

        except Exception as e:
            logger.error(f"Job submission failed: {e}")
            raise

    def get_job_status(self, job_id: str) -> Optional[str]:
        """
        Get job status.

        Args:
            job_id: Job identifier

        Returns:
            Job status string
        """
        try:
            if job_id in self.jobs:
                return self.jobs[job_id].status().name
            return None
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return None

    def get_job_result(
        self,
        job_id: str,
        timeout: int = 3600,
        wait: bool = True,
    ) -> Optional[Result]:
        """
        Retrieve job result.

        Args:
            job_id: Job identifier
            timeout: Timeout in seconds
            wait: Wait for job completion

        Returns:
            Result instance or None
        """
        try:
            if job_id not in self.jobs:
                logger.error(f"Job {job_id} not found")
                return None

            job = self.jobs[job_id]

            if wait:
                logger.info(f"Waiting for job {job_id} to complete...")
                job_monitor(job, interval=5)

            result = job.result()
            logger.info(f"Job result retrieved successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to retrieve job result: {e}")
            return None

    def monitor_job(self, job_id: str, interval: int = 5) -> None:
        """
        Monitor job execution.

        Args:
            job_id: Job identifier
            interval: Check interval in seconds
        """
        if job_id in self.jobs:
            job_monitor(self.jobs[job_id], interval=interval)
        else:
            logger.error(f"Job {job_id} not found")


class ResultParser:
    """Parses and processes quantum results."""

    @staticmethod
    def parse_counts(result: Result) -> Dict[str, Any]:
        """
        Parse measurement counts from result.

        Args:
            result: Qiskit Result instance

        Returns:
            Dictionary with count statistics
        """
        try:
            counts = result.get_counts(0)

            # Calculate statistics
            total_shots = sum(counts.values())
            probabilities = {
                state: count / total_shots for state, count in counts.items()
            }

            return {
                "counts": counts,
                "probabilities": probabilities,
                "total_shots": total_shots,
                "num_states": len(counts),
            }
        except Exception as e:
            logger.error(f"Result parsing failed: {e}")
            return {}

    @staticmethod
    def parse_statevector(result: Result) -> Optional[np.ndarray]:
        """
        Parse statevector from result.

        Args:
            result: Qiskit Result instance

        Returns:
            Statevector as numpy array
        """
        try:
            return np.array(result.get_statevector(0))
        except Exception as e:
            logger.error(f"Statevector parsing failed: {e}")
            return None

    @staticmethod
    def parse_expectation_value(result: Result) -> Optional[float]:
        """
        Parse expectation value from result.

        Args:
            result: Qiskit Result instance

        Returns:
            Expectation value
        """
        try:
            return result.data(0).evdata.evs
        except Exception as e:
            logger.error(f"Expectation value parsing failed: {e}")
            return None


class QiskitQuantumProcessor:
    """Main quantum processor orchestrating all Qiskit operations."""

    def __init__(
        self,
        backend_name: str = "qasm_simulator",
        transpiler_config: Optional[TranspilerConfig] = None,
    ) -> None:
        """
        Initialize quantum processor.

        Args:
            backend_name: Backend identifier
            transpiler_config: TranspilerConfig instance
        """
        self.env_manager = QiskitEnvironmentManager()
        self.transpiler = CircuitTranspiler(transpiler_config)
        self.backend_manager = BackendManager(self.env_manager)

        backend = self.backend_manager.get_backend(backend_name)
        if not backend:
            backend = self.backend_manager.get_backend("qasm_simulator")
            logger.warning(f"Falling back to qasm_simulator")

        self.backend = backend
        self.submission_manager = JobSubmissionManager(self.backend, self.transpiler)

        logger.info(f"Quantum processor initialized with {self.backend.name}")

    def execute_circuit(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        job_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute quantum circuit end-to-end.

        Args:
            circuit: Quantum circuit to execute
            shots: Number of shots
            job_name: Optional job name

        Returns:
            Dictionary with execution results
        """
        try:
            # Submit job
            job_id, job = self.submission_manager.submit_job(
                circuit, shots=shots, job_name=job_name
            )

            # Get result
            result = self.submission_manager.get_job_result(job_id)

            if result:
                # Parse results
                counts_info = ResultParser.parse_counts(result)
                statevector = ResultParser.parse_statevector(result)

                return {
                    "job_id": job_id,
                    "backend": self.backend.name,
                    "counts": counts_info,
                    "statevector": statevector,
                    "metadata": {
                        "shots": shots,
                        "timestamp": datetime.now().isoformat(),
                    }
                }
            else:
                logger.error("Failed to retrieve job result")
                return None

        except Exception as e:
            logger.error(f"Circuit execution failed: {e}")
            return None

    def switch_backend(self, backend_name: str) -> bool:
        """
        Switch to different backend.

        Args:
            backend_name: Backend identifier

        Returns:
            True if successful
        """
        backend = self.backend_manager.get_backend(backend_name)
        if backend:
            self.backend = backend
            self.submission_manager = JobSubmissionManager(self.backend, self.transpiler)
            logger.info(f"Switched to backend: {backend_name}")
            return True
        else:
            logger.error(f"Backend {backend_name} not available")
            return False

    def get_available_backends(self) -> Dict[str, List[str]]:
        """Get list of available backends."""
        return self.backend_manager.list_available_backends()

    def get_backend_properties(self) -> Dict[str, Any]:
        """Get properties of current backend."""
        return {
            "name": self.backend.name,
            "num_qubits": self.backend.num_qubits,
            "basis_gates": getattr(self.backend, "basis_gates", []),
            "coupling_map": getattr(self.backend, "coupling_map", None),
            "dt": getattr(self.backend, "dt", None),
        }
