"""
Quantum Feature Extractor for ML Pipeline Integration

Advanced feature extraction featuring:
- Classical image feature encoding into quantum states
- Quantum circuit execution with parameter optimization
- Quantum measurement and feature extraction
- Conversion to classical feature vectors
- Integration with ML inference engine
- Hybrid quantum-classical inference pipeline

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from quantum.qiskit_integration import QiskitQuantumProcessor
from quantum.negative_space_circuit import NegativeSpaceQuantumCircuit
from quantum.execution_strategy import QuantumExecutionEngine, ExecutionBackend
from quantum.hybrid_optimizer import HybridQuantumClassicalOptimizer

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """Preprocesses classical features for quantum encoding."""

    @staticmethod
    def normalize_features(
        features: np.ndarray,
        method: str = "minmax",
    ) -> np.ndarray:
        """
        Normalize features to [0, 1] or [-1, 1].

        Args:
            features: Input features
            method: Normalization method ('minmax', 'zscore')

        Returns:
            Normalized features
        """
        features = np.asarray(features, dtype=np.float32)

        if method == "minmax":
            # Min-max normalization to [0, 1]
            min_val = np.min(features)
            max_val = np.max(features)
            if max_val > min_val:
                features = (features - min_val) / (max_val - min_val)
            else:
                features = np.zeros_like(features)

        elif method == "zscore":
            # Z-score normalization
            mean = np.mean(features)
            std = np.std(features)
            if std > 0:
                features = (features - mean) / std

        return features

    @staticmethod
    def extract_key_features(
        image_data: np.ndarray,
        num_features: int = 6,
    ) -> np.ndarray:
        """
        Extract key features from image data.

        Args:
            image_data: Image array
            num_features: Number of features to extract

        Returns:
            Extracted features
        """
        image_data = np.asarray(image_data).flatten()

        if len(image_data) >= num_features:
            # Uniform sampling
            indices = np.linspace(0, len(image_data) - 1, num_features, dtype=int)
            features = image_data[indices]
        else:
            # Pad with zeros
            features = np.pad(image_data, (0, num_features - len(image_data)))

        return features

    @staticmethod
    def apply_dimensionality_reduction(
        features: np.ndarray,
        target_dim: int = 6,
        method: str = "pca",
    ) -> np.ndarray:
        """
        Reduce feature dimensionality.

        Args:
            features: Input features
            target_dim: Target dimension
            method: Reduction method ('pca', 'sampling')

        Returns:
            Reduced features
        """
        if method == "sampling":
            if len(features) > target_dim:
                indices = np.random.choice(len(features), target_dim, replace=False)
                return features[indices]
            else:
                return features

        # For PCA, would use sklearn but keeping simple
        return FeaturePreprocessor.extract_key_features(features, target_dim)


class QuantumFeatureExtractor:
    """Main quantum feature extractor for ML integration."""

    def __init__(
        self,
        num_qubits: int = 8,
        num_feature_qubits: int = 6,
        shots: int = 1024,
        backend: ExecutionBackend = ExecutionBackend.QASM_SIMULATOR,
    ) -> None:
        """
        Initialize quantum feature extractor.

        Args:
            num_qubits: Total qubits
            num_feature_qubits: Feature qubits
            shots: Measurement shots
            backend: Execution backend
        """
        self.num_qubits = num_qubits
        self.num_feature_qubits = num_feature_qubits
        self.shots = shots

        self.preprocessor = FeaturePreprocessor()
        self.circuit_builder = NegativeSpaceQuantumCircuit(
            num_qubits=num_qubits,
            num_feature_qubits=num_feature_qubits,
        )
        self.execution_engine = QuantumExecutionEngine(
            default_backend=backend,
            use_fallback=True,
        )

        logger.info(
            f"Initialized QuantumFeatureExtractor: "
            f"{num_qubits} qubits, {num_feature_qubits} feature qubits"
        )

    def extract_quantum_features(
        self,
        classical_features: np.ndarray,
        parameters: Optional[np.ndarray] = None,
        measurement_basis: str = "z",
    ) -> Dict[str, Any]:
        """
        Extract quantum features from classical input.

        Args:
            classical_features: Classical feature vector
            parameters: Optional circuit parameters
            measurement_basis: Measurement basis ('z', 'x', 'y')

        Returns:
            Quantum feature extraction results
        """
        try:
            # Preprocess features
            processed_features = self.preprocessor.normalize_features(classical_features)
            processed_features = self.preprocessor.extract_key_features(
                processed_features,
                num_features=self.num_feature_qubits
            )

            # Build quantum circuit
            circuit = self.circuit_builder.build_full_circuit(
                features=processed_features,
                parameters=parameters,
                measurement_basis=measurement_basis,
            )

            # Execute circuit
            execution_result = self.execution_engine.execute_circuit(
                circuit,
                shots=self.shots,
            )

            if not execution_result or not execution_result.get("success"):
                logger.error("Quantum circuit execution failed")
                return {"success": False, "error": "Execution failed"}

            # Extract quantum features from measurements
            counts = execution_result.get("counts", {})
            quantum_features = self._extract_features_from_counts(counts)

            return {
                "success": True,
                "quantum_features": quantum_features,
                "raw_counts": counts,
                "statistics": execution_result.get("statistics", {}),
                "execution_backend": execution_result.get("backend"),
                "circuit_depth": circuit.depth(),
                "num_qubits": circuit.num_qubits,
            }

        except Exception as e:
            logger.error(f"Quantum feature extraction failed: {e}")
            return {"success": False, "error": str(e)}

    def _extract_features_from_counts(
        self,
        counts: Dict[str, int],
    ) -> np.ndarray:
        """
        Extract feature vector from measurement counts.

        Args:
            counts: Measurement counts dictionary

        Returns:
            Feature vector
        """
        total_shots = sum(counts.values())
        num_states = len(counts)

        # Initialize feature vector
        features = np.zeros(self.num_feature_qubits)

        # Compute expectation values for each qubit
        for state_str, count in counts.items():
            prob = count / total_shots

            # Extract per-qubit probabilities
            for qubit_idx in range(min(len(state_str), self.num_feature_qubits)):
                if state_str[qubit_idx] == "1":
                    features[qubit_idx] += prob

        # Normalize
        features = features / np.max(features) if np.max(features) > 0 else features

        return features

    def extract_multi_basis_features(
        self,
        classical_features: np.ndarray,
        bases: List[str] = ["z", "x", "y"],
    ) -> Dict[str, Any]:
        """
        Extract features in multiple measurement bases.

        Args:
            classical_features: Classical features
            bases: List of measurement bases

        Returns:
            Multi-basis features
        """
        results = {}
        all_features = []

        for basis in bases:
            result = self.extract_quantum_features(
                classical_features,
                measurement_basis=basis,
            )

            if result.get("success"):
                features = result.get("quantum_features", np.array([]))
                results[basis] = features
                all_features.append(features)

        # Combine features from all bases
        combined_features = np.concatenate(all_features) if all_features else np.array([])

        return {
            "success": True,
            "basis_features": results,
            "combined_features": combined_features,
        }

    def extract_parameterized_features(
        self,
        classical_features: np.ndarray,
        parameter_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Extract features with parameterized circuits.

        Args:
            classical_features: Classical features
            parameter_ranges: Optional parameter ranges

        Returns:
            Parameterized features and optimal parameters
        """
        num_params = self.circuit_builder.ansatz_builder.get_num_parameters()

        # Initialize parameters randomly
        initial_params = np.random.rand(num_params) * 2 * np.pi

        # Set up optimizer for feature extraction
        def circuit_factory(params: np.ndarray) -> QuantumCircuit:
            return self.circuit_builder.build_full_circuit(
                features=classical_features,
                parameters=params,
            )

        optimizer = HybridQuantumClassicalOptimizer(
            circuit_factory=circuit_factory,
            num_parameters=num_params,
            optimizer_method="COBYLA",
        )

        # Optimize parameters
        optimal_params, _, metadata = optimizer.optimize(
            maxiter=50,
            initial_parameters=initial_params,
        )

        # Extract features with optimal parameters
        final_result = self.extract_quantum_features(
            classical_features,
            parameters=optimal_params,
        )

        return {
            **final_result,
            "optimal_parameters": optimal_params,
            "optimization_metadata": metadata,
        }


class HybridInferenceIntegrator:
    """Integrates quantum feature extraction with classical ML inference."""

    def __init__(
        self,
        quantum_extractor: QuantumFeatureExtractor,
        classical_model: Optional[Any] = None,
    ) -> None:
        """
        Initialize hybrid inference integrator.

        Args:
            quantum_extractor: QuantumFeatureExtractor instance
            classical_model: Optional classical ML model
        """
        self.quantum_extractor = quantum_extractor
        self.classical_model = classical_model
        logger.info("Initialized HybridInferenceIntegrator")

    def hybrid_inference(
        self,
        input_data: np.ndarray,
        use_quantum_features: bool = True,
        use_classical_features: bool = False,
        combine_method: str = "concatenate",
    ) -> Dict[str, Any]:
        """
        Perform hybrid quantum-classical inference.

        Args:
            input_data: Input data (image or feature vector)
            use_quantum_features: Include quantum features
            use_classical_features: Include classical features
            combine_method: How to combine feature types

        Returns:
            Inference results
        """
        features_dict = {}

        # Extract quantum features
        if use_quantum_features:
            q_result = self.quantum_extractor.extract_quantum_features(input_data)
            if q_result.get("success"):
                features_dict["quantum"] = q_result.get("quantum_features")

        # Keep classical features
        if use_classical_features:
            processed = self.quantum_extractor.preprocessor.normalize_features(input_data)
            features_dict["classical"] = processed

        # Combine features
        combined_features = self._combine_features(features_dict, combine_method)

        # Classical inference
        predictions = None
        if self.classical_model and combined_features is not None:
            try:
                predictions = self.classical_model.predict(combined_features)
            except Exception as e:
                logger.warning(f"Classical model inference failed: {e}")

        return {
            "features": combined_features,
            "quantum_features": features_dict.get("quantum"),
            "classical_features": features_dict.get("classical"),
            "predictions": predictions,
            "feature_dimensions": {k: len(v) for k, v in features_dict.items()},
        }

    def _combine_features(
        self,
        features_dict: Dict[str, np.ndarray],
        method: str = "concatenate",
    ) -> Optional[np.ndarray]:
        """
        Combine features from multiple sources.

        Args:
            features_dict: Dictionary of feature arrays
            method: Combination method

        Returns:
            Combined feature array
        """
        if not features_dict:
            return None

        if method == "concatenate":
            return np.concatenate(list(features_dict.values()))
        elif method == "stack":
            return np.stack(list(features_dict.values()))
        elif method == "mean":
            return np.mean(list(features_dict.values()), axis=0)
        else:
            return np.concatenate(list(features_dict.values()))

    def benchmark_hybrid_inference(
        self,
        test_data: List[np.ndarray],
        num_runs: int = 3,
    ) -> Dict[str, Any]:
        """
        Benchmark hybrid inference performance.

        Args:
            test_data: List of test samples
            num_runs: Number of runs

        Returns:
            Benchmark results
        """
        quantum_times = []
        total_times = []

        import time

        for sample in test_data:
            for _ in range(num_runs):
                start = time.time()
                self.hybrid_inference(sample)
                total_times.append(time.time() - start)

        return {
            "num_samples": len(test_data),
            "num_runs": num_runs,
            "total_samples_processed": len(test_data) * num_runs,
            "mean_inference_time": np.mean(total_times),
            "std_inference_time": np.std(total_times),
            "min_time": np.min(total_times),
            "max_time": np.max(total_times),
        }
