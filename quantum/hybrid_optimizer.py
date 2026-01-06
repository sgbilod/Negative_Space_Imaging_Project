"""
Hybrid Classical-Quantum Optimizer

Advanced optimization module featuring:
- Classical optimizers (COBYLA, SPSA, Adam)
- Cost function evaluation on quantum circuits
- Iterative parameter optimization
- Convergence analysis and tracking
- Parameter history management
- Quantum-classical feedback loops

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, Optimizer
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)


class ClassicalOptimizer:
    """Wrapper for classical optimization algorithms."""

    def __init__(
        self,
        method: str = "COBYLA",
        **kwargs: Any
    ) -> None:
        """
        Initialize classical optimizer.

        Args:
            method: Optimization method ('COBYLA', 'SPSA', 'Adam', 'L-BFGS-B')
            **kwargs: Additional optimizer parameters
        """
        self.method = method
        self.optimizer_params = kwargs
        self.iteration_count = 0
        self.best_value = float("inf")
        self.best_params = None
        logger.info(f"Initialized {method} optimizer")

    def optimize(
        self,
        objective: Callable,
        initial_params: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        maxiter: int = 100,
        callback: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Run optimization.

        Args:
            objective: Objective function to minimize
            initial_params: Initial parameter values
            bounds: Parameter bounds (lower, upper)
            maxiter: Maximum iterations
            callback: Callback function for each iteration

        Returns:
            Tuple of (optimal_params, minimum_value, metadata)
        """
        if self.method == "COBYLA":
            return self._optimize_cobyla(objective, initial_params, bounds, maxiter, callback)
        elif self.method == "SPSA":
            return self._optimize_spsa(objective, initial_params, maxiter, callback)
        elif self.method == "L-BFGS-B":
            return self._optimize_lbfgs(objective, initial_params, bounds, maxiter, callback)
        else:
            logger.error(f"Unknown optimizer: {self.method}")
            return initial_params, float("inf"), {}

    def _optimize_cobyla(
        self,
        objective: Callable,
        initial_params: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]],
        maxiter: int,
        callback: Optional[Callable],
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """COBYLA optimization."""
        options = {"maxiter": maxiter, "rhobeg": 1.0}
        options.update(self.optimizer_params)

        result = minimize(
            objective,
            initial_params,
            method="COBYLA",
            options=options,
            callback=callback,
        )

        return result.x, result.fun, {
            "success": result.success,
            "nfev": result.nfev,
            "nit": result.nit,
            "message": result.message,
        }

    def _optimize_spsa(
        self,
        objective: Callable,
        initial_params: np.ndarray,
        maxiter: int,
        callback: Optional[Callable],
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """SPSA (Simultaneous Perturbation Stochastic Approximation)."""
        params = initial_params.copy()
        learning_rate = self.optimizer_params.get("learning_rate", 0.1)
        perturbation = self.optimizer_params.get("perturbation", 0.1)

        best_value = objective(params)
        history = []

        for iteration in range(maxiter):
            # Random perturbation
            delta = np.random.normal(0, perturbation, size=len(params))

            # Gradient estimation
            plus_value = objective(params + delta)
            minus_value = objective(params - delta)

            gradient = (plus_value - minus_value) / (2 * perturbation) * delta

            # Parameter update
            params -= learning_rate * gradient / (iteration + 1)

            # Track best value
            current_value = objective(params)
            if current_value < best_value:
                best_value = current_value
                self.best_params = params.copy()

            history.append(current_value)

            if callback:
                callback({"nit": iteration, "value": current_value})

        return params, best_value, {"history": history, "nit": maxiter}

    def _optimize_lbfgs(
        self,
        objective: Callable,
        initial_params: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]],
        maxiter: int,
        callback: Optional[Callable],
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """L-BFGS-B optimization."""
        options = {"maxiter": maxiter, "gtol": 1e-6}
        options.update(self.optimizer_params)

        result = minimize(
            objective,
            initial_params,
            method="L-BFGS-B",
            bounds=bounds,
            options=options,
            callback=callback,
        )

        return result.x, result.fun, {
            "success": result.success,
            "nfev": result.nfev,
            "nit": result.nit,
        }


class CostFunctionManager:
    """Manages cost function evaluation on quantum circuits."""

    def __init__(
        self,
        circuit_factory: Callable[[np.ndarray], QuantumCircuit],
        observable: Optional[SparsePauliOp] = None,
    ) -> None:
        """
        Initialize cost function manager.

        Args:
            circuit_factory: Function that creates circuit given parameters
            observable: Observable to measure (Z if None)
        """
        self.circuit_factory = circuit_factory
        self.observable = observable or SparsePauliOp(["Z"])
        self.evaluation_count = 0
        self.evaluation_history: List[float] = []
        self.estimator = Estimator()

    def evaluate_circuit(
        self,
        parameters: np.ndarray,
        backend: Optional[Any] = None,
    ) -> float:
        """
        Evaluate cost function for given parameters.

        Args:
            parameters: Circuit parameters
            backend: Optional backend for execution

        Returns:
            Cost function value
        """
        try:
            # Create parameterized circuit
            circuit = self.circuit_factory(parameters)

            # Evaluate expectation value
            job = self.estimator.run(circuit, self.observable)
            result = job.result()

            # Extract expectation value
            exp_value = result.values[0].real

            self.evaluation_count += 1
            self.evaluation_history.append(exp_value)

            logger.debug(f"Evaluation {self.evaluation_count}: E = {exp_value:.6f}")

            return exp_value

        except Exception as e:
            logger.error(f"Cost function evaluation failed: {e}")
            return float("inf")

    def get_evaluation_statistics(self) -> Dict[str, float]:
        """Get statistics of evaluations."""
        if not self.evaluation_history:
            return {}

        history = np.array(self.evaluation_history)
        return {
            "min_value": float(np.min(history)),
            "max_value": float(np.max(history)),
            "mean_value": float(np.mean(history)),
            "std_value": float(np.std(history)),
            "total_evaluations": self.evaluation_count,
        }


class ParameterHistory:
    """Tracks parameter optimization history."""

    def __init__(self) -> None:
        """Initialize parameter history tracker."""
        self.iterations: List[int] = []
        self.parameters: List[np.ndarray] = []
        self.values: List[float] = []
        self.timestamps: List[float] = []

    def record(
        self,
        iteration: int,
        params: np.ndarray,
        value: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record optimization step.

        Args:
            iteration: Iteration number
            params: Circuit parameters
            value: Cost function value
            timestamp: Optional timestamp
        """
        self.iterations.append(iteration)
        self.parameters.append(params.copy())
        self.values.append(value)
        if timestamp is not None:
            self.timestamps.append(timestamp)

    def get_best_parameters(self) -> Optional[np.ndarray]:
        """Get best parameters found."""
        if not self.values:
            return None
        best_idx = np.argmin(self.values)
        return self.parameters[best_idx]

    def get_best_value(self) -> Optional[float]:
        """Get best value found."""
        if not self.values:
            return None
        return min(self.values)

    def get_history(self) -> Dict[str, Any]:
        """Get complete history."""
        return {
            "iterations": self.iterations,
            "values": self.values,
            "num_iterations": len(self.iterations),
            "best_value": self.get_best_value(),
        }

    def to_numpy_array(self) -> np.ndarray:
        """Convert history to numpy array."""
        return np.array(self.parameters)


class ConvergenceAnalyzer:
    """Analyzes convergence of optimization."""

    @staticmethod
    def compute_convergence_rate(
        values: List[float],
        window_size: int = 10,
    ) -> float:
        """
        Compute convergence rate.

        Args:
            values: Sequence of cost values
            window_size: Window size for averaging

        Returns:
            Convergence rate
        """
        if len(values) < 2 * window_size:
            return 0.0

        # Compare recent values to earlier values
        recent = np.mean(values[-window_size:])
        earlier = np.mean(values[-2*window_size:-window_size])

        if earlier == 0:
            return 0.0

        return (earlier - recent) / abs(earlier)

    @staticmethod
    def check_convergence(
        values: List[float],
        tolerance: float = 1e-6,
        patience: int = 10,
    ) -> Tuple[bool, str]:
        """
        Check if optimization has converged.

        Args:
            values: Sequence of cost values
            tolerance: Tolerance for convergence
            patience: Patience for early stopping

        Returns:
            Tuple of (converged, reason)
        """
        if len(values) < 2:
            return False, "Not enough iterations"

        # Check for stagnation
        recent_changes = np.abs(np.diff(values[-patience:]))
        max_change = np.max(recent_changes)

        if max_change < tolerance:
            return True, f"Converged (change={max_change:.2e})"

        # Check if improving
        if values[-1] > values[-2]:
            return False, "Cost increasing"

        return False, "Optimizing"

    @staticmethod
    def plot_convergence(
        values: List[float],
        title: str = "Optimization Convergence",
    ) -> None:
        """
        Plot convergence curve.

        Args:
            values: Sequence of cost values
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(values, 'b-', linewidth=2)
            plt.xlabel("Iteration")
            plt.ylabel("Cost Value")
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            logger.info("Convergence plot generated")

        except ImportError:
            logger.warning("Matplotlib not available for plotting")


class HybridQuantumClassicalOptimizer:
    """Main optimizer orchestrating quantum-classical hybrid optimization."""

    def __init__(
        self,
        circuit_factory: Callable[[np.ndarray], QuantumCircuit],
        observable: Optional[SparsePauliOp] = None,
        optimizer_method: str = "COBYLA",
        num_parameters: int = 10,
    ) -> None:
        """
        Initialize hybrid optimizer.

        Args:
            circuit_factory: Function creating circuit from parameters
            observable: Observable to measure
            optimizer_method: Classical optimizer method
            num_parameters: Number of circuit parameters
        """
        self.circuit_factory = circuit_factory
        self.num_parameters = num_parameters

        self.cost_manager = CostFunctionManager(circuit_factory, observable)
        self.optimizer = ClassicalOptimizer(method=optimizer_method)
        self.parameter_history = ParameterHistory()
        self.convergence_analyzer = ConvergenceAnalyzer()

        logger.info(
            f"Initialized HybridQuantumClassicalOptimizer: "
            f"{optimizer_method}, {num_parameters} parameters"
        )

    def optimize(
        self,
        maxiter: int = 100,
        initial_parameters: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        tolerance: float = 1e-6,
        patience: int = 10,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Run hybrid optimization.

        Args:
            maxiter: Maximum iterations
            initial_parameters: Initial circuit parameters
            bounds: Parameter bounds
            tolerance: Convergence tolerance
            patience: Early stopping patience

        Returns:
            Tuple of (optimal_params, min_value, metadata)
        """
        # Initialize parameters
        if initial_parameters is None:
            initial_parameters = np.random.rand(self.num_parameters) * 2 * np.pi

        # Set bounds
        if bounds is None:
            bounds = (np.zeros(self.num_parameters), 2 * np.pi * np.ones(self.num_parameters))

        logger.info("Starting hybrid quantum-classical optimization...")

        # Callback for tracking
        def optimization_callback(x_k: np.ndarray) -> None:
            if hasattr(x_k, '__len__'):
                value = self.cost_manager.evaluate_circuit(x_k)
                iteration = len(self.parameter_history.iterations)
                self.parameter_history.record(iteration, x_k, value)

                # Check convergence
                converged, reason = self.convergence_analyzer.check_convergence(
                    self.parameter_history.values,
                    tolerance=tolerance,
                    patience=patience
                )

                if converged:
                    logger.info(f"Converged: {reason}")

        # Run optimization
        optimal_params, min_value, opt_info = self.optimizer.optimize(
            objective=self.cost_manager.evaluate_circuit,
            initial_params=initial_parameters,
            bounds=bounds,
            maxiter=maxiter,
            callback=optimization_callback,
        )

        # Get final statistics
        stats = self.cost_manager.get_evaluation_statistics()
        history = self.parameter_history.get_history()

        metadata = {
            **opt_info,
            **stats,
            "final_cost": min_value,
            "optimization_history": history,
        }

        logger.info(f"Optimization complete: min_value = {min_value:.6f}")

        return optimal_params, min_value, metadata

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization run."""
        return {
            "num_parameters": self.num_parameters,
            "optimizer_method": self.optimizer.method,
            "total_evaluations": self.cost_manager.evaluation_count,
            "best_value": self.parameter_history.get_best_value(),
            "convergence_history": self.parameter_history.get_history(),
        }
