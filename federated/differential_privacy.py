"""
Differential Privacy Implementation for Federated Learning
DP-SGD with privacy budget tracking and composition analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class PrivacyMode(Enum):
    """Privacy guarantee types."""
    EPSILON_DELTA = "epsilon_delta"
    RENYI_DIFFERENTIAL_PRIVACY = "rdp"
    PURE_DIFFERENTIAL_PRIVACY = "pure"


@dataclass
class PrivacyBudget:
    """Privacy budget tracking with composition analysis."""

    epsilon: float = 1.0
    delta: float = 1e-5
    total_epsilon: float = 0.0
    total_delta: float = 0.0
    num_compositions: int = 0
    mode: PrivacyMode = PrivacyMode.EPSILON_DELTA
    history: List[dict] = field(default_factory=list)

    def update(self, epsilon: float, delta: float, mechanism: str = "unknown"):
        """Update privacy budget with composition."""
        if epsilon > self.epsilon - self.total_epsilon:
            logger.warning(
                f"Privacy budget exceeded: {self.total_epsilon + epsilon} > {self.epsilon}"
            )

        self.total_epsilon += epsilon
        self.total_delta += delta
        self.num_compositions += 1

        self.history.append({
            "mechanism": mechanism,
            "epsilon": epsilon,
            "delta": delta,
            "cumulative_epsilon": self.total_epsilon,
            "cumulative_delta": self.total_delta,
        })

        logger.info(
            f"Budget update: {mechanism} | ε={epsilon:.4f}, δ={delta:.2e} | "
            f"Total ε={self.total_epsilon:.4f}, δ={self.total_delta:.2e}"
        )

    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return (
            self.total_epsilon >= self.epsilon or
            self.total_delta >= self.delta
        )

    def remaining(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        return (
            max(0, self.epsilon - self.total_epsilon),
            max(0, self.delta - self.total_delta),
        )


class DifferentialPrivacyManager:
    """
    DP-SGD implementation with privacy accounting.
    Manages gradient clipping, noise addition, and privacy budget.
    """

    def __init__(
        self,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        gradient_clip_norm: float = 1.0,
        batch_size: int = 32,
        noise_multiplier: float = 1.0,
    ):
        """
        Initialize DP manager.

        Args:
            target_epsilon: Target epsilon for privacy guarantee
            target_delta: Target delta for privacy guarantee
            gradient_clip_norm: L2 clipping norm for gradients
            batch_size: Batch size for privacy accounting
            noise_multiplier: Multiplier for Gaussian noise
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.gradient_clip_norm = gradient_clip_norm
        self.batch_size = batch_size
        self.noise_multiplier = noise_multiplier

        self.privacy_budget = PrivacyBudget(
            epsilon=target_epsilon,
            delta=target_delta,
        )

        logger.info(
            f"DP Manager initialized: ε={target_epsilon}, δ={target_delta}, "
            f"clip_norm={gradient_clip_norm}, noise_multiplier={noise_multiplier}"
        )

    def clip_gradients(
        self,
        gradients: List[np.ndarray],
        norm: Optional[float] = None,
    ) -> List[np.ndarray]:
        """
        Clip gradients to maximum norm per sample.

        Args:
            gradients: List of gradient arrays
            norm: Override gradient_clip_norm

        Returns:
            Clipped gradients
        """
        clip_norm = norm or self.gradient_clip_norm
        clipped = []

        for grad in gradients:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > clip_norm:
                clipped_grad = grad * (clip_norm / (grad_norm + 1e-10))
            else:
                clipped_grad = grad
            clipped.append(clipped_grad)

        return clipped

    def add_gaussian_noise(
        self,
        gradients: List[np.ndarray],
        sensitivity: float = 1.0,
    ) -> List[np.ndarray]:
        """
        Add Gaussian noise for differential privacy.

        Args:
            gradients: Clipped gradients
            sensitivity: Sensitivity of function

        Returns:
            Noised gradients
        """
        noised = []
        noise_scale = self.noise_multiplier * sensitivity / self.batch_size

        for grad in gradients:
            noise = np.random.normal(0, noise_scale, grad.shape)
            noised_grad = grad + noise
            noised.append(noised_grad)

        return noised

    def compute_rdp(
        self,
        num_steps: int,
        orders: Optional[List[float]] = None,
    ) -> List[Tuple[float, float]]:
        """
        Compute Renyi Differential Privacy (RDP) for composition.

        Args:
            num_steps: Number of composition steps
            orders: RDP orders to compute

        Returns:
            List of (order, rdp_value) tuples
        """
        if orders is None:
            orders = [i + 2 for i in range(32)]

        rdps = []
        noise_std = self.noise_multiplier

        for order in orders:
            if order == 1:
                continue

            # RDP composition formula
            exponent = order * (
                np.log(1 + (noise_std**2) * (order - 1))
                - 2 * np.log(1 + (noise_std**2))
            ) / (2 * (noise_std**2))

            rdp = min(exponent / num_steps, 1e10)  # Avoid overflow
            rdps.append((order, rdp))

        return rdps

    def rdp_to_epsilon(
        self,
        rdps: List[Tuple[float, float]],
        delta: float,
    ) -> float:
        """
        Convert RDP to epsilon using conversion formula.

        Args:
            rdps: List of (order, rdp_value) tuples
            delta: Target delta

        Returns:
            Epsilon value
        """
        min_epsilon = float('inf')

        for order, rdp_value in rdps:
            epsilon = rdp_value + (2 * np.sqrt(rdp_value)) / np.sqrt(
                order - 1
            ) + np.log(1 / delta) / order
            min_epsilon = min(min_epsilon, epsilon)

        return min_epsilon

    def compute_epsilon(
        self,
        num_epochs: int,
        dataset_size: int,
        delta: Optional[float] = None,
    ) -> float:
        """
        Compute epsilon for given number of epochs.

        Args:
            num_epochs: Number of training epochs
            dataset_size: Size of training dataset
            delta: Target delta (use self.target_delta if None)

        Returns:
            Epsilon value
        """
        delta = delta or self.target_delta

        # Number of composition steps
        num_steps = (dataset_size // self.batch_size) * num_epochs

        # Compute RDP
        rdps = self.compute_rdp(num_steps)

        # Convert to epsilon
        epsilon = self.rdp_to_epsilon(rdps, delta)

        return epsilon

    def apply_privacy_protection(
        self,
        gradients: List[np.ndarray],
        mechanism: str = "dp-sgd",
    ) -> List[np.ndarray]:
        """
        Apply full DP protection: clip + noise.

        Args:
            gradients: Input gradients
            mechanism: Name of mechanism for logging

        Returns:
            Privacy-protected gradients
        """
        # Clip gradients
        clipped = self.clip_gradients(gradients)

        # Add noise
        noised = self.add_gaussian_noise(clipped)

        return noised

    def get_privacy_accounting(self, num_epochs: int, dataset_size: int) -> dict:
        """
        Get comprehensive privacy accounting.

        Args:
            num_epochs: Number of training epochs
            dataset_size: Size of training dataset

        Returns:
            Privacy accounting dictionary
        """
        epsilon = self.compute_epsilon(num_epochs, dataset_size)

        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "actual_epsilon": epsilon,
            "num_steps": (dataset_size // self.batch_size) * num_epochs,
            "gradient_clip_norm": self.gradient_clip_norm,
            "noise_multiplier": self.noise_multiplier,
            "privacy_budget": self.privacy_budget,
        }


class CompositionAnalyzer:
    """Analyze privacy composition across multiple mechanisms."""

    @staticmethod
    def parallel_composition(
        epsilon_values: List[float],
        delta_values: List[float],
    ) -> Tuple[float, float]:
        """
        Compute epsilon-delta for parallel composition.

        Args:
            epsilon_values: List of epsilon values
            delta_values: List of delta values

        Returns:
            (epsilon, delta) tuple
        """
        # Parallel composition: take max epsilon, sum delta
        epsilon = max(epsilon_values) if epsilon_values else 0
        delta = sum(delta_values)

        return epsilon, delta

    @staticmethod
    def sequential_composition(
        epsilon_values: List[float],
        delta_values: List[float],
    ) -> Tuple[float, float]:
        """
        Compute epsilon-delta for sequential composition.

        Args:
            epsilon_values: List of epsilon values
            delta_values: List of delta values

        Returns:
            (epsilon, delta) tuple
        """
        # Sequential composition: sum epsilon, sum delta
        epsilon = sum(epsilon_values)
        delta = sum(delta_values)

        return epsilon, delta

    @staticmethod
    def adaptive_composition(
        num_compositions: int,
        epsilon_per_composition: float,
        delta_per_composition: float,
    ) -> Tuple[float, float]:
        """
        Compute epsilon-delta for adaptive composition.

        Args:
            num_compositions: Number of composition steps
            epsilon_per_composition: Epsilon per step
            delta_per_composition: Delta per step

        Returns:
            (epsilon, delta) tuple
        """
        # Adaptive composition with sqrt(log(1/delta)) factor
        epsilon = (
            epsilon_per_composition * np.sqrt(2 * num_compositions * np.log(1 / delta_per_composition))
            + epsilon_per_composition * np.log(1 / delta_per_composition)
        )
        delta = num_compositions * delta_per_composition

        return epsilon, delta


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize DP manager
    dp_manager = DifferentialPrivacyManager(
        target_epsilon=1.0,
        target_delta=1e-5,
        gradient_clip_norm=1.0,
        batch_size=32,
        noise_multiplier=1.0,
    )

    # Simulate gradient clipping and noise
    gradients = [np.random.randn(10, 10) for _ in range(5)]
    protected = dp_manager.apply_privacy_protection(gradients)

    # Compute privacy accounting
    accounting = dp_manager.get_privacy_accounting(
        num_epochs=5,
        dataset_size=1000,
    )

    print(f"Privacy Accounting: {accounting}")
