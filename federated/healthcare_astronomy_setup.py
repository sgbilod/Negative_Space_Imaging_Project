"""
Healthcare and Astronomy Use Case Setup
Multi-institutional federated learning scenario.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from .simulation import SimulatedClient, ClientSimulationConfig, FederationSimulator
from .federated_server import FederatedServer

logger = logging.getLogger(__name__)


@dataclass
class InstitutionConfig:
    """Configuration for a participating institution."""

    institution_id: str
    institution_type: str  # "hospital", "observatory"
    num_local_samples: int
    num_local_clients: int
    data_domain: str  # "medical_imaging", "astronomical"
    privacy_critical: bool = True
    data_heterogeneity: float = 1.0


class MedicalImagingDataGenerator:
    """Generate synthetic medical imaging data."""

    @staticmethod
    def create_federated_datasets(
        num_institutions: int = 3,
        samples_per_institution: int = 500,
    ) -> Dict[str, np.ndarray]:
        """
        Create federated medical imaging datasets.

        Args:
            num_institutions: Number of hospitals
            samples_per_institution: Samples per hospital

        Returns:
            Dictionary of datasets
        """
        datasets = {}

        for i in range(num_institutions):
            # Simulate different imaging devices/protocols
            base_pattern = np.random.randn(28, 28) * 0.5 + 0.5

            institution_data = []
            for j in range(samples_per_institution):
                # Add institutional variation
                noise = np.random.randn(28, 28) * 0.1 * (i + 1)
                image = base_pattern + noise
                image = np.clip(image, 0, 1)
                institution_data.append(image)

            datasets[f"hospital_{i}"] = np.array(institution_data)

            logger.info(
                f"Hospital {i}: {len(institution_data)} images "
                f"(variation scale: {0.1 * (i + 1):.2f})"
            )

        return datasets


class AstronomicalObservationGenerator:
    """Generate synthetic astronomical observation data."""

    @staticmethod
    def create_federated_datasets(
        num_observatories: int = 2,
        observations_per_observatory: int = 300,
    ) -> Dict[str, np.ndarray]:
        """
        Create federated astronomical observation datasets.

        Args:
            num_observatories: Number of observatories
            observations_per_observatory: Observations per observatory

        Returns:
            Dictionary of datasets
        """
        datasets = {}

        for i in range(num_observatories):
            # Simulate different telescope types
            base_pattern = np.random.poisson(5, (28, 28))

            observatory_data = []
            for j in range(observations_per_observatory):
                # Add observatory-specific noise
                noise = np.random.poisson(i + 1, (28, 28))
                observation = (base_pattern + noise) / 20.0
                observation = np.clip(observation, 0, 1)
                observatory_data.append(observation)

            datasets[f"observatory_{i}"] = np.array(observatory_data)

            logger.info(
                f"Observatory {i}: {len(observatory_data)} observations "
                f"(noise scale: {i + 1})"
            )

        return datasets


class HealthcareAstronomySimulation:
    """Simulate federated learning across hospitals and observatories."""

    def __init__(
        self,
        num_hospitals: int = 3,
        num_observatories: int = 2,
        model: Optional[nn.Module] = None,
    ):
        """
        Initialize healthcare-astronomy simulation.

        Args:
            num_hospitals: Number of hospitals
            num_observatories: Number of observatories
            model: Model to use (create default if None)
        """
        self.num_hospitals = num_hospitals
        self.num_observatories = num_observatories

        if model is None:
            # Create simple CNN model
            model = self._create_default_model()

        self.simulator = FederationSimulator(
            model=model,
            num_clients=num_hospitals + num_observatories,
            device="cpu",
        )

        self.institution_configs: List[InstitutionConfig] = []
        self.datasets: Dict[str, np.ndarray] = {}
        self.results: Dict[str, Any] = {}

    def _create_default_model(self) -> nn.Module:
        """Create a simple CNN model for medical/astronomical data."""
        class SimpleConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.fc1 = nn.Linear(64 * 28 * 28, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                if len(x.shape) == 3:
                    x = x.unsqueeze(1)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        return SimpleConv()

    def setup_institutions(self):
        """Setup institutional configurations."""
        # Medical institutions (hospitals)
        for i in range(self.num_hospitals):
            config = InstitutionConfig(
                institution_id=f"hospital_{i}",
                institution_type="hospital",
                num_local_samples=500 + i * 100,
                num_local_clients=1,
                data_domain="medical_imaging",
                privacy_critical=True,
                data_heterogeneity=1.0 + i * 0.3,
            )
            self.institution_configs.append(config)

        # Astronomical institutions (observatories)
        for i in range(self.num_observatories):
            config = InstitutionConfig(
                institution_id=f"observatory_{i}",
                institution_type="observatory",
                num_local_samples=300 + i * 50,
                num_local_clients=1,
                data_domain="astronomical",
                privacy_critical=False,
                data_heterogeneity=1.2 + i * 0.2,
            )
            self.institution_configs.append(config)

        logger.info(f"Setup {len(self.institution_configs)} institutions")

    def create_datasets(self):
        """Create federated datasets for all institutions."""
        # Medical imaging from hospitals
        medical_datasets = MedicalImagingDataGenerator.create_federated_datasets(
            num_institutions=self.num_hospitals,
            samples_per_institution=500,
        )

        # Astronomical observations from observatories
        astronomical_datasets = AstronomicalObservationGenerator.create_federated_datasets(
            num_observatories=self.num_observatories,
            observations_per_observatory=300,
        )

        self.datasets = {**medical_datasets, **astronomical_datasets}

        logger.info(f"Created {len(self.datasets)} datasets")

    def create_clients(self):
        """Create federated clients for all institutions."""
        client_configs = []

        for idx, config in enumerate(self.institution_configs):
            dataset_key = list(self.datasets.keys())[idx]
            dataset = self.datasets[dataset_key]

            client_config = ClientSimulationConfig(
                client_id=config.institution_id,
                num_samples=len(dataset),
                iid_level=0.3 if config.institution_type == "hospital" else 0.6,
                data_heterogeneity=config.data_heterogeneity,
                straggler_probability=0.1 if idx % 2 == 0 else 0.0,
                dropout_probability=0.05,
                enable_dp=config.privacy_critical,
                dp_epsilon=1.0 if config.privacy_critical else 10.0,
            )
            client_configs.append(client_config)

        self.simulator.create_clients(client_configs)

        logger.info(f"Created {len(client_configs)} federated clients")

    def run_simulation(
        self,
        num_rounds: int = 10,
        local_epochs: int = 2,
    ) -> Dict[str, Any]:
        """
        Run healthcare-astronomy federated learning simulation.

        Args:
            num_rounds: Number of federation rounds
            local_epochs: Local epochs per client

        Returns:
            Simulation results
        """
        logger.info(
            f"Starting healthcare-astronomy simulation: "
            f"{self.num_hospitals} hospitals, "
            f"{self.num_observatories} observatories, "
            f"{num_rounds} rounds"
        )

        # Run simulation
        self.simulator.run_simulation(
            num_rounds=num_rounds,
            local_epochs=local_epochs,
        )

        # Get results
        self.results = self.simulator.get_simulation_summary()

        return self.results

    def evaluate_use_case(self) -> Dict[str, Any]:
        """
        Evaluate federated learning performance for healthcare-astronomy.

        Args:
            Returns:
            Evaluation results
        """
        if not self.results:
            logger.warning("No simulation results available")
            return {}

        metrics = self.results["metrics_history"]

        # Analyze institutional performance
        evaluation = {
            "overall": {
                "final_accuracy": self.results.get("final_accuracy", 0),
                "final_loss": self.results.get("final_loss", float('inf')),
                "total_communication": self.results.get("total_communication", 0),
                "min_epsilon": self.results.get("min_epsilon_reached", float('inf')),
            },
            "convergence": {
                "rounds": len(metrics),
                "converged": metrics[-1].convergence_metric < 0.01 if metrics else False,
                "final_convergence_metric": metrics[-1].convergence_metric if metrics else 0,
            },
            "privacy": {
                "avg_epsilon": np.mean([m.privacy_epsilon for m in metrics]),
                "min_epsilon": min([m.privacy_epsilon for m in metrics]),
                "privacy_budget_exhausted": any(
                    m.privacy_epsilon >= 1.0 for m in metrics
                ),
            },
            "robustness": {
                "avg_active_clients": np.mean([m.num_active_clients for m in metrics]),
                "total_stragglers": sum(m.num_stragglers for m in metrics),
                "straggler_rate": sum(m.num_stragglers for m in metrics) / (
                    len(metrics) * (self.num_hospitals + self.num_observatories) + 1e-10
                ),
            },
            "communication": {
                "total_bytes": sum(m.communication_cost for m in metrics),
                "avg_bytes_per_round": np.mean([m.communication_cost for m in metrics]),
            },
        }

        logger.info(f"Use case evaluation: {evaluation}")

        return evaluation

    def get_multi_institutional_summary(self) -> Dict[str, Any]:
        """Get summary for each institution."""
        summary = {
            "hospitals": [],
            "observatories": [],
        }

        for config in self.institution_configs:
            info = {
                "id": config.institution_id,
                "type": config.institution_type,
                "num_samples": config.num_local_samples,
                "data_heterogeneity": config.data_heterogeneity,
                "privacy_critical": config.privacy_critical,
            }

            if config.institution_type == "hospital":
                summary["hospitals"].append(info)
            else:
                summary["observatories"].append(info)

        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    sim = HealthcareAstronomySimulation(
        num_hospitals=3,
        num_observatories=2,
    )

    sim.setup_institutions()
    sim.create_datasets()
    sim.create_clients()

    results = sim.run_simulation(num_rounds=5)
    evaluation = sim.evaluate_use_case()

    print(f"Simulation results: {results}")
    print(f"Evaluation: {evaluation}")
