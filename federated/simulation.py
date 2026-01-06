"""
Federated Learning Simulation
Multi-client training with non-IID data, stragglers, and dropout.
"""

import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from .federated_client import FederatedClient, ClientTrainer
from .federated_server import FederatedServer, ModelAggregator, AggregationStrategy
from .differential_privacy import DifferentialPrivacyManager
from .communication import SecureSerializer

logger = logging.getLogger(__name__)


@dataclass
class ClientSimulationConfig:
    """Configuration for simulated client."""

    client_id: str
    num_samples: int
    iid_level: float = 0.5  # 0=non-IID, 1=IID
    data_heterogeneity: float = 1.0  # Data distribution shift
    straggler_probability: float = 0.0  # Probability of being slow
    dropout_probability: float = 0.0  # Probability of dropping out
    enable_dp: bool = True
    dp_epsilon: float = 1.0


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""

    round_number: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    global_accuracy: float = 0.0
    global_loss: float = 0.0
    communication_cost: int = 0  # Bytes transmitted
    privacy_epsilon: float = float('inf')
    privacy_delta: float = 0.0
    num_active_clients: int = 0
    num_stragglers: int = 0
    convergence_metric: float = 0.0
    training_time: float = 0.0
    client_metrics: Dict[str, Dict] = field(default_factory=dict)


class SimulatedClient:
    """Simulated federated learning client for testing."""

    def __init__(
        self,
        config: ClientSimulationConfig,
        model: nn.Module,
        device: str = "cpu",
    ):
        """
        Initialize simulated client.

        Args:
            config: Client configuration
            model: Model to train
            device: Torch device
        """
        self.config = config
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        # Create local data (synthetic)
        self._create_local_data()

        self.trainer = ClientTrainer(
            client_id=config.client_id,
            model=model,
            enable_dp=config.enable_dp,
            dp_epsilon=config.dp_epsilon,
        )

        self.is_active = True
        self.is_straggler = False
        self.local_metrics: List[Dict] = []

    def _create_local_data(self):
        """Create synthetic local data with heterogeneity."""
        # Base distribution
        base_data = np.random.randn(self.config.num_samples, 28, 28)

        # Apply distribution shift based on heterogeneity
        shift = self.config.data_heterogeneity * np.random.randn()
        self.local_data = base_data + shift

        logger.info(
            f"Client {self.config.client_id}: {self.config.num_samples} samples "
            f"(heterogeneity={self.config.data_heterogeneity:.2f})"
        )

    def simulate_dropout(self) -> bool:
        """Check if client drops out this round."""
        if np.random.random() < self.config.dropout_probability:
            self.is_active = False
            logger.warning(f"Client {self.config.client_id} dropped out")
            return True

        self.is_active = True
        return False

    def simulate_straggler(self) -> bool:
        """Check if client is a straggler this round."""
        if np.random.random() < self.config.straggler_probability:
            self.is_straggler = True
            logger.warning(f"Client {self.config.client_id} is straggler")
            return True

        self.is_straggler = False
        return False

    def train_round(
        self,
        global_parameters: Dict[str, np.ndarray],
        num_epochs: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Train for one round.

        Args:
            global_parameters: Global model parameters
            num_epochs: Local training epochs

        Returns:
            Training result or None if dropout
        """
        # Simulate dropout
        if self.simulate_dropout():
            return None

        # Simulate stragglers (add delay)
        if self.simulate_straggler():
            time.sleep(0.5)

        # Set global parameters
        self.trainer.set_model_parameters(global_parameters)

        # Create batches from local data
        batches = [
            self.local_data[i:i+32]
            for i in range(0, len(self.local_data), 32)
        ]

        # Train
        metrics_list = []
        for epoch in range(num_epochs):
            metrics = self.trainer.train_epoch(batches)
            metrics_list.append(metrics)

        # Get updated parameters
        updated_params = self.trainer.get_model_parameters()

        return {
            "client_id": self.config.client_id,
            "parameters": updated_params,
            "num_samples": len(self.local_data),
            "metrics": metrics_list,
            "is_straggler": self.is_straggler,
        }


class FederationSimulator:
    """Simulates multi-client federated learning."""

    def __init__(
        self,
        model: nn.Module,
        num_clients: int = 10,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.FED_AVG,
        device: str = "cpu",
    ):
        """
        Initialize simulator.

        Args:
            model: Global model template
            num_clients: Number of clients to simulate
            aggregation_strategy: Aggregation strategy
            device: Torch device
        """
        self.model = model
        self.num_clients = num_clients
        self.device = device

        self.server = FederatedServer(
            model=model,
            aggregation_strategy=aggregation_strategy,
        )

        self.aggregator = ModelAggregator(
            strategy=aggregation_strategy,
            model_template=model,
        )

        self.serializer = SecureSerializer()

        self.clients: List[SimulatedClient] = []
        self.simulation_metrics: List[SimulationMetrics] = []

        logger.info(
            f"FederationSimulator initialized: "
            f"{num_clients} clients, strategy={aggregation_strategy.value}"
        )

    def create_clients(
        self,
        configs: Optional[List[ClientSimulationConfig]] = None,
    ):
        """
        Create simulated clients.

        Args:
            configs: List of client configurations
        """
        if configs is None:
            # Create default configs
            configs = [
                ClientSimulationConfig(
                    client_id=f"client_{i}",
                    num_samples=np.random.randint(100, 500),
                    iid_level=np.random.uniform(0.3, 0.9),
                    data_heterogeneity=np.random.exponential(0.5),
                    straggler_probability=0.1 if i % 5 == 0 else 0.0,
                    dropout_probability=0.05,
                )
                for i in range(self.num_clients)
            ]

        self.clients = [
            SimulatedClient(
                config=config,
                model=type(self.model)(),  # Create new model instance
                device=self.device,
            )
            for config in configs
        ]

        logger.info(f"Created {len(self.clients)} clients")

    def run_simulation(
        self,
        num_rounds: int = 10,
        local_epochs: int = 1,
    ) -> List[SimulationMetrics]:
        """
        Run federated learning simulation.

        Args:
            num_rounds: Number of federation rounds
            local_epochs: Local training epochs per round

        Returns:
            List of simulation metrics
        """
        logger.info(f"Starting simulation: {num_rounds} rounds, {local_epochs} local epochs")

        for round_num in range(num_rounds):
            round_metrics = self._simulate_round(
                round_num,
                local_epochs,
            )
            self.simulation_metrics.append(round_metrics)

            # Log round summary
            logger.info(
                f"Round {round_num} | "
                f"Active clients: {round_metrics.num_active_clients} | "
                f"Loss: {round_metrics.global_loss:.4f} | "
                f"Accuracy: {round_metrics.global_accuracy:.4f} | "
                f"Comm: {round_metrics.communication_cost:,} bytes | "
                f"ε={round_metrics.privacy_epsilon:.4f}"
            )

        return self.simulation_metrics

    def _simulate_round(
        self,
        round_num: int,
        local_epochs: int,
    ) -> SimulationMetrics:
        """
        Simulate one federation round.

        Args:
            round_num: Round number
            local_epochs: Local epochs

        Returns:
            Round metrics
        """
        start_time = time.time()

        # Get current global model
        global_params = self._get_global_parameters()

        # Client training
        client_results = []
        active_clients = 0
        stragglers = 0

        for client in self.clients:
            result = client.train_round(global_params, local_epochs)

            if result:
                client_results.append(result)
                active_clients += 1
                if result["is_straggler"]:
                    stragglers += 1

        # Aggregation
        if client_results:
            # Create update objects for aggregation
            from .federated_server import ClientUpdate

            updates = [
                ClientUpdate(
                    client_id=r["client_id"],
                    round_number=round_num,
                    parameters=r["parameters"],
                    num_samples=r["num_samples"],
                )
                for r in client_results
            ]

            # Aggregate
            aggregated_params, convergence = self.aggregator.aggregate(updates)

            # Update global model
            self._set_global_parameters(aggregated_params)

        # Compute communication cost
        comm_cost = self._compute_communication_cost(client_results)

        # Compute privacy budget
        privacy_epsilon, privacy_delta = self._compute_privacy_budget()

        # Evaluate
        global_loss, global_accuracy = self._evaluate_global_model()

        # Create metrics
        metrics = SimulationMetrics(
            round_number=round_num,
            global_accuracy=global_accuracy,
            global_loss=global_loss,
            communication_cost=comm_cost,
            privacy_epsilon=privacy_epsilon,
            privacy_delta=privacy_delta,
            num_active_clients=active_clients,
            num_stragglers=stragglers,
            convergence_metric=convergence if client_results else 0,
            training_time=time.time() - start_time,
        )

        return metrics

    def _get_global_parameters(self) -> Dict[str, np.ndarray]:
        """Get current global model parameters."""
        params = {}
        for name, param in self.model.named_parameters():
            params[name] = param.cpu().detach().numpy()
        return params

    def _set_global_parameters(self, params: Dict[str, np.ndarray]):
        """Set global model parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.copy_(
                        torch.tensor(params[name], dtype=param.dtype)
                    )

    def _compute_communication_cost(
        self,
        client_results: List[Dict],
    ) -> int:
        """Compute total communication cost."""
        cost = 0
        for result in client_results:
            # Serialize to estimate size
            serialized = self.serializer.serialize_parameters(
                parameters=result["parameters"],
                client_id=result["client_id"],
                round_number=0,
            )
            cost += len(serialized.model_parameters)

        return cost

    def _compute_privacy_budget(self) -> Tuple[float, float]:
        """Compute current privacy budget."""
        # Aggregate from all clients
        total_epsilon = 0
        total_delta = 0

        for client in self.clients:
            if hasattr(client.trainer, 'dp_manager') and client.trainer.dp_manager:
                ep, de = client.trainer.dp_manager.privacy_budget.remaining()
                total_epsilon += ep
                total_delta += de

        avg_epsilon = total_epsilon / (len(self.clients) + 1e-10)
        avg_delta = total_delta / (len(self.clients) + 1e-10)

        return avg_epsilon, avg_delta

    def _evaluate_global_model(self) -> Tuple[float, float]:
        """Evaluate global model (placeholder)."""
        # In real implementation, would evaluate on server test set
        return np.random.random(), np.random.random()

    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get complete simulation summary."""
        metrics = self.simulation_metrics

        if not metrics:
            return {}

        losses = [m.global_loss for m in metrics]
        accuracies = [m.global_accuracy for m in metrics]
        comm_costs = [m.communication_cost for m in metrics]
        epsilons = [m.privacy_epsilon for m in metrics]

        return {
            "num_rounds": len(metrics),
            "num_clients": self.num_clients,
            "final_loss": losses[-1],
            "final_accuracy": accuracies[-1],
            "avg_loss": np.mean(losses),
            "avg_accuracy": np.mean(accuracies),
            "total_communication": sum(comm_costs),
            "avg_communication_per_round": np.mean(comm_costs),
            "min_epsilon_reached": min(epsilons),
            "metrics_history": metrics,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Simulation module loaded")
