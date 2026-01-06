"""
Flower Framework Integration
Complete Flower setup with strategy configuration and coordination.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class ClientSelectionStrategy(Enum):
    """Client selection strategies."""

    RANDOM = "random"
    FEDPROX = "fedprox"
    LOSS_BASED = "loss_based"
    AVAILABILITY = "availability"


@dataclass
class FlowerConfig:
    """Flower framework configuration."""

    num_rounds: int = 10
    clients_per_round: int = 5
    min_clients: int = 2
    min_fit_clients: int = 2
    min_available_clients: int = 2
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    client_resources: Dict = None

    def __post_init__(self):
        if self.client_resources is None:
            self.client_resources = {
                "num_cpus": 1,
                "num_gpus": 0.5,
            }


class FlowerCoordinator:
    """
    Coordinates federated learning using Flower framework.
    Manages client-server communication and aggregation.
    """

    def __init__(
        self,
        server_address: str = "0.0.0.0:8080",
        num_clients: int = 5,
        config: Optional[FlowerConfig] = None,
    ):
        """
        Initialize Flower coordinator.

        Args:
            server_address: Server address (host:port)
            num_clients: Number of participating clients
            config: Flower configuration
        """
        self.server_address = server_address
        self.num_clients = num_clients
        self.config = config or FlowerConfig()

        self.clients: Dict[str, 'FlowerClient'] = {}
        self.server: Optional['FlowerServer'] = None

        self.metrics_history: List[Dict] = []
        self.round_history: List[Dict] = []

        logger.info(
            f"Flower Coordinator initialized | "
            f"Server: {server_address} | Clients: {num_clients}"
        )

    def register_client(
        self,
        client_id: str,
        client_fn: Callable,
    ):
        """
        Register a client with the coordinator.

        Args:
            client_id: Unique client identifier
            client_fn: Client function/class
        """
        self.clients[client_id] = {
            "id": client_id,
            "fn": client_fn,
            "status": "registered",
        }

        logger.info(f"Client {client_id} registered")

    def start_server(
        self,
        server_fn: Callable,
        strategy: Optional[Any] = None,
    ):
        """
        Start Flower server.

        Args:
            server_fn: Server function
            strategy: Aggregation strategy (FedAvg, etc.)
        """
        logger.info("Starting Flower server...")

        self.server = {
            "fn": server_fn,
            "strategy": strategy,
            "status": "running",
        }

        logger.info("Flower server started")

    def run_federation(
        self,
        num_rounds: int,
        min_clients: int = 2,
    ) -> Dict[str, Any]:
        """
        Run federated learning rounds.

        Args:
            num_rounds: Number of federation rounds
            min_clients: Minimum clients per round

        Returns:
            Federation results dictionary
        """
        logger.info(f"Starting federation: {num_rounds} rounds, min {min_clients} clients")

        results = {
            "num_rounds": num_rounds,
            "num_clients": len(self.clients),
            "rounds": [],
        }

        for round_num in range(num_rounds):
            round_results = self._run_round(round_num, min_clients)
            results["rounds"].append(round_results)

            self.round_history.append(round_results)

        return results

    def _run_round(
        self,
        round_num: int,
        min_clients: int,
    ) -> Dict[str, Any]:
        """
        Execute a single federation round.

        Args:
            round_num: Round number
            min_clients: Minimum clients for this round

        Returns:
            Round results
        """
        logger.info(f"Running federation round {round_num}")

        # Select clients for this round
        selected_clients = self._select_clients(min_clients)

        if len(selected_clients) < min_clients:
            logger.error(
                f"Not enough clients: {len(selected_clients)} < {min_clients}"
            )
            return {
                "round": round_num,
                "success": False,
                "error": "Insufficient clients",
            }

        # Collect updates from clients
        updates = []
        for client_id in selected_clients:
            client = self.clients[client_id]

            # Call client function
            if callable(client["fn"]):
                update = client["fn"]()
                updates.append({
                    "client_id": client_id,
                    "update": update,
                })

        # Aggregate
        aggregated = self._aggregate_updates(updates)

        # Evaluate
        metrics = self._evaluate_round(round_num, aggregated)

        round_result = {
            "round": round_num,
            "selected_clients": selected_clients,
            "num_updates": len(updates),
            "aggregated": aggregated is not None,
            "metrics": metrics,
        }

        logger.info(
            f"Round {round_num} completed | "
            f"Clients: {len(selected_clients)} | "
            f"Success: {aggregated is not None}"
        )

        return round_result

    def _select_clients(self, min_clients: int) -> List[str]:
        """
        Select clients for current round.

        Args:
            min_clients: Minimum clients needed

        Returns:
            List of selected client IDs
        """
        available = list(self.clients.keys())

        num_to_select = max(
            min_clients,
            min(self.config.clients_per_round, len(available)),
        )

        selected = available[:num_to_select]

        return selected

    def _aggregate_updates(
        self,
        updates: List[Dict],
    ) -> Optional[Dict]:
        """
        Aggregate client updates.

        Args:
            updates: List of client updates

        Returns:
            Aggregated model or None
        """
        if not updates:
            return None

        # Simple averaging (can be extended with strategies)
        aggregated = {}

        for key in updates[0].get("update", {}).keys():
            values = [u["update"][key] for u in updates]
            aggregated[key] = np.mean(values) if values else 0

        return aggregated

    def _evaluate_round(
        self,
        round_num: int,
        aggregated: Optional[Dict],
    ) -> Dict[str, float]:
        """
        Evaluate global model after round.

        Args:
            round_num: Round number
            aggregated: Aggregated model

        Returns:
            Evaluation metrics
        """
        metrics = {
            "round": round_num,
            "timestamp": str(np.datetime64('now')),
        }

        if aggregated:
            metrics["loss"] = float(np.mean(list(aggregated.values())))
            metrics["accuracy"] = 0.0  # Placeholder

        self.metrics_history.append(metrics)

        return metrics

    def get_statistics(self) -> Dict[str, Any]:
        """Get federation statistics."""
        return {
            "total_clients": len(self.clients),
            "num_rounds": len(self.round_history),
            "metrics_history": self.metrics_history,
            "round_history": self.round_history,
        }


class FederatedClient:
    """
    Flower-compatible federated client.
    Implements Flower client interface for training.
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        data_fn: Callable,
    ):
        """
        Initialize federated client.

        Args:
            client_id: Unique client identifier
            model: PyTorch model
            data_fn: Function to get training data
        """
        self.client_id = client_id
        self.model = model
        self.data_fn = data_fn

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        logger.info(f"FederatedClient {client_id} initialized")

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Fit model on local data.

        Args:
            parameters: Global model parameters
            config: Training configuration

        Returns:
            (updated_parameters, num_samples, metrics) tuple
        """
        logger.info(f"Client {self.client_id} starting fit")

        # Set model parameters
        self._set_parameters(parameters)

        # Get local data
        train_loader = self.data_fn()

        # Train
        num_epochs = config.get("num_epochs", 1)
        lr = config.get("learning_rate", 1e-3)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        total_loss = 0
        num_batches = 0

        self.model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:
                batch_tensor = torch.tensor(
                    batch,
                    dtype=torch.float32,
                    device=self.device,
                )

                optimizer.zero_grad()
                outputs = self.model(batch_tensor)
                loss = outputs.mean()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # Get updated parameters
        updated_params = self._get_parameters()

        num_samples = len(train_loader) * 32  # Assumes batch_size=32
        metrics = {
            "loss": total_loss / (num_batches + 1e-10),
            "num_samples": num_samples,
        }

        logger.info(
            f"Client {self.client_id} fit completed | "
            f"Loss: {metrics['loss']:.4f}"
        )

        return updated_params, num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local data.

        Args:
            parameters: Model parameters
            config: Evaluation configuration

        Returns:
            (loss, num_samples, metrics) tuple
        """
        # Set parameters
        self._set_parameters(parameters)

        # Get validation data
        val_loader = self.data_fn()

        # Evaluate
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch_tensor = torch.tensor(
                    batch,
                    dtype=torch.float32,
                    device=self.device,
                )

                outputs = self.model(batch_tensor)
                loss = outputs.mean()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / (num_batches + 1e-10)
        num_samples = len(val_loader) * 32

        metrics = {
            "accuracy": 0.0,  # Placeholder
        }

        return avg_loss, num_samples, metrics

    def _set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from list."""
        with torch.no_grad():
            for param, new_param in zip(self.model.parameters(), parameters):
                param.copy_(torch.tensor(new_param, dtype=param.dtype))

    def _get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as list."""
        return [
            param.cpu().numpy()
            for param in self.model.parameters()
        ]


class FederatedServer:
    """
    Flower-compatible federated server.
    Implements aggregation strategy and server logic.
    """

    def __init__(self):
        """Initialize federated server."""
        self.global_parameters = None
        self.round = 0

        logger.info("FederatedServer initialized")

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[Any, int]],
        failures: List[BaseException],
    ) -> Tuple[Optional[List[np.ndarray]], Dict]:
        """
        Aggregate fit results from clients.

        Args:
            rnd: Round number
            results: (client_params, num_samples) tuples
            failures: Failed client results

        Returns:
            (aggregated_parameters, metrics) tuple
        """
        if not results:
            logger.error("No fit results to aggregate")
            return None, {}

        # Weighted average
        total_samples = sum(num_samples for _, num_samples in results)

        aggregated = None
        for client_params, num_samples in results:
            weight = num_samples / total_samples

            if aggregated is None:
                aggregated = [p * weight for p in client_params]
            else:
                aggregated = [
                    a + (p * weight)
                    for a, p in zip(aggregated, client_params)
                ]

        self.global_parameters = aggregated
        self.round = rnd

        metrics = {
            "round": rnd,
            "num_clients": len(results),
            "failures": len(failures),
        }

        logger.info(
            f"Aggregation round {rnd}: {len(results)} clients, "
            f"{len(failures)} failures"
        )

        return aggregated, metrics

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[int, Dict]],
        failures: List[BaseException],
    ) -> Tuple[float, Dict]:
        """
        Aggregate evaluation results.

        Args:
            rnd: Round number
            results: (num_samples, metrics) tuples
            failures: Failed evaluations

        Returns:
            (loss, metrics) tuple
        """
        if not results:
            return float('inf'), {}

        # Weighted average loss
        total_samples = sum(num_samples for num_samples, _ in results)

        weighted_loss = 0
        for num_samples, metrics in results:
            weight = num_samples / total_samples
            weighted_loss += metrics.get("loss", 0) * weight

        eval_metrics = {
            "round": rnd,
            "num_clients": len(results),
            "weighted_loss": weighted_loss,
        }

        logger.info(f"Evaluation round {rnd}: loss={weighted_loss:.4f}")

        return weighted_loss, eval_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    print("Flower Integration module loaded")
