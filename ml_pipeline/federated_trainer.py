"""
Federated Learning Integration with ML Pipeline
Bridges federated training with existing pipeline infrastructure.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from federated.federated_client import FederatedClient, ClientTrainer
from federated.federated_server import FederatedServer, AggregationStrategy
from federated.flower_integration import FlowerCoordinator, FlowerConfig

logger = logging.getLogger(__name__)


@dataclass
class FederatedTrainingConfig:
    """Configuration for federated training."""

    # Federated parameters
    num_federation_rounds: int = 10
    clients_per_round: int = 5
    local_epochs_per_round: int = 2
    learning_rate: float = 0.01

    # Privacy parameters
    enable_differential_privacy: bool = True
    target_epsilon: float = 1.0
    target_delta: float = 1e-5
    gradient_clipping_norm: float = 1.0

    # Communication parameters
    compression_enabled: bool = True
    compression_ratio: float = 0.1

    # Aggregation parameters
    aggregation_strategy: str = "fedavg"
    byzantine_robust: bool = False
    min_samples_for_aggregation: int = 2

    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_rounds: int = 5
    early_stopping_threshold: float = 0.001

    # Model checkpoint
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5
    checkpoint_dir: str = "./federated_checkpoints"


class FederatedTrainingMetrics:
    """Track metrics during federated training."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.round_metrics: List[Dict[str, Any]] = []
        self.client_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.privacy_budget_history: List[float] = []

    def add_round_metric(
        self,
        round_num: int,
        loss: float,
        accuracy: float,
        num_active_clients: int,
        communication_cost: int,
        convergence_metric: float = 0.0,
    ):
        """Add metric for a federation round."""
        metric = {
            "round": round_num,
            "loss": loss,
            "accuracy": accuracy,
            "num_active_clients": num_active_clients,
            "communication_cost": communication_cost,
            "convergence_metric": convergence_metric,
        }
        self.round_metrics.append(metric)

        logger.info(
            f"Round {round_num}: Loss={loss:.4f}, Accuracy={accuracy:.4f}, "
            f"Clients={num_active_clients}, Comm={communication_cost/1024:.1f}KB"
        )

    def add_client_metric(
        self,
        client_id: str,
        round_num: int,
        local_loss: float,
        local_accuracy: float,
        training_time: float,
    ):
        """Add metric for client training."""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = []

        metric = {
            "round": round_num,
            "local_loss": local_loss,
            "local_accuracy": local_accuracy,
            "training_time": training_time,
        }
        self.client_metrics[client_id].append(metric)

    def add_privacy_epsilon(self, epsilon: float):
        """Add current privacy epsilon value."""
        self.privacy_budget_history.append(epsilon)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if not self.round_metrics:
            return {}

        losses = [m["loss"] for m in self.round_metrics]
        accuracies = [m["accuracy"] for m in self.round_metrics]

        return {
            "final_loss": losses[-1],
            "final_accuracy": accuracies[-1],
            "avg_accuracy": np.mean(accuracies),
            "accuracy_improvement": accuracies[-1] - accuracies[0] if accuracies else 0,
            "min_epsilon": min(self.privacy_budget_history) if self.privacy_budget_history else float('inf'),
            "total_rounds": len(self.round_metrics),
        }


class FederatedTrainer:
    """Unified federated learning trainer."""

    def __init__(
        self,
        model: nn.Module,
        config: FederatedTrainingConfig,
        device: str = "cpu",
    ):
        """
        Initialize federated trainer.

        Args:
            model: PyTorch model
            config: Training configuration
            device: Device for training
        """
        self.model = model
        self.config = config
        self.device = device

        self.server: Optional[FederatedServer] = None
        self.coordinator: Optional[FlowerCoordinator] = None
        self.metrics = FederatedTrainingMetrics()

        logger.info(f"Initialized FederatedTrainer on {device}")

    def create_federated_clients(
        self,
        data_loaders: Dict[str, DataLoader],
        client_ids: Optional[List[str]] = None,
    ) -> Dict[str, FederatedClient]:
        """
        Create federated clients from data loaders.

        Args:
            data_loaders: Dictionary of client_id -> DataLoader
            client_ids: List of client IDs (use keys from data_loaders if None)

        Returns:
            Dictionary of client_id -> FederatedClient
        """
        if client_ids is None:
            client_ids = list(data_loaders.keys())

        clients = {}

        for client_id in client_ids:
            if client_id not in data_loaders:
                logger.warning(f"No data loader for client {client_id}")
                continue

            # Create client
            client = FederatedClient(
                client_id=client_id,
                model=self.model,
                train_loader=data_loaders[client_id],
                device=self.device,
                learning_rate=self.config.learning_rate,
                enable_dp=self.config.enable_differential_privacy,
                target_epsilon=self.config.target_epsilon,
            )

            clients[client_id] = client

        logger.info(f"Created {len(clients)} federated clients")

        return clients

    def setup_server(
        self,
        test_loader: Optional[DataLoader] = None,
    ) -> FederatedServer:
        """
        Setup federated server.

        Args:
            test_loader: Optional test DataLoader for evaluation

        Returns:
            FederatedServer instance
        """
        aggregation_strategy = AggregationStrategy[
            self.config.aggregation_strategy.upper()
        ]

        self.server = FederatedServer(
            model=self.model,
            aggregation_strategy=aggregation_strategy,
            device=self.device,
            test_loader=test_loader,
            early_stopping_patience=self.config.early_stopping_rounds,
        )

        logger.info(f"Server configured with {aggregation_strategy} aggregation")

        return self.server

    def run_federated_learning(
        self,
        clients: Dict[str, FederatedClient],
        server: FederatedServer,
        num_rounds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run federated learning rounds.

        Args:
            clients: Dictionary of clients
            server: Server instance
            num_rounds: Number of rounds (use config if None)

        Returns:
            Training results
        """
        if num_rounds is None:
            num_rounds = self.config.num_federation_rounds

        logger.info(f"Starting federated learning: {num_rounds} rounds")

        client_ids = list(clients.keys())

        for round_num in range(num_rounds):
            logger.info(f"\n--- Federation Round {round_num + 1}/{num_rounds} ---")

            # Select clients for this round
            selected_clients = self._select_clients(client_ids)

            # Local training
            client_updates = self._train_clients(
                selected_clients, clients, round_num
            )

            # Server aggregation
            if client_updates:
                aggregation_result = server.aggregate_round(client_updates)

                # Update metrics
                self.metrics.add_round_metric(
                    round_num=round_num + 1,
                    loss=aggregation_result.convergence_metric,
                    accuracy=0.0,  # Placeholder
                    num_active_clients=len(client_updates),
                    communication_cost=sum(
                        update.num_samples * 10000 for update in client_updates
                    ),
                    convergence_metric=aggregation_result.convergence_metric,
                )

                # Broadcast updated model
                global_state = server.get_global_model()
                for client in [clients[cid] for cid in selected_clients]:
                    client.set_model_parameters(global_state)

            # Early stopping check
            if (
                self.config.early_stopping_enabled
                and round_num > self.config.early_stopping_rounds
            ):
                recent_metrics = self.metrics.round_metrics[
                    -self.config.early_stopping_rounds :
                ]
                if self._check_convergence(recent_metrics):
                    logger.info(f"Early stopping at round {round_num + 1}")
                    break

            # Checkpoint saving
            if (
                self.config.save_checkpoints
                and (round_num + 1) % self.config.checkpoint_frequency == 0
            ):
                self._save_checkpoint(round_num + 1, server)

        logger.info("Federated learning completed")

        return self.metrics.get_summary()

    def _select_clients(self, client_ids: List[str]) -> List[str]:
        """Select clients for current round."""
        num_select = min(self.config.clients_per_round, len(client_ids))
        return np.random.choice(client_ids, num_select, replace=False).tolist()

    def _train_clients(
        self,
        selected_clients: List[str],
        clients: Dict[str, FederatedClient],
        round_num: int,
    ) -> List:
        """Train selected clients and collect updates."""
        client_updates = []

        for client_id in selected_clients:
            if client_id not in clients:
                continue

            client = clients[client_id]

            try:
                # Train locally
                for epoch in range(self.config.local_epochs_per_round):
                    metrics = client.train_local(epoch=epoch)

                    self.metrics.add_client_metric(
                        client_id=client_id,
                        round_num=round_num,
                        local_loss=metrics.loss,
                        local_accuracy=metrics.accuracy,
                        training_time=metrics.training_time,
                    )

                # Get update
                update = client.get_model_update()
                client_updates.append(update)

            except Exception as e:
                logger.warning(f"Client {client_id} training failed: {e}")

        return client_updates

    def _check_convergence(self, recent_metrics: List[Dict]) -> bool:
        """Check if training has converged."""
        if len(recent_metrics) < 2:
            return False

        losses = [m["loss"] for m in recent_metrics]
        improvement = losses[0] - losses[-1]

        return improvement < self.config.early_stopping_threshold

    def _save_checkpoint(self, round_num: int, server: FederatedServer):
        """Save model checkpoint."""
        try:
            import os
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)

            checkpoint_path = f"{self.config.checkpoint_dir}/round_{round_num}.pt"
            torch.save(server.global_model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            "config": {
                "num_rounds": self.config.num_federation_rounds,
                "clients_per_round": self.config.clients_per_round,
                "local_epochs": self.config.local_epochs_per_round,
                "aggregation": self.config.aggregation_strategy,
                "dp_enabled": self.config.enable_differential_privacy,
            },
            "results": self.metrics.get_summary(),
            "detailed_metrics": self.metrics.round_metrics,
        }


def compare_training_approaches(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Compare federated vs centralized training.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        device: Device for training

    Returns:
        Comparison results
    """
    comparison = {}

    # 1. Centralized training
    logger.info("\n=== CENTRALIZED TRAINING ===")
    centralized_model = model.__class__()
    centralized_trainer = _train_centralized(
        centralized_model, train_loader, test_loader, device
    )
    comparison["centralized"] = centralized_trainer

    # 2. Federated training
    logger.info("\n=== FEDERATED TRAINING ===")
    federated_model = model.__class__()
    config = FederatedTrainingConfig(num_federation_rounds=10)
    federated_trainer = FederatedTrainer(federated_model, config, device)

    # Note: Would need to split data into clients for actual test
    comparison["federated"] = {
        "config": config.__dict__,
    }

    return comparison


def _train_centralized(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    num_epochs: int = 10,
) -> Dict[str, Any]:
    """Helper function for centralized training."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    results = {"losses": [], "accuracies": []}

    model.to(device)

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Testing
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        results["losses"].append(avg_loss)
        results["accuracies"].append(accuracy)

        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Create trainer
    model = SimpleModel()
    config = FederatedTrainingConfig()
    trainer = FederatedTrainer(model, config, device="cpu")

    logger.info("FederatedTrainer created and ready for use")
