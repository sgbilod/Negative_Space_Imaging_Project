"""
Federated Client Implementation
Local training with privacy constraints and communication protocol.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn

from .differential_privacy import DifferentialPrivacyManager
from .data_privacy import DataPrivacyManager, DataValidator
from .communication import SecureSerializer, CommunicationProtocol

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics from local training."""

    epoch: int
    loss: float
    accuracy: float
    privacy_epsilon: float
    privacy_delta: float
    training_time: float
    data_size: int
    batch_size: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LocalDataManager:
    """
    Manages local dataset for federated learning.
    Data never leaves the client.
    """

    def __init__(
        self,
        client_id: str,
        data_path: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize local data manager.

        Args:
            client_id: Unique client identifier
            data_path: Path to local training data
            batch_size: Batch size for training
        """
        self.client_id = client_id
        self.batch_size = batch_size

        self.data_manager = DataPrivacyManager(
            client_id=client_id,
        )
        self.validator = DataValidator()

        self.training_data = None
        self.test_data = None

        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path: str) -> bool:
        """Load local training data."""
        return self.data_manager.load_local_data(data_path)

    def get_batches(self, shuffle: bool = True) -> List[np.ndarray]:
        """Get training data as batches."""
        return self.data_manager.split_into_batches(
            self.batch_size,
            shuffle=shuffle,
        )

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of local data (privacy-safe)."""
        return self.data_manager.create_data_summary()


class ClientTrainer:
    """
    Trains model locally on client with privacy constraints.
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        enable_dp: bool = True,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
    ):
        """
        Initialize client trainer.

        Args:
            client_id: Client identifier
            model: PyTorch model
            optimizer: Optimizer type ("adam", "sgd")
            learning_rate: Learning rate
            enable_dp: Enable differential privacy
            dp_epsilon: Target epsilon for DP
            dp_delta: Target delta for DP
        """
        self.client_id = client_id
        self.model = model
        self.learning_rate = learning_rate
        self.enable_dp = enable_dp

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Setup optimizer
        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
            )
        else:
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
            )

        # Setup privacy
        if enable_dp:
            self.dp_manager = DifferentialPrivacyManager(
                target_epsilon=dp_epsilon,
                target_delta=dp_delta,
            )
        else:
            self.dp_manager = None

        self.training_metrics: List[TrainingMetrics] = []

        logger.info(
            f"Client {client_id} initialized | DP: {enable_dp} "
            f"(ε={dp_epsilon}, δ={dp_delta})"
        )

    def train_epoch(
        self,
        train_loader: List[np.ndarray],
        val_loader: Optional[List[np.ndarray]] = None,
    ) -> TrainingMetrics:
        """
        Train for one epoch on local data.

        Args:
            train_loader: List of training batches
            val_loader: Optional validation batches

        Returns:
            TrainingMetrics object
        """
        import time

        self.model.train()
        start_time = time.time()

        total_loss = 0
        num_batches = 0
        epoch_num = len(self.training_metrics)

        for batch in train_loader:
            # Convert to tensor
            batch_tensor = torch.tensor(
                batch,
                dtype=torch.float32,
                device=self.device,
            )

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_tensor)

            # Dummy loss (should be computed properly in real implementation)
            loss = outputs.mean()

            # Backward pass
            loss.backward()

            # Get gradients
            gradients = [p.grad.cpu().numpy() for p in self.model.parameters()]

            # Apply privacy protection
            if self.dp_manager:
                protected_grads = self.dp_manager.apply_privacy_protection(
                    gradients,
                    mechanism="dp-sgd",
                )

                # Set protected gradients
                for param, protected_grad in zip(
                    self.model.parameters(),
                    protected_grads,
                ):
                    param.grad = torch.tensor(
                        protected_grad,
                        dtype=torch.float32,
                        device=self.device,
                    )

            # Optimizer step
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Compute metrics
        avg_loss = total_loss / (num_batches + 1e-10)
        accuracy = 0.0  # Placeholder
        training_time = time.time() - start_time

        # Compute privacy budget used
        if self.dp_manager:
            privacy_epsilon = self.dp_manager.compute_epsilon(
                num_epochs=epoch_num + 1,
                dataset_size=len(train_loader) * 32,  # Assumes batch_size=32
            )
            privacy_delta = self.dp_manager.target_delta
        else:
            privacy_epsilon = float('inf')
            privacy_delta = 0.0

        metrics = TrainingMetrics(
            epoch=epoch_num,
            loss=avg_loss,
            accuracy=accuracy,
            privacy_epsilon=privacy_epsilon,
            privacy_delta=privacy_delta,
            training_time=training_time,
            data_size=len(train_loader) * 32,
            batch_size=32,
        )

        self.training_metrics.append(metrics)

        logger.info(
            f"Epoch {epoch_num} | Loss: {avg_loss:.4f} | "
            f"ε={privacy_epsilon:.4f} | Time: {training_time:.2f}s"
        )

        return metrics

    def validate(
        self,
        val_loader: List[np.ndarray],
    ) -> Tuple[float, float]:
        """
        Validate model on local validation data.

        Args:
            val_loader: Validation batches

        Returns:
            (loss, accuracy) tuple
        """
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
        accuracy = 0.0  # Placeholder

        return avg_loss, accuracy

    def get_model_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get current model parameters for transmission.

        Args:
            Returns:
            Dictionary of parameters
        """
        parameters = {}
        for name, param in self.model.named_parameters():
            parameters[name] = param.cpu().detach().numpy()

        return parameters

    def set_model_parameters(self, parameters: Dict[str, np.ndarray]):
        """
        Set model parameters from aggregated global model.

        Args:
            parameters: Dictionary of parameters
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.copy_(
                        torch.tensor(
                            parameters[name],
                            dtype=param.dtype,
                            device=self.device,
                        )
                    )


class FederatedClient:
    """
    Main federated learning client interface.
    Coordinates local training, privacy, and communication.
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        data_path: str,
        server_address: str = "localhost",
        server_port: int = 8883,
        **trainer_kwargs,
    ):
        """
        Initialize federated client.

        Args:
            client_id: Unique client identifier
            model: PyTorch model
            data_path: Path to local training data
            server_address: Federated learning server address
            server_port: Server port
            **trainer_kwargs: Additional arguments for ClientTrainer
        """
        self.client_id = client_id
        self.server_address = server_address
        self.server_port = server_port

        # Initialize components
        self.data_manager = LocalDataManager(
            client_id=client_id,
            data_path=data_path,
        )

        self.trainer = ClientTrainer(
            client_id=client_id,
            model=model,
            **trainer_kwargs,
        )

        self.serializer = SecureSerializer()
        self.communication = CommunicationProtocol(
            client_id=client_id,
            server_address=server_address,
            server_port=server_port,
        )

        self.round = 0
        self.session_history: List[Dict] = []

    def receive_global_model(self) -> Optional[Dict[str, np.ndarray]]:
        """Receive global model from server."""
        success, data = self.communication.receive_with_retry()

        if not success or data is None:
            logger.error("Failed to receive global model")
            return None

        parameters = self.serializer.deserialize_parameters(data)
        return parameters

    def train_local(
        self,
        num_epochs: int = 1,
    ) -> List[TrainingMetrics]:
        """
        Execute local training.

        Args:
            num_epochs: Number of local training epochs

        Returns:
            List of training metrics
        """
        metrics_list = []

        for epoch in range(num_epochs):
            # Get training batches
            batches = self.data_manager.get_batches(shuffle=True)

            # Train one epoch
            metrics = self.trainer.train_epoch(batches)
            metrics_list.append(metrics)

        return metrics_list

    def send_model_update(self) -> bool:
        """
        Send trained model parameters to server.

        Args:
            Returns:
            True if successful
        """
        # Get parameters
        parameters = self.trainer.get_model_parameters()

        # Serialize
        serialized = self.serializer.serialize_parameters(
            parameters=parameters,
            client_id=self.client_id,
            round_number=self.round,
        )

        # Send
        success, message = self.communication.send_with_retry(serialized)

        if success:
            logger.info(f"Model update sent: {message}")
        else:
            logger.error(f"Model update failed: {message}")

        return success

    def federated_round(
        self,
        num_local_epochs: int = 1,
    ) -> Dict[str, Any]:
        """
        Execute one federated learning round.

        Args:
            num_local_epochs: Number of local training epochs

        Returns:
            Round summary
        """
        start_time = datetime.now()

        logger.info(f"Starting federated round {self.round}")

        # Receive global model
        global_params = self.receive_global_model()
        if global_params:
            self.trainer.set_model_parameters(global_params)

        # Train locally
        metrics_list = self.train_local(num_epochs=num_local_epochs)

        # Send model update
        success = self.send_model_update()

        # Update round number
        self.round += 1

        # Log session
        duration = (datetime.now() - start_time).total_seconds()

        round_summary = {
            "round": self.round - 1,
            "success": success,
            "duration": duration,
            "metrics": metrics_list,
            "timestamp": datetime.now().isoformat(),
        }

        self.session_history.append(round_summary)

        logger.info(f"Federated round {self.round - 1} completed in {duration:.2f}s")

        return round_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    # Note: In real usage, provide actual model and data
    print("Federated Client module loaded")
