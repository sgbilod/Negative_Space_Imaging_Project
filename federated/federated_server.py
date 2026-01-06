"""
Federated Server Implementation
Model aggregation, client management, and global evaluation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Aggregation strategies for federated learning."""

    FED_AVG = "fedavg"  # Basic averaging
    WEIGHTED_AVG = "weighted_avg"  # Weighted by sample count
    MED_AVG = "median"  # Median aggregation
    TRIMMED_MEAN = "trimmed_mean"  # Robust to outliers
    KRUM = "krum"  # Byzantine-robust


@dataclass
class ClientUpdate:
    """Client model update."""

    client_id: str
    round_number: int
    parameters: Dict[str, np.ndarray]
    num_samples: int
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AggregationResult:
    """Result of model aggregation."""

    round_number: int
    global_parameters: Dict[str, np.ndarray]
    num_clients: int
    convergence_metric: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ModelAggregator:
    """
    Aggregates model updates from multiple clients.
    Implements various aggregation strategies with robustness.
    """

    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.FED_AVG,
        model_template: Optional[nn.Module] = None,
    ):
        """
        Initialize aggregator.

        Args:
            strategy: Aggregation strategy to use
            model_template: Template model for compatibility checking
        """
        self.strategy = strategy
        self.model_template = model_template
        self.aggregation_history: List[AggregationResult] = []

    def aggregate(
        self,
        updates: List[ClientUpdate],
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Aggregate client updates into global model.

        Args:
            updates: List of client updates

        Returns:
            (aggregated_parameters, convergence_metric) tuple
        """
        if not updates:
            logger.error("No updates to aggregate")
            return {}, float('inf')

        if self.strategy == AggregationStrategy.FED_AVG:
            return self._fedavg(updates)
        elif self.strategy == AggregationStrategy.WEIGHTED_AVG:
            return self._weighted_avg(updates)
        elif self.strategy == AggregationStrategy.MED_AVG:
            return self._median_aggregation(updates)
        elif self.strategy == AggregationStrategy.TRIMMED_MEAN:
            return self._trimmed_mean(updates)
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using FedAvg")
            return self._fedavg(updates)

    def _fedavg(
        self,
        updates: List[ClientUpdate],
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Standard FedAvg aggregation (simple averaging).

        Args:
            updates: Client updates

        Returns:
            (aggregated_parameters, convergence_metric) tuple
        """
        if not updates:
            return {}, float('inf')

        aggregated = {}
        param_names = updates[0].parameters.keys()

        for param_name in param_names:
            stacked = np.stack([
                update.parameters[param_name]
                for update in updates
            ])

            aggregated[param_name] = np.mean(stacked, axis=0)

        # Compute convergence metric (variance of parameter updates)
        convergence = self._compute_convergence_metric(updates, aggregated)

        logger.info(
            f"FedAvg aggregation: {len(updates)} clients, "
            f"convergence={convergence:.6f}"
        )

        return aggregated, convergence

    def _weighted_avg(
        self,
        updates: List[ClientUpdate],
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Weighted averaging by number of samples.

        Args:
            updates: Client updates

        Returns:
            (aggregated_parameters, convergence_metric) tuple
        """
        total_samples = sum(update.num_samples for update in updates)

        aggregated = {}
        param_names = updates[0].parameters.keys()

        for param_name in param_names:
            weighted_param = None

            for update in updates:
                weight = update.num_samples / total_samples
                weighted = update.parameters[param_name] * weight

                if weighted_param is None:
                    weighted_param = weighted
                else:
                    weighted_param += weighted

            aggregated[param_name] = weighted_param

        convergence = self._compute_convergence_metric(updates, aggregated)

        logger.info(
            f"Weighted aggregation: {len(updates)} clients, "
            f"total_samples={total_samples}, convergence={convergence:.6f}"
        )

        return aggregated, convergence

    def _median_aggregation(
        self,
        updates: List[ClientUpdate],
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Robust median aggregation.

        Args:
            updates: Client updates

        Returns:
            (aggregated_parameters, convergence_metric) tuple
        """
        aggregated = {}
        param_names = updates[0].parameters.keys()

        for param_name in param_names:
            stacked = np.stack([
                update.parameters[param_name]
                for update in updates
            ])

            aggregated[param_name] = np.median(stacked, axis=0)

        convergence = self._compute_convergence_metric(updates, aggregated)

        logger.info(
            f"Median aggregation: {len(updates)} clients, "
            f"convergence={convergence:.6f}"
        )

        return aggregated, convergence

    def _trimmed_mean(
        self,
        updates: List[ClientUpdate],
        trim_fraction: float = 0.1,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Trimmed mean aggregation (robust to outliers).

        Args:
            updates: Client updates
            trim_fraction: Fraction to trim from both ends

        Returns:
            (aggregated_parameters, convergence_metric) tuple
        """
        aggregated = {}
        param_names = updates[0].parameters.keys()

        for param_name in param_names:
            stacked = np.stack([
                update.parameters[param_name]
                for update in updates
            ])

            # Trim and compute mean
            aggregated[param_name] = np.mean(
                np.sort(stacked, axis=0)[
                    int(len(updates) * trim_fraction):
                    int(len(updates) * (1 - trim_fraction))
                ],
                axis=0,
            )

        convergence = self._compute_convergence_metric(updates, aggregated)

        logger.info(
            f"Trimmed mean aggregation: {len(updates)} clients, "
            f"trim={trim_fraction}, convergence={convergence:.6f}"
        )

        return aggregated, convergence

    def _compute_convergence_metric(
        self,
        updates: List[ClientUpdate],
        aggregated: Dict[str, np.ndarray],
    ) -> float:
        """
        Compute convergence metric (variance of updates).

        Args:
            updates: Client updates
            aggregated: Aggregated parameters

        Returns:
            Convergence metric (lower is better)
        """
        total_variance = 0
        num_params = 0

        for param_name in aggregated.keys():
            for update in updates:
                diff = update.parameters[param_name] - aggregated[param_name]
                total_variance += np.sum(diff ** 2)
                num_params += diff.size

        convergence = np.sqrt(total_variance / (num_params + 1e-10))

        return convergence


class GlobalEvaluator:
    """
    Evaluates global model on server-side test data.
    """

    def __init__(
        self,
        model: nn.Module,
        test_data: Optional[np.ndarray] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize global evaluator.

        Args:
            model: Model to evaluate
            test_data: Server-side test data (optional)
            device: Device to run evaluation on
        """
        self.model = model
        self.test_data = test_data

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.evaluation_history: List[Dict] = []

    def evaluate(self) -> Optional[Dict[str, float]]:
        """
        Evaluate global model on test data.

        Args:
            Returns:
            Evaluation metrics dictionary
        """
        if self.test_data is None:
            logger.warning("No test data available for global evaluation")
            return None

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.test_data:
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

        metrics = {
            "loss": avg_loss,
            "accuracy": 0.0,  # Placeholder
            "timestamp": datetime.now().isoformat(),
        }

        self.evaluation_history.append(metrics)

        logger.info(f"Global evaluation: loss={avg_loss:.4f}")

        return metrics

    def update_model(self, parameters: Dict[str, np.ndarray]):
        """
        Update model with new parameters.

        Args:
            parameters: New model parameters
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


class FederatedServer:
    """
    Main federated learning server interface.
    Coordinates aggregation, evaluation, and global model management.
    """

    def __init__(
        self,
        model: nn.Module,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.FED_AVG,
        test_data: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 10,
        early_stopping_threshold: float = 1e-4,
    ):
        """
        Initialize federated server.

        Args:
            model: Global model
            aggregation_strategy: Strategy for aggregating updates
            test_data: Server-side test data (optional)
            early_stopping_rounds: Rounds without improvement before stopping
            early_stopping_threshold: Convergence threshold for early stopping
        """
        self.model = model
        self.aggregation_strategy = aggregation_strategy

        self.aggregator = ModelAggregator(
            strategy=aggregation_strategy,
            model_template=model,
        )

        self.evaluator = GlobalEvaluator(
            model=model,
            test_data=test_data,
        )

        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_threshold = early_stopping_threshold

        self.current_round = 0
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0

        self.round_history: List[Dict] = []

    def aggregate_round(
        self,
        client_updates: List[ClientUpdate],
    ) -> AggregationResult:
        """
        Aggregate a federated learning round.

        Args:
            client_updates: Updates from all participating clients

        Returns:
            AggregationResult
        """
        logger.info(f"Aggregating round {self.current_round} from {len(client_updates)} clients")

        # Aggregate parameters
        aggregated_params, convergence = self.aggregator.aggregate(
            client_updates
        )

        # Update global model
        self.evaluator.update_model(aggregated_params)

        # Evaluate on server test data
        metrics = self.evaluator.evaluate()

        # Check early stopping
        if metrics:
            current_loss = metrics.get("loss", float('inf'))

            if current_loss < self.best_loss - self.early_stopping_threshold:
                self.best_loss = current_loss
                self.rounds_without_improvement = 0
            else:
                self.rounds_without_improvement += 1

        # Store result
        result = AggregationResult(
            round_number=self.current_round,
            global_parameters=aggregated_params,
            num_clients=len(client_updates),
            convergence_metric=convergence,
        )

        # Log round
        round_log = {
            "round": self.current_round,
            "num_clients": len(client_updates),
            "convergence": convergence,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        self.round_history.append(round_log)
        self.current_round += 1

        return result

    def should_stop_early(self) -> bool:
        """
        Check if early stopping conditions are met.

        Args:
            Returns:
            True if should stop early
        """
        return (
            self.rounds_without_improvement >= self.early_stopping_rounds
        )

    def get_global_model(self) -> Dict[str, np.ndarray]:
        """
        Get current global model parameters.

        Args:
            Returns:
            Dictionary of model parameters
        """
        parameters = {}
        for name, param in self.model.named_parameters():
            parameters[name] = param.cpu().detach().numpy()

        return parameters

    def get_server_status(self) -> Dict[str, Any]:
        """
        Get current server status and statistics.

        Args:
            Returns:
            Status dictionary
        """
        return {
            "current_round": self.current_round,
            "best_loss": self.best_loss,
            "rounds_without_improvement": self.rounds_without_improvement,
            "should_stop": self.should_stop_early(),
            "num_rounds_completed": len(self.round_history),
            "evaluation_history": self.evaluator.evaluation_history,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    print("Federated Server module loaded")
