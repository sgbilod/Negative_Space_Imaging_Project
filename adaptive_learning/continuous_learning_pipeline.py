"""
Continuous Learning Pipeline for Negative Space Imaging

Handles online learning and incremental model updates with:
- Streaming data integration
- Sample weighting and importance sampling
- Continual learning without catastrophic forgetting
- Drift detection and adaptation
- Experience replay buffers
- Callback system for lifecycle events

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
import numpy as np
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExperienceReplayBuffer:
    """Experience replay buffer for continual learning."""

    def __init__(self, max_size: int = 10000, sampling_strategy: str = "uniform"):
        """Initialize replay buffer."""
        self.max_size = max_size
        self.sampling_strategy = sampling_strategy
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)

    def add(self, experience: Tuple[Any, Any, Any], priority: float = 1.0) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample_batch(self, batch_size: int) -> List[Tuple[Any, Any, Any]]:
        """Sample batch from buffer."""
        if len(self.buffer) == 0:
            return []

        if self.sampling_strategy == "uniform":
            indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        elif self.sampling_strategy == "priority":
            priorities = np.array(self.priorities)
            priorities = priorities / priorities.sum()
            indices = np.random.choice(
                len(self.buffer), min(batch_size, len(self.buffer)), p=priorities, replace=False
            )
        else:
            indices = np.arange(min(batch_size, len(self.buffer)))

        return [self.buffer[i] for i in indices]

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update experience priorities."""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority

    def get_size(self) -> int:
        """Get buffer size."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()
        self.priorities.clear()


class DriftDetector:
    """Detect distribution drift in data."""

    def __init__(self, window_size: int = 100, threshold: float = 0.1):
        """Initialize drift detector."""
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_mean = None
        self.baseline_std = None
        self.current_window = deque(maxlen=window_size)
        self.drift_detected = False

    def update(self, features: np.ndarray) -> bool:
        """Update detector with new features."""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self.baseline_mean is None:
            self.baseline_mean = features.mean(axis=0)
            self.baseline_std = features.std(axis=0)
            return False

        self.current_window.extend([f for f in features])

        if len(self.current_window) >= self.window_size:
            window_array = np.array(list(self.current_window))
            window_mean = window_array.mean(axis=0)

            # Detect drift using mean shift
            mean_shift = np.abs(window_mean - self.baseline_mean) / (self.baseline_std + 1e-6)
            max_shift = np.max(mean_shift)

            self.drift_detected = max_shift > self.threshold

            if self.drift_detected:
                logger.warning(f"Distribution drift detected: max_shift={max_shift:.4f}")
                # Update baseline
                self.baseline_mean = window_mean
                self.baseline_std = window_array.std(axis=0)

        return self.drift_detected

    def get_drift_score(self) -> float:
        """Get current drift score."""
        if self.baseline_mean is None or len(self.current_window) == 0:
            return 0.0

        window_array = np.array(list(self.current_window))
        window_mean = window_array.mean(axis=0)
        mean_shift = np.abs(window_mean - self.baseline_mean) / (self.baseline_std + 1e-6)
        return float(np.max(mean_shift))


class SampleWeighter:
    """Assign weights to samples for importance sampling."""

    def __init__(self, weighting_strategy: str = "uniform"):
        """Initialize sample weighter."""
        self.weighting_strategy = weighting_strategy
        self.sample_losses = deque(maxlen=1000)

    def compute_weights(self, losses: np.ndarray) -> np.ndarray:
        """Compute sample weights based on losses."""
        if self.weighting_strategy == "uniform":
            return np.ones(len(losses)) / len(losses)

        elif self.weighting_strategy == "inverse_loss":
            # Higher loss -> higher weight
            weights = 1.0 / (losses + 1e-6)
            return weights / weights.sum()

        elif self.weighting_strategy == "softmax":
            # Softmax of losses
            exp_losses = np.exp(losses / (np.std(losses) + 1e-6))
            return exp_losses / exp_losses.sum()

        else:
            return np.ones(len(losses)) / len(losses)

    def update_loss_history(self, losses: np.ndarray) -> None:
        """Update loss history."""
        self.sample_losses.extend(losses)

    def get_average_loss(self) -> float:
        """Get average loss."""
        if len(self.sample_losses) == 0:
            return 0.0
        return float(np.mean(list(self.sample_losses)))


class ContinualLearningCallback:
    """Base callback for continual learning lifecycle."""

    def on_batch_start(self, batch_idx: int) -> None:
        """Called at start of batch processing."""
        pass

    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any]) -> None:
        """Called at end of batch processing."""
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """Called at start of epoch."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called at end of epoch."""
        pass

    def on_drift_detected(self, drift_score: float) -> None:
        """Called when distribution drift is detected."""
        pass

    def on_model_update(self, update_info: Dict[str, Any]) -> None:
        """Called after model update."""
        pass


class ContinuousLearningPipeline:
    """Pipeline for continuous/online learning."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        buffer_size: int = 10000,
        drift_threshold: float = 0.1,
        enable_drift_detection: bool = True,
        callbacks: Optional[List[ContinualLearningCallback]] = None,
    ):
        """Initialize continuous learning pipeline."""
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.drift_threshold = drift_threshold
        self.enable_drift_detection = enable_drift_detection
        self.callbacks = callbacks or []

        # Initialize components
        self.replay_buffer = ExperienceReplayBuffer(max_size=buffer_size)
        self.drift_detector = DriftDetector(threshold=drift_threshold)
        self.sample_weighter = SampleWeighter(weighting_strategy="inverse_loss")

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Statistics
        self.total_samples_seen = 0
        self.total_updates = 0
        self.batch_count = 0

        logger.info("Continuous learning pipeline initialized")

    def process_batch(
        self,
        batch_data: torch.Tensor,
        batch_labels: torch.Tensor,
        loss_fn: Callable,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """Process a batch of data."""
        for callback in self.callbacks:
            callback.on_batch_start(self.batch_count)

        self.model.train()

        # Forward pass
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        outputs = self.model(batch_data)
        loss = loss_fn(outputs, batch_labels)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Add to replay buffer
        experience = (batch_data.detach(), batch_labels.detach(), outputs.detach())
        priority = float(loss.item())
        self.replay_buffer.add(experience, priority=priority)

        # Detect drift
        if self.enable_drift_detection:
            features = batch_data.cpu().detach().numpy().reshape(batch_data.shape[0], -1)
            drift_detected = self.drift_detector.update(features)
            if drift_detected:
                for callback in self.callbacks:
                    callback.on_drift_detected(self.drift_detector.get_drift_score())

        # Update statistics
        self.total_samples_seen += len(batch_data)
        self.total_updates += 1
        self.batch_count += 1

        metrics = {
            "batch_loss": float(loss.item()),
            "batch_size": len(batch_data),
            "total_samples_seen": self.total_samples_seen,
        }

        for callback in self.callbacks:
            callback.on_batch_end(self.batch_count, metrics)

        logger.debug(f"Batch {self.batch_count} processed: loss={loss.item():.4f}")

        return metrics

    def experience_replay(
        self,
        loss_fn: Callable,
        num_replay_batches: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """Perform experience replay training."""
        if self.replay_buffer.get_size() == 0:
            return {}

        self.model.train()
        replay_losses = []

        for _ in range(num_replay_batches):
            # Sample from replay buffer
            batch = self.replay_buffer.sample_batch(self.batch_size)

            if not batch:
                continue

            data_batch = torch.stack([exp[0] for exp in batch])
            label_batch = torch.stack([exp[1] for exp in batch])

            data_batch = data_batch.to(device)
            label_batch = label_batch.to(device)

            # Forward pass
            outputs = self.model(data_batch)
            loss = loss_fn(outputs, label_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            replay_losses.append(float(loss.item()))

        return {
            "avg_replay_loss": float(np.mean(replay_losses)) if replay_losses else 0.0,
            "num_replay_batches": num_replay_batches,
        }

    def update_with_weighted_samples(
        self,
        batch_data: torch.Tensor,
        batch_labels: torch.Tensor,
        losses: np.ndarray,
        loss_fn: Callable,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """Update model with weighted samples."""
        # Compute sample weights
        weights = self.sample_weighter.compute_weights(losses)
        self.sample_weighter.update_loss_history(losses)

        self.model.train()

        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        weights_tensor = torch.from_numpy(weights).to(device).float()

        outputs = self.model(batch_data)
        loss = loss_fn(outputs, batch_labels, weight=weights_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "weighted_loss": float(loss.item()),
            "avg_sample_weight": float(np.mean(weights)),
            "avg_loss": self.sample_weighter.get_average_loss(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_samples_seen": self.total_samples_seen,
            "total_updates": self.total_updates,
            "buffer_size": self.replay_buffer.get_size(),
            "drift_score": self.drift_detector.get_drift_score(),
            "drift_detected": self.drift_detector.drift_detected,
            "avg_loss": self.sample_weighter.get_average_loss(),
        }

    def reset(self) -> None:
        """Reset pipeline state."""
        self.replay_buffer.clear()
        self.drift_detector.drift_detected = False
        self.batch_count = 0
        self.total_updates = 0

        logger.info("Continuous learning pipeline reset")
