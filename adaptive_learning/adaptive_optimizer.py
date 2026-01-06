"""
Adaptive Optimizer Module for Negative Space Imaging

Implements dynamic learning rate and hyperparameter adjustment with:
- Dynamic learning rate scheduling
- Hyperparameter adjustment based on performance
- Adaptive batch size tuning
- Momentum and acceleration control
- Performance-based adaptation

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdaptiveScheduler:
    """Adaptive learning rate scheduler."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_lr: float = 1e-3,
        min_lr: float = 1e-6,
        max_lr: float = 1e-1,
        patience: int = 5,
    ):
        """Initialize adaptive scheduler."""
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience

        self.current_lr = initial_lr
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.step_count = 0

    def step(self, loss: float) -> None:
        """Step scheduler based on loss."""
        self.step_count += 1

        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Reduce learning rate if no improvement
        if self.patience_counter >= self.patience:
            self.current_lr = max(self.current_lr * 0.5, self.min_lr)
            self._update_lr()
            self.patience_counter = 0
            logger.info(f"Learning rate reduced to {self.current_lr:.6f}")

        # Anneal learning rate
        self._anneal_lr()

    def _anneal_lr(self) -> None:
        """Anneal learning rate."""
        # Cosine annealing
        annealed_lr = self.min_lr + (self.current_lr - self.min_lr) * (1 + np.cos(np.pi * self.step_count / 1000)) / 2
        self._update_lr(annealed_lr)

    def _update_lr(self, new_lr: Optional[float] = None) -> None:
        """Update optimizer learning rate."""
        lr = new_lr or self.current_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class HyperparameterAdapter:
    """Adapt hyperparameters based on performance."""

    def __init__(self, optimizer: optim.Optimizer):
        """Initialize hyperparameter adapter."""
        self.optimizer = optimizer
        self.metrics_history = []
        self.lr_history = []
        self.momentum_history = []

    def adapt_learning_rate(
        self,
        current_metric: float,
        target_metric: float = 0.9,
    ) -> None:
        """Adapt learning rate based on performance."""
        current_lr = self.optimizer.param_groups[0]["lr"]

        if current_metric < target_metric:
            # Performance below target, increase learning rate
            new_lr = min(current_lr * 1.1, 1e-1)
            logger.debug(f"Increasing learning rate: {current_lr:.6f} -> {new_lr:.6f}")
        else:
            # Performance above target, decrease learning rate for stability
            new_lr = max(current_lr * 0.95, 1e-6)
            logger.debug(f"Decreasing learning rate: {current_lr:.6f} -> {new_lr:.6f}")

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        self.lr_history.append(new_lr)

    def adapt_momentum(self, acceleration_factor: float = 0.99) -> None:
        """Adapt momentum for optimizer."""
        if "momentum" not in self.optimizer.param_groups[0]:
            return

        current_momentum = self.optimizer.param_groups[0]["momentum"]
        new_momentum = min(current_momentum * acceleration_factor, 0.999)

        for param_group in self.optimizer.param_groups:
            param_group["momentum"] = new_momentum

        self.momentum_history.append(new_momentum)
        logger.debug(f"Momentum adjusted to {new_momentum:.4f}")

    def adapt_weight_decay(self, metric_improvement: float, threshold: float = 0.001):
        """Adapt weight decay based on improvement rate."""
        if metric_improvement < threshold:
            # Small improvement, increase regularization
            current_decay = self.optimizer.param_groups[0].get("weight_decay", 0)
            new_decay = min(current_decay * 1.1, 1e-3)
            logger.debug(f"Increasing weight decay: {current_decay:.6f} -> {new_decay:.6f}")
        else:
            # Good improvement, decrease regularization
            current_decay = self.optimizer.param_groups[0].get("weight_decay", 0)
            new_decay = max(current_decay * 0.9, 1e-6)
            logger.debug(f"Decreasing weight decay: {current_decay:.6f} -> {new_decay:.6f}")

        for param_group in self.optimizer.param_groups:
            param_group["weight_decay"] = new_decay

    def get_status(self) -> Dict[str, Any]:
        """Get current optimizer status."""
        param_group = self.optimizer.param_groups[0]
        return {
            "learning_rate": param_group.get("lr", "N/A"),
            "momentum": param_group.get("momentum", "N/A"),
            "weight_decay": param_group.get("weight_decay", "N/A"),
            "history_length": len(self.metrics_history),
        }


class AdaptiveBatchSizer:
    """Adaptively adjust batch size based on performance."""

    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 256,
    ):
        """Initialize adaptive batch sizer."""
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.batch_size_history = [initial_batch_size]

    def adapt(self, gradient_norm: float, target_gradient_norm: float = 1.0) -> int:
        """Adapt batch size based on gradient norm."""
        if gradient_norm > target_gradient_norm * 1.5:
            # Gradients too large, increase batch size to reduce variance
            new_batch_size = min(int(self.current_batch_size * 1.5), self.max_batch_size)
            logger.debug(f"Increasing batch size: {self.current_batch_size} -> {new_batch_size}")
        elif gradient_norm < target_gradient_norm * 0.5:
            # Gradients too small, decrease batch size for faster updates
            new_batch_size = max(int(self.current_batch_size * 0.75), self.min_batch_size)
            logger.debug(f"Decreasing batch size: {self.current_batch_size} -> {new_batch_size}")
        else:
            # Gradient norm is good
            new_batch_size = self.current_batch_size

        self.current_batch_size = new_batch_size
        self.batch_size_history.append(new_batch_size)

        return new_batch_size

    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch_size

    def get_statistics(self) -> Dict[str, Any]:
        """Get batch size statistics."""
        return {
            "current_batch_size": self.current_batch_size,
            "min_batch_size": min(self.batch_size_history),
            "max_batch_size": max(self.batch_size_history),
            "avg_batch_size": float(np.mean(self.batch_size_history)),
        }


class AdaptiveOptimizer:
    """Complete adaptive optimizer combining multiple strategies."""

    def __init__(
        self,
        model: torch.nn.Module,
        initial_lr: float = 1e-3,
        initial_batch_size: int = 32,
    ):
        """Initialize adaptive optimizer."""
        self.model = model
        self.base_optimizer = optim.AdamW(model.parameters(), lr=initial_lr)

        # Components
        self.scheduler = AdaptiveScheduler(self.base_optimizer, initial_lr=initial_lr)
        self.adapter = HyperparameterAdapter(self.base_optimizer)
        self.batch_sizer = AdaptiveBatchSizer(initial_batch_size=initial_batch_size)

        # State
        self.step_count = 0
        self.metrics_history = []

    def step(
        self,
        loss: float,
        gradient_norm: Optional[float] = None,
        current_metric: Optional[float] = None,
    ) -> None:
        """Perform optimization step with adaptation."""
        # Step scheduler
        self.scheduler.step(loss)

        # Adapt learning rate based on performance
        if current_metric is not None:
            self.adapter.adapt_learning_rate(current_metric)

        # Adapt batch size if gradient norm available
        if gradient_norm is not None:
            self.batch_sizer.adapt(gradient_norm)

        self.step_count += 1
        self.metrics_history.append({
            "step": self.step_count,
            "loss": loss,
            "gradient_norm": gradient_norm,
            "metric": current_metric,
        })

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.scheduler.get_lr()

    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self.batch_sizer.get_batch_size()

    def get_status(self) -> Dict[str, Any]:
        """Get complete optimizer status."""
        return {
            "step_count": self.step_count,
            "learning_rate": self.get_learning_rate(),
            "batch_size": self.get_batch_size(),
            "scheduler_status": {
                "best_loss": self.scheduler.best_loss,
                "patience": self.scheduler.patience_counter,
            },
            "adapter_status": self.adapter.get_status(),
            "batch_sizer_status": self.batch_sizer.get_statistics(),
        }

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.base_optimizer.zero_grad()

    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass."""
        loss.backward()

    def optimize(self, gradient_norm: Optional[float] = None) -> None:
        """Step optimizer."""
        if gradient_norm is not None and gradient_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_norm)

        self.base_optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state."""
        self.base_optimizer.load_state_dict(state_dict)


class AdaptiveCallback:
    """Callback for adaptive optimization during training."""

    def __init__(self, adaptive_optimizer: AdaptiveOptimizer):
        """Initialize callback."""
        self.optimizer = adaptive_optimizer

    def on_batch_end(
        self,
        loss: float,
        gradient_norm: Optional[float] = None,
        metric: Optional[float] = None,
    ) -> None:
        """Called at end of training batch."""
        self.optimizer.step(loss, gradient_norm, metric)

    def on_epoch_end(self) -> Dict[str, Any]:
        """Called at end of training epoch."""
        return self.optimizer.get_status()

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.get_learning_rate()

    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self.optimizer.get_batch_size()
