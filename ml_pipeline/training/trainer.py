"""
Advanced ML Trainer Module for Negative Space Imaging

Comprehensive training engine with:
- DataLoader integration (PyTorch)
- Model initialization from config
- Training & validation loops
- Checkpoint management
- Distributed training support
- Optuna hyperparameter optimization
- Mixed precision training
- Early stopping & learning rate scheduling
- W&B integration for experiment tracking

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.parallel as nn_parallel
import wandb
from optuna.trial import Trial
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for training process."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision: bool = True,
        distributed: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "./checkpoints",
        enable_wandb: bool = True,
        wandb_project: str = "negative-space-imaging",
        wandb_entity: Optional[str] = None,
        enable_profiling: bool = False,
        profile_wait_steps: int = 1,
        profile_warmup_steps: int = 1,
        profile_active_steps: int = 3,
    ):
        """Initialize training configuration."""
        self.model_config = model_config
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.device = device
        self.mixed_precision = mixed_precision
        self.distributed = distributed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.enable_wandb = enable_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.enable_profiling = enable_profiling
        self.profile_wait_steps = profile_wait_steps
        self.profile_warmup_steps = profile_warmup_steps
        self.profile_active_steps = profile_active_steps


class CheckpointManager:
    """Manages model checkpoint saving and loading."""

    def __init__(self, checkpoint_dir: Path, keep_last_n: int = 5):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoint_history: List[Path] = []

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> Path:
        """Save model checkpoint."""
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict()
            if isinstance(model, (nn_parallel.DataParallel, nn_parallel.DistributedDataParallel))
            else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        if lr_scheduler is not None:
            checkpoint_data["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint_data, latest_path)
        self.checkpoint_history.append(latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint_data, best_path)
            logger.info(f"Saved best model checkpoint: {best_path}")

        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint_data, epoch_path)

        # Remove old checkpoints
        if len(self.checkpoint_history) > self.keep_last_n:
            old_checkpoint = self.checkpoint_history.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        return latest_path

    def load_checkpoint(self, model: nn.Module, optimizer: Optimizer, checkpoint_path: Path):
        """Load model checkpoint."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle DataParallel/DistributedDataParallel models
        if isinstance(model, (nn_parallel.DataParallel, nn_parallel.DistributedDataParallel)):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        return checkpoint


class EarlyStoppingCallback:
    """Early stopping callback for training."""

    def __init__(
        self,
        metric_name: str = "val_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
    ):
        """Initialize early stopping."""
        self.metric_name = metric_name
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_metric = None
        self.wait_count = 0
        self.stopped_epoch = None

    def __call__(self, current_metrics: Dict[str, float]) -> bool:
        """Check if training should stop."""
        current_metric = current_metrics.get(self.metric_name)

        if current_metric is None:
            return False

        if self.best_metric is None:
            self.best_metric = current_metric
            return False

        if current_metric < self.best_metric - self.min_delta:
            self.best_metric = current_metric
            self.wait_count = 0
            return False

        self.wait_count += 1
        if self.wait_count >= self.patience:
            self.stopped_epoch = self.wait_count
            return True

        return False


class Trainer:
    """Advanced trainer for Negative Space Imaging models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        loss_fn: Callable,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        """Initialize trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.loss_fn = loss_fn
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Setup distributed training
        if config.distributed:
            self.model = nn_parallel.DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True
            )
        elif torch.cuda.device_count() > 1:
            self.model = nn_parallel.DataParallel(self.model)

        # Setup optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup learning rate scheduler
        self.lr_scheduler = lr_scheduler

        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None

        # Checkpointing
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)

        # Early stopping
        self.early_stopping = EarlyStoppingCallback(patience=15)

        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        self.best_val_loss = float("inf")

        # W&B initialization
        if config.enable_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=vars(config),
                mode="online",
            )
            wandb.watch(self.model, log_freq=100)

        logger.info(f"Trainer initialized on device: {self.device}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [TRAIN]")
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with autocast():
                    outputs = self._forward_pass(batch)
                    loss = self.loss_fn(outputs, batch[1])
                    loss = loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                outputs = self._forward_pass(batch)
                loss = self.loss_fn(outputs, batch[1])
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(batch[1].cpu().numpy())

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        metrics = {"train_loss": avg_loss}

        # Compute additional metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        if all_targets.ndim > 1 and all_targets.shape[1] > 1:
            # Multi-class classification
            predictions = np.argmax(all_preds, axis=1)
            targets = np.argmax(all_targets, axis=1)
            accuracy = np.mean(predictions == targets)
            metrics["train_accuracy"] = accuracy

        self.train_metrics = metrics
        logger.info(f"Training metrics: {metrics}")

        return metrics

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="[VALIDATION]")
            for batch in pbar:
                outputs = self._forward_pass(batch)
                loss = self.loss_fn(outputs, batch[1])
                total_loss += loss.item()

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch[1].cpu().numpy())

                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.val_loader)
        metrics = {"val_loss": avg_loss}

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        if all_targets.ndim > 1 and all_targets.shape[1] > 1:
            predictions = np.argmax(all_preds, axis=1)
            targets = np.argmax(all_targets, axis=1)
            accuracy = np.mean(predictions == targets)
            metrics["val_accuracy"] = accuracy

        self.val_metrics = metrics
        logger.info(f"Validation metrics: {metrics}")

        return metrics

    def fit(
        self,
        num_epochs: Optional[int] = None,
        optuna_trial: Optional[Trial] = None,
        resume_from_checkpoint: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Train model for specified epochs."""
        num_epochs = num_epochs or self.config.num_epochs
        start_epoch = 0

        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            checkpoint = self.checkpoint_manager.load_checkpoint(
                self.model, self.optimizer, resume_from_checkpoint
            )
            start_epoch = checkpoint["epoch"] + 1

        for epoch in range(start_epoch, num_epochs):
            # Training
            train_metrics = self.train_epoch(epoch)

            # Learning rate scheduling
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Validation
            val_metrics = self.validate_epoch()

            # Combined metrics
            combined_metrics = {**train_metrics, **val_metrics}

            # W&B logging
            if self.config.enable_wandb:
                wandb.log(combined_metrics, step=epoch)

            # Checkpoint saving
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]

            self.checkpoint_manager.save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                combined_metrics,
                is_best=is_best,
                lr_scheduler=self.lr_scheduler,
            )

            # Optuna trial reporting
            if optuna_trial:
                optuna_trial.report(val_metrics["val_loss"], step=epoch)
                if optuna_trial.should_prune():
                    raise optuna.TrialPruned()

            # Early stopping
            if self.early_stopping(combined_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        if self.config.enable_wandb:
            wandb.finish()

        return {
            "best_val_loss": self.best_val_loss,
            "final_train_metrics": self.train_metrics,
            "final_val_metrics": self.val_metrics,
        }

    def _forward_pass(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Execute forward pass."""
        inputs, _ = batch
        inputs = inputs.to(self.device)
        return self.model(inputs)


class OptunaOptimizer:
    """Optuna-based hyperparameter optimization."""

    def __init__(
        self,
        model_factory: Callable,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        loss_fn: Callable,
        n_trials: int = 20,
    ):
        """Initialize Optuna optimizer."""
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.loss_fn = loss_fn
        self.n_trials = n_trials
        self.best_trial = None

    def objective(self, trial: Trial) -> float:
        """Objective function for optimization."""
        # Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        warmup_epochs = trial.suggest_int("warmup_epochs", 1, 10)

        # Create config with suggested parameters
        trial_config = TrainingConfig(
            model_config=self.config.model_config,
            batch_size=self.config.batch_size,
            num_epochs=self.config.num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            device=self.config.device,
            mixed_precision=self.config.mixed_precision,
            enable_wandb=False,  # Disable W&B for trials
        )

        # Create model
        model = self.model_factory()

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=trial_config,
            loss_fn=self.loss_fn,
            optimizer=optimizer,
        )

        # Train model
        result = trainer.fit(optuna_trial=trial)

        return result["best_val_loss"]

    def optimize(self) -> Dict[str, Any]:
        """Run optimization."""
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_trial = study.best_trial

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value}")
        logger.info(f"Best params: {study.best_params}")

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
        }


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    loss_fn: Callable,
    optimizer: Optional[Optimizer] = None,
    lr_scheduler: Optional[LRScheduler] = None,
) -> Trainer:
    """Factory function to create trainer."""
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
