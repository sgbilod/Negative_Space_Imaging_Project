"""
SSM Training Pipeline

Optimized training for state space models:
- Variable-length sequence handling
- Gradient accumulation for memory efficiency
- Mixed precision training (FP16/FP32)
- Learning rate scheduling and warmup
- Validation and early stopping
- Comprehensive metrics tracking
"""

import logging
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

logger = logging.getLogger(__name__)


class SSMTrainer:
    """
    Trainer for SSM models with full training pipeline.

    Features:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Validation metrics
    - Checkpoint management
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = True,
        task_type: str = "classification",
    ):
        """
        Initialize trainer.

        Args:
            model: SSM or Transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: Weight decay (L2 regularization)
            gradient_accumulation_steps: Steps for gradient accumulation
            mixed_precision: Use mixed precision training
            task_type: Task type for loss computation
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.task_type = task_type
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None

        # Loss function
        if task_type == "classification":
            self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        elif task_type == "regression":
            self.loss_fn = nn.MSELoss(reduction="mean")
        elif task_type == "anomaly_detection":
            self.loss_fn = nn.BCELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Metrics tracking
        self.train_history = {
            "loss": [],
            "accuracy": [] if task_type == "classification" else [],
        }
        self.val_history = {
            "loss": [],
            "accuracy": [] if task_type == "classification" else [],
        }

        logger.info(
            f"✓ Initialized trainer for {task_type} task on {device}"
        )

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Metrics dict with loss and accuracy
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            x, y = batch
            batch_size = x.size(0)

            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(x)
                    logits = outputs["logits"]
                    loss = self.loss_fn(logits, y)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(x)
                logits = outputs["logits"]
                loss = self.loss_fn(logits, y)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.gradient_accumulation_steps

            # Compute accuracy for classification
            if self.task_type == "classification":
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == y).sum().item()
                total_correct += correct
                total_samples += batch_size

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()

            num_batches += 1

        avg_loss = total_loss / num_batches
        metrics = {"loss": avg_loss}

        if self.task_type == "classification":
            accuracy = total_correct / total_samples
            metrics["accuracy"] = accuracy

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Metrics dict with loss and accuracy
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch
                batch_size = x.size(0)

                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(x)
                        logits = outputs["logits"]
                        loss = self.loss_fn(logits, y)
                else:
                    outputs = self.model(x)
                    logits = outputs["logits"]
                    loss = self.loss_fn(logits, y)

                total_loss += loss.item()

                # Compute accuracy
                if self.task_type == "classification":
                    predictions = torch.argmax(logits, dim=-1)
                    correct = (predictions == y).sum().item()
                    total_correct += correct
                    total_samples += batch_size

                num_batches += 1

        avg_loss = total_loss / num_batches
        metrics = {"loss": avg_loss}

        if self.task_type == "classification":
            accuracy = total_correct / total_samples
            metrics["accuracy"] = accuracy

        return metrics

    def fit(
        self,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        warmup_epochs: int = 5,
        scheduler_type: str = "cosine",
    ) -> Dict[str, Any]:
        """
        Train model for specified epochs with early stopping.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            warmup_epochs: Number of warmup epochs
            scheduler_type: Learning rate scheduler type

        Returns:
            Training history dict
        """
        # Learning rate scheduler
        total_steps = num_epochs * len(self.train_loader)
        warmup_steps = warmup_epochs * len(self.train_loader)

        if scheduler_type == "cosine":
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
            )
        elif scheduler_type == "onecycle":
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=0.1,
                total_steps=total_steps,
                pct_start=warmup_epochs / num_epochs,
            )
        else:
            scheduler = None

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if scheduler is not None:
                scheduler.step()

            # Record history
            self.train_history["loss"].append(train_metrics["loss"])
            self.val_history["loss"].append(val_metrics["loss"])

            if self.task_type == "classification":
                self.train_history["accuracy"].append(train_metrics["accuracy"])
                self.val_history["accuracy"].append(val_metrics["accuracy"])

            # Early stopping check
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1

            # Log progress
            log_msg = f"Epoch {epoch+1}/{num_epochs} - "
            log_msg += f"Train Loss: {train_metrics['loss']:.4f}, "
            log_msg += f"Val Loss: {val_metrics['loss']:.4f}"

            if self.task_type == "classification":
                log_msg += f", Train Acc: {train_metrics['accuracy']:.4f}, "
                log_msg += f"Val Acc: {val_metrics['accuracy']:.4f}"

            logger.info(log_msg)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }

    def test(self) -> Dict[str, float]:
        """
        Evaluate on test set.

        Returns:
            Test metrics
        """
        if self.test_loader is None:
            logger.warning("No test loader provided")
            return {}

        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.test_loader:
                x, y = batch
                batch_size = x.size(0)

                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)
                logits = outputs["logits"]
                loss = self.loss_fn(logits, y)

                total_loss += loss.item()

                if self.task_type == "classification":
                    predictions = torch.argmax(logits, dim=-1)
                    correct = (predictions == y).sum().item()
                    total_correct += correct
                    total_samples += batch_size

                num_batches += 1

        avg_loss = total_loss / num_batches
        metrics = {"loss": avg_loss}

        if self.task_type == "classification":
            accuracy = total_correct / total_samples
            metrics["accuracy"] = accuracy

        logger.info(f"Test metrics: {metrics}")
        return metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": {
                "train": self.train_history,
                "val": self.val_history,
            }
        }, path)
        logger.info(f"✓ Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_history = checkpoint["history"]["train"]
        self.val_history = checkpoint["history"]["val"]
        logger.info(f"✓ Loaded checkpoint from {path}")


# Export public API
__all__ = [
    "SSMTrainer",
]
