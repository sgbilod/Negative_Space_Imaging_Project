"""
Vision Transformer Fine-tuning Pipeline

Advanced fine-tuning pipeline with:
- Layer-wise learning rate configuration
- Gradual unfreezing strategy
- Custom training loop with validation
- Checkpoint management based on metrics
- Early stopping with patience
- W&B experiment tracking

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LayerWiseLRScheduler:
    """
    Layer-wise learning rate scheduler for Vision Transformer.

    Applies different learning rates to different layers based on depth.
    Earlier layers use smaller learning rates, later layers use larger rates.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 1e-4,
        max_lr: float = 1e-3,
        num_layers: int = 12,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
    ) -> None:
        """
        Initialize layer-wise LR scheduler.

        Args:
            optimizer: PyTorch optimizer
            base_lr: Base learning rate for early layers
            max_lr: Max learning rate for final layers
            num_layers: Total number of layers in model
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.num_layers = num_layers
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

        # Compute layer-wise learning rates (linear increase from base to max)
        self.layer_lrs = np.linspace(base_lr, max_lr, num_layers)

    def step(self) -> None:
        """Update learning rates for this epoch."""
        epoch = self.current_epoch

        # Warmup phase: linear increase from 0 to target LR
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            lrs = self.layer_lrs * warmup_factor
        else:
            # Cosine annealing after warmup
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            lrs = self.layer_lrs * cosine_factor

        # Set learning rates for each parameter group
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = float(lr)

        self.current_epoch += 1


class GradualUnfreezing:
    """
    Gradually unfreeze model layers during training.
    """

    def __init__(
        self,
        model: nn.Module,
        num_layers: int = 12,
        unfreeze_strategy: str = "linear",
    ) -> None:
        """
        Initialize gradual unfreezing.

        Args:
            model: Model to unfreeze
            num_layers: Number of layers to unfreeze
            unfreeze_strategy: Strategy for unfreezing ('linear' or 'exponential')
        """
        self.model = model
        self.num_layers = num_layers
        self.unfreeze_strategy = unfreeze_strategy
        self.current_unfrozen_layers = 0

    def unfreeze_layer(self, layer_idx: int) -> None:
        """
        Unfreeze specific layer.

        Args:
            layer_idx: Index of layer to unfreeze
        """
        if hasattr(self.model, "blocks") and layer_idx < len(self.model.blocks):
            for param in self.model.blocks[layer_idx].parameters():
                param.requires_grad = True
            logger.info(f"Unfroze layer {layer_idx}")

    def unfreeze_until(self, layer_idx: int) -> None:
        """
        Unfreeze all layers up to specified index.

        Args:
            layer_idx: Index to unfreeze until
        """
        for i in range(layer_idx + 1):
            self.unfreeze_layer(i)
        self.current_unfrozen_layers = layer_idx + 1

    def step(self, epoch: int, total_epochs: int) -> None:
        """
        Determine unfreezing based on epoch.

        Args:
            epoch: Current epoch
            total_epochs: Total training epochs
        """
        if self.unfreeze_strategy == "linear":
            # Unfreeze layers linearly over training
            layer_to_unfreeze = int(
                (epoch / total_epochs) * self.num_layers
            )
        elif self.unfreeze_strategy == "exponential":
            # Unfreeze layers exponentially (slow start, fast end)
            progress = epoch / total_epochs
            layer_to_unfreeze = int(
                self.num_layers * (progress ** 0.5)
            )
        else:
            layer_to_unfreeze = self.current_unfrozen_layers

        # Unfreeze new layers
        if layer_to_unfreeze > self.current_unfrozen_layers:
            self.unfreeze_until(layer_to_unfreeze - 1)


class ViTFineTuner:
    """
    Complete fine-tuning pipeline for Vision Transformer.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int = 2,
        base_lr: float = 1e-4,
        max_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        num_epochs: int = 100,
        device: str = "cuda",
        enable_wandb: bool = True,
        checkpoint_dir: str = "./checkpoints/vit",
        enable_gradual_unfreezing: bool = True,
        unfreeze_strategy: str = "linear",
    ) -> None:
        """
        Initialize ViT fine-tuner.

        Args:
            model: Vision Transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of output classes
            base_lr: Base learning rate for early layers
            max_lr: Max learning rate for final layers
            weight_decay: Weight decay for regularization
            warmup_epochs: Number of warmup epochs
            num_epochs: Total training epochs
            device: Device to use ('cuda' or 'cpu')
            enable_wandb: Whether to use Weights & Biases
            checkpoint_dir: Directory for saving checkpoints
            enable_gradual_unfreezing: Whether to gradually unfreeze layers
            unfreeze_strategy: Strategy for unfreezing
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Setup layer-wise parameter groups
        self.optimizer = self._setup_optimizer(base_lr, max_lr, weight_decay)

        # Learning rate scheduler
        self.lr_scheduler = LayerWiseLRScheduler(
            self.optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            num_layers=len(self.model.blocks) if hasattr(self.model, "blocks") else 12,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs,
        )

        # Gradual unfreezing
        self.gradual_unfreezing = None
        if enable_gradual_unfreezing and hasattr(self.model, "blocks"):
            self.gradual_unfreezing = GradualUnfreezing(
                self.model,
                num_layers=len(self.model.blocks),
                unfreeze_strategy=unfreeze_strategy,
            )

        # Metrics tracking
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        self.patience = 15
        self.patience_counter = 0
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []

        # W&B integration
        self.enable_wandb = enable_wandb
        if enable_wandb:
            wandb.init(
                project="negative-space-vit",
                config={
                    "base_lr": base_lr,
                    "max_lr": max_lr,
                    "weight_decay": weight_decay,
                    "warmup_epochs": warmup_epochs,
                    "num_epochs": num_epochs,
                    "enable_gradual_unfreezing": enable_gradual_unfreezing,
                    "unfreeze_strategy": unfreeze_strategy,
                },
            )

    def _setup_optimizer(
        self,
        base_lr: float,
        max_lr: float,
        weight_decay: float,
    ) -> Optimizer:
        """
        Setup layer-wise optimizer with different learning rates.

        Args:
            base_lr: Base learning rate
            max_lr: Max learning rate
            weight_decay: Weight decay coefficient

        Returns:
            Configured optimizer
        """
        param_groups: List[Dict[str, Any]] = []

        # Layer-wise parameter groups
        if hasattr(self.model, "blocks"):
            num_layers = len(self.model.blocks)
            lrs = np.linspace(base_lr, max_lr, num_layers)

            # Patch embedding and position embedding
            param_groups.append({
                "params": list(self.model.patch_embed.parameters()) +
                         [self.model.pos_embed, self.model.cls_token],
                "lr": float(lrs[0]),
                "weight_decay": weight_decay,
            })

            # Transformer blocks with layer-wise LR
            for i, block in enumerate(self.model.blocks):
                param_groups.append({
                    "params": list(block.parameters()),
                    "lr": float(lrs[i]),
                    "weight_decay": weight_decay,
                })

            # Classification head
            param_groups.append({
                "params": list(self.model.norm.parameters()) +
                         list(self.model.head.parameters()),
                "lr": float(max_lr),
                "weight_decay": weight_decay,
            })
        else:
            # Fallback: single learning rate
            param_groups.append({
                "params": self.model.parameters(),
                "lr": max_lr,
                "weight_decay": weight_decay,
            })

        optimizer = torch.optim.AdamW(param_groups)
        return optimizer

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Gradual unfreezing
        if self.gradual_unfreezing:
            self.gradual_unfreezing.step(epoch, self.num_epochs)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs} [TRAIN]",
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix({
                "loss": loss.item(),
                "acc": correct_predictions / total_samples,
            })

        # Learning rate scheduling
        self.lr_scheduler.step()

        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = correct_predictions / total_samples

        metrics = {
            "train_loss": avg_loss,
            "train_accuracy": avg_accuracy,
        }

        self.train_history.append(metrics)
        logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

        return metrics

    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc="[VALIDATION]",
            )

            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

                pbar.set_postfix({
                    "loss": loss.item(),
                    "acc": correct_predictions / total_samples,
                })

        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = correct_predictions / total_samples

        metrics = {
            "val_loss": avg_loss,
            "val_accuracy": avg_accuracy,
        }

        self.val_history.append(metrics)
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

        return metrics

    def fit(self) -> Dict[str, Any]:
        """
        Train model for specified epochs.

        Returns:
            Training results dictionary
        """
        logger.info(f"Starting fine-tuning for {self.num_epochs} epochs")
        start_time = time.time()

        for epoch in range(self.num_epochs):
            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            val_metrics = self.validate_epoch()

            # Combined metrics
            combined_metrics = {**train_metrics, **val_metrics}

            # W&B logging
            if self.enable_wandb:
                wandb.log(combined_metrics, step=epoch)

            # Checkpoint management
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_val_accuracy = val_metrics["val_accuracy"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if epoch % 10 == 0:
                    self._save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        elapsed_time = time.time() - start_time

        if self.enable_wandb:
            wandb.finish()

        return {
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "elapsed_time": elapsed_time,
        }

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": self.best_val_loss,
            "val_accuracy": self.best_val_accuracy,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def get_learning_rates(self) -> List[float]:
        """
        Get current learning rates for each parameter group.

        Returns:
            List of learning rates
        """
        return [param_group["lr"] for param_group in self.optimizer.param_groups]

    def export_to_onnx(
        self,
        output_path: str = "./model.onnx",
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    ) -> None:
        """
        Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            input_size: Input tensor size (batch, channels, height, width)
        """
        self.model.eval()
        dummy_input = torch.randn(input_size).to(self.device)

        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                opset_version=12,
                input_names=["image"],
                output_names=["logits"],
                dynamic_axes={"image": {0: "batch_size"}},
            )
            logger.info(f"Exported model to ONNX: {output_path}")
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            raise
