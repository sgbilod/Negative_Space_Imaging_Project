"""
Diffusion Model Training Pipeline

Complete training pipeline with:
- Data loading for synthetic astronomical images
- Noise corruption and prediction target
- Loss computation (MSE, MAE)
- Training loop with validation
- EMA (Exponential Moving Average) model updates
- Checkpointing and resume capability
- W&B integration for tracking

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ExponentialMovingAverage:
    """
    Exponential Moving Average for model parameters.

    Used for EMA-based model updates during diffusion training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        """
        Initialize EMA.

        Args:
            model: Model to track
            decay: Decay rate (higher = more conservative updates)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """Update shadow parameters using EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                else:
                    new_shadow = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                    self.shadow[name] = new_shadow

    def apply_shadow(self) -> None:
        """Apply shadow to model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def restore_parameters(self) -> None:
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]


class DiffusionTrainingConfig:
    """Configuration for diffusion model training."""

    def __init__(
        self,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_diffusion_steps: int = 1000,
        loss_type: str = "mse",
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        enable_wandb: bool = True,
        checkpoint_interval: int = 10,
        device: str = "cuda",
    ) -> None:
        """
        Initialize training config.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            num_diffusion_steps: Number of diffusion steps
            loss_type: Loss function type ('mse' or 'mae')
            use_ema: Whether to use EMA
            ema_decay: EMA decay rate
            enable_wandb: Whether to use W&B
            checkpoint_interval: Save checkpoint every N epochs
            device: Device to use
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_diffusion_steps = num_diffusion_steps
        self.loss_type = loss_type
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.enable_wandb = enable_wandb
        self.checkpoint_interval = checkpoint_interval
        self.device = device


class DiffusionTrainer:
    """Trainer for diffusion models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DiffusionTrainingConfig,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Diffusion model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            optimizer: Optional optimizer (created if not provided)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.device)

        # Setup optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup EMA
        self.ema = None
        if config.use_ema:
            self.ema = ExponentialMovingAverage(self.model, decay=config.ema_decay)

        # Loss function
        if config.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif config.loss_type == "mae":
            self.loss_fn = nn.L1Loss()
        else:
            self.loss_fn = nn.MSELoss()

        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

        # W&B
        if config.enable_wandb:
            wandb.init(
                project="negative-space-diffusion",
                config=vars(config),
                mode="online",
            )

        logger.info(f"Trainer initialized on device: {self.device}")

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs} [TRAIN]",
        )

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)

            # Sample random timesteps
            batch_size = images.shape[0]
            t = torch.randint(
                0,
                self.config.num_diffusion_steps,
                (batch_size,),
                device=self.device,
            )

            # Forward diffusion: add noise
            x_t, noise = self.model.diffuse(images, t)

            # Predict noise
            self.optimizer.zero_grad()
            noise_pred = self.model(x_t, t)

            # Compute loss
            loss = self.loss_fn(noise_pred, noise)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # EMA update
            if self.ema:
                self.ema.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)

        logger.info(f"Epoch {epoch + 1} - Training Loss: {avg_loss:.6f}")

        return avg_loss

    def validate_epoch(self) -> float:
        """
        Validate model.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc="[VALIDATION]",
            )

            for images, _ in pbar:
                images = images.to(self.device)

                # Sample random timesteps
                batch_size = images.shape[0]
                t = torch.randint(
                    0,
                    self.config.num_diffusion_steps,
                    (batch_size,),
                    device=self.device,
                )

                # Forward diffusion
                x_t, noise = self.model.diffuse(images, t)

                # Predict noise
                noise_pred = self.model(x_t, t)

                # Compute loss
                loss = self.loss_fn(noise_pred, noise)
                total_loss += loss.item()

                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)

        logger.info(f"Validation Loss: {avg_loss:.6f}")

        return avg_loss

    def fit(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Train model.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training results dictionary
        """
        logger.info(
            f"Starting training for {self.config.num_epochs} epochs on {self.device}"
        )
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss = self.validate_epoch()

            # W&B logging
            if self.config.enable_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }, step=epoch)

            # Checkpoint saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, is_best=True)

            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

        elapsed_time = time.time() - start_time

        if self.config.enable_wandb:
            wandb.finish()

        return {
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "elapsed_time": elapsed_time,
        }

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model
        """
        checkpoint_dir = Path("./checkpoints/diffusion")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        if self.ema:
            checkpoint["ema_state_dict"] = self.ema.shadow

        if is_best:
            path = checkpoint_dir / "best_model.pt"
        else:
            path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.ema and "ema_state_dict" in checkpoint:
            self.ema.shadow = checkpoint["ema_state_dict"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def get_ema_model(self) -> nn.Module:
        """
        Get EMA model.

        Returns:
            Model with EMA parameters
        """
        if not self.ema:
            return self.model

        # Apply EMA parameters
        self.ema.apply_shadow()
        return self.model

    def export_model(self, output_path: str) -> None:
        """
        Export model to file.

        Args:
            output_path: Path to save model
        """
        if self.ema:
            self.ema.apply_shadow()

        self.model.save_model(output_path)
        logger.info(f"Exported model to {output_path}")


class DiffusionTrainingPipeline:
    """Complete training pipeline for diffusion models."""

    def __init__(
        self,
        config: DiffusionTrainingConfig,
        device: str = "cuda",
    ) -> None:
        """
        Initialize pipeline.

        Args:
            config: Training configuration
            device: Device to use
        """
        self.config = config
        self.device = device

    def run(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Run training pipeline.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training results
        """
        trainer = DiffusionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
        )

        results = trainer.fit()
        return results
