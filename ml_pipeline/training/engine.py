"""
ML Pipeline Training - Model Training and Optimization

Handles model training, evaluation, and continuous learning for the pipeline.

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

from ..core.config import ModelConfig, PipelineConfig
from ..core.pipeline import DeviceManager
from ..models.registry import ModelRegistry, BaseModel

logger = logging.getLogger(__name__)


class TrainingEngine:
    """
    Model training and optimization engine.

    Features:
    - Automated model training with hyperparameter optimization
    - Continuous learning and model updates
    - Performance monitoring and early stopping
    - Multi-GPU training support
    """

    def __init__(
        self,
        config: PipelineConfig,
        model_registry: ModelRegistry,
        device_manager: DeviceManager
    ):
        self.config = config
        self.model_registry = model_registry
        self.device_manager = device_manager

        # Training settings
        self.training_dir = Path(config.model_dir) / "training"
        self.training_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized training engine")

    async def train_model(
        self,
        model_name: str,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train a model with the provided datasets.

        Args:
            model_name: Name of the model to train
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            training_config: Optional training configuration

        Returns:
            Training results and metrics
        """
        logger.info(f"Starting training for model: {model_name}")

        # Get model
        model = await self.model_registry.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not loaded")

        # Setup training configuration
        config = self._get_training_config(model_name, training_config)

        # Create data loaders
        train_loader = self._create_data_loader(train_dataset, config, shuffle=True)
        val_loader = self._create_data_loader(val_dataset, config, shuffle=False) if val_dataset else None

        # Setup training components
        optimizer = self._create_optimizer(model, config)
        scheduler = self._create_scheduler(optimizer, config)
        criterion = self._create_criterion(config)

        # Training loop
        best_model_state = None
        best_val_loss = float('inf')
        training_history = []

        for epoch in range(config["epochs"]):
            logger.info(f"Epoch {epoch + 1}/{config['epochs']}")

            # Train epoch
            train_metrics = await self._train_epoch(
                model, train_loader, optimizer, criterion, config
            )

            # Validate epoch
            val_metrics = None
            if val_loader:
                val_metrics = await self._validate_epoch(
                    model, val_loader, criterion, config
                )

            # Update learning rate
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["loss"] if val_metrics else train_metrics["loss"])
                else:
                    scheduler.step()

            # Record metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            }
            training_history.append(epoch_metrics)

            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_metrics['loss']:.4f}" +
                       (f", Val Loss: {val_metrics['loss']:.4f}" if val_metrics else ""))

            # Save best model
            val_loss = val_metrics["loss"] if val_metrics else train_metrics["loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.model.state_dict().copy()

            # Early stopping
            if self._should_early_stop(training_history, config):
                logger.info("Early stopping triggered")
                break

        # Save trained model
        if best_model_state:
            model.model.load_state_dict(best_model_state)
            await self._save_model_checkpoint(model, model_name, training_history)

        training_results = {
            "model_name": model_name,
            "epochs_completed": len(training_history),
            "best_val_loss": best_val_loss,
            "training_history": training_history,
            "final_train_metrics": training_history[-1]["train"],
            "final_val_metrics": training_history[-1]["val"],
        }

        logger.info(f"Training completed for {model_name}")
        return training_results

    def _get_training_config(self, model_name: str, custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get training configuration with defaults."""
        default_config = {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "adam",
            "scheduler": "cosine",
            "criterion": "cross_entropy",
            "early_stopping_patience": 10,
            "gradient_clip_norm": 1.0,
            "mixed_precision": True,
        }

        # Update with model-specific config
        model_config = self.config.get_model_config(model_name)
        if "training" in model_config.model_params:
            default_config.update(model_config.model_params["training"])

        # Update with custom config
        if custom_config:
            default_config.update(custom_config)

        return default_config

    def _create_data_loader(self, dataset: Dataset, config: Dict[str, Any], shuffle: bool) -> DataLoader:
        """Create data loader for training."""
        return DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
        )

    def _create_optimizer(self, model: BaseModel, config: Dict[str, Any]) -> optim.Optimizer:
        """Create optimizer for training."""
        optimizer_name = config["optimizer"].lower()

        if optimizer_name == "adam":
            return optim.Adam(
                model.model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"]
            )
        elif optimizer_name == "adamw":
            return optim.AdamW(
                model.model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"]
            )
        elif optimizer_name == "sgd":
            return optim.SGD(
                model.model.parameters(),
                lr=config["learning_rate"],
                momentum=0.9,
                weight_decay=config["weight_decay"]
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _create_scheduler(self, optimizer: optim.Optimizer, config: Dict[str, Any]) -> Optional[Any]:
        """Create learning rate scheduler."""
        scheduler_name = config["scheduler"].lower()

        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        else:
            return None

    def _create_criterion(self, config: Dict[str, Any]) -> nn.Module:
        """Create loss criterion."""
        criterion_name = config["criterion"].lower()

        if criterion_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif criterion_name == "mse":
            return nn.MSELoss()
        elif criterion_name == "bce":
            return nn.BCELoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")

    async def _train_epoch(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.model.train()
        total_loss = 0.0
        num_batches = 0

        scaler = torch.cuda.amp.GradScaler() if config["mixed_precision"] and torch.cuda.is_available() else None

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move to device
            if model.device:
                inputs = inputs.to(model.device)
                targets = targets.to(model.device)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model.model(inputs)
                loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), config["gradient_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), config["gradient_clip_norm"])
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    async def _validate_epoch(
        self,
        model: BaseModel,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        model.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move to device
                if model.device:
                    inputs = inputs.to(model.device)
                    targets = targets.to(model.device)

                # Forward pass
                outputs = model.model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def _should_early_stop(self, training_history: List[Dict], config: Dict[str, Any]) -> bool:
        """Check if early stopping should be triggered."""
        patience = config["early_stopping_patience"]
        if len(training_history) < patience + 1:
            return False

        # Check if validation loss hasn't improved
        recent_losses = [
            epoch["val"]["loss"] if epoch["val"] else epoch["train"]["loss"]
            for epoch in training_history[-patience-1:]
        ]

        # If the minimum loss in the last patience epochs is not better than patience epochs ago
        min_recent = min(recent_losses[-patience:])
        loss_patience_ago = recent_losses[0]

        return min_recent >= loss_patience_ago

    async def _save_model_checkpoint(
        self,
        model: BaseModel,
        model_name: str,
        training_history: List[Dict]
    ) -> None:
        """Save model checkpoint after training."""
        checkpoint_path = self.training_dir / f"{model_name}_checkpoint.pth"

        checkpoint = {
            "model_state_dict": model.model.state_dict(),
            "model_name": model_name,
            "training_history": training_history,
            "timestamp": time.time(),
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved model checkpoint: {checkpoint_path}")

    async def evaluate_model(
        self,
        model_name: str,
        test_dataset: Dataset,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test dataset.

        Args:
            model_name: Name of the model to evaluate
            test_dataset: Test dataset
            metrics: List of metrics to compute

        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")

        # Get model
        model = await self.model_registry.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not loaded")

        # Create test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        # Default metrics
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1"]

        # Evaluate
        results = await self._evaluate_model(model, test_loader, metrics)

        logger.info(f"Evaluation completed for {model_name}: {results}")
        return results

    async def _evaluate_model(
        self,
        model: BaseModel,
        test_loader: DataLoader,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Evaluate model with specified metrics."""
        model.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                if model.device:
                    inputs = inputs.to(model.device)
                    targets = targets.to(model.device)

                outputs = model.model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Compute metrics
        results = {}
        for metric in metrics:
            if metric == "accuracy":
                results["accuracy"] = np.mean(np.array(all_predictions) == np.array(all_targets))
            elif metric == "precision":
                results["precision"] = self._compute_precision(all_predictions, all_targets)
            elif metric == "recall":
                results["recall"] = self._compute_recall(all_predictions, all_targets)
            elif metric == "f1":
                precision = self._compute_precision(all_predictions, all_targets)
                recall = self._compute_recall(all_predictions, all_targets)
                results["f1"] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return results

    def _compute_precision(self, predictions: List[int], targets: List[int]) -> float:
        """Compute precision metric."""
        from sklearn.metrics import precision_score
        return precision_score(targets, predictions, average='weighted', zero_division=0)

    def _compute_recall(self, predictions: List[int], targets: List[int]) -> float:
        """Compute recall metric."""
        from sklearn.metrics import recall_score
        return recall_score(targets, predictions, average='weighted', zero_division=0)

    async def optimize_hyperparameters(
        self,
        model_name: str,
        train_dataset: Dataset,
        val_dataset: Dataset,
        param_space: Dict[str, List[Any]],
        max_trials: int = 20
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using grid search or random search.

        Args:
            model_name: Name of the model to optimize
            train_dataset: Training dataset
            val_dataset: Validation dataset
            param_space: Parameter search space
            max_trials: Maximum number of trials

        Returns:
            Best hyperparameters and results
        """
        logger.info(f"Optimizing hyperparameters for {model_name}")

        best_params = None
        best_score = float('-inf')

        # Simple grid search (can be extended to random search or Bayesian optimization)
        from itertools import product

        param_combinations = list(product(*param_space.values()))
        param_names = list(param_space.keys())

        for i, param_values in enumerate(param_combinations[:max_trials]):
            params = dict(zip(param_names, param_values))

            logger.info(f"Trial {i + 1}/{min(len(param_combinations), max_trials)}: {params}")

            # Train with these parameters
            training_config = {"epochs": 10, **params}  # Shorter training for optimization
            results = await self.train_model(model_name, train_dataset, val_dataset, training_config)

            # Evaluate performance (use negative validation loss as score)
            score = -results["best_val_loss"]

            if score > best_score:
                best_score = score
                best_params = params

        optimization_results = {
            "model_name": model_name,
            "best_params": best_params,
            "best_score": best_score,
            "trials_completed": min(len(param_combinations), max_trials),
        }

        logger.info(f"Hyperparameter optimization completed: {optimization_results}")
        return optimization_results
