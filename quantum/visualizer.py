"""
Quantum Visualization System
Copyright (c) 2025 Stephen Bilodeau
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from mpl_toolkits.mplot3d import Axes3D


class QuantumVisualizer:
    """Visualization system for quantum operations"""

    def __init__(self):
        self.fig = None
        self.axs = None

    def visualize_quantum_state(
        self,
        quantum_state: np.ndarray,
        title: str = "Quantum State Visualization"
    ) -> None:
        """Visualize quantum state in 3D"""
        self.fig = plt.figure(figsize=(15, 10))
        self.axs = []

        # Create subplots for each quantum metric
        metrics = ['Potential', 'Spin', 'Entanglement', 'Phase']
        for i in range(4):
            ax = self.fig.add_subplot(2, 2, i+1, projection='3d')
            self._plot_quantum_metric(ax, quantum_state[..., i], metrics[i])
            self.axs.append(ax)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def visualize_entanglement(
        self,
        patterns: Dict[str, Any],
        title: str = "Entanglement Patterns"
    ) -> None:
        """Visualize entanglement patterns"""
        self.fig = plt.figure(figsize=(15, 10))
        self.axs = []

        # Plot each pattern type
        for i, (name, pattern) in enumerate(patterns.items()):
            if isinstance(pattern, np.ndarray):
                ax = self.fig.add_subplot(2, 2, i+1, projection='3d')
                self._plot_pattern(ax, pattern, name.replace('_', ' ').title())
                self.axs.append(ax)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def visualize_probability_field(
        self,
        field: np.ndarray,
        title: str = "Probability Field"
    ) -> None:
        """Visualize probability field"""
        self.fig = plt.figure(figsize=(10, 8))
        ax = self.fig.add_subplot(111, projection='3d')

        self._plot_field(ax, field)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def _plot_quantum_metric(
        self,
        ax: Axes3D,
        data: np.ndarray,
        metric_name: str
    ) -> None:
        """Plot a single quantum metric"""
        x, y, z = np.indices(data.shape)
        scatter = ax.scatter(
            x.flatten(),
            y.flatten(),
            z.flatten(),
            c=data.flatten(),
            cmap='viridis',
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax)
        ax.set_title(f"{metric_name} Distribution")

    def _plot_pattern(self, ax: Axes3D, pattern: np.ndarray, name: str) -> None:
        """Plot an entanglement pattern"""
        x, y, z = np.indices(pattern.shape)
        mask = pattern > 0
        scatter = ax.scatter(
            x[mask],
            y[mask],
            z[mask],
            c=pattern[mask],
            cmap='plasma',
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax)
        ax.set_title(name)

    def _plot_field(self, ax: Axes3D, field: np.ndarray) -> None:
        """Plot probability field"""
        x, y, z = np.indices(field.shape)
        scatter = ax.scatter(
            x.flatten(),
            y.flatten(),
            z.flatten(),
            c=field.flatten(),
            cmap='magma',
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax)
        ax.set_title("Probability Distribution")
