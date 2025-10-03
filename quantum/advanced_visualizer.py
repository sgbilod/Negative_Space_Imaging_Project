"""
Advanced Quantum Visualization System
Copyright (c) 2025 Stephen Bilodeau
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from datetime import datetime


class AdvancedQuantumVisualizer:
    def cleanup(self):
        """Clean up visualization resources and close figures."""
        try:
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None
            plt.close('all')
            self.axs = []
            self.animations = []
            self.history = []
        except Exception as e:
            print(f"Visualization cleanup error: {e}")
    """Advanced visualization system with real-time capabilities"""

    def __init__(self):
        self.fig = None
        self.axs = []
        self.animations = []
        self.history = []
        self.max_history = 1000

    def initialize_real_time_display(self, num_plots: int = 4) -> None:
        """Initialize real-time display"""
        self.num_plots = num_plots
        plt.ion()  # Enable interactive mode
        plt.close('all')  # Close any existing figures

    def update_quantum_state(
        self,
        quantum_state: np.ndarray,
        metrics: Dict[str, Any] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update quantum state visualization with detailed metrics"""
        if not timestamp:
            timestamp = datetime.now()

        # Store in history
        self.history.append((timestamp, quantum_state))
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Clear any existing plots
        plt.clf()

        # Create new figure with subplots
        if self.fig is None:
            self.fig = plt.figure(figsize=(16, 12))

        self.axs = []

        # Create main subplots for quantum states
        channels = ['Potential', 'Spin', 'Entanglement', 'Phase']
        for i, channel in enumerate(channels):
            ax = self.fig.add_subplot(2, 2, i+1, projection='3d')
            self.axs.append(ax)
            self._plot_quantum_metric_advanced(
                ax,
                quantum_state[..., i],
                channel,
                timestamp
            )

            # Add metrics text if available
            if metrics and channel.lower() in metrics:
                metric_data = metrics[channel.lower()]
                if metric_data:
                    ax.text2D(
                        0.05, 0.95,
                        self._format_metrics(metric_data),
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment='top',
                        bbox=dict(
                            facecolor='white',
                            alpha=0.8
                        )
                    )

            # Add metrics if available
            if metrics and channel.lower() in metrics:
                metric_data = metrics[channel.lower()]
                ax.text2D(
                    0.05, 0.95,
                    self._format_metrics(metric_data),
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    bbox=dict(
                        facecolor='white',
                        alpha=0.8
                    )
                )

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to render

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for display"""
        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted = f"{key}: {value:.3f}"
            else:
                formatted = f"{key}: {value}"
            lines.append(formatted)
        return "\n".join(lines)

    def visualize_quantum_evolution(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """Visualize quantum state evolution over time"""
        if not self.history:
            return

        # Filter history by time range
        filtered_history = self._filter_history(start_time, end_time)

        # Create animation
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()
            timestamp, state = filtered_history[frame]
            self._plot_quantum_evolution(ax, state, timestamp)

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(filtered_history),
            interval=50
        )

        self.animations.append(anim)
        plt.show(block=False)  # Non-blocking show

    def visualize_entanglement_network(
        self,
        patterns: Dict[str, Any],
        threshold: float = 0.5
    ) -> None:
        """Visualize quantum entanglement network"""
        # Clear any existing figures
        plt.close('all')

        # Create new figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract entanglement data
        entanglements = patterns['primary_entanglements']
        positions = np.where(entanglements > threshold)

        # Plot entanglement nodes
        scatter = ax.scatter(
            positions[0],
            positions[1],
            positions[2],
            c=entanglements[positions],
            cmap='viridis',
            s=100,
            alpha=0.6
        )

        # Plot entanglement connections
        self._plot_entanglement_connections(ax, positions, entanglements)

        # Add colorbar to correct figure
        fig.colorbar(scatter, ax=ax, label='Entanglement Strength')
        ax.set_title('Quantum Entanglement Network')

        # Show non-blocking
        plt.show(block=False)

    def _plot_quantum_metric_advanced(
        self,
        ax: Axes3D,
        data: np.ndarray,
        metric_name: str,
        timestamp: datetime
    ) -> None:
        """Plot quantum metric with advanced visualization"""
        # Validate and normalize data
        if np.all(data == 0):
            data = np.zeros(data.shape)
        else:
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
            if np.max(np.abs(data)) > 0:
                data = data / np.max(np.abs(data))

        x, y, z = np.indices(data.shape)
        values = data.flatten()

        # Create scalar field visualization
        scatter = ax.scatter(
            x.flatten(),
            y.flatten(),
            z.flatten(),
            c=values,
            cmap='viridis',
            alpha=0.6,
            s=50 * (0.1 + np.abs(values))  # Size varies with value
        )

        # Add colorbar and labels
        plt.colorbar(scatter, ax=ax, label=f'{metric_name} Magnitude')
        ax.set_title(f'{metric_name} Distribution')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Add timestamp and metric info
        # Format timestamp nicely
        time_str = timestamp.strftime('%H:%M:%S.%f')[:-3]
        ax.set_title(f"{metric_name} at {time_str}")
        # Add colorbar to correct figure
        plt.gcf().colorbar(scatter, ax=ax)

        # Add quantum field lines
        self._add_field_lines(ax, data)

    def _add_field_lines(self, ax: Axes3D, data: np.ndarray) -> None:
        """Add quantum field lines to visualization"""
        # Calculate gradient field
        gx, gy, gz = np.gradient(data)

        # Plot field lines
        x, y, z = np.indices(data.shape)
        mask = np.random.random(data.shape) < 0.1  # Randomly sample points

        ax.quiver(
            x[mask],
            y[mask],
            z[mask],
            gx[mask],
            gy[mask],
            gz[mask],
            length=0.5,
            normalize=True,
            color='red',
            alpha=0.3
        )

    def _plot_quantum_evolution(
        self,
        ax: Axes3D,
        state: np.ndarray,
        timestamp: datetime
    ) -> None:
        """Plot quantum state evolution"""
        # Calculate aggregate quantum field
        field = np.mean(state, axis=-1)
        x, y, z = np.indices(field.shape)

        scatter = ax.scatter(
            x.flatten(),
            y.flatten(),
            z.flatten(),
            c=field.flatten(),
            cmap='plasma',
            alpha=0.6
        )

        time_str = timestamp.strftime('%H:%M:%S.%f')[:-3]
        ax.set_title(f"Quantum Evolution at {time_str}")
        plt.colorbar(scatter, ax=ax)

    def _plot_entanglement_connections(
        self,
        ax: Axes3D,
        positions: tuple,
        strengths: np.ndarray
    ) -> None:
        """Plot connections between entangled points"""
        for i in range(len(positions[0])):
            for j in range(i + 1, len(positions[0])):
                strength = min(
                    strengths[tuple([positions[k][i] for k in range(3)])],
                    strengths[tuple([positions[k][j] for k in range(3)])]
                )

                if strength > 0.7:  # Only show strong connections
                    ax.plot(
                        [positions[0][i], positions[0][j]],
                        [positions[1][i], positions[1][j]],
                        [positions[2][i], positions[2][j]],
                        'r-',
                        alpha=0.2 * strength
                    )

    def _filter_history(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> List:
        """Filter quantum history by time range"""
        if not start_time:
            start_time = self.history[0][0]
        if not end_time:
            end_time = self.history[-1][0]

        return [
            (t, s) for t, s in self.history
            if start_time <= t <= end_time
        ]
