"""
Advanced Quantum Visualization System
Copyright (c) 2025 Stephen Bilodeau
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for better stability
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

logger = logging.getLogger(__name__)

class VisualizationError(Exception):
    """Base exception for visualization errors."""
    pass

class AdvancedQuantumVisualizer:
    """Advanced visualization system with real-time capabilities and proper resource management."""

    def __init__(self):
        """Initialize the visualizer with proper error handling."""
        self.fig = None
        self.axs = []
        self.animations = []
        self.history: List[Tuple[datetime, np.ndarray]] = []
        self.max_history = 1000
        self._initialized = False

    def initialize_real_time_display(self, num_plots: int = 4) -> None:
        """Initialize real-time display with error handling.

        Args:
            num_plots: Number of subplot panels to create

        Raises:
            VisualizationError: If initialization fails
        """
        try:
            self.num_plots = num_plots
            plt.ion()  # Enable interactive mode
            plt.close('all')  # Close any existing figures
            self.fig = plt.figure(figsize=(16, 12))
            self._initialized = True
            logger.info("Real-time display initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize display: {e}")
            raise VisualizationError(f"Display initialization failed: {e}")

    def update_quantum_state(
        self,
        quantum_state: np.ndarray,
        metrics: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update quantum state visualization with defensive checks.

        Args:
            quantum_state: Numpy array of quantum state data
            metrics: Optional dict of metrics to display
            timestamp: Optional timestamp for the update

        Raises:
            VisualizationError: If update fails
        """
        if not self._initialized:
            logger.warning("Attempted update before initialization")
            return

        if quantum_state is None:
            logger.warning("Received null quantum state")
            return

        try:
            if not timestamp:
                timestamp = datetime.now()

            # Store in history with bounds checking
            self.history.append((timestamp, quantum_state.copy()))
            while len(self.history) > self.max_history:
                self.history.pop(0)

            # Clear existing plots safely
            if self.fig is not None:
                plt.figure(self.fig.number)
                plt.clf()

            # Update visualization
            self._update_plots(quantum_state, metrics)
            plt.draw()
            plt.pause(0.01)  # Small pause for display update

        except Exception as e:
            logger.error(f"Visualization update failed: {e}")
            raise VisualizationError(f"Failed to update visualization: {e}")

    def _update_plots(
        self,
        quantum_state: np.ndarray,
        metrics: Optional[Dict[str, Any]]
    ) -> None:
        """Internal method to update plot panels."""
        if self.fig is None:
            return

        try:
            # Create subplot grid
            grid_size = int(np.ceil(np.sqrt(self.num_plots)))

            # Plot 1: State magnitude
            ax1 = self.fig.add_subplot(grid_size, grid_size, 1)
            ax1.plot(np.abs(quantum_state))
            ax1.set_title('State Magnitude')

            # Plot 2: Phase
            ax2 = self.fig.add_subplot(grid_size, grid_size, 2)
            ax2.plot(np.angle(quantum_state))
            ax2.set_title('Phase')

            # Plot 3: Time evolution if history exists
            if self.history:
                ax3 = self.fig.add_subplot(grid_size, grid_size, 3)
                times = [t for t, _ in self.history]
                values = [np.mean(np.abs(s)) for _, s in self.history]
                ax3.plot(times, values)
                ax3.set_title('Time Evolution')

            # Plot 4: Metrics if provided
            if metrics and len(metrics) > 0:
                ax4 = self.fig.add_subplot(grid_size, grid_size, 4)
                ax4.axis('off')
                metric_text = '\n'.join(f'{k}: {v}' for k, v in metrics.items())
                ax4.text(0.1, 0.9, metric_text, transform=ax4.transAxes)

            self.fig.tight_layout()

        except Exception as e:
            logger.error(f"Plot update failed: {e}")
            raise VisualizationError(f"Failed to update plots: {e}")

    def cleanup(self) -> None:
        """Properly cleanup visualization resources."""
        try:
            # Stop any active animations
            for anim in self.animations:
                anim.event_source.stop()
            self.animations.clear()

            # Close all figures
            plt.close('all')

            # Clear references
            self.fig = None
            self.axs.clear()
            self._initialized = False

            logger.info("Visualization cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise VisualizationError(f"Failed to cleanup visualization: {e}")
