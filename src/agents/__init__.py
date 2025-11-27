"""
Agents Module - Negative Space Imaging Project

Provides autonomous agent orchestration and persistent memory management
for the imaging pipeline.

This module exports:
- DeepAgent Supervisor: Hierarchical task orchestration
- Specialized Agents: Acquisition, Reconstruction, Analysis
- Memory System: Persistent storage with decay and LSH-based similarity search

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

from __future__ import annotations

# =============================================================================
# Supervisor Exports
# =============================================================================

from src.agents.supervisor import (
    # Enums
    AgentState,
    TaskPriority,
    # Data classes
    AgentTask,
    AgentResult,
    # Base class
    ImagingAgent,
    # Specialized agents
    AcquisitionAgent,
    ReconstructionAgent,
    AnalysisAgent,
    # Supervisor
    DeepAgentSupervisor,
    # Factory
    create_imaging_supervisor,
)

# =============================================================================
# Memory System Exports
# =============================================================================

from src.agents.memory_system import (
    # Enums
    MemoryType,
    DecayStrategy,
    # Data class
    MemoryEntry,
    # Decay functions
    DecayFunction,
    # Spatial signature cache
    SpatialSignatureCache,
    # Memory manager
    PersistentMemoryManager,
    # Factory
    create_memory_manager,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Supervisor enums
    "AgentState",
    "TaskPriority",
    # Supervisor data classes
    "AgentTask",
    "AgentResult",
    # Agent base and implementations
    "ImagingAgent",
    "AcquisitionAgent",
    "ReconstructionAgent",
    "AnalysisAgent",
    # Supervisor
    "DeepAgentSupervisor",
    "create_imaging_supervisor",
    # Memory enums
    "MemoryType",
    "DecayStrategy",
    # Memory data class
    "MemoryEntry",
    # Memory utilities
    "DecayFunction",
    "SpatialSignatureCache",
    # Memory manager
    "PersistentMemoryManager",
    "create_memory_manager",
]

# =============================================================================
# Module Info
# =============================================================================

__version__ = "1.0.0"
__author__ = "Stephen Bilodeau"
