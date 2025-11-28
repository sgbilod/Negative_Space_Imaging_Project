#!/usr/bin/env python
"""
HPC Configuration Management
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides configuration management for High Performance Computing
cluster settings, including support for multiple HPC backends (SLURM, PBS, LSF).
"""

from __future__ import annotations

import os
import logging
import yaml
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class HPCBackend(Enum):
    """Supported HPC job scheduler backends."""
    SLURM = "slurm"
    PBS = "pbs"
    LSF = "lsf"
    LOCAL = "local"
    AUTO = "auto"


class ResourceUnit(Enum):
    """Resource measurement units."""
    CORES = "cores"
    GB = "GB"
    MB = "MB"
    NODES = "nodes"
    GPUS = "gpus"


@dataclass
class MemoryConfig:
    """Memory allocation configuration."""
    per_node: int = 64  # GB
    per_task: int = 8   # GB
    reserved: int = 4   # GB reserved for system
    swap_limit: int = 0  # GB, 0 = disabled

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "per_node": self.per_node,
            "per_task": self.per_task,
            "reserved": self.reserved,
            "swap_limit": self.swap_limit,
        }


@dataclass
class ComputeNodeConfig:
    """Configuration for compute nodes."""
    min_nodes: int = 1
    max_nodes: int = 16
    cpus_per_node: int = 32
    gpus_per_node: int = 0
    memory_gb: int = 128
    node_features: List[str] = field(default_factory=list)
    exclusive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "cpus_per_node": self.cpus_per_node,
            "gpus_per_node": self.gpus_per_node,
            "memory_gb": self.memory_gb,
            "node_features": self.node_features,
            "exclusive": self.exclusive,
        }


@dataclass
class QueueConfig:
    """Job queue configuration."""
    name: str = "default"
    priority: int = 100
    max_jobs: int = 100
    max_wall_time: str = "24:00:00"
    max_cpus: int = 1000
    preemptible: bool = False
    partition: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "priority": self.priority,
            "max_jobs": self.max_jobs,
            "max_wall_time": self.max_wall_time,
            "max_cpus": self.max_cpus,
            "preemptible": self.preemptible,
            "partition": self.partition,
        }


@dataclass
class JobSchedulingConfig:
    """Job scheduling configuration."""
    default_queue: str = "default"
    default_wall_time: str = "04:00:00"
    default_cpus: int = 4
    default_memory_gb: int = 16
    retry_count: int = 3
    retry_delay_seconds: int = 60
    dependency_timeout: int = 3600
    max_array_size: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_queue": self.default_queue,
            "default_wall_time": self.default_wall_time,
            "default_cpus": self.default_cpus,
            "default_memory_gb": self.default_memory_gb,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "dependency_timeout": self.dependency_timeout,
            "max_array_size": self.max_array_size,
        }


@dataclass
class NetworkConfig:
    """Network configuration for HPC cluster."""
    interconnect: str = "ethernet"  # ethernet, infiniband, omnipath
    bandwidth_gbps: float = 100.0
    latency_us: float = 1.0
    mpi_enabled: bool = True
    rdma_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interconnect": self.interconnect,
            "bandwidth_gbps": self.bandwidth_gbps,
            "latency_us": self.latency_us,
            "mpi_enabled": self.mpi_enabled,
            "rdma_enabled": self.rdma_enabled,
        }


@dataclass
class StorageConfig:
    """Storage configuration for HPC cluster."""
    scratch_path: str = "/scratch"
    home_path: str = "/home"
    project_path: str = "/project"
    parallel_fs: bool = True
    quota_gb: int = 1000
    purge_days: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scratch_path": self.scratch_path,
            "home_path": self.home_path,
            "project_path": self.project_path,
            "parallel_fs": self.parallel_fs,
            "quota_gb": self.quota_gb,
            "purge_days": self.purge_days,
        }


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    enabled: bool = True
    metrics_interval: int = 30
    log_level: str = "INFO"
    prometheus_enabled: bool = True
    grafana_dashboard: bool = True
    alert_email: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "metrics_interval": self.metrics_interval,
            "log_level": self.log_level,
            "prometheus_enabled": self.prometheus_enabled,
            "grafana_dashboard": self.grafana_dashboard,
            "alert_email": self.alert_email,
        }


@dataclass
class HPCConfig:
    """
    Main HPC configuration class.

    Manages all aspects of HPC cluster configuration including compute nodes,
    memory allocation, job scheduling, queues, networking, and storage.

    Example:
        >>> config = HPCConfig.from_environment()
        >>> config.backend
        <HPCBackend.SLURM: 'slurm'>
        >>> config.validate()
        True
    """

    # Core settings
    backend: HPCBackend = HPCBackend.AUTO
    cluster_name: str = "negative-space-hpc"
    environment: str = "production"

    # Component configurations
    compute: ComputeNodeConfig = field(default_factory=ComputeNodeConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    scheduling: JobSchedulingConfig = field(default_factory=JobSchedulingConfig)
    queues: List[QueueConfig] = field(default_factory=lambda: [QueueConfig()])
    network: NetworkConfig = field(default_factory=NetworkConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Advanced settings
    modules: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    custom_directives: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        if self.backend == HPCBackend.AUTO:
            self.backend = self._detect_backend()
        logger.info(f"HPC Config initialized with backend: {self.backend.value}")

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> HPCConfig:
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration file

        Returns:
            HPCConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HPCConfig:
        """
        Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            HPCConfig instance
        """
        backend = HPCBackend(data.get("backend", "auto"))

        compute_data = data.get("compute", {})
        compute = ComputeNodeConfig(**compute_data)

        memory_data = data.get("memory", {})
        memory = MemoryConfig(**memory_data)

        scheduling_data = data.get("scheduling", {})
        scheduling = JobSchedulingConfig(**scheduling_data)

        queues_data = data.get("queues", [{}])
        queues = [QueueConfig(**q) for q in queues_data]

        network_data = data.get("network", {})
        network = NetworkConfig(**network_data)

        storage_data = data.get("storage", {})
        storage = StorageConfig(**storage_data)

        monitoring_data = data.get("monitoring", {})
        monitoring = MonitoringConfig(**monitoring_data)

        return cls(
            backend=backend,
            cluster_name=data.get("cluster_name", "negative-space-hpc"),
            environment=data.get("environment", "production"),
            compute=compute,
            memory=memory,
            scheduling=scheduling,
            queues=queues,
            network=network,
            storage=storage,
            monitoring=monitoring,
            modules=data.get("modules", []),
            environment_variables=data.get("environment_variables", {}),
            custom_directives=data.get("custom_directives", {}),
        )

    @classmethod
    def from_environment(cls) -> HPCConfig:
        """
        Create configuration from environment variables.

        Reads configuration from environment variables prefixed with HPC_.

        Returns:
            HPCConfig instance
        """
        env_vars = {
            k.replace("HPC_", "").lower(): v
            for k, v in os.environ.items()
            if k.startswith("HPC_")
        }

        backend_str = env_vars.get("backend", "auto")
        backend = HPCBackend(backend_str)

        compute = ComputeNodeConfig(
            min_nodes=int(env_vars.get("min_nodes", 1)),
            max_nodes=int(env_vars.get("max_nodes", 16)),
            cpus_per_node=int(env_vars.get("cpus_per_node", 32)),
            gpus_per_node=int(env_vars.get("gpus_per_node", 0)),
        )

        memory = MemoryConfig(
            per_node=int(env_vars.get("memory_per_node", 64)),
            per_task=int(env_vars.get("memory_per_task", 8)),
        )

        return cls(
            backend=backend,
            cluster_name=env_vars.get("cluster_name", "negative-space-hpc"),
            environment=env_vars.get("environment", "production"),
            compute=compute,
            memory=memory,
        )

    def _detect_backend(self) -> HPCBackend:
        """
        Auto-detect the HPC backend.

        Returns:
            Detected HPCBackend
        """
        # Check for SLURM
        if os.path.exists("/usr/bin/sbatch") or os.getenv("SLURM_JOB_ID"):
            logger.info("Detected SLURM scheduler")
            return HPCBackend.SLURM

        # Check for PBS
        if os.path.exists("/usr/bin/qsub") or os.getenv("PBS_JOBID"):
            logger.info("Detected PBS scheduler")
            return HPCBackend.PBS

        # Check for LSF
        if os.path.exists("/usr/bin/bsub") or os.getenv("LSB_JOBID"):
            logger.info("Detected LSF scheduler")
            return HPCBackend.LSF

        logger.warning(
            "No HPC scheduler detected, using local mode. "
            "Set HPC_BACKEND environment variable to override."
        )
        return HPCBackend.LOCAL

    def validate(self) -> bool:
        """
        Validate the configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        errors = []

        # Validate compute settings
        if self.compute.min_nodes > self.compute.max_nodes:
            errors.append("min_nodes cannot exceed max_nodes")

        if self.compute.cpus_per_node < 1:
            errors.append("cpus_per_node must be at least 1")

        # Validate memory settings
        if self.memory.per_task > self.memory.per_node:
            errors.append("memory.per_task cannot exceed memory.per_node")

        # Validate scheduling settings
        if self.scheduling.retry_count < 0:
            errors.append("retry_count cannot be negative")

        # Validate queues
        if not self.queues:
            errors.append("At least one queue must be defined")

        for queue in self.queues:
            if queue.priority < 0 or queue.priority > 1000:
                errors.append(f"Queue {queue.name} priority must be 0-1000")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

        logger.info("Configuration validation passed")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "backend": self.backend.value,
            "cluster_name": self.cluster_name,
            "environment": self.environment,
            "compute": self.compute.to_dict(),
            "memory": self.memory.to_dict(),
            "scheduling": self.scheduling.to_dict(),
            "queues": [q.to_dict() for q in self.queues],
            "network": self.network.to_dict(),
            "storage": self.storage.to_dict(),
            "monitoring": self.monitoring.to_dict(),
            "modules": self.modules,
            "environment_variables": self.environment_variables,
            "custom_directives": self.custom_directives,
        }

    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            path: Path to save the configuration
        """
        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        logger.info(f"Configuration saved to {path}")

    def get_job_script_header(self, job_name: str = "nsi_job") -> str:
        """
        Generate job script header based on backend.

        Args:
            job_name: Name for the job

        Returns:
            Job script header string
        """
        if self.backend == HPCBackend.SLURM:
            return self._get_slurm_header(job_name)
        elif self.backend == HPCBackend.PBS:
            return self._get_pbs_header(job_name)
        elif self.backend == HPCBackend.LSF:
            return self._get_lsf_header(job_name)
        else:
            return "#!/bin/bash\n# Local execution mode\n"

    def _get_slurm_header(self, job_name: str) -> str:
        """Generate SLURM job script header."""
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --nodes={self.compute.min_nodes}",
            f"#SBATCH --ntasks-per-node={self.compute.cpus_per_node}",
            f"#SBATCH --time={self.scheduling.default_wall_time}",
            f"#SBATCH --mem={self.scheduling.default_memory_gb}G",
        ]

        if self.compute.gpus_per_node > 0:
            lines.append(f"#SBATCH --gres=gpu:{self.compute.gpus_per_node}")

        default_queue = next(
            (q for q in self.queues if q.name == self.scheduling.default_queue),
            self.queues[0] if self.queues else None
        )
        if default_queue and default_queue.partition:
            lines.append(f"#SBATCH --partition={default_queue.partition}")

        if self.compute.exclusive:
            lines.append("#SBATCH --exclusive")

        for key, value in self.custom_directives.items():
            lines.append(f"#SBATCH --{key}={value}")

        lines.append("")

        # Add module loads
        for module in self.modules:
            lines.append(f"module load {module}")

        # Add environment variables
        for key, value in self.environment_variables.items():
            lines.append(f"export {key}={value}")

        lines.append("")
        return "\n".join(lines)

    def _get_pbs_header(self, job_name: str) -> str:
        """Generate PBS job script header."""
        lines = [
            "#!/bin/bash",
            f"#PBS -N {job_name}",
            f"#PBS -l nodes={self.compute.min_nodes}:ppn={self.compute.cpus_per_node}",
            f"#PBS -l walltime={self.scheduling.default_wall_time}",
            f"#PBS -l mem={self.scheduling.default_memory_gb}gb",
        ]

        if self.compute.gpus_per_node > 0:
            lines.append(f"#PBS -l ngpus={self.compute.gpus_per_node}")

        for key, value in self.custom_directives.items():
            lines.append(f"#PBS {key}={value}")

        lines.append("")

        for module in self.modules:
            lines.append(f"module load {module}")

        for key, value in self.environment_variables.items():
            lines.append(f"export {key}={value}")

        lines.append("")
        return "\n".join(lines)

    def _get_lsf_header(self, job_name: str) -> str:
        """Generate LSF job script header."""
        lines = [
            "#!/bin/bash",
            f"#BSUB -J {job_name}",
            f"#BSUB -n {self.compute.min_nodes * self.compute.cpus_per_node}",
            f"#BSUB -W {self.scheduling.default_wall_time}",
            f"#BSUB -M {self.scheduling.default_memory_gb * 1024}",
        ]

        if self.compute.gpus_per_node > 0:
            lines.append(f"#BSUB -gpu \"num={self.compute.gpus_per_node}\"")

        for key, value in self.custom_directives.items():
            lines.append(f"#BSUB {key} {value}")

        lines.append("")

        for module in self.modules:
            lines.append(f"module load {module}")

        for key, value in self.environment_variables.items():
            lines.append(f"export {key}={value}")

        lines.append("")
        return "\n".join(lines)


def get_default_config() -> HPCConfig:
    """
    Get default HPC configuration.

    Returns:
        Default HPCConfig instance
    """
    return HPCConfig()


def load_config(
    path: Optional[Union[str, Path]] = None,
    use_env: bool = True
) -> HPCConfig:
    """
    Load HPC configuration from file or environment.

    Args:
        path: Optional path to configuration file
        use_env: Whether to use environment variables

    Returns:
        HPCConfig instance
    """
    if path:
        return HPCConfig.from_file(path)
    elif use_env:
        return HPCConfig.from_environment()
    else:
        return get_default_config()
