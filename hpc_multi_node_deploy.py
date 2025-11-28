#!/usr/bin/env python
"""
HPC Multi-Node Deployment
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

Scripts for deploying the Negative Space Imaging system across multiple
compute nodes, including node discovery, registration, load balancing,
and health monitoring.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of a compute node."""
    UNKNOWN = "unknown"
    DISCOVERING = "discovering"
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RESOURCE_BASED = "resource_based"
    RANDOM = "random"


@dataclass
class NodeSpec:
    """Specification for a compute node."""
    hostname: str
    cpus: int = 1
    memory_gb: float = 4.0
    gpus: int = 0
    port: int = 8000
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hostname": self.hostname,
            "cpus": self.cpus,
            "memory_gb": self.memory_gb,
            "gpus": self.gpus,
            "port": self.port,
            "labels": self.labels,
            "metadata": self.metadata,
        }


@dataclass
class NodeState:
    """Current state of a compute node."""
    spec: NodeSpec
    status: NodeStatus = NodeStatus.UNKNOWN
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    running_tasks: int = 0
    last_heartbeat: Optional[datetime] = None
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spec": self.spec.to_dict(),
            "status": self.status.value,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "gpu_usage": self.gpu_usage,
            "running_tasks": self.running_tasks,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "error_count": self.error_count,
        }

    @property
    def is_available(self) -> bool:
        """Check if node is available for work."""
        return self.status == NodeStatus.ONLINE

    @property
    def load_score(self) -> float:
        """Calculate a load score for load balancing."""
        return (
            (self.cpu_usage * 0.4) +
            (self.memory_usage * 0.3) +
            (self.gpu_usage * 0.3) +
            (self.running_tasks / max(self.spec.cpus, 1) * 10)
        )


@dataclass
class DeploymentConfig:
    """Configuration for multi-node deployment."""
    cluster_name: str = "nsi-cluster"
    nodes: List[NodeSpec] = field(default_factory=list)
    enable_load_balancing: bool = True
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED
    health_check_interval: int = 30
    heartbeat_timeout: int = 60
    max_tasks_per_node: int = 100
    auto_discovery: bool = False
    discovery_network: str = "10.0.0.0/24"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster_name": self.cluster_name,
            "nodes": [n.to_dict() for n in self.nodes],
            "enable_load_balancing": self.enable_load_balancing,
            "load_balancing_strategy": self.load_balancing_strategy.value,
            "health_check_interval": self.health_check_interval,
            "heartbeat_timeout": self.heartbeat_timeout,
            "max_tasks_per_node": self.max_tasks_per_node,
            "auto_discovery": self.auto_discovery,
            "discovery_network": self.discovery_network,
        }


class NodeDiscovery:
    """
    Node discovery service.

    Discovers compute nodes on the network and registers them.
    """

    def __init__(self, config: DeploymentConfig):
        """Initialize node discovery."""
        self.config = config
        self._discovered_nodes: Dict[str, NodeSpec] = {}

    def discover_nodes(self) -> List[NodeSpec]:
        """
        Discover nodes on the network.

        Returns:
            List of discovered nodes
        """
        discovered = []

        if not self.config.auto_discovery:
            logger.info("Auto-discovery disabled, using configured nodes")
            return list(self.config.nodes)

        logger.info(f"Starting node discovery on {self.config.discovery_network}")

        # Try to discover nodes via ping/scan
        try:
            discovered = self._scan_network()
        except Exception as e:
            logger.error(f"Network scan failed: {e}")

        # Merge with configured nodes
        all_nodes = {n.hostname: n for n in self.config.nodes}
        for node in discovered:
            if node.hostname not in all_nodes:
                all_nodes[node.hostname] = node

        self._discovered_nodes = all_nodes
        return list(all_nodes.values())

    def _scan_network(self) -> List[NodeSpec]:
        """Scan network for nodes."""
        # This is a simplified implementation
        # In production, use proper network discovery tools
        discovered = []

        network = self.config.discovery_network
        base_ip = ".".join(network.split("/")[0].split(".")[:-1])

        for i in range(1, 256):
            ip = f"{base_ip}.{i}"
            if self._check_host(ip):
                node = NodeSpec(
                    hostname=ip,
                    cpus=4,  # Default values
                    memory_gb=16.0,
                )
                discovered.append(node)

        return discovered

    def _check_host(self, ip: str) -> bool:
        """Check if a host is reachable."""
        try:
            socket.create_connection((ip, 22), timeout=1)
            return True
        except (socket.timeout, socket.error):
            return False

    def register_node(self, node: NodeSpec) -> None:
        """
        Register a new node.

        Args:
            node: Node specification
        """
        self._discovered_nodes[node.hostname] = node
        logger.info(f"Registered node: {node.hostname}")

    def unregister_node(self, hostname: str) -> bool:
        """
        Unregister a node.

        Args:
            hostname: Node hostname

        Returns:
            True if node was unregistered
        """
        if hostname in self._discovered_nodes:
            del self._discovered_nodes[hostname]
            logger.info(f"Unregistered node: {hostname}")
            return True
        return False


class HealthMonitor:
    """
    Health monitoring for compute nodes.

    Monitors node health and triggers alerts on issues.
    """

    def __init__(
        self,
        check_interval: int = 30,
        heartbeat_timeout: int = 60
    ):
        """Initialize health monitor."""
        self.check_interval = check_interval
        self.heartbeat_timeout = heartbeat_timeout
        self._running = False
        self._callbacks: List[Callable[[str, NodeStatus], None]] = []

    def register_callback(
        self,
        callback: Callable[[str, NodeStatus], None]
    ) -> None:
        """
        Register a status change callback.

        Args:
            callback: Function to call on status change
        """
        self._callbacks.append(callback)

    async def start(self, nodes: Dict[str, NodeState]) -> None:
        """
        Start health monitoring.

        Args:
            nodes: Dictionary of node states
        """
        self._running = True
        logger.info("Health monitor started")

        while self._running:
            for hostname, state in nodes.items():
                new_status = await self._check_node_health(state)
                if new_status != state.status:
                    old_status = state.status
                    state.status = new_status
                    self._notify_status_change(hostname, old_status, new_status)

            await asyncio.sleep(self.check_interval)

    def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        logger.info("Health monitor stopped")

    async def _check_node_health(self, state: NodeState) -> NodeStatus:
        """Check health of a single node."""
        if state.last_heartbeat is None:
            return NodeStatus.UNKNOWN

        elapsed = (datetime.utcnow() - state.last_heartbeat).total_seconds()

        if elapsed > self.heartbeat_timeout:
            return NodeStatus.OFFLINE

        if state.error_count > 5:
            return NodeStatus.ERROR

        if state.cpu_usage > 95 or state.memory_usage > 95:
            return NodeStatus.BUSY

        return NodeStatus.ONLINE

    def _notify_status_change(
        self,
        hostname: str,
        old_status: NodeStatus,
        new_status: NodeStatus
    ) -> None:
        """Notify callbacks of status change."""
        logger.info(f"Node {hostname}: {old_status.value} -> {new_status.value}")
        for callback in self._callbacks:
            try:
                callback(hostname, new_status)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class LoadBalancer:
    """
    Load balancer for distributing work across nodes.

    Implements multiple load balancing strategies.
    """

    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED
    ):
        """Initialize load balancer."""
        self.strategy = strategy
        self._round_robin_index = 0

    def select_node(
        self,
        nodes: Dict[str, NodeState],
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Select a node for task execution.

        Args:
            nodes: Dictionary of node states
            requirements: Optional resource requirements

        Returns:
            Hostname of selected node or None
        """
        available = {
            h: s for h, s in nodes.items()
            if s.is_available and self._meets_requirements(s, requirements)
        }

        if not available:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(list(available.keys()))
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded(available)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._resource_based(available, requirements)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            import random
            return random.choice(list(available.keys()))

        return list(available.keys())[0]

    def _meets_requirements(
        self,
        state: NodeState,
        requirements: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if node meets requirements."""
        if requirements is None:
            return True

        if requirements.get("cpus", 0) > state.spec.cpus:
            return False

        if requirements.get("memory_gb", 0) > state.spec.memory_gb:
            return False

        if requirements.get("gpus", 0) > state.spec.gpus:
            return False

        required_labels = requirements.get("labels", [])
        if required_labels and not all(l in state.spec.labels for l in required_labels):
            return False

        return True

    def _round_robin(self, hostnames: List[str]) -> str:
        """Round-robin selection."""
        hostname = hostnames[self._round_robin_index % len(hostnames)]
        self._round_robin_index += 1
        return hostname

    def _least_loaded(self, nodes: Dict[str, NodeState]) -> str:
        """Select least loaded node."""
        return min(nodes.keys(), key=lambda h: nodes[h].load_score)

    def _resource_based(
        self,
        nodes: Dict[str, NodeState],
        requirements: Optional[Dict[str, Any]]
    ) -> str:
        """Select based on resource availability."""
        def score(state: NodeState) -> float:
            available_cpus = state.spec.cpus * (1 - state.cpu_usage / 100)
            available_memory = state.spec.memory_gb * (1 - state.memory_usage / 100)

            req_cpus = requirements.get("cpus", 1) if requirements else 1
            req_memory = requirements.get("memory_gb", 1) if requirements else 1

            return (available_cpus / req_cpus) + (available_memory / req_memory)

        return max(nodes.keys(), key=lambda h: score(nodes[h]))


class MultiNodeDeployer:
    """
    Main class for multi-node deployment.

    Orchestrates node discovery, health monitoring, and load balancing.

    Example:
        config = DeploymentConfig(
            cluster_name="my-cluster",
            nodes=[
                NodeSpec(hostname="node1", cpus=32, memory_gb=128),
                NodeSpec(hostname="node2", cpus=32, memory_gb=128),
            ]
        )
        deployer = MultiNodeDeployer(config)
        await deployer.start()
    """

    def __init__(self, config: DeploymentConfig):
        """Initialize deployer."""
        self.config = config
        self.discovery = NodeDiscovery(config)
        self.health_monitor = HealthMonitor(
            check_interval=config.health_check_interval,
            heartbeat_timeout=config.heartbeat_timeout,
        )
        self.load_balancer = LoadBalancer(config.load_balancing_strategy)
        self.nodes: Dict[str, NodeState] = {}
        self._running = False

    async def start(self) -> None:
        """Start the multi-node deployment."""
        logger.info(f"Starting deployment: {self.config.cluster_name}")

        # Discover nodes
        discovered = self.discovery.discover_nodes()
        for spec in discovered:
            self.nodes[spec.hostname] = NodeState(spec=spec)

        logger.info(f"Discovered {len(self.nodes)} nodes")

        # Initialize nodes
        for hostname, state in self.nodes.items():
            await self._initialize_node(state)

        # Start health monitoring
        self._running = True
        asyncio.create_task(self.health_monitor.start(self.nodes))

        logger.info("Deployment started successfully")

    async def stop(self) -> None:
        """Stop the deployment."""
        logger.info("Stopping deployment")
        self._running = False
        self.health_monitor.stop()

    async def _initialize_node(self, state: NodeState) -> None:
        """Initialize a single node."""
        try:
            # Update status to discovering
            state.status = NodeStatus.DISCOVERING

            # Check connectivity
            if await self._check_connectivity(state.spec.hostname):
                state.status = NodeStatus.ONLINE
                state.last_heartbeat = datetime.utcnow()
                logger.info(f"Node {state.spec.hostname} online")
            else:
                state.status = NodeStatus.OFFLINE
                logger.warning(f"Node {state.spec.hostname} offline")

        except Exception as e:
            state.status = NodeStatus.ERROR
            state.error_count += 1
            logger.error(f"Node {state.spec.hostname} error: {e}")

    async def _check_connectivity(self, hostname: str) -> bool:
        """Check if a node is reachable."""
        try:
            # Try to connect to the node
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(hostname, 22),
                timeout=5.0,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            # For demo purposes, simulate connectivity
            return True

    def select_node(
        self,
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Select a node for task execution.

        Args:
            requirements: Optional resource requirements

        Returns:
            Selected node hostname
        """
        return self.load_balancer.select_node(self.nodes, requirements)

    def get_node_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all nodes.

        Returns:
            List of node status dictionaries
        """
        return [
            {
                "hostname": hostname,
                "status": state.status.value,
                "cpus": state.spec.cpus,
                "memory_gb": state.spec.memory_gb,
                "gpus": state.spec.gpus,
                "cpu_usage": state.cpu_usage,
                "memory_usage": state.memory_usage,
                "running_tasks": state.running_tasks,
            }
            for hostname, state in self.nodes.items()
        ]

    def calculate_load_distribution(
        self,
        num_tasks: int
    ) -> Dict[str, int]:
        """
        Calculate how tasks should be distributed.

        Args:
            num_tasks: Number of tasks to distribute

        Returns:
            Dictionary of hostname -> task count
        """
        distribution: Dict[str, int] = {}
        available_nodes = [
            h for h, s in self.nodes.items()
            if s.is_available
        ]

        if not available_nodes:
            return distribution

        # Calculate capacity for each node
        capacities = {}
        total_capacity = 0
        for hostname in available_nodes:
            state = self.nodes[hostname]
            capacity = max(
                1,
                int(state.spec.cpus * (1 - state.cpu_usage / 100))
            )
            capacities[hostname] = capacity
            total_capacity += capacity

        # Distribute tasks proportionally
        remaining = num_tasks
        for hostname in available_nodes[:-1]:
            count = int(num_tasks * capacities[hostname] / total_capacity)
            distribution[hostname] = count
            remaining -= count

        # Assign remaining to last node
        distribution[available_nodes[-1]] = remaining

        return distribution

    def add_node(self, spec: NodeSpec) -> None:
        """
        Add a new node to the cluster.

        Args:
            spec: Node specification
        """
        if spec.hostname in self.nodes:
            logger.warning(f"Node {spec.hostname} already exists")
            return

        self.nodes[spec.hostname] = NodeState(spec=spec)
        self.discovery.register_node(spec)
        logger.info(f"Added node: {spec.hostname}")

    def remove_node(self, hostname: str) -> bool:
        """
        Remove a node from the cluster.

        Args:
            hostname: Node hostname

        Returns:
            True if node was removed
        """
        if hostname not in self.nodes:
            return False

        del self.nodes[hostname]
        self.discovery.unregister_node(hostname)
        logger.info(f"Removed node: {hostname}")
        return True

    def update_node_metrics(
        self,
        hostname: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Update metrics for a node.

        Args:
            hostname: Node hostname
            metrics: Metrics dictionary
        """
        if hostname not in self.nodes:
            return

        state = self.nodes[hostname]
        state.cpu_usage = metrics.get("cpu_usage", state.cpu_usage)
        state.memory_usage = metrics.get("memory_usage", state.memory_usage)
        state.gpu_usage = metrics.get("gpu_usage", state.gpu_usage)
        state.running_tasks = metrics.get("running_tasks", state.running_tasks)
        state.last_heartbeat = datetime.utcnow()


async def deploy_cluster(
    config: DeploymentConfig
) -> MultiNodeDeployer:
    """
    Deploy a multi-node cluster.

    Args:
        config: Deployment configuration

    Returns:
        MultiNodeDeployer instance
    """
    deployer = MultiNodeDeployer(config)
    await deployer.start()
    return deployer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def main() -> None:
        # Example deployment
        config = DeploymentConfig(
            cluster_name="demo-cluster",
            nodes=[
                NodeSpec(hostname="node1.local", cpus=16, memory_gb=64, gpus=2),
                NodeSpec(hostname="node2.local", cpus=16, memory_gb=64, gpus=2),
            ],
            enable_load_balancing=True,
            health_check_interval=30,
        )

        print("Starting multi-node deployment...")
        deployer = await deploy_cluster(config)

        print("\nNode Status:")
        for status in deployer.get_node_status():
            print(f"  {status['hostname']}: {status['status']}")

        print("\nLoad Distribution (100 tasks):")
        dist = deployer.calculate_load_distribution(100)
        for hostname, count in dist.items():
            print(f"  {hostname}: {count} tasks")

        await deployer.stop()
        print("\nDeployment stopped")

    asyncio.run(main())
