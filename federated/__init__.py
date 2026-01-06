"""
Federated Learning Framework
Privacy-preserving federated learning for distributed training across hospitals and observatories.

Modules:
- flower_integration: Flower framework setup and coordination
- federated_client: Client-side training and privacy
- federated_server: Server-side aggregation and model management
- differential_privacy: DP-SGD implementation with privacy budgeting
- data_privacy: Local data handling and anonymization
- communication: Secure serialization and transmission
- deployment: Docker/Kubernetes deployment configuration
- healthcare_astronomy_setup: Multi-institutional scenario setup
"""

from .flower_integration import (
    FederatedClient,
    FederatedServer,
    FlowerCoordinator,
)
from .federated_client import ClientTrainer, LocalDataManager
from .federated_server import AggregationStrategy, GlobalEvaluator
from .differential_privacy import DifferentialPrivacyManager, PrivacyBudget
from .data_privacy import DataPrivacyManager, DataValidator
from .communication import SecureSerializer, CommunicationProtocol

__version__ = "1.0.0"
__all__ = [
    "FederatedClient",
    "FederatedServer",
    "FlowerCoordinator",
    "ClientTrainer",
    "LocalDataManager",
    "AggregationStrategy",
    "GlobalEvaluator",
    "DifferentialPrivacyManager",
    "PrivacyBudget",
    "DataPrivacyManager",
    "DataValidator",
    "SecureSerializer",
    "CommunicationProtocol",
]
