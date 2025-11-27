"""
Tool Configuration Management.

Manages feature flags and configuration for Advanced Tool Use features.
"""

from typing import Dict, List
from dataclasses import dataclass, field
from enum import Enum
import os


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ToolSearchConfig:
    enabled: bool = True
    default_strategy: str = "hybrid"
    max_results: int = 10
    min_score_threshold: float = 0.1
    always_loaded_token_budget: int = 5000


@dataclass
class PTCConfig:
    enabled: bool = True
    timeout_seconds: int = 60
    max_parallel_calls: int = 50
    max_tool_calls_per_execution: int = 100


@dataclass
class ToolExamplesConfig:
    enabled: bool = True
    max_examples_per_tool: int = 5
    include_expected_output: bool = True


@dataclass
class AdvancedToolUseConfig:
    environment: Environment = Environment.DEVELOPMENT
    tool_search: ToolSearchConfig = field(default_factory=ToolSearchConfig)
    ptc: PTCConfig = field(default_factory=PTCConfig)
    examples: ToolExamplesConfig = field(default_factory=ToolExamplesConfig)
    beta_header: str = "advanced-tool-use-2025-11-20"
    supported_models: List[str] = field(default_factory=lambda: [
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-5-20251101"
    ])

    @classmethod
    def from_environment(cls) -> 'AdvancedToolUseConfig':
        env_str = os.getenv("NSIP_ENVIRONMENT", "development")
        try:
            env = Environment(env_str)
        except ValueError:
            env = Environment.DEVELOPMENT

        config = cls(environment=env)

        if os.getenv("NSIP_TOOL_SEARCH_ENABLED"):
            config.tool_search.enabled = os.getenv("NSIP_TOOL_SEARCH_ENABLED") == "true"
        if os.getenv("NSIP_PTC_ENABLED"):
            config.ptc.enabled = os.getenv("NSIP_PTC_ENABLED") == "true"
        if os.getenv("NSIP_PTC_TIMEOUT"):
            config.ptc.timeout_seconds = int(os.getenv("NSIP_PTC_TIMEOUT"))

        return config

    def to_api_headers(self) -> Dict[str, str]:
        return {"anthropic-beta": self.beta_header}


config = AdvancedToolUseConfig.from_environment()
