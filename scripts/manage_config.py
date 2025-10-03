#!/usr/bin/env python
"""
Configuration Management System for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module manages all configuration aspects of the project:
1. Environment settings
2. Security parameters
3. Performance tuning
4. Integration configurations
5. User preferences
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class ConfigScope(Enum):
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USER = "user"
    INTEGRATION = "integration"

@dataclass
class ConfigValue:
    value: Any
    scope: ConfigScope
    description: str
    validation: Optional[str] = None

class ConfigurationManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_dir = project_root / "config"
        self.config_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        self.config_cache: Dict[str, ConfigValue] = {}

        # Initialize configuration files
        self._init_config_files()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("config_manager")
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(
            self.project_root / "logs" / "config.log"
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _init_config_files(self):
        """Initialize default configuration files."""
        default_configs = {
            "system_config.yaml": {
                "environment": {
                    "python_version": ">=3.8",
                    "opencv_version": ">=4.7.0",
                    "cuda_support": False
                },
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "security_config.json": {
                "security_level": "ENHANCED",
                "token_expiry_minutes": 60,
                "require_two_factor": True,
                "encryption_algorithm": "AES-256"
            },
            "performance_config.yaml": {
                "image_processing": {
                    "max_image_size": 4096,
                    "default_quality": 95,
                    "use_gpu": False
                },
                "multi_threading": {
                    "max_threads": 8,
                    "thread_pool_size": 4
                }
            },
            "integration_config.yaml": {
                "apis": {
                    "timeout_seconds": 30,
                    "retry_attempts": 3,
                    "verify_ssl": True
                },
                "services": {
                    "image_store": {
                        "type": "local",
                        "path": "data/images"
                    }
                }
            }
        }

        for filename, default_config in default_configs.items():
            config_path = self.config_dir / filename
            if not config_path.exists():
                with open(config_path, 'w') as f:
                    if filename.endswith('.json'):
                        json.dump(default_config, f, indent=2)
                    else:
                        yaml.safe_dump(default_config, f, indent=2)

    def get_config(self, key: str, scope: ConfigScope) -> Optional[ConfigValue]:
        """Get configuration value."""
        cache_key = f"{scope.value}:{key}"

        if cache_key in self.config_cache:
            return self.config_cache[cache_key]

        config_files = {
            ConfigScope.SYSTEM: "system_config.yaml",
            ConfigScope.SECURITY: "security_config.json",
            ConfigScope.PERFORMANCE: "performance_config.yaml",
            ConfigScope.INTEGRATION: "integration_config.yaml"
        }

        if scope not in config_files:
            self.logger.error(f"Invalid config scope: {scope}")
            return None

        config_path = self.config_dir / config_files[scope]
        if not config_path.exists():
            self.logger.error(f"Config file not found: {config_path}")
            return None

        try:
            with open(config_path) as f:
                if config_path.suffix == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)

            # Handle nested keys (e.g., "logging.level")
            value = config
            for part in key.split('.'):
                value = value[part]

            config_value = ConfigValue(
                value=value,
                scope=scope,
                description=f"Configuration value for {key}",
                validation=None  # TODO: Add validation rules
            )

            self.config_cache[cache_key] = config_value
            return config_value

        except Exception as e:
            self.logger.error(f"Error reading config {key}: {str(e)}")
            return None

    def set_config(self, key: str, value: Any, scope: ConfigScope) -> bool:
        """Set configuration value."""
        config_files = {
            ConfigScope.SYSTEM: "system_config.yaml",
            ConfigScope.SECURITY: "security_config.json",
            ConfigScope.PERFORMANCE: "performance_config.yaml",
            ConfigScope.INTEGRATION: "integration_config.yaml"
        }

        if scope not in config_files:
            self.logger.error(f"Invalid config scope: {scope}")
            return False

        config_path = self.config_dir / config_files[scope]
        if not config_path.exists():
            self.logger.error(f"Config file not found: {config_path}")
            return False

        try:
            # Read existing config
            with open(config_path) as f:
                if config_path.suffix == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)

            # Update nested value
            current = config
            parts = key.split('.')
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

            # Write updated config
            with open(config_path, 'w') as f:
                if config_path.suffix == '.json':
                    json.dump(config, f, indent=2)
                else:
                    yaml.safe_dump(config, f, indent=2)

            # Update cache
            cache_key = f"{scope.value}:{key}"
            self.config_cache[cache_key] = ConfigValue(
                value=value,
                scope=scope,
                description=f"Configuration value for {key}"
            )

            self.logger.info(f"Updated config {key} = {value}")
            return True

        except Exception as e:
            self.logger.error(f"Error setting config {key}: {str(e)}")
            return False

    def validate_configs(self) -> Dict[str, bool]:
        """Validate all configuration files."""
        results = {}

        for scope in ConfigScope:
            try:
                if scope == ConfigScope.USER:
                    continue  # Skip user configs for now

                config_files = {
                    ConfigScope.SYSTEM: "system_config.yaml",
                    ConfigScope.SECURITY: "security_config.json",
                    ConfigScope.PERFORMANCE: "performance_config.yaml",
                    ConfigScope.INTEGRATION: "integration_config.yaml"
                }

                config_path = self.config_dir / config_files[scope]
                if not config_path.exists():
                    results[scope.value] = False
                    continue

                with open(config_path) as f:
                    if config_path.suffix == '.json':
                        json.load(f)  # Validate JSON
                    else:
                        yaml.safe_load(f)  # Validate YAML

                results[scope.value] = True

            except Exception as e:
                self.logger.error(f"Config validation failed for {scope}: {str(e)}")
                results[scope.value] = False

        return results

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    config_manager = ConfigurationManager(project_root)

    # Validate all configs
    validation_results = config_manager.validate_configs()
    print("Configuration Validation Results:")
    for scope, is_valid in validation_results.items():
        print(f"{scope}: {'✅' if is_valid else '❌'}")
