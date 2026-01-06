"""
Configuration Loader Module
===========================

Centralized configuration loading utility supporting YAML and JSON formats.
This module provides a single, consistent interface for loading configuration
across the entire application.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("config_loader")

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.debug("PyYAML not available - YAML support disabled")


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    This is the centralized configuration loader used throughout the application.
    It supports both YAML and JSON formats and provides robust error handling.

    Args:
        config_path: Path to configuration file (YAML or JSON).
                    If None or empty, returns empty dict.

    Returns:
        Configuration dictionary, or empty dict if file not found/invalid

    Examples:
        >>> config = load_config("config/app.yaml")
        >>> db_host = config.get("database", {}).get("host")
    """
    if not config_path:
        logger.debug("No config path provided, returning empty config")
        return {}

    path = Path(config_path)

    # Check if file exists
    if not path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    try:
        # Determine format and load
        if path.suffix.lower() in {".yml", ".yaml"}:
            if not YAML_AVAILABLE:
                logger.warning(
                    f"YAML format requested but PyYAML not available. "
                    f"Install with: pip install pyyaml"
                )
                return {}

            logger.debug(f"Loading YAML config from {config_path}")
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                logger.info(f"Successfully loaded YAML config from {config_path}")
                return config

        elif path.suffix.lower() == ".json":
            logger.debug(f"Loading JSON config from {config_path}")
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f) or {}
                logger.info(f"Successfully loaded JSON config from {config_path}")
                return config

        else:
            # Try JSON first, then YAML as fallback
            logger.debug(f"Unknown format, attempting JSON for {config_path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f) or {}
                    logger.info(f"Loaded config as JSON from {config_path}")
                    return config
            except json.JSONDecodeError:
                if YAML_AVAILABLE:
                    logger.debug(f"JSON failed, trying YAML for {config_path}")
                    with open(path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f) or {}
                        logger.info(f"Loaded config as YAML from {config_path}")
                        return config
                else:
                    logger.error(f"Could not parse {config_path} as JSON or YAML")
                    return {}

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON config from {config_path}: {e}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML config from {config_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading config from {config_path}: {e}")
        return {}


def load_config_from_env(
    env_var: str,
    default_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load configuration from environment variable or default path.

    Useful for deployment scenarios where config path is specified via environment.

    Args:
        env_var: Environment variable name to check
        default_path: Path to use if env var not set

    Returns:
        Configuration dictionary

    Example:
        >>> config = load_config_from_env("APP_CONFIG", "config/default.yaml")
    """
    config_path = os.getenv(env_var, default_path)
    return load_config(config_path)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge override config into base config (deep merge).

    Args:
        base: Base configuration dictionary
        override: Configuration to merge in (takes precedence)

    Returns:
        Merged configuration dictionary

    Example:
        >>> base = load_config("config/base.yaml")
        >>> env_override = load_config("config/prod.yaml")
        >>> config = merge_configs(base, env_override)
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursive merge for nested dicts
            result[key] = merge_configs(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    print("=" * 70)
    print("CONFIG LOADER MODULE TEST")
    print("=" * 70)

    # Test 1: Load with None
    print(f"\nTest 1: Loading with None config path...")
    config = load_config(None)
    print(f"Result: {config}")
    assert config == {}, "Expected empty dict"

    # Test 2: Load non-existent file
    print(f"\nTest 2: Loading non-existent file...")
    config = load_config("/nonexistent/path/config_xyz.yaml")
    print(f"Result: {config}")
    assert config == {}, "Expected empty dict"

    # Test 3: Create and load test YAML
    if YAML_AVAILABLE:
        print(f"\nTest 3: Create and load YAML config...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_yaml_path = f.name
            test_config = {
                "database": {"host": "localhost", "port": 5432},
                "api": {"debug": True}
            }
            yaml.dump(test_config, f)

        config = load_config(test_yaml_path)
        print(f"Loaded: {config}")
        assert config == test_config, "Config mismatch"

        # Clean up
        import os
        os.remove(test_yaml_path)

    # Test 4: Create and load test JSON
    print(f"\nTest 4: Create and load JSON config...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_json_path = f.name
        test_config = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"debug": True}
        }
        json.dump(test_config, f)

    config = load_config(test_json_path)
    print(f"Loaded: {config}")
    assert config == test_config, "Config mismatch"

    # Clean up
    import os
    os.remove(test_json_path)

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
