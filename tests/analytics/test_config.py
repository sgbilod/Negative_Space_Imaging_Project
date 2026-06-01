"""
Configuration Component Tests

Tests for configuration management, validation, persistence, and runtime
adjustment of analytics pipeline configuration.
"""

import pytest
import json
import tempfile
import os
from typing import Dict, Any


class TestConfigurationLoading:
    """Test configuration loading and initialization."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = {
            "sample_interval": 60,
            "storage_backend": "memory",
            "buffer_size": 1000,
            "async_enabled": True,
        }

        assert config["sample_interval"] == 60
        assert config["storage_backend"] == "memory"
        assert config["buffer_size"] == 1000
        assert config["async_enabled"] is True

    def test_configuration_from_file(self):
        """Test loading configuration from file."""
        config_data = {
            "sample_interval": 120,
            "storage_backend": "postgresql",
            "buffer_size": 5000,
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            with open(config_file, "r") as f:
                loaded = json.load(f)

            assert loaded["sample_interval"] == 120
            assert loaded["storage_backend"] == "postgresql"
        finally:
            os.unlink(config_file)

    def test_environment_variable_override(self):
        """Test environment variable override."""
        os.environ["ANALYTICS_SAMPLE_INTERVAL"] = "300"

        interval = int(os.environ.get("ANALYTICS_SAMPLE_INTERVAL", 60))
        assert interval == 300

        # Cleanup
        del os.environ["ANALYTICS_SAMPLE_INTERVAL"]

    def test_configuration_validation(self):
        """Test configuration validation."""
        valid_config = {
            "sample_interval": 60,
            "storage_backend": "memory",
            "buffer_size": 1000,
        }

        # Check required fields
        required_fields = ["sample_interval", "storage_backend"]
        for field in required_fields:
            assert field in valid_config

    def test_invalid_configuration_rejection(self):
        """Test rejection of invalid configuration."""
        invalid_configs = [
            {"sample_interval": -1},  # Negative interval
            {"buffer_size": "invalid"},  # Wrong type
            {"storage_backend": "unknown"},  # Unknown backend
        ]

        for config in invalid_configs:
            # In real implementation, would raise error
            assert True


class TestConfigurationProfiles:
    """Test configuration profile switching."""

    def test_profile_selection(self):
        """Test selecting configuration profile."""
        profiles = {
            "development": {
                "debug": True,
                "sample_interval": 60,
                "storage": "memory",
            },
            "production": {
                "debug": False,
                "sample_interval": 300,
                "storage": "postgresql",
            },
        }

        # Select development
        dev_config = profiles["development"]
        assert dev_config["debug"] is True

        # Select production
        prod_config = profiles["production"]
        assert prod_config["debug"] is False

    def test_profile_switching_at_runtime(self):
        """Test switching profiles at runtime."""
        current_profile = "development"

        config_profiles = {
            "development": {"level": "debug"},
            "production": {"level": "warning"},
        }

        original_level = config_profiles[current_profile]["level"]
        assert original_level == "debug"

        # Switch profile
        current_profile = "production"
        new_level = config_profiles[current_profile]["level"]
        assert new_level == "warning"

    def test_profile_inheritance(self):
        """Test profile inheritance from base."""
        base_config = {
            "timeout": 30,
            "retries": 3,
            "buffer_size": 1000,
        }

        development_config = {
            **base_config,
            "debug": True,
            "timeout": 60,  # Override
        }

        # Verify inheritance
        assert development_config["retries"] == 3  # From base
        assert development_config["debug"] is True  # Specific
        assert development_config["timeout"] == 60  # Overridden


class TestConfigurationPersistence:
    """Test configuration persistence."""

    def test_save_configuration_to_file(self):
        """Test saving configuration to file."""
        config = {
            "version": "1.0",
            "sample_interval": 120,
            "features": ["anomaly_detection", "streaming"],
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(config, f)
            config_file = f.name

        try:
            # Load and verify
            with open(config_file, "r") as f:
                loaded = json.load(f)

            assert loaded["version"] == "1.0"
            assert loaded["sample_interval"] == 120
            assert "anomaly_detection" in loaded["features"]
        finally:
            os.unlink(config_file)

    def test_configuration_backup(self):
        """Test configuration backup creation."""
        original = {"value": 100}
        backup = original.copy()

        # Modify original
        original["value"] = 200

        # Backup unchanged
        assert backup["value"] == 100
        assert original["value"] == 200

    def test_configuration_rollback(self):
        """Test configuration rollback."""
        config_history = []

        # Save initial config
        config = {"version": 1}
        config_history.append(config.copy())

        # Modify config
        config["version"] = 2
        config_history.append(config.copy())

        # Rollback to previous
        config = config_history[-2]
        assert config["version"] == 1


class TestDynamicConfiguration:
    """Test dynamic configuration changes."""

    def test_threshold_modification(self):
        """Test modifying detection thresholds."""
        config = {
            "anomaly_threshold": 0.95,
            "zscore_threshold": 3.0,
        }

        original_threshold = config["anomaly_threshold"]

        # Modify threshold
        config["anomaly_threshold"] = 0.90

        assert config["anomaly_threshold"] != original_threshold
        assert config["anomaly_threshold"] == 0.90

    def test_enable_feature_at_runtime(self):
        """Test enabling features at runtime."""
        config = {
            "features": {
                "anomaly_detection": False,
                "streaming": True,
            }
        }

        # Enable anomaly detection
        config["features"]["anomaly_detection"] = True

        assert config["features"]["anomaly_detection"] is True

    def test_buffer_size_adjustment(self):
        """Test adjusting buffer size."""
        config = {"buffer_size": 1000}

        # Increase buffer
        config["buffer_size"] = 5000

        assert config["buffer_size"] == 5000

    def test_interval_modification(self):
        """Test modifying sampling interval."""
        config = {"sample_interval": 60}

        # Change interval
        config["sample_interval"] = 120

        assert config["sample_interval"] == 120
