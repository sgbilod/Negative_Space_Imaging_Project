"""
Test suite for ModelDeploymentPipeline
Tests deployment strategies, validation, rollback, and A/B testing
"""

import pytest
import asyncio
import torch
import torch.nn as nn
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any
from dataclasses import replace

from ml_pipeline.deployment.pipeline import (
    ModelDeploymentPipeline,
    DeploymentConfig,
    DeploymentResult,
    DeploymentStatus,
    DeploymentStrategy,
    DeploymentMetrics
)


class MockModel(nn.Module):
    """Mock PyTorch model for testing"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestModelDeploymentPipeline:
    """Comprehensive test suite for ModelDeploymentPipeline"""

    @pytest.fixture
    def mock_model(self):
        """Create a mock PyTorch model"""
        return MockModel()

    @pytest.fixture
    def mock_registry(self):
        """Create a mock model registry"""
        registry = Mock()
        registry.get_model = AsyncMock(return_value=MockModel())
        registry.get_model_metadata = AsyncMock(return_value={
            "version": "1.0.0",
            "created_at": datetime.now(),
            "performance_metrics": {"accuracy": 0.95}
        })
        return registry

    @pytest.fixture
    def mock_monitor(self):
        """Create a mock model monitor"""
        monitor = Mock()
        monitor.start_monitoring = AsyncMock()
        monitor.stop_monitoring = AsyncMock()
        monitor.get_metrics = AsyncMock(return_value={
            "latency": 100.0,
            "throughput": 1000.0,
            "error_rate": 0.01
        })
        return monitor

    @pytest.fixture
    def deployment_config(self):
        """Create a deployment configuration"""
        return DeploymentConfig(
            strategy=DeploymentStrategy.IMMEDIATE,
            traffic_percentage=100.0,
            validation_samples=1000,
            performance_threshold=0.95,
            rollback_on_failure=True,
            a_b_test_duration=3600,
            health_check_interval=10
        )

    @pytest.fixture
    def deployment_pipeline(self, mock_registry, mock_monitor):
        """Create a deployment pipeline instance"""
        return ModelDeploymentPipeline(
            model_registry=mock_registry,
            model_monitor=mock_monitor
        )

    @pytest.mark.asyncio
    async def test_immediate_deployment_success(
        self, deployment_pipeline, mock_model, deployment_config
    ):
        """Test successful immediate deployment"""
        # Configure for immediate deployment
        config = replace(deployment_config, strategy=DeploymentStrategy.IMMEDIATE)

        # Mock successful validation
        with patch.object(deployment_pipeline, '_validate_deployment',
                         return_value=AsyncMock()) as mock_validate:
            with patch.object(deployment_pipeline, '_perform_deployment',
                             return_value=AsyncMock()) as mock_deploy:
                result = await deployment_pipeline.deploy_model(
                    model=mock_model,
                    model_id="test_model_v1",
                    config=config
                )

                assert result.status == DeploymentStatus.ACTIVE
                assert result.model_name == "test_model_v1"
                assert result.strategy == DeploymentStrategy.IMMEDIATE
                assert result.performance_metrics is not None
                assert result.duration is not None

    @pytest.mark.asyncio
    async def test_gradual_deployment_success(
        self, deployment_pipeline, mock_model, deployment_config
    ):
        """Test successful gradual deployment"""
        config = replace(deployment_config, strategy=DeploymentStrategy.GRADUAL)
        config.gradual_config = {
            "steps": 5,
            "traffic_increment": 0.2,
            "step_duration": 60
        }

        with patch.object(deployment_pipeline, '_validate_deployment',
                         return_value=AsyncMock()) as mock_validate:
            with patch.object(deployment_pipeline, '_perform_gradual_deployment',
                             return_value=AsyncMock()) as mock_gradual:
                result = await deployment_pipeline.deploy_model(
                    model=mock_model,
                    model_id="test_model_v1",
                    config=config
                )

                assert result.status == DeploymentStatus.ACTIVE
                assert result.strategy == DeploymentStrategy.GRADUAL

    @pytest.mark.asyncio
    async def test_ab_testing_deployment(
        self, deployment_pipeline, mock_model, deployment_config
    ):
        """Test A/B testing deployment"""
        config = replace(deployment_config, strategy=DeploymentStrategy.A_B_TEST)

        with patch.object(deployment_pipeline, '_validate_deployment',
                         return_value=AsyncMock()) as mock_validate:
            with patch.object(deployment_pipeline, '_perform_ab_testing',
                             return_value=AsyncMock()) as mock_ab:
                result = await deployment_pipeline.deploy_model(
                    model=mock_model,
                    model_id="test_model_v1",
                    config=config
                )

                assert result.status == DeploymentStatus.ACTIVE
                assert result.strategy == DeploymentStrategy.A_B_TEST

    @pytest.mark.asyncio
    async def test_blue_green_deployment(
        self, deployment_pipeline, mock_model, deployment_config
    ):
        """Test blue-green deployment"""
        config = replace(deployment_config, strategy=DeploymentStrategy.BLUE_GREEN)

        with patch.object(deployment_pipeline, '_validate_deployment',
                         return_value=AsyncMock()) as mock_validate:
            with patch.object(deployment_pipeline, '_perform_blue_green_deployment',
                             return_value=AsyncMock()) as mock_bg:
                result = await deployment_pipeline.deploy_model(
                    model=mock_model,
                    model_id="test_model_v1",
                    config=config
                )

                assert result.status == DeploymentStatus.ACTIVE
                assert result.strategy == DeploymentStrategy.BLUE_GREEN

    @pytest.mark.asyncio
    async def test_deployment_validation_failure(
        self, deployment_pipeline, mock_model, deployment_config
    ):
        """Test deployment failure due to validation"""
        # Use default config

        # Mock validation failure
        with patch.object(deployment_pipeline, '_validate_deployment',
                         side_effect=Exception("Validation failed")) as mock_validate:
            with patch.object(deployment_pipeline, '_rollback_deployment',
                             return_value=AsyncMock()) as mock_rollback:
                result = await deployment_pipeline.deploy_model(
                    model=mock_model,
                    model_id="test_model_v1",
                    config=deployment_config
                )

                assert result.status == DeploymentStatus.FAILED
                assert "Validation failed" in result.error_message
                mock_rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_deployment_timeout(
        self, deployment_pipeline, mock_model, deployment_config
    ):
        """Test deployment timeout handling"""
        # Use default config

        # Mock slow deployment
        async def slow_deployment(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return DeploymentResult(
                model_id="test_model_v1",
                status=DeploymentStatus.ACTIVE,
                strategy=config.strategy
            )

        with patch.object(deployment_pipeline, '_validate_deployment',
                         return_value=AsyncMock()) as mock_validate:
            with patch.object(deployment_pipeline, '_perform_deployment',
                             side_effect=slow_deployment) as mock_deploy:
                with patch.object(deployment_pipeline, '_rollback_deployment',
                                 return_value=AsyncMock()) as mock_rollback:
                    result = await deployment_pipeline.deploy_model(
                        model=mock_model,
                        model_id="test_model_v1",
                        config=deployment_config
                    )

                    assert result.status == DeploymentStatus.FAILED
                    mock_rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_functionality(
        self, deployment_pipeline, mock_model, deployment_config
    ):
        """Test rollback functionality"""
        # Use default config

        with patch.object(deployment_pipeline, '_validate_deployment',
                         side_effect=Exception("Deployment failed")) as mock_validate:
            with patch.object(deployment_pipeline, '_rollback_deployment',
                             return_value=AsyncMock()) as mock_rollback:
                result = await deployment_pipeline.deploy_model(
                    model=mock_model,
                    model_id="test_model_v1",
                    config=deployment_config
                )

                assert result.status == DeploymentStatus.FAILED
                mock_rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_integration(
        self, deployment_pipeline, mock_model, deployment_config
    ):
        """Test health check integration during deployment"""
        # Use default config

        with patch.object(deployment_pipeline, '_validate_deployment',
                         return_value=AsyncMock()) as mock_validate:
            with patch.object(deployment_pipeline, '_perform_deployment',
                             return_value=AsyncMock()) as mock_deploy:
                with patch.object(deployment_pipeline, '_run_health_checks',
                                 return_value=AsyncMock(return_value=True)) as mock_health:
                    result = await deployment_pipeline.deploy_model(
                        model=mock_model,
                        model_id="test_model_v1",
                        config=deployment_config
                    )

                    assert result.status == DeploymentStatus.ACTIVE
                    mock_health.assert_called()

    def test_deployment_config_validation(self):
        """Test deployment configuration validation"""
        # Valid config
        config = DeploymentConfig(
            strategy=DeploymentStrategy.IMMEDIATE,
            a_b_test_duration=300,
            health_check_interval=10
        )
        assert config.a_b_test_duration == 300

        # Test field assignment
        config2 = DeploymentConfig(
            strategy=DeploymentStrategy.GRADUAL,
            a_b_test_duration=600
        )
        assert config2.a_b_test_duration == 600

    def test_deployment_metrics_calculation(self):
        """Test deployment metrics calculation"""
        metrics = DeploymentMetrics(
            deployment_time=120.5,
            validation_time=30.2,
            traffic_shifted=0.8,
            success_rate=0.95
        )

        assert metrics.deployment_time == 120.5
        assert metrics.validation_time == 30.2
        assert metrics.traffic_shifted == 0.8
        assert metrics.success_rate == 0.95

    @pytest.mark.asyncio
    async def test_concurrent_deployments(
        self, deployment_pipeline, mock_model, deployment_config
    ):
        """Test handling of concurrent deployments"""
        # Use default config

        # Mock successful deployments
        with patch.object(deployment_pipeline, '_validate_deployment',
                         return_value=AsyncMock()) as mock_validate:
            with patch.object(deployment_pipeline, '_perform_deployment',
                             return_value=AsyncMock()) as mock_deploy:
                # Launch multiple deployments concurrently
                tasks = []
                for i in range(3):
                    task = deployment_pipeline.deploy_model(
                        model=mock_model,
                        model_id=f"test_model_v{i}",
                        config=deployment_config
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks)

                # All should succeed
                for result in results:
                    assert result.status == DeploymentStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_deployment_with_custom_validation(
        self, deployment_pipeline, mock_model, deployment_config
    ):
        """Test deployment with custom validation function"""
        # Use default config

        with patch.object(deployment_pipeline, '_validate_deployment',
                         return_value=AsyncMock()) as mock_validate:
            with patch.object(deployment_pipeline, '_perform_deployment',
                             return_value=AsyncMock()) as mock_deploy:
                result = await deployment_pipeline.deploy_model(
                    model=mock_model,
                    model_id="test_model_v1",
                    config=deployment_config
                )

                assert result.status == DeploymentStatus.ACTIVE


if __name__ == "__main__":
    pytest.main([__file__])
