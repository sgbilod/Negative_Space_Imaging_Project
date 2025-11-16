#!/usr/bin/env python
"""
Unit tests for Advanced Acquisition Service
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

Tests cover:
- Service initialization and connector registration
- Local file acquisition
- Simulation connector patterns
- Batch acquisition
- Error handling and validation
- Metadata completeness
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from acquisition_service import (
    AcquisitionService,
    LocalFileConnector,
    SimulationConnector,
    ConnectorNotFoundError,
    AcquisitionError,
    SourceConnector,
    AcquisitionResult,
    acquire_sync
)


class TestAcquisitionService:
    """Test suite for AcquisitionService."""

    def test_service_initialization(self):
        """Test service initializes with default connectors."""
        service = AcquisitionService()

        assert "local" in service.connectors
        assert "simulation" in service.connectors
        assert isinstance(service.connectors["local"], LocalFileConnector)
        assert isinstance(service.connectors["simulation"], SimulationConnector)

    def test_connector_registration(self):
        """Test custom connector registration."""
        service = AcquisitionService()

        # Create mock connector
        class MockConnector(SourceConnector):
            def __init__(self):
                super().__init__("mock")

            async def connect(self, **config):
                self.connected = True

            async def acquire(self, target, **params):
                return AcquisitionResult(
                    data=b"mock_data",
                    metadata=None,
                    success=True
                )

            async def disconnect(self):
                self.connected = False

        mock = MockConnector()
        service.register_connector("mock", mock)

        assert "mock" in service.connectors
        assert service.connectors["mock"] == mock

    def test_connector_unregistration(self):
        """Test connector removal."""
        service = AcquisitionService()

        assert "local" in service.connectors
        service.unregister_connector("local")
        assert "local" not in service.connectors

    @pytest.mark.asyncio
    async def test_local_file_acquisition(self):
        """Test acquiring from local file."""
        service = AcquisitionService()

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            test_data = b"test_image_data_12345"
            f.write(test_data)
            temp_path = f.name

        try:
            # Acquire file
            result = await service.acquire(f"local://{temp_path}")

            assert result.success
            assert result.data == test_data
            assert result.metadata.source.type == "local_file"
            assert result.metadata.image.size_bytes == len(test_data)
            assert result.metadata.integrity.verified
            assert result.metadata.timing.acquisition_ms > 0

        finally:
            # Cleanup
            Path(temp_path).unlink()
            await service.close()

    @pytest.mark.asyncio
    async def test_local_file_not_found(self):
        """Test error handling for missing file."""
        service = AcquisitionService()

        result = await service.acquire("local:///nonexistent/file.raw")

        assert not result.success
        assert result.error is not None
        assert "not found" in result.error.lower()

        await service.close()

    @pytest.mark.asyncio
    async def test_simulation_test_pattern(self):
        """Test simulation connector test pattern."""
        service = AcquisitionService()

        result = await service.acquire(
            "simulation://test_pattern",
            width=64,
            height=64
        )

        assert result.success
        assert len(result.data) == 64 * 64  # 8-bit grayscale
        assert result.metadata.source.type == "simulation"
        assert result.metadata.image.dimensions == (64, 64)
        assert result.metadata.image.bit_depth == 8
        assert result.metadata.integrity.verified

        await service.close()

    @pytest.mark.asyncio
    async def test_simulation_patterns(self):
        """Test all simulation patterns."""
        service = AcquisitionService()

        patterns = ["test_pattern", "noise", "gradient", "checkerboard"]

        for pattern in patterns:
            result = await service.acquire(
                f"simulation://{pattern}",
                width=32,
                height=32
            )

            assert result.success, f"Pattern {pattern} failed"
            assert len(result.data) == 32 * 32
            assert result.metadata.source.identifier == pattern

        await service.close()

    @pytest.mark.asyncio
    async def test_batch_acquisition(self):
        """Test concurrent batch acquisition."""
        service = AcquisitionService()

        sources = [
            "simulation://test_pattern",
            "simulation://noise",
            "simulation://gradient"
        ]

        results = await service.acquire_batch(sources, width=32, height=32)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(len(r.data) == 32 * 32 for r in results)

        # Check unique acquisition IDs
        ids = [r.metadata.acquisition_id for r in results]
        assert len(set(ids)) == 3

        await service.close()

    @pytest.mark.asyncio
    async def test_batch_with_failures(self):
        """Test batch acquisition with some failures."""
        service = AcquisitionService()

        sources = [
            "simulation://test_pattern",
            "local:///nonexistent.raw",
            "simulation://noise"
        ]

        results = await service.acquire_batch(sources, width=32, height=32)

        assert len(results) == 3
        assert results[0].success  # simulation succeeds
        assert not results[1].success  # file not found fails
        assert results[2].success  # simulation succeeds

        await service.close()

    @pytest.mark.asyncio
    async def test_uri_parsing(self):
        """Test URI parsing with different formats."""
        service = AcquisitionService()

        # With protocol
        connector, target = service._parse_uri("simulation://test")
        assert connector == "simulation"
        assert target == "test"

        # Without protocol (defaults to local)
        connector, target = service._parse_uri("/path/to/file.raw")
        assert connector == "local"
        assert target == "/path/to/file.raw"

    @pytest.mark.asyncio
    async def test_connector_not_found(self):
        """Test error when connector doesn't exist."""
        service = AcquisitionService()

        with pytest.raises(ConnectorNotFoundError):
            await service.acquire("unknown_connector://target")

        await service.close()

    @pytest.mark.asyncio
    async def test_metadata_completeness(self):
        """Test that all metadata fields are populated."""
        service = AcquisitionService()

        result = await service.acquire("simulation://test_pattern")

        meta = result.metadata

        # Source metadata
        assert meta.source.type is not None
        assert meta.source.identifier is not None
        assert meta.source.verified is not None

        # Image metadata
        assert meta.image.format is not None
        assert meta.image.dimensions is not None
        assert meta.image.bit_depth is not None
        assert meta.image.color_space is not None
        assert meta.image.size_bytes is not None

        # Provenance metadata
        assert meta.provenance.captured_at is not None
        assert meta.provenance.instrument is not None

        # Integrity metadata
        assert meta.integrity.checksum is not None
        assert meta.integrity.algorithm is not None
        assert meta.integrity.verified is not None

        # Timing metadata
        assert meta.timing.acquisition_ms is not None
        assert meta.timing.total_ms is not None

        await service.close()

    @pytest.mark.asyncio
    async def test_checksum_verification(self):
        """Test checksum computation and verification."""
        service = AcquisitionService()

        result = await service.acquire("simulation://test_pattern", width=32, height=32)

        checksum = result.metadata.integrity.checksum

        # Checksum format: algorithm:digest
        assert ":" in checksum
        algo, digest = checksum.split(":", 1)

        assert algo.lower() == "sha256"
        assert len(digest) == 64  # SHA256 hex digest length

        await service.close()

    @pytest.mark.asyncio
    async def test_timing_measurement(self):
        """Test acquisition timing measurement."""
        service = AcquisitionService()

        result = await service.acquire("simulation://test_pattern")

        assert result.metadata.timing.acquisition_ms > 0
        assert result.metadata.timing.total_ms >= result.metadata.timing.acquisition_ms

        await service.close()

    @pytest.mark.asyncio
    async def test_custom_tags(self):
        """Test custom metadata tags."""
        service = AcquisitionService()

        result = await service.acquire("simulation://test_pattern")

        # Custom tags should exist and be modifiable
        assert hasattr(result.metadata, 'custom_tags')
        assert isinstance(result.metadata.custom_tags, dict)

        # Add custom tag
        result.metadata.custom_tags['experiment_id'] = 'exp-001'
        assert result.metadata.custom_tags['experiment_id'] == 'exp-001'

        await service.close()

    @pytest.mark.asyncio
    async def test_metadata_serialization(self):
        """Test metadata can be serialized to dict."""
        service = AcquisitionService()

        result = await service.acquire("simulation://test_pattern")

        meta_dict = result.metadata.to_dict()

        assert isinstance(meta_dict, dict)
        assert 'acquisition_id' in meta_dict
        assert 'source' in meta_dict
        assert 'image' in meta_dict
        assert 'provenance' in meta_dict
        assert 'integrity' in meta_dict
        assert 'timing' in meta_dict

        await service.close()

    def test_synchronous_acquisition(self):
        """Test synchronous wrapper function."""
        result = acquire_sync("simulation://test_pattern", width=16, height=16)

        assert result.success
        assert len(result.data) == 16 * 16


class TestLocalFileConnector:
    """Test suite for LocalFileConnector."""

    @pytest.mark.asyncio
    async def test_connector_lifecycle(self):
        """Test connector connect/disconnect lifecycle."""
        connector = LocalFileConnector()

        assert not connector.connected

        await connector.connect()
        assert connector.connected

        await connector.disconnect()
        assert not connector.connected


class TestSimulationConnector:
    """Test suite for SimulationConnector."""

    @pytest.mark.asyncio
    async def test_connector_lifecycle(self):
        """Test connector connect/disconnect lifecycle."""
        connector = SimulationConnector()

        assert not connector.connected

        await connector.connect()
        assert connector.connected

        await connector.disconnect()
        assert not connector.connected

    @pytest.mark.asyncio
    async def test_pattern_generation(self):
        """Test different pattern generation methods."""
        connector = SimulationConnector()
        await connector.connect()

        patterns = {
            "test_pattern": 256 * 256,
            "noise": 128 * 128,
            "gradient": 64 * 64,
            "checkerboard": 32 * 32
        }

        for pattern, expected_size in patterns.items():
            width = int(expected_size ** 0.5)
            result = await connector.acquire(pattern, width=width, height=width)

            assert result.success
            assert len(result.data) == expected_size

        await connector.disconnect()


@pytest.mark.asyncio
async def test_concurrent_performance():
    """Test concurrent acquisition performance."""
    import time

    service = AcquisitionService()

    # Prepare 10 sources
    sources = [f"simulation://test_pattern" for _ in range(10)]

    # Measure batch time
    start = time.time()
    results = await service.acquire_batch(sources, width=64, height=64)
    elapsed = time.time() - start

    assert all(r.success for r in results)
    assert len(results) == 10

    # Should be faster than sequential (10 * 10ms = 100ms)
    # Concurrent should be ~10-50ms depending on system
    print(f"\nBatch acquisition of 10 images: {elapsed*1000:.2f}ms")

    await service.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
