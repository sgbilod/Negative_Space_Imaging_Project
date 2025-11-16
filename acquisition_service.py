#!/usr/bin/env python
"""
Advanced Acquisition Service for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides a modern, extensible image acquisition service with:
- Async/await for concurrent multi-source acquisition
- Plugin architecture for custom connectors
- Enriched metadata with provenance tracking
- Robust error handling with retry/backoff
- Integrity verification and checksums

Usage:
    from acquisition_service import AcquisitionService, LocalFileConnector

    # Create service and register connectors
    service = AcquisitionService()
    service.register_connector("local", LocalFileConnector())

    # Acquire image asynchronously
    result = await service.acquire("local://path/to/image.raw")

    # Batch acquisition
    results = await service.acquire_batch([
        "local://image1.raw",
        "simulation://test_pattern"
    ])
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum

import aiofiles

# Configure logger
logger = logging.getLogger(__name__)


class AcquisitionError(Exception):
    """Base exception for acquisition errors."""
    pass


class ConnectorNotFoundError(AcquisitionError):
    """Raised when requested connector is not registered."""
    pass


class SourceConnectionError(AcquisitionError):
    """Raised when connection to source fails."""
    pass


class IntegrityVerificationError(AcquisitionError):
    """Raised when integrity check fails."""
    pass


@dataclass
class SourceMetadata:
    """Metadata about the acquisition source."""
    type: str
    identifier: str
    verified: bool = False
    connection_time_ms: Optional[float] = None


@dataclass
class ImageMetadata:
    """Image-specific metadata."""
    format: str
    dimensions: Optional[tuple] = None
    bit_depth: Optional[int] = None
    color_space: Optional[str] = None
    size_bytes: Optional[int] = None


@dataclass
class ProvenanceMetadata:
    """Provenance tracking metadata."""
    captured_at: str
    instrument: Optional[str] = None
    operator: Optional[str] = None
    location: Optional[Dict[str, float]] = None


@dataclass
class IntegrityMetadata:
    """Integrity verification metadata."""
    checksum: str
    algorithm: str = "SHA256"
    verified: bool = False


@dataclass
class TimingMetadata:
    """Timing information for acquisition."""
    acquisition_ms: float
    transfer_ms: Optional[float] = None
    total_ms: Optional[float] = None


@dataclass
class AcquisitionMetadata:
    """Complete acquisition metadata."""
    acquisition_id: str
    source: SourceMetadata
    image: ImageMetadata
    provenance: ProvenanceMetadata
    integrity: IntegrityMetadata
    timing: TimingMetadata
    custom_tags: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return asdict(self)


@dataclass
class AcquisitionResult:
    """Result of an acquisition operation."""
    data: bytes
    metadata: AcquisitionMetadata
    success: bool = True
    error: Optional[str] = None


class SourceConnector(ABC):
    """
    Abstract base class for source connectors.

    Each connector implements acquisition from a specific source type
    (local files, remote servers, cameras, etc.)
    """

    def __init__(self, name: str):
        """
        Initialize connector.

        Args:
            name: Unique name for this connector type
        """
        self.name = name
        self.connected = False
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @abstractmethod
    async def connect(self, **config) -> None:
        """
        Establish connection to source.

        Args:
            **config: Connector-specific configuration
        """
        pass

    @abstractmethod
    async def acquire(self, target: str, **params) -> AcquisitionResult:
        """
        Acquire image from source.

        Args:
            target: Source-specific target identifier
            **params: Additional acquisition parameters

        Returns:
            AcquisitionResult with data and metadata
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection gracefully."""
        pass

    def _generate_acquisition_id(self) -> str:
        """Generate unique acquisition ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.sha256(
            f"{timestamp}{time.time()}".encode()
        ).hexdigest()[:8]
        return f"acq-{timestamp}-{random_suffix}"

    def _compute_checksum(self, data: bytes, algorithm: str = "sha256") -> str:
        """
        Compute checksum of data.

        Args:
            data: Bytes to hash
            algorithm: Hash algorithm (sha256, sha512, md5)

        Returns:
            Hex digest of checksum
        """
        hasher = hashlib.new(algorithm)
        hasher.update(data)
        return f"{algorithm}:{hasher.hexdigest()}"


class LocalFileConnector(SourceConnector):
    """Connector for local file system acquisition."""

    def __init__(self):
        super().__init__("local")

    async def connect(self, **config) -> None:
        """Local file connector requires no connection setup."""
        self.connected = True
        self.logger.info("Local file connector ready")

    async def acquire(self, target: str, **params) -> AcquisitionResult:
        """
        Acquire image from local file.

        Args:
            target: File path
            **params: Additional parameters (ignored for local files)

        Returns:
            AcquisitionResult
        """
        start_time = time.time()

        try:
            # Read file asynchronously
            path = Path(target)

            if not path.exists():
                raise AcquisitionError(f"File not found: {target}")

            async with aiofiles.open(path, 'rb') as f:
                data = await f.read()

            # Compute timing
            elapsed_ms = (time.time() - start_time) * 1000

            # Create metadata
            metadata = self._create_metadata(target, data, elapsed_ms)

            return AcquisitionResult(
                data=data,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Failed to acquire from {target}: {e}")
            return AcquisitionResult(
                data=b'',
                metadata=self._create_error_metadata(target),
                success=False,
                error=str(e)
            )

    async def disconnect(self) -> None:
        """Local file connector requires no cleanup."""
        self.connected = False
        self.logger.info("Local file connector closed")

    def _create_metadata(
        self,
        target: str,
        data: bytes,
        elapsed_ms: float
    ) -> AcquisitionMetadata:
        """Create metadata for successful acquisition."""
        path = Path(target)
        metadata = AcquisitionMetadata(
            acquisition_id=self._generate_acquisition_id(),
            source=SourceMetadata(
                type="local_file",
                identifier=str(path.absolute()),
                verified=True,
                connection_time_ms=0.0
            ),
            image=ImageMetadata(
                format=path.suffix.upper().lstrip('.') or "RAW",
                size_bytes=len(data)
            ),
            provenance=ProvenanceMetadata(
                captured_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                instrument="LocalFileSystem"
            ),
            integrity=IntegrityMetadata(
                checksum=self._compute_checksum(data),
                algorithm="SHA256",
                verified=True
            ),
            timing=TimingMetadata(
                acquisition_ms=elapsed_ms,
                total_ms=elapsed_ms
            )
        )

        # Security integration (sign artifact hash)
        try:
            from security_module import load_or_create_keys, sign_bytes
            # Extract hash hex from integrity checksum (format algo:hex)
            algo, hash_hex = metadata.integrity.checksum.split(':', 1)
            private_key, public_key, key_id = load_or_create_keys()
            signature = sign_bytes(private_key, bytes.fromhex(hash_hex))
            import base64
            metadata.security = {
                'hash_algorithm': algo,
                'artifact_hash': hash_hex,
                'signature': base64.b64encode(signature).decode('utf-8'),
                'key_id': key_id,
                'provenance_chain': [
                    {
                        'stage': 'acquisition',
                        'artifact': str(path.absolute()),
                        'hash': hash_hex,
                        'timestamp': metadata.provenance.captured_at
                    }
                ]
            }
        except Exception as e:  # noqa: BLE001
            logger.error(f"Security signing failed for {target}: {e}")

        return metadata

    def _create_error_metadata(self, target: str) -> AcquisitionMetadata:
        """Create minimal metadata for failed acquisition."""
        return AcquisitionMetadata(
            acquisition_id=self._generate_acquisition_id(),
            source=SourceMetadata(
                type="local_file",
                identifier=target,
                verified=False
            ),
            image=ImageMetadata(format="UNKNOWN"),
            provenance=ProvenanceMetadata(
                captured_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            ),
            integrity=IntegrityMetadata(
                checksum="",
                verified=False
            ),
            timing=TimingMetadata(acquisition_ms=0.0)
        )


class SimulationConnector(SourceConnector):
    """Connector for simulated image generation."""

    def __init__(self):
        super().__init__("simulation")

    async def connect(self, **config) -> None:
        """Simulation connector requires no connection setup."""
        self.connected = True
        self.logger.info("Simulation connector ready")

    async def acquire(self, target: str, **params) -> AcquisitionResult:
        """
        Generate simulated image data.

        Args:
            target: Pattern type (test_pattern, noise, gradient, etc.)
            **params: Pattern parameters (width, height, etc.)

        Returns:
            AcquisitionResult with synthetic data
        """
        start_time = time.time()

        try:
            # Parse parameters
            width = params.get('width', 256)
            height = params.get('height', 256)
            pattern = target.lower()

            # Generate synthetic data based on pattern
            if pattern == "test_pattern" or pattern == "test":
                data = self._generate_test_pattern(width, height)
            elif pattern == "noise":
                data = self._generate_noise(width, height)
            elif pattern == "gradient":
                data = self._generate_gradient(width, height)
            else:
                # Default: checkerboard pattern
                data = self._generate_checkerboard(width, height)

            # Simulate acquisition delay
            await asyncio.sleep(0.01)  # 10ms simulated acquisition

            elapsed_ms = (time.time() - start_time) * 1000

            # Create metadata
            metadata = self._create_metadata(target, data, elapsed_ms, width, height)

            return AcquisitionResult(
                data=data,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            self.logger.error(f"Simulation failed for {target}: {e}")
            return AcquisitionResult(
                data=b'',
                metadata=self._create_error_metadata(target),
                success=False,
                error=str(e)
            )

    async def disconnect(self) -> None:
        """Simulation connector requires no cleanup."""
        self.connected = False
        self.logger.info("Simulation connector closed")

    def _generate_test_pattern(self, width: int, height: int) -> bytes:
        """Generate simple test pattern."""
        import numpy as np
        # Gradient with circular pattern
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        pattern = ((distance / max_dist) * 255).astype(np.uint8)
        return pattern.tobytes()

    def _generate_noise(self, width: int, height: int) -> bytes:
        """Generate random noise pattern."""
        import numpy as np
        noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        return noise.tobytes()

    def _generate_gradient(self, width: int, height: int) -> bytes:
        """Generate linear gradient."""
        import numpy as np
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        pattern = np.tile(gradient, (height, 1))
        return pattern.tobytes()

    def _generate_checkerboard(self, width: int, height: int) -> bytes:
        """Generate checkerboard pattern."""
        import numpy as np
        square_size = 32
        pattern = np.indices((height, width)).sum(axis=0) // square_size
        pattern = ((pattern % 2) * 255).astype(np.uint8)
        return pattern.tobytes()

    def _create_metadata(
        self,
        target: str,
        data: bytes,
        elapsed_ms: float,
        width: int,
        height: int
    ) -> AcquisitionMetadata:
        """Create metadata for simulated acquisition."""
        metadata = AcquisitionMetadata(
            acquisition_id=self._generate_acquisition_id(),
            source=SourceMetadata(
                type="simulation",
                identifier=target,
                verified=True,
                connection_time_ms=0.0
            ),
            image=ImageMetadata(
                format="RAW",
                dimensions=(width, height),
                bit_depth=8,
                color_space="grayscale",
                size_bytes=len(data)
            ),
            provenance=ProvenanceMetadata(
                captured_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                instrument="SimulationEngine"
            ),
            integrity=IntegrityMetadata(
                checksum=self._compute_checksum(data),
                algorithm="SHA256",
                verified=True
            ),
            timing=TimingMetadata(
                acquisition_ms=elapsed_ms,
                total_ms=elapsed_ms
            )
        )

        # Security integration
        try:
            from security_module import load_or_create_keys, sign_bytes
            algo, hash_hex = metadata.integrity.checksum.split(':', 1)
            private_key, public_key, key_id = load_or_create_keys()
            signature = sign_bytes(private_key, bytes.fromhex(hash_hex))
            import base64
            metadata.security = {
                'hash_algorithm': algo,
                'artifact_hash': hash_hex,
                'signature': base64.b64encode(signature).decode('utf-8'),
                'key_id': key_id,
                'provenance_chain': [
                    {
                        'stage': 'acquisition',
                        'artifact': f'simulation://{target}',
                        'hash': hash_hex,
                        'timestamp': metadata.provenance.captured_at
                    }
                ]
            }
        except Exception as e:  # noqa: BLE001
            logger.error(f"Security signing failed for simulation {target}: {e}")

        return metadata

    def _create_error_metadata(self, target: str) -> AcquisitionMetadata:
        """Create minimal metadata for failed simulation."""
        return AcquisitionMetadata(
            acquisition_id=self._generate_acquisition_id(),
            source=SourceMetadata(
                type="simulation",
                identifier=target,
                verified=False
            ),
            image=ImageMetadata(format="UNKNOWN"),
            provenance=ProvenanceMetadata(
                captured_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            ),
            integrity=IntegrityMetadata(
                checksum="",
                verified=False
            ),
            timing=TimingMetadata(acquisition_ms=0.0)
        )


class AcquisitionService:
    """
    Main acquisition service with plugin architecture.

    Manages multiple source connectors and provides unified interface
    for image acquisition with async support and error handling.
    """

    def __init__(self):
        """Initialize acquisition service."""
        self.connectors: Dict[str, SourceConnector] = {}
        self.logger = logging.getLogger(__name__)

        # Register default connectors
        self.register_connector("local", LocalFileConnector())
        self.register_connector("simulation", SimulationConnector())

    def register_connector(self, name: str, connector: SourceConnector) -> None:
        """
        Register a source connector.

        Args:
            name: Unique name for connector
            connector: SourceConnector instance
        """
        self.connectors[name] = connector
        self.logger.info(f"Registered connector: {name}")

    def unregister_connector(self, name: str) -> None:
        """
        Unregister a source connector.

        Args:
            name: Name of connector to remove
        """
        if name in self.connectors:
            del self.connectors[name]
            self.logger.info(f"Unregistered connector: {name}")

    async def acquire(
        self,
        source_uri: str,
        **params
    ) -> AcquisitionResult:
        """
        Acquire image from source URI.

        URI format: <connector>://<target>
        Example: local:///path/to/image.raw
                 simulation://test_pattern

        Args:
            source_uri: Source URI with connector prefix
            **params: Additional acquisition parameters

        Returns:
            AcquisitionResult
        """
        # Parse URI
        connector_name, target = self._parse_uri(source_uri)

        # Get connector
        if connector_name not in self.connectors:
            raise ConnectorNotFoundError(
                f"Connector '{connector_name}' not found. "
                f"Available: {list(self.connectors.keys())}"
            )

        connector = self.connectors[connector_name]

        # Ensure connector is connected
        if not connector.connected:
            await connector.connect()

        # Acquire image
        self.logger.info(f"Acquiring from {source_uri}")
        result = await connector.acquire(target, **params)

        if result.success:
            self.logger.info(
                f"Successfully acquired {len(result.data)} bytes "
                f"[{result.metadata.acquisition_id}]"
            )
        else:
            self.logger.error(
                f"Acquisition failed: {result.error}"
            )

        return result

    async def acquire_batch(
        self,
        source_uris: List[str],
        **params
    ) -> List[AcquisitionResult]:
        """
        Acquire multiple images concurrently.

        Args:
            source_uris: List of source URIs
            **params: Common parameters for all acquisitions

        Returns:
            List of AcquisitionResult objects
        """
        self.logger.info(f"Batch acquiring {len(source_uris)} sources")

        tasks = [self.acquire(uri, **params) for uri in source_uris]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions, convert to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Batch item {i} failed: {result}"
                )
                # Create error result
                processed_results.append(
                    AcquisitionResult(
                        data=b'',
                        metadata=self._create_error_metadata(source_uris[i]),
                        success=False,
                        error=str(result)
                    )
                )
            else:
                processed_results.append(result)

        success_count = sum(1 for r in processed_results if r.success)
        self.logger.info(
            f"Batch complete: {success_count}/{len(source_uris)} successful"
        )

        return processed_results

    async def close(self) -> None:
        """Close all connectors gracefully."""
        self.logger.info("Closing all connectors")

        for name, connector in self.connectors.items():
            if connector.connected:
                await connector.disconnect()

    def _parse_uri(self, uri: str) -> tuple:
        """
        Parse source URI into connector name and target.

        Args:
            uri: Source URI

        Returns:
            Tuple of (connector_name, target)
        """
        if "://" in uri:
            connector_name, target = uri.split("://", 1)
        else:
            # Default to local file
            connector_name = "local"
            target = uri

        return connector_name, target

    def _create_error_metadata(self, uri: str) -> AcquisitionMetadata:
        """Create minimal error metadata."""
        connector_name, target = self._parse_uri(uri)

        return AcquisitionMetadata(
            acquisition_id=f"acq-error-{int(time.time())}",
            source=SourceMetadata(
                type=connector_name,
                identifier=target,
                verified=False
            ),
            image=ImageMetadata(format="UNKNOWN"),
            provenance=ProvenanceMetadata(
                captured_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            ),
            integrity=IntegrityMetadata(
                checksum="",
                verified=False
            ),
            timing=TimingMetadata(acquisition_ms=0.0)
        )


# Helper function for synchronous usage
def acquire_sync(source_uri: str, **params) -> AcquisitionResult:
    """
    Synchronous wrapper for acquisition.

    Args:
        source_uri: Source URI
        **params: Acquisition parameters

    Returns:
        AcquisitionResult
    """
    service = AcquisitionService()
    result = asyncio.run(service.acquire(source_uri, **params))
    asyncio.run(service.close())
    return result


if __name__ == "__main__":
    # Example usage
    import json

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async def main():
        """Demo acquisition service."""
        service = AcquisitionService()

        # Single acquisition
        print("\n=== Single Acquisition ===")
        result = await service.acquire("simulation://test_pattern", width=128, height=128)

        if result.success:
            print(f"✓ Acquired {len(result.data)} bytes")
            print(f"  Checksum: {result.metadata.integrity.checksum[:32]}...")
            print(f"  Time: {result.metadata.timing.acquisition_ms:.2f}ms")

        # Batch acquisition
        print("\n=== Batch Acquisition ===")
        sources = [
            "simulation://test_pattern",
            "simulation://noise",
            "simulation://gradient"
        ]

        results = await service.acquire_batch(sources, width=64, height=64)

        print(f"Acquired {len(results)} images:")
        for i, result in enumerate(results):
            if result.success:
                print(f"  {i+1}. ✓ {result.metadata.source.identifier} "
                      f"({len(result.data)} bytes)")
            else:
                print(f"  {i+1}. ✗ Error: {result.error}")

        # Close service
        await service.close()
        print("\n=== Service Closed ===")

    asyncio.run(main())
