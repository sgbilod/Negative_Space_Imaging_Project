"""
Secure Communication Protocol for Federated Learning
TLS/SSL, model serialization, compression, and fault tolerance.
"""

import logging
import pickle
import json
import gzip
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
import hashlib
import ssl
import socket
import base64
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class SerializedModel:
    """Container for serialized model parameters."""

    timestamp: str
    client_id: str
    round_number: int
    model_parameters: bytes
    checksum: str
    compressed: bool
    compression_ratio: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SecureSerializer:
    """
    Serialization with security and efficiency measures.
    Handles model parameters, compression, and validation.
    """

    def __init__(
        self,
        use_compression: bool = True,
        compression_level: int = 6,
    ):
        """
        Initialize serializer.

        Args:
            use_compression: Enable gzip compression
            compression_level: Compression level (1-9)
        """
        self.use_compression = use_compression
        self.compression_level = compression_level

    def serialize_parameters(
        self,
        parameters: Dict[str, Any],
        client_id: str,
        round_number: int,
    ) -> SerializedModel:
        """
        Serialize model parameters with compression and checksum.

        Args:
            parameters: Model parameters dictionary
            client_id: Client identifier
            round_number: Federation round number

        Returns:
            SerializedModel object
        """
        # Pickle parameters
        pickled = pickle.dumps(parameters)
        original_size = len(pickled)

        # Compress if enabled
        if self.use_compression:
            compressed = gzip.compress(
                pickled,
                compresslevel=self.compression_level,
            )
            compressed_size = len(compressed)
            model_bytes = compressed
            ratio = original_size / (compressed_size + 1e-10)
        else:
            model_bytes = pickled
            compressed_size = original_size
            ratio = 1.0

        # Compute checksum
        checksum = hashlib.sha256(model_bytes).hexdigest()

        serialized = SerializedModel(
            timestamp=datetime.now().isoformat(),
            client_id=client_id,
            round_number=round_number,
            model_parameters=model_bytes,
            checksum=checksum,
            compressed=self.use_compression,
            compression_ratio=ratio,
            metadata={
                "original_size": original_size,
                "compressed_size": compressed_size,
                "num_parameters": len(parameters),
            },
        )

        logger.info(
            f"Parameters serialized: {original_size:,} → {compressed_size:,} bytes "
            f"({ratio:.2f}x compression)"
        )

        return serialized

    def deserialize_parameters(
        self,
        serialized: SerializedModel,
        verify_checksum: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Deserialize model parameters with verification.

        Args:
            serialized: SerializedModel object
            verify_checksum: Verify checksum before deserializing

        Returns:
            Parameters dictionary or None if verification fails
        """
        # Verify checksum
        if verify_checksum:
            computed_checksum = hashlib.sha256(
                serialized.model_parameters
            ).hexdigest()

            if computed_checksum != serialized.checksum:
                logger.error(
                    f"Checksum mismatch: {computed_checksum} != {serialized.checksum}"
                )
                return None

        try:
            # Decompress if needed
            if serialized.compressed:
                decompressed = gzip.decompress(serialized.model_parameters)
            else:
                decompressed = serialized.model_parameters

            # Unpickle parameters
            parameters = pickle.loads(decompressed)

            logger.info(f"Parameters deserialized: {len(parameters)} parameters")
            return parameters

        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            return None

    def quantize_parameters(
        self,
        parameters: Dict[str, Any],
        bits: int = 8,
    ) -> Dict[str, Any]:
        """
        Quantize parameters to reduce bandwidth.

        Args:
            parameters: Model parameters
            bits: Bits per parameter

        Returns:
            Quantized parameters
        """
        import numpy as np

        quantized = {}

        for name, param in parameters.items():
            if isinstance(param, np.ndarray):
                # Quantize to specified bits
                param_min = np.min(param)
                param_max = np.max(param)

                # Map to integer range
                levels = 2**bits - 1
                scaled = (param - param_min) / (param_max - param_min + 1e-10)
                quantized_int = np.round(scaled * levels).astype(np.uint8)

                # Store with scaling info
                quantized[name] = {
                    "quantized": quantized_int,
                    "min": float(param_min),
                    "max": float(param_max),
                    "bits": bits,
                }
            else:
                quantized[name] = param

        return quantized

    def dequantize_parameters(
        self,
        quantized: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Dequantize parameters back to full precision.

        Args:
            quantized: Quantized parameters

        Returns:
            Full precision parameters
        """
        import numpy as np

        dequantized = {}

        for name, param in quantized.items():
            if isinstance(param, dict) and "quantized" in param:
                # Dequantize
                q = param["quantized"]
                param_min = param["min"]
                param_max = param["max"]
                bits = param["bits"]

                levels = 2**bits - 1
                scaled = q.astype(np.float32) / levels
                dequantized[name] = (
                    scaled * (param_max - param_min) + param_min
                )
            else:
                dequantized[name] = param

        return dequantized


class CommunicationProtocol:
    """
    Secure communication protocol with TLS/SSL and reliability.
    """

    def __init__(
        self,
        client_id: str,
        server_address: str,
        server_port: int = 8883,
        use_tls: bool = True,
        retry_attempts: int = 3,
        timeout: int = 30,
    ):
        """
        Initialize communication protocol.

        Args:
            client_id: Client identifier
            server_address: Server hostname/IP
            server_port: Server port
            use_tls: Enable TLS encryption
            retry_attempts: Number of retry attempts
            timeout: Socket timeout in seconds
        """
        self.client_id = client_id
        self.server_address = server_address
        self.server_port = server_port
        self.use_tls = use_tls
        self.retry_attempts = retry_attempts
        self.timeout = timeout

        self.serializer = SecureSerializer()
        self.communication_history: List[Dict] = []

    def create_secure_socket(self) -> Optional[socket.socket]:
        """
        Create TLS/SSL secured socket.

        Args:
            Returns:
            Secure socket or None
        """
        try:
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)

            if self.use_tls:
                # Wrap with TLS
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE  # In production, use proper certificates

                secure_sock = context.wrap_socket(
                    sock,
                    server_hostname=self.server_address,
                )
                return secure_sock

            return sock

        except Exception as e:
            logger.error(f"Failed to create secure socket: {e}")
            return None

    def send_with_retry(
        self,
        data: SerializedModel,
    ) -> Tuple[bool, str]:
        """
        Send data with automatic retry on failure.

        Args:
            data: SerializedModel to send

        Returns:
            (success, message) tuple
        """
        for attempt in range(self.retry_attempts):
            try:
                sock = self.create_secure_socket()
                if not sock:
                    continue

                # Connect
                sock.connect((self.server_address, self.server_port))

                # Send data
                sock.sendall(data.model_parameters)

                sock.close()

                # Log successful send
                self._log_communication(
                    direction="send",
                    success=True,
                    data_size=len(data.model_parameters),
                    attempt=attempt + 1,
                )

                return True, "Data sent successfully"

            except Exception as e:
                logger.warning(
                    f"Send attempt {attempt + 1}/{self.retry_attempts} failed: {e}"
                )

                if attempt == self.retry_attempts - 1:
                    self._log_communication(
                        direction="send",
                        success=False,
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    return False, f"Send failed after {self.retry_attempts} attempts"

        return False, "Send failed"

    def receive_with_retry(
        self,
        buffer_size: int = 65536,
    ) -> Tuple[bool, Optional[bytes]]:
        """
        Receive data with automatic retry on failure.

        Args:
            buffer_size: Size of receive buffer

        Returns:
            (success, data) tuple
        """
        for attempt in range(self.retry_attempts):
            try:
                sock = self.create_secure_socket()
                if not sock:
                    continue

                # Connect (server initiates for receive)
                sock.bind(("0.0.0.0", self.server_port))
                sock.listen(1)

                conn, addr = sock.accept()
                data = conn.recv(buffer_size)
                conn.close()
                sock.close()

                # Log successful receive
                self._log_communication(
                    direction="receive",
                    success=True,
                    data_size=len(data),
                    attempt=attempt + 1,
                )

                return True, data

            except Exception as e:
                logger.warning(
                    f"Receive attempt {attempt + 1}/{self.retry_attempts} failed: {e}"
                )

                if attempt == self.retry_attempts - 1:
                    self._log_communication(
                        direction="receive",
                        success=False,
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    return False, None

        return False, None

    def _log_communication(
        self,
        direction: str,
        success: bool,
        data_size: int = 0,
        error: str = "",
        attempt: int = 1,
    ):
        """Log communication event."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "direction": direction,
            "success": success,
            "data_size": data_size,
            "error": error,
            "attempt": attempt,
            "client_id": self.client_id,
        }

        self.communication_history.append(log_entry)

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        if not self.communication_history:
            return {}

        total = len(self.communication_history)
        successful = sum(1 for log in self.communication_history if log["success"])
        total_bytes = sum(log.get("data_size", 0) for log in self.communication_history)

        return {
            "total_operations": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0,
            "total_bytes_transferred": total_bytes,
            "avg_bytes_per_operation": total_bytes / total if total > 0 else 0,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    serializer = SecureSerializer(use_compression=True)

    # Create dummy parameters
    parameters = {
        "layer1": [1.0, 2.0, 3.0],
        "layer2": [4.0, 5.0, 6.0],
    }

    # Serialize
    serialized = serializer.serialize_parameters(
        parameters,
        client_id="client_001",
        round_number=1,
    )

    print(f"Serialized size: {len(serialized.model_parameters)} bytes")

    # Deserialize
    deserialized = serializer.deserialize_parameters(serialized)
    print(f"Deserialized parameters: {deserialized}")
