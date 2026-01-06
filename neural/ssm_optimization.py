"""
SSM Production Optimization

Techniques for production deployment:
- TorchScript export for C++ serving
- ONNX export with operator fusion
- INT8 quantization (dynamic and static)
- Pruning strategies (magnitude, structured)
- Batch optimization for variable-length sequences
"""

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from pathlib import Path

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)


class SSMOptimizer:
    """Optimization toolkit for SSM models."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.original_size = self._get_model_size()

    @staticmethod
    def _get_model_size(model: Optional[nn.Module] = None) -> float:
        """Get model size in MB."""
        if model is None:
            return 0
        total_params = sum(p.numel() for p in model.parameters())
        return total_params * 4 / (1024 * 1024)  # Assuming float32

    # ==================== TORCHSCRIPT EXPORT ====================

    def export_torchscript_trace(
        self,
        output_path: str,
        example_input: torch.Tensor,
    ) -> str:
        """
        Export model to TorchScript via tracing.

        Suitable for models with data-dependent control flow.

        Args:
            output_path: Path to save traced model
            example_input: Example input tensor for tracing

        Returns:
            Path to saved model
        """
        try:
            # Trace the model
            traced_model = torch.jit.trace(
                self.model,
                example_input,
                check_trace=True,
            )

            # Save traced model
            traced_model.save(output_path)

            traced_size = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"✓ Exported TorchScript (trace): {output_path} ({traced_size:.2f}MB)")

            return output_path

        except Exception as e:
            logger.error(f"TorchScript trace export failed: {e}")
            raise

    def export_torchscript_script(self, output_path: str) -> str:
        """
        Export model to TorchScript via scripting (if model is compatible).

        Preserves control flow but requires model to be script-compatible.

        Args:
            output_path: Path to save scripted model

        Returns:
            Path to saved model
        """
        try:
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(output_path)

            scripted_size = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"✓ Exported TorchScript (script): {output_path} ({scripted_size:.2f}MB)")

            return output_path

        except Exception as e:
            logger.warning(f"TorchScript script export not supported: {e}")
            return None

    # ==================== ONNX EXPORT ====================

    def export_onnx(
        self,
        output_path: str,
        example_input: torch.Tensor,
        opset_version: int = 14,
        optimize: bool = True,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            example_input: Example input tensor
            opset_version: ONNX operator set version
            optimize: Apply operator fusion optimization

        Returns:
            Path to saved ONNX model
        """
        if not ONNX_AVAILABLE:
            logger.warning("ONNX not available, skipping ONNX export")
            return None

        try:
            # Export to ONNX
            torch.onnx.export(
                self.model,
                example_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False,
                dynamic_axes={
                    "input": {0: "batch_size", 1: "seq_len"},
                    "output": {0: "batch_size"},
                },
            )

            logger.info(f"✓ Exported ONNX model: {output_path}")

            # Optimize ONNX model if requested
            if optimize:
                self._optimize_onnx(output_path)

            onnx_size = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"  ONNX model size: {onnx_size:.2f}MB")

            return output_path

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise

    @staticmethod
    def _optimize_onnx(model_path: str):
        """Apply operator fusion optimizations to ONNX model."""
        if not ONNX_AVAILABLE:
            return

        try:
            from onnxruntime.transformers import optimizer

            opt_model = optimizer.optimize_model(
                model_path,
                model_type="bert",  # Use bert settings for transformer-like models
                num_heads=None,
                hidden_size=None,
                optimization_options=None,
            )

            opt_model.save_model_to_file(model_path)
            logger.info(f"✓ Applied ONNX optimizations to {model_path}")

        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")

    # ==================== QUANTIZATION ====================

    def quantize_dynamic(self, output_path: str) -> nn.Module:
        """
        Apply dynamic quantization (INT8).

        Quantizes weights to INT8 at runtime.

        Args:
            output_path: Path to save quantized model

        Returns:
            Quantized model
        """
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                qconfig_spec={torch.nn.Linear},  # Quantize linear layers
                dtype=torch.qint8,
            )

            torch.save(quantized_model.state_dict(), output_path)

            quantized_size = self._get_model_size(quantized_model)
            reduction = (self.original_size - quantized_size) / self.original_size * 100

            logger.info(
                f"✓ Dynamic quantization (INT8): {self.original_size:.2f}MB → {quantized_size:.2f}MB "
                f"({reduction:.1f}% reduction)"
            )

            return quantized_model

        except Exception as e:
            logger.error(f"Dynamic quantization failed: {e}")
            raise

    def quantize_static(
        self,
        output_path: str,
        calibration_data: torch.Tensor,
        qconfig: Optional[torch.quantization.QConfig] = None,
    ) -> nn.Module:
        """
        Apply static quantization (INT8) with calibration.

        Args:
            output_path: Path to save quantized model
            calibration_data: Calibration dataset
            qconfig: Quantization config (default: symmetric INT8)

        Returns:
            Quantized model
        """
        try:
            # Prepare model for quantization
            if qconfig is None:
                qconfig = torch.quantization.get_default_qconfig("fbgemm")

            model = self.model.eval()
            torch.quantization.prepare_qat(model, qconfig, inplace=True)

            # Calibrate with data
            with torch.no_grad():
                for batch in calibration_data:
                    if isinstance(batch, (list, tuple)):
                        model(*batch)
                    else:
                        model(batch)

            # Convert to quantized model
            torch.quantization.convert(model, inplace=True)

            torch.save(model.state_dict(), output_path)

            quantized_size = self._get_model_size(model)
            reduction = (self.original_size - quantized_size) / self.original_size * 100

            logger.info(
                f"✓ Static quantization (INT8): {self.original_size:.2f}MB → {quantized_size:.2f}MB "
                f"({reduction:.1f}% reduction)"
            )

            return model

        except Exception as e:
            logger.error(f"Static quantization failed: {e}")
            raise

    # ==================== PRUNING ====================

    def prune_magnitude(
        self,
        amount: float = 0.3,
        output_path: Optional[str] = None,
    ) -> nn.Module:
        """
        Apply magnitude-based pruning.

        Removes weights with smallest absolute values.

        Args:
            amount: Fraction of weights to prune (0-1)
            output_path: Path to save pruned model

        Returns:
            Pruned model
        """
        try:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=amount)

            # Remove pruning buffers
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.remove(module, "weight")

            pruned_size = self._get_model_size(self.model)
            reduction = (self.original_size - pruned_size) / self.original_size * 100

            logger.info(
                f"✓ Magnitude pruning ({amount*100:.1f}%): {self.original_size:.2f}MB → {pruned_size:.2f}MB "
                f"({reduction:.1f}% reduction)"
            )

            if output_path:
                torch.save(self.model.state_dict(), output_path)

            return self.model

        except Exception as e:
            logger.error(f"Magnitude pruning failed: {e}")
            raise

    def prune_structured(
        self,
        amount: float = 0.3,
        output_path: Optional[str] = None,
    ) -> nn.Module:
        """
        Apply structured pruning (remove entire filters/channels).

        Args:
            amount: Fraction of channels to prune
            output_path: Path to save pruned model

        Returns:
            Pruned model
        """
        try:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv1d):
                    prune.ln_structured(
                        module,
                        name="weight",
                        amount=amount,
                        n=2,  # Remove 2D structures (filters)
                    )

            # Remove pruning buffers
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv1d):
                    prune.remove(module, "weight")

            pruned_size = self._get_model_size(self.model)
            reduction = (self.original_size - pruned_size) / self.original_size * 100

            logger.info(
                f"✓ Structured pruning ({amount*100:.1f}%): {self.original_size:.2f}MB → {pruned_size:.2f}MB "
                f"({reduction:.1f}% reduction)"
            )

            if output_path:
                torch.save(self.model.state_dict(), output_path)

            return self.model

        except Exception as e:
            logger.error(f"Structured pruning failed: {e}")
            raise

    # ==================== BATCH OPTIMIZATION ====================

    def optimize_batch_padding(
        self,
        batch: torch.Tensor,
        target_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimize batch by padding to consistent length.

        Reduces computational overhead of variable-length batches.

        Args:
            batch: Input batch (num_samples, variable_seq_len, input_dim)
            target_length: Target sequence length (auto-compute if None)

        Returns:
            (padded_batch, attention_mask)
        """
        num_samples = batch.shape[0]
        input_dim = batch.shape[-1]

        if target_length is None:
            target_length = max(seq.shape[0] for seq in batch)

        # Create padded batch
        padded_batch = torch.zeros(
            num_samples,
            target_length,
            input_dim,
            device=batch.device,
            dtype=batch.dtype,
        )

        # Create attention mask
        attention_mask = torch.zeros(
            num_samples,
            target_length,
            device=batch.device,
            dtype=torch.bool,
        )

        # Fill padded batch and mask
        for i, seq in enumerate(batch):
            seq_len = seq.shape[0]
            padded_batch[i, :seq_len] = seq
            attention_mask[i, :seq_len] = True

        return padded_batch, attention_mask

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization recommendations."""
        return {
            "model_size_mb": self.original_size,
            "recommendations": [
                "1. Apply dynamic quantization (INT8) for 4× size reduction",
                "2. Export to TorchScript for C++ serving",
                "3. Use ONNX for cross-platform deployment",
                "4. Apply magnitude pruning (30%) for 2-3× speedup",
                "5. Batch pad variable-length sequences for efficiency",
            ],
            "expected_benefits": {
                "quantization": "4× size reduction, 1.5-2× speedup",
                "torchscript": "Streamlined serving, faster inference",
                "onnx": "Cross-platform compatibility",
                "pruning": "2-3× speedup, 1.5× size reduction",
                "batch_optimization": "10-30% throughput improvement",
            },
        }


def create_optimized_export(
    model: nn.Module,
    example_input: torch.Tensor,
    output_dir: str,
    optimize_for: str = "latency",  # latency, memory, or balance
) -> Dict[str, str]:
    """
    Create fully optimized model exports for production.

    Args:
        model: Model to optimize
        example_input: Example input for tracing
        output_dir: Directory to save exports
        optimize_for: Optimization target (latency, memory, balance)

    Returns:
        Dict mapping export type to file path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    optimizer = SSMOptimizer(model)
    exports = {}

    # TorchScript export
    try:
        ts_path = os.path.join(output_dir, "model_traced.pt")
        optimizer.export_torchscript_trace(ts_path, example_input)
        exports["torchscript_trace"] = ts_path
    except Exception as e:
        logger.warning(f"TorchScript export failed: {e}")

    # ONNX export
    if ONNX_AVAILABLE:
        try:
            onnx_path = os.path.join(output_dir, "model_optimized.onnx")
            optimizer.export_onnx(onnx_path, example_input, optimize=True)
            exports["onnx"] = onnx_path
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

    # Quantization
    if optimize_for in ["memory", "balance"]:
        try:
            quant_path = os.path.join(output_dir, "model_quantized_int8.pt")
            optimizer.quantize_dynamic(quant_path)
            exports["quantized_int8"] = quant_path
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")

    # Pruning
    if optimize_for in ["latency", "balance"]:
        try:
            pruned_path = os.path.join(output_dir, "model_pruned_30pct.pt")
            optimizer.prune_magnitude(amount=0.3, output_path=pruned_path)
            exports["pruned_30pct"] = pruned_path
        except Exception as e:
            logger.warning(f"Pruning failed: {e}")

    # Save report
    report = optimizer.get_optimization_report()
    logger.info(f"✓ Optimization report: {report}")

    return exports


# Export public API
__all__ = [
    "SSMOptimizer",
    "create_optimized_export",
]
