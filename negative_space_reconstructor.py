#!/usr/bin/env python
"""Negative Space Reconstructor (Research Prototype Stub)

This lightweight stub provides a minimal implementation sufficient for the
end-to-end research demo pipeline. It simulates the reconstruction and
negative space mapping process so downstream analysis can operate on
generated artifacts.

Responsibilities (Prototype):
 - Accept a processed image path
 - Extract trivial "features" (basic pixel statistics)
 - Produce a pseudo 3D reconstruction artifact (JSON)
 - Map negative space regions heuristically (threshold segmentation)
 - Provide a dictionary summary for inclusion in pipeline outputs

Future (Not implemented here):
 - Real feature extraction (edges, contours, semantic regions)
 - True 3D model generation
 - Blockchain / cryptographic integration
 - Advanced spatial analytics

NOTE: Keep implementation intentionally simple and deterministic so tests
and demos remain stable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PIL_AVAILABLE = False


@dataclass
class ReconstructionResult:
    image_path: str
    feature_count: int
    mean_intensity: float
    negative_space_ratio: float
    negative_space_regions: int
    width: int
    height: int


class NegativeSpaceReconstructor:
    """Minimal reconstructor used by the research demo pipeline."""

    def __init__(self) -> None:
        self._last_result: ReconstructionResult | None = None

    def reconstruct(self, processed_image_path: str) -> ReconstructionResult:
        """Run a minimal reconstruction pipeline on a processed image.

        Args:
            processed_image_path: Path to a preprocessed image file.

        Returns:
            ReconstructionResult with basic derived metrics.
        """
        if not os.path.exists(processed_image_path):
            raise FileNotFoundError(
                f"Processed image not found: {processed_image_path}"
            )

        if not PIL_AVAILABLE:
            raise RuntimeError("PIL is required for reconstruction prototype")

        im = Image.open(processed_image_path).convert("L")  # grayscale
        width, height = im.size
        pixels = list(im.getdata())
        total = len(pixels)
        mean_intensity = sum(pixels) / float(total) if total else 0.0

        # Heuristic negative space: pixels below 25th percentile
        sorted_px = sorted(pixels)
        threshold_index = int(0.25 * total)
        threshold_val = sorted_px[threshold_index] if total else 0
        negative_mask = [p <= threshold_val for p in pixels]
        negative_space_ratio = (
            sum(negative_mask) / float(total) if total else 0.0
        )

        # Very naive region count: scan rows and count transitions
        # Simple heuristic; replace with connected-component labeling later.
        regions = 0
        in_region = False
        for p in negative_mask:
            if p and not in_region:
                regions += 1
                in_region = True
            elif not p and in_region:
                in_region = False

        result = ReconstructionResult(
            image_path=processed_image_path,
            feature_count=3,  # mean, ratio, regions (placeholder)
            mean_intensity=mean_intensity,
            negative_space_ratio=negative_space_ratio,
            negative_space_regions=regions,
            width=width,
            height=height,
        )
        self._last_result = result
        return result

    def export(self, output_dir: str) -> str:
        """Export the last reconstruction result as JSON.

        Args:
            output_dir: Directory in which to place reconstruction artifact.

        Returns:
            Path to the JSON artifact.
        """
        if self._last_result is None:
            raise RuntimeError("No reconstruction result available to export")

        os.makedirs(output_dir, exist_ok=True)
        artifact_path = Path(output_dir) / "reconstruction_result.json"
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self._last_result), f, indent=2)
        return str(artifact_path)


def run_minimal_reconstruction(
    processed_image_path: str, output_dir: str
) -> Dict[str, Any]:
    """Helper used by the E2E demo.

    Returns a dictionary with artifact path + metrics for pipeline summary.
    """
    recon = NegativeSpaceReconstructor()
    result = recon.reconstruct(processed_image_path)
    artifact = recon.export(output_dir)
    return {
        "artifact": artifact,
        "metrics": asdict(result)
    }


if __name__ == "__main__":  # Manual quick test
    import argparse
    parser = argparse.ArgumentParser(
        description="Standalone reconstruction stub test"
    )
    parser.add_argument(
        "--image", required=True, help="Path to processed image"
    )
    parser.add_argument(
        "--out", default="reconstruction_out", help="Output directory"
    )
    args = parser.parse_args()
    data = run_minimal_reconstruction(args.image, args.out)
    print(json.dumps(data, indent=2))
