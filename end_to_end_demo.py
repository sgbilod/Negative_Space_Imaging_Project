#!/usr/bin/env python
"""Canonical End-to-End Research Demo for Negative Space Imaging Project

This script is the authoritative research prototype entrypoint (Track A).
Steps:
 1. Load configuration (YAML) if provided.
 2. Acquire (simulate or load) an image and persist raw artifact.
 3. Produce a processed image suitable for reconstruction.
 4. Run minimal reconstruction stub to derive negative space metrics.
 5. Convert image to analysis-friendly CSV and run analysis module.
 6. Generate metrics + summary manifest (summary.json).
 7. Optionally perform secure verification (threshold signatures).

Outputs placed under the specified output directory in subfolders:
  raw/            - raw acquisition artifacts
  processed/      - processed image
  reconstruction/ - reconstruction_result.json
  analysis/       - analysis exported artifacts (from DataAnalysisSystem)
  metrics/        - metrics.json (summary of core metrics)
  logs/           - e2e_demo.log (run log)
  summary.json    - manifest listing key artifact paths

Exit code: 0 on success, non-zero on failure.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:  # Optional dependency for YAML config
    import yaml
    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    YAML_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    PIL_AVAILABLE = False

from image_acquisition import (
    ImageAcquisition as CoreImageAcquisition,
    ImageFormat,
    AcquisitionMode,
)
from advanced_reconstructor import AdvancedReconstructor
from data_analysis_system import DataAnalysisSystem
from config.config_loader import load_config


def configure_logging(log_dir: Path, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "e2e_demo.log"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.debug("Logging configured at %s", log_path)


def acquire_image(output_raw_dir: Path) -> Path:
    """Simulate acquisition and store raw artifact."""
    output_raw_dir.mkdir(parents=True, exist_ok=True)
    acquisition = CoreImageAcquisition(
        format=ImageFormat.RAW,
        mode=AcquisitionMode.SIMULATION,
    )
    data, metadata = acquisition.acquire(
        source="simulated_image",
        width=512,
        height=512,
        pattern="negative_space",
        negative_space_regions=3,
    )
    raw_path = output_raw_dir / "raw_image.raw"
    with open(raw_path, "wb") as f:
        f.write(data)
    meta_path = output_raw_dir / "raw_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logging.info("Acquisition complete: %s", raw_path)
    return raw_path


def create_processed_image(raw_path: Path, output_processed_dir: Path) -> Path:
    """Convert raw byte stream to a PNG for downstream steps."""
    output_processed_dir.mkdir(parents=True, exist_ok=True)
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL required to create processed image artifact")
    # Interpret raw bytes as grayscale image (512x512 as per acquisition)
    with open(raw_path, "rb") as f:
        raw_bytes = f.read()
    import numpy as np  # local import to keep top-level minimal
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    size = 512 * 512
    if arr.size < size:
        # Pad if unexpectedly short
        arr = np.pad(
            arr, (0, size - arr.size), mode="constant", constant_values=0
        )
    arr = arr[:size].reshape(512, 512)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    processed_path = output_processed_dir / "processed_image.png"
    img.save(processed_path)
    logging.info("Processed image created: %s", processed_path)
    return processed_path


def run_reconstruction(
    processed_image_path: Path, output_recon_dir: Path
) -> Dict[str, Any]:
    """Run advanced reconstruction replacing the minimal prototype.

    Returns a dictionary compatible with previous structure containing
    an 'artifact' path and a 'metrics' dictionary (now enriched).
    """
    output_recon_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate advanced reconstructor (defaults: Otsu +
    # open/close morphology)
    reconstructor = AdvancedReconstructor()
    result = reconstructor.reconstruct(
        str(processed_image_path), str(output_recon_dir)
    )

    # Persist full JSON summary (parity with previous stub artifact name)
    artifact_path = output_recon_dir / "reconstruction_result.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Compute legacy fields for compatibility (mean_intensity, dimensions)
    from PIL import Image  # local import
    import numpy as np
    im = Image.open(processed_image_path).convert("L")
    width, height = im.size
    pixels = np.array(im.getdata(), dtype=np.float32)
    mean_intensity = float(pixels.mean()) if pixels.size else 0.0

    # Extract spatial distribution metrics (short variable names
    # for readability)
    spatial_entropy = (
        result.aggregate_metrics.spatial_distribution.spatial_entropy
    )
    spatial_cluster_coeff = (
        result.aggregate_metrics.spatial_distribution.clustering_coefficient
    )

    metrics_enriched = {
        # Legacy prototype fields (maintain downstream compatibility)
        "image_path": str(processed_image_path),
        "feature_count": 0,  # placeholder (prototype used 3)
        "mean_intensity": mean_intensity,
        "negative_space_ratio": result.aggregate_metrics.negative_space_ratio,
        "negative_space_regions": result.aggregate_metrics.total_regions,
        "width": width,
        "height": height,
        # Advanced aggregate metrics
        "total_regions": result.aggregate_metrics.total_regions,
        "mean_region_area": result.aggregate_metrics.mean_region_area,
        "median_region_area": result.aggregate_metrics.median_region_area,
        "region_area_std": result.aggregate_metrics.region_area_std,
        "largest_region_area": result.aggregate_metrics.largest_region_area,
        "smallest_region_area": result.aggregate_metrics.smallest_region_area,
        "mean_convexity": result.aggregate_metrics.mean_convexity,
        "mean_circularity": result.aggregate_metrics.mean_circularity,
        "mean_eccentricity": result.aggregate_metrics.mean_eccentricity,
        "spatial_entropy": spatial_entropy,
        "spatial_clustering_coefficient": spatial_cluster_coeff,
        "processing_time_ms": result.provenance.processing_time_ms,
        "segmentation_strategy": result.segmentation_strategy,
        "morphology_operations": result.morphology_operations,
    }

    logging.info(
        "Advanced reconstruction complete: regions=%d ratio=%.3f artifact=%s",
        result.aggregate_metrics.total_regions,
        result.aggregate_metrics.negative_space_ratio,
        artifact_path,
    )

    return {
        "artifact": str(artifact_path),
        "metrics": metrics_enriched,
    }


def prepare_analysis_input(
    processed_image_path: Path, analysis_dir: Path
) -> Path:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL required for analysis input preparation")
    img = Image.open(processed_image_path).convert("L")
    pixels = list(img.getdata())
    # Downsample for lightweight analysis (every 8th pixel)
    sampled = pixels[::8]
    csv_path = analysis_dir / "pixels.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("intensity\n")
        for p in sampled:
            f.write(f"{p}\n")
    logging.info("Analysis input CSV written: %s", csv_path)
    return csv_path


def run_analysis(csv_path: Path, analysis_dir: Path) -> Dict[str, Any]:
    system = DataAnalysisSystem(config_path=None)
    # Restrict to fast statistical analysis for demo speed
    results = system.analyze_data(
        data_path=str(csv_path),
        analysis_types=["statistical"],
        output_prefix="e2e_demo",
        visualization=False,
    )
    logging.info("Analysis completed")
    return results


def export_metrics(
    metrics_dir: Path,
    reconstruction: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stats_summary = analysis.get("statistical", {}).get("summary", {})
    payload = {
        "reconstruction": reconstruction.get("metrics"),
        "analysis_mean_values": {
            k: v.get("mean")
            for k, v in stats_summary.items()
            if isinstance(v, dict)
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
    metrics_path = metrics_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Metrics exported: %s", metrics_path)
    return metrics_path


def secure_verify(
    processed_image_path: Path, signatures: int, threshold: int
) -> bool:
    cmd = [
        sys.executable,
        "secure_imaging_workflow.py",
        "--mode",
        "threshold",
        "--signatures",
        str(signatures),
        "--threshold",
        str(threshold),
        "--image",
        str(processed_image_path),
    ]
    logging.info("Running secure verification: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
        logging.debug("Secure verification output:\n%s", result.stdout)
        return "Verification successful" in result.stdout
    except subprocess.CalledProcessError as e:
        logging.error("Secure verification failed: %s", e)
        logging.error("stderr: %s", e.stderr)
        return False


def write_summary(
    output_dir: Path, artifacts: Dict[str, Path | str | Dict[str, Any]]
) -> Path:
    summary_path = output_dir / "summary.json"
    manifest = {
        k: (str(v) if not isinstance(v, dict) else v)
        for k, v in artifacts.items()
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logging.info("Summary written: %s", summary_path)
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonical Research E2E Demo")
    parser.add_argument(
        "--config", default="project_config.yaml",
        help="Path to project config (YAML/JSON)",
    )
    parser.add_argument(
        "--output-dir", default="output/e2e_demo_run",
        help="Directory for all artifacts",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--secure-verify", action="store_true",
        help="Run threshold signature verification step",
    )
    parser.add_argument(
        "--signatures", type=int, default=5,
        help="Number of signatures (if secure verify)",
    )
    parser.add_argument(
        "--threshold", type=int, default=3,
        help="Threshold (k) for verification",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    raw_dir = output_root / "raw"
    processed_dir = output_root / "processed"
    recon_dir = output_root / "reconstruction"
    analysis_dir = output_root / "analysis"
    metrics_dir = output_root / "metrics"
    logs_dir = output_root / "logs"

    configure_logging(logs_dir, args.verbose)

    logging.info("Starting End-to-End Research Demo")
    logging.info("Output root: %s", output_root)
    config = load_config(args.config)
    if config:
        logging.info("Loaded configuration from %s", args.config)
    else:
        logging.info("No configuration loaded or file missing; using defaults")

    try:
        # Acquisition + processing
        logging.info("STEP 1: Acquisition")
        raw_path = acquire_image(raw_dir)
        logging.info("STEP 2: Processing")
        processed_image = create_processed_image(raw_path, processed_dir)

        # Reconstruction
        logging.info("STEP 3: Reconstruction")
        reconstruction_data = run_reconstruction(processed_image, recon_dir)

        # Analysis
        logging.info("STEP 4: Analysis Preparation")
        csv_input = prepare_analysis_input(processed_image, analysis_dir)
        logging.info("STEP 5: Analysis Execution")
        analysis_results = run_analysis(csv_input, analysis_dir)

        # Metrics
        logging.info("STEP 6: Metrics Export")
        metrics_path = export_metrics(
            metrics_dir, reconstruction_data, analysis_results
        )

        # Optional secure verification
        verification_status = None
        if args.secure_verify:
            logging.info("STEP 7: Secure Verification")
            verification_status = secure_verify(
                processed_image, args.signatures, args.threshold
            )
            logging.info("Secure verification success=%s", verification_status)

        # Summary
        logging.info("STEP 8: Summary Generation")
        summary_path = write_summary(
            output_root,
            {
                "raw_image": raw_path,
                "processed_image": processed_image,
                "reconstruction_artifact": reconstruction_data.get("artifact"),
                "reconstruction_metrics": reconstruction_data.get("metrics"),
                "analysis_results_file_prefix": analysis_dir,
                "metrics_file": metrics_path,
                "secure_verification_success": verification_status,
                "config_used": args.config,
            },
        )

        logging.info("Demo completed successfully")
        print(json.dumps({"summary": str(summary_path)}, indent=2))
        return 0
    except Exception as e:  # noqa: BLE001
        logging.exception("Demo failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
