#!/usr/bin/env python
"""Smoke tests for canonical end-to-end research demo.

Checks:
 - Exit code success
 - Artifact directories exist
 - summary.json & metrics.json contain expected fields
 - Optional secure verification succeeds for multiple variants
 - Runtime stays under MAX_SECONDS threshold
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import unittest
import pytest

MAX_SECONDS = 20  # Upper bound for demo completion time


@pytest.mark.smoke
class TestEndToEndDemo(unittest.TestCase):
    """Canonical pipeline smoke tests (with timing & secure variants)."""

    def setUp(self) -> None:
        self.output_dir = Path("output/test_e2e_demo_run")
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def tearDown(self) -> None:
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @pytest.mark.smoke
    def test_demo_produces_artifacts(self):
        cmd = [
            sys.executable,
            "end_to_end_demo.py",
            "--output-dir",
            str(self.output_dir),
        ]
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(
            elapsed, MAX_SECONDS,
            f"Demo runtime {elapsed:.2f}s exceeds {MAX_SECONDS}s",
        )

        # Directories
        for sub in [
            "raw",
            "processed",
            "reconstruction",
            "analysis",
            "metrics",
            "logs",
        ]:
            self.assertTrue(
                (self.output_dir / sub).exists(), f"Missing directory: {sub}"
            )

        # Summary
        summary_path = self.output_dir / "summary.json"
        self.assertTrue(summary_path.exists(), "summary.json missing")
        summary = json.load(summary_path.open("r", encoding="utf-8"))
        for key in [
            "raw_image",
            "processed_image",
            "reconstruction_artifact",
            "reconstruction_metrics",
            "metrics_file",
        ]:
            self.assertIn(key, summary, f"Missing summary key: {key}")

        # Metrics
        metrics_path = Path(summary["metrics_file"])
        self.assertTrue(metrics_path.exists(), "metrics.json missing")
        metrics = json.load(metrics_path.open("r", encoding="utf-8"))
        self.assertIn("reconstruction", metrics)
        recon_metrics = metrics["reconstruction"]
        for field in [
            "mean_intensity",
            "negative_space_ratio",
            "negative_space_regions",
            "width",
            "height",
        ]:
            self.assertIn(field, recon_metrics)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "signatures,threshold", [(5, 3), (4, 2), (6, 4)]
    )
    def test_secure_verification_variants(self, signatures, threshold):
        variant_dir = self.output_dir / f"secure_{signatures}_{threshold}"
        cmd = [
            sys.executable,
            "end_to_end_demo.py",
            "--output-dir",
            str(variant_dir),
            "--secure-verify",
            "--signatures",
            str(signatures),
            "--threshold",
            str(threshold),
        ]
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(
            elapsed, MAX_SECONDS,
            f"Secure variant runtime {elapsed:.2f}s exceeds {MAX_SECONDS}s",
        )
        summary_path = variant_dir / "summary.json"
        self.assertTrue(summary_path.exists(), "summary.json missing")
        summary = json.load(summary_path.open("r", encoding="utf-8"))
        self.assertTrue(
            summary.get("secure_verification_success"),
            "Secure verification did not succeed",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
