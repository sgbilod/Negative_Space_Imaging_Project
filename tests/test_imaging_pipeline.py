"""
Tests for the clean 4-stage imaging pipeline (src/imaging_pipeline.py).
Runs without opencv, torch, or heavy ML deps.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from imaging_pipeline import (
    ImageLoader,
    NegativeSpaceDetector,
    AIEnhancer,
    Visualizer,
    run_pipeline,
)


def _make_synthetic_image(h=256, w=256):
    """256x256 RGB: dark background (0.05) with a bright square in the centre."""
    img = np.full((h, w, 3), 0.05, dtype=np.float32)
    img[80:176, 80:176] = 0.9
    return img


def _save_png(arr: np.ndarray, path: str):
    from PIL import Image
    rgb = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(path)


# ─── test 1 ──────────────────────────────────────────────────────────────────

def test_load_png():
    """ImageLoader returns float32 array in [0, 1]."""
    img = _make_synthetic_image()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        _save_png(img, tmp)
        loader = ImageLoader()
        result = loader.load(tmp)
        assert result.dtype == np.float32
        assert result.ndim in (2, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    finally:
        os.unlink(tmp)


# ─── test 2 ──────────────────────────────────────────────────────────────────

def test_detect_negative_space():
    """NegativeSpaceDetector returns bool mask and ratio in [0, 1]."""
    img = _make_synthetic_image()
    detector = NegativeSpaceDetector()
    result = detector.detect(img)

    assert "mask" in result
    assert "regions" in result
    assert "negative_space_ratio" in result

    assert result["mask"].dtype == bool
    assert result["mask"].shape == img.shape[:2]
    assert 0.0 <= result["negative_space_ratio"] <= 1.0
    assert len(result["regions"]) > 0


# ─── test 3 ──────────────────────────────────────────────────────────────────

def test_enhance_no_model():
    """AIEnhancer fallback (no model) returns same-shape float32 array."""
    img = _make_synthetic_image()
    detector = NegativeSpaceDetector()
    mask = detector.detect(img)["mask"]

    enhancer = AIEnhancer(models_dir=None)
    enhanced = enhancer.enhance(img, mask)

    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.float32
    assert enhanced.min() >= 0.0
    assert enhanced.max() <= 1.0


# ─── test 4 ──────────────────────────────────────────────────────────────────

def test_visualize_overlay():
    """Visualizer.overlay returns same spatial shape as input (H, W, 3)."""
    img = _make_synthetic_image()
    detector = NegativeSpaceDetector()
    detection = detector.detect(img)

    viz = Visualizer()
    annotated = viz.overlay(img, detection["mask"], detection["regions"])

    assert annotated.shape[:2] == img.shape[:2]
    assert annotated.ndim == 3
    assert annotated.shape[2] == 3


# ─── test 5 ──────────────────────────────────────────────────────────────────

def test_full_pipeline():
    """run_pipeline on a synthetic PNG creates output file and returns expected keys."""
    img = _make_synthetic_image()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        in_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        out_path = f.name

    try:
        _save_png(img, in_path)
        result = run_pipeline(in_path, out_path)

        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
        assert "regions" in result
        assert "negative_space_ratio" in result
        assert result["output_path"] == out_path
        assert 0.0 <= result["negative_space_ratio"] <= 1.0
        # The synthetic image has a large dark background — expect meaningful negative space
        assert result["negative_space_ratio"] > 0.1
    finally:
        os.unlink(in_path)
        os.unlink(out_path)
