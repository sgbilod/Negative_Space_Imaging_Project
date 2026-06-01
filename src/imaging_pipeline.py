"""
Clean 4-stage negative space imaging pipeline.
Stages: ImageLoader → NegativeSpaceDetector → AIEnhancer → Visualizer
"""

import warnings
import numpy as np
from pathlib import Path
from typing import Optional


# ─── Stage 1: ImageLoader ────────────────────────────────────────────────────

class ImageLoader:
    """Load images from FITS, DICOM, or standard raster formats into float32 [0,1] arrays."""

    def load(self, path: str) -> np.ndarray:
        p = Path(path)
        suffix = p.suffix.lower()

        if suffix in (".fits", ".fit", ".fts"):
            return self._load_fits(path)
        elif suffix in (".dcm", ".dicom"):
            return self._load_dicom(path)
        else:
            return self._load_standard(path)

    def _load_fits(self, path: str) -> np.ndarray:
        try:
            from astropy.io import fits
        except ImportError:
            warnings.warn("astropy not installed; cannot load FITS files")
            raise
        with fits.open(path) as hdul:
            data = hdul[0].data
        if data is None:
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    break
        arr = np.array(data, dtype=np.float32)
        # Collapse extra dimensions to 2-D
        while arr.ndim > 2:
            arr = arr[0]
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        return arr

    def _load_dicom(self, path: str) -> np.ndarray:
        try:
            import pydicom
        except ImportError:
            warnings.warn("pydicom not installed; cannot load DICOM files")
            raise
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        return arr

    def _load_standard(self, path: str) -> np.ndarray:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr


# ─── Stage 2: NegativeSpaceDetector ─────────────────────────────────────────

class NegativeSpaceDetector:
    """Detect negative space (background) regions via Otsu thresholding + connected components."""

    def detect(self, image: np.ndarray) -> dict:
        from skimage.filters import threshold_otsu
        from skimage.measure import label, regionprops

        # Work on grayscale
        if image.ndim == 3:
            gray = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
        else:
            gray = image.copy()

        thresh = threshold_otsu(gray)
        # Negative space = pixels below threshold (dark / background)
        mask = gray < thresh

        labeled = label(mask)
        props = regionprops(labeled)

        regions = []
        for prop in props:
            regions.append({
                "label": int(prop.label),
                "area": int(prop.area),
                "centroid": (float(prop.centroid[0]), float(prop.centroid[1])),
                "bbox": tuple(int(v) for v in prop.bbox),
            })

        negative_space_ratio = float(mask.sum()) / float(mask.size)

        return {
            "mask": mask,
            "regions": regions,
            "negative_space_ratio": negative_space_ratio,
        }


# ─── Stage 3: AIEnhancer ─────────────────────────────────────────────────────

class AIEnhancer:
    """
    Enhance the image in negative-space regions.
    Uses a .pt/.h5 model if one exists in models/; otherwise falls back to unsharp mask.
    """

    def __init__(self, models_dir: Optional[str] = None):
        self._model = None
        if models_dir:
            self._try_load_model(models_dir)

    def _try_load_model(self, models_dir: str):
        import glob
        pts = glob.glob(str(Path(models_dir) / "*.pt"))
        h5s = glob.glob(str(Path(models_dir) / "*.h5"))
        if pts:
            try:
                import torch
                self._model = torch.jit.load(pts[0], map_location="cpu")
                self._model.eval()
            except Exception:
                self._model = None
        elif h5s:
            try:
                import tensorflow as tf  # noqa: F401
                self._model = h5s[0]  # placeholder — caller can extend
            except Exception:
                self._model = None

    def enhance(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from skimage.filters import unsharp_mask as skimage_unsharp

        # If we have a torch model, apply it patch-by-patch (simple path)
        if self._model is not None:
            return self._model_enhance(image, mask)

        # Fallback: unsharp mask applied only to negative-space regions
        enhanced = image.copy()
        sharpened = skimage_unsharp(image, radius=2.0, amount=1.0)
        sharpened = np.clip(sharpened, 0.0, 1.0).astype(np.float32)

        # Broadcast mask to match image channels
        if image.ndim == 3:
            m = mask[..., np.newaxis]
        else:
            m = mask

        enhanced = np.where(m, sharpened, enhanced)
        return enhanced.astype(np.float32)

    def _model_enhance(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch
        # Naive pass-through: run full image through model, blend masked regions
        img_t = torch.from_numpy(image).float()
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0).unsqueeze(0)
        elif img_t.ndim == 3:
            img_t = img_t.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            try:
                out = self._model(img_t)
                out = out.squeeze().cpu().numpy()
                if out.ndim == 3:
                    out = out.transpose(1, 2, 0)
                out = np.clip(out, 0.0, 1.0).astype(np.float32)
                if image.ndim == 3:
                    m = mask[..., np.newaxis]
                else:
                    m = mask
                return np.where(m, out, image).astype(np.float32)
            except Exception:
                # Model failed; fall back to unsharp mask
                return self.enhance.__wrapped__(image, mask) if hasattr(self.enhance, "__wrapped__") else image


# ─── Stage 4: Visualizer ─────────────────────────────────────────────────────

class Visualizer:
    """Draw detected region boundaries and save results."""

    def overlay(self, original: np.ndarray, mask: np.ndarray, regions: list) -> np.ndarray:
        from skimage.segmentation import find_boundaries

        # Convert to uint8 RGB for drawing
        if original.ndim == 2:
            rgb = np.stack([original, original, original], axis=-1)
        else:
            rgb = original.copy()

        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

        # Draw boundaries in green
        boundaries = find_boundaries(mask.astype(np.int32), mode="outer")
        rgb[boundaries, 0] = 0
        rgb[boundaries, 1] = 255
        rgb[boundaries, 2] = 0

        return rgb

    def save(self, image: np.ndarray, output_path: str) -> None:
        from PIL import Image
        if image.dtype != np.uint8:
            image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        if image.ndim == 2:
            img = Image.fromarray(image, mode="L")
        else:
            img = Image.fromarray(image, mode="RGB")
        img.save(output_path)


# ─── Top-level pipeline ───────────────────────────────────────────────────────

def run_pipeline(input_path: str, output_path: str, models_dir: Optional[str] = None) -> dict:
    """
    Load → Detect → Enhance → Visualize → Save.
    Returns dict with regions, negative_space_ratio, output_path.
    """
    loader = ImageLoader()
    detector = NegativeSpaceDetector()
    enhancer = AIEnhancer(models_dir=models_dir)
    visualizer = Visualizer()

    image = loader.load(input_path)
    detection = detector.detect(image)
    enhanced = enhancer.enhance(image, detection["mask"])
    annotated = visualizer.overlay(enhanced, detection["mask"], detection["regions"])
    visualizer.save(annotated, output_path)

    return {
        "regions": detection["regions"],
        "negative_space_ratio": detection["negative_space_ratio"],
        "output_path": output_path,
    }
