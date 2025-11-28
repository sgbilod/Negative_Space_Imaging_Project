#!/usr/bin/env python
"""
Pipeline Demonstration
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This module provides a comprehensive demonstration of the Negative Space Imaging
end-to-end pipeline including acquisition, preprocessing, analysis, and visualization.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def print_header(title: str) -> None:
    """Print section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_step(step: int, description: str) -> None:
    """Print step indicator."""
    print(f"\n[Step {step}] {description}")
    print("-" * 40)


class PipelineDemo:
    """
    Comprehensive pipeline demonstration.
    
    Demonstrates the complete workflow from image acquisition
    through analysis and result visualization.
    """

    def __init__(
        self,
        output_dir: str = "./demo_output",
        use_gpu: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.use_gpu = use_gpu
        self.results: Dict[str, Any] = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_demo(self, mode: str = "full") -> Dict[str, Any]:
        """
        Run the pipeline demonstration.
        
        Args:
            mode: Demo mode (full, quick, acquisition_only, analysis_only)
            
        Returns:
            Dictionary of results
        """
        print_header("Negative Space Imaging Pipeline Demo")
        print(f"Mode: {mode}")
        print(f"Output: {self.output_dir}")
        print(f"GPU: {'Enabled' if self.use_gpu else 'Disabled'}")

        start_time = time.time()

        if mode in ("full", "quick", "acquisition_only"):
            self._demo_acquisition()

        if mode in ("full", "quick"):
            self._demo_preprocessing()

        if mode in ("full", "analysis_only"):
            self._demo_analysis()

        if mode == "full":
            self._demo_visualization()

        self._demo_summary()

        total_time = time.time() - start_time
        self.results["total_time"] = total_time
        print(f"\nTotal demo time: {total_time:.2f}s")

        return self.results

    def _demo_acquisition(self) -> None:
        """Demonstrate image acquisition."""
        print_step(1, "Image Acquisition")

        try:
            from integrated_acquisition_system import (
                IntegratedAcquisitionSystem,
                AcquisitionSourceType,
            )

            system = IntegratedAcquisitionSystem(max_workers=1)

            # Simulate acquisition
            import asyncio

            async def acquire():
                result = await system.acquire(
                    AcquisitionSourceType.SIMULATION,
                    {
                        "width": 512,
                        "height": 512,
                        "pattern": "negative_space",
                    },
                )
                return result

            result = asyncio.run(acquire())

            if result.success and result.image_data is not None:
                print(f"  ✓ Acquired image: {result.image_data.shape}")
                self.results["acquisition"] = {
                    "success": True,
                    "shape": list(result.image_data.shape),
                    "time": result.processing_time,
                }
                self._sample_image = result.image_data
            else:
                print(f"  ✗ Acquisition failed: {result.error_message}")
                self._create_fallback_image()

        except ImportError as e:
            print(f"  ! Import error: {e}")
            self._create_fallback_image()

    def _create_fallback_image(self) -> None:
        """Create fallback test image."""
        print("  Creating fallback test image...")
        # Create image with dark regions (negative space)
        image = np.random.randint(120, 200, (512, 512), dtype=np.uint8)
        
        # Add circular dark regions
        for _ in range(5):
            cx, cy = np.random.randint(50, 462, 2)
            radius = np.random.randint(30, 80)
            y, x = np.ogrid[:512, :512]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            image[mask] = np.random.randint(10, 50)

        self._sample_image = image.astype(np.float32)
        self.results["acquisition"] = {
            "success": True,
            "shape": list(image.shape),
            "fallback": True,
        }
        print(f"  ✓ Created fallback image: {image.shape}")

    def _demo_preprocessing(self) -> None:
        """Demonstrate preprocessing pipeline."""
        print_step(2, "Preprocessing")

        if not hasattr(self, "_sample_image"):
            self._create_fallback_image()

        try:
            from realtime_preprocessing import (
                RealtimePreprocessingPipeline,
                ImageFrame,
            )

            pipeline = RealtimePreprocessingPipeline()
            pipeline.start()

            # Process frame
            frame = ImageFrame(
                frame_id="demo_frame",
                data=self._sample_image.astype(np.float32),
            )

            # Submit and get result
            pipeline.submit_frame(frame.frame_id, frame.data)
            time.sleep(0.5)
            
            result = pipeline.get_result(timeout=2.0)
            pipeline.stop()

            if result:
                print(f"  ✓ Preprocessed frame: {result.frame_id}")
                print(f"    Quality: {result.quality.value if result.quality else 'N/A'}")
                print(f"    Stages: {len(result.processing_history)}")
                self._preprocessed_image = result.data
                self.results["preprocessing"] = {
                    "success": True,
                    "quality": result.quality.value if result.quality else None,
                    "stages": len(result.processing_history),
                }
            else:
                print("  ! No result from pipeline, using original image")
                self._preprocessed_image = self._sample_image

        except ImportError as e:
            print(f"  ! Import error: {e}")
            self._preprocessed_image = self._sample_image
            self.results["preprocessing"] = {"skipped": True}

    def _demo_analysis(self) -> None:
        """Demonstrate negative space analysis."""
        print_step(3, "Negative Space Analysis")

        if not hasattr(self, "_preprocessed_image"):
            if hasattr(self, "_sample_image"):
                self._preprocessed_image = self._sample_image
            else:
                self._create_fallback_image()
                self._preprocessed_image = self._sample_image

        image = self._preprocessed_image

        # Simple negative space detection
        print("  Analyzing negative space regions...")
        
        # Threshold to find dark regions
        threshold = np.percentile(image, 30)
        negative_space_mask = image < threshold

        # Find connected regions
        from scipy import ndimage
        labeled, num_features = ndimage.label(negative_space_mask)
        
        print(f"  ✓ Detected {num_features} negative space regions")

        # Analyze regions
        regions = []
        for i in range(1, num_features + 1):
            region_mask = labeled == i
            area = np.sum(region_mask)
            if area > 100:  # Minimum size filter
                y_coords, x_coords = np.where(region_mask)
                centroid = (np.mean(x_coords), np.mean(y_coords))
                mean_intensity = np.mean(image[region_mask])
                regions.append({
                    "id": i,
                    "area": int(area),
                    "centroid": [float(c) for c in centroid],
                    "mean_intensity": float(mean_intensity),
                })

        print(f"  ✓ Significant regions: {len(regions)}")
        for region in regions[:5]:  # Show first 5
            print(f"    - Region {region['id']}: area={region['area']}, "
                  f"centroid=({region['centroid'][0]:.0f}, {region['centroid'][1]:.0f})")

        self._analysis_results = {
            "total_regions": num_features,
            "significant_regions": len(regions),
            "regions": regions[:10],  # Keep top 10
            "negative_space_fraction": float(np.sum(negative_space_mask) / image.size),
        }
        self._negative_space_mask = negative_space_mask
        
        self.results["analysis"] = self._analysis_results

    def _demo_visualization(self) -> None:
        """Demonstrate result visualization."""
        print_step(4, "Visualization")

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap

            # Create figure with multiple panels
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            # Original image
            axes[0, 0].imshow(self._sample_image, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')

            # Preprocessed image
            if hasattr(self, '_preprocessed_image'):
                axes[0, 1].imshow(self._preprocessed_image, cmap='gray')
                axes[0, 1].set_title('Preprocessed Image')
            axes[0, 1].axis('off')

            # Negative space mask
            if hasattr(self, '_negative_space_mask'):
                axes[1, 0].imshow(self._negative_space_mask, cmap='viridis')
                axes[1, 0].set_title('Negative Space Detection')
            axes[1, 0].axis('off')

            # Overlay visualization
            if hasattr(self, '_preprocessed_image') and hasattr(self, '_negative_space_mask'):
                overlay = self._preprocessed_image.copy()
                overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
                rgb_overlay = np.stack([overlay, overlay, overlay], axis=-1)
                rgb_overlay[self._negative_space_mask, 0] = 1.0  # Red for negative space
                rgb_overlay[self._negative_space_mask, 1] *= 0.3
                rgb_overlay[self._negative_space_mask, 2] *= 0.3
                axes[1, 1].imshow(rgb_overlay)
                axes[1, 1].set_title('Negative Space Overlay')
            axes[1, 1].axis('off')

            plt.tight_layout()

            # Save figure
            output_path = self.output_dir / "pipeline_demo_results.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Saved visualization to {output_path}")
            self.results["visualization"] = {
                "success": True,
                "path": str(output_path),
            }

        except ImportError as e:
            print(f"  ! Visualization skipped: {e}")
            self.results["visualization"] = {"skipped": True}

    def _demo_summary(self) -> None:
        """Print demo summary."""
        print_step(5, "Summary")

        print("\nPipeline Demo Results:")
        print("-" * 40)

        if "acquisition" in self.results:
            acq = self.results["acquisition"]
            status = "✓" if acq.get("success") else "✗"
            print(f"  {status} Acquisition: {acq.get('shape', 'N/A')}")

        if "preprocessing" in self.results:
            pre = self.results["preprocessing"]
            if pre.get("skipped"):
                print("  - Preprocessing: Skipped")
            else:
                status = "✓" if pre.get("success") else "✗"
                print(f"  {status} Preprocessing: Quality={pre.get('quality', 'N/A')}")

        if "analysis" in self.results:
            ana = self.results["analysis"]
            print(f"  ✓ Analysis: {ana.get('significant_regions', 0)} regions detected")
            print(f"    Negative space: {ana.get('negative_space_fraction', 0)*100:.1f}%")

        if "visualization" in self.results:
            viz = self.results["visualization"]
            if viz.get("skipped"):
                print("  - Visualization: Skipped")
            else:
                print(f"  ✓ Visualization: {viz.get('path', 'N/A')}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Negative Space Imaging Pipeline Demo"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "acquisition_only", "analysis_only"],
        default="full",
        help="Demo mode",
    )
    parser.add_argument(
        "--output",
        default="./demo_output",
        help="Output directory",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    demo = PipelineDemo(
        output_dir=args.output,
        use_gpu=args.gpu,
    )

    try:
        results = demo.run_demo(mode=args.mode)
        return 0 if results.get("analysis") else 1
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
