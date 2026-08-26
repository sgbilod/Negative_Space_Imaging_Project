"""
GPU-Accelerated Image Processing Pipeline
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

from gpu.acceleration import GPUManager
from gpu.utils import (
    DeviceContext,
    gpu_timer,
    optimize_tensor_memory,
    get_optimal_device
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("GPUImageProcessing")


class GPUImageProcessor:
    """GPU-accelerated image processing pipeline."""

    def __init__(
        self,
        config: Optional[Dict[str, any]] = None,
        device_id: Optional[int] = None
    ):
        self.config = config or {}

        # Initialize GPU manager
        self.gpu_manager = GPUManager(
            config=self.config,
            device_ids=[device_id] if device_id is not None else None
        )

        # Set up default device
        self.device = get_optimal_device()
        logger.info(f"Using device: {self.device}")

    def load_image(
        self,
        image: Union[np.ndarray, torch.Tensor, str, Path]
    ) -> torch.Tensor:
        """Load image to GPU memory."""
        try:
            if isinstance(image, (str, Path)):
                # Load image from file
                import cv2
                image = cv2.imread(str(image))
                if image is None:
                    raise ValueError(f"Failed to load image: {image}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to tensor and optimize memory
            image_tensor = optimize_tensor_memory(image)

            # Move to GPU
            return image_tensor.to(self.device)

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise

    def preprocess(
        self,
        image: torch.Tensor,
        normalize: bool = True,
        resize: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Preprocess image on GPU."""
        try:
            with gpu_timer("Preprocessing"):
                # Add batch dimension if needed
                if image.dim() == 3:
                    image = image.unsqueeze(0)

                # Resize if specified
                if resize is not None:
                    image = F.interpolate(
                        image,
                        size=resize,
                        mode='bilinear',
                        align_corners=False
                    )

                # Normalize if requested
                if normalize:
                    image = image.float() / 255.0

                return image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def apply_filters(
        self,
        image: torch.Tensor,
        filters: List[Dict[str, any]]
    ) -> torch.Tensor:
        """Apply multiple filters to image on GPU."""
        try:
            with gpu_timer("Applying filters"):
                result = image

                for filter_config in filters:
                    filter_type = filter_config['type']
                    params = filter_config.get('params', {})

                    if filter_type == 'gaussian_blur':
                        kernel_size = params.get('kernel_size', 5)
                        sigma = params.get('sigma', 1.0)
                        result = self._apply_gaussian_blur(
                            result,
                            kernel_size,
                            sigma
                        )

                    elif filter_type == 'sharpen':
                        strength = params.get('strength', 1.0)
                        result = self._apply_sharpening(
                            result,
                            strength
                        )

                    elif filter_type == 'contrast':
                        factor = params.get('factor', 1.0)
                        result = self._adjust_contrast(
                            result,
                            factor
                        )

                    elif filter_type == 'custom':
                        kernel = params.get('kernel')
                        if kernel is not None:
                            kernel = torch.tensor(
                                kernel,
                                device=self.device
                            )
                            result = self._apply_custom_filter(
                                result,
                                kernel
                            )

                return result

        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            raise

    def _apply_gaussian_blur(
        self,
        image: torch.Tensor,
        kernel_size: int,
        sigma: float
    ) -> torch.Tensor:
        """Apply Gaussian blur filter."""
        return F.gaussian_blur(
            image,
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma)
        )

    def _apply_sharpening(
        self,
        image: torch.Tensor,
        strength: float = 1.0
    ) -> torch.Tensor:
        """Apply sharpening filter."""
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ], device=self.device) * strength

        return self._apply_custom_filter(image, kernel)

    def _adjust_contrast(
        self,
        image: torch.Tensor,
        factor: float = 1.0
    ) -> torch.Tensor:
        """Adjust image contrast."""
        mean = image.mean(dim=[2, 3], keepdim=True)
        return (image - mean) * factor + mean

    def _apply_custom_filter(
        self,
        image: torch.Tensor,
        kernel: torch.Tensor
    ) -> torch.Tensor:
        """Apply custom convolution filter."""
        # Ensure kernel has correct shape
        if kernel.dim() == 2:
            kernel = kernel.unsqueeze(0).unsqueeze(0)

        # Apply filter to each channel
        channels = []
        for c in range(image.size(1)):
            channel = F.conv2d(
                image[:, c:c+1],
                kernel,
                padding=kernel.size(-1) // 2
            )
            channels.append(channel)

        return torch.cat(channels, dim=1)

    def detect_features(
        self,
        image: torch.Tensor,
        method: str = 'sobel',
        threshold: float = 0.1
    ) -> torch.Tensor:
        """Detect image features on GPU."""
        try:
            with gpu_timer("Feature detection"):
                if method == 'sobel':
                    # Sobel filters
                    sobel_x = torch.tensor([
                        [-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]
                    ], device=self.device).float()

                    sobel_y = torch.tensor([
                        [-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]
                    ], device=self.device).float()

                    # Apply filters
                    dx = self._apply_custom_filter(
                        image,
                        sobel_x
                    )
                    dy = self._apply_custom_filter(
                        image,
                        sobel_y
                    )

                    # Compute gradient magnitude
                    magnitude = torch.sqrt(dx.pow(2) + dy.pow(2))

                    # Threshold
                    return magnitude > threshold

                else:
                    raise ValueError(f"Unknown feature detection method: {method}")

        except Exception as e:
            logger.error(f"Error detecting features: {e}")
            raise

    def enhance_image(
        self,
        image: torch.Tensor,
        method: str = 'super_resolution',
        scale_factor: int = 2
    ) -> torch.Tensor:
        """Enhance image quality on GPU."""
        try:
            with gpu_timer("Image enhancement"):
                if method == 'super_resolution':
                    # Simple bicubic upscaling
                    return F.interpolate(
                        image,
                        scale_factor=scale_factor,
                        mode='bicubic',
                        align_corners=False
                    )

                else:
                    raise ValueError(f"Unknown enhancement method: {method}")

        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            raise

    def combine_images(
        self,
        images: List[torch.Tensor],
        method: str = 'average'
    ) -> torch.Tensor:
        """Combine multiple images on GPU."""
        try:
            with gpu_timer("Combining images"):
                if len(images) == 0:
                    raise ValueError("No images to combine")

                # Ensure all images are on the correct device
                images = [img.to(self.device) for img in images]

                if method == 'average':
                    return torch.stack(images).mean(dim=0)

                elif method == 'maximum':
                    return torch.stack(images).max(dim=0)[0]

                elif method == 'minimum':
                    return torch.stack(images).min(dim=0)[0]

                else:
                    raise ValueError(f"Unknown combination method: {method}")

        except Exception as e:
            logger.error(f"Error combining images: {e}")
            raise

    def save_image(
        self,
        image: torch.Tensor,
        path: Union[str, Path]
    ):
        """Save processed image to file."""
        try:
            with gpu_timer("Saving image"):
                # Move to CPU and convert to numpy
                image_np = image.cpu().numpy()

                # Ensure proper range
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)

                # Save image
                import cv2
                cv2.imwrite(
                    str(path),
                    cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                )

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            raise

    def process_batch(
        self,
        images: List[Union[np.ndarray, torch.Tensor, str, Path]],
        operations: List[Dict[str, any]]
    ) -> List[torch.Tensor]:
        """Process a batch of images on GPU."""
        try:
            # Load all images to GPU
            tensors = [
                self.load_image(img) for img in images
            ]

            # Process each operation
            results = []
            for op in operations:
                op_type = op['type']
                params = op.get('params', {})

                if op_type == 'preprocess':
                    tensors = [
                        self.preprocess(
                            img,
                            **params
                        ) for img in tensors
                    ]

                elif op_type == 'filters':
                    tensors = [
                        self.apply_filters(
                            img,
                            params.get('filters', [])
                        ) for img in tensors
                    ]

                elif op_type == 'detect_features':
                    tensors = [
                        self.detect_features(
                            img,
                            **params
                        ) for img in tensors
                    ]

                elif op_type == 'enhance':
                    tensors = [
                        self.enhance_image(
                            img,
                            **params
                        ) for img in tensors
                    ]

                elif op_type == 'combine':
                    tensors = [self.combine_images(tensors, **params)]

            return tensors

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise

    def cleanup(self):
        """Clean up GPU resources."""
        self.gpu_manager.cleanup()


# Example usage
if __name__ == "__main__":
    # Create processor
    processor = GPUImageProcessor()

    # Load test image
    image_path = "test_image.jpg"
    if Path(image_path).exists():
        # Process image
        image = processor.load_image(image_path)

        # Apply preprocessing
        image = processor.preprocess(
            image,
            normalize=True,
            resize=(512, 512)
        )

        # Apply filters
        filtered = processor.apply_filters(image, [
            {
                'type': 'gaussian_blur',
                'params': {'kernel_size': 5, 'sigma': 1.0}
            },
            {
                'type': 'sharpen',
                'params': {'strength': 1.5}
            }
        ])

        # Detect features
        features = processor.detect_features(
            filtered,
            method='sobel',
            threshold=0.1
        )

        # Enhance image
        enhanced = processor.enhance_image(
            filtered,
            method='super_resolution',
            scale_factor=2
        )

        # Save results
        processor.save_image(enhanced, "enhanced.jpg")

        # Cleanup
        processor.cleanup()
