"""
Diffusion Model API Service

REST API endpoint layer for diffusion model inference and reconstruction.

Provides:
- Image generation endpoints
- Reconstruction service
- Batch processing
- Real-time inference
- Health checks

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import base64
import io
import json
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class DiffusionServiceConfig:
    """Configuration for diffusion service."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        max_batch_size: int = 32,
        max_image_size: Tuple[int, int] = (512, 512),
        default_num_steps: int = 50,
        cache_models: bool = True,
    ) -> None:
        """
        Initialize service config.

        Args:
            model_path: Path to model checkpoint
            device: Device to use
            max_batch_size: Maximum batch size for inference
            max_image_size: Maximum image size
            default_num_steps: Default number of inference steps
            cache_models: Whether to cache loaded models
        """
        self.model_path = model_path
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_image_size = max_image_size
        self.default_num_steps = default_num_steps
        self.cache_models = cache_models


class ImageProcessor:
    """Handle image encoding/decoding for API requests."""

    @staticmethod
    def image_to_base64(image: np.ndarray) -> str:
        """
        Convert numpy image to base64 string.

        Args:
            image: Image array (H, W, C) or (H, W)

        Returns:
            Base64 encoded image string
        """
        # Normalize to 0-255 if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(image, mode='L')

        # Encode to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return img_str

    @staticmethod
    def base64_to_image(img_str: str) -> np.ndarray:
        """
        Convert base64 string to numpy image.

        Args:
            img_str: Base64 encoded image string

        Returns:
            Image array
        """
        img_data = base64.b64decode(img_str)
        buffer = io.BytesIO(img_data)
        pil_image = Image.open(buffer)

        return np.array(pil_image)

    @staticmethod
    def preprocess_image(image: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: Input image array
            target_size: Target size

        Returns:
            Preprocessed tensor
        """
        # Resize if needed
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize(target_size, Image.LANCZOS)

        # Convert to tensor
        img_array = np.array(pil_image) / 255.0
        tensor = torch.from_numpy(img_array).float()

        # Add batch dimension and move to GPU if needed
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif len(tensor.shape) == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        return tensor


class DiffusionService:
    """
    Main diffusion service for API integration.

    Handles model loading, inference, and result formatting.
    """

    def __init__(self, config: DiffusionServiceConfig) -> None:
        """
        Initialize service.

        Args:
            config: Service configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.model_loaded = False
        self.image_processor = ImageProcessor()

        # Load model if path provided
        if config.model_path:
            self._load_model()

    def _load_model(self) -> bool:
        """
        Load diffusion model.

        Returns:
            Whether loading succeeded
        """
        try:
            # Import model here to avoid dependency issues
            from neural.diffusion_model_prototype import DiffusionFactory

            logger.info(f"Loading model from {self.config.model_path}")
            self.model = DiffusionFactory.create_model()

            # Load checkpoint if it exists
            try:
                checkpoint = torch.load(
                    self.config.model_path,
                    map_location=self.device
                )
                self.model.load_state_dict(checkpoint)
                logger.info("Model checkpoint loaded successfully")
            except FileNotFoundError:
                logger.warning(f"Checkpoint not found at {self.config.model_path}")

            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint.

        Returns:
            Health status
        """
        return {
            "status": "healthy",
            "model_loaded": self.model_loaded,
            "device": str(self.device),
            "max_batch_size": self.config.max_batch_size,
        }

    def generate(
        self,
        num_samples: int = 1,
        num_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_intermediate: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate new images.

        Args:
            num_samples: Number of samples to generate
            num_steps: Number of inference steps
            batch_size: Batch size for generation
            return_intermediate: Whether to return intermediate steps

        Returns:
            Generation results with base64 encoded images
        """
        if not self.model_loaded:
            return {"error": "Model not loaded"}

        num_steps = num_steps or self.config.default_num_steps
        batch_size = batch_size or self.config.max_batch_size
        num_samples = min(num_samples, self.config.max_batch_size * 10)

        try:
            with torch.no_grad():
                # Generate in batches
                all_samples = []

                for i in range(0, num_samples, batch_size):
                    current_batch_size = min(batch_size, num_samples - i)
                    samples = self.model.sample(
                        num_samples=current_batch_size,
                        num_steps=num_steps
                    )
                    all_samples.append(samples)

                generated = torch.cat(all_samples, dim=0)

            # Convert to numpy and encode
            images_base64 = []

            for i in range(len(generated)):
                img = generated[i].cpu().numpy()
                if img.ndim == 3:
                    img = img.transpose(1, 2, 0)

                img_b64 = self.image_processor.image_to_base64(img)
                images_base64.append(img_b64)

            return {
                "status": "success",
                "num_samples": len(images_base64),
                "images": images_base64,
                "num_steps": num_steps,
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "error": f"Generation failed: {str(e)}",
                "status": "error",
            }

    def reconstruct(
        self,
        image_base64: str,
        target_size: Optional[Tuple[int, int]] = None,
        num_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Reconstruct/denoise an image.

        Args:
            image_base64: Base64 encoded input image
            target_size: Target size for reconstruction
            num_steps: Number of inference steps
            guidance_scale: Guidance scale for reconstruction

        Returns:
            Reconstruction result with base64 encoded image
        """
        if not self.model_loaded:
            return {"error": "Model not loaded"}

        num_steps = num_steps or self.config.default_num_steps
        target_size = target_size or self.config.max_image_size

        try:
            # Decode input image
            image = self.image_processor.base64_to_image(image_base64)

            # Preprocess
            tensor = self.image_processor.preprocess_image(image, target_size)
            tensor = tensor.to(self.device)

            # Reconstruct
            with torch.no_grad():
                reconstructed = self.model.reconstruct(
                    tensor,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale
                )

            # Convert result
            result_img = reconstructed.cpu().numpy().squeeze()
            if result_img.ndim == 3:
                result_img = result_img.transpose(1, 2, 0)

            result_b64 = self.image_processor.image_to_base64(result_img)

            return {
                "status": "success",
                "image": result_b64,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "output_size": list(result_img.shape),
            }

        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            return {
                "error": f"Reconstruction failed: {str(e)}",
                "status": "error",
            }

    def batch_reconstruct(
        self,
        images_base64: list,
        target_size: Optional[Tuple[int, int]] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Reconstruct multiple images.

        Args:
            images_base64: List of base64 encoded images
            target_size: Target size
            num_steps: Number of inference steps

        Returns:
            Batch reconstruction results
        """
        results = []

        for img_b64 in images_base64:
            result = self.reconstruct(
                img_b64,
                target_size=target_size,
                num_steps=num_steps
            )
            results.append(result)

        return {
            "status": "success" if all(r.get("status") == "success" for r in results) else "partial_error",
            "num_processed": len(results),
            "results": results,
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get service configuration.

        Returns:
            Configuration dictionary
        """
        return {
            "max_batch_size": self.config.max_batch_size,
            "max_image_size": self.config.max_image_size,
            "default_num_steps": self.config.default_num_steps,
            "device": str(self.device),
        }


class DiffusionServiceFactory:
    """Factory for creating service instances."""

    _instance: Optional[DiffusionService] = None

    @classmethod
    def get_service(
        cls,
        config: Optional[DiffusionServiceConfig] = None,
        force_reload: bool = False,
    ) -> DiffusionService:
        """
        Get or create service instance.

        Args:
            config: Configuration for new instance
            force_reload: Force reload even if instance exists

        Returns:
            Service instance
        """
        if cls._instance is None or force_reload:
            if config is None:
                config = DiffusionServiceConfig()
            cls._instance = DiffusionService(config)

        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset service instance."""
        cls._instance = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Diffusion API service module loaded successfully")
