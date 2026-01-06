"""
Synthetic Negative Space Data Generation Script

Generate synthetic astronomical images with negative space patterns:
- Gaussian, Poisson, and mixed noise
- Multiple sizes: 64x64, 128x128, 256x256
- Dataset: 5,000 training + 1,000 validation images
- Realistic astronomical degradation patterns
- Background variation and stellar sources

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AstronomicalImageGenerator:
    """Generate synthetic astronomical images with negative space patterns."""

    def __init__(
        self,
        image_size: int = 256,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize generator.

        Args:
            image_size: Size of generated images
            seed: Random seed for reproducibility
        """
        self.image_size = image_size
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def generate_background(
        self,
        background_type: str = "gradient",
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """
        Generate astronomical background.

        Args:
            background_type: Type of background ('uniform', 'gradient', 'structured')
            noise_level: Background noise level

        Returns:
            Background image
        """
        if background_type == "uniform":
            background = np.ones((self.image_size, self.image_size)) * 100
        elif background_type == "gradient":
            x = np.linspace(0, 1, self.image_size)
            y = np.linspace(0, 1, self.image_size)
            xx, yy = np.meshgrid(x, y)
            background = 100 + 50 * (xx + yy) / 2
        elif background_type == "structured":
            # Create structured background with variations
            background = 100 * np.ones((self.image_size, self.image_size))
            # Add large-scale variations
            variation = ndimage.gaussian_filter(
                np.random.randn(self.image_size, self.image_size),
                sigma=20,
            )
            background = background + 20 * variation
        else:
            background = np.ones((self.image_size, self.image_size)) * 100

        # Add noise
        background = background + noise_level * np.random.randn(*background.shape)
        background = np.clip(background, 0, 255)

        return background

    def generate_stellar_sources(
        self,
        num_sources: int = 5,
        brightness_range: Tuple[float, float] = (50, 200),
        size_range: Tuple[float, float] = (1, 5),
    ) -> np.ndarray:
        """
        Generate stellar sources.

        Args:
            num_sources: Number of sources to generate
            brightness_range: Range of source brightness
            size_range: Range of source sizes (sigma for Gaussian)

        Returns:
            Image with stellar sources
        """
        image = np.zeros((self.image_size, self.image_size))

        for _ in range(num_sources):
            # Random position
            x = np.random.randint(self.image_size)
            y = np.random.randint(self.image_size)

            # Random brightness and size
            brightness = np.random.uniform(*brightness_range)
            size = np.random.uniform(*size_range)

            # Create Gaussian source
            yy, xx = np.ogrid[:self.image_size, :self.image_size]
            gaussian = brightness * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * size ** 2))
            image = image + gaussian

        image = np.clip(image, 0, 255)
        return image

    def add_gaussian_noise(
        self,
        image: np.ndarray,
        noise_level: float = 10,
    ) -> np.ndarray:
        """
        Add Gaussian noise.

        Args:
            image: Input image
            noise_level: Noise standard deviation

        Returns:
            Noisy image
        """
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255)

    def add_poisson_noise(
        self,
        image: np.ndarray,
        photon_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Add Poisson noise (photon noise).

        Args:
            image: Input image
            photon_scale: Photon scale factor

        Returns:
            Image with Poisson noise
        """
        # Scale image to photon counts
        photon_image = image / 255 * 1000 * photon_scale
        # Sample from Poisson distribution
        noisy_counts = np.random.poisson(photon_image)
        # Scale back to 0-255 range
        noisy_image = np.clip(noisy_counts / (1000 * photon_scale) * 255, 0, 255)
        return noisy_image

    def add_mixed_noise(
        self,
        image: np.ndarray,
        gaussian_level: float = 5,
        poisson_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Add mixed Gaussian and Poisson noise.

        Args:
            image: Input image
            gaussian_level: Gaussian noise standard deviation
            poisson_scale: Poisson noise scale

        Returns:
            Image with mixed noise
        """
        # Add Poisson noise first
        image = self.add_poisson_noise(image, poisson_scale)
        # Then add Gaussian noise
        image = self.add_gaussian_noise(image, gaussian_level)
        return image

    def generate_negative_space_pattern(
        self,
        pattern_type: str = "circular",
        pattern_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate negative space pattern.

        Args:
            pattern_type: Type of negative space pattern
            pattern_size: Size of pattern (pixels)

        Returns:
            Pattern mask
        """
        pattern = np.ones((self.image_size, self.image_size))

        if pattern_size is None:
            pattern_size = self.image_size // 4

        center_x = self.image_size // 2
        center_y = self.image_size // 2

        if pattern_type == "circular":
            yy, xx = np.ogrid[:self.image_size, :self.image_size]
            mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= pattern_size ** 2
            pattern[mask] = 0
        elif pattern_type == "rectangular":
            x_start = center_x - pattern_size // 2
            x_end = center_x + pattern_size // 2
            y_start = center_y - pattern_size // 2
            y_end = center_y + pattern_size // 2
            pattern[y_start:y_end, x_start:x_end] = 0
        elif pattern_type == "annular":
            yy, xx = np.ogrid[:self.image_size, :self.image_size]
            dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
            inner_mask = dist <= pattern_size * 0.7
            outer_mask = dist <= pattern_size
            pattern[inner_mask] = 1
            pattern[outer_mask & ~inner_mask] = 0

        return pattern

    def generate_synthetic_image(
        self,
        background_type: str = "gradient",
        num_sources: int = 5,
        noise_type: str = "mixed",
        pattern_type: str = "circular",
        apply_degradation: bool = True,
    ) -> np.ndarray:
        """
        Generate complete synthetic astronomical image.

        Args:
            background_type: Type of background
            num_sources: Number of stellar sources
            noise_type: Type of noise
            pattern_type: Type of negative space pattern
            apply_degradation: Whether to apply degradation

        Returns:
            Generated image
        """
        # Generate background
        image = self.generate_background(background_type)

        # Generate and add stellar sources
        sources = self.generate_stellar_sources(num_sources=num_sources)
        image = image + sources

        # Apply negative space pattern
        pattern = self.generate_negative_space_pattern(pattern_type=pattern_type)
        image = image * pattern

        # Add noise
        if noise_type == "gaussian":
            image = self.add_gaussian_noise(image, noise_level=10)
        elif noise_type == "poisson":
            image = self.add_poisson_noise(image, photon_scale=1.0)
        elif noise_type == "mixed":
            image = self.add_mixed_noise(image)

        # Normalize to 0-1 range
        image = np.clip(image, 0, 255) / 255.0

        return image

    def generate_batch(
        self,
        num_images: int = 100,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate batch of synthetic images.

        Args:
            num_images: Number of images to generate
            **kwargs: Arguments for generate_synthetic_image

        Returns:
            Batch of generated images
        """
        images = []
        pbar = range(num_images)
        try:
            from tqdm import tqdm
            pbar = tqdm(pbar, desc="Generating synthetic images")
        except ImportError:
            pass

        for _ in pbar:
            image = self.generate_synthetic_image(**kwargs)
            images.append(image)

        # Stack into batch
        batch = np.stack(images, axis=0)
        return batch


class SyntheticAstronomicalDataset(Dataset):
    """Dataset of synthetic astronomical images."""

    def __init__(
        self,
        num_images: int = 1000,
        image_size: int = 256,
        generator: Optional[AstronomicalImageGenerator] = None,
        transform: Optional[callable] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize dataset.

        Args:
            num_images: Number of images in dataset
            image_size: Image size
            generator: Image generator
            transform: Optional image transform
            seed: Random seed
        """
        self.num_images = num_images
        self.image_size = image_size
        self.transform = transform

        if generator is None:
            generator = AstronomicalImageGenerator(image_size=image_size, seed=seed)

        self.generator = generator

        # Generate all images
        logger.info(f"Generating {num_images} synthetic images...")
        self.images = generator.generate_batch(
            num_images=num_images,
            background_type=np.random.choice(["uniform", "gradient", "structured"]),
            num_sources=np.random.randint(3, 10),
            noise_type=np.random.choice(["gaussian", "poisson", "mixed"]),
            pattern_type=np.random.choice(["circular", "rectangular", "annular"]),
        )

    def __len__(self) -> int:
        """Get dataset length."""
        return self.num_images

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item from dataset.

        Args:
            idx: Index

        Returns:
            Image tensor and label
        """
        image = self.images[idx]

        # Convert to tensor and add channel dimension
        image = torch.from_numpy(image).float()
        if image.dim() == 2:
            image = image.unsqueeze(0)

        # Repeat single channel to 3 channels
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if self.transform:
            image = self.transform(image)

        # Binary label: has negative space or not
        label = 1 if (image.sum() < image.numel() * 0.5) else 0

        return image, label


def generate_dataset(
    output_dir: str = "./datasets/synthetic",
    image_sizes: List[int] = None,
    num_train: int = 5000,
    num_val: int = 1000,
    seed: int = 42,
) -> None:
    """
    Generate and save synthetic dataset.

    Args:
        output_dir: Directory to save dataset
        image_sizes: List of image sizes to generate
        num_train: Number of training images
        num_val: Number of validation images
        seed: Random seed
    """
    if image_sizes is None:
        image_sizes = [64, 128, 256]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_size in image_sizes:
        logger.info(f"\nGenerating {image_size}x{image_size} dataset...")

        # Generate training set
        train_dataset = SyntheticAstronomicalDataset(
            num_images=num_train,
            image_size=image_size,
            seed=seed,
        )

        # Generate validation set
        val_dataset = SyntheticAstronomicalDataset(
            num_images=num_val,
            image_size=image_size,
            seed=seed + 1,
        )

        # Save datasets
        train_path = output_path / f"synthetic_train_{image_size}x{image_size}.npz"
        val_path = output_path / f"synthetic_val_{image_size}x{image_size}.npz"

        np.savez_compressed(
            train_path,
            images=train_dataset.images,
        )

        np.savez_compressed(
            val_path,
            images=val_dataset.images,
        )

        logger.info(f"Saved training set to {train_path}")
        logger.info(f"Saved validation set to {val_path}")


def load_synthetic_dataset(
    dataset_path: str,
    image_size: int = 256,
) -> SyntheticAstronomicalDataset:
    """
    Load synthetic dataset from file.

    Args:
        dataset_path: Path to dataset file
        image_size: Image size

    Returns:
        Loaded dataset
    """
    data = np.load(dataset_path)
    images = data["images"]

    dataset = SyntheticAstronomicalDataset(
        num_images=len(images),
        image_size=image_size,
    )
    dataset.images = images

    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_dataset(
        output_dir="./datasets/synthetic_negative_space",
        image_sizes=[64, 128, 256],
        num_train=5000,
        num_val=1000,
    )
