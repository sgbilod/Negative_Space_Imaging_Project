"""
Diffusion Model Prototype for Negative Space Reconstruction

Complete diffusion-based reconstruction pipeline with:
- Multiple scheduler support (DDPM, DDIM, PNDM)
- Custom negative space reconstruction pipeline
- Configurable noise schedules
- Stochastic and deterministic sampling
- Inference with variable steps (20-1000)
- Quality evaluation metrics

Copyright (c) 2025 Stephen Bilodeau. All Rights Reserved.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMPipeline, DDPMPipeline, PNDMPipeline
from torch import Tensor

logger = logging.getLogger(__name__)


class NoiseSchedule(Enum):
    """Noise schedule types."""
    LINEAR = "linear"
    COSINE = "cosine"
    SQRT = "sqrt"
    QUADRATIC = "quadratic"


class SamplingStrategy(Enum):
    """Sampling strategies."""
    STOCHASTIC = "stochastic"
    DETERMINISTIC = "deterministic"
    DDIM = "ddim"


class SchedulerType(Enum):
    """Available schedulers."""
    DDPM = "ddpm"
    DDIM = "ddim"
    PNDM = "pndm"


class DiffusionConfig:
    """Configuration for diffusion model."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        noise_schedule: NoiseSchedule = NoiseSchedule.LINEAR,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        image_size: int = 256,
        channels: int = 3,
    ) -> None:
        """
        Initialize diffusion config.

        Args:
            num_timesteps: Number of diffusion steps
            noise_schedule: Noise schedule type
            beta_start: Starting beta value
            beta_end: Ending beta value
            image_size: Image size (assumed square)
            channels: Number of image channels
        """
        self.num_timesteps = num_timesteps
        self.noise_schedule = noise_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.channels = channels

        # Compute noise schedule
        self.betas = self._compute_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # Pre-compute variances
        self.variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.log_variance = np.log(np.clip(self.variance, a_min=1e-20, a_max=None))

    def _compute_betas(self) -> np.ndarray:
        """
        Compute noise schedule betas.

        Returns:
            Beta schedule array
        """
        if self.noise_schedule == NoiseSchedule.LINEAR:
            betas = np.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.noise_schedule == NoiseSchedule.COSINE:
            # Cosine schedule
            def alpha_bar(t):
                return np.cos(((t / self.num_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2

            alphas_cumprod = np.array(
                [alpha_bar(t) for t in range(self.num_timesteps)]
            )
            betas = 1.0 - alphas_cumprod / np.append(1.0, alphas_cumprod[:-1])
            betas = np.clip(betas, a_min=0.0001, a_max=0.9999)
        elif self.noise_schedule == NoiseSchedule.SQRT:
            # Square root schedule
            betas = np.sqrt(
                np.linspace(self.beta_start ** 2, self.beta_end ** 2, self.num_timesteps)
            )
        elif self.noise_schedule == NoiseSchedule.QUADRATIC:
            # Quadratic schedule
            betas = (
                np.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_timesteps) ** 2
            )
        else:
            betas = np.linspace(self.beta_start, self.beta_end, self.num_timesteps)

        return betas.astype(np.float32)


class SimpleUNet(nn.Module):
    """
    Simple UNet model for diffusion.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 128,
        num_blocks: int = 3,
        attention_resolutions: Tuple[int, ...] = (16, 8),
    ) -> None:
        """
        Initialize UNet.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            channels: Base number of channels
            num_blocks: Number of residual blocks
            attention_resolutions: Resolutions to apply attention
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_blocks = num_blocks

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

        # Encoder
        self.encoder = nn.ModuleList()
        in_ch = in_channels
        h = 256
        for _ in range(num_blocks):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, channels, 3, padding=1),
                    nn.GroupNorm(8, channels),
                    nn.SiLU(),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.GroupNorm(8, channels),
                    nn.SiLU(),
                )
            )
            in_ch = channels

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.GroupNorm(8, channels * 2),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )

        # Decoder
        self.decoder = nn.ModuleList()
        in_ch = channels
        for _ in range(num_blocks):
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch * 2, channels, 3, padding=1),
                    nn.GroupNorm(8, channels),
                    nn.SiLU(),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.GroupNorm(8, channels),
                    nn.SiLU(),
                )
            )
            in_ch = channels

        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels + in_channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 1),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width)
            t: Time step tensor of shape (batch,)

        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        # Time embedding
        t_emb = self.time_embed(t.float().unsqueeze(-1) / 1000.0)  # (batch, channels)
        t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1)  # (batch, channels, 1, 1)

        # Store input for skip connections
        h_input = x
        h = x

        # Encoder
        encoder_outputs = []
        for encoder_block in self.encoder:
            h = encoder_block(h)
            encoder_outputs.append(h)
            h = F.avg_pool2d(h, 2)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        for decoder_block in self.decoder:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = torch.cat([h, encoder_outputs.pop()], dim=1)
            h = decoder_block(h)

        # Final output with skip connection
        h = torch.cat([h, h_input], dim=1)
        h = self.final_conv(h)

        return h


class DiffusionModel(nn.Module):
    """
    Complete diffusion model for negative space reconstruction.
    """

    def __init__(
        self,
        config: DiffusionConfig,
        unet_channels: int = 128,
        device: str = "cuda",
    ) -> None:
        """
        Initialize diffusion model.

        Args:
            config: Diffusion configuration
            unet_channels: Base channels for UNet
            device: Device to use
        """
        super().__init__()
        self.config = config
        self.device = torch.device(device)

        # UNet for noise prediction
        self.unet = SimpleUNet(
            in_channels=config.channels,
            out_channels=config.channels,
            channels=unet_channels,
        ).to(self.device)

        # Convert numpy arrays to tensors
        self.register_buffer(
            "betas",
            torch.from_numpy(config.betas).to(self.device),
        )
        self.register_buffer(
            "alphas_cumprod",
            torch.from_numpy(config.alphas_cumprod).to(self.device),
        )
        self.register_buffer(
            "alphas_cumprod_prev",
            torch.from_numpy(config.alphas_cumprod_prev).to(self.device),
        )
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(torch.from_numpy(config.alphas_cumprod)).to(self.device),
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - torch.from_numpy(config.alphas_cumprod)).to(self.device),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass: predict noise.

        Args:
            x: Noisy input tensor
            t: Time step tensor

        Returns:
            Predicted noise
        """
        return self.unet(x, t)

    def diffuse(
        self,
        x0: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward diffusion process: add noise to image.

        Args:
            x0: Original image tensor of shape (batch, channels, height, width)
            t: Time step tensor of shape (batch,)

        Returns:
            Noisy image and noise tensors
        """
        sqrt_alphas_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        sqrt_alphas_t = sqrt_alphas_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_t = sqrt_one_minus_alphas_t.view(-1, 1, 1, 1)

        # Sample random noise
        noise = torch.randn_like(x0).to(self.device)

        # Add noise: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        x_t = sqrt_alphas_t * x0 + sqrt_one_minus_alphas_t * noise

        return x_t, noise

    def denoise(
        self,
        x_t: Tensor,
        t: Tensor,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """
        Single denoising step.

        Args:
            x_t: Current noisy image
            t: Current time step
            guidance_scale: Guidance scale for conditional generation

        Returns:
            Denoised image for previous timestep
        """
        # Predict noise
        noise_pred = self.forward(x_t, t)

        # Compute mean
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_t_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)

        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        sqrt_one_minus_alpha_t_prev = torch.sqrt(1.0 - alpha_t_prev)

        # Compute predicted x_0
        x_0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / torch.sqrt(alpha_t)

        # Clip x_0_pred to valid range
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # Posterior mean
        posterior_mean = (
            sqrt_alpha_t_prev * beta_t / (1.0 - alpha_t) * x_0_pred +
            torch.sqrt(1.0 - beta_t * (1.0 - alpha_t_prev) / (1.0 - alpha_t)) * (x_t - sqrt_one_minus_alpha_t * noise_pred) / torch.sqrt(alpha_t)
        )

        return posterior_mean

    def sample(
        self,
        num_samples: int = 1,
        image_size: Optional[int] = None,
        num_steps: int = 50,
        sampling_strategy: SamplingStrategy = SamplingStrategy.DETERMINISTIC,
    ) -> Tensor:
        """
        Generate samples from noise.

        Args:
            num_samples: Number of samples to generate
            image_size: Image size (if different from config)
            num_steps: Number of denoising steps
            sampling_strategy: Sampling strategy

        Returns:
            Generated images tensor
        """
        image_size = image_size or self.config.image_size

        # Start from pure noise
        x_t = torch.randn(
            num_samples,
            self.config.channels,
            image_size,
            image_size,
            device=self.device,
        )

        # Denoising steps
        timesteps = torch.linspace(
            self.config.num_timesteps - 1,
            0,
            num_steps,
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=self.device)

                if sampling_strategy == SamplingStrategy.DETERMINISTIC:
                    x_t = self.denoise(x_t, t_tensor)
                elif sampling_strategy == SamplingStrategy.STOCHASTIC:
                    # Add noise for stochastic sampling
                    x_t = self.denoise(x_t, t_tensor)
                    if t > 0:
                        noise = torch.randn_like(x_t) * torch.sqrt(self.betas[t])
                        x_t = x_t + noise

        # Clip to valid range
        x_t = torch.clamp(x_t, -1.0, 1.0)

        return x_t

    def reconstruct(
        self,
        degraded_image: Tensor,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """
        Reconstruct degraded image using diffusion.

        Args:
            degraded_image: Input degraded image
            num_steps: Number of reconstruction steps
            guidance_scale: Guidance scale

        Returns:
            Reconstructed image
        """
        # Forward diffusion to get initial noisy state
        t_0 = torch.tensor([self.config.num_timesteps - 1] * degraded_image.shape[0])
        x_t, _ = self.diffuse(degraded_image, t_0)

        # Reverse diffusion
        timesteps = torch.linspace(
            self.config.num_timesteps - 1,
            0,
            num_steps,
            dtype=torch.long,
        )

        with torch.no_grad():
            for t in timesteps:
                t_tensor = torch.full(
                    (degraded_image.shape[0],),
                    t,
                    dtype=torch.long,
                    device=self.device,
                )
                x_t = self.denoise(x_t, t_tensor, guidance_scale=guidance_scale)

        return torch.clamp(x_t, -1.0, 1.0)

    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get model configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "num_timesteps": self.config.num_timesteps,
            "noise_schedule": self.config.noise_schedule.value,
            "beta_start": float(self.config.beta_start),
            "beta_end": float(self.config.beta_end),
            "image_size": self.config.image_size,
            "channels": self.config.channels,
        }

    def save_model(self, path: str) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save model
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.get_config_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved model checkpoint to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model checkpoint from {path}")


class DiffusionFactory:
    """Factory for creating diffusion models."""

    @staticmethod
    def create_model(
        num_timesteps: int = 1000,
        noise_schedule: NoiseSchedule = NoiseSchedule.LINEAR,
        image_size: int = 256,
        channels: int = 3,
        unet_channels: int = 128,
        device: str = "cuda",
    ) -> DiffusionModel:
        """
        Create a diffusion model.

        Args:
            num_timesteps: Number of diffusion timesteps
            noise_schedule: Noise schedule type
            image_size: Image size
            channels: Number of channels
            unet_channels: Base UNet channels
            device: Device to use

        Returns:
            Initialized diffusion model
        """
        config = DiffusionConfig(
            num_timesteps=num_timesteps,
            noise_schedule=noise_schedule,
            image_size=image_size,
            channels=channels,
        )

        model = DiffusionModel(
            config=config,
            unet_channels=unet_channels,
            device=device,
        )

        return model

    @staticmethod
    def create_model_fast(
        image_size: int = 256,
        device: str = "cuda",
    ) -> DiffusionModel:
        """
        Create fast diffusion model (fewer timesteps).

        Args:
            image_size: Image size
            device: Device to use

        Returns:
            Fast diffusion model
        """
        return DiffusionFactory.create_model(
            num_timesteps=100,
            noise_schedule=NoiseSchedule.LINEAR,
            image_size=image_size,
            channels=3,
            unet_channels=64,
            device=device,
        )

    @staticmethod
    def create_model_high_quality(
        image_size: int = 256,
        device: str = "cuda",
    ) -> DiffusionModel:
        """
        Create high-quality diffusion model (more timesteps).

        Args:
            image_size: Image size
            device: Device to use

        Returns:
            High-quality diffusion model
        """
        return DiffusionFactory.create_model(
            num_timesteps=1000,
            noise_schedule=NoiseSchedule.COSINE,
            image_size=image_size,
            channels=3,
            unet_channels=256,
            device=device,
        )
