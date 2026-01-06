"""
UNet++ (Nested U-Net) Architecture for Medical Image Segmentation
=================================================================

This module implements the UNet++ architecture, an advanced variant of the U-Net
designed for superior medical image segmentation performance. UNet++ introduces
dense skip connections and deep supervision to capture fine details in medical
imaging, particularly suited for "negative space" analysis in CT/MRI scans.

Key Features:
- Dense skip connections for better feature propagation
- Deep supervision for improved training stability
- Nested convolutional blocks for multi-scale feature extraction
- Optimized for medical imaging applications

Author: Negative Space Imaging Project Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """
    Basic convolutional block with batch normalization and ReLU activation.

    This block performs two consecutive 3x3 convolutions followed by batch
    normalization and ReLU activation, which is the standard building block
    in U-Net architectures.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the convolutional block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetPlusPlus(nn.Module):
    """
    UNet++ (Nested U-Net) architecture for medical image segmentation.

    UNet++ extends the traditional U-Net by introducing dense skip connections
    and deep supervision. This allows for better feature propagation across
    different scales and more stable training for medical imaging tasks.

    The architecture consists of an encoder-decoder structure with nested
    convolutional blocks and dense skip pathways between corresponding levels.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512, 1024],
        deep_supervision: bool = True
    ) -> None:
        """
        Initialize the UNet++ model.

        Args:
            in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
            out_channels: Number of output channels (e.g., 1 for binary segmentation)
            features: List of feature channels for each level of the network
            deep_supervision: Whether to use deep supervision during training
        """
        super(UNetPlusPlus, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.deep_supervision = deep_supervision
        self.depth = len(features) - 1

        # Encoder blocks (X_0,0 to X_0,4)
        self.encoder_blocks = nn.ModuleList()
        for i, feature in enumerate(features):
            if i == 0:
                self.encoder_blocks.append(
                    ConvBlock(in_channels, feature)
                )
            else:
                self.encoder_blocks.append(
                    ConvBlock(features[i-1], feature)
                )

        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder blocks with dense skip connections
        self.decoder_blocks = nn.ModuleList()

        # Create nested decoder blocks
        for level in range(self.depth):
            level_blocks = nn.ModuleList()
            for sub_level in range(level + 1):
                if sub_level == 0:
                    # First block in each level connects to encoder
                    in_ch = features[level + 1] + features[level]
                else:
                    # Subsequent blocks connect to previous sub-levels
                    in_ch = features[level] * 2

                level_blocks.append(
                    ConvBlock(in_ch, features[level])
                )
            self.decoder_blocks.append(level_blocks)

        # Upsampling layers
        self.upsample = nn.ModuleList()
        for i in range(self.depth):
            self.upsample.append(
                nn.ConvTranspose2d(
                    features[i], features[i], kernel_size=2, stride=2
                )
            )

        # Deep supervision heads (optional)
        if self.deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
            for i in range(self.depth):
                self.deep_supervision_heads.append(
                    nn.Conv2d(features[i], out_channels, kernel_size=1)
                )

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        logger.info(f"Initialized UNet++ with depth {self.depth}, features {features}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the UNet++ network.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Dictionary containing:
            - 'output': Final segmentation mask
            - 'deep_outputs': List of intermediate outputs for deep supervision (if enabled)
        """
        try:
            # Encoder path
            encoder_outputs = []
            current = x

            for i, block in enumerate(self.encoder_blocks):
                current = block(current)
                encoder_outputs.append(current)
                if i < len(self.encoder_blocks) - 1:
                    current = self.pool(current)

            # Nested decoder with dense skip connections
            decoder_outputs = [encoder_outputs]

            for level in range(self.depth):
                level_outputs = []

                for sub_level in range(level + 1):
                    if sub_level == 0:
                        # Connect to encoder output
                        upsampled = self.upsample[level](encoder_outputs[level + 1])
                        # Concatenate with corresponding encoder output
                        concat_input = torch.cat([
                            upsampled,
                            encoder_outputs[level]
                        ], dim=1)
                    else:
                        # Connect to previous sub-level output
                        upsampled = self.upsample[level](level_outputs[sub_level - 1])
                        # Concatenate with all previous sub-levels in this level
                        concat_tensors = [upsampled] + [
                            decoder_outputs[level][i] for i in range(sub_level)
                        ]
                        concat_input = torch.cat(concat_tensors, dim=1)

                    # Apply convolutional block
                    output = self.decoder_blocks[level][sub_level](concat_input)
                    level_outputs.append(output)

                decoder_outputs.append(level_outputs)

            # Deep supervision outputs
            deep_outputs = []
            if self.deep_supervision:
                for i in range(self.depth):
                    deep_output = self.deep_supervision_heads[i](
                        decoder_outputs[i + 1][-1]
                    )
                    # Upsample to original size
                    deep_output = F.interpolate(
                        deep_output, size=x.shape[2:], mode='bilinear', align_corners=False
                    )
                    deep_outputs.append(deep_output)

            # Final output
            final_output = self.final_conv(decoder_outputs[-1][-1])
            final_output = F.interpolate(
                final_output, size=x.shape[2:], mode='bilinear', align_corners=False
            )

            result = {
                'output': final_output,
                'deep_outputs': deep_outputs if self.deep_supervision else []
            }

            return result

        except Exception as e:
            logger.error(f"Error in UNet++ forward pass: {e}")
            raise RuntimeError(f"UNet++ forward pass failed: {str(e)}") from e


class SegmentationUNetPlusPlus:
    """
    High-level interface for UNet++ segmentation model.

    This class provides a complete interface for building, training, and using
    the UNet++ model for medical image segmentation tasks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Optional[List[int]] = None,
        device: str = 'auto'
    ) -> None:
        """
        Initialize the segmentation model.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: Feature channels for each level (default: [64, 128, 256, 512, 1024])
            device: Device to run model on ('auto', 'cpu', 'cuda')
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features or [64, 128, 256, 512, 1024]

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[UNetPlusPlus] = None
        self.criterion = nn.BCEWithLogitsLoss() if out_channels == 1 else nn.CrossEntropyLoss()
        self.optimizer: Optional[torch.optim.Optimizer] = None

        logger.info(f"Initialized SegmentationUNetPlusPlus on device: {self.device}")

    def build_model(self) -> UNetPlusPlus:
        """
        Build and initialize the UNet++ model.

        Returns:
            The initialized UNet++ model
        """
        try:
            self.model = UNetPlusPlus(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                features=self.features
            ).to(self.device)

            # Initialize optimizer
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=1e-5
            )

            logger.info("Successfully built UNet++ model")
            return self.model

        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise RuntimeError(f"Model building failed: {str(e)}") from e

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        deep_supervision_weight: float = 0.5
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            inputs: Input images of shape (batch_size, in_channels, height, width)
            targets: Target masks of shape (batch_size, out_channels, height, width)
            deep_supervision_weight: Weight for deep supervision loss

        Returns:
            Dictionary containing loss values
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        try:
            self.model.train()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            main_loss = self.criterion(outputs['output'], targets)

            # Deep supervision loss
            deep_loss = 0.0
            if outputs['deep_outputs']:
                for deep_output in outputs['deep_outputs']:
                    deep_loss += self.criterion(deep_output, targets)
                deep_loss /= len(outputs['deep_outputs'])

            total_loss = main_loss + deep_supervision_weight * deep_loss

            total_loss.backward()
            self.optimizer.step()

            return {
                'total_loss': total_loss.item(),
                'main_loss': main_loss.item(),
                'deep_loss': deep_loss.item() if outputs['deep_outputs'] else 0.0
            }

        except Exception as e:
            logger.error(f"Training step failed: {e}")
            raise RuntimeError(f"Training step failed: {str(e)}") from e

    def predict_mask(
        self,
        inputs: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Predict segmentation masks for input images.

        Args:
            inputs: Input images of shape (batch_size, in_channels, height, width)
            threshold: Threshold for binary segmentation (only used for binary tasks)

        Returns:
            Predicted masks of shape (batch_size, out_channels, height, width)
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        try:
            self.model.eval()
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                masks = torch.sigmoid(outputs['output']) if self.out_channels == 1 else outputs['output']

                if self.out_channels == 1 and threshold is not None:
                    masks = (masks > threshold).float()

            return masks.cpu()

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}") from e

    def save_model(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'config': {
                    'in_channels': self.in_channels,
                    'out_channels': self.out_channels,
                    'features': self.features
                }
            }, path)
            logger.info(f"Model saved to {path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise RuntimeError(f"Model saving failed: {str(e)}") from e

    def load_model(self, path: str) -> None:
        """
        Load the model from disk.

        Args:
            path: Path to load the model from
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)

            if self.model is None:
                config = checkpoint.get('config', {})
                self.model = UNetPlusPlus(
                    in_channels=config.get('in_channels', self.in_channels),
                    out_channels=config.get('out_channels', self.out_channels),
                    features=config.get('features', self.features)
                ).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])

            if self.optimizer and checkpoint.get('optimizer_state_dict'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logger.info(f"Model loaded from {path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e


# Convenience function for creating a medical imaging segmentation model
def create_medical_segmentation_model(
    in_channels: int = 1,
    device: str = 'auto'
) -> SegmentationUNetPlusPlus:
    """
    Create a UNet++ model optimized for medical image segmentation.

    Args:
        in_channels: Number of input channels (1 for CT/MRI, 3 for RGB)
        device: Device to run the model on

    Returns:
        Configured SegmentationUNetPlusPlus instance
    """
    return SegmentationUNetPlusPlus(
        in_channels=in_channels,
        out_channels=1,  # Binary segmentation for medical imaging
        features=[64, 128, 256, 512, 1024],  # Standard UNet++ configuration
        device=device
    )</content>
<parameter name="filePath">c:\Users\sgbil\Negative_Space_Imaging_Project\segmentation_unet_plus.py
