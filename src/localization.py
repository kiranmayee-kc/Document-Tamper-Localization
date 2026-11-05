"""Localization Decoder for Pixel-Level Document Tamper Mask Prediction.

Decodes fused spatial feature maps into high-resolution single-channel tamper
logits matching the ground-truth mask dimensions (512x512).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Upsampling + Refinement Block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        x_up = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.conv(x_up)


class LocalizationDecoder(nn.Module):
    """Progressive Localization Decoder.

    Dimensions:
    - Input: [B, in_channels, 64, 64]
    - Stage 1 (64x64 -> 128x128): in_channels -> 64
    - Stage 2 (128x128 -> 256x256): 64 -> 32
    - Stage 3 (256x256 -> 512x512): 32 -> 16
    - Head (1x1 Conv): 16 -> 1
    - Output: [B, 1, 512, 512] raw tamper logits (unbounded)
    """

    def __init__(
        self,
        in_channels: int = 128,
        target_size: tuple = (512, 512),
    ):
        super().__init__()
        self.target_size = target_size

        self.up1 = DecoderBlock(in_channels, 64)    # -> 128x128
        self.up2 = DecoderBlock(64, 32)             # -> 256x256
        self.up3 = DecoderBlock(32, 16)             # -> 512x512

        self.head = nn.Conv2d(16, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: [B, in_channels, 64, 64] fused feature map

        Returns:
            [B, 1, 512, 512] raw tamper logits
        """
        b, c, h, w = x.shape

        x1 = self.up1(x, (h * 2, w * 2))     # [B, 64, 128, 128]
        x2 = self.up2(x1, (h * 4, w * 4))    # [B, 32, 256, 256]
        x3 = self.up3(x2, self.target_size)  # [B, 16, 512, 512]

        logits = self.head(x3)               # [B, 1, 512, 512]
        return logits
