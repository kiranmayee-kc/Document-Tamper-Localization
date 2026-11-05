"""SRM (Spatial Rich Model) Noise Residual Feature Extraction Branch.

Applies fixed high-pass / steganalysis residual filters to isolate manipulation
noise and local edge inconsistencies, followed by a lightweight learnable CNN
encoder producing spatial feature maps aligned with other branches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_srm_filter_bank() -> torch.Tensor:
    """Creates standard 3x3 Spatial Rich Model (SRM) high-pass residual filter kernels.

    Uses 3 established SRM kernels (Fridrich & Kodovsky, 2012):
    1. 1st order / cross residual: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]] / 4.0
    2. 2nd order / edge residual:  [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]] / 4.0
    3. Square / point residual:    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]] / 8.0

    Applied across 3 RGB channels independently -> 3 filters * 3 channels = 9 residual channels.

    Returns:
        torch.Tensor of shape [9, 1, 3, 3] for grouped convolution or [9, 3, 3, 3].
    """
    k1 = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]) / 4.0
    k2 = torch.tensor([[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]]) / 4.0
    k3 = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]) / 8.0

    base_kernels = [k1, k2, k3]  # 3 kernels

    # Build 9x3x3x3 weights: each of the 3 kernels applied to each of R, G, B channels
    weights = torch.zeros(9, 3, 3, 3, dtype=torch.float32)
    out_idx = 0
    for k in base_kernels:
        for c in range(3):
            weights[out_idx, c, :, :] = k
            out_idx += 1

    return weights


class SRMBranch(nn.Module):
    """SRM Noise Residual Branch.

    Dimensions:
    - Input: [B, 3, 512, 512] RGB image tensor
    - Fixed SRM High-Pass Filter Bank: 9 output residual maps at [B, 9, 512, 512]
    - Learnable CNN Downsampling Encoder:
      - Stage 1: Conv 9 -> 32, stride 2, BN, ReLU -> [B, 32, 256, 256]
      - Stage 2: Conv 32 -> 64, stride 2, BN, ReLU -> [B, 64, 128, 128]
      - Stage 3: Conv 64 -> out_channels, stride 2, BN, ReLU -> [B, out_channels, 64, 64]
      - Stage 4: Residual Conv refinement block -> [B, out_channels, 64, 64]
    - Output Feature Map: [B, out_channels, 64, 64]
    """

    def __init__(self, out_channels: int = 128):
        super().__init__()
        self.out_channels = out_channels

        # Fixed, non-learnable SRM High-Pass Conv2d layer
        self.srm_conv = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1, bias=False)
        srm_weights = create_srm_filter_bank()
        self.srm_conv.weight = nn.Parameter(srm_weights, requires_grad=False)

        # Learnable CNN Encoder (matches 64x64 spatial resolution)
        self.encoder = nn.Sequential(
            # 512x512 -> 256x256
            nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 256x256 -> 128x128
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 128x128 -> 64x64
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Refinement block at 64x64
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def extract_residuals(self, x: torch.Tensor) -> torch.Tensor:
        """Extract high-pass noise residuals using fixed SRM filters."""
        residuals = self.srm_conv(x)  # [B, 9, 512, 512]
        return residuals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: [B, 3, 512, 512] RGB image tensor

        Returns:
            [B, out_channels, 64, 64] Spatial feature map
        """
        residuals = self.extract_residuals(x)
        features = self.encoder(residuals)  # [B, out_channels, 64, 64]
        return features
