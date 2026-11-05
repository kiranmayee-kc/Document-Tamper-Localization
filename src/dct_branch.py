"""DCT (Discrete Cosine Transform) Frequency Feature Extraction Branch.

Applies a deterministic, block-wise 2D DCT-II (8x8 non-overlapping blocks) across
the image luminance channel, followed by a lightweight learnable CNN encoder to
capture localized frequency artifacts caused by document tampering and compression
inconsistencies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_dct8x8_filter_bank() -> torch.Tensor:
    """Generates the orthogonal 2D DCT-II basis filter bank for 8x8 blocks.

    Returns:
        torch.Tensor of shape [64, 1, 8, 8] containing the 64 orthonormal DCT basis filters.
    """
    filters = torch.zeros(64, 1, 8, 8, dtype=torch.float32)
    k = 0
    for u in range(8):
        alpha_u = 1.0 / math.sqrt(8.0) if u == 0 else math.sqrt(2.0 / 8.0)
        for v in range(8):
            alpha_v = 1.0 / math.sqrt(8.0) if v == 0 else math.sqrt(2.0 / 8.0)
            for i in range(8):
                for j in range(8):
                    val = (
                        alpha_u
                        * alpha_v
                        * math.cos((2.0 * i + 1.0) * u * math.pi / 16.0)
                        * math.cos((2.0 * j + 1.0) * v * math.pi / 16.0)
                    )
                    filters[k, 0, i, j] = val
            k += 1
    return filters


class DCTBranch(nn.Module):
    """DCT Frequency Branch.

    Dimensions:
    - Input: [B, 3, 512, 512] RGB image or [B, 1, 512, 512] Luminance
    - Luminance Conversion: Y = 0.299*R + 0.587*G + 0.114*B -> [B, 1, 512, 512]
    - Block DCT-II (8x8, stride 8): [B, 64, 64, 64]
      (64 frequency coefficients per 8x8 spatial block, 64x64 grid)
    - Learnable CNN Encoder:
      - 3 Residual Conv Blocks with BatchNorm and ReLU
    - Output Feature Map: [B, out_channels, 64, 64]
    """

    def __init__(self, out_channels: int = 128):
        super().__init__()
        self.out_channels = out_channels

        # Fixed, non-learnable 8x8 DCT Conv2d layer
        self.dct_conv = nn.Conv2d(1, 64, kernel_size=8, stride=8, padding=0, bias=False)
        dct_filters = create_dct8x8_filter_bank()
        self.dct_conv.weight = nn.Parameter(dct_filters, requires_grad=False)

        # Luminance weights for RGB input
        self.register_buffer(
            "rgb_to_y",
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1),
        )

        # Lightweight Learnable CNN Encoder
        # Input: 64 channels at 64x64 resolution
        self.encoder = nn.Sequential(
            # Layer 1: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Layer 2: Residual Block 128 -> 128
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Layer 3: Projection to out_channels
            nn.Conv2d(128, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def extract_dct(self, x: torch.Tensor) -> torch.Tensor:
        """Extract deterministic block-wise 2D DCT coefficients."""
        if x.shape[1] == 3:
            # Convert to Luminance (Y)
            y = (x * self.rgb_to_y).sum(dim=1, keepdim=True)
        else:
            y = x
        # Apply block DCT
        dct_coeff = self.dct_conv(y)  # [B, 64, 64, 64]
        return dct_coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: [B, 3, 512, 512] RGB image tensor (or raw image in [0, 1])

        Returns:
            [B, out_channels, 64, 64] Spatial feature map
        """
        dct_coeff = self.extract_dct(x)
        features = self.encoder(dct_coeff)  # [B, out_channels, 64, 64]
        return features
