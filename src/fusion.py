"""Spatial Feature Fusion Module for Multi-Branch Document Forensics.

Combines spatial representations from RGB, DCT, and SRM branches through
channel projection, concatenation, and convolutional feature interaction blocks.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFeatureFusion(nn.Module):
    """Spatial Feature Fusion Module.

    Dimensions:
    - Input: List of active branch feature maps, each of shape [B, C_in, 64, 64]
    - Projection: 1x1 Conv + BN + ReLU to project each active branch to proj_channels
    - Concatenation: [B, N_active * proj_channels, 64, 64]
    - Fusion Block: Two 3x3 Conv layers with residual skip connection
    - Output: [B, fused_channels, 64, 64]
    """

    def __init__(
        self,
        branch_channels: List[int],
        fused_channels: int = 128,
    ):
        """Args:

        branch_channels: List of channel counts for each expected active branch.
        fused_channels: Desired output channel dimension of the fused feature
        map.
        """
        super().__init__()
        self.num_branches = len(branch_channels)
        self.fused_channels = fused_channels

        # Individual 1x1 projection for each incoming branch
        proj_dim = fused_channels
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, proj_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
            )
            for in_c in branch_channels
        ])

        total_cat_dim = self.num_branches * proj_dim

        # Fusion convolution block
        self.fusion_block = nn.Sequential(
            nn.Conv2d(total_cat_dim, fused_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )

        # Residual shortcut if concatenated channels != fused_channels
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(total_cat_dim, fused_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fused_channels),
            )
            if total_cat_dim != fused_channels
            else nn.Identity()
        )

        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        """Args:

        feature_list: List of tensors [B, C_i, 64, 64] corresponding to each
        active branch.

        Returns:
            [B, fused_channels, 64, 64] Fused spatial feature map
        """
        if len(feature_list) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} feature maps, got {len(feature_list)}"
            )

        projected = [proj(feat) for proj, feat in zip(self.projections, feature_list)]

        cat_feat = torch.cat(projected, dim=1)  # [B, N*proj_dim, 64, 64]
        out = self.fusion_block(cat_feat) + self.shortcut(cat_feat)
        return self.final_relu(out)  # [B, fused_channels, 64, 64]
