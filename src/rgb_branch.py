"""RGB Feature Extraction Branch using ResNet-18.

Extracts multi-scale spatial feature maps from document images using an
ImageNet-pretrained ResNet-18 backbone, maintaining high-resolution spatial
representations for tamper localization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class RGBBranch(nn.Module):
    """Multi-Scale Spatial RGB Feature Extractor based on ResNet-18.

    Spatial Resolution Flow for 512x512 input:
    - Input: [B, 3, 512, 512]
    - Stem (conv1 + maxpool): [B, 64, 128, 128]
    - Layer1: [B, 64, 128, 128]  (stride 1, 1/4 res)
    - Layer2: [B, 128, 64, 64]   (stride 2, 1/8 res)
    - Layer3: [B, 256, 32, 32]   (stride 2, 1/16 res)
    - Layer4: [B, 512, 16, 16]   (stride 2, 1/32 res)

    Multi-scale aggregation fuses Layer1-Layer4 into a unified spatial feature map:
    - Output Feature Shape: [B, out_channels, 64, 64] (stride 8 / 1/8 resolution)
    """

    def __init__(
        self,
        pretrained: bool = True,
        out_channels: int = 128,
    ):
        super().__init__()
        self.out_channels = out_channels

        # Load ResNet-18
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                resnet = models.resnet18(weights=weights)
            except Exception:
                # Fallback if network download unavailable
                resnet = models.resnet18(weights=None)
        else:
            resnet = models.resnet18(weights=None)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels, 128x128
        self.layer2 = resnet.layer2  # 128 channels, 64x64
        self.layer3 = resnet.layer3  # 256 channels, 32x32
        self.layer4 = resnet.layer4  # 512 channels, 16x16

        # Lateral projections to target 64x64 spatial resolution
        self.lat_l1 = nn.Sequential(
            nn.Conv2d(64, out_channels // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
        )  # 128x128 -> 64x64

        self.lat_l2 = nn.Sequential(
            nn.Conv2d(128, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
        )  # 64x64 -> 64x64

        self.lat_l3 = nn.Sequential(
            nn.Conv2d(256, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
        )  # 32x32 -> 64x64 (via upsample)

        self.lat_l4 = nn.Sequential(
            nn.Conv2d(512, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
        )  # 16x16 -> 64x64 (via upsample)

        # Refinement convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: [B, 3, 512, 512] RGB image tensor (ImageNet normalized)

        Returns:
            [B, out_channels, 64, 64] Spatial feature map
        """
        # Backbone forward
        x0 = self.relu(self.bn1(self.conv1(x)))  # [B, 64, 256, 256]
        x1 = self.layer1(self.maxpool(x0))       # [B, 64, 128, 128]
        x2 = self.layer2(x1)                     # [B, 128, 64, 64]
        x3 = self.layer3(x2)                     # [B, 256, 32, 32]
        x4 = self.layer4(x3)                     # [B, 512, 16, 16]

        # Multi-scale aggregation at 64x64 resolution
        f1 = self.lat_l1(x1)  # [B, out_channels//4, 64, 64]
        f2 = self.lat_l2(x2)  # [B, out_channels//4, 64, 64]
        f3 = F.interpolate(self.lat_l3(x3), size=(64, 64), mode="bilinear", align_corners=False)
        f4 = F.interpolate(self.lat_l4(x4), size=(64, 64), mode="bilinear", align_corners=False)

        # Concat along channel dimension: 4 * (out_channels//4) = out_channels
        f_cat = torch.cat([f1, f2, f3, f4], dim=1)  # [B, out_channels, 64, 64]
        out = self.fusion_conv(f_cat)               # [B, out_channels, 64, 64]

        return out
