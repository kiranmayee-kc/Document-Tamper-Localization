"""TamperDoc Model Architecture.

Modular multi-branch neural network for document tamper localization combining:
- Pretrained ResNet-18 RGB branch
- Block-wise 2D-DCT frequency branch
- Fixed SRM high-pass residual noise branch
- Spatial feature fusion
- Progressive localization decoder
"""

from typing import Dict, Optional
import torch
import torch.nn as nn

from src.rgb_branch import RGBBranch
from src.dct_branch import DCTBranch
from src.srm_branch import SRMBranch
from src.fusion import SpatialFeatureFusion
from src.localization import LocalizationDecoder

"""TamperDoc Multi-Branch Document Tamper Localization Model.

    Ablation Configs:
    - Experiment A: rgb=True,  dct=False, srm=False (RGB Only)
    - Experiment B: rgb=True,  dct=True,  srm=False (RGB + DCT)
    - Experiment C: rgb=True,  dct=False, srm=True  (RGB + SRM)
    - Experiment D: rgb=True,  dct=True,  srm=True  (Full Multi-Modal)
    """


class TamperDocModel(nn.Module):
    
    def __init__(
        self,
        use_rgb: bool = True,
        use_dct: bool = True,
        use_srm: bool = True,
        pretrained_rgb: bool = True,
        branch_channels: int = 128,
        fused_channels: int = 128,
        target_size: tuple = (512, 512),
    ):
        super().__init__()
        if not any([use_rgb, use_dct, use_srm]):
            raise ValueError("At least one branch (RGB, DCT, or SRM) must be enabled.")

        self.use_rgb = use_rgb
        self.use_dct = use_dct
        self.use_srm = use_srm
        self.target_size = target_size

        # 1. RGB Branch
        if self.use_rgb:
            self.rgb_branch = RGBBranch(
                pretrained=pretrained_rgb,
                out_channels=branch_channels,
            )
        else:
            self.rgb_branch = None

        # 2. DCT Branch
        if self.use_dct:
            self.dct_branch = DCTBranch(
                out_channels=branch_channels,
            )
        else:
            self.dct_branch = None

        # 3. SRM Branch
        if self.use_srm:
            self.srm_branch = SRMBranch(
                out_channels=branch_channels,
            )
        else:
            self.srm_branch = None

        # Active branch channel list
        active_channels = []
        if self.use_rgb:
            active_channels.append(branch_channels)
        if self.use_dct:
            active_channels.append(branch_channels)
        if self.use_srm:
            active_channels.append(branch_channels)

        # 4. Spatial Feature Fusion
        self.fusion = SpatialFeatureFusion(
            branch_channels=active_channels,
            fused_channels=fused_channels,
        )

        # 5. Localization Decoder
        self.decoder = LocalizationDecoder(
            in_channels=fused_channels,
            target_size=target_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        raw_x: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:

        x: [B, 3, 512, 512] RGB image tensor (ImageNet normalized)
        raw_x: [B, 3, 512, 512] optional unnormalized image in [0, 1] for DCT /
        SRM

        Returns:
            Dict containing:
                "tamper_logits": [B, 1, 512, 512] raw logits
                "fused_features": [B, fused_channels, 64, 64]
                "branch_features": dict of active branch representations
        """
        # Default raw_x to x if not provided
        img_for_physics = raw_x if raw_x is not None else x

        active_feats = []
        branch_feats = {}

        # 1. RGB Features
        if self.use_rgb:
            rgb_f = self.rgb_branch(x)
            active_feats.append(rgb_f)
            branch_feats["rgb"] = rgb_f

        # 2. DCT Features
        if self.use_dct:
            dct_f = self.dct_branch(img_for_physics)
            active_feats.append(dct_f)
            branch_feats["dct"] = dct_f

        # 3. SRM Features
        if self.use_srm:
            srm_f = self.srm_branch(img_for_physics)
            active_feats.append(srm_f)
            branch_feats["srm"] = srm_f

        # 4. Fusion
        fused = self.fusion(active_feats)

        # 5. Localization Decoder
        logits = self.decoder(fused)

        return {
            "tamper_logits": logits,
            "fused_features": fused,
            "branch_features": branch_feats,
        }
