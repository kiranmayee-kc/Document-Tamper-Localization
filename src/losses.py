"""Loss Functions for Document Tamper Localization.

Combines Binary Cross Entropy with Logits and Soft Dice Loss for pixel-level
segmentation of forged regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation.

    Computes:
        Dice = 1 - (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:

        logits: [B, 1, H, W] unbounded model predictions
        targets: [B, 1, H, W] binary ground truth {0.0, 1.0}

        Returns:
            Scalar tensor representing Dice loss
        """
        probs = torch.sigmoid(logits)

        # Flatten across spatial dimensions per batch sample
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()


class CombinedLocalizationLoss(nn.Module):
    """Combined BCE + Dice Loss for Tamper Localization.

    Formula:
        Loss = BCEWithLogits(logits, targets) + lambda_dice * SoftDice(logits, targets)
    """

    def __init__(
        self,
        lambda_dice: float = 1.0,
        pos_weight: float = None,
        smooth_dice: float = 1.0,
    ):
        """Args:

        lambda_dice: Weight multiplier for the Dice loss component (default:
        1.0).
        pos_weight: Optional positive class weighting for BCE to balance sparse
        tamper regions.
        smooth_dice: Smoothing coefficient for Dice loss denominator.
        """
        super().__init__()
        self.lambda_dice = lambda_dice
        pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.dice = SoftDiceLoss(smooth=smooth_dice)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """Args:

        logits: [B, 1, H, W] model output
        targets: [B, 1, H, W] binary float mask

        Returns:
            Dict with 'total_loss', 'bce_loss', and 'dice_loss'
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = bce_loss + self.lambda_dice * dice_loss

        return {
            "loss": total_loss,
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
        }
