"""Evaluation Metrics for Pixel-Level Document Tamper Localization.

Computes Precision, Recall, F1 / Dice Score, and Intersection-over-Union (IoU)
with support for both cumulative dataset-wide tracking and single-batch scoring.
"""

from typing import Dict
import numpy as np
import torch


class LocalizationMetrics:
    """Computes and accumulates pixel-level binary segmentation metrics.

    Metrics:
    - Precision = TP / (TP + FP + eps)
    - Recall    = TP / (TP + FN + eps)
    - F1 / Dice = 2 * TP / (2 * TP + FP + FN + eps)
    - IoU       = TP / (TP + FP + FN + eps)
    """

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        """Reset running accumulation counters."""
        self.total_tp = 0.0
        self.total_fp = 0.0
        self.total_fn = 0.0
        self.total_tn = 0.0
        self.sample_count = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update cumulative counts from a batch of predictions and ground truths.

        Args:
            logits: [B, 1, H, W] model output logits
            targets: [B, 1, H, W] ground-truth binary masks {0.0, 1.0}
        """
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).to(torch.float32)
        targets = (targets > 0.0).to(torch.float32)

        tp = (preds * targets).sum().item()
        fp = (preds * (1.0 - targets)).sum().item()
        fn = ((1.0 - preds) * targets).sum().item()
        tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_tn += tn
        self.sample_count += targets.size(0)

    def compute(self) -> Dict[str, float]:
        """Compute dataset-wide metrics from accumulated totals.

        Returns:
            Dict containing 'precision', 'recall', 'f1', 'iou', and 'pixel_accuracy'
        """
        tp = self.total_tp
        fp = self.total_fp
        fn = self.total_fn
        tn = self.total_tn

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn + self.eps)
        iou = tp / (tp + fp + fn + self.eps)
        pixel_acc = (tp + tn) / (tp + tn + fp + fn + self.eps)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "pixel_accuracy": float(pixel_acc),
            "samples_evaluated": int(self.sample_count),
        }

    @staticmethod
    def batch_metrics(
        logits: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-7,
    ) -> Dict[str, float]:
        """Convenience method to compute metrics for a single batch."""
        meter = LocalizationMetrics(threshold=threshold, eps=eps)
        meter.update(logits, targets)
        return meter.compute()
