"""Regional Ground-Truth Diagnostic for DocTamper-Synthesized Document.

Calculates precise probability distributions inside vs. outside the ground-truth
tampered text region for both the synthesized tampered document and authentic original.

Methodology:
- Exact model checkpoint (checkpoints/best_model.pt, Epoch 3)
- Exact inference preprocessing (ImageNet normalization for RGB, [0,1] for DCT/SRM)
- torch.inference_mode(), CPU
- Pixel-exact masking using results/demo_inputs/doctamper_synthesized_mask.png
- Compares global, in-GT, and out-of-GT distributions
- Analyzes sub-threshold response ladder (0.01, 0.02, 0.05, 0.10, 0.25, 0.50)
- Produces diagnostic visualization with GT-region zoom
"""

import json
import os
import sys
import time
from typing import Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from src.model import TamperDocModel

# Paths
CHECKPOINT_PATH = os.path.abspath("checkpoints/best_model.pt")
TAMPERED_PATH = os.path.abspath("results/demo_inputs/doctamper_synthesized_tampered.jpg")
ORIGINAL_PATH = os.path.abspath("results/demo_inputs/og current bill.jpg")
GT_MASK_PATH  = os.path.abspath("results/demo_inputs/doctamper_synthesized_mask.png")
OUTPUT_FIG_PATH = os.path.abspath("results/demo_synthesized/gt_region_diagnosis.png")

TARGET_SIZE = (512, 512)
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


def load_model(checkpoint_path: str) -> TamperDocModel:
    """Loads TamperDoc model with weights restored from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = TamperDocModel(
        use_rgb=True,
        use_dct=True,
        use_srm=True,
        pretrained_rgb=False,
        branch_channels=128,
        fused_channels=128,
        target_size=TARGET_SIZE,
    )
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def preprocess_image(image_path: str) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Loads and preprocesses image with standard model pipeline."""
    pil_img = Image.open(image_path).convert("RGB")
    if pil_img.size != TARGET_SIZE:
        raise ValueError(f"Image {image_path} size is {pil_img.size}, expected {TARGET_SIZE}")
    
    img_np = np.asarray(pil_img, dtype=np.float32) / 255.0
    raw_tensor = TF.to_tensor(pil_img)
    norm_tensor = TF.normalize(raw_tensor, mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    
    return img_np, norm_tensor.unsqueeze(0), raw_tensor.unsqueeze(0)


def compute_regional_stats(prob_map: np.ndarray, gt_mask: np.ndarray, label: str) -> dict:
    """Computes detailed statistical breakdown globally, inside GT, and outside GT."""
    total_pixels = prob_map.size
    gt_bool = (gt_mask > 0)
    bg_bool = (~gt_bool)

    gt_pixels = int(np.sum(gt_bool))
    bg_pixels = int(np.sum(bg_bool))

    # 1. Global
    g_min = float(np.min(prob_map))
    g_max = float(np.max(prob_map))
    g_mean = float(np.mean(prob_map))
    g_std = float(np.std(prob_map))
    g_p95 = float(np.percentile(prob_map, 95.0))
    g_p99 = float(np.percentile(prob_map, 99.0))
    g_p999 = float(np.percentile(prob_map, 99.9))

    # 2. Inside GT Region
    gt_probs = prob_map[gt_bool]
    if gt_pixels > 0:
        gt_min = float(np.min(gt_probs))
        gt_max = float(np.max(gt_probs))
        gt_mean = float(np.mean(gt_probs))
        gt_median = float(np.median(gt_probs))
        gt_std = float(np.std(gt_probs))
        gt_p95 = float(np.percentile(gt_probs, 95.0))
        gt_p99 = float(np.percentile(gt_probs, 99.0))
    else:
        gt_min = gt_max = gt_mean = gt_median = gt_std = gt_p95 = gt_p99 = 0.0

    # 3. Outside GT Region (Background / Authentic text)
    bg_probs = prob_map[bg_bool]
    bg_min = float(np.min(bg_probs))
    bg_max = float(np.max(bg_probs))
    bg_mean = float(np.mean(bg_probs))
    bg_std = float(np.std(bg_probs))
    bg_p95 = float(np.percentile(bg_probs, 95.0))
    bg_p99 = float(np.percentile(bg_probs, 99.0))

    # 4. Threshold Ladder inside GT Region
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50]
    gt_threshold_counts = {}
    for t in thresholds:
        cnt = int(np.sum(gt_probs >= t))
        pct = (cnt / gt_pixels * 100.0) if gt_pixels > 0 else 0.0
        gt_threshold_counts[f"ge_{str(t).replace('.', '_')}"] = {
            "threshold": t,
            "count": cnt,
            "percentage": pct,
        }

    return {
        "label": label,
        "total_pixels": total_pixels,
        "gt_pixel_count": gt_pixels,
        "bg_pixel_count": bg_pixels,
        "global": {
            "min": g_min,
            "max": g_max,
            "mean": g_mean,
            "std": g_std,
            "p95": g_p95,
            "p99": g_p99,
            "p99_9": g_p999,
        },
        "gt_region": {
            "min": gt_min,
            "max": gt_max,
            "mean": gt_mean,
            "median": gt_median,
            "std": gt_std,
            "p95": gt_p95,
            "p99": gt_p99,
            "threshold_ladder": gt_threshold_counts,
        },
        "outside_gt": {
            "min": bg_min,
            "max": bg_max,
            "mean": bg_mean,
            "std": bg_std,
            "p95": bg_p95,
            "p99": bg_p99,
        },
    }


def generate_diagnostic_visualization(
    orig_np: np.ndarray,
    tamp_np: np.ndarray,
    gt_mask: np.ndarray,
    tamp_prob: np.ndarray,
    orig_prob: np.ndarray,
    save_path: str,
) -> None:
    """Creates a 6-panel diagnostic figure with global and zoom views."""
    # Find bounding box of GT mask with padding for zoom
    y_idx, x_idx = np.where(gt_mask > 0)
    pad = 20
    y1 = max(0, int(y_idx.min()) - pad)
    y2 = min(512, int(y_idx.max()) + pad)
    x1 = max(0, int(x_idx.min()) - pad)
    x2 = min(512, int(x_idx.max()) + pad)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 1: Global Views
    # (0, 0) Original Document
    axes[0, 0].imshow(orig_np)
    axes[0, 0].set_title("1. Original Document", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # (0, 1) Synthesized Tampered Document
    axes[0, 1].imshow(tamp_np)
    axes[0, 1].set_title("2. Tampered Document (Copy-Paste)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # (0, 2) Ground-Truth Mask
    axes[0, 2].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"3. Ground-Truth Mask ({int(np.sum(gt_mask))} pixels)", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    # Row 2: Model Responses & Region Zooms
    # (1, 0) Model Continuous Probability Map (Full Range)
    p_min, p_max = float(np.min(tamp_prob)), float(np.max(tamp_prob))
    im_p = axes[1, 0].imshow(tamp_prob, cmap="inferno", vmin=p_min, vmax=p_max)
    axes[1, 0].set_title(f"4. Full Probability Map [{p_min:.5f}, {p_max:.5f}]", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")
    cbar = plt.colorbar(im_p, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    # (1, 1) Zoomed Tampered Crop with GT Contour Outline
    tamp_crop = tamp_np[y1:y2, x1:x2].copy()
    gt_crop = gt_mask[y1:y2, x1:x2]
    contours, _ = cv2.findContours(gt_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tamp_crop_bgr = (tamp_crop * 255).astype(np.uint8)
    cv2.drawContours(tamp_crop_bgr, contours, -1, (255, 0, 0), 1)  # Red outline
    axes[1, 1].imshow(tamp_crop_bgr)
    axes[1, 1].set_title(f"5. Zoomed Tampered Text (y:[{y1},{y2}], x:[{x1},{x2}])", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    # (1, 2) Zoomed Probability Map with GT Contour Outline
    prob_crop = tamp_prob[y1:y2, x1:x2]
    im_zoom = axes[1, 2].imshow(prob_crop, cmap="inferno", vmin=p_min, vmax=p_max)
    # Overlay contour on top of heatmap
    for c in contours:
        pts = c.squeeze()
        if len(pts.shape) == 2:
            axes[1, 2].plot(pts[:, 0], pts[:, 1], color="cyan", linewidth=1.5)
    axes[1, 2].set_title(f"6. Zoomed Probability Response (Cyan=GT)", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")
    cbar_z = plt.colorbar(im_zoom, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar_z.ax.tick_params(labelsize=9)

    fig.suptitle("TamperDoc — Ground-Truth Region Diagnostic Analysis", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_formatted_report(tamp_stats: dict, orig_stats: dict) -> None:
    """Prints the comprehensive diagnostic report to console."""
    ts_g = tamp_stats["global"]
    ts_gt = tamp_stats["gt_region"]
    ts_out = tamp_stats["outside_gt"]
    ts_lad = ts_gt["threshold_ladder"]

    os_g = orig_stats["global"]
    os_gt = orig_stats["gt_region"]
    os_out = orig_stats["outside_gt"]
    os_lad = os_gt["threshold_ladder"]

    gt_cnt = tamp_stats["gt_pixel_count"]

    print("=" * 85)
    print("TAMPERDOC — GROUND-TRUTH REGIONAL INFERENCE DIAGNOSTIC REPORT")
    print("=" * 85)

    # 1. Global Stats
    print("\n1. GLOBAL PROBABILITY STATISTICS (Full 512x512 Canvas = 262,144 pixels):")
    print(f"   Metric            | Tampered Document | Authentic Original | Difference (T - O)")
    print(f"   ------------------+-------------------+--------------------+--------------------")
    print(f"   Minimum           | {ts_g['min']:17.6f} | {os_g['min']:18.6f} | {ts_g['min']-os_g['min']:+18.6f}")
    print(f"   Maximum           | {ts_g['max']:17.6f} | {os_g['max']:18.6f} | {ts_g['max']-os_g['max']:+18.6f}")
    print(f"   Mean              | {ts_g['mean']:17.6f} | {os_g['mean']:18.6f} | {ts_g['mean']-os_g['mean']:+18.6f}")
    print(f"   Std Dev           | {ts_g['std']:17.6f} | {os_g['std']:18.6f} | {ts_g['std']-os_g['std']:+18.6f}")
    print(f"   95.0th Percentile | {ts_g['p95']:17.6f} | {os_g['p95']:18.6f} | {ts_g['p95']-os_g['p95']:+18.6f}")
    print(f"   99.0th Percentile | {ts_g['p99']:17.6f} | {os_g['p99']:18.6f} | {ts_g['p99']-os_g['p99']:+18.6f}")
    print(f"   99.9th Percentile | {ts_g['p99_9']:17.6f} | {os_g['p99_9']:18.6f} | {ts_g['p99_9']-os_g['p99_9']:+18.6f}")

    # 2. Inside GT Region
    print(f"\n2. INSIDE GROUND-TRUTH TAMPERED REGION ONLY ({gt_cnt:,} pixels):")
    print(f"   Metric            | Tampered Document | Authentic Original | Difference (T - O)")
    print(f"   ------------------+-------------------+--------------------+--------------------")
    print(f"   Mean Probability  | {ts_gt['mean']:17.6f} | {os_gt['mean']:18.6f} | {ts_gt['mean']-os_gt['mean']:+18.6f}")
    print(f"   Median Probability| {ts_gt['median']:17.6f} | {os_gt['median']:18.6f} | {ts_gt['median']-os_gt['median']:+18.6f}")
    print(f"   Maximum Prob      | {ts_gt['max']:17.6f} | {os_gt['max']:18.6f} | {ts_gt['max']-os_gt['max']:+18.6f}")
    print(f"   Minimum Prob      | {ts_gt['min']:17.6f} | {os_gt['min']:18.6f} | {ts_gt['min']-os_gt['min']:+18.6f}")
    print(f"   Std Dev           | {ts_gt['std']:17.6f} | {os_gt['std']:18.6f} | {ts_gt['std']-os_gt['std']:+18.6f}")
    print(f"   95.0th Percentile | {ts_gt['p95']:17.6f} | {os_gt['p95']:18.6f} | {ts_gt['p95']-os_gt['p95']:+18.6f}")
    print(f"   99.0th Percentile | {ts_gt['p99']:17.6f} | {os_gt['p99']:18.6f} | {ts_gt['p99']-os_gt['p99']:+18.6f}")

    # 3. Outside GT Region
    print(f"\n3. OUTSIDE GROUND-TRUTH REGION ({tamp_stats['bg_pixel_count']:,} pixels):")
    print(f"   Metric            | Tampered Document | Authentic Original | Difference (T - O)")
    print(f"   ------------------+-------------------+--------------------+--------------------")
    print(f"   Mean Probability  | {ts_out['mean']:17.6f} | {os_out['mean']:18.6f} | {ts_out['mean']-os_out['mean']:+18.6f}")
    print(f"   Maximum Prob      | {ts_out['max']:17.6f} | {os_out['max']:18.6f} | {ts_out['max']-os_out['max']:+18.6f}")
    print(f"   95.0th Percentile | {ts_out['p95']:17.6f} | {os_out['p95']:18.6f} | {ts_out['p95']-os_out['p95']:+18.6f}")
    print(f"   99.0th Percentile | {ts_out['p99']:17.6f} | {os_out['p99']:18.6f} | {ts_out['p99']-os_out['p99']:+18.6f}")

    # 4. Regional Comparisons
    print("\n4. IN-REGION VS. BACKGROUND & GLOBAL COMPARISONS:")
    gt_vs_bg_ratio_tamp = ts_gt["mean"] / max(ts_out["mean"], 1e-9)
    gt_vs_bg_ratio_orig = os_gt["mean"] / max(os_out["mean"], 1e-9)
    print(f"   Tampered Document:")
    print(f"     • GT-region Mean ({ts_gt['mean']:.6f}) vs. Background Mean ({ts_out['mean']:.6f}) -> Ratio: {gt_vs_bg_ratio_tamp:.3f}x")
    print(f"     • GT-region Max  ({ts_gt['max']:.6f}) vs. Global Max     ({ts_g['max']:.6f})   -> Ratio: {ts_gt['max']/max(ts_g['max'],1e-9):.3f}x")
    print(f"   Authentic Original Document:")
    print(f"     • Target-region Mean ({os_gt['mean']:.6f}) vs. Background Mean ({os_out['mean']:.6f}) -> Ratio: {gt_vs_bg_ratio_orig:.3f}x")
    print(f"     • Target-region Max  ({os_gt['max']:.6f}) vs. Global Max     ({os_g['max']:.6f})   -> Ratio: {os_gt['max']/max(os_g['max'],1e-9):.3f}x")

    # 5. Threshold Response Ladder
    print(f"\n5. SUB-THRESHOLD ACTIVATION LADDER INSIDE GT REGION ({gt_cnt:,} pixels):")
    print(f"   Threshold Level  | Tampered Count (%)      | Authentic Original Count (%)")
    print(f"   -----------------+-------------------------+-----------------------------")
    for key in ts_lad:
        t_item = ts_lad[key]
        o_item = os_lad[key]
        t_val = t_item['threshold']
        print(f"   Prob >= {t_val:4.2f}    | {t_item['count']:4d} / {gt_cnt} ({t_item['percentage']:6.2f}%)   | {o_item['count']:4d} / {gt_cnt} ({o_item['percentage']:6.2f}%)")

    # 6. Conclusion
    print("\n6. LOCALIZED RESPONSE EVALUATION:")
    elevated_mean = ts_gt["mean"] > ts_out["mean"]
    elevated_vs_orig = ts_gt["mean"] > os_gt["mean"]
    print(f"   • Is GT-region mean elevated above document background? {'YES' if elevated_mean else 'NO'} ({ts_gt['mean']:.6f} vs {ts_out['mean']:.6f})")
    print(f"   • Is GT-region mean elevated compared to original digit? {'YES' if elevated_vs_orig else 'NO'} ({ts_gt['mean']:.6f} vs {os_gt['mean']:.6f})")
    print(f"   • Max probability inside GT tampered digit:             {ts_gt['max']:.6f} ({ts_gt['max']*100:.3f}%)")
    print(f"   • Global decision threshold:                            0.500000 (50.00%)")
    print(f"   • Deficit to decision boundary:                         {0.5 - ts_gt['max']:.6f} ({((0.5 - ts_gt['max'])/0.5)*100:.1f}% below threshold)")
    print("=" * 85)


def main():
    torch.set_num_threads(4)

    # 1. Load model & GT mask
    print("Loading model and inputs...")
    model = load_model(CHECKPOINT_PATH)
    gt_mask = np.asarray(Image.open(GT_MASK_PATH).convert("L"), dtype=np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    # 2. Preprocess images
    orig_np, orig_norm_x, orig_raw_x = preprocess_image(ORIGINAL_PATH)
    tamp_np, tamp_norm_x, tamp_raw_x = preprocess_image(TAMPERED_PATH)

    # 3. Inference
    with torch.inference_mode():
        tamp_out = model(tamp_norm_x, raw_x=tamp_raw_x)
        tamp_logits = tamp_out["tamper_logits"]
        tamp_prob = torch.sigmoid(tamp_logits).squeeze().cpu().numpy()

        orig_out = model(orig_norm_x, raw_x=orig_raw_x)
        orig_logits = orig_out["tamper_logits"]
        orig_prob = torch.sigmoid(orig_logits).squeeze().cpu().numpy()

    # 4. Regional stats
    tamp_stats = compute_regional_stats(tamp_prob, gt_mask, label="Tampered Document")
    orig_stats = compute_regional_stats(orig_prob, gt_mask, label="Authentic Original")

    # 5. Print report
    print_formatted_report(tamp_stats, orig_stats)

    # 6. Save visualization
    generate_diagnostic_visualization(
        orig_np, tamp_np, gt_mask, tamp_prob, orig_prob, OUTPUT_FIG_PATH
    )
    print(f"\nSaved diagnostic visualization -> {OUTPUT_FIG_PATH}")


if __name__ == "__main__":
    main()
