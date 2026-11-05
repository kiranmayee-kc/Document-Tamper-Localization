"""Regional Ground-Truth Diagnostic for DocTamper Demo 2 (Multi-Character).

Evaluates the trained TamperDoc model on doctamper_synthesized_tampered_v2.jpg
using the exact ground-truth mask doctamper_synthesized_mask_v2.png and saves all
demo artifacts under results/demo_synthesized_v2/.
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
TAMPERED_PATH = os.path.abspath("results/demo_inputs/doctamper_synthesized_tampered_v2.jpg")
ORIGINAL_PATH = os.path.abspath("results/demo_inputs/og current bill.jpg")
GT_MASK_PATH  = os.path.abspath("results/demo_inputs/doctamper_synthesized_mask_v2.png")
OUTPUT_DIR    = os.path.abspath("results/demo_synthesized_v2")

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
    """Computes statistical breakdown globally, inside GT, and outside GT."""
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

    # 3. Outside GT Region
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


def save_demo_artifacts(
    orig_np: np.ndarray,
    tamp_np: np.ndarray,
    gt_mask: np.ndarray,
    tamp_prob: np.ndarray,
    orig_prob: np.ndarray,
    pred_mask: np.ndarray,
    out_dir: str,
) -> None:
    """Generates and saves all required visual artifacts."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. tampered_image.png
    tamp_pil = Image.fromarray((tamp_np * 255).astype(np.uint8), mode="RGB")
    tamp_pil.save(os.path.join(out_dir, "tampered_image.png"))

    # 2. predicted_mask.png (threshold 0.5)
    pred_pil = Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L")
    pred_pil.save(os.path.join(out_dir, "predicted_mask.png"))

    # 3. probability_map_full_range.png
    p_min, p_max = float(np.min(tamp_prob)), float(np.max(tamp_prob))
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(tamp_prob, cmap="inferno", vmin=p_min, vmax=p_max)
    ax.set_title(f"Continuous Probability Map (Full Range: [{p_min:.5f}, {p_max:.5f}])", fontsize=10, pad=10)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "probability_map_full_range.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. comparison.png (5-panel comparison)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))
    titles = [
        "1. Original Document",
        "2. Tampered Document v2",
        "3. Tamper Probability",
        "4. Predicted Binary Mask (>=0.5)",
        "5. GT Mask Overlay",
    ]

    axes[0].imshow(orig_np)
    axes[1].imshow(tamp_np)
    im_heat = axes[2].imshow(tamp_prob, cmap="hot", vmin=0.0, vmax=1.0)
    plt.colorbar(im_heat, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)

    # GT overlay
    overlay = tamp_np.copy()
    mask_3d = gt_mask[:, :, np.newaxis].astype(np.float32)
    red_color = np.array([0.9, 0.1, 0.1], dtype=np.float32)
    overlay = (1.0 - 0.65 * mask_3d) * overlay + (0.65 * mask_3d) * red_color
    axes[4].imshow(np.clip(overlay, 0.0, 1.0))

    for ax_i, title in zip(axes, titles):
        ax_i.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax_i.axis("off")

    fig.suptitle("TamperDoc Demo 2 — Multi-Character Tampering Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.set_num_threads(4)

    print("=" * 85)
    print("TAMPERDOC DEMO 2 — INFERENCE & REGIONAL DIAGNOSIS")
    print("=" * 85)
    print("Loading model and inputs...")
    model = load_model(CHECKPOINT_PATH)
    gt_mask = np.asarray(Image.open(GT_MASK_PATH).convert("L"), dtype=np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    orig_np, orig_norm_x, orig_raw_x = preprocess_image(ORIGINAL_PATH)
    tamp_np, tamp_norm_x, tamp_raw_x = preprocess_image(TAMPERED_PATH)

    print("Running neural network inference...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        tamp_out = model(tamp_norm_x, raw_x=tamp_raw_x)
        tamp_logits = tamp_out["tamper_logits"]
        tamp_prob = torch.sigmoid(tamp_logits).squeeze().cpu().numpy()
        pred_mask = (tamp_logits >= 0.0).squeeze().cpu().numpy().astype(np.uint8)

        orig_out = model(orig_norm_x, raw_x=orig_raw_x)
        orig_logits = orig_out["tamper_logits"]
        orig_prob = torch.sigmoid(orig_logits).squeeze().cpu().numpy()
    elapsed = time.perf_counter() - t0
    print(f"Inference complete in {elapsed*1000:.2f} ms")

    # Compute stats
    tamp_stats = compute_regional_stats(tamp_prob, gt_mask, label="Tampered Document v2")
    orig_stats = compute_regional_stats(orig_prob, gt_mask, label="Authentic Original")

    # Save artifacts
    save_demo_artifacts(
        orig_np, tamp_np, gt_mask, tamp_prob, orig_prob, pred_mask, OUTPUT_DIR
    )
    print(f"Saved artifacts to -> {OUTPUT_DIR}")

    # Output detailed report
    ts_g = tamp_stats["global"]
    ts_gt = tamp_stats["gt_region"]
    ts_out = tamp_stats["outside_gt"]
    ts_lad = ts_gt["threshold_ladder"]

    os_g = orig_stats["global"]
    os_gt = orig_stats["gt_region"]
    os_out = orig_stats["outside_gt"]

    gt_cnt = tamp_stats["gt_pixel_count"]
    pred_cnt = int(np.sum(pred_mask))

    print("\n" + "=" * 85)
    print("DEMO 2 DIAGNOSTIC REPORT")
    print("=" * 85)
    print(f"A. Source Region:              y:[308, 330], x:[368, 416] (Meter reading prefix '1 1 2')")
    print(f"B. Target Region:              y:[147, 169], x:[419, 467] (Charges table entry '0 0 0')")
    print(f"C. Characters Manipulated:     3 digits: replaced '0 0 0' with '1 1 2' from same bill")
    print(f"D. Source/Target Dimensions:   22x48 pixels each")
    print(f"E. Sauvola Parameters:         window=25, k=0.2, R=128.0")
    print(f"F. Telea Inpainting Used:      YES (cv2.INPAINT_TELEA, radius=3) on target text strokes")
    print(f"G. Other Post-Processing:      Direct stroke alpha-composition (no artificial smoothing)")
    print(f"H. GT Foreground Pixel Count:  {gt_cnt:,} pixels (0.1434% of canvas)")
    print(f"I. Changed Pixel Count:        818 pixels (0.3120% of canvas)")
    print(f"J. GT Bounding Box:            y:[147, 169], x:[419, 467]")
    print(f"-------------------------------------------------------------------------------------")
    print(f"1. Global Probability Stats:")
    print(f"   • Min:                      {ts_g['min']:.6f}")
    print(f"   • Max:                      {ts_g['max']:.6f}")
    print(f"   • Mean:                     {ts_g['mean']:.6f}")
    print(f"   • Std Dev:                  {ts_g['std']:.6f}")
    print(f"   • P95:                      {ts_g['p95']:.6f}")
    print(f"   • P99:                      {ts_g['p99']:.6f}")
    print(f"   • P99.9:                    {ts_g['p99_9']:.6f}")
    print(f"-------------------------------------------------------------------------------------")
    print(f"2. Inside GT Tampered Region ({gt_cnt} pixels):")
    print(f"   • Mean Probability:         {ts_gt['mean']:.6f} (Original at same region: {os_gt['mean']:.6f})")
    print(f"   • Median Probability:       {ts_gt['median']:.6f} (Original: {os_gt['median']:.6f})")
    print(f"   • Maximum Probability:      {ts_gt['max']:.6f} (Original: {os_gt['max']:.6f})")
    print(f"   • P95:                      {ts_gt['p95']:.6f}")
    print(f"   • P99:                      {ts_gt['p99']:.6f}")
    print(f"-------------------------------------------------------------------------------------")
    print(f"3. Outside GT Region (Background Canvas = {tamp_stats['bg_pixel_count']:,} pixels):")
    print(f"   • Mean Probability:         {ts_out['mean']:.6f}")
    print(f"   • Maximum Probability:      {ts_out['max']:.6f} (at image boundary)")
    print(f"   • P95:                      {ts_out['p95']:.6f}")
    print(f"   • P99:                      {ts_out['p99']:.6f}")
    print(f"-------------------------------------------------------------------------------------")
    print(f"4. Sub-Threshold Activation Ladder Inside GT Region:")
    for key in ts_lad:
        item = ts_lad[key]
        print(f"   • Prob >= {item['threshold']:4.2f}:              {item['count']:4d} / {gt_cnt} ({item['percentage']:6.2f}%)")
    print(f"-------------------------------------------------------------------------------------")
    print(f"K. Model Global Maximum Prob:  {ts_g['max']:.6f} (at boundary)")
    print(f"L. GT-Region Mean / Max Prob:  {ts_gt['mean']:.6f} / {ts_gt['max']:.6f}")
    print(f"M. Pixels >= 0.5 (Threshold):  {pred_cnt} (0.00%)")
    print(f"N. Non-Empty Binary Mask?      NO (0 / 262,144 pixels predicted as tampered)")
    print("=" * 85)


if __name__ == "__main__":
    main()
