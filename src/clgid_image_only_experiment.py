"""Independent Image-Only TamperDoc Inference Experiment on clgid.jpg.

Performs controlled background patch transfer (5%-10% area) on blank region of clgid.jpg.
Executes pure image-only model inference with checkpoints/best_model.pt (threshold >= 0.5)
without creating, loading, or using any Ground Truth mask.

Outputs:
  - results/clgid_image_only/clgid_tampered.jpg
  - results/clgid_image_only/inference/probability_map.png
  - results/clgid_image_only/inference/predicted_mask.png
  - results/clgid_image_only/inference/inference_visualization.png
  - results/clgid_image_only/inference/diagnostics.json
"""

import json
import os
import sys
from typing import Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# ---------------------------------------------------------------------------
# Paths & Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_IMAGE_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "clgid.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "clgid_image_only")
INFERENCE_DIR = os.path.join(OUTPUT_DIR, "inference")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")

TAMPERED_IMAGE_PATH = os.path.join(OUTPUT_DIR, "clgid_tampered.jpg")
PROB_MAP_PATH = os.path.join(INFERENCE_DIR, "probability_map.png")
PRED_MASK_PATH = os.path.join(INFERENCE_DIR, "predicted_mask.png")
INFER_VIS_PATH = os.path.join(INFERENCE_DIR, "inference_visualization.png")
DIAGNOSTICS_PATH = os.path.join(INFERENCE_DIR, "diagnostics.json")

TARGET_SIZE = (512, 512)

# Background patch region configuration (60x260 = 15,600 px = 5.95% of image)
# Target: blank background strip between header and ID number
TARGET_REGION = {"y1": 75, "y2": 135, "x1": 200, "x2": 460}
SOURCE_REGION = {"y1": 75, "y2": 135, "x1": 30, "x2": 290}


def create_tampered_image() -> Tuple[np.ndarray, dict]:
    """Transfers a background patch into target blank background region with mild edge blending."""
    if not os.path.exists(SRC_IMAGE_PATH):
        raise FileNotFoundError(f"Source image not found: {SRC_IMAGE_PATH}")

    pil_img = Image.open(SRC_IMAGE_PATH)
    width, height = pil_img.size
    if (width, height) != TARGET_SIZE:
        raise ValueError(f"Expected {TARGET_SIZE}, got {width}x{height}")

    img_bgr = cv2.imread(SRC_IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    ty1, ty2, tx1, tx2 = TARGET_REGION["y1"], TARGET_REGION["y2"], TARGET_REGION["x1"], TARGET_REGION["x2"]
    sy1, sy2, sx1, sx2 = SOURCE_REGION["y1"], SOURCE_REGION["y2"], SOURCE_REGION["x1"], SOURCE_REGION["x2"]

    src_patch = img_rgb[sy1:sy2, sx1:sx2].copy()
    target_area = img_rgb[ty1:ty2, tx1:tx2].copy()

    # Mild edge feathering for realistic forensic compositing
    h, w = src_patch.shape[:2]
    feather_mask = np.ones((h, w), dtype=np.float32)
    border = 6
    for i in range(border):
        val = (i + 1) / (border + 1)
        feather_mask[i, :] = np.minimum(feather_mask[i, :], val)
        feather_mask[h - 1 - i, :] = np.minimum(feather_mask[h - 1 - i, :], val)
        feather_mask[:, i] = np.minimum(feather_mask[:, i], val)
        feather_mask[:, w - 1 - i] = np.minimum(feather_mask[:, w - 1 - i], val)

    feather_3d = feather_mask[:, :, np.newaxis]
    blended_patch = (feather_3d * src_patch.astype(np.float32) + (1.0 - feather_3d) * target_area.astype(np.float32))
    blended_patch = np.clip(blended_patch, 0, 255).astype(np.uint8)

    tampered_rgb = img_rgb.copy()
    tampered_rgb[ty1:ty2, tx1:tx2] = blended_patch

    # Save tampered JPEG
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Image.fromarray(tampered_rgb, mode="RGB").save(TAMPERED_IMAGE_PATH, format="JPEG", quality=95)

    patch_area = h * w
    patch_pct = (patch_area / (512 * 512)) * 100.0

    metadata = {
        "tamper_type": "Background patch transfer",
        "target_region": TARGET_REGION,
        "source_region": SOURCE_REGION,
        "manipulated_area_pixels": patch_area,
        "manipulated_area_pct": patch_pct,
    }

    return tampered_rgb, metadata


def run_image_only_inference(tampered_image_path: str) -> dict:
    """Executes model inference strictly from the tampered image with no GT mask."""
    os.makedirs(INFERENCE_DIR, exist_ok=True)

    from src.model import TamperDocModel

    device = torch.device("cpu")
    model = TamperDocModel(
        use_rgb=True,
        use_dct=True,
        use_srm=True,
        pretrained_rgb=False,
        branch_channels=128,
        fused_channels=128,
        target_size=(512, 512),
    )
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Standard preprocessing directly from tampered image
    pil_img = Image.open(tampered_image_path).convert("RGB")
    raw_tensor = TF.to_tensor(pil_img)
    norm_tensor = TF.normalize(raw_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    norm_x = norm_tensor.unsqueeze(0).to(device)
    raw_x = raw_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(norm_x, raw_x=raw_x)
        logits = outputs["tamper_logits"]

    logits_np = logits.squeeze().cpu().numpy()
    probs_np = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Threshold >= 0.5
    pred_mask = (probs_np >= 0.5).astype(np.uint8)

    # 1. Save Probability Map
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(probs_np, cmap="hot", vmin=0.0, vmax=max(float(probs_np.max()), 1e-4))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Sigmoid Probability Map (max={probs_np.max():.4f})", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(PROB_MAP_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Save Predicted Mask PNG
    Image.fromarray(pred_mask * 255, mode="L").save(PRED_MASK_PATH, format="PNG")

    # 3. Save Inference Visualization (Input image, Probability map, Predicted mask)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(np.array(pil_img))
    axes[0].set_title("1. Model Input: clgid_tampered.jpg", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    im_p = axes[1].imshow(probs_np, cmap="hot", vmin=0.0, vmax=max(float(probs_np.max()), 1e-4))
    plt.colorbar(im_p, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(f"2. Sigmoid Probabilities (max={probs_np.max():.4f})", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    pred_count = int(np.sum(pred_mask))
    axes[2].set_title(f"3. Predicted Mask ({pred_count:,} px >= 0.5)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle("Image-Only TamperDoc Inference on clgid_tampered.jpg", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(INFER_VIS_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Calculate Image-Only Metrics
    max_prob = float(probs_np.max())
    max_y, max_x = np.unravel_index(np.argmax(probs_np), probs_np.shape)

    pred_pixels = int(np.sum(pred_mask))
    pred_pct = (pred_pixels / (512 * 512)) * 100.0

    if pred_pixels > 0:
        py_indices, px_indices = np.where(pred_mask > 0)
        min_x = int(px_indices.min())
        max_x_box = int(px_indices.max())
        min_y = int(py_indices.min())
        max_y_box = int(py_indices.max())
        box_w = max_x_box - min_x + 1
        box_h = max_y_box - min_y + 1
        box_area = box_w * box_h
    else:
        min_x, max_x_box, min_y, max_y_box = 0, 0, 0, 0
        box_w, box_h, box_area = 0, 0, 0

    diagnostics = {
        "input_image": os.path.basename(tampered_image_path),
        "maximum_probability": max_prob,
        "max_prob_location": {"x": int(max_x), "y": int(max_y)},
        "predicted_pixels_ge_05": pred_pixels,
        "predicted_percentage": pred_pct,
        "predicted_bounding_box": {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x_box,
            "max_y": max_y_box,
        },
        "predicted_mask_width": box_w,
        "predicted_mask_height": box_h,
        "predicted_mask_area": box_area,
    }

    with open(DIAGNOSTICS_PATH, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    return diagnostics


def print_final_report(metadata: dict, diag: dict) -> None:
    """Prints the strict image-only measured report."""
    print("\n" + "=" * 80)
    print("IMAGE-ONLY TAMPERDOC INFERENCE EXPERIMENT REPORT")
    print("=" * 80)

    print("\n[A] EXPERIMENTAL SETUP")
    print(f"    Input:                      {diag['input_image']}")
    print(f"    Tamper Type:                {metadata['tamper_type']}")
    print(f"    Approximate Manipulated Area: {metadata['manipulated_area_pixels']:,} px ({metadata['manipulated_area_pct']:.2f}% of image)")
    print(f"    Ground Truth Mask:          NONE (Pure image-only inference)")

    print("\n[B] MODEL PREDICTION & LOCALIZATION METRICS")
    print(f"    Maximum Probability:        {diag['maximum_probability']:.6f} ({diag['maximum_probability']*100:.4f}%)")
    loc = diag['max_prob_location']
    print(f"    Strongest Response Location: x={loc['x']}, y={loc['y']}")
    print(f"    Predicted Pixels (>= 0.5):  {diag['predicted_pixels_ge_05']:,}")
    print(f"    Predicted Mask Percentage:  {diag['predicted_percentage']:.4f}%")

    bbox = diag['predicted_bounding_box']
    print("\n[C] PREDICTED MASK GEOMETRY")
    print(f"    Predicted Bounding Box:     min_x={bbox['min_x']}, min_y={bbox['min_y']}, max_x={bbox['max_x']}, max_y={bbox['max_y']}")
    print(f"    Predicted Mask Width:       {diag['predicted_mask_width']} px")
    print(f"    Predicted Mask Height:      {diag['predicted_mask_height']} px")
    print(f"    Predicted Mask Area:        {diag['predicted_mask_area']:,} px")

    print("\n[D] GENERATED ARTIFACTS")
    print(f"    Tampered Image:             {TAMPERED_IMAGE_PATH}")
    print(f"    Probability Map:            {PROB_MAP_PATH}")
    print(f"    Predicted Mask:             {PRED_MASK_PATH}")
    print(f"    Visualization Plot:         {INFER_VIS_PATH}")
    print(f"    Diagnostics JSON:           {DIAGNOSTICS_PATH}")

    print("\n" + "=" * 80)
    print("IMAGE-ONLY EXPERIMENT COMPLETE")
    print("=" * 80)


def main():
    print("=" * 80)
    print("STARTING IMAGE-ONLY TAMPERDOC EXPERIMENT ON clgid.jpg")
    print("=" * 80)

    # 1. Synthesize Tampered Image
    print("[1/3] Creating controlled background patch transfer...")
    _, metadata = create_tampered_image()
    print(f"  [OK] Tampered image created: {TAMPERED_IMAGE_PATH}")

    # 2. Run Image-Only Inference
    print("\n[2/3] Executing image-only model inference (no GT mask)...")
    diag = run_image_only_inference(TAMPERED_IMAGE_PATH)
    print(f"  [OK] Inference artifacts saved to {INFERENCE_DIR}")

    # 3. Print Report
    print("\n[3/3] Generating image-only report...")
    print_final_report(metadata, diag)


if __name__ == "__main__":
    main()
