"""TamperDoc Independent Image-Only Text Tampering Experiment (Demo 7).

Replaces 'History' with 'Life' in 'World History' -> 'World Life' in sample demo 7.jpg.
Strictly preserves 'World', all surrounding courses, grades, credits, tables, and borders.
Executes pure image-only TamperDoc inference (threshold >= 0.5) without any Ground Truth mask.

Outputs:
  - results/demo7_world_life/text_tampered.jpg
  - results/demo7_world_life/inference/probability_map.png
  - results/demo7_world_life/inference/predicted_mask.png
  - results/demo7_world_life/inference/inference_visualization.png
  - results/demo7_world_life/inference/diagnostics.json
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
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths & Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_IMAGE_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "sample demo 7.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "demo7_world_life")
INFERENCE_DIR = os.path.join(OUTPUT_DIR, "inference")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")

TAMPERED_IMAGE_PATH = os.path.join(OUTPUT_DIR, "text_tampered.jpg")
PROB_MAP_PATH = os.path.join(INFERENCE_DIR, "probability_map.png")
PRED_MASK_PATH = os.path.join(INFERENCE_DIR, "predicted_mask.png")
INFER_VIS_PATH = os.path.join(INFERENCE_DIR, "inference_visualization.png")
DIAGNOSTICS_PATH = os.path.join(INFERENCE_DIR, "diagnostics.json")

TARGET_SIZE = (512, 512)
TARGET_ORIGINAL_TEXT = "World History"
TARGET_NEW_TEXT = "World Life"


def render_supersampled_text(
    text: str,
    font_path: str,
    font_size: float,
    text_color: Tuple[int, int, int],
    scale_factor: int = 8,
    blur_sigma: float = 0.35,
) -> np.ndarray:
    """Renders high-resolution text and downsamples with Lanczos for realistic antialiasing."""
    scaled_font_size = int(round(font_size * scale_factor))
    font = ImageFont.truetype(font_path, scaled_font_size)

    dummy_img = Image.new("RGBA", (200, 50), (0, 0, 0, 0))
    dummy_draw = ImageDraw.Draw(dummy_img)
    dummy_draw.text((10, 5), text, font=font, fill=(*text_color, 255))

    final_w = dummy_img.width // scale_factor
    final_h = dummy_img.height // scale_factor
    rendered = dummy_img.resize((final_w, final_h), Image.Resampling.LANCZOS)
    rend_np = np.array(rendered)

    if blur_sigma > 0:
        rend_np[:, :, :3] = cv2.GaussianBlur(rend_np[:, :, :3], (3, 3), blur_sigma)

    # Crop tight around alpha
    alpha = rend_np[:, :, 3]
    nz_y, nz_x = np.where(alpha > 15)
    tight = rend_np[nz_y.min() : nz_y.max() + 1, nz_x.min() : nz_x.max() + 1]
    return tight


def create_tampered_image() -> Tuple[np.ndarray, dict]:
    """Replaces 'History' with 'Life' while keeping 'World' and everything else untouched."""
    if not os.path.exists(SRC_IMAGE_PATH):
        raise FileNotFoundError(f"Source image not found: {SRC_IMAGE_PATH}")

    pil_img = Image.open(SRC_IMAGE_PATH)
    width, height = pil_img.size
    if (width, height) != TARGET_SIZE:
        raise ValueError(f"Expected {TARGET_SIZE}, got {width}x{height}")

    img_bgr = cv2.imread(SRC_IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Coordinates of "History":
    # y in [155, 169], x in [71, 132]
    ty1, ty2 = 155, 169
    tx1, tx2 = 71, 132

    target_crop = img_rgb[ty1:ty2, tx1:tx2].copy()
    gray_crop = cv2.cvtColor(target_crop, cv2.COLOR_RGB2GRAY)

    # Selective stroke inpainting
    stroke_mask = (gray_crop < 190).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inpaint_mask = cv2.dilate(stroke_mask, kernel, iterations=1)
    inpainted = cv2.inpaint(cv2.cvtColor(target_crop, cv2.COLOR_RGB2BGR), inpaint_mask, 3, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

    # Paste inpainted background into image
    tampered_rgb = img_rgb.copy()
    tampered_rgb[ty1:ty2, tx1:tx2] = inpainted_rgb

    # Render replacement "Life"
    font_path = "C:/Windows/Fonts/arialbd.ttf"
    text_color = (44, 42, 40)
    tight_life = render_supersampled_text(
        text="Life",
        font_path=font_path,
        font_size=7.6,
        text_color=text_color,
        scale_factor=8,
        blur_sigma=0.35,
    )

    th, tw = tight_life.shape[:2]
    x_place = 71
    y_place = 166 - th + 1  # baseline alignment at y=166

    alpha = tight_life[:, :, 3].astype(np.float32) / 255.0
    txt = tight_life[:, :, :3].astype(np.float32)

    for c in range(3):
        tampered_rgb[y_place : y_place + th, x_place : x_place + tw, c] = (
            alpha * txt[:, :, c]
            + (1.0 - alpha) * tampered_rgb[y_place : y_place + th, x_place : x_place + tw, c].astype(np.float32)
        ).astype(np.uint8)

    # Save tampered JPEG image
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Image.fromarray(tampered_rgb, mode="RGB").save(TAMPERED_IMAGE_PATH, format="JPEG", quality=95)

    metadata = {
        "tamper_type": "Course Title Text Replacement ('World History' -> 'World Life')",
        "original_course_title": TARGET_ORIGINAL_TEXT,
        "new_course_title": TARGET_NEW_TEXT,
        "replaced_word": "History -> Life",
        "untouched_word": "World",
        "tampered_region": {
            "y1": int(y_place),
            "y2": int(y_place + th),
            "x1": int(x_place),
            "x2": int(x_place + tw),
            "width": int(tw),
            "height": int(th),
            "area_px": int(tw * th),
        },
        "inpainted_clean_box": {
            "y1": ty1, "y2": ty2, "x1": tx1, "x2": tx2,
            "width": tx2 - tx1, "height": ty2 - ty1, "area_px": (tx2 - tx1) * (ty2 - ty1),
        },
        "font_family": "Arial Bold (arialbd.ttf)",
        "font_size": 7.6,
        "text_color_rgb": list(text_color),
        "source_image": SRC_IMAGE_PATH,
        "tampered_image": TAMPERED_IMAGE_PATH,
    }

    return tampered_rgb, metadata


def run_image_only_inference(tampered_rgb: np.ndarray, metadata: dict) -> dict:
    """Executes strictly image-only model inference with NO Ground Truth mask."""
    os.makedirs(INFERENCE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    from model import TamperDocModel

    model = TamperDocModel().to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # Preprocessing
    raw_x = TF.to_tensor(tampered_rgb).unsqueeze(0).to(device)
    norm_x = TF.normalize(raw_x.clone(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Inference forward pass
    with torch.no_grad():
        outputs = model(norm_x, raw_x=raw_x)
        logits = outputs["tamper_logits"]
        probs = torch.sigmoid(logits)

    prob_map = probs.squeeze().cpu().numpy()
    pred_mask = (prob_map >= 0.5).astype(np.uint8)

    # Image-Only Quantitative Diagnostics
    total_pixels = prob_map.size
    pred_positive_pixels = int(np.sum(pred_mask))
    pred_positive_pct = float(pred_positive_pixels / total_pixels * 100.0)

    max_prob = float(np.max(prob_map))
    min_prob = float(np.min(prob_map))
    mean_prob = float(np.mean(prob_map))
    median_prob = float(np.median(prob_map))

    # Strongest response location
    max_y, max_x = np.unravel_index(np.argmax(prob_map), prob_map.shape)

    # Connected components on predicted mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask)
    components = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        components.append({
            "component_id": int(i),
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "area_px": int(area),
            "centroid": [float(cx), float(cy)],
        })

    # Overall predicted bounding box
    if pred_positive_pixels > 0:
        py, px = np.where(pred_mask > 0)
        p_bbox = {
            "x1": int(px.min()),
            "y1": int(py.min()),
            "x2": int(px.max() + 1),
            "y2": int(py.max() + 1),
            "width": int(px.max() - px.min() + 1),
            "height": int(py.max() - py.min() + 1),
            "area_px": int((px.max() - px.min() + 1) * (py.max() - py.min() + 1)),
        }
    else:
        p_bbox = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "width": 0, "height": 0, "area_px": 0}

    # Visual Artifacts
    # 1. Probability Map
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(prob_map, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("TamperDoc Sigmoid Probability Map", fontsize=11)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tampering Probability")
    plt.tight_layout()
    plt.savefig(PROB_MAP_PATH, dpi=150, bbox_inches="tight")
    alt_dir = os.path.join(PROJECT_ROOT, "results", "text_only_test", "inference")
    os.makedirs(alt_dir, exist_ok=True)
    plt.savefig(os.path.join(alt_dir, "probability_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Predicted Mask
    Image.fromarray(pred_mask * 255, mode="L").save(PRED_MASK_PATH)
    Image.fromarray(pred_mask * 255, mode="L").save(os.path.join(alt_dir, "predicted_mask.png"))

    # 3. 3-Panel Inspection Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(tampered_rgb)
    axes[0].set_title("Tampered Document ('World Life')", fontsize=10)
    axes[0].axis("off")

    im1 = axes[1].imshow(prob_map, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Probability Map (Max: {max_prob:.4f})", fontsize=10)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Predicted Mask (>=0.5: {pred_positive_pixels} px / {pred_positive_pct:.2f}%)", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(INFER_VIS_PATH, dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(alt_dir, "inference_visualization.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Diagnostics JSON
    diagnostics = {
        "experiment_name": "Demo 7 Image-Only Text Tampering ('World History' -> 'World Life')",
        "source_image": SRC_IMAGE_PATH,
        "tampered_image": TAMPERED_IMAGE_PATH,
        "checkpoint": CHECKPOINT_PATH,
        "device": str(device),
        "target_original_text": TARGET_ORIGINAL_TEXT,
        "target_new_text": TARGET_NEW_TEXT,
        "synthesis_metadata": metadata,
        "inference_metrics": {
            "decision_threshold": 0.5,
            "maximum_probability": max_prob,
            "max_probability": max_prob,
            "strongest_response_x": int(max_x),
            "strongest_response_y": int(max_y),
            "strongest_response_probability": max_prob,
            "min_probability": min_prob,
            "mean_probability": mean_prob,
            "median_probability": median_prob,
            "predicted_tampered_pixels": pred_positive_pixels,
            "predicted_tampered_percentage": pred_positive_pct,
            "total_pixels": total_pixels,
            "predicted_bounding_box": p_bbox,
            "predicted_mask_width": p_bbox["width"],
            "predicted_mask_height": p_bbox["height"],
            "predicted_mask_area": p_bbox["area_px"],
            "num_detected_components": len(components),
            "detected_components": components,
        },
        "artifacts": {
            "tampered_image": TAMPERED_IMAGE_PATH,
            "probability_map": PROB_MAP_PATH,
            "predicted_mask": PRED_MASK_PATH,
            "inference_visualization": INFER_VIS_PATH,
            "diagnostics_json": DIAGNOSTICS_PATH,
        },
    }

    with open(DIAGNOSTICS_PATH, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    with open(os.path.join(alt_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    return diagnostics


def main():
    print("=" * 70)
    print("TAMPERDOC DEMO 7: 'World History' -> 'World Life' TEXT TAMPERING")
    print(f"Source: {SRC_IMAGE_PATH}")
    print(f"Target: '{TARGET_ORIGINAL_TEXT}' -> '{TARGET_NEW_TEXT}'")
    print("=" * 70)

    # 1. Synthesize tampered document
    tampered_rgb, meta = create_tampered_image()
    print(f"[1/2] Tampered image synthesized and saved to: {TAMPERED_IMAGE_PATH}")

    # 2. Run Image-Only Inference
    diag = run_image_only_inference(tampered_rgb, meta)
    print(f"[2/2] Inference completed. Diagnostics saved to: {DIAGNOSTICS_PATH}")
    print("=" * 70)
    print("INFERENCE SUMMARY:")
    metrics = diag["inference_metrics"]
    print(f"  Maximum Probability:           {metrics['max_probability']:.4f}")
    print(f"  Strongest-Response (x, y):     ({metrics['strongest_response_x']}, {metrics['strongest_response_y']})")
    print(f"  Strongest-Response Prob:       {metrics['strongest_response_probability']:.4f}")
    print(f"  Mean / Median Probability:     {metrics['mean_probability']:.4f} / {metrics['median_probability']:.4f}")
    print(f"  Predicted Pixels (>=0.5):      {metrics['predicted_tampered_pixels']} ({metrics['predicted_tampered_percentage']:.3f}%)")
    print(f"  Predicted Bounding Box:        {metrics['predicted_bounding_box']}")
    print(f"  Detected Components:           {metrics['num_detected_components']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
