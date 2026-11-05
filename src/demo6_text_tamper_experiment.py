"""Independent Image-Only Text Tampering Experiment on sample demo 6 allt.jpg.

Performs controlled, text-only modification replacing academic program/department text with:
  'Electronics and Communication Engineering'

Strict Constraints:
- Text-only manipulation: No background patch transfer, no watermark, no paper texture modifications.
- Visual and forensic consistency: Font family (bold sans-serif), stroke thickness, baseline alignment,
  local contrast, scan softness, and natural antialiasing.
- Strict Image-Only Inference: Absolutely NO Ground Truth mask created, loaded, or evaluated.
- Model: checkpoints/best_model.pt (threshold >= 0.5).

Outputs:
  - results/text_only_test/text_tampered.jpg
  - results/text_only_test/inference/probability_map.png
  - results/text_only_test/inference/predicted_mask.png
  - results/text_only_test/inference/inference_visualization.png
  - results/text_only_test/inference/diagnostics.json
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
SRC_IMAGE_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "sample demo 6 allt.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "text_only_test")
INFERENCE_DIR = os.path.join(OUTPUT_DIR, "inference")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")

TAMPERED_IMAGE_PATH = os.path.join(OUTPUT_DIR, "text_tampered.jpg")
PROB_MAP_PATH = os.path.join(INFERENCE_DIR, "probability_map.png")
PRED_MASK_PATH = os.path.join(INFERENCE_DIR, "predicted_mask.png")
INFER_VIS_PATH = os.path.join(INFERENCE_DIR, "inference_visualization.png")
DIAGNOSTICS_PATH = os.path.join(INFERENCE_DIR, "diagnostics.json")

TARGET_SIZE = (512, 512)
TARGET_TEXT = "Electronics and Communication Engineering"


def render_supersampled_text(
    text: str,
    font_path: str,
    font_size: float,
    text_color: Tuple[int, int, int],
    scale_factor: int = 8,
    blur_sigma: float = 0.35,
) -> Tuple[np.ndarray, int, int]:
    """Renders high-resolution text and downsamples with Lanczos for realistic antialiasing."""
    scaled_font_size = int(round(font_size * scale_factor))
    font = ImageFont.truetype(font_path, scaled_font_size)

    dummy_img = Image.new("RGBA", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=(*text_color, 255))

    final_w = int(round(tw / scale_factor))
    final_h = int(round(th / scale_factor))
    rendered = canvas.resize((final_w, final_h), Image.Resampling.LANCZOS)
    rend_np = np.array(rendered)

    if blur_sigma > 0:
        rend_np[:, :, :3] = cv2.GaussianBlur(rend_np[:, :, :3], (3, 3), blur_sigma)

    return rend_np, final_w, final_h


def create_tampered_image() -> Tuple[np.ndarray, dict]:
    """Applies text-only tampering to replace program text with 'Electronics and Communication Engineering'."""
    if not os.path.exists(SRC_IMAGE_PATH):
        raise FileNotFoundError(f"Source image not found: {SRC_IMAGE_PATH}")

    pil_img = Image.open(SRC_IMAGE_PATH)
    width, height = pil_img.size
    if (width, height) != TARGET_SIZE:
        raise ValueError(f"Expected {TARGET_SIZE}, got {width}x{height}")

    img_bgr = cv2.imread(SRC_IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Coordinates of target department text line:
    # y in [135, 149], x in [44, 350]
    ty1, ty2 = 135, 149
    tx1, tx2 = 44, 350

    target_crop = img_rgb[ty1:ty2, tx1:tx2].copy()
    gray_crop = cv2.cvtColor(target_crop, cv2.COLOR_RGB2GRAY)

    # Stroke-level segmentation for inpainting
    text_mask = (gray_crop < 88).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inpaint_mask = cv2.dilate(text_mask, kernel, iterations=1)

    # OpenCV Telea inpainting on text strokes only
    inpainted = cv2.inpaint(cv2.cvtColor(target_crop, cv2.COLOR_RGB2BGR), inpaint_mask, 3, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

    # Render replacement text: "Electronics and Communication Engineering"
    font_path = "C:/Windows/Fonts/arialbd.ttf"
    text_color = (46, 43, 41)
    rendered_rgba, fw, fh = render_supersampled_text(
        text=TARGET_TEXT,
        font_path=font_path,
        font_size=7.2,
        text_color=text_color,
        scale_factor=8,
        blur_sigma=0.35,
    )

    # Paste inpainted background into image
    tampered_rgb = img_rgb.copy()
    tampered_rgb[ty1:ty2, tx1:tx2] = inpainted_rgb

    # Composite replacement text onto tampered image at exact baseline
    # Baseline alignment: y ~ 145 -> y_start = 146 - fh
    y_pos = 146 - fh
    x_pos = tx1 + 1

    alpha = rendered_rgba[:, :, 3].astype(np.float32) / 255.0
    text_rgb_layer = rendered_rgba[:, :, :3].astype(np.float32)

    h_fit = min(fh, 512 - y_pos)
    w_fit = min(fw, 512 - x_pos)

    for c in range(3):
        tampered_rgb[y_pos : y_pos + h_fit, x_pos : x_pos + w_fit, c] = (
            alpha[:h_fit, :w_fit] * text_rgb_layer[:h_fit, :w_fit, c]
            + (1.0 - alpha[:h_fit, :w_fit]) * tampered_rgb[y_pos : y_pos + h_fit, x_pos : x_pos + w_fit, c].astype(np.float32)
        ).astype(np.uint8)

    # Save tampered JPEG image
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Image.fromarray(tampered_rgb, mode="RGB").save(TAMPERED_IMAGE_PATH, format="JPEG", quality=95)

    metadata = {
        "tamper_type": "Text-Only Program Name Replacement",
        "target_string": TARGET_TEXT,
        "tampered_region": {
            "y1": int(y_pos),
            "y2": int(y_pos + h_fit),
            "x1": int(x_pos),
            "x2": int(x_pos + w_fit),
            "width": int(w_fit),
            "height": int(h_fit),
            "area_px": int(w_fit * h_fit),
        },
        "font_family": "Arial Bold (arialbd.ttf)",
        "font_size": 7.2,
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

    # Preprocessing: standard ImageNet normalization
    raw_x = TF.to_tensor(tampered_rgb).unsqueeze(0).to(device)
    norm_x = TF.normalize(raw_x.clone(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Inference forward pass
    with torch.no_grad():
        outputs = model(norm_x, raw_x=raw_x)
        logits = outputs["tamper_logits"]
        probs = torch.sigmoid(logits)

    prob_map = probs.squeeze().cpu().numpy()
    pred_mask = (prob_map >= 0.5).astype(np.uint8)

    # Quantitative diagnostics (strictly Image-Only)
    total_pixels = prob_map.size
    pred_positive_pixels = int(np.sum(pred_mask))
    pred_positive_pct = float(pred_positive_pixels / total_pixels * 100.0)

    max_prob = float(np.max(prob_map))
    min_prob = float(np.min(prob_map))
    mean_prob = float(np.mean(prob_map))
    median_prob = float(np.median(prob_map))

    # Location of maximum probability
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

    # Save visual outputs
    # 1. Probability Map (Viridis color scale)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(prob_map, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("TamperDoc Sigmoid Probability Map", fontsize=11)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tampering Probability")
    plt.tight_layout()
    plt.savefig(PROB_MAP_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Predicted Mask (Binary)
    Image.fromarray(pred_mask * 255, mode="L").save(PRED_MASK_PATH)

    # 3. 3-Panel Inspection Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(tampered_rgb)
    axes[0].set_title("Tampered Document (Image Only)", fontsize=10)
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
    plt.close(fig)

    # Package diagnostics
    diagnostics = {
        "experiment_name": "Image-Only Text Tampering Localization (sample demo 6 allt.jpg)",
        "source_image": SRC_IMAGE_PATH,
        "tampered_image": TAMPERED_IMAGE_PATH,
        "checkpoint": CHECKPOINT_PATH,
        "device": str(device),
        "target_text": TARGET_TEXT,
        "synthesis_metadata": metadata,
        "inference_metrics": {
            "decision_threshold": 0.5,
            "max_probability": max_prob,
            "max_probability_location": {"x": int(max_x), "y": int(max_y)},
            "min_probability": min_prob,
            "mean_probability": mean_prob,
            "median_probability": median_prob,
            "predicted_tampered_pixels": pred_positive_pixels,
            "predicted_tampered_percentage": pred_positive_pct,
            "total_pixels": total_pixels,
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

    return diagnostics


def main():
    print("=" * 70)
    print("TAMPERDOC IMAGE-ONLY TEXT TAMPERING EXPERIMENT")
    print(f"Source: {SRC_IMAGE_PATH}")
    print(f"Target Replacement: '{TARGET_TEXT}'")
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
    print(f"  Max Probability:           {metrics['max_probability']:.4f} at (x={metrics['max_probability_location']['x']}, y={metrics['max_probability_location']['y']})")
    print(f"  Mean / Median Probability: {metrics['mean_probability']:.4f} / {metrics['median_probability']:.4f}")
    print(f"  Predicted Pixels (>=0.5):  {metrics['predicted_tampered_pixels']} ({metrics['predicted_tampered_percentage']:.3f}%)")
    print(f"  Detected Components:       {metrics['num_detected_components']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
