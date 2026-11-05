"""Demo 4: End-to-End Selective Tampering Synthesis and Evaluation (1120.00 -> 2436.00).

Synthesizes a realistic tampering demonstration on 'results/demo_inputs/og current bill.jpg'
modifying Bill Amount 1120.00 -> 2436.00 using genuine source digit crops (2, 4, 3, 6)
and Sauvola-guided selective copy-paste.

Followed by model inference with checkpoints/best_model.pt.
"""

import json
import os
import sys
from typing import Dict, List, Tuple

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
# Configuration & Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "og current bill.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "demo4_2436")
INFERENCE_DIR = os.path.join(OUTPUT_DIR, "inference")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")

SYNTHESIZED_IMAGE_PATH = os.path.join(OUTPUT_DIR, "synthesized_2436.jpg")
GT_MASK_PATH = os.path.join(OUTPUT_DIR, "gt_mask_2436.png")
VERIFICATION_PATH = os.path.join(OUTPUT_DIR, "verification_2436.png")
CHECK_IMAGE_PATH = os.path.join(OUTPUT_DIR, "bill_amount_check_2436.png")

PROB_MAP_PATH = os.path.join(INFERENCE_DIR, "probability_map.png")
PRED_MASK_PATH = os.path.join(INFERENCE_DIR, "predicted_mask.png")
COMPARISON_PATH = os.path.join(INFERENCE_DIR, "comparison.png")
DIAGNOSTICS_PATH = os.path.join(INFERENCE_DIR, "diagnostics.json")

TARGET_SIZE = (512, 512)
SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.2
SAUVOLA_R = 128.0

# ---------------------------------------------------------------------------
# Digit Configurations
# ---------------------------------------------------------------------------
DIGIT_REPLACEMENTS = [
    {
        "pos_id": 1,
        "original_digit": "1",
        "replacement_digit": "2",
        "target": {"y1": 308, "y2": 330, "x1": 368, "x2": 382},
        "source": {"y1": 308, "y2": 330, "x1": 398, "x2": 414},
        "source_desc": "digit '2' from Bill Amount field (pos 3)",
    },
    {
        "pos_id": 2,
        "original_digit": "1",
        "replacement_digit": "4",
        "target": {"y1": 308, "y2": 330, "x1": 383, "x2": 396},
        "source": {"y1": 55, "y2": 75, "x1": 435, "x2": 449},
        "source_desc": "digit '4' from '11.46' in Electricity Duty row",
    },
    {
        "pos_id": 3,
        "original_digit": "2",
        "replacement_digit": "3",
        "target": {"y1": 308, "y2": 330, "x1": 397, "x2": 413},
        "source": {"y1": 276, "y2": 298, "x1": 440, "x2": 455},
        "source_desc": "digit '3' from '-0.36' in Loss/Gain row",
    },
    {
        "pos_id": 4,
        "original_digit": "0",
        "replacement_digit": "6",
        "target": {"y1": 308, "y2": 330, "x1": 414, "x2": 429},
        "source": {"y1": 55, "y2": 75, "x1": 449, "x2": 463},
        "source_desc": "digit '6' from '11.46' in Electricity Duty row",
    },
]


def validate_input_image(path: str) -> np.ndarray:
    """Strictly validates input document image."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    pil_img = Image.open(path)
    img_format = (pil_img.format or "").upper()
    if img_format not in ("JPEG", "JPG"):
        raise ValueError(f"Input image must be a JPEG, got format: '{img_format}'")

    width, height = pil_img.size
    if (width, height) != TARGET_SIZE:
        raise ValueError(f"Input dimensions must be {TARGET_SIZE}, got {width}x{height}")

    if pil_img.mode != "RGB":
        raise ValueError(f"Input image mode must be RGB, got '{pil_img.mode}'")

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Failed to read image with OpenCV: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def compute_sauvola_binarization(
    gray: np.ndarray,
    window_size: int = SAUVOLA_WINDOW,
    k: float = SAUVOLA_K,
    R: float = SAUVOLA_R,
) -> np.ndarray:
    """Computes exact Sauvola local adaptive binarization."""
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (window_size, window_size))
    sq_mean = cv2.blur(gray_f ** 2, (window_size, window_size))
    variance = np.maximum(sq_mean - mean ** 2, 0.0)
    std = np.sqrt(variance)
    threshold = mean * (1.0 + k * (std / R - 1.0))
    return (gray_f < threshold).astype(np.uint8)


def synthesize_tampered_document(
    img_rgb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Performs Demo 4 synthesis replacing 1120.00 with 2436.00."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sauvola_fg = compute_sauvola_binarization(gray)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    global_inpaint_mask = np.zeros((512, 512), dtype=np.uint8)
    gt_mask = np.zeros((512, 512), dtype=np.uint8)

    per_digit_stats: List[dict] = []

    for rep in DIGIT_REPLACEMENTS:
        t = rep["target"]
        s = rep["source"]
        ty1, ty2, tx1, tx2 = t["y1"], t["y2"], t["x1"], t["x2"]
        target_h, target_w = ty2 - ty1, tx2 - tx1

        target_fg_crop = sauvola_fg[ty1:ty2, tx1:tx2]
        dilated_target_fg = cv2.dilate(target_fg_crop, dilate_kernel, iterations=2)
        global_inpaint_mask[ty1:ty2, tx1:tx2] = np.maximum(
            global_inpaint_mask[ty1:ty2, tx1:tx2], dilated_target_fg
        )

        sy1, sy2, sx1, sx2 = s["y1"], s["y2"], s["x1"], s["x2"]
        source_fg_crop_raw = sauvola_fg[sy1:sy2, sx1:sx2]
        source_fg_crop = cv2.resize(source_fg_crop_raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        per_digit_stats.append({
            "pos_id": rep["pos_id"],
            "original_digit": rep["original_digit"],
            "replacement_digit": rep["replacement_digit"],
            "target_bbox": t,
            "source_bbox": s,
            "source_desc": rep["source_desc"],
            "target_shape_hw": (target_h, target_w),
            "source_shape_hw": (sy2 - sy1, sx2 - sx1),
            "source_fg_pixels_raw": int(np.sum(source_fg_crop_raw)),
            "source_fg_pixels_resized": int(np.sum(source_fg_crop)),
        })

    # Clean target region using Telea inpainting
    cleaned_rgb = cv2.inpaint(
        img_rgb, global_inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )
    tampered_rgb = cleaned_rgb.copy()

    # Composite source RGB crops using Sauvola masks
    for rep, stats in zip(DIGIT_REPLACEMENTS, per_digit_stats):
        t = rep["target"]
        s = rep["source"]
        ty1, ty2, tx1, tx2 = t["y1"], t["y2"], t["x1"], t["x2"]
        target_h, target_w = ty2 - ty1, tx2 - tx1
        sy1, sy2, sx1, sx2 = s["y1"], s["y2"], s["x1"], s["x2"]

        src_rgb_crop = img_rgb[sy1:sy2, sx1:sx2]
        src_fg_crop = sauvola_fg[sy1:sy2, sx1:sx2]

        stroke_mask = cv2.resize(src_fg_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        src_h, src_w = sy2 - sy1, sx2 - sx1
        interp = cv2.INTER_AREA if (src_h >= target_h and src_w >= target_w) else cv2.INTER_LINEAR
        src_rgb_resized = cv2.resize(src_rgb_crop, (target_w, target_h), interpolation=interp)

        # Composite source RGB crop with Sauvola mask over cleaned background
        mask_3d = stroke_mask[:, :, np.newaxis].astype(np.float32)
        cleaned_target = cleaned_rgb[ty1:ty2, tx1:tx2].astype(np.float32)
        src_crop_f = src_rgb_resized.astype(np.float32)

        blended = (1.0 - mask_3d) * cleaned_target + mask_3d * src_crop_f
        tampered_rgb[ty1:ty2, tx1:tx2] = np.clip(blended, 0, 255).astype(np.uint8)

        gt_mask[ty1:ty2, tx1:tx2] = np.maximum(gt_mask[ty1:ty2, tx1:tx2], stroke_mask)

    diff = np.max(np.abs(img_rgb.astype(np.int32) - tampered_rgb.astype(np.int32)), axis=-1)
    y_diff, x_diff = np.where(diff > 0)
    changed_pixel_count = int(len(y_diff))
    changed_pixel_pct = (changed_pixel_count / (512 * 512)) * 100.0
    changed_bbox = {
        "y_min": int(y_diff.min()) if changed_pixel_count > 0 else 0,
        "y_max": int(y_diff.max()) if changed_pixel_count > 0 else 0,
        "x_min": int(x_diff.min()) if changed_pixel_count > 0 else 0,
        "x_max": int(x_diff.max()) if changed_pixel_count > 0 else 0,
    }

    metadata = {
        "transformation": "1120.00 -> 2436.00 (Bill Amount field)",
        "per_digit_stats": per_digit_stats,
        "sauvola_params": {"window": SAUVOLA_WINDOW, "k": SAUVOLA_K, "R": SAUVOLA_R},
        "inpaint_method": "cv2.INPAINT_TELEA (radius=3)",
        "changed_pixel_count": changed_pixel_count,
        "changed_pixel_pct": changed_pixel_pct,
        "changed_bbox": changed_bbox,
        "gt_foreground_pixels": int(np.sum(gt_mask)),
        "gt_foreground_pct": (float(np.sum(gt_mask)) / (512 * 512)) * 100.0,
    }

    return tampered_rgb, gt_mask, metadata, sauvola_fg


def save_synthesis_outputs(
    original_rgb: np.ndarray,
    sauvola_fg: np.ndarray,
    tampered_rgb: np.ndarray,
    gt_mask: np.ndarray,
    metadata: dict,
) -> None:
    """Saves the 4 required synthesis outputs in results/demo4_2436/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Tampered Image
    Image.fromarray(tampered_rgb, mode="RGB").save(SYNTHESIZED_IMAGE_PATH, format="JPEG", quality=95)

    # 2. GT Mask
    Image.fromarray(gt_mask, mode="L").save(GT_MASK_PATH, format="PNG")

    # 3. Verification Figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("1. Original Document", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(sauvola_fg, cmap="gray")
    axes[1].set_title(f"2. Sauvola Mask (w={SAUVOLA_WINDOW}, k={SAUVOLA_K})", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(tampered_rgb)
    axes[2].set_title("3. Tampered Document (2436.00)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    overlay = tampered_rgb.copy().astype(np.float32) / 255.0
    mask_3d = gt_mask[:, :, np.newaxis].astype(np.float32)
    red_color = np.array([0.9, 0.1, 0.1], dtype=np.float32)
    alpha = 0.65 * mask_3d
    overlay = np.clip((1.0 - alpha) * overlay + alpha * red_color, 0.0, 1.0)
    axes[3].imshow(overlay)
    axes[3].set_title(f"4. GT Mask Overlay ({metadata['gt_foreground_pixels']} px)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    fig.suptitle("Demo 4 (1120.00 -> 2436.00) Synthesis Verification", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(VERIFICATION_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Bill Amount Check Figure
    orig_crop = original_rgb[295:345, 100:480]
    tamp_crop = tampered_rgb[295:345, 100:480]
    gt_crop = gt_mask[295:345, 100:480]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    axes[0].imshow(orig_crop)
    axes[0].set_title("ORIGINAL Bill Amount Field (Bill Amount 1120.00)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(tamp_crop)
    axes[1].set_title("TAMPERED Bill Amount Field (Bill Amount 2436.00)", fontsize=12, fontweight="bold", color="blue")
    axes[1].axis("off")

    overlay_crop = tamp_crop.copy().astype(np.float32) / 255.0
    mask_3d_crop = gt_crop[:, :, np.newaxis].astype(np.float32)
    overlay_crop = (1.0 - 0.7 * mask_3d_crop) * overlay_crop + 0.7 * mask_3d_crop * red_color
    axes[2].imshow(overlay_crop)
    axes[2].set_title(f"GT Mask Overlay ({metadata['gt_foreground_pixels']} foreground pixels)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(CHECK_IMAGE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_model_inference(
    tampered_rgb: np.ndarray,
    gt_mask: np.ndarray,
) -> dict:
    """Runs model inference on the synthesized image with checkpoints/best_model.pt."""
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

    # Standard preprocessing
    pil_img = Image.fromarray(tampered_rgb, mode="RGB")
    raw_tensor = TF.to_tensor(pil_img)
    norm_tensor = TF.normalize(raw_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    norm_x = norm_tensor.unsqueeze(0).to(device)
    raw_x = raw_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(norm_x, raw_x=raw_x)
        logits = outputs["tamper_logits"]

    logits_np = logits.squeeze().cpu().numpy()
    probs_np = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Binary prediction at threshold >= 0.5
    pred_mask = (probs_np >= 0.5).astype(np.uint8)

    # 1. Save Probability Map
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(probs_np, cmap="hot", vmin=0.0, vmax=max(probs_np.max(), 1e-4))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Demo 4 Sigmoid Probability Map (max={probs_np.max():.4f})", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(PROB_MAP_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Save Predicted Mask
    Image.fromarray(pred_mask * 255, mode="L").save(PRED_MASK_PATH, format="PNG")

    # 3. Save Comparison Figure
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].imshow(tampered_rgb)
    axes[0].set_title("1. Synthesized Document", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title(f"2. GT Mask ({int(np.sum(gt_mask))} px)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    im_prob = axes[2].imshow(probs_np, cmap="hot", vmin=0.0, vmax=max(probs_np.max(), 1e-4))
    plt.colorbar(im_prob, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title(f"3. Probabilities (max={probs_np.max():.4f})", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(pred_mask, cmap="gray")
    axes[3].set_title(f"4. Predicted Mask ({int(np.sum(pred_mask))} px >= 0.5)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    fig.suptitle("Demo 4 (2436.00) TamperDoc Model Inference & Localization", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(COMPARISON_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Quantitative Metrics
    gt_bool = gt_mask.astype(bool)
    bg_bool = ~gt_bool
    gt_probs = probs_np[gt_bool]
    bg_probs = probs_np[bg_bool]

    gt_mean_prob = float(gt_probs.mean()) if len(gt_probs) > 0 else 0.0
    gt_max_prob = float(gt_probs.max()) if len(gt_probs) > 0 else 0.0
    bg_mean_prob = float(bg_probs.mean()) if len(bg_probs) > 0 else 0.0
    bg_max_prob = float(bg_probs.max()) if len(bg_probs) > 0 else 0.0
    gt_px_above_05 = int(np.sum(gt_probs >= 0.5)) if len(gt_probs) > 0 else 0
    pred_px_above_05 = int(np.sum(pred_mask))

    gt_bg_ratio = gt_mean_prob / bg_mean_prob if bg_mean_prob > 1e-12 else float("inf")

    diagnostics = {
        "global_logit_min": float(logits_np.min()),
        "global_logit_max": float(logits_np.max()),
        "global_logit_mean": float(logits_np.mean()),
        "global_prob_min": float(probs_np.min()),
        "global_prob_max": float(probs_np.max()),
        "global_prob_mean": float(probs_np.mean()),
        "gt_pixel_count": int(np.sum(gt_mask)),
        "gt_mean_probability": gt_mean_prob,
        "gt_max_probability": gt_max_prob,
        "gt_pixels_ge_05": gt_px_above_05,
        "gt_fraction_detected": float(gt_px_above_05 / max(int(np.sum(gt_mask)), 1)),
        "bg_mean_probability": bg_mean_prob,
        "bg_max_probability": bg_max_prob,
        "predicted_pixels_ge_05": pred_px_above_05,
        "gt_bg_probability_ratio": gt_bg_ratio,
    }

    # 4. Save Diagnostics JSON
    with open(DIAGNOSTICS_PATH, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    return diagnostics


def print_final_report(metadata: dict, diagnostics: dict) -> None:
    """Prints clear, un-hardcoded, exact measured results."""
    print("\n" + "=" * 80)
    print("DEMO 4 COMPREHENSIVE EXPERIMENT REPORT (Bill Amount 1120.00 -> 2436.00)")
    print("=" * 80)

    print("\n[A] SOURCE & TARGET COORDINATES")
    for s in metadata["per_digit_stats"]:
        src = s["source_bbox"]
        tgt = s["target_bbox"]
        print(f"    Pos {s['pos_id']} ({s['original_digit']} -> {s['replacement_digit']}):")
        print(f"      Source: {s['source_desc']}")
        print(f"              y:[{src['y1']},{src['y2']}], x:[{src['x1']},{src['x2']}]  (h={src['y2']-src['y1']}, w={src['x2']-src['x1']})")
        print(f"      Target: y:[{tgt['y1']},{tgt['y2']}], x:[{tgt['x1']},{tgt['x2']}]  (h={tgt['y2']-tgt['y1']}, w={tgt['x2']-tgt['x1']})")
        print(f"      Source FG Pixels (raw): {s['source_fg_pixels_raw']} | Target Slot FG: {s['source_fg_pixels_resized']}")

    print("\n[B] SYNTHESIS QUANTITATIVE METRICS")
    print(f"    Total Ground-Truth Pixels:      {metadata['gt_foreground_pixels']:,} ({metadata['gt_foreground_pct']:.4f}%)")
    print(f"    Total Changed Pixels:           {metadata['changed_pixel_count']:,} ({metadata['changed_pixel_pct']:.4f}%)")
    cb = metadata["changed_bbox"]
    print(f"    Changed Bounding Box:           y:[{cb['y_min']},{cb['y_max']}], x:[{cb['x_min']},{cb['x_max']}]")

    print("\n[C] MODEL INFERENCE & LOCALIZATION RESULTS")
    print(f"    Maximum Probability (Global):   {diagnostics['global_prob_max']:.6f} ({diagnostics['global_prob_max']*100:.4f}%)")
    print(f"    GT-Region Mean Probability:     {diagnostics['gt_mean_probability']:.6f} ({diagnostics['gt_mean_probability']*100:.4f}%)")
    print(f"    GT-Region Max Probability:      {diagnostics['gt_max_probability']:.6f} ({diagnostics['gt_max_probability']*100:.4f}%)")
    print(f"    Background Mean Probability:    {diagnostics['bg_mean_probability']:.6f} ({diagnostics['bg_mean_probability']*100:.4f}%)")
    print(f"    GT / Background Ratio:          {diagnostics['gt_bg_probability_ratio']:.2f}x")
    print(f"    GT Pixels >= 0.5:               {diagnostics['gt_pixels_ge_05']:,} / {diagnostics['gt_pixel_count']:,} ({diagnostics['gt_fraction_detected']*100:.2f}%)")
    print(f"    Total Predicted Pixels >= 0.5:  {diagnostics['predicted_pixels_ge_05']:,}")

    print("\n[D] ARTIFACT LOCATIONS")
    print(f"    Synthesized Document:           {SYNTHESIZED_IMAGE_PATH}")
    print(f"    Ground-Truth Mask:              {GT_MASK_PATH}")
    print(f"    Synthesis Verification:         {VERIFICATION_PATH}")
    print(f"    Bill Amount Check Image:        {CHECK_IMAGE_PATH}")
    print(f"    Probability Map:                {PROB_MAP_PATH}")
    print(f"    Predicted Mask:                 {PRED_MASK_PATH}")
    print(f"    Comparison Figure:              {COMPARISON_PATH}")
    print(f"    Diagnostics JSON:               {DIAGNOSTICS_PATH}")

    print("\n" + "=" * 80)
    print("DEMO 4 EXECUTION COMPLETE")
    print("=" * 80)


def main():
    print("=" * 80)
    print("STARTING DEMO 4: 1120.00 -> 2436.00 SYNTHESIS & INFERENCE")
    print("=" * 80)

    # 1. Validate Input
    print("[1/4] Validating input image...")
    img_rgb = validate_input_image(INPUT_IMAGE_PATH)
    print("  [OK] Input image valid (512x512 RGB JPEG)")

    # 2. Synthesize
    print("\n[2/4] Synthesizing tampered image (1120.00 -> 2436.00)...")
    tampered_rgb, gt_mask, metadata, sauvola_fg = synthesize_tampered_document(img_rgb)
    save_synthesis_outputs(img_rgb, sauvola_fg, tampered_rgb, gt_mask, metadata)
    print(f"  [OK] Synthesized outputs saved to {OUTPUT_DIR}")

    # 3. Model Inference
    print("\n[3/4] Running model inference with checkpoints/best_model.pt...")
    diagnostics = run_model_inference(tampered_rgb, gt_mask)
    print(f"  [OK] Inference outputs saved to {INFERENCE_DIR}")

    # 4. Report
    print("\n[4/4] Generating report...")
    print_final_report(metadata, diagnostics)


if __name__ == "__main__":
    main()
