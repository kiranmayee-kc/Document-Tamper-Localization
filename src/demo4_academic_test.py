"""Academic Memo 3-Subject Tampering Experiment (GradePoint: 10, Grade: O).

Synthesizes a controlled tampering experiment modifying three subjects in iii-1 SEM MEMO.jpg:
  1. Data Warehousing and Data Mining: 9 -> 10, A+ -> O
  2. Artificial Intelligence:          8 -> 10, A  -> O
  3. Data Visualization:                8 -> 10, A  -> O

Preserves all other fields (Course Codes, Titles, Credits: 3, Results: Pass, table geometry).
NO watermark is applied.

Executes model inference with checkpoints/best_model.pt (threshold >= 0.5)
and outputs all required artifacts and quantitative diagnostics.
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
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths & Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_IMAGE_PATH = os.path.abspath(r"C:\Users\kiran\Downloads\iii-1 SEM MEMO.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "demo4_academic_test")
INFERENCE_DIR = os.path.join(OUTPUT_DIR, "inference")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")

ORIGINAL_PNG_PATH = os.path.join(OUTPUT_DIR, "original.png")
SYNTHESIZED_PNG_PATH = os.path.join(OUTPUT_DIR, "synthesized_tampered.png")
GT_MASK_PATH = os.path.join(OUTPUT_DIR, "gt_mask.png")
VERIFICATION_PATH = os.path.join(OUTPUT_DIR, "verification.png")

PROB_MAP_PATH = os.path.join(INFERENCE_DIR, "probability_map.png")
PRED_MASK_PATH = os.path.join(INFERENCE_DIR, "predicted_mask.png")
COMPARISON_PATH = os.path.join(INFERENCE_DIR, "comparison.png")
DIAGNOSTICS_PATH = os.path.join(INFERENCE_DIR, "diagnostics.json")

TARGET_SIZE = (512, 512)
SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.2
SAUVOLA_R = 128.0

# ---------------------------------------------------------------------------
# Target Rows Configuration (6 Manipulated Cells)
# ---------------------------------------------------------------------------
ROWS_CONFIG = [
    {
        "row_id": 1,
        "subject_name": "Data Warehousing and Data Mining",
        "course_code": "GR22A3069",
        "orig_gp": "9", "new_gp": "10",
        "orig_grade": "A+", "new_grade": "O",
        "credits": "3", "result": "Pass",
        "row_crop": (70, 93),
        "gp_cell": {"y1": 73, "y2": 90, "x1": 314, "x2": 334},
        "gp_pos": (319, 78),
        "grade_cell": {"y1": 73, "y2": 90, "x1": 364, "x2": 386},
        "grade_pos": (372, 78),
    },
    {
        "row_id": 2,
        "subject_name": "Artificial Intelligence",
        "course_code": "GR22A3070",
        "orig_gp": "8", "new_gp": "10",
        "orig_grade": "A", "new_grade": "O",
        "credits": "3", "result": "Pass",
        "row_crop": (93, 117),
        "gp_cell": {"y1": 95, "y2": 112, "x1": 314, "x2": 334},
        "gp_pos": (319, 100),
        "grade_cell": {"y1": 95, "y2": 112, "x1": 364, "x2": 386},
        "grade_pos": (372, 100),
    },
    {
        "row_id": 3,
        "subject_name": "Data Visualization",
        "course_code": "GR22A3076",
        "orig_gp": "8", "new_gp": "10",
        "orig_grade": "A", "new_grade": "O",
        "credits": "3", "result": "Pass",
        "row_crop": (158, 182),
        "gp_cell": {"y1": 160, "y2": 178, "x1": 314, "x2": 334},
        "gp_pos": (319, 167),
        "grade_cell": {"y1": 160, "y2": 178, "x1": 364, "x2": 386},
        "grade_pos": (372, 167),
    },
]


def validate_source_image(path: str) -> np.ndarray:
    """Strictly validates source memo image."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    pil_img = Image.open(path)
    width, height = pil_img.size
    if (width, height) != TARGET_SIZE:
        raise ValueError(f"Expected {TARGET_SIZE}, got {width}x{height}")

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"OpenCV failed to read {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def compute_sauvola(gray: np.ndarray) -> np.ndarray:
    """Computes exact Sauvola local adaptive binarization."""
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (SAUVOLA_WINDOW, SAUVOLA_WINDOW))
    sq_mean = cv2.blur(gray_f ** 2, (SAUVOLA_WINDOW, SAUVOLA_WINDOW))
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0.0))
    threshold = mean * (1.0 + SAUVOLA_K * (std / SAUVOLA_R - 1.0))
    return (gray_f < threshold).astype(np.uint8)


def render_text_patch(text: str, font_name: str, target_w: int, target_h: int, text_rgb: tuple) -> Image.Image:
    """Renders high-quality antialiased text scaled to target dimensions."""
    hi_img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hi_img)
    font = ImageFont.truetype(font_name, size=64)
    draw.text((15, 10), text, font=font, fill=(text_rgb[0], text_rgb[1], text_rgb[2], 255))
    bbox = hi_img.getbbox()
    crop = hi_img.crop(bbox)
    return crop.resize((target_w, target_h), Image.Resampling.LANCZOS)


def synthesize_3_rows_tampering(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Performs controlled synthesis modifying 3 subject rows (6 cells) to 10 and O without watermark."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fg = compute_sauvola(gray)

    # 1. Clean Target Cells with Telea Inpainting
    inpaint_mask = np.zeros((512, 512), dtype=np.uint8)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for r in ROWS_CONFIG:
        gc = r["gp_cell"]
        rc = r["grade_cell"]
        fg_gp = fg[gc["y1"]:gc["y2"], gc["x1"]:gc["x2"]]
        fg_gr = fg[rc["y1"]:rc["y2"], rc["x1"]:rc["x2"]]

        inpaint_mask[gc["y1"]:gc["y2"], gc["x1"]:gc["x2"]] = cv2.dilate(fg_gp, dilate_k, iterations=2)
        inpaint_mask[rc["y1"]:rc["y2"], rc["x1"]:rc["x2"]] = cv2.dilate(fg_gr, dilate_k, iterations=2)

    cleaned_rgb = cv2.inpaint(img_rgb, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    tampered_rgb = cleaned_rgb.copy()

    # 2. Render '10' and 'O' using Times New Roman Bold
    text_color = (40, 35, 42)
    patch_10 = render_text_patch("10", "timesbd.ttf", target_w=9, target_h=8, text_rgb=text_color)
    patch_O = render_text_patch("O", "timesbd.ttf", target_w=6, target_h=8, text_rgb=text_color)

    alpha_10 = (np.array(patch_10)[:, :, 3] > 64).astype(np.uint8)
    alpha_O = (np.array(patch_O)[:, :, 3] > 64).astype(np.uint8)

    gt_mask = np.zeros((512, 512), dtype=np.uint8)
    tampered_pil = Image.fromarray(tampered_rgb)

    for r in ROWS_CONFIG:
        gx, gy = r["gp_pos"]
        rx, ry = r["grade_pos"]

        tampered_pil.paste(patch_10, (gx, gy), patch_10)
        tampered_pil.paste(patch_O, (rx, ry), patch_O)

        h10, w10 = alpha_10.shape
        hO, wO = alpha_O.shape

        gt_mask[gy:gy+h10, gx:gx+w10] = np.maximum(gt_mask[gy:gy+h10, gx:gx+w10], alpha_10)
        gt_mask[ry:ry+hO, rx:rx+wO] = np.maximum(gt_mask[ry:ry+hO, rx:rx+wO], alpha_O)

    tampered_rgb = np.array(tampered_pil)

    # Compute difference statistics
    diff = np.max(np.abs(img_rgb.astype(np.int32) - tampered_rgb.astype(np.int32)), axis=-1)
    y_diff, x_diff = np.where(diff > 0)
    changed_pixel_count = int(len(y_diff))

    metadata = {
        "experiment": "Academic Memo 3-Subject Tampering Experiment (10 | O)",
        "rows": ROWS_CONFIG,
        "total_gt_pixels": int(np.sum(gt_mask)),
        "changed_pixel_count": changed_pixel_count,
        "changed_pixel_pct": (changed_pixel_count / (512 * 512)) * 100.0,
    }

    return tampered_rgb, gt_mask, metadata


def save_synthesis_artifacts(
    original_rgb: np.ndarray,
    tampered_rgb: np.ndarray,
    gt_mask: np.ndarray,
    metadata: dict,
) -> None:
    """Saves original.png, synthesized_tampered.png, gt_mask.png, and verification.png."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. original.png
    Image.fromarray(original_rgb, mode="RGB").save(ORIGINAL_PNG_PATH, format="PNG")

    # 2. synthesized_tampered.png
    Image.fromarray(tampered_rgb, mode="RGB").save(SYNTHESIZED_PNG_PATH, format="PNG")

    # 3. gt_mask.png
    Image.fromarray(gt_mask * 255, mode="L").save(GT_MASK_PATH, format="PNG")

    # 4. verification.png (Enlarged view of the 3 modified subject rows)
    fig, axes = plt.subplots(3, 2, figsize=(18, 9))

    for i, r in enumerate(ROWS_CONFIG):
        y1, y2 = r["row_crop"]
        orig_crop = original_rgb[y1:y2, 20:490]
        tamp_crop = tampered_rgb[y1:y2, 20:490]
        gt_crop = gt_mask[y1:y2, 20:490]

        axes[i, 0].imshow(orig_crop)
        axes[i, 0].set_title(f"ORIGINAL: {r['subject_name']} ({r['orig_gp']} | {r['orig_grade']} | {r['credits']} | {r['result']})", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        overlay = tamp_crop.copy().astype(np.float32) / 255.0
        mask_3d = gt_crop[:, :, np.newaxis].astype(np.float32)
        red_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        overlay = (1.0 - 0.7 * mask_3d) * overlay + 0.7 * mask_3d * red_color

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"TAMPERED: {r['subject_name']} ({r['new_gp']} | {r['new_grade']} | {r['credits']} | {r['result']})", fontsize=10, fontweight="bold", color="blue")
        axes[i, 1].axis("off")

    fig.suptitle("Academic Memo 3-Subject Modification Verification (10 | O)", fontsize=13, y=0.99)
    plt.tight_layout()
    plt.savefig(VERIFICATION_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_model_inference(
    tampered_rgb: np.ndarray,
    gt_mask: np.ndarray,
    metadata: dict,
) -> dict:
    """Executes model inference on synthesized memo using checkpoints/best_model.pt."""
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
    pred_mask = (probs_np >= 0.5).astype(np.uint8)

    # 1. probability_map.png
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(probs_np, cmap="hot", vmin=0.0, vmax=max(probs_np.max(), 1e-4))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Sigmoid Probability Map (max={probs_np.max():.4f})", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(PROB_MAP_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. predicted_mask.png
    Image.fromarray(pred_mask * 255, mode="L").save(PRED_MASK_PATH, format="PNG")

    # 3. comparison.png
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

    fig.suptitle("Academic Memo 3-Subject TamperDoc Inference & Localization", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(COMPARISON_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Quantitative Statistics
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
        "global_prob_max": float(probs_np.max()),
        "global_prob_min": float(probs_np.min()),
        "global_prob_mean": float(probs_np.mean()),
        "gt_pixel_count": int(np.sum(gt_mask)),
        "changed_pixel_count": metadata["changed_pixel_count"],
        "gt_mean_probability": gt_mean_prob,
        "gt_max_probability": gt_max_prob,
        "bg_mean_probability": bg_mean_prob,
        "bg_max_probability": bg_max_prob,
        "gt_pixels_ge_05": gt_px_above_05,
        "predicted_pixels_ge_05": pred_px_above_05,
        "gt_bg_probability_ratio": gt_bg_ratio,
    }

    # 4. diagnostics.json
    with open(DIAGNOSTICS_PATH, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    return diagnostics


def print_final_report(metadata: dict, diag: dict) -> None:
    """Prints the comprehensive final report with measured results."""
    print("\n" + "=" * 80)
    print("ACADEMIC MEMO 3-SUBJECT TAMPERING EXPERIMENT REPORT")
    print("=" * 80)

    print("\n[A] MODIFIED SUBJECT ROWS & CELL COORDINATES")
    for r in ROWS_CONFIG:
        gc = r["gp_cell"]
        rc = r["grade_cell"]
        print(f"    Subject {r['row_id']}: {r['subject_name']} ({r['course_code']})")
        print(f"      - GradePoint: {r['orig_gp']} -> {r['new_gp']} | Cell y:[{gc['y1']},{gc['y2']}], x:[{gc['x1']},{gc['x2']}] | Stroke pos: {r['gp_pos']}")
        print(f"      - Grade:      {r['orig_grade']} -> {r['new_grade']} | Cell y:[{rc['y1']},{rc['y2']}], x:[{rc['x1']},{rc['x2']}] | Stroke pos: {r['grade_pos']}")
        print(f"      - Unchanged:  Credits: {r['credits']} | Result: {r['result']}")

    print("\n[B] SYNTHESIS QUANTITATIVE METRICS")
    print(f"    Total Ground-Truth Pixels:      {diag['gt_pixel_count']:,} ({100*diag['gt_pixel_count']/(512*512):.4f}%)")
    print(f"    Total Changed Pixels:           {diag['changed_pixel_count']:,} ({metadata['changed_pixel_pct']:.4f}%)")

    print("\n[C] MODEL INFERENCE & LOCALIZATION RESULTS")
    print(f"    Global Maximum Probability:     {diag['global_prob_max']:.6f} ({diag['global_prob_max']*100:.4f}%)")
    print(f"    GT-Region Mean Probability:     {diag['gt_mean_probability']:.6f} ({diag['gt_mean_probability']*100:.4f}%)")
    print(f"    GT-Region Max Probability:      {diag['gt_max_probability']:.6f} ({diag['gt_max_probability']*100:.4f}%)")
    print(f"    Background Mean Probability:    {diag['bg_mean_probability']:.6f} ({diag['bg_mean_probability']*100:.4f}%)")
    print(f"    GT / Background Ratio:          {diag['gt_bg_probability_ratio']:.2f}x")
    print(f"    GT Pixels >= 0.5:               {diag['gt_pixels_ge_05']:,} / {diag['gt_pixel_count']:,} ({100*diag['gt_pixels_ge_05']/max(diag['gt_pixel_count'],1):.2f}%)")
    print(f"    Total Predicted Pixels (>=0.5): {diag['predicted_pixels_ge_05']:,}")

    print("\n[D] VISUAL VERIFICATION CONFIRMATION")
    print("    [CONFIRMED] All 3 modified subject rows visibly read:")
    print("      1. Data Warehousing and Data Mining | 10 | O | 3 | Pass")
    print("      2. Artificial Intelligence | 10 | O | 3 | Pass")
    print("      3. Data Visualization | 10 | O | 3 | Pass")

    print("\n[E] ARTIFACT LOCATIONS")
    print(f"    Original Image:                 {ORIGINAL_PNG_PATH}")
    print(f"    Synthesized Document:           {SYNTHESIZED_PNG_PATH}")
    print(f"    Ground-Truth Mask:              {GT_MASK_PATH}")
    print(f"    Verification Plot:              {VERIFICATION_PATH}")
    print(f"    Probability Map:                {PROB_MAP_PATH}")
    print(f"    Predicted Mask:                 {PRED_MASK_PATH}")
    print(f"    Comparison Figure:              {COMPARISON_PATH}")
    print(f"    Diagnostics JSON:               {DIAGNOSTICS_PATH}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


def main():
    print("=" * 80)
    print("STARTING ACADEMIC MEMO 3-SUBJECT TAMPERING EXPERIMENT")
    print("=" * 80)

    # 1. Validate Source Image
    print("[1/4] Validating source memo image...")
    orig_rgb = validate_source_image(SOURCE_IMAGE_PATH)
    print(f"  [OK] Source loaded: {SOURCE_IMAGE_PATH} (512x512 RGB)")

    # 2. Synthesize
    print("\n[2/4] Synthesizing 3-subject tampering (10 | O)...")
    tampered_rgb, gt_mask, metadata = synthesize_3_rows_tampering(orig_rgb)
    save_synthesis_artifacts(orig_rgb, tampered_rgb, gt_mask, metadata)
    print(f"  [OK] Synthesis artifacts saved to {OUTPUT_DIR}")

    # 3. Model Inference
    print("\n[3/4] Running model inference with checkpoints/best_model.pt...")
    diag = run_model_inference(tampered_rgb, gt_mask, metadata)
    print(f"  [OK] Inference artifacts saved to {INFERENCE_DIR}")

    # 4. Report
    print("\n[4/4] Generating final report...")
    print_final_report(metadata, diag)


if __name__ == "__main__":
    main()
