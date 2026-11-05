"""DocTamper Demo 3: Corrected Multi-Character Selective Tampering Synthesis Tool.

Deterministic, single-image reproduction of the official DocTamper (CVPR 2023)
Selective Tampering Synthesis methodology — replaces Bill Amount '1120.00' -> '2436.00'
by replacing 4 individual digits (positions 1-4) via per-character selective copy-paste
from confirmed real digit strokes in the same original document.

Methodology:
1. Input Verification: Strict 512x512 RGB JPEG validation of results/demo_inputs/og current bill.jpg.
2. Per-Character Selective Copy-Paste: 4 independent digit replacements (1->2, 1->4, 2->3, 0->6).
3. Sauvola Foreground Extraction: Local window binarization (w=25, k=0.2, R=128).
4. Target Text Removal: Classical OpenCV inpainting (cv2.INPAINT_TELEA, radius=3) on target
   digit strokes prior to source stroke placement.
5. Stroke-Only Transfer: Transfers ONLY foreground digit strokes into the cleaned target slot.
   The rectangular RGB source crop is NEVER pasted, and source paper background is discarded.
6. Ground-Truth Mask Generation: Exact 512x512 uint8 binary mask ({0, 1}) marking inserted strokes.
7. Validation & Verification: Programmatic validation + 4-panel visual report + enlarged check image.

Confirmed Real Source Characters in og current bill.jpg:
  '2' -> y:[308, 330], x:[398, 414]  (from original Bill Amount field pos 3)
  '4' -> y:[55, 75],   x:[435, 449]  (from '11.46' in Electricity Duty row)
  '3' -> y:[276, 298], x:[440, 455]  (from '-0.36' in Loss/Gain row)
  '6' -> y:[55, 75],   x:[449, 463]  (from '11.46' in Electricity Duty row)

NO generative AI, diffusion models, neural inpainting, font rendering, or AI image editing
is used.
"""

import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "og current bill.jpg")
OUTPUT_TAMPERED_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "doctamper_synthesized_tampered_v3.jpg")
OUTPUT_MASK_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "doctamper_synthesized_mask_v3.png")
OUTPUT_VERIF_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "synthesis_verification_v3.png")
OUTPUT_CHECK_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "demo3_bill_amount_check.png")

TARGET_SIZE = (512, 512)

# Sauvola Parameters (as documented in DocTamper CVPR 2023)
SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.2
SAUVOLA_R = 128.0

# ---------------------------------------------------------------------------
# Per-Character Replacement Configuration (1120.00 -> 2436.00)
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
    """Strictly validates that the input image exists, is 512x512, RGB, and JPEG."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    pil_img = Image.open(path)
    img_format = (pil_img.format or "").upper()
    if img_format not in ("JPEG", "JPG"):
        raise ValueError(f"Input image must be a JPEG, got format: '{img_format}'")

    width, height = pil_img.size
    if (width, height) != TARGET_SIZE:
        raise ValueError(
            f"Input dimensions must be exactly {TARGET_SIZE[0]}x{TARGET_SIZE[1]}, "
            f"got {width}x{height}. Silent resizing is prohibited."
        )

    if pil_img.mode != "RGB":
        raise ValueError(f"Input image mode must be RGB, got '{pil_img.mode}'")

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Failed to read image with OpenCV: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def compute_sauvola_binarization(
    gray: np.ndarray,
    window_size: int = SAUVOLA_WINDOW,
    k: float = SAUVOLA_K,
    R: float = SAUVOLA_R,
) -> np.ndarray:
    """Computes exact Sauvola local adaptive binarization for document text."""
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (window_size, window_size))
    sq_mean = cv2.blur(gray_f ** 2, (window_size, window_size))
    variance = np.maximum(sq_mean - mean ** 2, 0.0)
    std = np.sqrt(variance)
    threshold = mean * (1.0 + k * (std / R - 1.0))
    binary_fg = (gray_f < threshold).astype(np.uint8)
    return binary_fg


def synthesize_tampered_document_v3(
    img_rgb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Applies stroke-only selective copy-paste synthesis for Demo 3 (1120.00 -> 2436.00)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sauvola_fg = compute_sauvola_binarization(gray)

    # 1. Build Inpaint Mask to clean target slots
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    global_inpaint_mask = np.zeros((512, 512), dtype=np.uint8)
    gt_mask = np.zeros((512, 512), dtype=np.uint8)

    per_digit_stats: List[dict] = []

    for rep in DIGIT_REPLACEMENTS:
        t = rep["target"]
        s = rep["source"]
        ty1, ty2, tx1, tx2 = t["y1"], t["y2"], t["x1"], t["x2"]
        target_h, target_w = ty2 - ty1, tx2 - tx1

        # Target foreground
        target_fg_crop = sauvola_fg[ty1:ty2, tx1:tx2]
        target_fg_pixel_count = int(np.sum(target_fg_crop))

        # Dilate target foreground by 2 iterations to cleanly remove target text
        dilated_target_fg = cv2.dilate(target_fg_crop, dilate_kernel, iterations=2)
        global_inpaint_mask[ty1:ty2, tx1:tx2] = np.maximum(
            global_inpaint_mask[ty1:ty2, tx1:tx2], dilated_target_fg
        )

        # Source foreground
        sy1, sy2, sx1, sx2 = s["y1"], s["y2"], s["x1"], s["x2"]
        source_fg_crop_raw = sauvola_fg[sy1:sy2, sx1:sx2]
        source_fg_pixel_count_raw = int(np.sum(source_fg_crop_raw))

        # Resize stroke mask using nearest neighbor
        source_fg_crop = cv2.resize(source_fg_crop_raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        source_fg_pixel_count_resized = int(np.sum(source_fg_crop))

        per_digit_stats.append({
            "pos_id": rep["pos_id"],
            "original_digit": rep["original_digit"],
            "replacement_digit": rep["replacement_digit"],
            "target_bbox": t,
            "source_bbox": s,
            "source_desc": rep["source_desc"],
            "target_shape_hw": (target_h, target_w),
            "source_shape_hw": (sy2 - sy1, sx2 - sx1),
            "target_fg_pixels": target_fg_pixel_count,
            "source_fg_pixels_raw": source_fg_pixel_count_raw,
            "source_fg_pixels_resized": source_fg_pixel_count_resized,
            "resize_applied": not ((sy2 - sy1) == target_h and (sx2 - sx1) == target_w),
        })

    # 2. Clean Target Digits with OpenCV Telea Inpainting (radius=3)
    cleaned_rgb = cv2.inpaint(
        img_rgb, global_inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )
    tampered_rgb = cleaned_rgb.copy()

    # 3. Stroke-Only Transfer (source rectangular crop is NEVER pasted)
    for rep, stats in zip(DIGIT_REPLACEMENTS, per_digit_stats):
        t = rep["target"]
        s = rep["source"]
        ty1, ty2, tx1, tx2 = t["y1"], t["y2"], t["x1"], t["x2"]
        target_h, target_w = ty2 - ty1, tx2 - tx1
        sy1, sy2, sx1, sx2 = s["y1"], s["y2"], s["x1"], s["x2"]

        src_rgb_crop = img_rgb[sy1:sy2, sx1:sx2]
        src_fg_crop = sauvola_fg[sy1:sy2, sx1:sx2]

        # Resize stroke mask
        stroke_mask = cv2.resize(src_fg_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Resize source RGB strokes
        src_h, src_w = sy2 - sy1, sx2 - sx1
        interp = cv2.INTER_AREA if (src_h >= target_h and src_w >= target_w) else cv2.INTER_LINEAR
        src_rgb_resized = cv2.resize(src_rgb_crop, (target_w, target_h), interpolation=interp)

        # Transfer ONLY foreground stroke pixels onto cleaned background
        fg_indices = (stroke_mask == 1)
        target_slot = tampered_rgb[ty1:ty2, tx1:tx2]
        target_slot[fg_indices] = src_rgb_resized[fg_indices]
        tampered_rgb[ty1:ty2, tx1:tx2] = target_slot

        # Mark in GT mask (values in {0, 1})
        gt_mask[ty1:ty2, tx1:tx2] = np.maximum(gt_mask[ty1:ty2, tx1:tx2], stroke_mask)

    # 4. Compute differences & statistics
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
        "transformation": "1120.00 -> 2436.00 (Bill Amount field, 4 digits replaced)",
        "per_digit_stats": per_digit_stats,
        "sauvola_params": {"window": SAUVOLA_WINDOW, "k": SAUVOLA_K, "R": SAUVOLA_R},
        "inpaint_method": "cv2.INPAINT_TELEA (radius=3)",
        "inpaint_reason": "Background restoration under removed target strokes before stroke transfer",
        "transfer_method": "Stroke-only foreground placement (source background discarded)",
        "changed_pixel_count": changed_pixel_count,
        "changed_pixel_pct": changed_pixel_pct,
        "changed_bbox": changed_bbox,
        "gt_foreground_pixels": int(np.sum(gt_mask)),
        "gt_foreground_pct": (float(np.sum(gt_mask)) / (512 * 512)) * 100.0,
    }

    return tampered_rgb, gt_mask, metadata, sauvola_fg


def save_verification_figures(
    original_rgb: np.ndarray,
    sauvola_fg: np.ndarray,
    tampered_rgb: np.ndarray,
    gt_mask: np.ndarray,
    metadata: dict,
    verif_path: str,
    check_path: str,
) -> None:
    """Generates the 4-panel verification plot and enlarged Bill Amount check image."""
    # 1. 4-panel synthesis verification figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original_rgb)
    axes[0].set_title("1. Original Document", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(sauvola_fg, cmap="gray")
    axes[1].set_title(f"2. Sauvola Text Mask (w={SAUVOLA_WINDOW}, k={SAUVOLA_K})", fontsize=11, fontweight="bold")
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
    axes[3].set_title(f"4. GT Mask Overlay ({int(np.sum(gt_mask))} px)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    fig.suptitle("DocTamper Demo 3 (1120.00 -> 2436.00) Synthesis Verification", fontsize=13, y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(verif_path), exist_ok=True)
    plt.savefig(verif_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Enlarged Bill Amount check figure
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
    axes[2].set_title(f"GT Mask Overlay ({int(np.sum(gt_mask))} foreground stroke pixels)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(check_path), exist_ok=True)
    plt.savefig(check_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_post_synthesis_validation(
    original_rgb: np.ndarray,
    tampered_rgb: np.ndarray,
    gt_mask: np.ndarray,
    metadata: dict,
) -> None:
    """Runs automated programmatic validation against all required constraints."""
    # 1. Original document must remain unchanged
    assert os.path.exists(INPUT_IMAGE_PATH), "Original input image missing!"
    reloaded_orig = np.asarray(Image.open(INPUT_IMAGE_PATH).convert("RGB"))
    assert np.array_equal(original_rgb, reloaded_orig), "Original image was modified!"

    # 2. Tampered image shape/dtype
    assert tampered_rgb.shape == (512, 512, 3), f"Tampered shape {tampered_rgb.shape} != (512, 512, 3)"
    assert tampered_rgb.dtype == np.uint8, "Tampered dtype must be uint8"
    assert not np.isnan(tampered_rgb.astype(np.float32)).any(), "Tampered image contains NaN"

    # 3. GT mask constraints
    assert gt_mask.shape == (512, 512), f"GT mask shape {gt_mask.shape} != (512, 512)"
    assert gt_mask.dtype == np.uint8, "GT mask dtype must be uint8"
    unique_vals = set(np.unique(gt_mask))
    assert unique_vals.issubset({0, 1}), f"GT mask unique values {unique_vals} not subset of {{0, 1}}"
    assert np.sum(gt_mask) > 0, "GT mask is empty!"
    assert np.sum(gt_mask) < (512 * 512), "GT mask covers the entire image!"

    # 4. Changed pixels constraints (localized to Bill Amount)
    assert metadata["changed_pixel_count"] > 0, "No pixels were changed!"
    assert metadata["changed_pixel_pct"] < 5.0, "Too many pixels changed (must be localized text edit)!"

    # 5. Overlap check
    diff = np.max(np.abs(original_rgb.astype(np.int32) - tampered_rgb.astype(np.int32)), axis=-1)
    diff_mask = (diff > 0).astype(np.uint8)
    overlap = np.sum(gt_mask & diff_mask)
    assert overlap > 0, "GT mask does not overlap with actual changed pixels!"

    # 6. Check that '.00' and other fields are unchanged
    # Dot at x:[430, 437], '00' at x:[438, 468]
    dot_zeros_orig = original_rgb[308:330, 430:470]
    dot_zeros_tamp = tampered_rgb[308:330, 430:470]
    assert np.array_equal(dot_zeros_orig, dot_zeros_tamp), "The '.00' field was modified!"


def print_synthesis_report(metadata: dict) -> None:
    """Prints the comprehensive synthesis report."""
    print("\n" + "=" * 80)
    print("DOCTAMPER DEMO 3 SYNTHESIS REPORT (1120.00 -> 2436.00)")
    print("=" * 80)

    print("\n[A] TRANSFORMATION & METHODOLOGY")
    print(f"    Transformation:       {metadata['transformation']}")
    print(f"    Methodology:          Selective Copy-Paste (stroke-only foreground transfer)")
    print(f"    Target Cleaning:      {metadata['inpaint_method']}")
    print(f"    Sauvola Parameters:   window={SAUVOLA_WINDOW}, k={SAUVOLA_K}, R={SAUVOLA_R}")
    print(f"    Source Paper Copied:  NO (strokes only, target background preserved)")
    print(f"    AI / Font Used:       NO (all strokes from real document pixels)")

    print("\n[B] PER-DIGIT REPLACEMENT DETAILS")
    for s in metadata["per_digit_stats"]:
        t = s["target_bbox"]
        src = s["source_bbox"]
        resize_note = "YES (resized to fit)" if s["resize_applied"] else "NO (same size)"
        print(f"    Pos {s['pos_id']}: '{s['original_digit']}' -> '{s['replacement_digit']}'")
        print(f"           Source: {s['source_desc']}")
        print(f"                   y:[{src['y1']},{src['y2']}], x:[{src['x1']},{src['x2']}]  ({s['source_shape_hw'][0]}x{s['source_shape_hw'][1]})")
        print(f"           Target: y:[{t['y1']},{t['y2']}], x:[{t['x1']},{t['x2']}]  ({s['target_shape_hw'][0]}x{s['target_shape_hw'][1]})")
        print(f"           Resize Applied: {resize_note}")
        print(f"           Source FG Pixels (raw): {s['source_fg_pixels_raw']} | Target Slot FG: {s['source_fg_pixels_resized']}")

    print("\n[C] QUANTITATIVE SYNTHESIS METRICS")
    print(f"    Total Changed Pixels: {metadata['changed_pixel_count']:,} ({metadata['changed_pixel_pct']:.4f}%)")
    print(f"    Total GT Pixels:      {metadata['gt_foreground_pixels']:,} ({metadata['gt_foreground_pct']:.4f}%)")
    cb = metadata["changed_bbox"]
    print(f"    Changed Bounding Box: y:[{cb['y_min']},{cb['y_max']}], x:[{cb['x_min']},{cb['x_max']}]")

    print("\n[D] ARTIFACTS GENERATED")
    print(f"    Tampered Document:    {OUTPUT_TAMPERED_PATH}")
    print(f"    Ground-Truth Mask:    {OUTPUT_MASK_PATH}")
    print(f"    Verification Figure:  {OUTPUT_VERIF_PATH}")
    print(f"    Bill Amount Check:    {OUTPUT_CHECK_PATH}")

    print("\n" + "=" * 80)
    print("SYNTHESIS COMPLETE — MODEL INFERENCE NOT EXECUTED (AWAITING VISUAL APPROVAL)")
    print("=" * 80)


def main():
    print("=" * 80)
    print("DOCTAMPER DEMO 3 — SYNTHESIZING BILL AMOUNT 1120.00 -> 2436.00")
    print("=" * 80)
    print(f"Input Document:       {INPUT_IMAGE_PATH}")
    print(f"Output Tampered:      {OUTPUT_TAMPERED_PATH}")
    print(f"Output GT Mask:       {OUTPUT_MASK_PATH}")
    print(f"Output Verification:  {OUTPUT_VERIF_PATH}")
    print(f"Output Check Image:   {OUTPUT_CHECK_PATH}")
    print("-" * 80)

    # 1. Validate Original
    print("\n[1/4] Validating input image...")
    original_rgb = validate_input_image(INPUT_IMAGE_PATH)
    print("  [OK] Input verified: 512x512 RGB JPEG")

    # 2. Synthesize
    print("\n[2/4] Executing stroke-only selective copy-paste synthesis...")
    tampered_rgb, gt_mask, metadata, sauvola_fg = synthesize_tampered_document_v3(original_rgb)
    print("  [OK] Synthesis complete.")

    # 3. Save Artifacts
    print("\n[3/4] Saving output files...")
    os.makedirs(os.path.dirname(OUTPUT_TAMPERED_PATH), exist_ok=True)

    tampered_pil = Image.fromarray(tampered_rgb, mode="RGB")
    tampered_pil.save(OUTPUT_TAMPERED_PATH, format="JPEG", quality=95)
    print(f"  [OK] Saved tampered JPEG -> {OUTPUT_TAMPERED_PATH}")

    mask_pil = Image.fromarray(gt_mask, mode="L")
    mask_pil.save(OUTPUT_MASK_PATH, format="PNG")
    print(f"  [OK] Saved GT Mask PNG   -> {OUTPUT_MASK_PATH}")

    save_verification_figures(
        original_rgb, sauvola_fg, tampered_rgb, gt_mask, metadata,
        OUTPUT_VERIF_PATH, OUTPUT_CHECK_PATH
    )
    print(f"  [OK] Saved verification  -> {OUTPUT_VERIF_PATH}")
    print(f"  [OK] Saved visual check  -> {OUTPUT_CHECK_PATH}")

    # 4. Programmatic Validation
    print("\n[4/4] Running automated validation checks...")
    run_post_synthesis_validation(original_rgb, tampered_rgb, gt_mask, metadata)
    print("  [OK] All validation assertions passed.")

    # Report
    print_synthesis_report(metadata)


if __name__ == "__main__":
    main()
