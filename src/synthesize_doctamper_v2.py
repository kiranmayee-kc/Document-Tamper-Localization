"""DocTamper Demo 2: Multi-Character Selective Tampering Synthesis Tool.

Deterministic, single-image reproduction of the official DocTamper (CVPR 2023)
Selective Tampering Synthesis methodology for a larger multi-character text region (3 digits).

Methodology:
1. Input Verification: Strict 512x512 RGB JPEG validation of results/demo_inputs/og current bill.jpg.
2. Text Region Identification: Selective Copy-Paste of 3-character field ('1 1 2' -> '0 0 0').
3. Sauvola Foreground Extraction: Local window binarization (w=25, k=0.2, R=128).
4. Target Text Removal: Classical OpenCV inpainting (cv2.INPAINT_TELEA, radius=3) on target text strokes.
5. Source Text Transfer: Deterministic transfer of source stroke pixels into the cleaned target region.
6. Ground-Truth Mask Generation: Exact 512x512 uint8 binary mask ({0, 1}) marking tampered pixels.
7. Validation & Verification: Programmatic validation of output dimensions, values, and visual report.

NO generative AI, diffusion models, neural inpainting, or AI image editing is used.
"""

import os
import sys
from typing import Dict, Tuple

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
INPUT_IMAGE_PATH = os.path.abspath("results/demo_inputs/og current bill.jpg")
OUTPUT_TAMPERED_PATH = os.path.abspath("results/demo_inputs/doctamper_synthesized_tampered_v2.jpg")
OUTPUT_MASK_PATH = os.path.abspath("results/demo_inputs/doctamper_synthesized_mask_v2.png")
OUTPUT_VERIF_PATH = os.path.abspath("results/demo_inputs/synthesis_verification_v2.png")

TARGET_SIZE = (512, 512)

# Sauvola Parameters (as documented in DocTamper CVPR 2023)
SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.2
SAUVOLA_R = 128.0

# Selective Copy-Paste Region Coordinates (3-character field within the same document)
# Target: 3 digits '0 0 0' in the charges table row at y in [147, 169], x in [419, 467] (22x48)
TARGET_BBOX = {"y1": 147, "y2": 169, "x1": 419, "x2": 467, "desc": "3-digit charges entry '0 0 0'"}

# Source: 3 digits '1 1 2' from the meter reading row at y in [308, 330], x in [368, 416] (22x48)
SOURCE_BBOX = {"y1": 308, "y2": 330, "x1": 368, "x2": 416, "desc": "3-digit reading prefix '1 1 2'"}


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
            f"Input dimensions must be exactly {TARGET_SIZE[0]}x{TARGET_SIZE[1]}, got {width}x{height}. "
            f"Silent resizing is prohibited."
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
    sq_mean = cv2.blur(gray_f**2, (window_size, window_size))
    variance = np.maximum(sq_mean - mean**2, 0.0)
    std = np.sqrt(variance)

    threshold = mean * (1.0 + k * (std / R - 1.0))
    binary_fg = (gray_f < threshold).astype(np.uint8)
    return binary_fg


def synthesize_tampered_document_v2(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Applies DocTamper Selective Copy-Paste synthesis for Demo 2 (3-digit replacement)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Sauvola Foreground Extraction
    sauvola_fg = compute_sauvola_binarization(
        gray, window_size=SAUVOLA_WINDOW, k=SAUVOLA_K, R=SAUVOLA_R
    )

    ty1, ty2, tx1, tx2 = TARGET_BBOX["y1"], TARGET_BBOX["y2"], TARGET_BBOX["x1"], TARGET_BBOX["x2"]
    sy1, sy2, sx1, sx2 = SOURCE_BBOX["y1"], SOURCE_BBOX["y2"], SOURCE_BBOX["x1"], SOURCE_BBOX["x2"]

    target_fg_crop = sauvola_fg[ty1:ty2, tx1:tx2]
    source_fg_crop = sauvola_fg[sy1:sy2, sx1:sx2]

    source_fg_pixel_count = int(np.sum(source_fg_crop))
    target_fg_pixel_count = int(np.sum(target_fg_crop))

    # 2. Target Text Removal (Classical OpenCV Telea Inpainting)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_target_fg = cv2.dilate(target_fg_crop, dilate_kernel, iterations=1)

    inpaint_mask = np.zeros((512, 512), dtype=np.uint8)
    inpaint_mask[ty1:ty2, tx1:tx2] = dilated_target_fg

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    cleaned_bgr = cv2.inpaint(img_rgb, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    cleaned_rgb = cleaned_bgr

    # 3. Source Text Transfer (Selective Copy-Paste)
    tampered_rgb = cleaned_rgb.copy()
    src_rgb_crop = img_rgb[sy1:sy2, sx1:sx2].copy()

    src_fg_3d = source_fg_crop[:, :, np.newaxis].astype(np.float32)
    cleaned_target_crop = cleaned_rgb[ty1:ty2, tx1:tx2].astype(np.float32)
    src_crop_f = src_rgb_crop.astype(np.float32)

    blended_target = (1.0 - src_fg_3d) * cleaned_target_crop + src_fg_3d * src_crop_f
    tampered_rgb[ty1:ty2, tx1:tx2] = np.clip(blended_target, 0, 255).astype(np.uint8)

    # 4. Ground-Truth Mask Generation (Strictly {0, 1}, single-channel uint8)
    gt_mask = np.zeros((512, 512), dtype=np.uint8)
    gt_mask[ty1:ty2, tx1:tx2] = source_fg_crop

    # Measure changed pixels
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
        "source_bbox": SOURCE_BBOX,
        "target_bbox": TARGET_BBOX,
        "source_fg_pixels": source_fg_pixel_count,
        "target_fg_pixels": target_fg_pixel_count,
        "inpaint_method": "cv2.INPAINT_TELEA (radius=3)",
        "inpaint_reason": "Cleanly restore local document paper texture under target text prior to stroke transfer",
        "changed_pixel_count": changed_pixel_count,
        "changed_pixel_pct": changed_pixel_pct,
        "changed_bbox": changed_bbox,
        "gt_foreground_pixels": int(np.sum(gt_mask)),
        "gt_foreground_pct": (float(np.sum(gt_mask)) / (512 * 512)) * 100.0,
    }
    return tampered_rgb, gt_mask, metadata


def save_verification_figure(
    original_rgb: np.ndarray,
    sauvola_fg: np.ndarray,
    tampered_rgb: np.ndarray,
    gt_mask: np.ndarray,
    save_path: str,
) -> None:
    """Generates a 4-panel verification figure for inspection."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original_rgb)
    axes[0].set_title("1. Original Document", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(sauvola_fg, cmap="gray")
    axes[1].set_title(f"2. Sauvola Text Mask (w={SAUVOLA_WINDOW}, k={SAUVOLA_K})", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(tampered_rgb)
    axes[2].set_title("3. Synthesized Tampered Document v2", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    overlay = tampered_rgb.copy().astype(np.float32) / 255.0
    mask_3d = gt_mask[:, :, np.newaxis].astype(np.float32)
    red_color = np.array([0.9, 0.1, 0.1], dtype=np.float32)
    alpha = 0.65 * mask_3d
    overlay = (1.0 - alpha) * overlay + alpha * red_color
    overlay = np.clip(overlay, 0.0, 1.0)

    axes[3].imshow(overlay)
    axes[3].set_title(f"4. GT Mask Overlay ({int(np.sum(gt_mask))} px)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    fig.suptitle("DocTamper Demo 2 (3-Digit Field) Synthesis Verification", fontsize=13, y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_post_synthesis_validation(
    original_rgb: np.ndarray,
    tampered_rgb: np.ndarray,
    gt_mask: np.ndarray,
    metadata: dict,
) -> None:
    """Runs automated programmatic validation against all required constraints."""
    assert os.path.exists(INPUT_IMAGE_PATH), "Original input image missing!"
    reloaded_orig = np.asarray(Image.open(INPUT_IMAGE_PATH).convert("RGB"))
    assert np.array_equal(original_rgb, reloaded_orig), "Original image was modified!"

    assert tampered_rgb.shape == (512, 512, 3), f"Tampered shape {tampered_rgb.shape} != (512, 512, 3)"
    assert tampered_rgb.dtype == np.uint8, "Tampered dtype must be uint8"
    assert not np.isnan(tampered_rgb).any(), "Tampered image contains NaN"
    assert not np.isinf(tampered_rgb).any(), "Tampered image contains Inf"

    assert gt_mask.shape == (512, 512), f"GT mask shape {gt_mask.shape} != (512, 512)"
    assert gt_mask.dtype == np.uint8, "GT mask dtype must be uint8"
    unique_vals = set(np.unique(gt_mask))
    assert unique_vals.issubset({0, 1}), f"GT mask unique values {unique_vals} not subset of {{0, 1}}"
    assert np.sum(gt_mask) > 0, "GT mask is empty!"
    assert np.sum(gt_mask) < (512 * 512), "GT mask covers the entire image!"

    assert metadata["changed_pixel_count"] > 0, "No pixels were changed!"
    assert metadata["changed_pixel_pct"] < 5.0, "Too many pixels changed (should be local text edit)!"

    diff = np.max(np.abs(original_rgb.astype(np.int32) - tampered_rgb.astype(np.int32)), axis=-1)
    diff_mask = (diff > 0).astype(np.uint8)
    overlap = np.sum(gt_mask & diff_mask)
    assert overlap > 0, "GT mask does not overlap with actual changed pixels!"


def main():
    print("=" * 80)
    print("DOCTAMPER DEMO 2 — MULTI-CHARACTER TAMPERING SYNTHESIS")
    print("=" * 80)
    print(f"Input Document:       {INPUT_IMAGE_PATH}")
    print(f"Output Tampered:      {OUTPUT_TAMPERED_PATH}")
    print(f"Output GT Mask:       {OUTPUT_MASK_PATH}")
    print(f"Output Verification:  {OUTPUT_VERIF_PATH}")
    print(f"Methodology:          Selective Copy-Paste (3-Character Field)")
    print("-" * 80)

    # 1. Validate & Load Original Document
    print("[1/5] Validating input image...")
    original_rgb = validate_input_image(INPUT_IMAGE_PATH)
    print(f"  [OK] Original image verified: 512x512 RGB JPEG")

    # 2. Execute Synthesis
    print("\n[2/5] Executing Selective Copy-Paste synthesis (3-digit field)...")
    print(f"  Target: {TARGET_BBOX['desc']} at y:[{TARGET_BBOX['y1']}, {TARGET_BBOX['y2']}], x:[{TARGET_BBOX['x1']}, {TARGET_BBOX['x2']}] (size: {TARGET_BBOX['y2']-TARGET_BBOX['y1']}x{TARGET_BBOX['x2']-TARGET_BBOX['x1']})")
    print(f"  Source: {SOURCE_BBOX['desc']} at y:[{SOURCE_BBOX['y1']}, {SOURCE_BBOX['y2']}], x:[{SOURCE_BBOX['x1']}, {SOURCE_BBOX['x2']}] (size: {SOURCE_BBOX['y2']-SOURCE_BBOX['y1']}x{SOURCE_BBOX['x2']-SOURCE_BBOX['x1']})")
    print(f"  Sauvola Parameters: window={SAUVOLA_WINDOW}, k={SAUVOLA_K}, R={SAUVOLA_R}")

    tampered_rgb, gt_mask, metadata = synthesize_tampered_document_v2(original_rgb)

    # 3. Save Output Artifacts
    print("\n[3/5] Saving synthesized outputs...")
    os.makedirs(os.path.dirname(OUTPUT_TAMPERED_PATH), exist_ok=True)

    tampered_pil = Image.fromarray(tampered_rgb, mode="RGB")
    tampered_pil.save(OUTPUT_TAMPERED_PATH, format="JPEG", quality=95)
    print(f"  [OK] Saved tampered JPEG -> {OUTPUT_TAMPERED_PATH}")

    mask_pil = Image.fromarray(gt_mask, mode="L")
    mask_pil.save(OUTPUT_MASK_PATH, format="PNG")
    print(f"  [OK] Saved GT Mask PNG  -> {OUTPUT_MASK_PATH}")

    # 4. Save Verification Figure
    print("\n[4/5] Generating verification figure...")
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    sauvola_fg = compute_sauvola_binarization(gray)
    save_verification_figure(
        original_rgb, sauvola_fg, tampered_rgb, gt_mask, OUTPUT_VERIF_PATH
    )
    print(f"  [OK] Saved verification plot -> {OUTPUT_VERIF_PATH}")

    # 5. Programmatic Validation
    print("\n[5/5] Running automated validation checks...")
    run_post_synthesis_validation(original_rgb, tampered_rgb, gt_mask, metadata)
    print("  [OK] All validation checks passed successfully.")

    print("\n" + "=" * 80)
    print("DEMO 2 SYNTHESIS COMPLETE & FROZEN")
    print("=" * 80)
    print(f"Changed Pixels:       {metadata['changed_pixel_count']:,} ({metadata['changed_pixel_pct']:.4f}%)")
    print(f"GT Foreground Pixels: {metadata['gt_foreground_pixels']:,} ({metadata['gt_foreground_pct']:.4f}%)")
    print(f"GT Bounding Box:      y:[{TARGET_BBOX['y1']}, {TARGET_BBOX['y2']}], x:[{TARGET_BBOX['x1']}, {TARGET_BBOX['x2']}]")
    print("=" * 80)


if __name__ == "__main__":
    main()
