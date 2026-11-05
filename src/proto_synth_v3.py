"""Prototype synthesis script for Demo 3 (1120.00 -> 2436.00)."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "og current bill.jpg")
OUTPUT_CHECK_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "demo3_bill_amount_check.png")

SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.2
SAUVOLA_R = 128.0

DIGIT_REPLACEMENTS = [
    {
        "pos_id": 1,
        "original_digit": "1",
        "replacement_digit": "2",
        "target": {"y1": 309, "y2": 329, "x1": 368, "x2": 381},
        "source": {"y1": 308, "y2": 330, "x1": 398, "x2": 414},
        "source_desc": "digit '2' from Bill Amount field (pos 3)",
    },
    {
        "pos_id": 2,
        "original_digit": "1",
        "replacement_digit": "4",
        "target": {"y1": 309, "y2": 329, "x1": 383, "x2": 395},
        "source": {"y1": 55, "y2": 75, "x1": 435, "x2": 449},
        "source_desc": "digit '4' from '11.46' in Electricity Duty row",
    },
    {
        "pos_id": 3,
        "original_digit": "2",
        "replacement_digit": "3",
        "target": {"y1": 309, "y2": 329, "x1": 397, "x2": 413},
        "source": {"y1": 276, "y2": 298, "x1": 440, "x2": 455},
        "source_desc": "digit '3' from '-0.36' in Loss/Gain row",
    },
    {
        "pos_id": 4,
        "original_digit": "0",
        "replacement_digit": "6",
        "target": {"y1": 309, "y2": 329, "x1": 413, "x2": 428},
        "source": {"y1": 55, "y2": 75, "x1": 449, "x2": 463},
        "source_desc": "digit '6' from '11.46' in Electricity Duty row",
    },
]

def load_image(path):
    img_bgr = cv2.imread(path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def sauvola(gray):
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (SAUVOLA_WINDOW, SAUVOLA_WINDOW))
    sq_mean = cv2.blur(gray_f ** 2, (SAUVOLA_WINDOW, SAUVOLA_WINDOW))
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0.0))
    threshold = mean * (1.0 + SAUVOLA_K * (std / SAUVOLA_R - 1.0))
    return (gray_f < threshold).astype(np.uint8)

def main():
    img_rgb = load_image(INPUT_IMAGE_PATH)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sauvola_fg = sauvola(gray)

    # Clean target digits using Telea inpainting on dilated target foreground
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    global_inpaint_mask = np.zeros((512, 512), dtype=np.uint8)
    gt_mask = np.zeros((512, 512), dtype=np.uint8)

    for rep in DIGIT_REPLACEMENTS:
        t = rep["target"]
        ty1, ty2, tx1, tx2 = t["y1"], t["y2"], t["x1"], t["x2"]
        target_fg_crop = sauvola_fg[ty1:ty2, tx1:tx2]
        dilated_target_fg = cv2.dilate(target_fg_crop, dilate_kernel, iterations=1)
        global_inpaint_mask[ty1:ty2, tx1:tx2] = np.maximum(
            global_inpaint_mask[ty1:ty2, tx1:tx2], dilated_target_fg
        )

    # Inpaint to remove original 1120 digits
    cleaned_rgb = cv2.inpaint(
        img_rgb, global_inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )
    tampered_rgb = cleaned_rgb.copy()

    # Stroke-only transfer
    for rep in DIGIT_REPLACEMENTS:
        t = rep["target"]
        s = rep["source"]
        ty1, ty2, tx1, tx2 = t["y1"], t["y2"], t["x1"], t["x2"]
        target_h, target_w = ty2 - ty1, tx2 - tx1
        sy1, sy2, sx1, sx2 = s["y1"], s["y2"], s["x1"], s["x2"]

        # Extract source stroke mask and source RGB
        src_fg_crop = sauvola_fg[sy1:sy2, sx1:sx2]
        src_rgb_crop = img_rgb[sy1:sy2, sx1:sx2]

        # Resize stroke mask using nearest neighbour
        stroke_mask = cv2.resize(src_fg_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # Resize source RGB using linear/area
        src_h, src_w = sy2 - sy1, sx2 - sx1
        interp = cv2.INTER_AREA if (src_h >= target_h and src_w >= target_w) else cv2.INTER_LINEAR
        src_rgb_resized = cv2.resize(src_rgb_crop, (target_w, target_h), interpolation=interp)

        # Place ONLY foreground stroke pixels onto cleaned background
        fg_indices = (stroke_mask == 1)
        target_slot = tampered_rgb[ty1:ty2, tx1:tx2]
        target_slot[fg_indices] = src_rgb_resized[fg_indices]
        tampered_rgb[ty1:ty2, tx1:tx2] = target_slot

        # Mark in GT mask
        gt_mask[ty1:ty2, tx1:tx2] = np.maximum(gt_mask[ty1:ty2, tx1:tx2], stroke_mask)

    # Generate enlarged check figure
    # Bill Amount crop: y:[295, 345], x:[100, 480]
    orig_crop = img_rgb[295:345, 100:480]
    tamp_crop = tampered_rgb[295:345, 100:480]
    gt_crop = gt_mask[295:345, 100:480]

    # Create visual check
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    
    axes[0].imshow(orig_crop)
    axes[0].set_title("ORIGINAL Bill Amount Field (Bill Amount 1120.00)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(tamp_crop)
    axes[1].set_title("TAMPERED Bill Amount Field (Bill Amount 2436.00)", fontsize=12, fontweight="bold", color="blue")
    axes[1].axis("off")

    # Overlay
    overlay = tamp_crop.copy().astype(np.float32) / 255.0
    mask_3d = gt_crop[:, :, np.newaxis].astype(np.float32)
    red_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    overlay = (1.0 - 0.7 * mask_3d) * overlay + 0.7 * mask_3d * red_color
    axes[2].imshow(overlay)
    axes[2].set_title(f"GT Mask Overlay ({int(np.sum(gt_mask))} foreground pixels inserted)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_CHECK_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved check image: {OUTPUT_CHECK_PATH}")

if __name__ == "__main__":
    main()
