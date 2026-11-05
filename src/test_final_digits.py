"""Test exact isolated digit crops for 2, 4, 3, 6."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "og current bill.jpg")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "test_final_digits.png")

SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.2
SAUVOLA_R = 128.0

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
    fg = sauvola(gray)

    digits = [
        ("Digit '2' (from Bill Amount 1120.00)", 308, 330, 398, 414),
        ("Digit '4' (from 11.46 in Electricity Duty)", 55, 75, 435, 449),
        ("Digit '3' (from -0.36 in Loss/Gain)", 276, 298, 440, 455),
        ("Digit '6' (from -0.36 in Loss/Gain)", 276, 298, 455, 469),
        ("Digit '6' (from 11.46 in Electricity Duty)", 55, 75, 449, 463),
    ]

    fig, axes = plt.subplots(len(digits), 3, figsize=(15, 3 * len(digits)))

    for i, (label, y1, y2, x1, x2) in enumerate(digits):
        crop_rgb = img_rgb[y1:y2, x1:x2]
        crop_fg = fg[y1:y2, x1:x2]
        scale = 10
        rgb_big = cv2.resize(crop_rgb, (crop_rgb.shape[1]*scale, crop_rgb.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        fg_vis = cv2.cvtColor((crop_fg*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        fg_big = cv2.resize(fg_vis, (fg_vis.shape[1]*scale, fg_vis.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

        ctx_pad = 10
        cy1, cy2 = max(0, y1-ctx_pad), min(512, y2+ctx_pad)
        cx1, cx2 = max(0, x1-ctx_pad), min(512, x2+ctx_pad)
        ctx = img_rgb[cy1:cy2, cx1:cx2].copy()
        cv2.rectangle(ctx, (x1-cx1, y1-cy1), (x2-cx1, y2-cy1), (255, 0, 0), 1)
        ctx_big = cv2.resize(ctx, (ctx.shape[1]*3, ctx.shape[0]*3), interpolation=cv2.INTER_NEAREST)

        axes[i, 0].imshow(ctx_big)
        axes[i, 0].set_title(f"Context: {label}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(rgb_big)
        axes[i, 1].set_title(f"RGB ({y2-y1}x{x2-x1}) y:[{y1},{y2}] x:[{x1},{x2}]", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(fg_big)
        axes[i, 2].set_title(f"Sauvola FG (pixels={int(np.sum(crop_fg))})", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
