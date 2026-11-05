"""Find exact bounding boxes for '4' and '6' in '11.46'."""

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
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "exact_1146_digits.png")

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

    # In y:[50, 80], x:[380, 480], find connected components
    crop = fg[50:80, 380:480] * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(crop)
    
    print("Components in 11.46 region (y:[50,80], x:[380,480]):")
    comps = []
    for idx in range(1, num):
        area = stats[idx, cv2.CC_STAT_AREA]
        cx = stats[idx, cv2.CC_STAT_LEFT]
        cy = stats[idx, cv2.CC_STAT_TOP]
        cw = stats[idx, cv2.CC_STAT_WIDTH]
        ch = stats[idx, cv2.CC_STAT_HEIGHT]
        y1, y2 = 50 + cy, 50 + cy + ch
        x1, x2 = 380 + cx, 380 + cx + cw
        print(f"  idx {idx}: y:[{y1},{y2}] (h={ch}), x:[{x1},{x2}] (w={cw}), area={area}")
        comps.append((y1, y2, x1, x2, area))

    # Also let's inspect the exact candidates
    # '2': y:[308,330], x:[398,414] (from 1120.00 pos 3)
    # '4': from 11.46
    # '3': y:[276,298], x:[440,455] (from -0.36)
    # '6': from 11.46 (or y:[276,298], x:[455,469] from -0.36)

    # Let's visualize all 4 chosen digits side-by-side!
    # Let's first test candidate coordinates
    candidates = [
        ("Digit '2' (from Bill Amount)", 308, 330, 398, 414),
        ("Digit '4' (from '11.46')", 54, 75, 429, 444),
        ("Digit '3' (from '-0.36')", 276, 298, 440, 455),
        ("Digit '6' (from '11.46')", 54, 75, 444, 459),
        ("Digit '6' alt (from '-0.36')", 276, 298, 455, 469),
    ]

    fig, axes = plt.subplots(len(candidates), 3, figsize=(15, 3 * len(candidates)))

    for i, (label, y1, y2, x1, x2) in enumerate(candidates):
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
        axes[i, 0].set_title(f"Context: {label}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(rgb_big)
        axes[i, 1].set_title(f"RGB Crop ({y2-y1}x{x2-x1}) y:[{y1},{y2}] x:[{x1},{x2}]", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(fg_big)
        axes[i, 2].set_title(f"Sauvola FG (pixels={int(np.sum(crop_fg))})", fontsize=10)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
