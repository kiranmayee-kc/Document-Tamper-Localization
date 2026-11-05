"""Find exact location of '11.46' and '24.40'."""

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
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "find_1146.png")

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

    # Let's inspect y:[0, 95] and x:[350, 512]
    # Let's crop 3 bands: y:[0, 30], y:[25, 55], y:[50, 85]
    bands = [
        ("Top Row y:[0, 30] ('24.40'?)", 0, 30, 350, 512),
        ("Row 2 y:[25, 55] ('100.00'?)", 25, 55, 350, 512),
        ("Row 3 y:[50, 80] ('11.46'?)", 50, 80, 350, 512),
    ]

    fig, axes = plt.subplots(len(bands), 2, figsize=(15, 3 * len(bands)))

    for i, (label, y1, y2, x1, x2) in enumerate(bands):
        crop_rgb = img_rgb[y1:y2, x1:x2]
        crop_fg = fg[y1:y2, x1:x2]
        
        scale = 6
        rgb_big = cv2.resize(crop_rgb, (crop_rgb.shape[1]*scale, crop_rgb.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        fg_vis = cv2.cvtColor((crop_fg*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        fg_big = cv2.resize(fg_vis, (fg_vis.shape[1]*scale, fg_vis.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

        axes[i, 0].imshow(rgb_big)
        axes[i, 0].set_title(f"RGB: {label}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(fg_big)
        axes[i, 1].set_title(f"Sauvola FG: {label}", fontsize=10)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
