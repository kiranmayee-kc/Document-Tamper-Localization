"""Test background patch transfer on clgid.jpg."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SRC_PATH = os.path.abspath("results/demo_inputs/clgid.jpg")
TEST_OUT = os.path.abspath("results/demo_inputs/test_clgid_patch.png")

def main():
    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Let's inspect target blank background area:
    # Target: y:[75, 135], x:[200, 460] -> 60 x 260 = 15,600 px (5.95%)
    # Source: y:[75, 135], x:[30, 290]  -> 60 x 260 = 15,600 px (5.95%)
    
    # Or another blank region:
    # Let's check the patch area and difference
    src_patch = img_rgb[75:135, 30:290].copy()
    target_area = img_rgb[75:135, 200:460].copy()

    # Mild blending (e.g. slight feathering at edges to integrate naturally)
    h, w = src_patch.shape[:2]
    feather_mask = np.ones((h, w), dtype=np.float32)
    border = 6
    for i in range(border):
        val = (i + 1) / (border + 1)
        feather_mask[i, :] = np.minimum(feather_mask[i, :], val)
        feather_mask[h - 1 - i, :] = np.minimum(feather_mask[h - 1 - i, :], val)
        feather_mask[:, i] = np.minimum(feather_mask[:, i], val)
        feather_mask[:, w - 1 - i] = np.minimum(feather_mask[:, w - 1 - i], val)
    
    feather_3d = feather_mask[:, :, np.newaxis]
    blended_patch = (feather_3d * src_patch.astype(np.float32) + (1.0 - feather_3d) * target_area.astype(np.float32))
    blended_patch = np.clip(blended_patch, 0, 255).astype(np.uint8)

    tampered_rgb = img_rgb.copy()
    tampered_rgb[75:135, 200:460] = blended_patch

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original clgid.jpg", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(tampered_rgb)
    axes[1].set_title(f"Tampered clgid.jpg (patch {h}x{w}={h*w} px, {100*h*w/(512*512):.2f}%)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    diff = np.max(np.abs(img_rgb.astype(np.int32) - tampered_rgb.astype(np.int32)), axis=-1)
    im_diff = axes[2].imshow(diff, cmap="viridis")
    plt.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title(f"Absolute Pixel Difference (changed px={int(np.sum(diff > 0))})", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(TEST_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {TEST_OUT}")

if __name__ == "__main__":
    main()
