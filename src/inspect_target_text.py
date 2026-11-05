"""Inspect the candidate text region in sample demo 6 allt.jpg."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 6 allt.jpg")
OUT_PATH = os.path.abspath("results/demo_inputs/sample_demo6_text_crop.png")

def main():
    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Crop region between y=100 and y=200
    crop = img_rgb[110:180, 20:490]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].imshow(crop)
    axes[0].set_title("Region y=110..180, x=20..490")
    
    # Add pixel grid
    for y_rel in range(0, crop.shape[0], 10):
        y_abs = 110 + y_rel
        axes[0].axhline(y_rel, color="cyan", linewidth=0.5, alpha=0.5)
        axes[0].text(5, y_rel, f"y={y_abs}", color="red", fontsize=8)
    for x_rel in range(0, crop.shape[1], 25):
        x_abs = 20 + x_rel
        axes[0].axvline(x_rel, color="yellow", linewidth=0.5, alpha=0.5)
        axes[0].text(x_rel, 5, f"x={x_abs}", color="blue", fontsize=7, rotation=90)

    # Grayscale view
    gray_crop = cv2.cvtColor(img_bgr[110:180, 20:490], cv2.COLOR_BGR2GRAY)
    axes[1].imshow(gray_crop, cmap="gray")
    axes[1].set_title("Grayscale Crop with Intensity Scale")
    
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
