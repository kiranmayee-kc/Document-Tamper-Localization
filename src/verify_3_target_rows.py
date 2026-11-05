"""Verify exact 3 target rows."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_PATH = r"C:\Users\kiran\Downloads\iii-1 SEM MEMO.jpg"
OUTPUT_PATH = os.path.abspath("results/demo_inputs/verify_3_target_rows.png")

def main():
    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    rows = [
        ("1. Data Warehousing and Data Mining", 70, 93),
        ("2. Artificial Intelligence", 93, 117),
        ("3. Data Visualization", 158, 182),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 6))

    for i, (label, y1, y2) in enumerate(rows):
        crop = img_rgb[y1:y2, 20:490]
        axes[i].imshow(crop)
        axes[i].set_title(label, fontsize=10, fontweight="bold")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
