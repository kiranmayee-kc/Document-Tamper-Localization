"""Inspect lines in Block 3 and Block 4."""
import os, sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(8, 2, figsize=(14, 12))
for i in range(8):
    y1 = 280 + i * 11
    y2 = y1 + 12
    # Block 3
    axes[i, 0].imshow(img_rgb[y1:y2, 25:170])
    axes[i, 0].set_title(f"B3 Row {i}: y={y1}..{y2}")
    # Block 4
    axes[i, 1].imshow(img_rgb[y1:y2, 265:410])
    axes[i, 1].set_title(f"B4 Row {i}: y={y1}..{y2}")

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_rows_b3_b4.png", dpi=200, bbox_inches="tight")
print("Saved rows B3 and B4.")
