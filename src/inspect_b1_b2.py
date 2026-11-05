"""Inspect lines in Block 1, 2, 3, 4 to find World History."""
import os, sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Let's inspect the course name rows in Block 1 (Top Left):
# y from 135 to 240, x from 25 to 170
# Let's crop 8 horizontal slices in Block 1 and 8 in Block 2
fig, axes = plt.subplots(8, 2, figsize=(14, 12))
for i in range(8):
    y1 = 140 + i * 11
    y2 = y1 + 12
    # Block 1
    axes[i, 0].imshow(img_rgb[y1:y2, 25:170])
    axes[i, 0].set_title(f"B1 Row {i}: y={y1}..{y2}")
    # Block 2
    axes[i, 1].imshow(img_rgb[y1:y2, 265:410])
    axes[i, 1].set_title(f"B2 Row {i}: y={y1}..{y2}")

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_rows_b1_b2.png", dpi=200, bbox_inches="tight")
print("Saved rows B1 and B2.")
