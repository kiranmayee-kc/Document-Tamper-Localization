"""Analyze the exact 'World History' row in Block 1."""
import os, sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Target line is around y=155..168, x=25..150
row_crop = img_rgb[153:169, 25:150]
row_gray = gray[153:169, 25:150]

fig, axes = plt.subplots(3, 1, figsize=(12, 6))

# 1. RGB crop
axes[0].imshow(row_crop)
axes[0].set_title("Original Crop: y=153..169, x=25..150")
for x in range(0, 125, 5):
    axes[0].axvline(x, color="yellow", linewidth=0.5, alpha=0.5)
    axes[0].text(x, 2, f"{25+x}", color="blue", fontsize=6, rotation=90)
for y in range(0, 16, 2):
    axes[0].axhline(y, color="red", linewidth=0.5, alpha=0.4)
    axes[0].text(1, y+1, f"{153+y}", color="red", fontsize=6)

# 2. Grayscale intensity
axes[1].imshow(row_gray, cmap="gray")
axes[1].set_title("Grayscale Crop")

# 3. Horizontal and vertical profiles
# Text stroke binarization
# Background is ~240-255 (white paper), text is dark (< 150)
text_bin = (row_gray < 160).astype(np.uint8) * 255
axes[2].imshow(text_bin, cmap="gray")
axes[2].set_title("Binarized Strokes (< 160)")

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_world_history_detail.png", dpi=200, bbox_inches="tight")
print("Saved detail crop.")

# Calculate background and text median RGB
bg_pixels = row_crop[row_gray > 220]
text_pixels = row_crop[row_gray < 120]

print(f"Background RGB (median): {np.median(bg_pixels, axis=0)}")
print(f"Text RGB (median): {np.median(text_pixels, axis=0)}")
