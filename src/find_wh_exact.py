"""Find exact coordinates of World History."""
import os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Let's search across the image by saving zoomed-in 4 quadrants of the table:
# y=120..220, y=220..320, y=320..420
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
for i, (y1, y2) in enumerate([(120, 220), (220, 320), (320, 420)]):
    axes[i].imshow(img_rgb[y1:y2, 10:480])
    axes[i].set_title(f"y={y1}..{y2}")
    # Add fine grid
    for y in range(0, 100, 10):
        axes[i].axhline(y, color="red", alpha=0.3)
        axes[i].text(2, y+7, f"{y1+y}", color="red", fontsize=8)
    for x in range(0, 470, 20):
        axes[i].axvline(x, color="blue", alpha=0.3)
        axes[i].text(x, 8, f"{10+x}", color="blue", fontsize=7, rotation=90)

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_zoom_search.png", dpi=200, bbox_inches="tight")
print("Saved zoom search.")
