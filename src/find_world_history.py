"""Find 'World History' in sample demo 7.jpg."""
import os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Let's crop vertical strips covering the table region (likely y in [100, 350], x in [20, 480])
# Let's slice into 4 rows and save an annotated image
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
slices = [
    (100, 160),
    (160, 220),
    (220, 280),
    (280, 340),
]

for i, (y1, y2) in enumerate(slices):
    crop = img_rgb[y1:y2, 10:500]
    axes[i].imshow(crop)
    axes[i].set_title(f"y = {y1} .. {y2}")
    for y_rel in range(0, y2 - y1, 10):
        axes[i].axhline(y_rel, color="cyan", linewidth=0.5, alpha=0.5)
        axes[i].text(2, y_rel + 7, f"{y1 + y_rel}", color="red", fontsize=7)
    for x_rel in range(0, 490, 20):
        axes[i].axvline(x_rel, color="yellow", linewidth=0.5, alpha=0.5)
        axes[i].text(x_rel, 5, f"{10 + x_rel}", color="blue", fontsize=6, rotation=90)

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_table_slices.png", dpi=200, bbox_inches="tight")
print("Saved table slices.")
