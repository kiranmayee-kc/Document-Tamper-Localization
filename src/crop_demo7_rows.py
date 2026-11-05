"""Search for 'World History' in sample demo 7.jpg with detailed row crops."""
import os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Let's crop individual table sections
# Left column courses vs right column courses
# Academic transcript tables usually have two columns or a central table
fig, axes = plt.subplots(6, 1, figsize=(14, 12))
row_ranges = [
    (140, 180),
    (180, 220),
    (220, 260),
    (260, 300),
    (300, 340),
    (340, 380),
]

for idx, (y1, y2) in enumerate(row_ranges):
    crop = img_rgb[y1:y2, 10:500]
    axes[idx].imshow(crop)
    axes[idx].set_title(f"Row {idx+1}: y={y1}..{y2}")
    for x in range(0, 490, 25):
        axes[idx].axvline(x, color="yellow", linewidth=0.5, alpha=0.4)
        axes[idx].text(x, 5, f"{10+x}", color="blue", fontsize=6, rotation=90)
    for y in range(0, y2 - y1, 5):
        axes[idx].axhline(y, color="cyan", linewidth=0.5, alpha=0.3)
        axes[idx].text(2, y + 4, f"{y1+y}", color="red", fontsize=6)

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_detailed_rows.png", dpi=200, bbox_inches="tight")
print("Saved detailed rows.")
