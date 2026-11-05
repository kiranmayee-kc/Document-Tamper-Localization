"""Inspect candidate lines in detail."""
import os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Let's inspect crops:
crops = [
    ("Line 190..235", 190, 235),
    ("Line 245..285", 245, 285),
    ("Line 140..185", 140, 185),
    ("Line 285..330", 285, 330),
    ("Line 330..375", 330, 375),
    ("Line 375..420", 375, 420),
]

fig, axes = plt.subplots(len(crops), 1, figsize=(14, 12))
for idx, (title, y1, y2) in enumerate(crops):
    crop = img_rgb[y1:y2, 10:490]
    axes[idx].imshow(crop)
    axes[idx].set_title(f"{title} (y={y1}..{y2})")
    for x in range(0, 480, 20):
        axes[idx].axvline(x, color="yellow", linewidth=0.5, alpha=0.4)
        axes[idx].text(x, 4, f"{10+x}", color="blue", fontsize=6, rotation=90)
    for y in range(0, y2 - y1, 5):
        axes[idx].axhline(y, color="cyan", linewidth=0.5, alpha=0.3)
        axes[idx].text(2, y + 3, f"{y1+y}", color="red", fontsize=6)

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_candidate_crops.png", dpi=200, bbox_inches="tight")
print("Saved candidate crops.")
