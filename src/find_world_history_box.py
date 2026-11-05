"""Pinpoint 'World History' in sample demo 7.jpg."""
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

# Let's inspect the text regions in each semester block.
# Usually high school / college transcripts have semesters:
# e.g., Grade 9 / Grade 10 / Grade 11 / Grade 12 or Fall / Spring
# Let's check the words in rows:
# Let's write out crops of every single line from y=100 to y=420 with high resolution and pixel coordinates

fig, axes = plt.subplots(12, 1, figsize=(16, 18))
y_intervals = [
    (110, 135),
    (135, 160),
    (160, 185),
    (185, 210),
    (210, 235),
    (235, 260),
    (260, 285),
    (285, 310),
    (310, 335),
    (335, 360),
    (360, 385),
    (385, 410),
]

for idx, (y1, y2) in enumerate(y_intervals):
    crop = img_rgb[y1:y2, 15:495]
    axes[idx].imshow(crop)
    axes[idx].set_title(f"Row {idx:02d}: y={y1}..{y2}")
    for x in range(0, 480, 20):
        axes[idx].axvline(x, color="yellow", linewidth=0.5, alpha=0.3)
        axes[idx].text(x, 3, f"{15+x}", color="blue", fontsize=6, rotation=90)
    for y in range(0, y2-y1, 5):
        axes[idx].axhline(y, color="red", linewidth=0.5, alpha=0.3)
        axes[idx].text(2, y+3, f"{y1+y}", color="red", fontsize=6)

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_precise_grid.png", dpi=200, bbox_inches="tight")
print("Saved precise grid.")
