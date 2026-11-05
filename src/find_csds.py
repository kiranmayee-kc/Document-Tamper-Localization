"""Find 'B. Tech - CSDS' in college Demo 5.jpg."""
import os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/college Demo 5.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Let's inspect the text regions:
# Header is usually at top (y=0..120)
# Student details (Name, Roll, Branch/Course, Valid Thru) usually in middle (y=120..350)
fig, axes = plt.subplots(6, 1, figsize=(14, 12))
slices = [
    (120, 170),
    (170, 220),
    (220, 270),
    (270, 320),
    (320, 370),
    (370, 420),
]

for idx, (y1, y2) in enumerate(slices):
    crop = img_rgb[y1:y2, 10:500]
    axes[idx].imshow(crop)
    axes[idx].set_title(f"y={y1}..{y2}")
    for x in range(0, 490, 25):
        axes[idx].axvline(x, color="yellow", linewidth=0.5, alpha=0.4)
        axes[idx].text(x, 5, f"{10+x}", color="blue", fontsize=6, rotation=90)
    for y in range(0, y2 - y1, 5):
        axes[idx].axhline(y, color="cyan", linewidth=0.5, alpha=0.3)
        axes[idx].text(2, y + 4, f"{y1+y}", color="red", fontsize=6)

plt.tight_layout()
plt.savefig("results/demo_inputs/college_demo5_slices.png", dpi=200, bbox_inches="tight")
print("Saved college_demo5_slices.png")
