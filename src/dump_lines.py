"""Dump individual candidate text lines."""
import os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

lines_to_inspect = [
    (140, 154),
    (157, 167),
    (173, 184),
    (198, 212),
    (214, 225),
    (231, 242),
    (255, 269),
    (272, 283),
    (289, 302),
    (313, 324),
    (329, 340),
    (346, 359),
    (371, 384),
]

fig, axes = plt.subplots(len(lines_to_inspect), 1, figsize=(14, 14))
for idx, (y1, y2) in enumerate(lines_to_inspect):
    crop = img_rgb[y1-2:y2+2, :]
    axes[idx].imshow(crop)
    axes[idx].set_title(f"Line {idx}: y={y1}..{y2}")
    for x in range(0, 512, 25):
        axes[idx].axvline(x, color="yellow", linewidth=0.5, alpha=0.3)
        axes[idx].text(x, 2, f"{x}", color="blue", fontsize=6, rotation=90)

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_table_lines_dump.png", dpi=200, bbox_inches="tight")
print("Saved table lines dump.")
