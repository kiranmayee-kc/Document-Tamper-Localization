"""Inspect the exact region around y=295..325, x=240..460 in college Demo 5.jpg."""
import os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/college Demo 5.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Zoom in on y=295..325, x=220..460
crop = img_rgb[295:325, 220:460]

fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(crop)
ax.set_title("Region y=295..325, x=220..460 ('B. Tech - CSDS')")
for x in range(0, crop.shape[1], 10):
    ax.axvline(x, color="yellow", linewidth=0.5, alpha=0.5)
    ax.text(x, 2, f"{220+x}", color="blue", fontsize=7, rotation=90)
for y in range(0, crop.shape[0], 5):
    ax.axhline(y, color="red", linewidth=0.5, alpha=0.4)
    ax.text(2, y+3, f"{295+y}", color="red", fontsize=7)

plt.tight_layout()
plt.savefig("results/demo_inputs/college_demo5_csds_detail.png", dpi=200, bbox_inches="tight")
print("Saved college_demo5_csds_detail.png")
