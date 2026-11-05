"""Analyze course titles in each table block."""
import os, sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Let's crop each course column in the transcript
# Left column of courses is usually around x=30..180
# Right column of courses is usually around x=260..420
# Let's inspect all rows with high magnification
fig, axes = plt.subplots(10, 2, figsize=(14, 20))

# Let's check rows at y:
# 140..170, 170..200, 200..230, 230..260, 260..290, 290..320, 320..350, 350..380, 380..410, 410..440
y_steps = [(140+i*28, 140+(i+1)*28) for i in range(10)]

for i, (y1, y2) in enumerate(y_steps):
    # Left course column
    crop_l = img_rgb[y1:y2, 25:200]
    axes[i, 0].imshow(crop_l)
    axes[i, 0].set_title(f"L{i}: y={y1}..{y2}, x=25..200")
    
    # Right course column
    crop_r = img_rgb[y1:y2, 255:430]
    axes[i, 1].imshow(crop_r)
    axes[i, 1].set_title(f"R{i}: y={y1}..{y2}, x=255..430")

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_all_rows_zoomed.png", dpi=200, bbox_inches="tight")
print("Saved zoomed courses.")
