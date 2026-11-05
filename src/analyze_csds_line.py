"""Measure exact coordinates, colors, and baseline of 'B. Tech - CSDS'."""
import os, sys
import cv2
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/college Demo 5.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# In the crop y=300..320, x=240..420
# Let's inspect column profiles
crop_gray = gray[303:319, 240:420]
crop_rgb = img_rgb[303:319, 240:420]

# Background is white/cream paper
# Let's inspect column averages
col_mean = np.mean(crop_gray, axis=0)

for x_rel, val in enumerate(col_mean):
    x_abs = 240 + x_rel
    if val < 240:
        print(f"x={x_abs:3d}: gray={val:5.1f}")

# Measure background RGB and text RGB
bg_sample = img_rgb[298:304, 300:360]
print("Background RGB median:", np.median(bg_sample, axis=(0, 1)))

# Text pixels (gray < 160)
text_mask = crop_gray < 160
print("Text RGB median:", np.median(crop_rgb[text_mask], axis=0))
