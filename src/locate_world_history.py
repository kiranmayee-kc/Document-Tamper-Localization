"""Locate 'World History' specifically across the image."""
import os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Let's inspect rows around y=120 to y=480 in 20-pixel increments
# In transcripts, courses are often listed under course columns
# Let's check both left column (e.g. x=30..250) and right column (e.g. x=260..480)
y_starts = list(range(120, 440, 20))
fig, axes = plt.subplots(len(y_starts), 2, figsize=(16, 28))

for idx, y_s in enumerate(y_starts):
    y_e = y_s + 20
    # Left col
    crop_l = img_rgb[y_s:y_e, 20:250]
    axes[idx, 0].imshow(crop_l)
    axes[idx, 0].set_title(f"Left y={y_s}..{y_e}")
    # Right col
    crop_r = img_rgb[y_s:y_e, 250:480]
    axes[idx, 1].imshow(crop_r)
    axes[idx, 1].set_title(f"Right y={y_s}..{y_e}")

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_all_course_crops.png", dpi=200, bbox_inches="tight")
print("Saved all course crops.")
