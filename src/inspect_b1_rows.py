"""Test font matches across all rows in Block 1."""
import os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Compare character 'L' and 'i' and 'f' and 'e' in other words:
# 'Physical' has 'l', 'i', 's', 'c', 'a', 'l'
# 'Spanish' has 'i', 's', 'h'
# 'English' has 'E', 'n', 'g', 'l', 'i', 's', 'h'
# Let's inspect them at 4x zoom

fig, axes = plt.subplots(6, 1, figsize=(10, 8))
# English 9: y ~ 138..150
# Algebra I: y ~ 147..159
# World History: y ~ 156..168
# Physical Science: y ~ 165..177
# Spanish I: y ~ 174..186
# Health: y ~ 183..195

for i in range(6):
    y1 = 138 + i * 9
    y2 = y1 + 12
    crop = img_rgb[y1:y2, 35:160]
    axes[i].imshow(crop)
    axes[i].set_title(f"Row {i+1}: y={y1}..{y2}")

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_block1_all_rows.png", dpi=200, bbox_inches="tight")
print("Saved Block 1 rows.")
