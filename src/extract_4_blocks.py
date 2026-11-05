"""Extract and display all course names with large zoom."""
import os, sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Let's crop individual rows from the 4 semester blocks
# Block 1 (Top Left): y=120..250, x=20..240
# Block 2 (Top Right): y=120..250, x=260..480
# Block 3 (Bottom Left): y=260..400, x=20..240
# Block 4 (Bottom Right): y=260..400, x=260..480

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes[0, 0].imshow(img_rgb[110:250, 20:250])
axes[0, 0].set_title("Block 1 (Top Left: y=110..250, x=20..250)")

axes[0, 1].imshow(img_rgb[110:250, 260:490])
axes[0, 1].set_title("Block 2 (Top Right: y=110..250, x=260..490)")

axes[1, 0].imshow(img_rgb[250:410, 20:250])
axes[1, 0].set_title("Block 3 (Bottom Left: y=250..410, x=20..250)")

axes[1, 1].imshow(img_rgb[250:410, 260:490])
axes[1, 1].set_title("Block 4 (Bottom Right: y=250..410, x=260..490)")

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_4_blocks.png", dpi=200, bbox_inches="tight")
print("Saved 4 blocks.")
