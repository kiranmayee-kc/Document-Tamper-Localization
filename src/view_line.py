"""Scan words in line y=135..153."""
import os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 6 allt.jpg")

img_bgr = cv2.imread(SRC_PATH)
line = img_bgr[135:153, 0:512]

fig, ax = plt.subplots(figsize=(16, 3))
ax.imshow(cv2.cvtColor(line, cv2.COLOR_BGR2RGB))
ax.set_title("Line y=135..153 (x=0..512)")
for x in range(0, 512, 20):
    ax.axvline(x, color="yellow", linewidth=0.5, alpha=0.5)
    ax.text(x, 2, f"{x}", color="red", fontsize=6, rotation=90)

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo6_full_line.png", dpi=200, bbox_inches="tight")
print("Saved full line view")
