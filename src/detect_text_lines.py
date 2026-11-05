"""Search for 'World History' by scanning text regions."""
import os, sys
import cv2
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Let's inspect text lines in the transcript.
# Let's write a script that crops every line of text in the document and saves them labeled.
# Academic transcript tables have rows with text like "English", "Algebra", "World History", "Biology", etc.
# Let's extract horizontal projection profile to detect all text lines.

# Threshold text
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Horizontal projection
proj = np.sum(thresh, axis=1)

lines = []
in_line = False
start_y = 0
for y, val in enumerate(proj):
    if val > 1000 and not in_line:
        in_line = True
        start_y = y
    elif val <= 1000 and in_line:
        in_line = False
        if y - start_y > 4:
            lines.append((start_y, y))

print(f"Found {len(lines)} candidate text lines:")
for i, (y1, y2) in enumerate(lines):
    line_crop = gray[y1:y2, :]
    print(f"Line {i:02d}: y=[{y1:3d}, {y2:3d}] (h={y2-y1:2d})")

# Let's save a plot with each detected line labeled
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
for i, (y1, y2) in enumerate(lines):
    ax.axhline(y1, color="lime", linewidth=0.5, alpha=0.7)
    ax.axhline(y2, color="red", linewidth=0.5, alpha=0.7)
    ax.text(5, (y1+y2)//2, f"L{i}", color="blue", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_detected_lines.png", dpi=150, bbox_inches="tight")
print("Saved detected lines.")
