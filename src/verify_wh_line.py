"""Find the exact row matching 'World History'."""
import os, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Check all 4 blocks for "World History"
# Let's test template matching with font Arial, size 7..10 for "World" and "History"
results = []
for block_name, by1, by2, bx1, bx2 in [
    ("Block 1 (Grade 9)", 130, 240, 25, 180),
    ("Block 2 (Grade 10)", 130, 240, 260, 420),
    ("Block 3 (Grade 11)", 270, 390, 25, 180),
    ("Block 4 (Grade 12)", 270, 390, 260, 420),
]:
    block_gray = gray[by1:by2, bx1:bx2]
    # Check for "History" template
    for fsize in [7, 8, 9, 10]:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", fsize)
        dummy = Image.new("L", (80, 20), 255)
        d = ImageDraw.Draw(dummy)
        d.text((2, 2), "History", font=font, fill=0)
        bbox = dummy.getbbox()
        if bbox is None: continue
        tmpl = np.array(dummy.crop(bbox))
        
        res = cv2.matchTemplate(block_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        min_v, max_v, min_l, max_l = cv2.minMaxLoc(res)
        abs_x = bx1 + max_l[0]
        abs_y = by1 + max_l[1]
        results.append((max_v, block_name, fsize, abs_x, abs_y, bbox[2]-bbox[0], bbox[3]-bbox[1]))

results.sort(key=lambda x: x[0], reverse=True)
print("Top 5 matches for 'History':")
for r in results[:5]:
    print(f"Score: {r[0]:.4f} | Block: {r[1]} | Size: {r[2]} | Pos: (x={r[3]}, y={r[4]}) | Dim: {r[5]}x{r[6]}")

# Top match coordinates:
best = results[0]
bx, by = best[3], best[4]
bw, bh = best[5], best[6]

# Let's inspect the entire line containing this match:
line_y1 = max(0, by - 6)
line_y2 = min(512, by + bh + 6)
line_x1 = max(0, bx - 60)
line_x2 = min(512, bx + bw + 60)

fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(img_rgb[line_y1:line_y2, line_x1:line_x2])
ax.set_title(f"Best match region: y={line_y1}..{line_y2}, x={line_x1}..{line_x2}")
for x in range(0, line_x2 - line_x1, 10):
    ax.axvline(x, color="yellow", linewidth=0.5, alpha=0.4)
    ax.text(x, 2, f"{line_x1+x}", color="blue", fontsize=7, rotation=90)
for y in range(0, line_y2 - line_y1, 2):
    ax.axhline(y, color="red", linewidth=0.5, alpha=0.3)
    ax.text(1, y+1, f"{line_y1+y}", color="red", fontsize=6)

plt.tight_layout()
plt.savefig("results/demo_inputs/sample_demo7_world_history_found.png", dpi=200, bbox_inches="tight")
print("Saved sample_demo7_world_history_found.png")
