"""Find exact coordinates of 'Data Visualization' (Credits: 3)."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_PATH = r"C:\Users\kiran\Downloads\iii-1 SEM MEMO.jpg"
OUTPUT_PATH = os.path.abspath("results/demo_inputs/find_datavis.png")

SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.2
SAUVOLA_R = 128.0

def sauvola(gray):
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (SAUVOLA_WINDOW, SAUVOLA_WINDOW))
    sq_mean = cv2.blur(gray_f ** 2, (SAUVOLA_WINDOW, SAUVOLA_WINDOW))
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0.0))
    threshold = mean * (1.0 + SAUVOLA_K * (std / SAUVOLA_R - 1.0))
    return (gray_f < threshold).astype(np.uint8)

def main():
    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fg = sauvola(gray)

    # Inspect all rows from y=60 to y=280 in bands of ~22px
    bands = [
        ("Row 1: Data Warehousing and Data Mining", 70, 95),
        ("Row 2: Artificial Intelligence", 95, 120),
        ("Row 3: Data Warehousing Lab", 120, 145),
        ("Row 4: AI Lab using Python", 145, 170),
        ("Row 5: Data Visualization (3 credits)", 168, 193),
        ("Row 6: Data Visualization Lab (1.5 credits)", 193, 218),
        ("Row 7: Cloud Computing", 218, 243),
        ("Row 8: Intro to IoT", 243, 275),
    ]

    fig, axes = plt.subplots(len(bands), 2, figsize=(16, 2.5 * len(bands)))

    for i, (label, y1, y2) in enumerate(bands):
        crop_rgb = img_rgb[y1:y2, 20:490]
        crop_fg = fg[y1:y2, 20:490]

        # Connected components in GP and Grade
        crop_comp = fg[y1:y2, 305:400] * 255
        num, labels, stats, _ = cv2.connectedComponentsWithStats(crop_comp)
        print(f"\n{label} y:[{y1},{y2}]:")
        for idx in range(1, num):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area < 5: continue
            cx, cy = stats[idx, cv2.CC_STAT_LEFT], stats[idx, cv2.CC_STAT_TOP]
            cw, ch = stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT]
            print(f"  Comp {idx}: y:[{y1+cy},{y1+cy+ch}] (h={ch}), x:[{305+cx},{305+cx+cw}] (w={cw}), area={area}")

        axes[i, 0].imshow(crop_rgb)
        axes[i, 0].set_title(label, fontsize=9, fontweight="bold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(crop_fg, cmap="gray")
        axes[i, 1].set_title(f"FG: {label}", fontsize=9)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
