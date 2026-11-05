"""Fine-grained inspection of target row (Row 2: Artificial Intelligence)."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_PATH = r"C:\Users\kiran\Downloads\iii-1 SEM MEMO.jpg"
OUTPUT_PATH = os.path.abspath("results/demo_inputs/memo_row2_inspect.png")

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

    # Let's inspect rows 1 to 4: y in [70, 170]
    # In particular, row 2 (Artificial Intelligence): y in [95, 125]
    row2_rgb = img_rgb[95:125, 20:490]
    row2_fg = fg[95:125, 20:490]

    # Let's zoom into each column of row 2:
    # 1. Course Code: x in [25, 95]
    # 2. Course Title: x in [95, 305]
    # 3. GradePoint ('8'): x in [315, 350]
    # 4. Grade ('A'): x in [360, 390]
    # 5. Credits ('3'): x in [405, 430]
    # 6. Result ('Pass'): x in [440, 485]

    fig, axes = plt.subplots(4, 1, figsize=(15, 10))

    axes[0].imshow(row2_rgb)
    axes[0].set_title("Row 2: Full Row RGB (y:[95,125], x:[20,490])", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(row2_fg, cmap="gray")
    axes[1].set_title("Row 2: Sauvola Foreground", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Zoomed GradePoint ('8') and Grade ('A')
    gp_crop = img_rgb[100:122, 320:345]
    gr_crop = img_rgb[100:122, 365:385]

    # Find connected components in row 2
    crop_comp = fg[95:125, 305:400] * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(crop_comp)
    print("Components in GradePoint & Grade area (y:[95,125], x:[305,400]):")
    for idx in range(1, num):
        area = stats[idx, cv2.CC_STAT_AREA]
        cx = stats[idx, cv2.CC_STAT_LEFT]
        cy = stats[idx, cv2.CC_STAT_TOP]
        cw = stats[idx, cv2.CC_STAT_WIDTH]
        ch = stats[idx, cv2.CC_STAT_HEIGHT]
        print(f"  Comp {idx}: y:[{95+cy},{95+cy+ch}] (h={ch}), x:[{305+cx},{305+cx+cw}] (w={cw}), area={area}")

    # Display enlarged GP and Grade
    gp_big = cv2.resize(gp_crop, (gp_crop.shape[1]*8, gp_crop.shape[0]*8), interpolation=cv2.INTER_NEAREST)
    gr_big = cv2.resize(gr_crop, (gr_crop.shape[1]*8, gr_crop.shape[0]*8), interpolation=cv2.INTER_NEAREST)

    combined_zoom = np.hstack([gp_big, np.ones((gp_big.shape[0], 20, 3), dtype=np.uint8)*200, gr_big])
    axes[2].imshow(combined_zoom)
    axes[2].set_title("Enlarged GradePoint ('8') and Grade ('A') in Row 2", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Check font style across other rows for reference (e.g. '7' in course codes, 'B' or 'A+' in grades)
    row1_rgb = img_rgb[70:95, 305:400]
    row2_rgb_sub = img_rgb[95:125, 305:400]
    row3_rgb = img_rgb[125:150, 305:400]
    stack_rgb = np.vstack([row1_rgb, row2_rgb_sub, row3_rgb])
    axes[3].imshow(stack_rgb)
    axes[3].set_title("Rows 1, 2, 3 (GP & Grade columns)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
