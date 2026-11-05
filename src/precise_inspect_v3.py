"""Precise inspection of confirmed digit locations.

Digits needed: 2, 4, 3, 6
Sources:
  '2' — Bill Amount '1120.00' pos-3   y:[307,331], x:[397,413]
  '4' — Electricity Duty '11.46'  (need exact bbox)
  '3' — Loss/Gain '-0.36'         (need exact bbox)
  '6' — Loss/Gain '-0.36' or '11.46' (need exact bbox)

Also checks 24.40 at top for backup '4'.

Grid lines at 25px confirmed:
  '11.46' is at y ~ [75, 100] (row-band 75-100)
  '-0.36' is at y ~ [270, 305] (row-band 270-305)
  '24.40' is at y ~ [0, 40] (very top)
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "og current bill.jpg")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "demo3_precise_inspect.png")

SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.2
SAUVOLA_R = 128.0


def load_image(path):
    img_bgr = cv2.imread(path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def sauvola(gray):
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (SAUVOLA_WINDOW, SAUVOLA_WINDOW))
    sq_mean = cv2.blur(gray_f ** 2, (SAUVOLA_WINDOW, SAUVOLA_WINDOW))
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0.0))
    threshold = mean * (1.0 + SAUVOLA_K * (std / SAUVOLA_R - 1.0))
    return (gray_f < threshold).astype(np.uint8)


def show_region(ax_rgb, ax_fg, img_rgb, fg, y1, y2, x1, x2, label, scale=12):
    crop_rgb = img_rgb[y1:y2, x1:x2]
    crop_fg  = fg[y1:y2, x1:x2]
    rgb_big  = cv2.resize(crop_rgb, (crop_rgb.shape[1]*scale, crop_rgb.shape[0]*scale),
                          interpolation=cv2.INTER_NEAREST)
    fg_vis   = cv2.cvtColor((crop_fg*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    fg_big   = cv2.resize(fg_vis, (fg_vis.shape[1]*scale, fg_vis.shape[0]*scale),
                          interpolation=cv2.INTER_NEAREST)
    fg_sum   = int(np.sum(crop_fg))
    ax_rgb.imshow(rgb_big)
    ax_rgb.set_title(f"{label}\ny:[{y1},{y2}] x:[{x1},{x2}]\nfg={fg_sum}", fontsize=9)
    ax_rgb.axis("off")
    ax_fg.imshow(fg_big)
    ax_fg.set_title(f"Sauvola FG", fontsize=9)
    ax_fg.axis("off")
    return fg_sum


def find_digit_components(fg, y1, y2, x1, x2, min_fg=40):
    """Find connected components in the region and return their bboxes."""
    crop = fg[y1:y2, x1:x2].copy().astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(crop)
    results = []
    for idx in range(1, num):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < min_fg:
            continue
        cx = stats[idx, cv2.CC_STAT_LEFT]
        cy = stats[idx, cv2.CC_STAT_TOP]
        cw = stats[idx, cv2.CC_STAT_WIDTH]
        ch = stats[idx, cv2.CC_STAT_HEIGHT]
        # convert back to full image coords
        results.append({
            "area": int(area),
            "y1": y1 + cy, "y2": y1 + cy + ch,
            "x1": x1 + cx, "x2": x1 + cx + cw,
        })
    results.sort(key=lambda r: r["x1"])  # left-to-right
    return results


def main():
    img_rgb = load_image(INPUT_IMAGE_PATH)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fg = sauvola(gray)

    # Scan the full doc left half (x:0-350) and right half (x:350-512)
    # to find exact row positions for 11.46 and -0.36
    print("Scanning full image for rows with numeric content (x:350-512):")
    for y in range(0, 500, 5):
        crop = fg[y:y+5, 350:512]
        s = int(np.sum(crop))
        if s > 50:
            print(f"  y[{y},{y+5}]: fg={s}")

    print("\nScanning full image for rows with content (x:0-350):")
    for y in range(0, 500, 5):
        crop = fg[y:y+5, 0:350]
        s = int(np.sum(crop))
        if s > 100:
            print(f"  y[{y},{y+5}]: fg={s}")

    # Specifically scan for '11.46' row — it's around y=[65,95] based on visual
    print("\nFine scan y:[50,120] x:[350,512]:")
    for y in range(50, 120):
        crop = fg[y:y+1, 350:512]
        s = int(np.sum(crop))
        if s > 5:
            print(f"  y={y}: fg={s}")

    # Fine scan '-0.36' row
    print("\nFine scan y:[265,310] x:[350,512]:")
    for y in range(265, 310):
        crop = fg[y:y+1, 350:512]
        s = int(np.sum(crop))
        if s > 5:
            print(f"  y={y}: fg={s}")

    # Find components in the '11.46' area
    print("\n--- Components in estimated '11.46' region y:[65,100] x:[380,512] ---")
    comps_1146 = find_digit_components(fg, 65, 100, 380, 512, min_fg=30)
    for c in comps_1146:
        print(f"  y:[{c['y1']},{c['y2']}] x:[{c['x1']},{c['x2']}]  area={c['area']}")

    # Find components in the '-0.36' area
    print("\n--- Components in estimated '-0.36' region y:[268,305] x:[380,512] ---")
    comps_036 = find_digit_components(fg, 268, 305, 380, 512, min_fg=30)
    for c in comps_036:
        print(f"  y:[{c['y1']},{c['y2']}] x:[{c['x1']},{c['x2']}]  area={c['area']}")

    # Find components in '24.40' at top
    print("\n--- Components in '24.40' top region y:[0,50] x:[380,512] ---")
    comps_2440 = find_digit_components(fg, 0, 50, 380, 512, min_fg=20)
    for c in comps_2440:
        print(f"  y:[{c['y1']},{c['y2']}] x:[{c['x1']},{c['x2']}]  area={c['area']}")

    # Find components in Bill Amount '1120.00'
    print("\n--- Components in Bill Amount '1120.00' region y:[305,335] x:[360,475] ---")
    comps_ba = find_digit_components(fg, 305, 335, 360, 475, min_fg=30)
    for c in comps_ba:
        print(f"  y:[{c['y1']},{c['y2']}] x:[{c['x1']},{c['x2']}]  area={c['area']}")

    # Now produce a visual inspection with the regions we will use
    # (best guesses based on above; will refine after seeing output)
    regions = [
        # Wide context crops
        ("11.46 row context",      "—",   60, 100, 360, 512),
        ("-0.36 row context",      "—",  265, 310, 360, 512),
        ("24.40 row context",      "—",    0,  50, 360, 512),
        ("1120.00 row context",    "—",  303, 335, 355, 480),
        # Individual digit crops (will set after component analysis)
    ]

    n = len(regions)
    fig, axes = plt.subplots(n, 2, figsize=(20, 5 * n))
    for i, (label, _, y1, y2, x1, x2) in enumerate(regions):
        show_region(axes[i][0], axes[i][1], img_rgb, fg, y1, y2, x1, x2, label, scale=6)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPrecise inspection -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
