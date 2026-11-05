"""Fine-grained digit hunt: scan for 4 and 6 across right-side numeric columns.

Outputs crops of the numeric value columns at each row,
plus zoomed crops of all promising digit regions.
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
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "demo3_digit_hunt.png")

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


def show_region(img_rgb, fg, y1, y2, x1, x2, scale=6):
    crop_rgb = img_rgb[y1:y2, x1:x2]
    crop_fg  = fg[y1:y2, x1:x2]
    rgb_big  = cv2.resize(crop_rgb, (crop_rgb.shape[1]*scale, crop_rgb.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    fg_vis   = (crop_fg * 255).astype(np.uint8)
    fg_big   = cv2.resize(fg_vis, (fg_vis.shape[1]*scale, fg_vis.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    return rgb_big, fg_big, int(np.sum(crop_fg))


def main():
    img_rgb = load_image(INPUT_IMAGE_PATH)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fg = sauvola(gray)

    # Show the full-width scan of the right half (numeric columns) per row band
    # The numeric values are approximately x:[350,510]
    # Scan every ~25px band of the full image height

    # First show the whole image with a grid overlay to understand layout
    fig0, ax0 = plt.subplots(figsize=(10, 20))
    ax0.imshow(img_rgb)
    ax0.set_title("Full document — annotating numeric column x:[350,512]")
    for y in range(0, 512, 25):
        ax0.axhline(y, color='cyan', linewidth=0.4, alpha=0.5)
    ax0.axvline(350, color='red', linewidth=1)
    ax0.set_xlim(0, 512); ax0.set_ylim(512, 0)
    plt.tight_layout()
    full_path = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "demo3_full_annotated.png")
    plt.savefig(full_path, dpi=100, bbox_inches="tight")
    plt.close(fig0)
    print(f"Full annotated -> {full_path}")

    # Now fine-grained inspection: numeric column x:[350,512], all rows in bands of ~25px
    # Focus on rows that have numeric content (based on previous row scans)
    numeric_x1, numeric_x2 = 340, 512

    row_bands = [
        (128, 155, "row-band 128-155"),
        (155, 180, "row-band 155-180"),
        (180, 210, "row-band 180-210"),
        (210, 240, "row-band 210-240"),
        (240, 270, "row-band 240-270"),
        (270, 305, "row-band 270-305"),
        (305, 340, "row-band 305-340"),
        (340, 375, "row-band 340-375"),
        (375, 410, "row-band 375-410"),
        (405, 445, "row-band 405-445"),
        (440, 475, "row-band 440-475"),
    ]

    n = len(row_bands)
    fig, axes = plt.subplots(n, 2, figsize=(20, 3 * n))

    for i, (y1, y2, label) in enumerate(row_bands):
        rgb_big, fg_big, fg_sum = show_region(img_rgb, fg, y1, y2, numeric_x1, numeric_x2, scale=4)
        axes[i][0].imshow(rgb_big)
        axes[i][0].set_title(f"RGB × 4 | y:[{y1},{y2}] x:[{numeric_x1},{numeric_x2}] | {label}", fontsize=9)
        axes[i][0].axis("off")
        axes[i][1].imshow(fg_big, cmap="gray")
        axes[i][1].set_title(f"Sauvola FG × 4 | fg_sum={fg_sum}", fontsize=9)
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Digit hunt figure -> {OUTPUT_PATH}")

    # --- Also zoom into specific candidate digit cells ---
    # Bill Amount '1120.00' is at y:[307,331], x:[368,468]
    # Let's look at each character slot individually in the full-amount row
    # and nearby rows to find 4 and 6
    print("\n=== Individual digit slot inspection across numeric rows ===")
    # Examine cols 350..512 in x-strips of ~15px
    x_cols = list(range(350, 512, 15))
    for row_y1, row_y2, row_label in row_bands:
        parts = []
        for xc in x_cols:
            xc2 = min(xc + 15, 512)
            crop_fg = fg[row_y1:row_y2, xc:xc2]
            s = int(np.sum(crop_fg))
            if s > 30:
                parts.append(f"x[{xc},{xc2}]={s}")
        if parts:
            print(f"  {row_label}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
