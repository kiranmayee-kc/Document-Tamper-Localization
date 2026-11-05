"""Inspect candidate source digit locations in the original bill.

Outputs a large multi-panel image so we can visually confirm
each source coordinate actually contains the intended digit
BEFORE using it in the synthesis.

Candidates to check:
  '2' — from original Bill Amount '1120' (pos 3 of the original field)
  '4' — from '11.46' text in the document
  '3' — any verified '3' occurrence
  '6' — from '11.46' text in the document

Also scans a wider area to help locate '3' and backup digits.
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
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "demo_inputs", "demo3_source_inspection.png")

SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.2
SAUVOLA_R = 128.0


def load_image(path):
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def sauvola(gray):
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (SAUVOLA_WINDOW, SAUVOLA_WINDOW))
    sq_mean = cv2.blur(gray_f ** 2, (SAUVOLA_WINDOW, SAUVOLA_WINDOW))
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0.0))
    threshold = mean * (1.0 + SAUVOLA_K * (std / SAUVOLA_R - 1.0))
    return (gray_f < threshold).astype(np.uint8)


def enlarge_crop(crop_rgb, scale=8):
    h, w = crop_rgb.shape[:2]
    return cv2.resize(crop_rgb, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def enlarge_mask(mask, scale=8):
    h, w = mask.shape[:2]
    vis = (mask * 255).astype(np.uint8)
    vis3 = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    return cv2.resize(vis3, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def inspect_region(img_rgb, fg, y1, y2, x1, x2, label, scale=8):
    """Returns (rgb_enlarged, fg_enlarged, fg_sum) for a region."""
    crop_rgb = img_rgb[y1:y2, x1:x2]
    crop_fg = fg[y1:y2, x1:x2]
    fg_sum = int(np.sum(crop_fg))
    return enlarge_crop(crop_rgb, scale), enlarge_mask(crop_fg, scale), fg_sum


def main():
    img_rgb = load_image(INPUT_IMAGE_PATH)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fg = sauvola(gray)

    # -----------------------------------------------------------------------
    # Candidate regions to inspect
    # Each entry: (label, intended_digit, y1, y2, x1, x2, notes)
    # -----------------------------------------------------------------------
    candidates = [
        # --- Original Bill Amount field: '1120.00'
        # Full row approximately y:[307,331], x:[368,468]
        # We know '2' is around x:[397,413] in the original field (pos 3)
        ("BillAmt-pos3 (orig '2')", "2",  307, 331, 397, 413, "original '2' in 1120"),
        ("BillAmt-fullrow",         "—",  303, 335, 360, 475, "full Bill Amount row"),

        # --- '11.46' text --- the user says '4' and '6' are here
        # Need to find where '11.46' appears.  Try scanning multiple rows.
        ("Row-scan y130-160",       "—",  130, 165,  30, 200, "header area scan"),
        ("Row-scan y155-185",       "—",  155, 190,  30, 220, "scan area"),
        ("Row-scan y180-210",       "—",  180, 210,  30, 220, "scan area"),
        ("Row-scan y200-230",       "—",  200, 230,  30, 220, "scan area"),
        ("Row-scan y220-255",       "—",  220, 255,  30, 250, "scan area"),
        ("Row-scan y250-285",       "—",  250, 285,  30, 250, "scan area"),
        ("Row-scan y280-315",       "—",  280, 315,  30, 250, "scan area"),
        ("Row-scan y340-375",       "—",  340, 375,  30, 250, "scan area"),
        ("Row-scan y375-410",       "—",  375, 410,  30, 250, "scan area"),
        ("Row-scan y405-445",       "—",  405, 445,  30, 250, "scan area"),
        ("Row-scan y440-475",       "—",  440, 475,  30, 250, "scan area"),

        # --- Previous Demo3 "4" source (produced 'A') --- confirm it's wrong
        ("OldSrc '4' (produced A)", "?",  146, 166,  46,  62, "old source — was wrong"),
        # --- Previous Demo3 "5" source (produced 'C') --- confirm it's wrong
        ("OldSrc '5' (produced C)", "?",  437, 455,  63,  78, "old source — was wrong"),
        # --- Previous Demo3 "3" source
        ("OldSrc '3'",              "?",  376, 394, 243, 256, "old source"),
    ]

    n = len(candidates)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    if n == 1:
        axes = [axes]

    print(f"{'Label':<35} {'Intended':>8} {'FG sum':>8}  Notes")
    print("-" * 70)

    for i, (label, digit, y1, y2, x1, x2, notes) in enumerate(candidates):
        rgb_big, fg_big, fg_sum = inspect_region(img_rgb, fg, y1, y2, x1, x2, label)
        print(f"{label:<35} {digit:>8} {fg_sum:>8}  {notes}")

        # Context view (zoomed in image, 2x the region with border)
        pad = 10
        py1 = max(0, y1 - pad); py2 = min(511, y2 + pad)
        px1 = max(0, x1 - pad); px2 = min(511, x2 + pad)
        ctx = img_rgb[py1:py2, px1:px2].copy()
        # Draw rectangle on context
        ctx_draw = ctx.copy()
        ry1 = y1 - py1; ry2 = y2 - py1; rx1 = x1 - px1; rx2 = x2 - px1
        cv2.rectangle(ctx_draw, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)
        ctx_big = cv2.resize(ctx_draw, (ctx_draw.shape[1]*3, ctx_draw.shape[0]*3), interpolation=cv2.INTER_NEAREST)

        axes[i][0].imshow(ctx_big)
        axes[i][0].set_title(f"{label}\ny:[{y1},{y2}] x:[{x1},{x2}] — {notes}", fontsize=8)
        axes[i][0].axis("off")

        axes[i][1].imshow(rgb_big)
        axes[i][1].set_title(f"RGB crop ×8 | intended='{digit}'", fontsize=8)
        axes[i][1].axis("off")

        axes[i][2].imshow(fg_big)
        axes[i][2].set_title(f"Sauvola FG ×8 | fg_sum={fg_sum}", fontsize=8)
        axes[i][2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved inspection figure -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
