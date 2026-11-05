"""Test synthesis on exact 3 subject rows (10 | O)."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC_PATH = r"C:\Users\kiran\Downloads\iii-1 SEM MEMO.jpg"
OUTPUT_PATH = os.path.abspath("results/demo_inputs/test_3_rows_correct.png")

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

def render_text_patch(text: str, font_name: str, target_w: int, target_h: int, text_rgb: tuple):
    hi_img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hi_img)
    font = ImageFont.truetype(font_name, size=64)
    draw.text((15, 10), text, font=font, fill=(text_rgb[0], text_rgb[1], text_rgb[2], 255))
    bbox = hi_img.getbbox()
    crop = hi_img.crop(bbox)
    return crop.resize((target_w, target_h), Image.Resampling.LANCZOS)

def main():
    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fg = sauvola(gray)

    rows_config = [
        {
            "name": "Data Warehousing and Data Mining",
            "row_y1": 70, "row_y2": 93,
            "gp_cell": (73, 90, 314, 334),
            "gp_pos": (319, 78),
            "grade_cell": (73, 90, 364, 386),
            "grade_pos": (372, 78),
        },
        {
            "name": "Artificial Intelligence",
            "row_y1": 93, "row_y2": 117,
            "gp_cell": (95, 112, 314, 334),
            "gp_pos": (319, 100),
            "grade_cell": (95, 112, 364, 386),
            "grade_pos": (372, 100),
        },
        {
            "name": "Data Visualization",
            "row_y1": 158, "row_y2": 182,
            "gp_cell": (160, 178, 314, 334),
            "gp_pos": (319, 167),
            "grade_cell": (160, 178, 364, 386),
            "grade_pos": (372, 167),
        },
    ]

    inpaint_mask = np.zeros((512, 512), dtype=np.uint8)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for r in rows_config:
        gy1, gy2, gx1, gx2 = r["gp_cell"]
        ry1, ry2, rx1, rx2 = r["grade_cell"]
        
        fg_gp = fg[gy1:gy2, gx1:gx2]
        fg_gr = fg[ry1:ry2, rx1:rx2]

        inpaint_mask[gy1:gy2, gx1:gx2] = cv2.dilate(fg_gp, dilate_k, iterations=2)
        inpaint_mask[ry1:ry2, rx1:rx2] = cv2.dilate(fg_gr, dilate_k, iterations=2)

    cleaned_rgb = cv2.inpaint(img_rgb, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    tampered_rgb = cleaned_rgb.copy()

    text_color = (40, 35, 42)
    patch_10 = render_text_patch("10", "timesbd.ttf", target_w=9, target_h=8, text_rgb=text_color)
    patch_O = render_text_patch("O", "timesbd.ttf", target_w=6, target_h=8, text_rgb=text_color)

    alpha_10 = (np.array(patch_10)[:, :, 3] > 64).astype(np.uint8)
    alpha_O = (np.array(patch_O)[:, :, 3] > 64).astype(np.uint8)

    gt_mask = np.zeros((512, 512), dtype=np.uint8)
    tampered_pil = Image.fromarray(tampered_rgb)

    for r in rows_config:
        gx, gy = r["gp_pos"]
        rx, ry = r["grade_pos"]

        tampered_pil.paste(patch_10, (gx, gy), patch_10)
        tampered_pil.paste(patch_O, (rx, ry), patch_O)

        h10, w10 = alpha_10.shape
        hO, wO = alpha_O.shape

        gt_mask[gy:gy+h10, gx:gx+w10] = np.maximum(gt_mask[gy:gy+h10, gx:gx+w10], alpha_10)
        gt_mask[ry:ry+hO, rx:rx+wO] = np.maximum(gt_mask[ry:ry+hO, rx:rx+wO], alpha_O)

    tampered_rgb = np.array(tampered_pil)

    # Verification Figure showing all 3 rows
    fig, axes = plt.subplots(3, 2, figsize=(18, 9))

    for i, r in enumerate(rows_config):
        y1, y2 = r["row_y1"], r["row_y2"]
        orig_crop = img_rgb[y1:y2, 20:490]
        tamp_crop = tampered_rgb[y1:y2, 20:490]
        gt_crop = gt_mask[y1:y2, 20:490]

        axes[i, 0].imshow(orig_crop)
        axes[i, 0].set_title(f"ORIGINAL: {r['name']}", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        overlay = tamp_crop.copy().astype(np.float32) / 255.0
        mask_3d = gt_crop[:, :, np.newaxis].astype(np.float32)
        red_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        overlay = (1.0 - 0.7 * mask_3d) * overlay + 0.7 * mask_3d * red_color

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"TAMPERED (10 | O): {r['name']}", fontsize=10, fontweight="bold", color="blue")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
