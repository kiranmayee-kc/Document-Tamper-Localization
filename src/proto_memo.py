"""Prototype synthesis and verification for academic memo test."""

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
OUTPUT_TEST_PATH = os.path.abspath("results/demo_inputs/memo_tamper_test.png")

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

def render_character_patch(char: str, font_name: str, target_w: int, target_h: int, text_rgb: tuple):
    """Renders an antialiased character with alpha channel at exact target dimensions."""
    hi_size = 128
    hi_img = Image.new("RGBA", (hi_size, hi_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hi_img)
    font = ImageFont.truetype(font_name, size=80)
    
    # Draw centered
    draw.text((20, 10), char, font=font, fill=(text_rgb[0], text_rgb[1], text_rgb[2], 255))
    bbox = hi_img.getbbox()
    crop = hi_img.crop(bbox)
    
    # Resize with Lanczos to target dimensions
    resized = crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return resized

def main():
    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fg = sauvola(gray)

    # Coordinates for Row 2:
    # GradePoint cell: y:[98, 114], x:[316, 332]
    # Original '8' stroke: y:[100, 108], x:[321, 327] (8x6)
    gp_y1, gp_y2, gp_x1, gp_x2 = 98, 114, 316, 332
    
    # Grade cell: y:[98, 114], x:[368, 384]
    # Original 'A' stroke: y:[100, 108], x:[373, 379] (8x6)
    gr_y1, gr_y2, gr_x1, gr_x2 = 98, 114, 368, 384

    # 1. Target Inpainting: Clean original '8' and 'A'
    inpaint_mask = np.zeros((512, 512), dtype=np.uint8)
    
    # Extract foreground for '8' and 'A'
    fg_8 = fg[gp_y1:gp_y2, gp_x1:gp_x2]
    fg_A = fg[gr_y1:gr_y2, gr_x1:gr_x2]

    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inpaint_mask[gp_y1:gp_y2, gp_x1:gp_x2] = cv2.dilate(fg_8, dilate_k, iterations=2)
    inpaint_mask[gr_y1:gr_y2, gr_x1:gr_x2] = cv2.dilate(fg_A, dilate_k, iterations=2)

    cleaned_rgb = cv2.inpaint(img_rgb, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    tampered_rgb = cleaned_rgb.copy()

    # Get local background and text color
    bg_color_gp = np.median(cleaned_rgb[gp_y1:gp_y2, gp_x1:gp_x2], axis=(0, 1)).astype(int)
    bg_color_gr = np.median(cleaned_rgb[gr_y1:gr_y2, gr_x1:gr_x2], axis=(0, 1)).astype(int)
    
    # Text stroke color from nearby original digits ('9' above and '8' below)
    text_color = (40, 35, 42)

    # 2. Render '7' and 'B' using timesbd.ttf (Times New Roman Bold)
    char_7 = render_character_patch("7", "timesbd.ttf", target_w=6, target_h=8, text_rgb=text_color)
    char_B = render_character_patch("B", "timesbd.ttf", target_w=6, target_h=8, text_rgb=text_color)

    # Convert to PIL for alpha compositing onto cleaned image
    tampered_pil = Image.fromarray(tampered_rgb)
    
    # Exact placement (matching vertical alignment y=100..108 and centered in cell)
    # GP placement: x=321, y=100
    # Grade placement: x=373, y=100
    pos_7 = (321, 100)
    pos_B = (373, 100)

    tampered_pil.paste(char_7, pos_7, char_7)
    tampered_pil.paste(char_B, pos_B, char_B)
    tampered_rgb = np.array(tampered_pil)

    # 3. Ground Truth Mask ({0, 1} uint8)
    gt_mask = np.zeros((512, 512), dtype=np.uint8)
    
    # Extract alpha masks of the rendered characters
    alpha_7 = (np.array(char_7)[:, :, 3] > 64).astype(np.uint8)
    alpha_B = (np.array(char_B)[:, :, 3] > 64).astype(np.uint8)

    h7, w7 = alpha_7.shape
    hB, wB = alpha_B.shape

    gt_mask[pos_7[1]:pos_7[1]+h7, pos_7[0]:pos_7[0]+w7] = alpha_7
    gt_mask[pos_B[1]:pos_B[1]+hB, pos_B[0]:pos_B[0]+wB] = alpha_B

    # 4. Add Watermark
    # Place at bottom empty margin area (y=440..460, x=50..460)
    watermark_pil = Image.fromarray(tampered_rgb)
    draw_wm = ImageDraw.Draw(watermark_pil)
    wm_font = ImageFont.truetype("arialbd.ttf", size=13)
    wm_text = "SYNTHETIC TAMPER TEST — NOT A VALID DOCUMENT"
    
    # Watermark position centered at bottom
    draw_wm.text((70, 445), wm_text, font=wm_font, fill=(180, 50, 50))
    tampered_with_wm = np.array(watermark_pil)

    # Visual Inspection Figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))

    orig_crop = img_rgb[90:130, 20:490]
    tamp_crop = tampered_with_wm[90:130, 20:490]
    gt_crop = gt_mask[90:130, 20:490]

    axes[0].imshow(orig_crop)
    axes[0].set_title("ORIGINAL ROW: GR22A3070 | Artificial Intelligence | 8 | A | 3 | Pass", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(tamp_crop)
    axes[1].set_title("TAMPERED ROW: GR22A3070 | Artificial Intelligence | 7 | B | 3 | Pass", fontsize=11, fontweight="bold", color="blue")
    axes[1].axis("off")

    overlay = tamp_crop.copy().astype(np.float32) / 255.0
    mask_3d = gt_crop[:, :, np.newaxis].astype(np.float32)
    red_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    overlay = (1.0 - 0.7 * mask_3d) * overlay + 0.7 * mask_3d * red_color
    axes[2].imshow(overlay)
    axes[2].set_title(f"GT Mask Overlay ({int(np.sum(gt_mask))} pixels in manipulated cells)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_TEST_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved test figure: {OUTPUT_TEST_PATH}")

if __name__ == "__main__":
    main()
