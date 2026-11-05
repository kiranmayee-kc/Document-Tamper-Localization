"""Test inpainting 'History' and rendering 'Life'."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")

def render_supersampled_word(word: str, font_path: str, font_size: float, text_color: tuple, scale_factor: int = 8, blur_sigma: float = 0.3):
    font = ImageFont.truetype(font_path, int(round(font_size * scale_factor)))
    dummy = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), word, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    pad = 5 * scale_factor
    canvas = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad - bbox[0], pad - bbox[1]), word, font=font, fill=(*text_color, 255))
    
    final_w = (tw + 2 * pad) // scale_factor
    final_h = (th + 2 * pad) // scale_factor
    rendered = canvas.resize((final_w, final_h), Image.Resampling.LANCZOS)
    rend_np = np.array(rendered)
    
    if blur_sigma > 0:
        rend_np[:, :, :3] = cv2.GaussianBlur(rend_np[:, :, :3], (3, 3), blur_sigma)
        
    return rend_np, bbox, pad // scale_factor

def main():
    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Context region: y in [150, 172], x in [25, 160]
    context = img_rgb[150:172, 25:160].copy()
    
    # "History" bounding region: y in [155, 169], x in [71, 132]
    ty1, ty2 = 155, 169
    tx1, tx2 = 71, 132
    
    # Text stroke segmentation for inpainting
    target_crop = img_rgb[ty1:ty2, tx1:tx2].copy()
    gray_crop = cv2.cvtColor(target_crop, cv2.COLOR_RGB2GRAY)
    
    # Stroke mask
    stroke_mask = (gray_crop < 190).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inpaint_mask = cv2.dilate(stroke_mask, kernel, iterations=1)
    
    inpainted = cv2.inpaint(cv2.cvtColor(target_crop, cv2.COLOR_RGB2BGR), inpaint_mask, 3, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    
    # Test rendering "Life" with different fonts and font sizes
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    axes[0].imshow(context)
    axes[0].set_title("Original: 'World History'")
    
    font_options = [
        ("Arial Regular sz=8.0", "C:/Windows/Fonts/arial.ttf", 8.0, (44, 42, 40), 0.35),
        ("Arial Regular sz=8.5", "C:/Windows/Fonts/arial.ttf", 8.5, (44, 42, 40), 0.35),
        ("Arial Bold sz=7.5", "C:/Windows/Fonts/arialbd.ttf", 7.5, (44, 42, 40), 0.35),
        ("Arial Bold sz=8.0", "C:/Windows/Fonts/arialbd.ttf", 8.0, (44, 42, 40), 0.35),
    ]
    
    for idx, (label, fpath, fsize, color, blur) in enumerate(font_options):
        # Inpaint target area in full image
        test_img = img_rgb.copy()
        test_img[ty1:ty2, tx1:tx2] = inpainted_rgb
        
        # Render "Life"
        rend_rgba, bbox, pad = render_supersampled_word("Life", fpath, fsize, color, scale_factor=8, blur_sigma=blur)
        
        # Start "Life" right after "World" (World ends at x=67, space of 4px -> start at x=72)
        # Align baseline: baseline of "World" is y=166
        # Top of capital 'L' should be at y=157 -> y_pos
        rw, rh = rend_rgba.shape[1], rend_rgba.shape[0]
        
        # Crop tight around non-zero alpha in rend_rgba
        alpha_channel = rend_rgba[:, :, 3]
        nz_y, nz_x = np.where(alpha_channel > 10)
        if len(nz_y) > 0:
            min_y, max_y = nz_y.min(), nz_y.max()
            min_x, max_x = nz_x.min(), nz_x.max()
            tight_rgba = rend_rgba[min_y:max_y+1, min_x:max_x+1]
            th, tw = tight_rgba.shape[:2]
            
            # Place at baseline 166:
            # For "Life", baseline is bottom of 'L', 'i', 'f', 'e' (at bottom of tight crop)
            x_place = 72
            y_place = 166 - th + 1
            
            # Blend
            alpha = tight_rgba[:, :, 3].astype(np.float32) / 255.0
            txt = tight_rgba[:, :, :3].astype(np.float32)
            for c in range(3):
                test_img[y_place:y_place+th, x_place:x_place+tw, c] = (
                    alpha * txt[:, :, c] + (1.0 - alpha) * test_img[y_place:y_place+th, x_place:x_place+tw, c].astype(np.float32)
                ).astype(np.uint8)
                
        crop_res = test_img[150:172, 25:160]
        axes[idx+1].imshow(crop_res)
        axes[idx+1].set_title(f"Option {idx+1}: {label} (placed at x={x_place}, y={y_place}, size={tw}x{th})")
        axes[idx+1].axhline(16, color="red", alpha=0.3, linestyle="--")
        
    plt.tight_layout()
    plt.savefig("results/demo_inputs/test_life_render_options.png", dpi=200, bbox_inches="tight")
    print("Saved test_life_render_options.png")

if __name__ == "__main__":
    main()
