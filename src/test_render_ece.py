"""Test rendering 'Electronics and Communication Engineering'."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 6 allt.jpg")

def render_text_strip(text: str, font_path: str, font_size: int, text_color: tuple, scale_factor: int = 4):
    """Renders text with supersampling for crisp, natural antialiasing."""
    font = ImageFont.truetype(font_path, font_size * scale_factor)
    
    # Measure text bounding box
    dummy_img = Image.new("RGBA", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    # Render on transparent canvas
    pad = 10 * scale_factor
    canvas = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=(*text_color, 255))
    
    # Resize back with Lanczos antialiasing
    final_w = (tw + 2 * pad) // scale_factor
    final_h = (th + 2 * pad) // scale_factor
    rendered = canvas.resize((final_w, final_h), Image.Resampling.LANCZOS)
    return np.array(rendered)

def main():
    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Target text line y=136..150
    # In the original line: "in " ends at x ~ 43.
    # The department text spans from x=45 to x=350.
    # Let's clean the region x=45..350, y=136..150
    ty1, ty2 = 136, 150
    tx1, tx2 = 44, 350
    
    target_crop = img_rgb[ty1:ty2, tx1:tx2].copy()
    gray_crop = cv2.cvtColor(target_crop, cv2.COLOR_RGB2GRAY)
    
    # Text stroke mask for inpainting
    text_mask = (gray_crop < 90).astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)
    inpaint_mask = cv2.dilate(text_mask, kernel, iterations=1)
    
    # Inpaint to get clean background
    inpainted = cv2.inpaint(cv2.cvtColor(target_crop, cv2.COLOR_RGB2BGR), inpaint_mask, 3, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    
    # Average text color in original:
    # Text color is approximately (45, 42, 40)
    text_rgb = (48, 45, 42)
    
    # Try different font styles: Arial bold or Arial regular
    font_arial = "C:/Windows/Fonts/arial.ttf"
    font_arialbd = "C:/Windows/Fonts/arialbd.ttf"
    
    # Render "Electronics and Communication Engineering"
    # Height of capital letters should be ~8px
    text_str = "Electronics and Communication Engineering"
    rendered_rgba = render_text_strip(text_str, font_arialbd, font_size=8, text_color=text_rgb, scale_factor=4)
    
    # Alpha blend onto inpainted background
    rh, rw = rendered_rgba.shape[:2]
    print(f"Rendered text size: {rw}x{rh}")
    
    # Composite onto inpainted crop
    composite_crop = inpainted_rgb.copy()
    
    # Position text at baseline y_offset ~ 1, x_offset ~ 2
    x_off, y_off = 3, 1
    
    # Alpha blend
    alpha = rendered_rgba[:, :, 3].astype(np.float32) / 255.0
    text_rgb_layer = rendered_rgba[:, :, :3].astype(np.float32)
    
    # Ensure crop fits
    fit_h = min(rh, composite_crop.shape[0] - y_off)
    fit_w = min(rw, composite_crop.shape[1] - x_off)
    
    for c in range(3):
        composite_crop[y_off:y_off+fit_h, x_off:x_off+fit_w, c] = (
            alpha[:fit_h, :fit_w] * text_rgb_layer[:fit_h, :fit_w, c] +
            (1.0 - alpha[:fit_h, :fit_w]) * composite_crop[y_off:y_off+fit_h, x_off:x_off+fit_w, c].astype(np.float32)
        ).astype(np.uint8)
        
    fig, axes = plt.subplots(3, 1, figsize=(12, 6))
    axes[0].imshow(target_crop)
    axes[0].set_title("Original text region")
    axes[1].imshow(inpainted_rgb)
    axes[1].set_title("Inpainted background")
    axes[2].imshow(composite_crop)
    axes[2].set_title(f"Tampered text: '{text_str}'")
    plt.tight_layout()
    plt.savefig("results/demo_inputs/test_text_render.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved test_text_render.png")

if __name__ == "__main__":
    main()
