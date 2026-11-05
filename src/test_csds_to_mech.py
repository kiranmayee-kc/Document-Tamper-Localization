"""Test inpainting 'CSDS' and rendering 'MECH'."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC_PATH = os.path.abspath("results/demo_inputs/college Demo 5.jpg")

def render_supersampled_word(word: str, font_path: str, font_size: float, text_color: tuple, scale_factor: int = 8, blur_sigma: float = 0.35):
    font = ImageFont.truetype(font_path, int(round(font_size * scale_factor)))
    dummy = Image.new("RGBA", (200, 50), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    d.text((10, 5), word, font=font, fill=(*text_color, 255))
    
    final_w = dummy.width // scale_factor
    final_h = dummy.height // scale_factor
    rendered = dummy.resize((final_w, final_h), Image.Resampling.LANCZOS)
    rend_np = np.array(rendered)
    
    if blur_sigma > 0:
        rend_np[:, :, :3] = cv2.GaussianBlur(rend_np[:, :, :3], (3, 3), blur_sigma)
        
    alpha = rend_np[:, :, 3]
    nz_y, nz_x = np.where(alpha > 15)
    tight = rend_np[nz_y.min():nz_y.max()+1, nz_x.min():nz_x.max()+1]
    return tight

def main():
    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # "CSDS" region: y in [303, 318], x in [360, 395]
    ty1, ty2 = 303, 318
    tx1, tx2 = 360, 395
    
    target_crop = img_rgb[ty1:ty2, tx1:tx2].copy()
    gray_crop = cv2.cvtColor(target_crop, cv2.COLOR_RGB2GRAY)
    
    # Stroke inpainting
    stroke_mask = (gray_crop < 140).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inpaint_mask = cv2.dilate(stroke_mask, kernel, iterations=1)
    
    inpainted = cv2.inpaint(cv2.cvtColor(target_crop, cv2.COLOR_RGB2BGR), inpaint_mask, 3, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    
    # Test rendering "MECH"
    font_options = [
        ("Arial Bold sz=8.0", "C:/Windows/Fonts/arialbd.ttf", 8.0, (45, 55, 56), 0.35),
        ("Arial Bold sz=8.5", "C:/Windows/Fonts/arialbd.ttf", 8.5, (45, 55, 56), 0.35),
        ("Arial Regular sz=8.5", "C:/Windows/Fonts/arial.ttf", 8.5, (45, 55, 56), 0.35),
        ("Calibri Bold sz=9.0", "C:/Windows/Fonts/calibrib.ttf", 9.0, (45, 55, 56), 0.35),
    ]
    
    fig, axes = plt.subplots(len(font_options) + 1, 1, figsize=(12, 10))
    axes[0].imshow(img_rgb[298:322, 235:410])
    axes[0].set_title("Original: 'B. Tech - CSDS'")
    
    for idx, (label, fpath, fsize, color, blur) in enumerate(font_options):
        test_img = img_rgb.copy()
        test_img[ty1:ty2, tx1:tx2] = inpainted_rgb
        
        tight_mech = render_supersampled_word("MECH", fpath, fsize, color, scale_factor=8, blur_sigma=blur)
        th, tw = tight_mech.shape[:2]
        
        # Place "MECH" starting at x=362
        x_place = 362
        y_place = 315 - th + 1  # baseline alignment
        
        alpha = tight_mech[:, :, 3].astype(np.float32) / 255.0
        txt = tight_mech[:, :, :3].astype(np.float32)
        
        for c in range(3):
            test_img[y_place:y_place+th, x_place:x_place+tw, c] = (
                alpha * txt[:, :, c] + (1.0 - alpha) * test_img[y_place:y_place+th, x_place:x_place+tw, c].astype(np.float32)
            ).astype(np.uint8)
            
        axes[idx+1].imshow(test_img[298:322, 235:410])
        axes[idx+1].set_title(f"Option {idx+1}: {label} (x={x_place}, y={y_place}, size={tw}x{th})")
        axes[idx+1].axhline(17, color="red", alpha=0.3, linestyle="--")
        
    plt.tight_layout()
    plt.savefig("results/demo_inputs/test_mech_options.png", dpi=200, bbox_inches="tight")
    print("Saved test_mech_options.png")

if __name__ == "__main__":
    main()
