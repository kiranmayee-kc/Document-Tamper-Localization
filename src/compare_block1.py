"""Compare rendered 'World Life' with surrounding course rows."""
import os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")

def render_tight_word(word: str, font_path: str, font_size: float, text_color: tuple, scale_factor: int = 8, blur_sigma: float = 0.35):
    font = ImageFont.truetype(font_path, int(round(font_size * scale_factor)))
    dummy = Image.new("RGBA", (200, 50), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    d.text((10, 5), word, font=font, fill=(*text_color, 255))
    
    # Downsample
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
    
    # Inpaint "History"
    ty1, ty2 = 155, 169
    tx1, tx2 = 71, 132
    target_crop = img_rgb[ty1:ty2, tx1:tx2].copy()
    gray_crop = cv2.cvtColor(target_crop, cv2.COLOR_RGB2GRAY)
    stroke_mask = (gray_crop < 190).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inpaint_mask = cv2.dilate(stroke_mask, kernel, iterations=1)
    inpainted = cv2.inpaint(cv2.cvtColor(target_crop, cv2.COLOR_RGB2BGR), inpaint_mask, 3, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    
    # Build tampered image
    tampered_img = img_rgb.copy()
    tampered_img[ty1:ty2, tx1:tx2] = inpainted_rgb
    
    # Try Arial Regular 8.0 vs Arial Bold 7.5
    tight_life = render_tight_word("Life", "C:/Windows/Fonts/arialbd.ttf", 7.6, (44, 42, 40), scale_factor=8, blur_sigma=0.35)
    th, tw = tight_life.shape[:2]
    
    x_place = 71
    y_place = 166 - th + 1
    
    alpha = tight_life[:, :, 3].astype(np.float32) / 255.0
    txt = tight_life[:, :, :3].astype(np.float32)
    for c in range(3):
        tampered_img[y_place:y_place+th, x_place:x_place+tw, c] = (
            alpha * txt[:, :, c] + (1.0 - alpha) * tampered_img[y_place:y_place+th, x_place:x_place+tw, c].astype(np.float32)
        ).astype(np.uint8)
        
    # Plot original Block 1 vs tampered Block 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    axes[0].imshow(img_rgb[125:240, 20:250])
    axes[0].set_title("Original Block 1 ('World History')")
    axes[1].imshow(tampered_img[125:240, 20:250])
    axes[1].set_title("Tampered Block 1 ('World Life')")
    
    plt.tight_layout()
    plt.savefig("results/demo_inputs/sample_demo7_block1_compare.png", dpi=200, bbox_inches="tight")
    print("Saved block1 compare.")

if __name__ == "__main__":
    main()
