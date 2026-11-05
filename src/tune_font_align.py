"""Fine-tune font rendering and baseline alignment for demo 6 text tampering."""
import os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 6 allt.jpg")

def test_font_alignments():
    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Context line: y=130..155, x=10..400
    # Original text: "in COMPUTER SCIENCE AND ENGINEERING (DATA SCIENCE) (CSD), under ..."
    # Let's inspect "in " at x=28..42. Baseline is around y=145. Top of 'i' is around y=139.
    # Height of lowercase letter is ~6px, uppercase is ~8px.
    
    # We want to replace "COMPUTER SCIENCE AND ENGINEERING (DATA SCIENCE) (CSD)" with:
    # "Electronics and Communication Engineering"
    # Starting at x=44, ending before ", under"
    
    # Inpaint region: y=135..148, x=43..355
    ty1, ty2 = 135, 149
    tx1, tx2 = 44, 355
    
    target_crop = img_rgb[ty1:ty2, tx1:tx2].copy()
    gray_crop = cv2.cvtColor(target_crop, cv2.COLOR_RGB2GRAY)
    
    # Inpainting
    text_mask = (gray_crop < 85).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inpaint_mask = cv2.dilate(text_mask, kernel, iterations=1)
    inpainted = cv2.inpaint(cv2.cvtColor(target_crop, cv2.COLOR_RGB2BGR), inpaint_mask, 3, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    
    # Font tests:
    # 1. Arial Bold, size 8
    # 2. Arial Bold, size 7.5
    # 3. Arial Bold, size 7
    # 4. Arial Regular, size 8
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 8))
    
    for idx, (font_path, fsize, blur_sigma) in enumerate([
        ("C:/Windows/Fonts/arialbd.ttf", 7.5, 0.4),
        ("C:/Windows/Fonts/arialbd.ttf", 7.0, 0.3),
        ("C:/Windows/Fonts/arial.ttf", 7.5, 0.3),
        ("C:/Windows/Fonts/ARIALNB.TTF", 7.5, 0.3),
    ]):
        # Supersampled rendering
        scale = 8
        font = ImageFont.truetype(font_path, int(fsize * scale))
        text = "Electronics and Communication Engineering"
        
        dummy = Image.new("RGBA", (10, 10))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        
        canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((-bbox[0], -bbox[1]), text, font=font, fill=(45, 42, 40, 255))
        
        # Downscale
        fw = int(round(tw / scale))
        fh = int(round(th / scale))
        rendered = canvas.resize((fw, fh), Image.Resampling.LANCZOS)
        rend_np = np.array(rendered)
        
        # Apply slight gaussian blur to match scanner PSF
        if blur_sigma > 0:
            rend_np[:, :, :3] = cv2.GaussianBlur(rend_np[:, :, :3], (3, 3), blur_sigma)
        
        # Composite onto full line context (y=130..155, x=10..450)
        context = img_rgb[130:155, 10:450].copy()
        # Inpaint target part of context
        # Offset of target relative to context: y: 5..19, x: 34..345
        context[5:19, 34:345] = inpainted_rgb
        
        # Place rendered text at baseline y ~ 145 -> rel y in context ~ 15 - fh + 1
        y_pos = 14 - fh + 2  # align baseline
        x_pos = 35
        
        alpha = rend_np[:, :, 3].astype(np.float32) / 255.0
        txt = rend_np[:, :, :3].astype(np.float32)
        
        h_use = min(fh, context.shape[0] - y_pos)
        w_use = min(fw, context.shape[1] - x_pos)
        
        for c in range(3):
            context[y_pos:y_pos+h_use, x_pos:x_pos+w_use, c] = (
                alpha[:h_use, :w_use] * txt[:h_use, :w_use, c] +
                (1.0 - alpha[:h_use, :w_use]) * context[y_pos:y_pos+h_use, x_pos:x_pos+w_use, c].astype(np.float32)
            ).astype(np.uint8)
            
        axes[idx].imshow(context)
        axes[idx].set_title(f"Option {idx+1}: {os.path.basename(font_path)} size={fsize} sigma={blur_sigma} (rendered size: {fw}x{fh})")
        axes[idx].axhline(15, color="red", alpha=0.3, linestyle="--")
        
    plt.tight_layout()
    plt.savefig("results/demo_inputs/font_align_options.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved font_align_options.png")

if __name__ == "__main__":
    test_font_alignments()
