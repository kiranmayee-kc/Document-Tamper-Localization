"""Test rendering matching '7' and 'B' characters for the academic memo."""

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
OUTPUT_PATH = os.path.abspath("results/demo_inputs/font_matching_test.png")

def main():
    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Original '8' and 'A' crops
    orig_8 = img_rgb[98:112, 319:329]  # 14x10
    orig_A = img_rgb[98:112, 371:381]  # 14x10
    orig_9 = img_rgb[73:87, 319:329]   # 14x10
    orig_Ap = img_rgb[73:87, 369:383]  # 14x14

    # Background color in this table cell:
    bg_color = np.median(img_rgb[98:112, 330:345], axis=(0,1))
    text_color = np.min(img_rgb[98:112, 320:328], axis=(0,1))
    print(f"Background RGB: {bg_color}, Text min RGB: {text_color}")

    # Let's test font rendering with times.ttf, georgia.ttf, cambria.ttf, etc.
    fonts_to_test = ["times.ttf", "timesbd.ttf", "georgia.ttf", "georgiab.ttf", "cambria.ttc", "arial.ttf", "arialbd.ttf"]
    
    fig, axes = plt.subplots(len(fonts_to_test) + 1, 4, figsize=(12, 2 * (len(fonts_to_test) + 1)))

    # Row 0: Original 8, A, 9, A+
    for ax, crop, label in zip(axes[0], [orig_8, orig_A, orig_9, orig_Ap], ["Orig '8'", "Orig 'A'", "Orig '9'", "Orig 'A+'"]):
        big = cv2.resize(crop, (crop.shape[1]*8, crop.shape[0]*8), interpolation=cv2.INTER_NEAREST)
        ax.imshow(big)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axis("off")

    for i, font_name in enumerate(fonts_to_test, 1):
        try:
            # Render at high resolution (e.g., size=64) then scale down to ~8px height
            hi_img = Image.new("RGBA", (128, 128), (255, 255, 255, 0))
            draw = ImageDraw.Draw(hi_img)
            font = ImageFont.truetype(font_name, size=64)
            
            # Draw '7'
            draw.text((32, 20), "7", font=font, fill=(int(text_color[0]), int(text_color[1]), int(text_color[2]), 255))
            # Crop to bbox
            bbox_7 = hi_img.getbbox()
            crop_7 = hi_img.crop(bbox_7)
            
            # Draw 'B'
            hi_img_B = Image.new("RGBA", (128, 128), (255, 255, 255, 0))
            draw_B = ImageDraw.Draw(hi_img_B)
            draw_B.text((32, 20), "B", font=font, fill=(int(text_color[0]), int(text_color[1]), int(text_color[2]), 255))
            bbox_B = hi_img_B.getbbox()
            crop_B = hi_img_B.crop(bbox_B)

            # Resize down: '7' to height=8, width=5-6
            r7 = crop_7.resize((6, 8), Image.Resampling.LANCZOS)
            rB = crop_B.resize((6, 8), Image.Resampling.LANCZOS)

            # Paste onto local background patch
            patch_7 = np.tile(bg_color, (14, 10, 1)).astype(np.uint8)
            patch_B = np.tile(bg_color, (14, 10, 1)).astype(np.uint8)

            p7_pil = Image.fromarray(patch_7)
            pB_pil = Image.fromarray(patch_B)

            p7_pil.paste(r7, (2, 3), r7)
            pB_pil.paste(rB, (2, 3), rB)

            p7_np = np.array(p7_pil)
            pB_np = np.array(pB_pil)

            p7_big = cv2.resize(p7_np, (p7_np.shape[1]*8, p7_np.shape[0]*8), interpolation=cv2.INTER_NEAREST)
            pB_big = cv2.resize(pB_np, (pB_np.shape[1]*8, pB_np.shape[0]*8), interpolation=cv2.INTER_NEAREST)

            axes[i, 0].imshow(p7_big)
            axes[i, 0].set_title(f"{font_name} '7'", fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(pB_big)
            axes[i, 1].set_title(f"{font_name} 'B'", fontsize=9)
            axes[i, 1].axis("off")

            # Draw test row placement
            axes[i, 2].axis("off")
            axes[i, 3].axis("off")

        except Exception as e:
            axes[i, 0].text(0.1, 0.5, f"Error: {e}", fontsize=8)
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")
            axes[i, 2].axis("off")
            axes[i, 3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved font matching test: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
