"""Analyze the target text line in sample demo 6 allt.jpg and test font rendering."""
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

def main():
    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # In line y=136..148:
    # "in COMPUTER SCIENCE AND ENGINEERING (DATA SCIENCE) (CSD), under ..."
    # Let's inspect x=35 to x=320
    target_crop = img_rgb[134:152, 25:350]
    
    # Let's test inpainting on the text portion
    gray_crop = cv2.cvtColor(target_crop, cv2.COLOR_RGB2GRAY)
    
    # Text stroke segmentation
    # Background in this line is around 120-135, text strokes are around 30-70
    text_mask = (gray_crop < 85).astype(np.uint8) * 255
    # Dilate text mask slightly for inpainting
    kernel = np.ones((2, 2), np.uint8)
    inpaint_mask = cv2.dilate(text_mask, kernel, iterations=1)
    
    # Telea inpainting
    inpainted_crop = cv2.inpaint(cv2.cvtColor(target_crop, cv2.COLOR_RGB2BGR), inpaint_mask, 3, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_crop, cv2.COLOR_BGR2RGB)
    
    # Test rendering "Electronics and Communication Engineering"
    # Or "ELECTRONICS AND COMMUNICATION ENGINEERING"
    # Let's check the font size: character height in the original text is about 8-9 pixels (capital letters)
    # With supersampling (say 4x), render at font size ~32-36, then resize to match width/height
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 6))
    axes[0].imshow(target_crop)
    axes[0].set_title("Original Crop (y=134..152, x=25..350)")
    
    axes[1].imshow(inpaint_mask, cmap="gray")
    axes[1].set_title("Inpaint Mask (Text Strokes)")
    
    axes[2].imshow(inpainted_rgb)
    axes[2].set_title("Inpainted Clean Background")
    
    plt.tight_layout()
    plt.savefig("results/demo_inputs/test_inpaint_line.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved test_inpaint_line.png")

if __name__ == "__main__":
    main()
