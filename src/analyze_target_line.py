"""Detailed analysis of target text line in sample demo 6 allt.jpg."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 6 allt.jpg")

def main():
    img_bgr = cv2.imread(SRC_PATH)
    
    # Target line is around y=138..150
    # Let's inspect y=135..152, x=25..350
    line_crop = img_bgr[135:153, 25:400]
    gray = cv2.cvtColor(line_crop, cv2.COLOR_BGR2GRAY)
    
    # Text pixels are dark, background is lighter
    bg_sample = img_bgr[130:135, 100:200]
    bg_median = np.median(bg_sample, axis=(0, 1))
    
    # Darkest pixels (text stroke centers)
    text_mask = gray < 80
    text_median = np.median(line_crop[text_mask], axis=0) if np.any(text_mask) else [0,0,0]
    
    print(f"Background RGB (median): {bg_median[::-1]}")
    print(f"Text RGB (median): {text_median[::-1]}")
    print(f"Line height in pixels: {line_crop.shape[0]}")
    print(f"Gray min: {gray.min()}, max: {gray.max()}, mean: {gray.mean():.1f}")

if __name__ == "__main__":
    main()
