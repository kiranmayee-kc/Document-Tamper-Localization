"""Pinpoint exact bounding box of 'B. Tech - CSDS' in college Demo 5.jpg."""
import os, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC_PATH = os.path.abspath("results/demo_inputs/college Demo 5.jpg")
img_bgr = cv2.imread(SRC_PATH)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Search for "B. Tech" or "CSDS" or "Tech" with template matching
for word in ["CSDS", "Tech", "B. Tech"]:
    for fpath in ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibrib.ttf"]:
        for fsize in [8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]:
            font = ImageFont.truetype(fpath, fsize)
            dummy = Image.new("L", (150, 40), 255)
            d = ImageDraw.Draw(dummy)
            d.text((5, 5), word, font=font, fill=0)
            bbox = dummy.getbbox()
            if bbox is None: continue
            tmpl = np.array(dummy.crop(bbox))
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            min_v, max_v, min_l, max_l = cv2.minMaxLoc(res)
            if max_v > 0.45:
                print(f"Word '{word}' | Font {os.path.basename(fpath)} sz={fsize} | Score={max_v:.3f} at loc={max_l}")
