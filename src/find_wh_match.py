"""Template match 'World History' across lines."""
import os, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Render template of "World" and "History"
font_arial = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 9)
for word in ["World", "History"]:
    dummy = Image.new("L", (100, 20), 255)
    d = ImageDraw.Draw(dummy)
    d.text((5, 2), word, font=font_arial, fill=0)
    # Find bounding box
    bbox = dummy.getbbox()
    tmpl = np.array(dummy.crop(bbox))
    
    res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(f"Word '{word}': max_val={max_val:.3f} at loc={max_loc}")

# Let's also check with arial bold, arial narrow, times new roman, calibri
for font_name, font_file in [
    ("Arial", "C:/Windows/Fonts/arial.ttf"),
    ("Arial Bold", "C:/Windows/Fonts/arialbd.ttf"),
    ("Times New Roman", "C:/Windows/Fonts/times.ttf"),
    ("Calibri", "C:/Windows/Fonts/calibri.ttf"),
    ("Courier New", "C:/Windows/Fonts/cour.ttf"),
]:
    for size in [7, 8, 9, 10, 11]:
        f = ImageFont.truetype(font_file, size)
        dummy = Image.new("L", (100, 20), 255)
        d = ImageDraw.Draw(dummy)
        d.text((5, 2), "History", font=f, fill=0)
        bbox = dummy.getbbox()
        if bbox is None: continue
        tmpl = np.array(dummy.crop(bbox))
        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.45:
            print(f"Font {font_name} sz={size}: match={max_val:.3f} at loc={max_loc}")
