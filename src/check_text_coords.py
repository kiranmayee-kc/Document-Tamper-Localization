"""Check OCR / exact text in line 135..153."""
import os, sys
import cv2
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 6 allt.jpg")
img_bgr = cv2.imread(SRC_PATH)

# Let's inspect where "in COMPUTER SCIENCE AND ENGINEERING" or similar text is
# Let's check y=136..148:
# x=28..45: "in "
# x=45..360: "COMPUTER SCIENCE AND ENGINEERING (DATA SCIENCE) (CSD)" or similar
print("Target image shape:", img_bgr.shape)

# Let's test inpainting on the text portion
# Say x=45 to x=360, y=137 to y=148
# Let's inspect horizontal profile
crop = img_bgr[136:148, 25:480]
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
print("Crop shape:", crop.shape)
