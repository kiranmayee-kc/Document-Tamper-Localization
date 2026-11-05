"""Find the exact bounding box of 'World History'."""
import os, sys
import cv2
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Let's inspect the entire image for text "World History"
# In high school transcripts:
# Grade 9 (Freshman): e.g. English 9, Algebra I, World History, Physical Science...
# Grade 10: English 10, Geometry, US History, Biology...
# Let's search across all table sections:
# Let's find connected components in the text regions

# Binarize with threshold
_, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

# Let's test several regions where "World History" could be:
# Let's save a crop around y=190..230 and y=140..180 and y=230..270
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Let's inspect rows around y=170..270 in detail
for y_s, y_e in [(140, 190), (190, 240), (240, 290), (290, 340), (340, 390)]:
    crop = img_rgb[y_s:y_e, 20:250]
    # Check if there is text
    print(f"Region y={y_s}..{y_e}, x=20..250: mean gray = {np.mean(gray[y_s:y_e, 20:250]):.1f}")

# Let's write a script that analyzes every row on left (x=20..250) and right (x=260..490)
