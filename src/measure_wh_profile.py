"""Measure exact extents of 'World' and 'History'."""
import os, sys
import cv2
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Let's inspect x in [25, 140], y in [155, 167]
crop = gray[155:167, 25:140]
col_profile = np.mean(crop, axis=0)

# Dark columns (text strokes) vs light columns (spaces)
for x_rel, val in enumerate(col_profile):
    x_abs = 25 + x_rel
    print(f"x={x_abs:3d}: gray={val:5.1f}")
