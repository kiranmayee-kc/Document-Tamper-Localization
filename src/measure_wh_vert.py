"""Analyze font characteristics of the table text in sample demo 7.jpg."""
import os, sys
import cv2
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 7.jpg")
img_bgr = cv2.imread(SRC_PATH)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Vertical profile of "World" (x in 42..67) and "History" (x in 72..128)
crop_w = gray[150:172, 42:67]
crop_h = gray[150:172, 72:128]

for y_rel in range(crop_h.shape[0]):
    y_abs = 150 + y_rel
    v_w = np.mean(crop_w[y_rel, :])
    v_h = np.mean(crop_h[y_rel, :])
    print(f"y={y_abs:3d}: 'World'={v_w:5.1f}, 'History'={v_h:5.1f}")
