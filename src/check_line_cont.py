"""Inspect line continuation after tampered text."""
import os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_PATH = os.path.abspath("results/demo_inputs/sample demo 6 allt.jpg")
img_bgr = cv2.imread(SRC_PATH)
line_crop = img_bgr[133:152, 0:512]

# Let's inspect where ", under" or similar starts
# In the original image: "COMPUTER SCIENCE AND ENGINEERING (DATA SCIENCE) (CSD), under ..."
# If we replace "COMPUTER SCIENCE AND ENGINEERING (DATA SCIENCE) (CSD)" with "Electronics and Communication Engineering",
# where should ", under ..." start?
# Let's check coordinates.
print("Line width:", line_crop.shape[1])
