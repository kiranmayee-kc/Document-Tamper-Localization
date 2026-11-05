"""Inspect iii-1 SEM MEMO.jpg and locate target row."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SRC_PATH = r"C:\Users\kiran\Downloads\iii-1 SEM MEMO.jpg"
OUTPUT_INSPECT_PATH = os.path.abspath("results/demo_inputs/memo_inspect.png")

def main():
    if not os.path.exists(SRC_PATH):
        print(f"Error: {SRC_PATH} not found!")
        return

    pil_img = Image.open(SRC_PATH)
    print(f"Format: {pil_img.format}, Size: {pil_img.size}, Mode: {pil_img.mode}")

    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"OpenCV shape: {img_rgb.shape}")

    # Let's save a resized overview with coordinate grid for inspection
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(img_rgb)
    ax.set_title(f"iii-1 SEM MEMO.jpg ({w}x{h})")
    for y in range(0, h, 100):
        ax.axhline(y, color="cyan", linewidth=0.5, alpha=0.6)
    for x in range(0, w, 100):
        ax.axvline(x, color="yellow", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_INSPECT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_INSPECT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overview: {OUTPUT_INSPECT_PATH}")

if __name__ == "__main__":
    main()
