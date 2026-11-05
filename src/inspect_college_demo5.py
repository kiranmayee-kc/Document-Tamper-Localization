"""Inspect college Demo 5.jpg to locate 'B. Tech - CSDS'."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SRC_PATH = os.path.abspath("results/demo_inputs/college Demo 5.jpg")
OUT_PATH = os.path.abspath("results/demo_inputs/college_demo5_inspect.png")

def main():
    pil_img = Image.open(SRC_PATH)
    print(f"Format: {pil_img.format}, Size: {pil_img.size}, Mode: {pil_img.mode}")

    img_bgr = cv2.imread(SRC_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"Dimensions: {w}x{h}, Channels: {img_rgb.shape[2]}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_rgb)
    ax.set_title(f"college Demo 5.jpg ({w}x{h}) with coordinate grid")
    for y in range(0, h, 25):
        ax.axhline(y, color="cyan", linewidth=0.5, alpha=0.5)
        ax.text(5, y - 2, f"{y}", color="red", fontsize=6)
    for x in range(0, w, 50):
        ax.axvline(x, color="yellow", linewidth=0.5, alpha=0.5)
        ax.text(x + 2, 15, f"{x}", color="blue", fontsize=6, rotation=90)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
