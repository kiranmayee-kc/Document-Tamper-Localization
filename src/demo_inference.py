"""TamperDoc — Real-Document Demo & Diagnostic Inference Pipeline.

Standalone inference script for evaluating single external document images or
comparing original vs. tampered document pairs using the trained TamperDoc model.
Includes detailed logit and probability diagnostic statistics to analyze model responses.

Usage Examples:
  Single image inference:
    python src/demo_inference.py --image "results/demo_inputs/ed cur bill.jpg"

  Original vs. Tampered comparison:
    python src/demo_inference.py --original "results/demo_inputs/og current bill.jpg" --tampered "results/demo_inputs/ed cur bill.jpg"

  Custom checkpoint & output directory:
    python src/demo_inference.py --image "path/to/doc.jpg" --checkpoint "checkpoints/best_model.pt" --output-dir "results/demo"
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from src.model import TamperDocModel

# Default paths
DEFAULT_CHECKPOINT = os.path.abspath("checkpoints/best_model.pt")
DEFAULT_OUTPUT_DIR = os.path.abspath("results/demo")
TARGET_SIZE = (512, 512)
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


def validate_and_load_image(image_path: str, label: str = "Input") -> Tuple[Image.Image, np.ndarray, torch.Tensor, torch.Tensor]:
    """Validates image file properties and prepares model input tensors.

    Checks:
    - File exists and is readable
    - Format is JPEG / JPG
    - Mode is RGB (3 channels)
    - Strict dimensions: exactly 512x512 (no silent resizing)

    Returns:
        pil_img: PIL Image in RGB
        img_np: numpy array [512, 512, 3] in float32 [0, 1]
        normalized_tensor: [1, 3, 512, 512] normalized for RGB branch
        raw_tensor: [1, 3, 512, 512] unnormalized in [0, 1] for DCT/SRM
    """
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"{label} image file not found: {abs_path}")

    if not os.path.isfile(abs_path):
        raise ValueError(f"{label} path is not a valid file: {abs_path}")

    try:
        pil_img = Image.open(abs_path)
    except Exception as e:
        raise ValueError(f"Unable to read {label} image: {e}")

    # Validate format
    img_format = (pil_img.format or "").upper()
    if img_format not in ("JPEG", "JPG"):
        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in (".jpg", ".jpeg"):
            raise ValueError(
                f"{label} image must be a JPEG/JPG file, got format='{img_format}', ext='{ext}'"
            )

    # Validate dimensions strictly (NO silent resizing)
    width, height = pil_img.size
    if (width, height) != TARGET_SIZE:
        raise ValueError(
            f"{label} image dimensions must be exactly {TARGET_SIZE[0]}x{TARGET_SIZE[1]}, "
            f"got {width}x{height}. Please provide a 512x512 image or resize manually beforehand."
        )

    # Ensure RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # To tensor [3, 512, 512] in [0, 1]
    raw_tensor = TF.to_tensor(pil_img)

    # Normalized RGB tensor for pretrained backbone
    normalized_tensor = TF.normalize(
        raw_tensor, mean=NORMALIZE_MEAN, std=NORMALIZE_STD
    )

    # Add batch dimension [1, 3, 512, 512]
    raw_tensor_batched = raw_tensor.unsqueeze(0)
    norm_tensor_batched = normalized_tensor.unsqueeze(0)

    # NumPy float representation [512, 512, 3] in [0, 1]
    img_np = np.asarray(pil_img, dtype=np.float32) / 255.0

    return pil_img, img_np, norm_tensor_batched, raw_tensor_batched


def load_model(checkpoint_path: str) -> Tuple[TamperDocModel, dict]:
    """Loads TamperDoc model with weights restored from checkpoint."""
    abs_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Checkpoint file not found: {abs_path}")

    checkpoint = torch.load(abs_path, map_location="cpu")

    model = TamperDocModel(
        use_rgb=True,
        use_dct=True,
        use_srm=True,
        pretrained_rgb=False,
        branch_channels=128,
        fused_channels=128,
        target_size=TARGET_SIZE,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, checkpoint


def create_localization_overlay(img_np: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Creates a transparent red highlight overlay on top of the RGB image."""
    overlay = img_np.copy()
    red_color = np.array([0.85, 0.15, 0.15], dtype=np.float32)  # High-visibility red
    mask_3d = pred_mask[:, :, np.newaxis].astype(np.float32)

    blend_factor = alpha * mask_3d
    overlay = (1.0 - blend_factor) * overlay + blend_factor * red_color
    return np.clip(overlay, 0.0, 1.0)


def compute_diagnostic_statistics(
    logits_2d: np.ndarray,
    prob_map_2d: np.ndarray,
    label: str = "Tampered Image",
) -> dict:
    """Computes comprehensive diagnostic statistics on raw logits and sigmoid probabilities."""
    total_pixels = logits_2d.size

    # 1. Raw Logits stats
    logit_min = float(np.min(logits_2d))
    logit_max = float(np.max(logits_2d))
    logit_mean = float(np.mean(logits_2d))
    logit_std = float(np.std(logits_2d))

    # 2. Sigmoid Probabilities stats
    prob_min = float(np.min(prob_map_2d))
    prob_max = float(np.max(prob_map_2d))
    prob_mean = float(np.mean(prob_map_2d))
    prob_std = float(np.std(prob_map_2d))
    prob_p95 = float(np.percentile(prob_map_2d, 95.0))
    prob_p99 = float(np.percentile(prob_map_2d, 99.0))
    prob_p999 = float(np.percentile(prob_map_2d, 99.9))

    # 3. Threshold counts
    count_ge_01 = int(np.sum(prob_map_2d >= 0.1))
    count_ge_025 = int(np.sum(prob_map_2d >= 0.25))
    count_ge_05 = int(np.sum(prob_map_2d >= 0.5))

    pct_ge_01 = (count_ge_01 / total_pixels) * 100.0
    pct_ge_025 = (count_ge_025 / total_pixels) * 100.0
    pct_ge_05 = (count_ge_05 / total_pixels) * 100.0

    # 4. Top 20 highest-probability pixels
    flat_indices = np.argsort(prob_map_2d.ravel())[-20:][::-1]
    top_20 = []
    for rank, idx in enumerate(flat_indices, 1):
        y, x = np.unravel_index(idx, prob_map_2d.shape)
        top_20.append({
            "rank": rank,
            "y": int(y),
            "x": int(x),
            "probability": float(prob_map_2d[y, x]),
            "logit": float(logits_2d[y, x]),
        })

    stats = {
        "label": label,
        "total_pixels": total_pixels,
        "logits": {
            "min": logit_min,
            "max": logit_max,
            "mean": logit_mean,
            "std": logit_std,
        },
        "probabilities": {
            "min": prob_min,
            "max": prob_max,
            "mean": prob_mean,
            "std": prob_std,
            "p95": prob_p95,
            "p99": prob_p99,
            "p99_9": prob_p999,
        },
        "threshold_counts": {
            "ge_0_1": {"count": count_ge_01, "percentage": pct_ge_01},
            "ge_0_25": {"count": count_ge_025, "percentage": pct_ge_025},
            "ge_0_5": {"count": count_ge_05, "percentage": pct_ge_05},
        },
        "top_20_pixels": top_20,
    }
    return stats


def print_diagnostic_report(stats: dict) -> None:
    """Prints a structured diagnostic report to the console."""
    label = stats["label"]
    lg = stats["logits"]
    pr = stats["probabilities"]
    tc = stats["threshold_counts"]
    top20 = stats["top_20_pixels"]
    total = stats["total_pixels"]

    print("\n" + "=" * 80)
    print(f"DIAGNOSTIC OUTPUT ANALYSIS — {label.upper()}")
    print("=" * 80)
    print("1. Raw Logits Statistics:")
    print(f"   Minimum:             {lg['min']:+10.6f}")
    print(f"   Maximum:             {lg['max']:+10.6f}")
    print(f"   Mean:                {lg['mean']:+10.6f}")
    print(f"   Standard Deviation:  {lg['std']:10.6f}")

    print("\n2. Sigmoid Probabilities Statistics:")
    print(f"   Minimum:             {pr['min']:10.6f}")
    print(f"   Maximum:             {pr['max']:10.6f}")
    print(f"   Mean:                {pr['mean']:10.6f}")
    print(f"   Standard Deviation:  {pr['std']:10.6f}")
    print(f"   95.0th Percentile:   {pr['p95']:10.6f}")
    print(f"   99.0th Percentile:   {pr['p99']:10.6f}")
    print(f"   99.9th Percentile:   {pr['p99_9']:10.6f}")

    print("\n3. Pixel Threshold Counts:")
    print(f"   Probability >= 0.10: {tc['ge_0_1']['count']:7,d} / {total:,d} ({tc['ge_0_1']['percentage']:6.3f}%)")
    print(f"   Probability >= 0.25: {tc['ge_0_25']['count']:7,d} / {total:,d} ({tc['ge_0_25']['percentage']:6.3f}%)")
    print(f"   Probability >= 0.50: {tc['ge_0_5']['count']:7,d} / {total:,d} ({tc['ge_0_5']['percentage']:6.3f}%)  [Decision Threshold]")

    print("\n4. Top 20 Highest-Probability Pixels:")
    print("   Rank |  Coord (Y, X)  |  Probability  |  Raw Logit")
    print("   -----+----------------+---------------+-----------")
    for item in top20:
        print(f"   {item['rank']:4d} | ({item['y']:3d}, {item['x']:3d})     |    {item['probability']:8.6f}   |  {item['logit']:+9.5f}")
    print("=" * 80)


def save_visualizations(
    output_dir: str,
    tampered_pil: Image.Image,
    tampered_np: np.ndarray,
    prob_map: np.ndarray,
    pred_mask: np.ndarray,
    original_pil: Optional[Image.Image] = None,
    original_np: Optional[np.ndarray] = None,
    orig_prob_map: Optional[np.ndarray] = None,
) -> dict:
    """Saves output images and comparison figures to the destination directory.

    Returns:
        dict mapping artifact name to saved file path
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}

    # 1. Save input / tampered image copy
    tampered_copy_path = os.path.join(output_dir, "tampered_image.png")
    tampered_pil.save(tampered_copy_path)
    saved_files["tampered_image"] = tampered_copy_path

    # 2. Save predicted binary mask (0 = black, 255 = white)
    pred_mask_path = os.path.join(output_dir, "predicted_mask.png")
    mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L")
    mask_pil.save(pred_mask_path)
    saved_files["predicted_mask"] = pred_mask_path

    # 3. Save standard probability map heatmap with [0, 1] colorbar
    prob_map_path = os.path.join(output_dir, "probability_map.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(prob_map, cmap="hot", vmin=0.0, vmax=1.0)
    ax.set_title("Predicted Tamper Probability Map (Scale: [0, 1])", fontsize=11, pad=10)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(prob_map_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files["probability_map"] = prob_map_path

    # 4. Save continuous full-range probability map WITHOUT thresholding (auto dynamic range)
    prob_map_full_range_path = os.path.join(output_dir, "probability_map_full_range.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    p_min, p_max = float(np.min(prob_map)), float(np.max(prob_map))
    # If dynamic range is extremely narrow, add epsilon for display
    vmin = p_min
    vmax = p_max if (p_max - p_min) > 1e-6 else p_min + 1e-3
    im_full = ax.imshow(prob_map, cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(f"Continuous Probability Map (Full Dynamic Range: [{p_min:.5f}, {p_max:.5f}])", fontsize=10, pad=10)
    ax.axis("off")
    cbar_full = plt.colorbar(im_full, ax=ax, fraction=0.046, pad=0.04)
    cbar_full.ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(prob_map_full_range_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files["probability_map_full_range"] = prob_map_full_range_path

    # 5. Save localization overlay image
    overlay_path = os.path.join(output_dir, "localization_overlay.png")
    overlay_np = create_localization_overlay(tampered_np, pred_mask)
    overlay_pil = Image.fromarray((overlay_np * 255).astype(np.uint8), mode="RGB")
    overlay_pil.save(overlay_path)
    saved_files["localization_overlay"] = overlay_path

    # 6. Optional Original vs Tampered Comparison Figure
    if original_pil is not None and original_np is not None:
        original_copy_path = os.path.join(output_dir, "original_image.png")
        original_pil.save(original_copy_path)
        saved_files["original_image"] = original_copy_path

        comparison_path = os.path.join(output_dir, "comparison.png")
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))
        titles = [
            "1. Original Document",
            "2. Tampered Input",
            "3. Tamper Probability",
            "4. Predicted Binary Mask",
            "5. Localization Overlay",
        ]

        axes[0].imshow(original_np)
        axes[1].imshow(tampered_np)
        im_heat = axes[2].imshow(prob_map, cmap="hot", vmin=0.0, vmax=1.0)
        plt.colorbar(im_heat, ax=axes[2], fraction=0.046, pad=0.04)
        axes[3].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[4].imshow(overlay_np)

        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
            ax.axis("off")

        fig.suptitle("TamperDoc — Document Tampering Localization Analysis", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files["comparison"] = comparison_path

        # 7. Additional Diagnostic Comparison Figure (Original vs Tampered Probability Maps & Difference)
        if orig_prob_map is not None:
            diag_comp_path = os.path.join(output_dir, "diagnostic_probability_comparison.png")
            diff_map = np.abs(prob_map - orig_prob_map)

            fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
            ax_titles = [
                "Original Prob Map",
                "Tampered Prob Map",
                "Absolute Prob Difference |T - O|",
                "Tampered Overlay",
            ]

            im0 = axes[0].imshow(orig_prob_map, cmap="hot", vmin=0.0, vmax=1.0)
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(prob_map, cmap="hot", vmin=0.0, vmax=1.0)
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(diff_map, cmap="magma")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            axes[3].imshow(overlay_np)

            for ax, title in zip(axes, ax_titles):
                ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
                ax.axis("off")

            fig.suptitle("TamperDoc — Diagnostic Probability Difference Analysis", fontsize=13, y=1.02)
            plt.tight_layout()
            plt.savefig(diag_comp_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files["diagnostic_probability_comparison"] = diag_comp_path

    return saved_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="TamperDoc Real-Document Demo / Inference & Diagnostic Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Single image inference & diagnosis:
    python src/demo_inference.py --image "results/demo_inputs/ed cur bill.jpg"

  Original vs. Tampered comparison:
    python src/demo_inference.py --original "results/demo_inputs/og current bill.jpg" --tampered "results/demo_inputs/ed cur bill.jpg"

  Custom checkpoint & output:
    python src/demo_inference.py --image "sample.jpg" --checkpoint "checkpoints/best_model.pt" --output-dir "results/demo"
""",
    )
    # Image options
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to 512x512 JPEG image to inspect for tampering.",
    )
    parser.add_argument(
        "--tampered",
        type=str,
        default=None,
        help="Path to 512x512 JPEG tampered document image.",
    )
    parser.add_argument(
        "--original",
        type=str,
        default=None,
        help="Optional path to 512x512 JPEG authentic/original document image for visual and statistical comparison.",
    )
    # System & Model options
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save demo results and visualizations (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of PyTorch CPU threads to utilize (default: 4).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine tampered image path from --image or --tampered
    tampered_input_path = args.tampered if args.tampered is not None else args.image
    if tampered_input_path is None:
        print("Error: Must provide an image to evaluate via --image <path> or --tampered <path>.")
        print("Run with --help for detailed usage instructions.")
        sys.exit(1)

    torch.set_num_threads(args.threads)

    print("=" * 80)
    print("TAMPERDOC — REAL-DOCUMENT INFERENCE & DIAGNOSTIC PIPELINE")
    print("=" * 80)
    print(f"Tampered Input:  {tampered_input_path}")
    if args.original:
        print(f"Original Input:  {args.original}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Output Directory:{args.output_dir}")
    print(f"CPU Threads:     {args.threads}")
    print(f"Decision Rule:   logits >= 0.0  (mathematically equivalent to sigmoid(logits) >= 0.5)")
    print("-" * 80)

    # 1. Validate & Load Tampered Input Image
    print("[1/5] Validating and preprocessing input image(s)...")
    tampered_pil, tampered_np, norm_x, raw_x = validate_and_load_image(
        tampered_input_path, label="Tampered"
    )
    print(f"  [OK] Tampered image: {tampered_pil.size[0]}x{tampered_pil.size[1]} RGB JPEG")

    # Optional Original Image
    original_pil = None
    original_np = None
    orig_norm_x = None
    orig_raw_x = None
    if args.original:
        original_pil, original_np, orig_norm_x, orig_raw_x = validate_and_load_image(
            args.original, label="Original"
        )
        print(f"  [OK] Original image: {original_pil.size[0]}x{original_pil.size[1]} RGB JPEG")

    # 2. Load Checkpoint & Model
    print("\n[2/5] Loading trained TamperDoc model...")
    model, ckpt = load_model(args.checkpoint)
    ckpt_epoch = ckpt.get("epoch", "unknown")
    best_val_f1 = ckpt.get("best_val_f1", ckpt.get("val_stats", {}).get("val_f1", "unknown"))
    print(f"  [OK] Model loaded successfully.")
    print(f"       Checkpoint Epoch: {ckpt_epoch}")
    print(f"       Best Val F1:      {best_val_f1}")

    # 3. Inference on Tampered Image
    print("\n[3/5] Running neural network inference on tampered document...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(norm_x, raw_x=raw_x)
        logits = outputs["tamper_logits"]  # [1, 1, 512, 512]
        prob_tensor = torch.sigmoid(logits)
        pred_mask_tensor = logits >= 0.0
    inference_time = time.perf_counter() - t0

    # Sanity checks on output
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise RuntimeError("Model output contains NaN or Inf values!")

    logits_np = logits.squeeze().cpu().numpy()  # [512, 512] float32
    prob_map = prob_tensor.squeeze().cpu().numpy()  # [512, 512] float32 in [0, 1]
    pred_mask = pred_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)  # [512, 512] uint8 {0, 1}

    # Descriptive Statistics
    total_pixels = 512 * 512
    tampered_pixels = int(np.sum(pred_mask))
    tamper_percentage = (tampered_pixels / total_pixels) * 100.0

    print(f"  [OK] Tampered inference finished in {inference_time*1000:.2f} ms ({inference_time:.3f} s)")
    print(f"       Output Tensor Shape:      {list(logits.shape)}")
    print(f"       Predicted Tamper Pixels:  {tampered_pixels:,} / {total_pixels:,}")
    print(f"       Predicted Tamper Fraction:{tamper_percentage:.2f}% (descriptive metric)")

    # 4. Optional Inference on Original Image & Comparison
    orig_prob_map = None
    orig_logits_np = None
    if args.original and orig_norm_x is not None and orig_raw_x is not None:
        print("\n[4/5] Running neural network inference on authentic/original document...")
        t0_orig = time.perf_counter()
        with torch.inference_mode():
            orig_outputs = model(orig_norm_x, raw_x=orig_raw_x)
            orig_logits = orig_outputs["tamper_logits"]
            orig_prob_tensor = torch.sigmoid(orig_logits)
        orig_inference_time = time.perf_counter() - t0_orig

        if torch.isnan(orig_logits).any() or torch.isinf(orig_logits).any():
            raise RuntimeError("Original model output contains NaN or Inf values!")

        orig_logits_np = orig_logits.squeeze().cpu().numpy()
        orig_prob_map = orig_prob_tensor.squeeze().cpu().numpy()
        orig_pred_mask = (orig_logits >= 0.0).squeeze().cpu().numpy().astype(np.uint8)
        orig_tampered_pixels = int(np.sum(orig_pred_mask))

        print(f"  [OK] Original inference finished in {orig_inference_time*1000:.2f} ms ({orig_inference_time:.3f} s)")
        print(f"       Predicted Tamper Pixels:  {orig_tampered_pixels:,} / {total_pixels:,}")
    else:
        print("\n[4/5] Skipping original image inference (no --original provided)...")

    # 5. Diagnostic Statistics Computation & Output
    tampered_stats = compute_diagnostic_statistics(
        logits_np, prob_map, label=f"Tampered Document ({os.path.basename(tampered_input_path)})"
    )
    print_diagnostic_report(tampered_stats)

    if orig_logits_np is not None and orig_prob_map is not None:
        orig_stats = compute_diagnostic_statistics(
            orig_logits_np, orig_prob_map, label=f"Original Document ({os.path.basename(args.original)})"
        )
        print_diagnostic_report(orig_stats)

        # Difference Statistics
        diff_prob = np.abs(prob_map - orig_prob_map)
        mean_abs_diff = float(np.mean(diff_prob))
        max_abs_diff = float(np.max(diff_prob))

        print("\n" + "=" * 80)
        print("DIAGNOSTIC COMPARISON: ORIGINAL VS. TAMPERED PROBABILITY MAPS")
        print("=" * 80)
        print(f"Mean Absolute Probability Difference:   {mean_abs_diff:10.6f}")
        print(f"Maximum Absolute Probability Difference:{max_abs_diff:10.6f}")
        print("=" * 80)

    # 6. Save Visualizations
    print("\n[5/5] Generating and saving visual artifacts...")
    saved_files = save_visualizations(
        output_dir=args.output_dir,
        tampered_pil=tampered_pil,
        tampered_np=tampered_np,
        prob_map=prob_map,
        pred_mask=pred_mask,
        original_pil=original_pil,
        original_np=original_np,
        orig_prob_map=orig_prob_map,
    )
    for name, path in saved_files.items():
        print(f"  Saved {name:32s} -> {path}")

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)
    print(f"Input Document:              {tampered_input_path}")
    print(f"Input Dimensions:            512x512 RGB")
    print(f"Checkpoint Path:             {args.checkpoint}")
    print(f"Checkpoint Epoch:            {ckpt_epoch}")
    print(f"Model Output Shape:          {list(logits.shape)}")
    print(f"Decision Threshold:          logits >= 0.0 (sigmoid >= 0.5)")
    print(f"Inference Latency:           {inference_time*1000:.2f} ms")
    print(f"Predicted Tampered Pixels:   {tampered_pixels:,} ({tamper_percentage:.2f}%)")
    print(f"Output Artifacts Saved To:   {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
