"""
TamperDoc intermediate-feature visualizer (normal Python script; no Streamlit).

Run from the project root:
    python visualize_features.py --image "path/to/document.jpg"

Optional:
    python visualize_features.py --image "path/to/document.jpg" \
        --checkpoint "checkpoints/best_model.pt" \
        --output-dir "results/feature_visualizations"

Input must be exactly 512x512 pixels, matching the existing inference pipeline.
This script loads the existing checkpoint; it does not train or modify the model.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, os.path.abspath("."))
from src.demo_inference import load_model

TARGET_SIZE = (512, 512)
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
THRESHOLD = 0.5


def normalize_for_display(array):
    """Min-max normalization for visualization only."""
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.float32)
    lo = float(array[finite].min())
    hi = float(array[finite].max())
    result = np.zeros_like(array, dtype=np.float32)
    if hi - lo > 1e-8:
        result[finite] = (array[finite] - lo) / (hi - lo)
    return result


def feature_activation_map(tensor):
    """Summarize [B,C,H,W] learned features as mean absolute activation."""
    if not torch.is_tensor(tensor) or tensor.ndim != 4:
        shape = tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor).__name__
        raise ValueError(f"Expected [B,C,H,W] tensor, got {shape}")
    arr = tensor.detach().float().cpu()[0].abs().mean(dim=0).numpy()
    return normalize_for_display(arr)


def raw_intermediate_map(tensor):
    """Summarize [B,C,H,W], [C,H,W], or [H,W] intermediate tensors."""
    if not torch.is_tensor(tensor):
        raise TypeError("Extractor did not return a PyTorch tensor.")
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        return normalize_for_display(arr)
    if arr.ndim == 3:
        return normalize_for_display(np.mean(np.abs(arr), axis=0))
    raise ValueError(f"Cannot visualize tensor with shape {arr.shape}")


def save_heatmap(array, title, path, cmap="inferno", fixed_range=True):
    fig, ax = plt.subplots(figsize=(6, 6))
    if fixed_range:
        im = ax.imshow(array, cmap=cmap, vmin=0, vmax=1)
    else:
        im = ax.imshow(array, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_overlay(image_np, mask, path, alpha=0.45):
    overlay = image_np.copy()
    red = np.array([0.85, 0.15, 0.15], dtype=np.float32)
    mask3 = mask[..., None].astype(np.float32)
    overlay = (1 - alpha * mask3) * overlay + (alpha * mask3) * red
    Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(
        description="Save actual TamperDoc intermediate feature visualizations."
    )
    parser.add_argument("--image", required=True, help="Path to a 512x512 document image.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output-dir", default="results/feature_visualizations")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    if image.size != TARGET_SIZE:
        raise ValueError(
            f"Image must be exactly 512x512; received {image.size}. "
            "No automatic resizing is applied."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_num_threads(args.threads)

    image_np = np.asarray(image, dtype=np.float32) / 255.0
    raw_x = TF.to_tensor(image).unsqueeze(0)
    norm_x = TF.normalize(raw_x.clone(), mean=NORMALIZE_MEAN, std=NORMALIZE_STD)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint)

    print("Running inference...")
    with torch.inference_mode():
        outputs = model(norm_x, raw_x=raw_x)
        logits = outputs["tamper_logits"]
        probability = torch.sigmoid(logits)
        predicted_mask = (probability >= THRESHOLD).squeeze().cpu().numpy().astype(np.uint8)

    if not torch.isfinite(logits).all():
        raise RuntimeError("Model output contains NaN or Inf values.")

    prob_map = probability.squeeze().cpu().numpy()
    saved = []

    # Save input, probability, binary mask, and overlay.
    input_path = os.path.join(args.output_dir, "input_document.png")
    image.save(input_path)
    saved.append(input_path)

    probability_path = os.path.join(args.output_dir, "tamper_probability_map.png")
    save_heatmap(prob_map, "Tamper Probability (0–1)", probability_path, cmap="hot")
    saved.append(probability_path)

    mask_path = os.path.join(args.output_dir, "predicted_binary_mask.png")
    Image.fromarray((predicted_mask * 255).astype(np.uint8)).save(mask_path)
    saved.append(mask_path)

    overlay_path = os.path.join(args.output_dir, "localization_overlay.png")
    save_overlay(image_np, predicted_mask, overlay_path)
    saved.append(overlay_path)

    # Learned branch activations returned by the model.
    branch_features = outputs.get("branch_features", {})
    feature_titles = {
        "rgb": "ResNet RGB Feature Activation (mean |activation|)",
        "dct": "DCT Branch Feature Activation (mean |activation|)",
        "srm": "SRM Branch Feature Activation (mean |activation|)",
    }
    for key, title in feature_titles.items():
        if key not in branch_features:
            print(f"WARNING: outputs['branch_features']['{key}'] not found; skipping.")
            continue
        tensor = branch_features[key]
        feature_map = feature_activation_map(tensor)
        out_path = os.path.join(args.output_dir, f"{key}_feature_activation.png")
        save_heatmap(feature_map, title, out_path, cmap="viridis")
        saved.append(out_path)
        print(f"{key} feature tensor shape: {tuple(tensor.shape)}")

    # Fused representation returned by the model.
    fused = outputs.get("fused_features")
    if fused is not None:
        fused_map = feature_activation_map(fused)
        fused_path = os.path.join(args.output_dir, "fused_feature_activation.png")
        save_heatmap(
            fused_map,
            "Fused Feature Activation (mean |activation|)",
            fused_path,
            cmap="viridis",
        )
        saved.append(fused_path)
        print(f"Fused feature tensor shape: {tuple(fused.shape)}")
    else:
        print("WARNING: outputs['fused_features'] not found; skipping fused map.")

    # Optional raw DCT coefficients and SRM residuals, if methods are exposed.
    extractors = [
        ("dct_branch", "extract_dct", "raw_dct_coefficient_summary.png",
         "Raw DCT Coefficient Magnitude Summary", "magma"),
        ("srm_branch", "extract_residuals", "raw_srm_residual_summary.png",
         "Raw SRM Residual Magnitude Summary", "gray"),
    ]
    for branch_name, method_name, filename, title, cmap in extractors:
        branch = getattr(model, branch_name, None)
        method = getattr(branch, method_name, None) if branch is not None else None
        if method is None:
            print(f"NOTE: {branch_name}.{method_name} not exposed; skipping raw visualization.")
            continue
        try:
            with torch.inference_mode():
                raw_tensor = method(raw_x)
            raw_map = raw_intermediate_map(raw_tensor)
            out_path = os.path.join(args.output_dir, filename)
            save_heatmap(raw_map, title, out_path, cmap=cmap)
            saved.append(out_path)
            print(f"{branch_name} raw tensor shape: {tuple(raw_tensor.shape)}")
        except Exception as exc:
            print(f"WARNING: Could not generate {title}: {exc}")

    print("\nInference summary")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Maximum probability: {float(prob_map.max()):.6f}")
    print(f"Mean probability: {float(prob_map.mean()):.6f}")
    print(f"Predicted positive pixels: {int(predicted_mask.sum()):,} / {predicted_mask.size:,}")
    print(f"Decision threshold: probability >= {THRESHOLD:.2f}")
    print(f"\nSaved {len(saved)} files to: {os.path.abspath(args.output_dir)}")
    for path in saved:
        print(f"  {path}")
    print(
        "\nNote: feature maps are normalized summaries for display. "
        "They are not ground-truth masks or independent tampering decisions."
    )


if __name__ == "__main__":
    main()
