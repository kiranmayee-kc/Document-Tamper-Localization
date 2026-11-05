"""Final evaluation on a fixed 10,000-sample subset of DocTamperV1-Testingset.

Protocol:
  - Select exactly 10,000 unique indices from the 30,000-sample Testingset
    using random seed 42 (selection is model-independent).
  - Save selected indices + metadata to  configs/test_subset_10k.json
  - Sanity-check checkpoint, indices, and model output shape before running.
  - Inference: batch_size=16, threads=4, num_workers=0,
               torch.inference_mode(), logits >= 0.0
  - Accumulate pixel-level TP/FP/FN/TN across all samples.
  - Save metrics to  results/test_evaluation_10k.json
  - Save overlay visualizations to  results/test_visualizations/
  - Print concise final summary.

No training. No backpropagation. No synthetic data. No fabricated metrics.
"""

import json
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import DocTamperLMDBDataset
from src.model import TamperDocModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = os.path.abspath("checkpoints/best_model.pt")
TESTSET_PATH    = os.path.abspath("dataset/DocTamper/DocTamperV1-Testingset")
SUBSET_CFG_PATH = os.path.abspath("configs/test_subset_10k.json")
RESULTS_PATH    = os.path.abspath("results/test_evaluation_10k.json")
VIS_DIR         = os.path.abspath("results/test_visualizations")

NUM_SUBSET      = 10_000
SEED            = 42
BATCH_SIZE      = 16
THREADS         = 4
NUM_WORKERS     = 0
LOG_INTERVAL    = 10   # print progress every N batches (~160 samples / ~50 s)
NUM_VIS_SAMPLES = 20   # representative visualizations to save

EPS = 1e-7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_and_save_subset(full_size: int) -> list:
    """Select 10,000 unique indices with seed 42 and save to configs/."""
    rng = random.Random(SEED)
    indices = rng.sample(range(full_size), NUM_SUBSET)
    indices_sorted = sorted(indices)   # deterministic ordering for DataLoader

    os.makedirs(os.path.dirname(SUBSET_CFG_PATH), exist_ok=True)
    meta = {
        "description": "Fixed 10,000-sample evaluation subset of DocTamperV1-Testingset",
        "seed": SEED,
        "full_testset_size": full_size,
        "subset_size": NUM_SUBSET,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "indices": indices_sorted,
    }
    with open(SUBSET_CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {NUM_SUBSET} indices to: {SUBSET_CFG_PATH}")
    return indices_sorted


def load_model(checkpoint_path: str) -> tuple:
    """Load TamperDocModel from checkpoint, return (model, epoch, best_val_f1)."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model = TamperDocModel(
        use_rgb=True,
        use_dct=True,
        use_srm=True,
        pretrained_rgb=False,
        branch_channels=128,
        fused_channels=128,
        target_size=(512, 512),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    epoch      = ckpt.get("epoch", "unknown")
    best_val_f1 = ckpt.get("best_val_f1", None)
    if best_val_f1 is None:
        best_val_f1 = ckpt.get("val_stats", {}).get("val_f1", "unknown")

    return model, epoch, best_val_f1


def save_visualization(raw_img_uint8: np.ndarray,
                       gt_mask: np.ndarray,
                       pred_mask: np.ndarray,
                       prob_map: np.ndarray,
                       sample_index: int,
                       save_dir: str) -> None:
    """Save 4-panel (image / GT / prediction / probability) PNG."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    titles = ["Input Image", "Ground Truth", "Prediction (≥0.5)", "Probability Map"]

    # Panel 0: RGB image (uint8 → [0,1] float)
    axes[0].imshow(raw_img_uint8.transpose(1, 2, 0).astype(np.float32) / 255.0)

    # Panel 1: Ground-truth mask (binary)
    axes[1].imshow(gt_mask.squeeze(), cmap="gray", vmin=0, vmax=1)

    # Panel 2: Binary prediction
    axes[2].imshow(pred_mask.squeeze(), cmap="gray", vmin=0, vmax=1)

    # Panel 3: Probability heatmap
    im = axes[3].imshow(prob_map.squeeze(), cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Test Sample #{sample_index}", fontsize=11, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"sample_{sample_index:06d}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("TAMPERDOC — FINAL EVALUATION (10,000-SAMPLE FIXED TESTINGSET SUBSET)")
    print("=" * 80)
    print(f"Started at:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checkpoint:   {CHECKPOINT_PATH}")
    print(f"Dataset:      {TESTSET_PATH}")
    print(f"Subset size:  {NUM_SUBSET} samples (seed={SEED})")
    print(f"Batch size:   {BATCH_SIZE}")
    print(f"Threads:      {THREADS}")
    print(f"Workers:      {NUM_WORKERS}")
    print(f"Threshold:    logits >= 0.0  (== sigmoid >= 0.5)")
    print(f"Results:      {RESULTS_PATH}")
    print(f"Vis dir:      {VIS_DIR}")
    print("=" * 80)

    torch.set_num_threads(THREADS)

    # ------------------------------------------------------------------
    # 1. Load full Testingset to get true size, then select subset
    # ------------------------------------------------------------------
    print("\n[1/5] Loading dataset and selecting 10,000-sample subset ...")
    full_dataset = DocTamperLMDBDataset(lmdb_dir=TESTSET_PATH, is_train=False)
    full_size = len(full_dataset)
    print(f"  Full Testingset size: {full_size}")

    assert full_size >= NUM_SUBSET, \
        f"Testingset has only {full_size} samples, need {NUM_SUBSET}"

    subset_indices = select_and_save_subset(full_size)

    # ------------------------------------------------------------------
    # 2. Sanity checks
    # ------------------------------------------------------------------
    print("\n[2/5] Running sanity checks ...")

    assert len(subset_indices) == NUM_SUBSET, \
        f"Expected {NUM_SUBSET} indices, got {len(subset_indices)}"
    assert len(set(subset_indices)) == NUM_SUBSET, \
        "Indices are not unique!"
    assert all(0 <= idx < full_size for idx in subset_indices), \
        "Some indices are out of Testingset range!"
    print(f"  [OK] Indices: {NUM_SUBSET} unique, all in [0, {full_size})")

    # Load model
    print("  Loading checkpoint ...")
    model, ckpt_epoch, best_val_f1 = load_model(CHECKPOINT_PATH)
    print(f"  [OK] Checkpoint loaded | epoch={ckpt_epoch} | best_val_f1={best_val_f1}")

    # Build subset DataLoader
    eval_subset    = Subset(full_dataset, subset_indices)
    eval_dataloader = DataLoader(
        eval_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # Single-batch shape / NaN check
    print("  Running 1-batch shape/NaN sanity check ...")
    sanity_iter = iter(eval_dataloader)
    sanity_batch = next(sanity_iter)
    del sanity_iter  # release iterator immediately

    with torch.inference_mode():
        sanity_out = model(sanity_batch["image"], raw_x=sanity_batch["raw_image"])
        sanity_logits = sanity_out["tamper_logits"]

    expected_shape = (BATCH_SIZE, 1, 512, 512)
    assert tuple(sanity_logits.shape) == expected_shape, \
        f"Expected output shape {expected_shape}, got {tuple(sanity_logits.shape)}"
    assert not torch.isnan(sanity_logits).any(), "NaN values detected in logits!"
    assert not torch.isinf(sanity_logits).any(), "Inf values detected in logits!"
    print(f"  [OK] Output shape: {list(sanity_logits.shape)} | No NaN/Inf")

    # Rebuild DataLoader (fresh iterator without consuming any batches)
    eval_dataloader = DataLoader(
        eval_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    total_batches = len(eval_dataloader)
    print(f"  [OK] Sanity checks passed. Proceeding with {total_batches} batches.")

    # ------------------------------------------------------------------
    # 3. Full inference loop
    # ------------------------------------------------------------------
    print(f"\n[3/5] Running inference on {NUM_SUBSET} samples ...")
    print(f"      Progress updates every {LOG_INTERVAL} batches ({LOG_INTERVAL * BATCH_SIZE} samples)\n")

    # Metric accumulators
    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0
    sample_count = 0

    # For visualizations: collect every Nth sample across the full run
    vis_interval = max(1, total_batches // NUM_VIS_SAMPLES)
    vis_samples = []  # list of dicts for saving

    nan_inf_count = 0
    eval_start = time.perf_counter()
    last_log_time = eval_start

    with torch.inference_mode():
        for batch_idx, batch in enumerate(eval_dataloader):
            images     = batch["image"]      # [B,3,512,512] normalized
            raw_images = batch["raw_image"]  # [B,3,512,512] uint8 as float
            masks      = batch["mask"]       # [B,1,512,512] {0,1}

            # Forward pass
            outputs = model(images, raw_x=raw_images)
            logits  = outputs["tamper_logits"]  # [B,1,512,512]

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_inf_count += 1

            # Pixel-level binary prediction
            preds   = logits >= 0.0    # bool [B,1,512,512]
            targets = masks > 0.0      # bool [B,1,512,512]

            tp = int(torch.count_nonzero(preds & targets))
            fp = int(torch.count_nonzero(preds & (~targets)))
            fn = int(torch.count_nonzero((~preds) & targets))
            tn = int(torch.count_nonzero((~preds) & (~targets)))

            tp_total += tp
            fp_total += fp
            fn_total += fn
            tn_total += tn
            sample_count += masks.size(0)

            # Collect visualization sample once per vis_interval batches
            if batch_idx % vis_interval == 0 and len(vis_samples) < NUM_VIS_SAMPLES:
                b = 0  # first item in batch
                prob_map = torch.sigmoid(logits[b]).numpy()           # [1,512,512]
                pred_mask = (prob_map >= 0.5).astype(np.float32)
                vis_samples.append({
                    "index":       batch["index"][b].item() if isinstance(batch["index"][b], torch.Tensor) else int(subset_indices[batch_idx * BATCH_SIZE]),
                    "raw_image":   raw_images[b].numpy().astype(np.uint8) if raw_images[b].max() > 1.0 else (raw_images[b].numpy() * 255).astype(np.uint8),
                    "gt_mask":     masks[b].numpy(),
                    "pred_mask":   pred_mask,
                    "prob_map":    prob_map,
                })

            # Progress logging
            if (batch_idx + 1) % LOG_INTERVAL == 0 or batch_idx == total_batches - 1:
                now = time.perf_counter()
                elapsed = now - eval_start
                avg_speed = sample_count / max(elapsed, 1e-6)
                remaining = (NUM_SUBSET - sample_count) / max(avg_speed, 1e-6)
                pct = 100.0 * sample_count / NUM_SUBSET

                # Running metrics
                run_f1 = (2.0 * tp_total) / (2.0 * tp_total + fp_total + fn_total + EPS)
                run_iou = tp_total / (tp_total + fp_total + fn_total + EPS)

                print(
                    f"  Batch {batch_idx+1:4d}/{total_batches} | "
                    f"Samples {sample_count:6d}/{NUM_SUBSET} ({pct:5.1f}%) | "
                    f"Elapsed {elapsed/60:5.1f}min | "
                    f"ETA {remaining/60:5.1f}min | "
                    f"Speed {avg_speed:5.2f} samp/s | "
                    f"Running F1 {run_f1:.4f} | IoU {run_iou:.4f}",
                    flush=True,
                )

    total_eval_time = time.perf_counter() - eval_start
    avg_throughput  = sample_count / max(total_eval_time, 1e-6)

    # ------------------------------------------------------------------
    # 4. Compute final metrics
    # ------------------------------------------------------------------
    print(f"\n[4/5] Computing final metrics ...")

    precision  = tp_total / (tp_total + fp_total + EPS)
    recall     = tp_total / (tp_total + fn_total + EPS)
    f1         = (2.0 * tp_total) / (2.0 * tp_total + fp_total + fn_total + EPS)
    iou        = tp_total / (tp_total + fp_total + fn_total + EPS)
    pixel_acc  = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total + EPS)
    specificity = tn_total / (tn_total + fp_total + EPS)

    results = {
        "evaluation_id":         "test_evaluation_10k",
        "timestamp":             datetime.utcnow().isoformat() + "Z",
        "checkpoint":            CHECKPOINT_PATH,
        "checkpoint_epoch":      ckpt_epoch,
        "best_val_f1_from_ckpt": best_val_f1,
        "dataset":               TESTSET_PATH,
        "subset_config":         SUBSET_CFG_PATH,
        "full_testset_size":     full_size,
        "evaluated_samples":     sample_count,
        "subset_seed":           SEED,
        "batch_size":            BATCH_SIZE,
        "threads":               THREADS,
        "num_workers":           NUM_WORKERS,
        "threshold":             "logits >= 0.0 (== sigmoid >= 0.5)",
        "inference_mode":        True,
        "nan_inf_batches":       nan_inf_count,
        "metrics": {
            "f1_dice":     round(f1, 6),
            "iou":         round(iou, 6),
            "precision":   round(precision, 6),
            "recall":      round(recall, 6),
            "pixel_accuracy": round(pixel_acc, 6),
            "specificity": round(specificity, 6),
        },
        "pixel_counts": {
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "tn": tn_total,
        },
        "timing": {
            "total_seconds":    round(total_eval_time, 2),
            "total_minutes":    round(total_eval_time / 60, 2),
            "avg_throughput_sps": round(avg_throughput, 3),
        },
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {RESULTS_PATH}")

    # ------------------------------------------------------------------
    # 5. Save visualizations
    # ------------------------------------------------------------------
    print(f"\n[5/5] Saving {len(vis_samples)} visualizations to {VIS_DIR} ...")
    for vis in vis_samples:
        save_visualization(
            raw_img_uint8=vis["raw_image"],
            gt_mask=vis["gt_mask"],
            pred_mask=vis["pred_mask"],
            prob_map=vis["prob_map"],
            sample_index=vis["index"],
            save_dir=VIS_DIR,
        )
    print(f"  Saved {len(vis_samples)} overlay PNGs.")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  Checkpoint:           {CHECKPOINT_PATH}")
    print(f"  Checkpoint epoch:     {ckpt_epoch}  (best_val_f1={best_val_f1})")
    print(f"  Evaluated samples:    {sample_count} / {NUM_SUBSET}")
    print(f"  Threshold:            logits >= 0.0  (== sigmoid >= 0.5)")
    print(f"  NaN/Inf batches:      {nan_inf_count}")
    print("-" * 80)
    print(f"  F1 / Dice:            {f1:.6f}")
    print(f"  IoU:                  {iou:.6f}")
    print(f"  Precision:            {precision:.6f}")
    print(f"  Recall:               {recall:.6f}")
    print(f"  Pixel Accuracy:       {pixel_acc:.6f}")
    print(f"  Specificity:          {specificity:.6f}")
    print("-" * 80)
    print(f"  Total eval time:      {total_eval_time/60:.2f} min ({total_eval_time:.1f} s)")
    print(f"  Average throughput:   {avg_throughput:.2f} samples/sec")
    print(f"  Results saved:        {RESULTS_PATH}")
    print(f"  Visualizations saved: {VIS_DIR}/")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
