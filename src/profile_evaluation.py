"""Controlled Diagnostic Throughput & Timing Profiler (256 Real Test Samples).

Profiles the exact inference pipeline of src/evaluate.py on the first 256 samples of DocTamperV1-Testingset:
- Batch size: 16 (16 batches)
- PyTorch CPU threads: 4
- num_workers: 0
- torch.inference_mode()
- logit sign thresholding (logits >= 0.0)

Measures individual component durations:
1. Data loading time (LMDB read, PIL decode, tensor transform)
2. Model forward inference time
3. Metric accumulation time
4. End-to-end throughput (samples/sec)
"""

import os
import sys
import time
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import DocTamperLMDBDataset
from src.model import TamperDocModel

CHECKPOINT_PATH = os.path.abspath("checkpoints/best_model.pt")
TESTSET_PATH    = os.path.abspath("dataset/DocTamper/DocTamperV1-Testingset")
NUM_SAMPLES     = 256
BATCH_SIZE      = 16
THREADS         = 4


def main():
    print("=" * 80)
    print(f"CONTROLLED EVALUATION THROUGHPUT PROFILER ({NUM_SAMPLES} REAL TEST SAMPLES)")
    print("=" * 80)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Dataset:    {TESTSET_PATH}")
    print(f"Threads:    {THREADS}")
    print(f"Batch Size: {BATCH_SIZE} ({NUM_SAMPLES // BATCH_SIZE} batches)")
    print(f"Workers:    0 (single-process)")

    torch.set_num_threads(THREADS)

    # 1. Load Model
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = TamperDocModel(
        use_rgb=True,
        use_dct=True,
        use_srm=True,
        pretrained_rgb=False,
        branch_channels=128,
        fused_channels=128,
        target_size=(512, 512),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # 2. Dataset Subset (first 256 samples)
    raw_dataset = DocTamperLMDBDataset(lmdb_dir=TESTSET_PATH, is_train=False)
    test_subset = Subset(raw_dataset, list(range(NUM_SAMPLES)))
    dataloader  = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    sample_count = 0
    eps = 1e-7

    total_data_time = 0.0
    total_fwd_time  = 0.0
    total_metric_time = 0.0

    nan_inf_detected = False
    verified_shape = None

    print("\n--- Starting Detailed Batch-by-Batch Profiling ---", flush=True)
    overall_start = time.perf_counter()

    data_start = time.perf_counter()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            data_elapsed = time.perf_counter() - data_start
            total_data_time += data_elapsed

            images     = batch["image"]
            raw_images = batch["raw_image"]
            masks      = batch["mask"]

            # Model Forward Pass
            fwd_start = time.perf_counter()
            outputs = model(images, raw_x=raw_images)
            logits  = outputs["tamper_logits"]
            fwd_elapsed = time.perf_counter() - fwd_start
            total_fwd_time += fwd_elapsed

            if verified_shape is None:
                verified_shape = list(logits.shape)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_inf_detected = True

            # Metric Accumulation
            metric_start = time.perf_counter()
            preds   = logits >= 0.0
            targets = masks > 0.0

            tp = int(torch.count_nonzero(preds & targets))
            fp = int(torch.count_nonzero(preds & (~targets)))
            fn = int(torch.count_nonzero((~preds) & targets))
            tn = int(torch.count_nonzero((~preds) & (~targets)))

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            sample_count += masks.size(0)

            metric_elapsed = time.perf_counter() - metric_start
            total_metric_time += metric_elapsed

            batch_total = data_elapsed + fwd_elapsed + metric_elapsed
            batch_speed = BATCH_SIZE / max(batch_total, 1e-6)

            print(f"  Batch {batch_idx+1:02d}/{len(dataloader):02d} | "
                  f"Data: {data_elapsed*1000:6.1f}ms | "
                  f"Fwd: {fwd_elapsed*1000:6.1f}ms | "
                  f"Metric: {metric_elapsed*1000:5.1f}ms | "
                  f"Total: {batch_total*1000:6.1f}ms | "
                  f"Speed: {batch_speed:5.2f} samp/s", flush=True)

            data_start = time.perf_counter()

    overall_total = time.perf_counter() - overall_start
    avg_speed = NUM_SAMPLES / overall_total

    precision = total_tp / (total_tp + total_fp + eps)
    recall    = total_tp / (total_tp + total_fn + eps)
    f1        = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + eps)
    iou       = total_tp / (total_tp + total_fp + total_fn + eps)
    pixel_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)

    print("\n" + "=" * 80)
    print("PROFILING SUMMARY RESULTS (256 SAMPLES)")
    print("=" * 80)
    print(f"Total Samples Evaluated:      {sample_count}")
    print(f"Total Batches Evaluated:      {len(dataloader)} (batch_size={BATCH_SIZE})")
    print(f"Verified Model Output Shape:  {verified_shape} (Expected: [{BATCH_SIZE}, 1, 512, 512])")
    print(f"Any NaN / Inf Values:         {nan_inf_detected}")
    print(f"Gradients Enabled:            {torch.is_grad_enabled()} (False)")
    print(f"Requires Grad on Params:      {any(p.requires_grad for p in model.parameters())} (False)")
    print("-" * 80)
    print(f"Component Time Breakdown:")
    print(f"  Data Loading & Prep Time:   {total_data_time:6.2f} s ({total_data_time/overall_total*100:5.1f}%) | {total_data_time/NUM_SAMPLES*1000:5.1f} ms/sample")
    print(f"  Model Forward Pass Time:    {total_fwd_time:6.2f} s ({total_fwd_time/overall_total*100:5.1f}%) | {total_fwd_time/NUM_SAMPLES*1000:5.1f} ms/sample")
    print(f"  Metric Reduction Time:      {total_metric_time:6.2f} s ({total_metric_time/overall_total*100:5.1f}%) | {total_metric_time/NUM_SAMPLES*1000:5.1f} ms/sample")
    print(f"  Total Wall-Clock Time:      {overall_total:6.2f} s ({overall_total/60:5.2f} min)")
    print(f"  Measured Overall Throughput:{avg_speed:6.2f} samples/sec")
    print("-" * 80)
    print(f"256-Sample Intermediate Metrics (Diagnostic only; not full test metrics):")
    print(f"  F1 / Dice:  {f1:.6f}")
    print(f"  IoU:        {iou:.6f}")
    print(f"  Precision:  {precision:.6f}")
    print(f"  Recall:     {recall:.6f}")
    print(f"  Pixel Acc:  {pixel_acc:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
