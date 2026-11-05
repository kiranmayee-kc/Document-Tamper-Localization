"""CPU Inference Performance Benchmark on Real DocTamper Test Samples (num_workers=0).

Evaluates inference throughput across:
- Batch sizes: 4, 8, 16, 32
- PyTorch thread counts: 4, 8, 12
- DataLoader workers: 0 (strictly single-process)
- Baseline vs. Optimized inference pipelines

Uses 64 real samples from DocTamperV1-Testingset.
Strictly read-only; does not modify model weights, dataset, or evaluation rules.
"""

import ctypes
import ctypes.wintypes
import gc
import json
import os
import sys
import time
from typing import Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import DocTamperLMDBDataset
from src.metrics import LocalizationMetrics
from src.model import TamperDocModel

CHECKPOINT_PATH = os.path.abspath("checkpoints/best_model.pt")
TESTSET_PATH    = os.path.abspath("dataset/DocTamper/DocTamperV1-Testingset")
RESULTS_PATH    = os.path.abspath("results/inference_benchmark.json")
BENCHMARK_SAMPLES = 64


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_process_ram_mb() -> float:
    """Returns current process RAM Working Set in Megabytes."""
    try:
        pmc = ProcessMemoryCounters()
        pmc.cb = ctypes.sizeof(ProcessMemoryCounters)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(pmc),
            pmc.cb,
        )
        return pmc.WorkingSetSize / (1024 * 1024)
    except Exception:
        return 0.0


def load_model(checkpoint_path: str) -> TamperDocModel:
    """Loads trained TamperDoc model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
    return model


@torch.no_grad()
def run_baseline_inference(model: TamperDocModel, dataloader: DataLoader) -> Tuple[Dict, float, float]:
    """Runs baseline inference using torch.no_grad() and sigmoid thresholding."""
    meter = LocalizationMetrics(threshold=0.5)
    ram_start = get_process_ram_mb()
    start_time = time.perf_counter()

    for batch in dataloader:
        images     = batch["image"]
        raw_images = batch["raw_image"]
        masks      = batch["mask"]

        outputs = model(images, raw_x=raw_images)
        logits  = outputs["tamper_logits"]

        meter.update(logits, masks)

    elapsed = time.perf_counter() - start_time
    ram_end = get_process_ram_mb()
    metrics = meter.compute()
    return metrics, elapsed, max(ram_start, ram_end)


def run_optimized_inference(model: TamperDocModel, dataloader: DataLoader) -> Tuple[Dict, float, float]:
    """Runs optimized inference using torch.inference_mode() and logit sign thresholding."""
    ram_start = get_process_ram_mb()
    start_time = time.perf_counter()

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    sample_count = 0
    eps = 1e-7

    with torch.inference_mode():
        for batch in dataloader:
            images     = batch["image"]
            raw_images = batch["raw_image"]
            masks      = batch["mask"]

            outputs = model(images, raw_x=raw_images)
            logits  = outputs["tamper_logits"]

            # Equivalent to sigmoid(logits) >= 0.5 without computing transcendentals
            preds = logits >= 0.0
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

    elapsed = time.perf_counter() - start_time
    ram_end = get_process_ram_mb()

    precision = total_tp / (total_tp + total_fp + eps)
    recall    = total_tp / (total_tp + total_fn + eps)
    f1        = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + eps)
    iou       = total_tp / (total_tp + total_fp + total_fn + eps)
    pixel_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "pixel_accuracy": float(pixel_acc),
        "samples_evaluated": sample_count,
    }
    return metrics, elapsed, max(ram_start, ram_end)


def main():
    print("=" * 80)
    print(f"TAMPERDOC CPU INFERENCE BENCHMARK ({BENCHMARK_SAMPLES} REAL TEST SAMPLES, num_workers=0)")
    print("=" * 80)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Dataset:    {TESTSET_PATH}")

    model = load_model(CHECKPOINT_PATH)
    raw_dataset = DocTamperLMDBDataset(lmdb_dir=TESTSET_PATH, is_train=False)
    test_subset = Subset(raw_dataset, list(range(BENCHMARK_SAMPLES)))

    batch_sizes   = [4, 8, 16, 32]
    thread_counts = [4, 8, 12]

    results_matrix = []

    # 1. Warm-up run
    torch.set_num_threads(8)
    warmup_loader = DataLoader(test_subset, batch_size=4, shuffle=False, num_workers=0)
    print("\nWarming up PyTorch CPU kernels (1 batch)...", flush=True)
    with torch.no_grad():
        w_batch = next(iter(warmup_loader))
        _ = model(w_batch["image"], raw_x=w_batch["raw_image"])
    print("Warm-up complete.\n", flush=True)

    # 2. Benchmark Baseline vs Optimized across matrix
    print("Running Systematic Performance Matrix (num_workers=0)...", flush=True)

    for threads in thread_counts:
        torch.set_num_threads(threads)
        for bs in batch_sizes:
            try:
                loader = DataLoader(
                    test_subset,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )

                # A. Run Baseline Pipeline
                gc.collect()
                base_metrics, base_elapsed, base_ram = run_baseline_inference(model, loader)
                base_throughput = BENCHMARK_SAMPLES / base_elapsed

                # B. Run Optimized Pipeline
                gc.collect()
                opt_metrics, opt_elapsed, opt_ram = run_optimized_inference(model, loader)
                opt_throughput = BENCHMARK_SAMPLES / opt_elapsed

                # C. Check mathematical equivalence
                max_diff = max(
                    abs(opt_metrics["f1"] - base_metrics["f1"]),
                    abs(opt_metrics["iou"] - base_metrics["iou"]),
                    abs(opt_metrics["precision"] - base_metrics["precision"]),
                    abs(opt_metrics["recall"] - base_metrics["recall"]),
                    abs(opt_metrics["pixel_accuracy"] - base_metrics["pixel_accuracy"]),
                )
                eq_pass = (max_diff < 1e-6)

                record = {
                    "threads": threads,
                    "batch_size": bs,
                    "num_workers": 0,
                    "base_throughput": round(base_throughput, 3),
                    "base_elapsed_s": round(base_elapsed, 2),
                    "base_ram_mb": round(base_ram, 1),
                    "opt_throughput": round(opt_throughput, 3),
                    "opt_elapsed_s": round(opt_elapsed, 2),
                    "opt_ram_mb": round(opt_ram, 1),
                    "speedup_ratio": round(opt_throughput / base_throughput, 3),
                    "max_metric_diff": float(max_diff),
                    "metrics_equal": eq_pass,
                    "f1": opt_metrics["f1"],
                    "iou": opt_metrics["iou"],
                    "status": "PASS",
                }
                results_matrix.append(record)

                print(f"  Threads: {threads:2d} | BS: {bs:2d} | Base: {base_throughput:5.2f} samp/s ({base_elapsed:5.2f}s) | "
                      f"Opt: {opt_throughput:5.2f} samp/s ({opt_elapsed:5.2f}s, {opt_ram:6.1f}MB) | Speedup: {record['speedup_ratio']:.2f}x | Equiv: {eq_pass}", flush=True)

            except Exception as e:
                print(f"  [FAIL] Threads: {threads} | BS: {bs} | Error: {e}", flush=True)
                results_matrix.append({
                    "threads": threads,
                    "batch_size": bs,
                    "num_workers": 0,
                    "status": f"FAILED: {e}",
                })

    # 3. Save benchmark JSON
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "tamperdoc_cpu_inference_matrix_singleprocess",
            "samples": BENCHMARK_SAMPLES,
            "checkpoint": CHECKPOINT_PATH,
            "results": results_matrix,
        }, f, indent=2)

    # 4. Print Summary Table
    print("\n" + "=" * 95)
    print(f"{'Threads':<8} {'BatchSize':<10} {'Base Thpt':<12} {'Opt Thpt':<12} {'Speedup':<10} {'Opt Time(s)':<13} {'RAM (MB)':<10} {'Equivalence':<12}")
    print("-" * 95)
    for r in results_matrix:
        if r["status"] == "PASS":
            equiv_str = str(r["metrics_equal"])
            print(f"{r['threads']:<8} {r['batch_size']:<10} {r['base_throughput']:<5.2f} samp/s  {r['opt_throughput']:<5.2f} samp/s  {r['speedup_ratio']:<4.2f}x      {r['opt_elapsed_s']:<13.2f} {r['opt_ram_mb']:<10.1f} {equiv_str:<12}")
    print("=" * 95)


if __name__ == "__main__":
    main()
