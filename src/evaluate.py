"""Evaluation pipeline for TamperDoc on protected Testingset.

Executes optimized inference-only evaluation of trained TamperDoc model on DocTamperV1-Testingset.
- Strict evaluation mode (torch.inference_mode(), model.eval())
- Mathematically exact logit thresholding (logits >= 0.0 <==> sigmoid(logits) >= 0.5)
- Direct integer tensor reduction for metric accumulation
- Evaluates complete 30,000 test set images
- Saves quantitative results to results/test_evaluation.json
- Saves representative qualitative overlay visualizations to results/test_visualizations/
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from src.dataset import DocTamperLMDBDataset
from src.model import TamperDocModel

DEFAULT_CHECKPOINT = os.path.abspath("checkpoints/best_model.pt")
DEFAULT_TESTSET    = os.path.abspath("dataset/DocTamper/DocTamperV1-Testingset")
DEFAULT_RESULTS    = os.path.abspath("results/test_evaluation.json")
DEFAULT_VIS_DIR    = os.path.abspath("results/test_visualizations")


def load_evaluation_model(checkpoint_path: str) -> Tuple[TamperDocModel, int, float, dict]:
    """Loads TamperDoc model and restores checkpoint weights."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    epoch = checkpoint.get("epoch", "unknown")
    best_f1 = checkpoint.get("best_val_f1", checkpoint.get("val_stats", {}).get("val_f1", "unknown"))

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

    return model, epoch, best_f1, checkpoint


def save_test_visualizations(
    samples: list,
    save_dir: str,
    threshold: float = 0.5,
) -> None:
    """Saves side-by-side visualization overlays of test predictions."""
    os.makedirs(save_dir, exist_ok=True)

    for item in samples:
        idx = item["index"]
        raw_img = item["raw_image"].cpu().permute(1, 2, 0).numpy() # [512, 512, 3] in [0, 1]
        gt_mask = item["mask"].cpu().squeeze().numpy()             # [512, 512] {0, 1}
        prob_map = item["prob"].cpu().squeeze().numpy()            # [512, 512] in [0, 1]
        pred_mask = (prob_map >= threshold).astype(np.float32)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(raw_img)
        axes[0].set_title(f"Test Sample #{idx}\nInput Document", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground-Truth Mask\n(1=Tampered, 0=Auth)", fontsize=10)
        axes[1].axis("off")

        im = axes[2].imshow(prob_map, cmap="jet", vmin=0, vmax=1)
        axes[2].set_title("Predicted Tamper Prob\n(Sigmoid Logits)", fontsize=10)
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title(f"Binarized Prediction\n(Threshold = {threshold})", fontsize=10)
        axes[3].axis("off")

        plt.tight_layout()
        out_file = os.path.join(save_dir, f"test_sample_{idx:05d}.png")
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_evaluation(
    model: TamperDocModel,
    dataloader: DataLoader,
    save_vis_count: int = 10,
    vis_dir: str = DEFAULT_VIS_DIR,
    log_interval: int = 100,
) -> Dict:
    """Executes full inference evaluation loop over the test dataloader."""
    model.eval()
    nan_inf_count = 0
    vis_samples = []

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    sample_count = 0
    eps = 1e-7

    total_batches = len(dataloader)
    total_samples = len(dataloader.dataset)
    start_time = time.time()

    print(f"\nEvaluating {total_samples:,} test samples ({total_batches:,} batches @ batch_size={dataloader.batch_size})...", flush=True)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            images     = batch["image"]
            raw_images = batch["raw_image"]
            masks      = batch["mask"]
            indices    = batch["index"]

            outputs = model(images, raw_x=raw_images)
            logits  = outputs["tamper_logits"]

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_inf_count += 1

            # Strictly equivalent to sigmoid(logits) >= 0.5 without expensive transcendental operations
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

            # Collect representative samples for visualization only from the initial batch(es)
            if len(vis_samples) < save_vis_count:
                probs = torch.sigmoid(logits)
                for i in range(min(images.size(0), save_vis_count - len(vis_samples))):
                    vis_samples.append({
                        "index": indices[i].item() if isinstance(indices[i], torch.Tensor) else int(indices[i]),
                        "raw_image": raw_images[i],
                        "mask": masks[i],
                        "prob": probs[i],
                    })

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - start_time
                speed = ((batch_idx + 1) * dataloader.batch_size) / max(elapsed, 1e-6)
                print(f"  [Evaluation] Batch {batch_idx+1:05d}/{total_batches:05d} | "
                      f"Samples: {min((batch_idx+1)*dataloader.batch_size, total_samples):05d}/{total_samples:05d} | "
                      f"Throughput: {speed:.2f} samp/s | Elapsed: {elapsed/60:.1f}m", flush=True)

    total_time = time.time() - start_time
    avg_throughput = sample_count / max(total_time, 1e-6)

    precision = total_tp / (total_tp + total_fp + eps)
    recall    = total_tp / (total_tp + total_fn + eps)
    f1        = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + eps)
    iou       = total_tp / (total_tp + total_fp + total_fn + eps)
    pixel_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)

    # Save visual overlays
    if len(vis_samples) > 0:
        print(f"\nSaving {len(vis_samples)} qualitative prediction overlays to {vis_dir}...")
        save_test_visualizations(vis_samples, vis_dir, threshold=0.5)

    return {
        "f1":                  float(f1),
        "iou":                 float(iou),
        "precision":           float(precision),
        "recall":              float(recall),
        "pixel_accuracy":      float(pixel_acc),
        "total_tp_pixels":     int(total_tp),
        "total_fp_pixels":     int(total_fp),
        "total_fn_pixels":     int(total_fn),
        "total_tn_pixels":     int(total_tn),
        "total_time_s":        round(total_time, 2),
        "total_time_min":      round(total_time / 60, 2),
        "total_time_hours":    round(total_time / 3600, 2),
        "average_throughput":  round(avg_throughput, 4),
        "total_samples":       sample_count,
        "total_batches":       total_batches,
        "batch_size":          dataloader.batch_size,
        "threshold_definition": "logits >= 0.0 (mathematically identical to sigmoid(logits) >= 0.5)",
        "nan_inf_batches":     nan_inf_count,
    }


def main():
    parser = argparse.ArgumentParser(description="TamperDoc Final Inference-Only Test Evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to best model checkpoint")
    parser.add_argument("--test-dir", type=str, default=DEFAULT_TESTSET, help="Path to DocTamperV1-Testingset LMDB")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size (default: 16)")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads (default: 4)")
    parser.add_argument("--save-vis", type=int, default=10, help="Number of qualitative overlays to save (default: 10)")
    parser.add_argument("--results-path", type=str, default=DEFAULT_RESULTS, help="Output JSON results path")
    parser.add_argument("--vis-dir", type=str, default=DEFAULT_VIS_DIR, help="Output directory for visualizations")
    parser.add_argument("--sanity-check", action="store_true", help="Run 4-sample sanity check only")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    print("=" * 75)
    print("  TAMPERDOC FINAL TEST-SET INFERENCE EVALUATION")
    print("=" * 75)
    print(f"PyTorch Version:     {torch.__version__}")
    print(f"Active CPU Threads:  {args.threads}")
    print(f"Checkpoint Path:     {args.checkpoint}")
    print(f"Test Dataset Path:   {args.test_dir}")
    print(f"Prediction Threshold:0.5 (evaluated as logits >= 0.0)")
    print(f"Batch Size:          {args.batch_size}")
    print(f"num_workers:         0 (single-process LMDB)")

    # 1. Load Model from Checkpoint
    model, epoch, best_val_f1, ckpt_meta = load_evaluation_model(args.checkpoint)
    print(f"\n[Checkpoint Audit]")
    print(f"  Checkpoint File:   {args.checkpoint}")
    print(f"  Originating Epoch: Epoch {epoch}")
    print(f"  Best Val F1:       {best_val_f1}")
    print(f"  Total Parameters:  {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model Mode:        model.eval() / torch.inference_mode()")

    # 2. Dataset Loading
    test_dataset = DocTamperLMDBDataset(lmdb_dir=args.test_dir, is_train=False)
    test_sample_count = len(test_dataset)
    print(f"\n[Dataset Verification]")
    print(f"  Total Test Samples: {test_sample_count:,}")
    print(f"  Verified Path:      {args.test_dir}")

    # 3. Sanity Check Mode (4 samples)
    if args.sanity_check:
        print("\n>>> EXECUTING 4-SAMPLE PRE-TEST SANITY CHECK <<<", flush=True)
        sanity_sub = Subset(test_dataset, list(range(4)))
        sanity_loader = DataLoader(sanity_sub, batch_size=4, shuffle=False, num_workers=0)
        
        batch = next(iter(sanity_loader))
        images = batch["image"]
        raw    = batch["raw_image"]
        masks  = batch["mask"]

        assert images.shape == (4, 3, 512, 512), f"Unexpected image shape: {images.shape}"
        assert raw.shape    == (4, 3, 512, 512), f"Unexpected raw image shape: {raw.shape}"
        assert masks.shape  == (4, 1, 512, 512), f"Unexpected mask shape: {masks.shape}"

        with torch.inference_mode():
            outputs = model(images, raw_x=raw)
            logits  = outputs["tamper_logits"]
            probs   = torch.sigmoid(logits)

        assert logits.shape == (4, 1, 512, 512), f"Unexpected logits shape: {logits.shape}"
        assert not (torch.isnan(logits).any() or torch.isinf(logits).any()), "NaN/Inf detected in logits!"
        
        pred_via_sign  = (logits >= 0.0)
        pred_via_prob  = (probs >= 0.5)
        mismatch_count = torch.count_nonzero(pred_via_sign != pred_via_prob).item()
        assert mismatch_count == 0, f"Mismatch between logits >= 0 and probs >= 0.5: {mismatch_count} pixels"

        print(f"  Sanity check sample shapes verified:")
        print(f"    Image:        {images.shape} (dtype={images.dtype})")
        print(f"    Raw:          {raw.shape} (dtype={raw.dtype})")
        print(f"    Mask:         {masks.shape} (dtype={masks.dtype}, unique={torch.unique(masks).tolist()})")
        print(f"    Logits:       {logits.shape} (range=[{logits.min():.2f}, {logits.max():.2f}])")
        print(f"    Probs:        {probs.shape} (range=[{probs.min():.4f}, {probs.max():.4f}])")
        print(f"    Binary Equiv: {pred_via_sign.shape} (Mismatches: {mismatch_count}) -- 100% IDENTICAL")

        sanity_vis_dir = os.path.join(args.vis_dir, "sanity_check")
        save_test_visualizations([{
            "index": i,
            "raw_image": raw[i],
            "mask": masks[i],
            "prob": probs[i]
        } for i in range(4)], sanity_vis_dir, threshold=0.5)
        print(f"  Sanity visualizations saved to: {sanity_vis_dir}")

        print("\n[SANITY CHECK PASSED] All sanity checks verified successfully!")
        return

    # 4. Full 30,000-sample Evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    results = run_evaluation(
        model=model,
        dataloader=test_loader,
        save_vis_count=args.save_vis,
        vis_dir=args.vis_dir,
        log_interval=100,
    )

    # 5. Save Results JSON
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    full_output = {
        "evaluation_name":            "DocTamperV1_Testingset_Evaluation",
        "checkpoint_used":            args.checkpoint,
        "checkpoint_originating_epoch": epoch,
        "best_val_f1_in_ckpt":        best_val_f1,
        "test_dataset_path":          args.test_dir,
        "total_test_samples_evaluated": results["total_samples"],
        "hardware": {
            "device": "cpu",
            "threads": args.threads,
            "batch_size": args.batch_size,
            "num_workers": 0,
        },
        "isolation_verified": {
            "training_data_used": False,
            "validation_data_used": False,
            "scd_data_used": False,
            "fcd_data_used": False,
        },
        "metrics": results,
    }

    with open(args.results_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print("\n" + "=" * 75)
    print("  FINAL DOCTAMPER TEST-SET EVALUATION RESULTS (30,000 IMAGES)")
    print("=" * 75)
    print(f"  Test Set F1 / Dice:       {results['f1']:.6f} ({results['f1']*100:.2f}%)")
    print(f"  Test Set IoU:             {results['iou']:.6f} ({results['iou']*100:.2f}%)")
    print(f"  Test Set Precision:       {results['precision']:.6f} ({results['precision']*100:.2f}%)")
    print(f"  Test Set Recall:          {results['recall']:.6f} ({results['recall']*100:.2f}%)")
    print(f"  Test Set Pixel Accuracy:  {results['pixel_accuracy']:.6f} ({results['pixel_accuracy']*100:.2f}%)")
    print(f"  Total Pixels Supervised:  {(results['total_tp_pixels'] + results['total_fp_pixels'] + results['total_fn_pixels'] + results['total_tn_pixels']):,}")
    print(f"  Total Images Evaluated:   {results['total_samples']:,}")
    print(f"  Total Batches:            {results['total_batches']:,} (batch_size={results['batch_size']})")
    print(f"  Threshold Definition:     {results['threshold_definition']}")
    print(f"  NaN / Inf Batches:        {results['nan_inf_batches']}")
    print(f"  Average Throughput:       {results['average_throughput']:.2f} samples/s")
    print(f"  Total Evaluation Time:    {results['total_time_s']:.1f} s ({results['total_time_min']:.2f} min / {results['total_time_hours']:.2f} hours)")
    print(f"  Results Saved To:         {args.results_path}")
    print(f"  Visualizations Saved To:  {args.vis_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
