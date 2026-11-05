"""Controlled Learning Comparison: Full Backprop (A) vs Partially Frozen RGB (B).

Trains two independently-initialized TamperDocModel instances for 3 epochs each
on the same 256 training samples and evaluates on the same 64 validation samples
from configs/dev_subset_10k.json.

Configuration A -- Full Backprop
  All ResNet-18 layers trainable; DCT, SRM, Fusion, Decoder trainable.

Configuration B -- Partially Frozen
  ResNet-18 conv1/stem + layer1 + layer2 frozen.
  ResNet-18 layer3 + layer4 trainable; DCT, SRM, Fusion, Decoder trainable.

Both models start from fresh ImageNet-pretrained weights (never share state).

Results saved to: results/freezing_learning_comparison.json
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import DocTamperLMDBDataset
from src.losses import CombinedLocalizationLoss
from src.metrics import LocalizationMetrics
from src.model import TamperDocModel

# Constants
CONFIG_PATH  = os.path.abspath("configs/dev_subset_10k.json")
DATASET_PATH = os.path.abspath("dataset/DocTamper/DocTamperV1-TrainingSet")
RESULTS_PATH = os.path.abspath("results/freezing_learning_comparison.json")

TRAIN_SAMPLES = 256
VAL_SAMPLES   = 64
BATCH_SIZE    = 4
NUM_EPOCHS    = 3
NUM_THREADS   = 8
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
SEED          = 42


def freeze_rgb_early_stages(model):
    """Freezes ResNet-18 conv1/stem, layer1, layer2 in-place."""
    rgb = model.rgb_branch
    for p in rgb.conv1.parameters():
        p.requires_grad = False
    for p in rgb.bn1.parameters():
        p.requires_grad = False
    for p in rgb.layer1.parameters():
        p.requires_grad = False
    for p in rgb.layer2.parameters():
        p.requires_grad = False


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_loaders(dataset, train_indices, val_indices, seed):
    """Creates reproducible train and val DataLoaders sharing a single dataset instance."""
    train_sub = Subset(dataset, train_indices)
    val_sub   = Subset(dataset, val_indices)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_sub,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=g,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, epoch_idx, config_name):
    model.train()
    running_loss  = 0.0
    running_bce   = 0.0
    running_dice  = 0.0
    nan_inf_count = 0
    n_batches     = 0
    epoch_start = time.time()

    for batch_idx, batch in enumerate(loader):
        images = batch["image"]
        raw    = batch["raw_image"]
        masks  = batch["mask"]

        optimizer.zero_grad()
        outputs = model(images, raw_x=raw)
        logits  = outputs["tamper_logits"]

        losses = criterion(logits, masks)
        loss = losses["loss"]

        if not torch.isfinite(loss):
            nan_inf_count += 1
            print(f"  [WARN] {config_name} epoch {epoch_idx+1} batch {batch_idx+1}: "
                  f"non-finite loss {loss.item():.6f}")
            continue

        loss.backward()
        optimizer.step()

        running_loss += losses["loss"].item()
        running_bce  += losses["bce_loss"].item()
        running_dice += losses["dice_loss"].item()
        n_batches    += 1

        if (batch_idx + 1) % 16 == 0 or (batch_idx + 1) == len(loader):
            print(f"    batch {batch_idx+1:02d}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} (BCE: {losses['bce_loss'].item():.4f}, "
                  f"Dice: {losses['dice_loss'].item():.4f})", flush=True)

    elapsed = time.time() - epoch_start
    denom = max(n_batches, 1)
    samples_processed = n_batches * BATCH_SIZE
    throughput = samples_processed / max(elapsed, 1e-6)

    return {
        "train_loss":         running_loss / denom,
        "train_bce_loss":     running_bce  / denom,
        "train_dice_loss":    running_dice / denom,
        "train_time_s":       round(elapsed, 2),
        "train_samples_sec":  round(throughput, 4),
        "nan_inf_batches":    nan_inf_count,
    }


def validate(model, loader, criterion):
    model.eval()
    running_loss  = 0.0
    running_bce   = 0.0
    running_dice  = 0.0
    n_batches     = 0
    meter = LocalizationMetrics(threshold=0.5)
    val_start = time.time()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"]
            raw    = batch["raw_image"]
            masks  = batch["mask"]

            outputs = model(images, raw_x=raw)
            logits  = outputs["tamper_logits"]

            losses = criterion(logits, masks)

            if torch.isfinite(losses["loss"]):
                running_loss += losses["loss"].item()
                running_bce  += losses["bce_loss"].item()
                running_dice += losses["dice_loss"].item()
                n_batches    += 1

            meter.update(logits, masks)

    elapsed = time.time() - val_start
    denom   = max(n_batches, 1)
    metrics = meter.compute()

    return {
        "val_loss":           running_loss / denom,
        "val_bce_loss":       running_bce  / denom,
        "val_dice_loss":      running_dice / denom,
        "val_precision":      metrics["precision"],
        "val_recall":         metrics["recall"],
        "val_f1":             metrics["f1"],
        "val_iou":            metrics["iou"],
        "val_pixel_accuracy": metrics["pixel_accuracy"],
        "val_time_s":         round(elapsed, 2),
    }


def run_config(name, freeze, dataset, train_indices, val_indices):
    print(f"\n{'='*70}")
    print(f"  CONFIGURATION {name} -- {'PARTIALLY FROZEN RGB' if freeze else 'FULL RGB BACKPROP'}")
    print(f"{'='*70}")

    # Create fresh model instance with ImageNet pretrained weights
    model = TamperDocModel(
        use_rgb=True,
        use_dct=True,
        use_srm=True,
        pretrained_rgb=True,
        branch_channels=128,
        fused_channels=128,
        target_size=(512, 512),
    )

    if freeze:
        freeze_rgb_early_stages(model)

    trainable_params = count_trainable(model)
    total_params     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({trainable_params/total_params*100:.2f}% trainable)")

    criterion = CombinedLocalizationLoss(lambda_dice=1.0, pos_weight=None)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Initialize reproducible dataloaders with seed for this configuration
    train_loader, val_loader = create_loaders(dataset, train_indices, val_indices, seed=SEED)

    epoch_results = []
    config_start  = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        print(f"\n  [Epoch {epoch+1}/{NUM_EPOCHS} - Config {name}]", flush=True)

        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, epoch, name)
        val_stats   = validate(model, val_loader, criterion)

        elapsed = time.time() - epoch_start
        print(f"    Train: Loss={train_stats['train_loss']:.4f} (BCE={train_stats['train_bce_loss']:.4f}, Dice={train_stats['train_dice_loss']:.4f}) | Time={train_stats['train_time_s']:.1f}s ({train_stats['train_samples_sec']:.2f} samples/s)")
        print(f"    Val:   Loss={val_stats['val_loss']:.4f} (BCE={val_stats['val_bce_loss']:.4f}, Dice={val_stats['val_dice_loss']:.4f})")
        print(f"    Val Metrics: F1={val_stats['val_f1']:.4f} | IoU={val_stats['val_iou']:.4f} | Precision={val_stats['val_precision']:.4f} | Recall={val_stats['val_recall']:.4f} | PixelAcc={val_stats['val_pixel_accuracy']:.4f}")
        print(f"    Epoch Wall Time: {elapsed:.1f}s | NaN/Inf: {train_stats['nan_inf_batches']}")

        row = {
            "epoch": epoch + 1,
            "epoch_wall_clock_time_s": round(elapsed, 2),
        }
        row.update(train_stats)
        row.update(val_stats)
        epoch_results.append(row)

    total_time = time.time() - config_start
    print(f"\n  Config {name} Total Time: {total_time:.1f}s ({total_time/60:.2f} min)")

    print(f"\n--- EPOCH SUMMARY FOR CONFIGURATION {name} ---")
    for r in epoch_results:
        print(f"  Epoch {r['epoch']}: TrainLoss={r['train_loss']:.4f}, ValLoss={r['val_loss']:.4f}, ValF1={r['val_f1']:.4f}, ValIoU={r['val_iou']:.4f}, Time={r['epoch_wall_clock_time_s']:.1f}s")

    return {
        "configuration":    name,
        "description":      "Partially frozen (stem+layer1+layer2 frozen)" if freeze else "Full backprop (all ResNet-18 trainable)",
        "freeze_rgb_early": freeze,
        "trainable_params": trainable_params,
        "total_params":     total_params,
        "total_time_s":     round(total_time, 2),
        "total_time_min":   round(total_time / 60, 2),
        "epochs":           epoch_results,
    }


def print_comparison_table(result_a, result_b):
    metrics = [
        ("train_loss",         "Train Loss"),
        ("train_bce_loss",     "Train BCE Loss"),
        ("train_dice_loss",    "Train Dice Loss"),
        ("train_time_s",       "Train Time (s)"),
        ("train_samples_sec",  "Train Speed (samp/s)"),
        ("val_loss",           "Val Loss"),
        ("val_bce_loss",       "Val BCE Loss"),
        ("val_dice_loss",      "Val Dice Loss"),
        ("val_f1",             "Val F1 / Dice"),
        ("val_iou",            "Val IoU"),
        ("val_precision",      "Val Precision"),
        ("val_recall",         "Val Recall"),
        ("val_pixel_accuracy", "Val Pixel Accuracy"),
        ("val_time_s",         "Val Time (s)"),
        ("epoch_wall_clock_time_s", "Epoch Total Time (s)"),
    ]

    print(f"\n{'='*84}")
    print("  CONTROLLED LEARNING COMPARISON: CONFIGURATION A vs CONFIGURATION B")
    print(f"{'='*84}")
    header = f"{'Metric':<26} | {'A (Ep 1)':>9} {'A (Ep 2)':>9} {'A (Ep 3)':>9} | {'B (Ep 1)':>9} {'B (Ep 2)':>9} {'B (Ep 3)':>9}"
    print(header)
    print("-" * 84)

    for key, label in metrics:
        row_a = [e[key] for e in result_a["epochs"]]
        row_b = [e[key] for e in result_b["epochs"]]
        line = f"{label:<26} |"
        for v in row_a:
            line += f" {v:>9.4f}" if isinstance(v, float) else f" {v:>9}"
        line += " |"
        for v in row_b:
            line += f" {v:>9.4f}" if isinstance(v, float) else f" {v:>9}"
        print(line)

    print("-" * 84)
    print(f"{'Trainable Params':<26} | {result_a['trainable_params']:>29,} | {result_b['trainable_params']:>29,}")
    print(f"{'Total Time (s)':<26} | {result_a['total_time_s']:>29.1f} | {result_b['total_time_s']:>29.1f}")
    print(f"{'Total Time (min)':<26} | {result_a['total_time_min']:>29.2f} | {result_b['total_time_min']:>29.2f}")
    print(f"{'='*84}\n")


def main():
    print("STARTING CONTROLLED FREEZING LEARNING COMPARISON", flush=True)

    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(SEED)

    print(f"PyTorch Version: {torch.__version__} | Active CPU Threads: {NUM_THREADS}")
    print(f"Frozen Split Config: {CONFIG_PATH}")
    print(f"Dataset LMDB Path:   {DATASET_PATH}")
    print(f"Results Save Path:   {RESULTS_PATH}")

    # Load frozen split
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_indices = cfg["train_indices"][:TRAIN_SAMPLES]
    val_indices   = cfg["val_indices"][:VAL_SAMPLES]

    print(f"\nDataset Selection:")
    print(f"  Training Samples:   {len(train_indices)} (first {TRAIN_SAMPLES} of {cfg['train_size']})")
    print(f"  Validation Samples: {len(val_indices)} (first {VAL_SAMPLES} of {cfg['val_size']})")
    print(f"  Batch Size:         {BATCH_SIZE}")
    print(f"  Training Epochs:    {NUM_EPOCHS}")
    print(f"  Learning Rate:      {LR}")
    print(f"  Weight Decay:       {WEIGHT_DECAY}")
    print(f"  Loss Function:      CombinedLocalizationLoss (lambda_dice=1.0, pos_weight=None)")
    print(f"  Optimizer:          AdamW")

    assert len(train_indices) == TRAIN_SAMPLES
    assert len(val_indices)   == VAL_SAMPLES

    # Instantiate single dataset instance
    dataset = DocTamperLMDBDataset(lmdb_dir=DATASET_PATH, is_train=False)

    # 1. Run Configuration A (Full Backprop)
    print("\n>>> Executing Configuration A (Full Backprop) <<<", flush=True)
    result_a = run_config("A", freeze=False, dataset=dataset, train_indices=train_indices, val_indices=val_indices)

    # 2. Run Configuration B (Partially Frozen)
    print("\n>>> Executing Configuration B (Partially Frozen RGB) <<<", flush=True)
    result_b = run_config("B", freeze=True, dataset=dataset, train_indices=train_indices, val_indices=val_indices)

    # 3. Save JSON results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    output = {
        "experiment":             "controlled_freezing_learning_comparison",
        "dataset_split_config":   CONFIG_PATH,
        "train_samples":          TRAIN_SAMPLES,
        "val_samples":            VAL_SAMPLES,
        "batch_size":             BATCH_SIZE,
        "num_epochs":             NUM_EPOCHS,
        "num_cpu_threads":        NUM_THREADS,
        "learning_rate":          LR,
        "weight_decay":           WEIGHT_DECAY,
        "loss_config":            {"lambda_dice": 1.0, "pos_weight": None},
        "optimizer":              "AdamW",
        "seed":                   SEED,
        "train_indices_used":     train_indices,
        "val_indices_used":       val_indices,
        "configuration_A":        result_a,
        "configuration_B":        result_b,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults successfully written to: {RESULTS_PATH}", flush=True)

    # 4. Print Side-by-Side Comparison
    print_comparison_table(result_a, result_b)

    print("CONTROLLED FREEZING LEARNING COMPARISON COMPLETE", flush=True)


if __name__ == "__main__":
    main()
