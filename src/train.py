"""Main Development Training Pipeline for TamperDoc (Configuration B).

Configuration:
- Model: Multi-Branch TamperDocModel (RGB + DCT + SRM)
- Strategy: Configuration B (ResNet-18 stem + layer1 + layer2 frozen; layer3 + layer4 trainable)
- Dataset: DocTamperV1-TrainingSet via configs/dev_subset_10k.json (8,500 train, 1,500 val)
- Optimizer: AdamW(lr=1e-4, weight_decay=1e-4)
- Loss: CombinedLocalizationLoss (lambda_dice=1.0, pos_weight=None)
- Hardware: CPU (8 threads), Batch size = 4
- Checkpointing: Per-epoch checkpoints + best validation F1 checkpoint + resume support
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
from torch.utils.data import DataLoader, Subset

from src.dataset import DocTamperLMDBDataset
from src.losses import CombinedLocalizationLoss
from src.metrics import LocalizationMetrics
from src.model import TamperDocModel

# Default paths
CONFIG_PATH     = os.path.abspath("configs/dev_subset_10k.json")
DATASET_PATH    = os.path.abspath("dataset/DocTamper/DocTamperV1-TrainingSet")
CHECKPOINT_DIR  = os.path.abspath("checkpoints")
RESULTS_PATH    = os.path.abspath("results/training_log.json")


def freeze_rgb_early_stages(model: TamperDocModel) -> None:
    """Freezes ResNet-18 conv1/stem, layer1, layer2 in-place (Configuration B)."""
    rgb = model.rgb_branch
    for p in rgb.conv1.parameters():
        p.requires_grad = False
    for p in rgb.bn1.parameters():
        p.requires_grad = False
    for p in rgb.layer1.parameters():
        p.requires_grad = False
    for p in rgb.layer2.parameters():
        p.requires_grad = False


def verify_freeze_state(model: TamperDocModel) -> Dict[str, bool]:
    """Inspects and returns the trainability of each module."""
    rgb = model.rgb_branch
    stem_frozen = (
        all(not p.requires_grad for p in rgb.conv1.parameters()) and
        all(not p.requires_grad for p in rgb.bn1.parameters())
    )
    l1_frozen = all(not p.requires_grad for p in rgb.layer1.parameters())
    l2_frozen = all(not p.requires_grad for p in rgb.layer2.parameters())
    l3_trainable = any(p.requires_grad for p in rgb.layer3.parameters())
    l4_trainable = any(p.requires_grad for p in rgb.layer4.parameters())
    lat_trainable = all([
        any(p.requires_grad for p in rgb.lat_l1.parameters()),
        any(p.requires_grad for p in rgb.lat_l2.parameters()),
        any(p.requires_grad for p in rgb.lat_l3.parameters()),
        any(p.requires_grad for p in rgb.lat_l4.parameters()),
        any(p.requires_grad for p in rgb.fusion_conv.parameters()),
    ])
    dct_trainable = any(p.requires_grad for p in model.dct_branch.encoder.parameters())
    srm_trainable = any(p.requires_grad for p in model.srm_branch.encoder.parameters())
    fusion_trainable = any(p.requires_grad for p in model.fusion.parameters())
    decoder_trainable = any(p.requires_grad for p in model.decoder.parameters())

    return {
        "stem_frozen": stem_frozen,
        "layer1_frozen": l1_frozen,
        "layer2_frozen": l2_frozen,
        "layer3_trainable": l3_trainable,
        "layer4_trainable": l4_trainable,
        "lateral_trainable": lat_trainable,
        "dct_encoder_trainable": dct_trainable,
        "srm_encoder_trainable": srm_trainable,
        "fusion_trainable": fusion_trainable,
        "decoder_trainable": decoder_trainable,
    }


def load_dataset_splits(config_path: str) -> Tuple[list, list, dict]:
    """Loads and validates the exact 8,500 train and 1,500 val indices from frozen config."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_indices = cfg["train_indices"]
    val_indices   = cfg["val_indices"]

    assert len(train_indices) == 8500, f"Expected 8500 train indices, found {len(train_indices)}"
    assert len(val_indices) == 1500, f"Expected 1500 val indices, found {len(val_indices)}"
    assert len(set(train_indices) & set(val_indices)) == 0, "Train and validation overlap detected!"

    return train_indices, val_indices, cfg


def build_dataloaders(
    dataset_path: str,
    train_indices: list,
    val_indices: list,
    batch_size: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DocTamperLMDBDataset]:
    """Creates training and validation dataloaders reusing a single LMDB dataset instance."""
    dataset = DocTamperLMDBDataset(lmdb_dir=dataset_path, is_train=False)

    train_sub = Subset(dataset, train_indices)
    val_sub   = Subset(dataset, val_indices)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_sub,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=g,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader, dataset


def train_one_epoch(
    model: TamperDocModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLocalizationLoss,
    epoch_idx: int,
    log_interval: int = 100,
) -> Dict[str, float]:
    """Executes one full training epoch."""
    model.train()
    running_loss  = 0.0
    running_bce   = 0.0
    running_dice  = 0.0
    nan_inf_count = 0
    n_batches     = 0
    epoch_start   = time.time()
    total_batches = len(loader)

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
            print(f"  [WARN] Epoch {epoch_idx+1} Batch {batch_idx+1}: Non-finite loss {loss.item():.6f}")
            continue

        loss.backward()
        optimizer.step()

        running_loss += losses["loss"].item()
        running_bce  += losses["bce_loss"].item()
        running_dice += losses["dice_loss"].item()
        n_batches    += 1

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            cur_time = time.time() - epoch_start
            cur_speed = ((batch_idx + 1) * loader.batch_size) / max(cur_time, 1e-6)
            print(f"  [Epoch {epoch_idx+1}] Batch {batch_idx+1:04d}/{total_batches:04d} | "
                  f"Loss: {loss.item():.4f} (BCE: {losses['bce_loss'].item():.4f}, Dice: {losses['dice_loss'].item():.4f}) | "
                  f"Speed: {cur_speed:.2f} samp/s | Elapsed: {cur_time/60:.1f}m", flush=True)

    elapsed = time.time() - epoch_start
    denom = max(n_batches, 1)
    samples_processed = n_batches * loader.batch_size
    throughput = samples_processed / max(elapsed, 1e-6)

    return {
        "train_loss":        running_loss / denom,
        "train_bce_loss":    running_bce  / denom,
        "train_dice_loss":   running_dice / denom,
        "train_time_s":      round(elapsed, 2),
        "train_samples_sec": round(throughput, 4),
        "nan_inf_batches":   nan_inf_count,
    }


def validate(
    model: TamperDocModel,
    loader: DataLoader,
    criterion: CombinedLocalizationLoss,
    log_interval: int = 100,
) -> Dict[str, float]:
    """Evaluates the model on validation set."""
    model.eval()
    running_loss  = 0.0
    running_bce   = 0.0
    running_dice  = 0.0
    n_batches     = 0
    meter         = LocalizationMetrics(threshold=0.5)
    val_start     = time.time()
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
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

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                print(f"  [Validation] Batch {batch_idx+1:04d}/{total_batches:04d}", flush=True)

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


def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    epoch: int,
    is_best: bool = False,
) -> str:
    """Saves checkpoint for current epoch and optionally best checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    best_path = os.path.join(checkpoint_dir, "best_model.pt")

    torch.save(state, epoch_path)
    torch.save(state, latest_path)
    if is_best:
        torch.save(state, best_path)
        print(f"  >>> New Best Validation F1 ({state.get('best_val_f1', 0.0):.4f})! Saved to {best_path} <<<")

    print(f"  Saved checkpoint to {epoch_path}")
    return epoch_path


def load_checkpoint(
    checkpoint_path: str,
    model: TamperDocModel,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, float]:
    """Safely loads a checkpoint and restores model and optimizer states."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("  Restored optimizer state from checkpoint.")
    
    start_epoch = checkpoint["epoch"] + 1
    best_val_f1 = checkpoint.get("best_val_f1", -1.0)
    print(f"Successfully loaded checkpoint '{checkpoint_path}' (Epoch {checkpoint['epoch']} completed, resuming from Epoch {start_epoch}, Best Val F1: {best_val_f1:.4f})")
    return start_epoch, best_val_f1


def main():
    parser = argparse.ArgumentParser(description="TamperDoc Main Development Training (Config B)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--threads", type=int, default=8, help="PyTorch CPU threads (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true", help="Verify configuration without training")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    print("=" * 75)
    print("  TAMPERDOC MAIN DEVELOPMENT TRAINING -- CONFIGURATION B")
    print("=" * 75)
    print(f"PyTorch Version:     {torch.__version__}")
    print(f"Active CPU Threads:  {args.threads}")
    print(f"Split Config:        {CONFIG_PATH}")
    print(f"Dataset LMDB:        {DATASET_PATH}")
    print(f"Checkpoints Dir:     {CHECKPOINT_DIR}")
    print(f"Results Log:         {RESULTS_PATH}")

    # 1. Load exact splits
    train_indices, val_indices, split_cfg = load_dataset_splits(CONFIG_PATH)
    print(f"\n[Split Verification]")
    print(f"  Total LMDB Dataset:  {split_cfg['total_samples']:,} samples")
    print(f"  Training Subset:     {len(train_indices):,} samples")
    print(f"  Validation Subset:   {len(val_indices):,} samples")
    print(f"  Split Seed:          {split_cfg['seed']}")

    # 2. Build model & apply Config B freeze
    model = TamperDocModel(
        use_rgb=True,
        use_dct=True,
        use_srm=True,
        pretrained_rgb=True,
        branch_channels=128,
        fused_channels=128,
        target_size=(512, 512),
    )
    freeze_rgb_early_stages(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    freeze_status    = verify_freeze_state(model)

    print(f"\n[Model Architecture & Freeze Status]")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  Frozen Parameters:    {total_params - trainable_params:,}")
    print(f"  Stem Frozen:          {freeze_status['stem_frozen']}")
    print(f"  Layer 1 Frozen:       {freeze_status['layer1_frozen']}")
    print(f"  Layer 2 Frozen:       {freeze_status['layer2_frozen']}")
    print(f"  Layer 3 Trainable:    {freeze_status['layer3_trainable']}")
    print(f"  Layer 4 Trainable:    {freeze_status['layer4_trainable']}")
    print(f"  DCT Trainable:        {freeze_status['dct_encoder_trainable']}")
    print(f"  SRM Trainable:        {freeze_status['srm_encoder_trainable']}")
    print(f"  Fusion Trainable:     {freeze_status['fusion_trainable']}")
    print(f"  Decoder Trainable:    {freeze_status['decoder_trainable']}")

    # 3. Criterion & Optimizer
    criterion = CombinedLocalizationLoss(lambda_dice=1.0, pos_weight=None)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 4. Resume if requested
    start_epoch = 1
    best_val_f1 = -1.0
    training_history = []

    # Preserve existing history from results log if available
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                existing_log = json.load(f)
                training_history = existing_log.get("history", [])
                print(f"  Loaded existing history with {len(training_history)} epoch(s) from {RESULTS_PATH}")
        except Exception as e:
            print(f"  [Warning] Could not load existing history: {e}")

    if args.resume:
        start_epoch, resume_best_f1 = load_checkpoint(args.resume, model, optimizer)
        if resume_best_f1 > 0:
            best_val_f1 = resume_best_f1
        # If history contains epochs >= start_epoch (e.g. from interrupted run), trim to < start_epoch
        training_history = [h for h in training_history if h.get("epoch", 0) < start_epoch]

    if args.dry_run:
        print(f"\n[DRY RUN VERIFICATION SUCCESSFUL] Next epoch to train: {start_epoch}. Target epochs: {args.epochs}. Exiting without training.")
        return

    # 5. Build Dataloaders
    train_loader, val_loader, dataset = build_dataloaders(
        DATASET_PATH, train_indices, val_indices, batch_size=args.batch_size
    )

    # 6. Training Loop
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    print(f"\n>>> Starting Training Loop: Epoch {start_epoch} to {args.epochs} <<<", flush=True)
    total_train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n{'='*70}\n  EPOCH {epoch}/{args.epochs}\n{'='*70}", flush=True)

        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, epoch - 1)
        val_stats   = validate(model, val_loader, criterion)

        elapsed = time.time() - epoch_start
        is_best = val_stats["val_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_stats["val_f1"]

        epoch_record = {
            "epoch": epoch,
            "epoch_wall_clock_time_s": round(elapsed, 2),
            "best_val_f1_so_far": round(best_val_f1, 4),
        }
        epoch_record.update(train_stats)
        epoch_record.update(val_stats)
        training_history.append(epoch_record)

        # Save checkpoint
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_stats": train_stats,
            "val_stats": val_stats,
            "best_val_f1": best_val_f1,
            "config": {
                "trainable_params": trainable_params,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
            }
        }
        save_checkpoint(checkpoint_state, CHECKPOINT_DIR, epoch, is_best=is_best)

        # Update JSON log
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "experiment": "main_development_training_config_b",
                "train_size": len(train_indices),
                "val_size": len(val_indices),
                "batch_size": args.batch_size,
                "total_epochs": args.epochs,
                "history": training_history,
            }, f, indent=2)

        print(f"\n  Epoch {epoch} Summary: TrainLoss={train_stats['train_loss']:.4f} (BCE={train_stats['train_bce_loss']:.4f}, Dice={train_stats['train_dice_loss']:.4f}) | "
              f"ValLoss={val_stats['val_loss']:.4f} (BCE={val_stats['val_bce_loss']:.4f}, Dice={val_stats['val_dice_loss']:.4f}) | "
              f"ValF1={val_stats['val_f1']:.4f} | ValIoU={val_stats['val_iou']:.4f} | ValPrec={val_stats['val_precision']:.4f} | ValRec={val_stats['val_recall']:.4f} | Time={elapsed/60:.1f}m", flush=True)

    total_elapsed = time.time() - total_train_start
    print(f"\n{'='*70}\n  TRAINING COMPLETE -- Resumed Run Time: {total_elapsed/3600:.2f} hours\n{'='*70}")


if __name__ == "__main__":
    main()
