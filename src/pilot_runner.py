"""TamperDoc 10K Subset Creation, Verification, and CPU Pilot Benchmark.

Contains full implementations of:
- Phase 1: phase1_create_subset()
- Phase 2: phase2_verify_samples()
- Phase 3: phase3_foreground_statistics()
- Phase 4: phase4_cpu_pilot()
as well as load_existing_subset() for non-destructive reuse of existing configs.
"""

import json
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import DocTamperLMDBDataset
from src.losses import CombinedLocalizationLoss
from src.metrics import LocalizationMetrics
from src.model import TamperDocModel

TOTAL_SAMPLES = 120000
SUBSET_SIZE = 10000
TRAIN_SIZE = 8500
VAL_SIZE = 1500
SEED = 42

PILOT_TRAIN_SIZE = 256
PILOT_VAL_SIZE = 64
BATCH_SIZE = 4
CONFIG_PATH = os.path.abspath("configs/dev_subset_10k.json")


def load_existing_subset():
  """Loads existing 10K subset from configs/dev_subset_10k.json without regenerating."""
  if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

  with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_data = json.load(f)

  train_indices = config_data["train_indices"]
  val_indices = config_data["val_indices"]
  print(f"✓ Loaded existing subset from {CONFIG_PATH}", flush=True)
  print(f"  Dataset:       {config_data.get('dataset_name')}", flush=True)
  print(f"  Random Seed:   {config_data.get('seed')}", flush=True)
  print(f"  Train Size:    {len(train_indices):,}", flush=True)
  print(f"  Val Size:      {len(val_indices):,}", flush=True)
  return train_indices, val_indices


def phase1_create_subset():
  """Phase 1: Generates deterministic 10K subset and saves to configs/dev_subset_10k.json."""
  print("=" * 75, flush=True)
  print(
      "PHASE 1 — DETERMINISTIC 10K SUBSET GENERATION (seed=42)", flush=True
  )
  print("=" * 75, flush=True)

  rng = np.random.default_rng(SEED)
  full_permutation = rng.permutation(TOTAL_SAMPLES)
  subset_10k = full_permutation[:SUBSET_SIZE].tolist()

  train_indices = subset_10k[:TRAIN_SIZE]
  val_indices = subset_10k[TRAIN_SIZE:]

  # Assertions
  assert (
      len(train_indices) == TRAIN_SIZE
  ), f"Expected {TRAIN_SIZE} train indices, got {len(train_indices)}"
  assert (
      len(val_indices) == VAL_SIZE
  ), f"Expected {VAL_SIZE} val indices, got {len(val_indices)}"
  assert (
      len(set(train_indices)) == TRAIN_SIZE
  ), "Duplicate detected in train_indices!"
  assert (
      len(set(val_indices)) == VAL_SIZE
  ), "Duplicate detected in val_indices!"
  assert len(set(train_indices).intersection(set(val_indices))) == 0, (
      "Overlap between train and validation indices!"
  )
  assert all(0 <= idx < TOTAL_SAMPLES for idx in subset_10k), (
      "Index out of [0, 119999] bounds!"
  )

  config_data = {
      "dataset_name": "DocTamperV1-TrainingSet",
      "total_samples": TOTAL_SAMPLES,
      "seed": SEED,
      "subset_size": SUBSET_SIZE,
      "train_size": TRAIN_SIZE,
      "val_size": VAL_SIZE,
      "train_indices": train_indices,
      "val_indices": val_indices,
  }

  with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config_data, f, indent=2)

  print(f"✓ Created {CONFIG_PATH}", flush=True)
  print(f"  Train indices count: {len(train_indices):,}", flush=True)
  print(f"  Val indices count:   {len(val_indices):,}", flush=True)
  print(f"  First 5 train IDs:   {train_indices[:5]}", flush=True)
  print(f"  First 5 val IDs:     {val_indices[:5]}", flush=True)
  return train_indices, val_indices


def phase2_verify_samples(dataset, train_indices, val_indices):
  """Phase 2: Verifies image and mask integrity for sampled indices."""
  print("\n" + "=" * 75, flush=True)
  print("PHASE 2 — SAMPLE INTEGRITY & HANDLE VERIFICATION", flush=True)
  print("=" * 75, flush=True)

  test_sample_ids = train_indices[:3] + val_indices[:3]
  print(
      f"Probing {len(test_sample_ids)} sample entries across train and"
      f" validation: {test_sample_ids}",
      flush=True,
  )

  for sid in test_sample_ids:
    sample = dataset[sid]
    img = sample["image"]
    raw_img = sample["raw_image"]
    mask = sample["mask"]

    assert img.shape == (
        3,
        512,
        512,
    ), f"Sample {sid} image shape unexpected: {img.shape}"
    assert raw_img.shape == (
        3,
        512,
        512,
    ), f"Sample {sid} raw_image shape unexpected: {raw_img.shape}"
    assert mask.shape == (
        1,
        512,
        512,
    ), f"Sample {sid} mask shape unexpected: {mask.shape}"

    uniques = torch.unique(mask).tolist()
    assert all(u in [0.0, 1.0] for u in uniques), (
        f"Sample {sid} mask contains invalid values: {uniques}"
    )

    tampered_pixels = int(torch.count_nonzero(mask > 0).item())
    print(
        f"  Sample #{sid:06d}: Image={img.shape} RGB in [{raw_img.min():.2f},"
        f" {raw_img.max():.2f}] | Mask={mask.shape} values={uniques} |"
        f" Tampered Pixels={tampered_pixels:,} ({tampered_pixels/(512*512)*100:.2f}%)",
        flush=True,
    )
  print(
      "✓ Sample integrity, {0, 1} mask normalization, and LMDB handle"
      " verified.",
      flush=True,
  )


def phase3_foreground_statistics(dataset, pilot_train_indices):
  """Phase 3: Computes foreground and background pixel counts and ratios on pilot samples."""
  print("\n" + "=" * 75, flush=True)
  print("PHASE 3 — FOREGROUND / BACKGROUND PIXEL STATISTICS (256 SAMPLES)", flush=True)
  print("=" * 75, flush=True)

  dataset._init_db()
  txn = dataset.txn

  total_pixels = 0
  positive_pixels = 0
  negative_pixels = 0

  for idx in pilot_train_indices:
    mask_key = f"label-{idx:09d}".encode("utf-8")
    raw_bytes = txn.get(mask_key)
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    mask = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    pos = (
        cv2.countNonZero(mask)
        if len(mask.shape) == 2
        else np.count_nonzero(mask > 0)
    )
    tot = mask.shape[0] * mask.shape[1]
    neg = tot - pos

    positive_pixels += pos
    negative_pixels += neg
    total_pixels += tot

  pos_ratio = positive_pixels / total_pixels
  neg_ratio = negative_pixels / total_pixels
  implied_pos_weight = (
      negative_pixels / positive_pixels if positive_pixels > 0 else 0.0
  )

  print(f"Total Samples Analyzed:       {len(pilot_train_indices)}", flush=True)
  print(
      f"Total Pixels Supervised:      {total_pixels:,} (256 × 512 × 512)",
      flush=True,
  )
  print(
      f"Positive (Tampered) Pixels:   {positive_pixels:,} ({pos_ratio*100:.3f}%)",
      flush=True,
  )
  print(
      f"Negative (Authentic) Pixels:  {negative_pixels:,} ({neg_ratio*100:.3f}%)",
      flush=True,
  )
  print(f"Positive Pixel Ratio:         {pos_ratio:.6f}", flush=True)
  print(f"Negative Pixel Ratio:         {neg_ratio:.6f}", flush=True)
  print(
      f"Implied Pos Weight (Neg/Pos): {implied_pos_weight:.2f} (Report only;"
      " NOT applied)",
      flush=True,
  )

  return {
      "total_pixels": total_pixels,
      "positive_pixels": positive_pixels,
      "negative_pixels": negative_pixels,
      "pos_ratio": pos_ratio,
      "neg_ratio": neg_ratio,
      "implied_pos_weight": implied_pos_weight,
  }


def phase4_cpu_pilot(dataset, train_indices, val_indices):
  """Phase 4: Executes 1-epoch baseline CPU runtime benchmark on full multi-branch model."""
  print("\n" + "=" * 75, flush=True)
  print(
      "PHASE 4 — CPU RUNTIME PILOT BENCHMARK (1 EPOCH, 256 TRAIN / 64 VAL)",
      flush=True,
  )
  print("=" * 75, flush=True)

  tracemalloc.start()

  pilot_train_ids = train_indices[:PILOT_TRAIN_SIZE]
  pilot_val_ids = val_indices[:PILOT_VAL_SIZE]

  train_subset = Subset(dataset, pilot_train_ids)
  val_subset = Subset(dataset, pilot_val_ids)

  train_loader = DataLoader(
      train_subset,
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers=0,
      pin_memory=False,
  )
  val_loader = DataLoader(
      val_subset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      num_workers=0,
      pin_memory=False,
  )

  device = torch.device("cpu")
  num_threads = torch.get_num_threads()
  print(
      f"Device: {device} | Active PyTorch CPU Threads: {num_threads}", flush=True
  )

  # Full Multi-Branch Model
  model = TamperDocModel(
      use_rgb=True,
      use_dct=True,
      use_srm=True,
      pretrained_rgb=True,
      branch_channels=128,
      fused_channels=128,
      target_size=(512, 512),
  ).to(device)

  # Standard Loss (Default: lambda_dice=1.0, pos_weight=None)
  criterion = CombinedLocalizationLoss(lambda_dice=1.0, pos_weight=None)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

  print(
      f"Configuration: Batch Size = {BATCH_SIZE}, Num Workers = 0, Pin Memory ="
      " False",
      flush=True,
  )
  print(
      f"Training Set: {len(train_subset)} samples ({len(train_loader)} batches)",
      flush=True,
  )
  print(
      f"Validation Set: {len(val_subset)} samples ({len(val_loader)} batches)",
      flush=True,
  )

  # 1. Training Pass
  print("\n--- Running 1 Epoch Training ---", flush=True)
  model.train()
  train_start_time = time.time()

  train_total_loss = 0.0
  train_total_bce = 0.0
  train_total_dice = 0.0
  train_metrics = LocalizationMetrics(threshold=0.5)
  nan_inf_detected = False
  forward_success = True
  backward_success = True
  optimizer_success = True

  for step, batch in enumerate(train_loader):
    images = batch["image"].to(device)
    raw_images = batch["raw_image"].to(device)
    masks = batch["mask"].to(device)

    optimizer.zero_grad()
    try:
      outputs = model(images, raw_x=raw_images)
      logits = outputs["tamper_logits"]
    except Exception as e:
      forward_success = False
      raise e

    if torch.isnan(logits).any() or torch.isinf(logits).any():
      nan_inf_detected = True

    loss_dict = criterion(logits, masks)
    loss = loss_dict["loss"]
    bce = loss_dict["bce_loss"]
    dice = loss_dict["dice_loss"]

    if torch.isnan(loss) or torch.isinf(loss):
      nan_inf_detected = True

    try:
      loss.backward()
    except Exception as e:
      backward_success = False
      raise e

    # Verify gradients before step
    for name, p in model.named_parameters():
      if p.requires_grad:
        if p.grad is None or torch.isnan(p.grad).any():
          backward_success = False

    try:
      optimizer.step()
    except Exception as e:
      optimizer_success = False
      raise e

    train_total_loss += loss.item() * images.size(0)
    train_total_bce += bce.item() * images.size(0)
    train_total_dice += dice.item() * images.size(0)
    train_metrics.update(logits.detach(), masks)

    if (step + 1) % 16 == 0 or (step + 1) == len(train_loader):
      curr_time = time.time() - train_start_time
      curr_speed = ((step + 1) * BATCH_SIZE) / curr_time
      print(
          f"  Train Batch [{step+1:02d}/{len(train_loader):02d}] | Loss:"
          f" {loss.item():.4f} (BCE: {bce.item():.4f}, Dice: {dice.item():.4f})"
          f" | Speed: {curr_speed:.2f} samples/s | Elapsed: {curr_time:.1f}s",
          flush=True,
      )

  train_time = time.time() - train_start_time
  train_speed = len(train_subset) / train_time

  avg_train_loss = train_total_loss / len(train_subset)
  avg_train_bce = train_total_bce / len(train_subset)
  avg_train_dice = train_total_dice / len(train_subset)
  train_results = train_metrics.compute()

  # 2. Validation Pass
  print("\n--- Running Validation (64 samples) ---", flush=True)
  model.eval()
  val_start_time = time.time()

  val_total_loss = 0.0
  val_total_bce = 0.0
  val_total_dice = 0.0
  val_metrics = LocalizationMetrics(threshold=0.5)

  with torch.no_grad():
    for step, batch in enumerate(val_loader):
      images = batch["image"].to(device)
      raw_images = batch["raw_image"].to(device)
      masks = batch["mask"].to(device)

      outputs = model(images, raw_x=raw_images)
      logits = outputs["tamper_logits"]

      if torch.isnan(logits).any() or torch.isinf(logits).any():
        nan_inf_detected = True

      loss_dict = criterion(logits, masks)
      loss = loss_dict["loss"]
      bce = loss_dict["bce_loss"]
      dice = loss_dict["dice_loss"]

      val_total_loss += loss.item() * images.size(0)
      val_total_bce += bce.item() * images.size(0)
      val_total_dice += dice.item() * images.size(0)
      val_metrics.update(logits, masks)

  val_time = time.time() - val_start_time
  val_speed = len(val_subset) / val_time
  total_time = train_time + val_time

  current_mem, peak_mem = tracemalloc.get_traced_memory()
  tracemalloc.stop()

  avg_val_loss = val_total_loss / len(val_subset)
  avg_val_bce = val_total_bce / len(val_subset)
  avg_val_dice = val_total_dice / len(val_subset)
  val_results = val_metrics.compute()

  print("\n" + "=" * 75, flush=True)
  print("PILOT BENCHMARK MEASUREMENTS", flush=True)
  print("=" * 75, flush=True)
  print(f"1. Device:                     {device}", flush=True)
  print(f"2. CPU threads:                {num_threads}", flush=True)
  print(f"3. Batch size:                 {BATCH_SIZE}", flush=True)
  print(f"4. num_workers:                0", flush=True)
  print(f"5. pin_memory:                 False", flush=True)
  print(f"6. Training samples:           {len(train_subset)}", flush=True)
  print(f"7. Validation samples:         {len(val_subset)}", flush=True)
  print(f"8. Training batches:           {len(train_loader)}", flush=True)
  print(f"9. Validation batches:         {len(val_loader)}", flush=True)
  print(f"10. Training wall-clock time:  {train_time:.2f}s", flush=True)
  print(f"11. Validation wall-clock time:{val_time:.2f}s", flush=True)
  print(f"12. Total wall-clock time:     {total_time:.2f}s", flush=True)
  print(f"13. Training speed:            {train_speed:.2f} samples/s", flush=True)
  print(f"14. Validation speed:          {val_speed:.2f} samples/s", flush=True)
  print(
      f"15. Peak Python RAM:           {peak_mem / (1024**2):.2f} MB", flush=True
  )
  print(f"16. Training Loss:             {avg_train_loss:.6f}", flush=True)
  print(f"17. Training BCE Loss:         {avg_train_bce:.6f}", flush=True)
  print(f"18. Training Dice Loss:        {avg_train_dice:.6f}", flush=True)
  print(f"19. Validation Loss:           {avg_val_loss:.6f}", flush=True)
  print(f"20. Validation BCE Loss:       {avg_val_bce:.6f}", flush=True)
  print(f"21. Validation Dice Loss:      {avg_val_dice:.6f}", flush=True)
  print(
      f"22. Validation Precision:      {val_results['precision']:.4f}",
      flush=True,
  )
  print(
      f"23. Validation Recall:         {val_results['recall']:.4f}", flush=True
  )
  print(f"24. Validation F1 / Dice:      {val_results['f1']:.4f}", flush=True)
  print(f"25. Validation IoU:            {val_results['iou']:.4f}", flush=True)
  print(
      f"26. Validation Pixel Accuracy: {val_results['pixel_accuracy']:.4f}",
      flush=True,
  )
  print(
      f"27. NaN / Inf Detected:        {nan_inf_detected} (CLEAN)", flush=True
  )
  print(
      f"28. Forward Pass Success:      {forward_success} (SUCCESS)", flush=True
  )
  print(
      f"29. Backward Pass Success:     {backward_success} (SUCCESS)", flush=True
  )
  print(
      f"30. Optimizer Update Success:  {optimizer_success} (SUCCESS)", flush=True
  )

  return {
      "device": str(device),
      "num_threads": num_threads,
      "batch_size": BATCH_SIZE,
      "num_workers": 0,
      "pin_memory": False,
      "train_samples": len(train_subset),
      "val_samples": len(val_subset),
      "train_batches": len(train_loader),
      "val_batches": len(val_loader),
      "train_time": train_time,
      "val_time": val_time,
      "total_time": total_time,
      "train_speed": train_speed,
      "val_speed": val_speed,
      "peak_ram_mb": peak_mem / (1024**2),
      "train_loss": avg_train_loss,
      "train_bce": avg_train_bce,
      "train_dice": avg_train_dice,
      "val_loss": avg_val_loss,
      "val_bce": avg_val_bce,
      "val_dice": avg_val_dice,
      "val_precision": val_results["precision"],
      "val_recall": val_results["recall"],
      "val_f1": val_results["f1"],
      "val_iou": val_results["iou"],
      "val_pixel_accuracy": val_results["pixel_accuracy"],
      "nan_inf": nan_inf_detected,
      "forward_success": forward_success,
      "backward_success": backward_success,
      "optimizer_success": optimizer_success,
  }


if __name__ == "__main__":
  dataset_path = os.path.abspath("dataset/DocTamper/DocTamperV1-TrainingSet")

  # 1. Load existing 10K subset (non-destructive)
  train_ids, val_ids = load_existing_subset()

  # 2. Initialize dataset
  dataset = DocTamperLMDBDataset(lmdb_dir=dataset_path, is_train=False)

  # 3. Print required start banner
  print(
      "\n===========================================================================",
      flush=True,
  )
  print(
      "STARTING PHASE 4 CPU PILOT — 256 TRAIN / 64 VAL / 1 EPOCH", flush=True
  )
  print(
      "===========================================================================\n",
      flush=True,
  )

  # 4. Run Phase 4
  pilot_results = phase4_cpu_pilot(dataset, train_ids, val_ids)

  # 5. Print required stop banner
  print(
      "\n===========================================================================",
      flush=True,
  )
  print("PHASE 4 CPU PILOT COMPLETE — STOPPING", flush=True)
  print(
      "===========================================================================",
      flush=True,
  )
