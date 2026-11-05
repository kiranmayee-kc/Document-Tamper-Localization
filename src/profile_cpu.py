"""CPU Profiling Script for TamperDoc.

Measures fine-grained component-level execution timings (Data Loading, RGB Branch,
DCT Branch, SRM Branch, Spatial Fusion, Localization Decoder, Loss Calculation,
and Backward Pass) across 4, 8, and 12 PyTorch CPU threads on 4 real DocTamper
training samples.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import DocTamperLMDBDataset
from src.losses import CombinedLocalizationLoss
from src.model import TamperDocModel

CONFIG_PATH = os.path.abspath("configs/dev_subset_10k.json")
DATASET_PATH = os.path.abspath("dataset/DocTamper/DocTamperV1-TrainingSet")
THREAD_CONFIGS = [4, 8, 12]
BATCH_SIZE = 4


def load_profile_batch():
  """Loads 4 real samples from the existing frozen training set."""
  with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_data = json.load(f)
  train_indices = config_data["train_indices"][:BATCH_SIZE]

  dataset = DocTamperLMDBDataset(lmdb_dir=DATASET_PATH, is_train=False)
  subset = Subset(dataset, train_indices)
  loader = DataLoader(
      subset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      num_workers=0,
      pin_memory=False,
  )

  t0 = time.perf_counter()
  batch = next(iter(loader))
  load_time = time.perf_counter() - t0

  images = batch["image"]
  raw_images = batch["raw_image"]
  masks = batch["mask"]

  return images, raw_images, masks, load_time, dataset


def report_parameter_counts(model):
  """Reports parameter breakdown across modules."""
  print("\n" + "=" * 75, flush=True)
  print("MODEL PARAMETER BREAKDOWN", flush=True)
  print("=" * 75, flush=True)

  modules = [
      ("RGB Branch (ResNet-18)", model.rgb_branch),
      ("DCT Branch (Basis + CNN)", model.dct_branch),
      ("SRM Branch (Filters + CNN)", model.srm_branch),
      ("Spatial Feature Fusion", model.fusion),
      ("Localization Decoder", model.decoder),
  ]

  total_all = sum(p.numel() for p in model.parameters())
  trainable_all = sum(
      p.numel() for p in model.parameters() if p.requires_grad
  )
  fixed_all = sum(
      p.numel() for p in model.parameters() if not p.requires_grad
  )

  for name, mod in modules:
    tot = sum(p.numel() for p in mod.parameters())
    trn = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    fix = sum(p.numel() for p in mod.parameters() if not p.requires_grad)
    pct = (tot / total_all) * 100
    print(
        f"  {name:<30} | Total: {tot:>10,} ({pct:5.2f}%) | Trainable:"
        f" {trn:>10,} | Fixed: {fix:>6,}",
        flush=True,
    )

  print("-" * 75, flush=True)
  print(
      f"  {'TOTAL MODEL':<30} | Total: {total_all:>10,} (100.0%) | Trainable:"
      f" {trainable_all:>10,} | Fixed: {fixed_all:>6,}",
      flush=True,
  )
  print("=" * 75 + "\n", flush=True)


def profile_threads(num_threads, images, raw_images, masks, data_load_time):
  torch.set_num_threads(num_threads)
  device = torch.device("cpu")

  # Instantiate clean model and loss
  model = TamperDocModel(
      use_rgb=True,
      use_dct=True,
      use_srm=True,
      pretrained_rgb=True,
      branch_channels=128,
      fused_channels=128,
      target_size=(512, 512),
  ).to(device)
  criterion = CombinedLocalizationLoss(lambda_dice=1.0, pos_weight=None)

  model.train()

  # Warmup pass (1 forward+backward) to prime CPU caches and JIT paths
  out_w = model(images, raw_x=raw_images)
  loss_w = criterion(out_w["tamper_logits"], masks)["loss"]
  loss_w.backward()
  model.zero_grad()

  # Measured Step-by-Step Profiling
  # 1. RGB Branch
  t_start_rgb = time.perf_counter()
  rgb_feat = model.rgb_branch(images)
  t_rgb = time.perf_counter() - t_start_rgb

  # 2. DCT Branch
  t_start_dct = time.perf_counter()
  dct_feat = model.dct_branch(raw_images)
  t_dct = time.perf_counter() - t_start_dct

  # 3. SRM Branch
  t_start_srm = time.perf_counter()
  srm_feat = model.srm_branch(raw_images)
  t_srm = time.perf_counter() - t_start_srm

  # 4. Fusion Module
  t_start_fusion = time.perf_counter()
  fused_feat = model.fusion([rgb_feat, dct_feat, srm_feat])
  t_fusion = time.perf_counter() - t_start_fusion

  # 5. Localization Decoder
  t_start_decoder = time.perf_counter()
  logits = model.decoder(fused_feat)
  t_decoder = time.perf_counter() - t_start_decoder

  total_forward_time = t_rgb + t_dct + t_srm + t_fusion + t_decoder

  # 6. Loss Calculation
  t_start_loss = time.perf_counter()
  loss_dict = criterion(logits, masks)
  total_loss = loss_dict["loss"]
  t_loss = time.perf_counter() - t_start_loss

  # 7. Backward Pass
  t_start_backward = time.perf_counter()
  total_loss.backward()
  t_backward = time.perf_counter() - t_start_backward

  total_fwd_bwd_time = total_forward_time + t_loss + t_backward
  total_with_data_time = total_fwd_bwd_time + data_load_time
  samples_per_sec = BATCH_SIZE / total_with_data_time

  nan_inf = (
      torch.isnan(logits).any().item()
      or torch.isinf(logits).any().item()
      or torch.isnan(total_loss).item()
      or torch.isinf(total_loss).item()
  )

  return {
      "threads": num_threads,
      "data_load_time": data_load_time,
      "t_rgb": t_rgb,
      "t_dct": t_dct,
      "t_srm": t_srm,
      "t_fusion": t_fusion,
      "t_decoder": t_decoder,
      "t_forward": total_forward_time,
      "t_loss": t_loss,
      "t_backward": t_backward,
      "t_total_fwd_bwd": total_fwd_bwd_time,
      "t_total_batch": total_with_data_time,
      "samples_per_sec": samples_per_sec,
      "nan_inf": nan_inf,
      "loss_val": total_loss.item(),
  }


def main():
  print("=" * 75, flush=True)
  print("TAMPERDOC CPU EXECUTION PROFILING ON 4 REAL SAMPLES", flush=True)
  print("=" * 75, flush=True)

  # 1. Load Data
  images, raw_images, masks, load_time, dataset = load_profile_batch()
  print(
      f"Loaded 4 real samples (Shape: {images.shape}) in {load_time*1000:.2f} ms",
      flush=True,
  )

  # 2. Parameter Breakdown
  init_model = TamperDocModel(
      use_rgb=True,
      use_dct=True,
      use_srm=True,
      pretrained_rgb=True,
      branch_channels=128,
      fused_channels=128,
      target_size=(512, 512),
  )
  report_parameter_counts(init_model)
  del init_model

  # 3. Profile Thread Configurations
  results = []
  for th in THREAD_CONFIGS:
    print(f"--- Profiling with {th} PyTorch CPU Threads ---", flush=True)
    res = profile_threads(th, images, raw_images, masks, load_time)
    results.append(res)
    print(
        f"  Total Batch Time: {res['t_total_batch']:.4f}s | Forward:"
        f" {res['t_forward']:.4f}s | Backward: {res['t_backward']:.4f}s |"
        f" Speed: {res['samples_per_sec']:.2f} samples/s",
        flush=True,
    )

  # 4. Comprehensive Timing Table
  print("\n" + "=" * 95, flush=True)
  print("FINE-GRAINED CPU PROFILING RESULTS SUMMARY (4 SAMPLES / BATCH)", flush=True)
  print("=" * 95, flush=True)
  header = (
      f"{'Component':<28} | {'4 Threads (s)':<14} | {'8 Threads (s)':<14} |"
      f" {'12 Threads (s)':<14}"
  )
  print(header, flush=True)
  print("-" * 95, flush=True)

  r4, r8, r12 = results[0], results[1], results[2]

  rows = [
      (
          "1. Data Loading (Batch of 4)",
          r4["data_load_time"],
          r8["data_load_time"],
          r12["data_load_time"],
      ),
      ("2. RGB Branch Forward", r4["t_rgb"], r8["t_rgb"], r12["t_rgb"]),
      ("3. DCT Branch Forward", r4["t_dct"], r8["t_dct"], r12["t_dct"]),
      ("4. SRM Branch Forward", r4["t_srm"], r8["t_srm"], r12["t_srm"]),
      (
          "5. Spatial Feature Fusion",
          r4["t_fusion"],
          r8["t_fusion"],
          r12["t_fusion"],
      ),
      (
          "6. Localization Decoder",
          r4["t_decoder"],
          r8["t_decoder"],
          r12["t_decoder"],
      ),
      ("7. Loss Calculation", r4["t_loss"], r8["t_loss"], r12["t_loss"]),
      (
          "8. Backward Pass (Autograd)",
          r4["t_backward"],
          r8["t_backward"],
          r12["t_backward"],
      ),
      ("-" * 28, "-" * 14, "-" * 14, "-" * 14),
      (
          "TOTAL FORWARD PASS",
          r4["t_forward"],
          r8["t_forward"],
          r12["t_forward"],
      ),
      (
          "TOTAL FWD + LOSS + BWD",
          r4["t_total_fwd_bwd"],
          r8["t_total_fwd_bwd"],
          r12["t_total_fwd_bwd"],
      ),
      (
          "TOTAL BATCH TIME (incl Data)",
          r4["t_total_batch"],
          r8["t_total_batch"],
          r12["t_total_batch"],
      ),
      (
          "THROUGHPUT (samples/sec)",
          f"{r4['samples_per_sec']:.2f}",
          f"{r8['samples_per_sec']:.2f}",
          f"{r12['samples_per_sec']:.2f}",
      ),
      (
          "NaN / Inf Detected",
          str(r4["nan_inf"]),
          str(r8["nan_inf"]),
          str(r12["nan_inf"]),
      ),
  ]

  for item in rows:
    if isinstance(item[1], float):
      print(
          f"{item[0]:<28} | {item[1]:>12.4f} s | {item[2]:>12.4f} s |"
          f" {item[3]:>12.4f} s",
          flush=True,
      )
    else:
      print(
          f"{item[0]:<28} | {item[1]:>14} | {item[2]:>14} | {item[3]:>14}",
          flush=True,
      )
  print("=" * 95 + "\n", flush=True)


if __name__ == "__main__":
  main()
