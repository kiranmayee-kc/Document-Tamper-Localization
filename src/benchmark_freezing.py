"""Controlled Benchmark: Full Backprop vs. Partially Frozen ResNet-18 RGB Backbone.

Compares Configuration A (Full Backprop) vs Configuration B (Partially Frozen RGB:
conv1/stem, layer1, layer2 frozen; layer3, layer4 and all other branches trainable)
on 4 real DocTamper samples at 8 PyTorch CPU threads.
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
BATCH_SIZE = 4
NUM_THREADS = 8


def load_real_samples():
  """Loads the exact 4 real training samples from dev_subset_10k.json."""
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

  batch = next(iter(loader))
  images = batch["image"]
  raw_images = batch["raw_image"]
  masks = batch["mask"]

  return images, raw_images, masks, train_indices


def freeze_rgb_early_stages(model: TamperDocModel):
  """Freezes ResNet-18 conv1/stem, layer1, and layer2.

  Keeps layer3, layer4, lateral connections, DCT, SRM, Fusion, and Decoder
  trainable.
  """
  rgb = model.rgb_branch

  # Freeze conv1, bn1
  for p in rgb.conv1.parameters():
    p.requires_grad = False
  for p in rgb.bn1.parameters():
    p.requires_grad = False

  # Freeze layer1
  for p in rgb.layer1.parameters():
    p.requires_grad = False

  # Freeze layer2
  for p in rgb.layer2.parameters():
    p.requires_grad = False


def verify_freeze_state(model: TamperDocModel):
  """Inspects and returns the exact trainability status of all modules."""
  rgb = model.rgb_branch

  stem_trainable = any(
      p.requires_grad for p in rgb.conv1.parameters()
  ) or any(p.requires_grad for p in rgb.bn1.parameters())
  l1_trainable = any(p.requires_grad for p in rgb.layer1.parameters())
  l2_trainable = any(p.requires_grad for p in rgb.layer2.parameters())
  l3_trainable = any(p.requires_grad for p in rgb.layer3.parameters())
  l4_trainable = any(p.requires_grad for p in rgb.layer4.parameters())

  lat_trainable = all([
      any(p.requires_grad for p in rgb.lat_l1.parameters()),
      any(p.requires_grad for p in rgb.lat_l2.parameters()),
      any(p.requires_grad for p in rgb.lat_l3.parameters()),
      any(p.requires_grad for p in rgb.lat_l4.parameters()),
      any(p.requires_grad for p in rgb.fusion_conv.parameters()),
  ])

  dct_trainable = any(
      p.requires_grad for p in model.dct_branch.encoder.parameters()
  )
  srm_trainable = any(
      p.requires_grad for p in model.srm_branch.encoder.parameters()
  )
  fusion_trainable = any(p.requires_grad for p in model.fusion.parameters())
  decoder_trainable = any(p.requires_grad for p in model.decoder.parameters())

  # Check fixed basis kernels
  dct_basis_fixed = not model.dct_branch.dct_conv.weight.requires_grad
  srm_basis_fixed = not model.srm_branch.srm_conv.weight.requires_grad

  return {
      "stem_trainable": stem_trainable,
      "l1_trainable": l1_trainable,
      "l2_trainable": l2_trainable,
      "l3_trainable": l3_trainable,
      "l4_trainable": l4_trainable,
      "lat_trainable": lat_trainable,
      "dct_trainable": dct_trainable,
      "srm_trainable": srm_trainable,
      "fusion_trainable": fusion_trainable,
      "decoder_trainable": decoder_trainable,
      "dct_basis_fixed": dct_basis_fixed,
      "srm_basis_fixed": srm_basis_fixed,
  }


def benchmark_configuration(name, model, images, raw_images, masks):
  criterion = CombinedLocalizationLoss(lambda_dice=1.0, pos_weight=None)
  model.train()

  # Warmup pass (1 forward + loss + backward) to eliminate cold-start cache/JIT variance
  out_w = model(images, raw_x=raw_images)
  loss_w = criterion(out_w["tamper_logits"], masks)["loss"]
  loss_w.backward()
  model.zero_grad()

  # Measured Pass
  t_fwd_start = time.perf_counter()
  outputs = model(images, raw_x=raw_images)
  logits = outputs["tamper_logits"]
  t_fwd = time.perf_counter() - t_fwd_start

  t_loss_start = time.perf_counter()
  loss_dict = criterion(logits, masks)
  total_loss = loss_dict["loss"]
  t_loss = time.perf_counter() - t_loss_start

  backward_success = True
  t_bwd_start = time.perf_counter()
  try:
    total_loss.backward()
  except Exception as e:
    backward_success = False
    raise e
  t_bwd = time.perf_counter() - t_bwd_start

  total_time = t_fwd + t_loss + t_bwd
  throughput = BATCH_SIZE / total_time

  nan_inf = (
      torch.isnan(logits).any().item()
      or torch.isinf(logits).any().item()
      or torch.isnan(total_loss).item()
      or torch.isinf(total_loss).item()
  )

  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(
      p.numel() for p in model.parameters() if p.requires_grad
  )
  frozen_params = sum(
      p.numel() for p in model.parameters() if not p.requires_grad
  )

  freeze_status = verify_freeze_state(model)

  return {
      "name": name,
      "total_params": total_params,
      "trainable_params": trainable_params,
      "frozen_params": frozen_params,
      "t_fwd": t_fwd,
      "t_loss": t_loss,
      "t_bwd": t_bwd,
      "t_total": total_time,
      "throughput": throughput,
      "nan_inf": nan_inf,
      "backward_success": backward_success,
      "freeze_status": freeze_status,
      "loss_value": total_loss.item(),
  }


def main():
  print("=" * 85, flush=True)
  print(
      "CONTROLLED BENCHMARK: FULL BACKPROP (A) VS PARTIALLY FROZEN RGB (B)",
      flush=True,
  )
  print("=" * 85, flush=True)

  torch.set_num_threads(NUM_THREADS)
  device = torch.device("cpu")
  print(
      f"Device: {device} | Active PyTorch Threads: {torch.get_num_threads()}",
      flush=True,
  )

  # Load identical 4 real samples
  images, raw_images, masks, sample_ids = load_real_samples()
  print(
      f"Loaded 4 Real DocTamper Samples (Indices: {sample_ids}, Image Tensor:"
      f" {images.shape})",
      flush=True,
  )

  # -------------------------------------------------------------
  # CONFIGURATION A: Baseline (Full Model Trainable)
  # -------------------------------------------------------------
  print("\n--- Running Configuration A (Full Baseline) ---", flush=True)
  model_a = TamperDocModel(
      use_rgb=True,
      use_dct=True,
      use_srm=True,
      pretrained_rgb=True,
      branch_channels=128,
      fused_channels=128,
      target_size=(512, 512),
  ).to(device)

  res_a = benchmark_configuration(
      "Config A (Full Baseline)", model_a, images, raw_images, masks
  )
  del model_a

  # -------------------------------------------------------------
  # CONFIGURATION B: Partially Frozen RGB Backbone
  # -------------------------------------------------------------
  print(
      "--- Running Configuration B (Partially Frozen RGB) ---", flush=True
  )
  model_b = TamperDocModel(
      use_rgb=True,
      use_dct=True,
      use_srm=True,
      pretrained_rgb=True,
      branch_channels=128,
      fused_channels=128,
      target_size=(512, 512),
  ).to(device)

  freeze_rgb_early_stages(model_b)

  res_b = benchmark_configuration(
      "Config B (Partially Frozen RGB)", model_b, images, raw_images, masks
  )
  del model_b

  # -------------------------------------------------------------
  # Detailed Comparison Table
  # -------------------------------------------------------------
  print("\n" + "=" * 105, flush=True)
  print("BENCHMARK COMPARISON TABLE (8 CPU Threads, Batch Size = 4)", flush=True)
  print("=" * 105, flush=True)
  header = (
      f"{'Configuration':<32} | {'Trainable Params':<18} | {'Forward (s)':<12}"
      f" | {'Loss (s)':<9} | {'Backward (s)':<13} | {'Total (s)':<10} |"
      f" {'Samples/sec':<11}"
  )
  print(header, flush=True)
  print("-" * 105, flush=True)

  print(
      f"{res_a['name']:<32} | {res_a['trainable_params']:>12,} ({res_a['trainable_params']/res_a['total_params']*100:.1f}%) |"
      f" {res_a['t_fwd']:>10.4f} s | {res_a['t_loss']:>7.4f} s |"
      f" {res_a['t_bwd']:>11.4f} s | {res_a['t_total']:>8.4f} s |"
      f" {res_a['throughput']:>9.2f} s/s",
      flush=True,
  )
  print(
      f"{res_b['name']:<32} | {res_b['trainable_params']:>12,} ({res_b['trainable_params']/res_b['total_params']*100:.1f}%) |"
      f" {res_b['t_fwd']:>10.4f} s | {res_b['t_loss']:>7.4f} s |"
      f" {res_b['t_bwd']:>11.4f} s | {res_b['t_total']:>8.4f} s |"
      f" {res_b['throughput']:>9.2f} s/s",
      flush=True,
  )
  print("=" * 105, flush=True)

  # Key Metrics
  bwd_reduction = (
      (res_a["t_bwd"] - res_b["t_bwd"]) / res_a["t_bwd"]
  ) * 100
  total_speedup = (
      (res_b["throughput"] - res_a["throughput"]) / res_a["throughput"]
  ) * 100
  time_ratio = res_a["t_total"] / res_b["t_total"]

  print("\n--- Summary Performance Deltas ---", flush=True)
  print(
      f"• Backward Pass Time Reduction: {bwd_reduction:.2f}% (from"
      f" {res_a['t_bwd']:.4f}s down to {res_b['t_bwd']:.4f}s)",
      flush=True,
  )
  print(
      f"• Total Step Speedup:           {time_ratio:.2f}x ({total_speedup:+.1f}%"
      f" throughput increase: {res_a['throughput']:.2f} ->"
      f" {res_b['throughput']:.2f} samples/s)",
      flush=True,
  )
  print(
      f"• Trainable Parameter Reduction:{res_a['trainable_params'] - res_b['trainable_params']:,} parameters frozen"
      f" ({res_b['trainable_params']:,} remaining trainable)",
      flush=True,
  )

  # Stage Verification Breakdown
  print("\n--- Detailed Trainability Status Verification ---", flush=True)
  print(
      f"Configuration B Stage Status:\n"
      f"  - ResNet-18 Stem (conv1+bn1): {'TRAINABLE' if res_b['freeze_status']['stem_trainable'] else 'FROZEN (requires_grad=False)'}\n"
      f"  - ResNet-18 Layer 1:          {'TRAINABLE' if res_b['freeze_status']['l1_trainable'] else 'FROZEN (requires_grad=False)'}\n"
      f"  - ResNet-18 Layer 2:          {'TRAINABLE' if res_b['freeze_status']['l2_trainable'] else 'FROZEN (requires_grad=False)'}\n"
      f"  - ResNet-18 Layer 3:          {'TRAINABLE (requires_grad=True)' if res_b['freeze_status']['l3_trainable'] else 'FROZEN'}\n"
      f"  - ResNet-18 Layer 4:          {'TRAINABLE (requires_grad=True)' if res_b['freeze_status']['l4_trainable'] else 'FROZEN'}\n"
      f"  - Lateral Projections:        {'TRAINABLE (requires_grad=True)' if res_b['freeze_status']['lat_trainable'] else 'FROZEN'}\n"
      f"  - DCT Branch Encoder:         {'TRAINABLE (requires_grad=True)' if res_b['freeze_status']['dct_trainable'] else 'FROZEN'}\n"
      f"  - SRM Branch Encoder:         {'TRAINABLE (requires_grad=True)' if res_b['freeze_status']['srm_trainable'] else 'FROZEN'}\n"
      f"  - Spatial Feature Fusion:     {'TRAINABLE (requires_grad=True)' if res_b['freeze_status']['fusion_trainable'] else 'FROZEN'}\n"
      f"  - Localization Decoder:       {'TRAINABLE (requires_grad=True)' if res_b['freeze_status']['decoder_trainable'] else 'FROZEN'}",
      flush=True,
  )

  print("=" * 85, flush=True)
  print("BENCHMARK COMPLETED SUCCESSFULLY. NO OPTIMIZER STEPS TAKEN.", flush=True)
  print("=" * 85, flush=True)


if __name__ == "__main__":
  main()
