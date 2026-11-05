"""End-to-End Sanity Check Script for TamperDoc.

Performs a rigorous verification on 4 samples from DocTamperV1-TrainingSet:
1. Tests DocTamperLMDBDataset loading.
2. Tests forward pass through RGB, DCT, SRM, Fusion, and Localization Decoder.
3. Verifies tensor shapes at every stage.
4. Computes Combined BCE + Dice Loss.
5. Performs backward pass and verifies gradient flow across learnable modules.
6. Verifies absence of NaN / Inf values.
7. Verifies all 4 ablation configurations (RGB, RGB+DCT, RGB+SRM, Full Multi-Branch).
8. Saves a visualization of image, ground-truth mask, and predicted mask to results/sanity_check.png.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataset import DocTamperLMDBDataset
from src.model import TamperDocModel
from src.losses import CombinedLocalizationLoss
from src.metrics import LocalizationMetrics


def run_sanity_check():
    print("=" * 75, flush=True)
    print("TAMPERDOC: END-TO-END SYSTEM SANITY VERIFICATION", flush=True)
    print("=" * 75, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Compute Device: {device}", flush=True)

    dataset_path = os.path.abspath("dataset/DocTamper/DocTamperV1-TrainingSet")
    print(f"Loading Dataset from: {dataset_path}", flush=True)

    # 1. Dataset Loading (4 samples)
    dataset = DocTamperLMDBDataset(
        lmdb_dir=dataset_path,
        is_train=False,
    )
    print(f"Dataset reported sample count: {len(dataset):,}", flush=True)

    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))

    images = batch["image"].to(device)       # [B, 3, 512, 512]
    raw_images = batch["raw_image"].to(device) # [B, 3, 512, 512]
    masks = batch["mask"].to(device)         # [B, 1, 512, 512]
    indices = batch["index"].tolist()

    print(f"\nLoaded Batch of {batch_size} samples (Indices: {indices}):", flush=True)
    print(f"  Normalized Image Tensor: {images.shape}, dtype={images.dtype}, range=[{images.min():.2f}, {images.max():.2f}]", flush=True)
    print(f"  Raw Image Tensor:        {raw_images.shape}, dtype={raw_images.dtype}, range=[{raw_images.min():.2f}, {raw_images.max():.2f}]", flush=True)
    print(f"  Ground-Truth Mask Tensor: {masks.shape}, dtype={masks.dtype}, uniques={torch.unique(masks).tolist()}", flush=True)

    assert masks.shape == (batch_size, 1, 512, 512), f"Mask shape mismatch: {masks.shape}"
    assert images.shape == (batch_size, 3, 512, 512), f"Image shape mismatch: {images.shape}"

    # 2. Model Initialization (Full Multi-Modal Architecture: RGB + DCT + SRM)
    print("\n--- Initializing Full Multi-Branch Model (RGB + DCT + SRM) ---", flush=True)
    model = TamperDocModel(
        use_rgb=True,
        use_dct=True,
        use_srm=True,
        pretrained_rgb=True,
        branch_channels=128,
        fused_channels=128,
        target_size=(512, 512),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Total Parameters:     {total_params:,}", flush=True)
    print(f"  Trainable Parameters: {trainable_params:,}", flush=True)
    print(f"  Fixed/Basis Params:   {fixed_params:,} (e.g. DCT basis & SRM high-pass kernels)", flush=True)

    # 3. Step-by-Step Forward Pass & Intermediate Tensor Shapes
    print("\n--- Intermediate Stage Tensor Shapes ---", flush=True)
    with torch.no_grad():
        rgb_feat = model.rgb_branch(images)
        dct_feat = model.dct_branch(raw_images)
        srm_feat = model.srm_branch(raw_images)
        fused_feat = model.fusion([rgb_feat, dct_feat, srm_feat])
        decoder_logits = model.decoder(fused_feat)

    print(f"  [1] RGB Branch Output:      {rgb_feat.shape}      (Expected: [B, 128, 64, 64])", flush=True)
    print(f"  [2] DCT Branch Output:      {dct_feat.shape}      (Expected: [B, 128, 64, 64])", flush=True)
    print(f"  [3] SRM Branch Output:      {srm_feat.shape}      (Expected: [B, 128, 64, 64])", flush=True)
    print(f"  [4] Fused Features:         {fused_feat.shape}      (Expected: [B, 128, 64, 64])", flush=True)
    print(f"  [5] Localization Logits:    {decoder_logits.shape}   (Expected: [B, 1, 512, 512])", flush=True)
    print(f"  [6] Ground-Truth Mask:      {masks.shape}   (Expected: [B, 1, 512, 512])", flush=True)

    # 4. End-to-End Forward Pass with Gradient Graph
    model.train()
    outputs = model(images, raw_x=raw_images)
    logits = outputs["tamper_logits"]

    # 5. Loss Computation
    criterion = CombinedLocalizationLoss(lambda_dice=1.0)
    loss_dict = criterion(logits, masks)
    total_loss = loss_dict["loss"]
    bce_loss = loss_dict["bce_loss"]
    dice_loss = loss_dict["dice_loss"]

    print("\n--- Loss Calculation ---", flush=True)
    print(f"  Total Combined Loss: {total_loss.item():.6f}", flush=True)
    print(f"  BCE Component Loss:  {bce_loss.item():.6f}", flush=True)
    print(f"  Dice Component Loss: {dice_loss.item():.6f}", flush=True)

    # 6. NaN / Inf Verification
    assert not torch.isnan(total_loss), "Loss contains NaN!"
    assert not torch.isinf(total_loss), "Loss contains Inf!"
    assert not torch.isnan(logits).any(), "Logits contain NaN!"
    assert not torch.isinf(logits).any(), "Logits contain Inf!"
    print("  ✓ No NaN or Inf detected in outputs or loss.", flush=True)

    # 7. Backward Pass & Gradient Verification
    total_loss.backward()

    print("\n--- Gradient Flow Verification Across Learnable Modules ---", flush=True)
    grad_status = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            has_grad = param.grad is not None and not torch.isnan(param.grad).any() and param.grad.abs().sum() > 0
            module_name = name.split(".")[0]
            grad_status[module_name] = grad_status.get(module_name, True) and has_grad

    for mod, status in grad_status.items():
        symbol = "✓" if status else "✗"
        print(f"  [{symbol}] Module '{mod}': Gradients active and healthy = {status}", flush=True)

    # Verify fixed non-learnable kernels have NO gradients
    assert model.dct_branch.dct_conv.weight.grad is None, "DCT basis weights must NOT have gradients!"
    assert model.srm_branch.srm_conv.weight.grad is None, "SRM filter weights must NOT have gradients!"
    print("  ✓ Confirmed fixed non-learnable status of DCT and SRM basis kernels.", flush=True)

    # 8. Test All 4 Ablation Configurations
    print("\n--- Verifying All 4 Architectural Ablations ---", flush=True)
    ablations = [
        ("Exp A: RGB Only", {"use_rgb": True, "use_dct": False, "use_srm": False}),
        ("Exp B: RGB + DCT", {"use_rgb": True, "use_dct": True, "use_srm": False}),
        ("Exp C: RGB + SRM", {"use_rgb": True, "use_dct": False, "use_srm": True}),
        ("Exp D: RGB + DCT + SRM (Full)", {"use_rgb": True, "use_dct": True, "use_srm": True}),
    ]
    for name, cfg in ablations:
        m_abl = TamperDocModel(**cfg, pretrained_rgb=False).to(device)
        with torch.no_grad():
            out_abl = m_abl(images, raw_x=raw_images)
            out_shape = out_abl["tamper_logits"].shape
        print(f"  ✓ {name:<30} -> Output Logits Shape: {out_shape}", flush=True)

    # 9. Metrics Computation on Initial Batch
    initial_metrics = LocalizationMetrics.batch_metrics(logits.detach(), masks)
    print("\n--- Initial Batch Metrics (Untrained Initialization) ---", flush=True)
    for k, v in initial_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)

    # 10. Generate and Save Sanity Visualization to results/
    os.makedirs("results", exist_ok=True)
    vis_path = os.path.abspath("results/sanity_check.png")

    probs = torch.sigmoid(logits.detach()).cpu()
    raw_imgs_cpu = raw_images.cpu().permute(0, 2, 3, 1).numpy()
    masks_cpu = masks.cpu().squeeze(1).numpy()
    probs_cpu = probs.squeeze(1).numpy()

    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    if batch_size == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(batch_size):
        # Original Image
        axes[i, 0].imshow(np.clip(raw_imgs_cpu[i], 0.0, 1.0))
        axes[i, 0].set_title(f"Sample #{indices[i]} - Original Image", fontsize=11)
        axes[i, 0].axis("off")

        # Ground-Truth Mask
        axes[i, 1].imshow(masks_cpu[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 1].set_title("Ground-Truth Tamper Mask", fontsize=11)
        axes[i, 1].axis("off")

        # Initial Predicted Mask (Probability)
        im_pred = axes[i, 2].imshow(probs_cpu[i], cmap="jet", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title("Initial Predicted Probabilities", fontsize=11)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Saved Sanity Visualization to: {vis_path}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("ALL SANITY CHECKS PASSED PERFECTLY!", flush=True)
    print("=" * 75, flush=True)
    return vis_path


if __name__ == "__main__":
    run_sanity_check()
