"""
Training script for KL-VAE.

Usage:
    python train_vae.py --train
    python train_vae.py --test
    python train_vae.py --encode_all
"""

import os
import csv
import json
import argparse
import copy
import re
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib

from model_vae import KLVAE, PatchDiscriminator3D
from utils import seed_torch, save_checkpoint, load_checkpoint
import config

from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure
)


# ============================================================
# Dataset: loads all individual PET scans from subject folders
# ============================================================

def parse_month_from_name(filename):
    """Return integer month from names containing M00/M06/...; otherwise None."""
    session_match = re.search(r"ses-M(\d{1,3})", filename)
    if session_match:
        return int(session_match.group(1))
    match = re.search(r"M(\d{1,3})", filename)
    if match:
        return int(match.group(1))
    return None


def _format_months(months):
    if months is None:
        return "all"
    return ",".join(f"M{month:02d}" for month in months)


class PETScanDataset(Dataset):
    """
    Dataset that walks through data/{split}/{subject_id}/*.nii.gz
    and loads each individual scan as a training sample.
    """
    def __init__(self, data_root, split='train', allowed_months=None):
        self.scans = []
        self.month_counts = {}
        self.allowed_months = None if allowed_months is None else set(allowed_months)
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            print(f"WARNING: {split_dir} does not exist")
            return
        for subject in sorted(os.listdir(split_dir)):
            subj_dir = os.path.join(split_dir, subject)
            if not os.path.isdir(subj_dir):
                continue
            for f in sorted(os.listdir(subj_dir)):
                if f.endswith('.nii.gz') or f.endswith('.nii'):
                    month = parse_month_from_name(f)
                    if self.allowed_months is not None and month not in self.allowed_months:
                        continue
                    self.scans.append((os.path.join(subj_dir, f), month))
                    if month is not None:
                        self.month_counts[month] = self.month_counts.get(month, 0) + 1

        month_summary = " ".join(
            f"M{month:02d}:{count}" for month, count in sorted(self.month_counts.items())
        )
        print(
            f"[{split}] Found {len(self.scans)} scans "
            f"(months={_format_months(allowed_months)}"
            f"{'; ' + month_summary if month_summary else ''})"
        )

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        path, month = self.scans[idx]
        data = nib.load(path).get_fdata().astype(np.float32)
        data = data[np.newaxis, ...]  # (1, D, H, W)
        month_value = -1 if month is None else month
        return torch.from_numpy(data), os.path.basename(path), torch.tensor(month_value, dtype=torch.float32)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(pred, gt):
    """PSNR and SSIM with dynamic data range."""
    if pred.ndim == 4:
        pred = pred.unsqueeze(1)
    if gt.ndim == 4:
        gt = gt.unsqueeze(1)
    pred, gt = pred.float(), gt.float()

    combined = torch.cat([pred, gt], dim=0)
    dr = float(combined.max() - combined.min())
    if dr < 1e-6:
        dr = 2.0

    dims = tuple(range(1, pred.ndim))
    psnr_val = peak_signal_noise_ratio(pred, gt, data_range=dr, reduction='none', dim=dims)
    ssim_val = structural_similarity_index_measure(
        pred, gt, data_range=dr, reduction='none',
        gaussian_kernel=False, kernel_size=7
    )
    return psnr_val.sum().item(), ssim_val.sum().item()


# ============================================================
# Hinge GAN losses
# ============================================================

def hinge_d_loss(real_logits, fake_logits):
    return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())


def hinge_g_loss(fake_logits):
    return -fake_logits.mean()


def get_allowed_months(month_args):
    if month_args is None or len(month_args) == 0:
        return None
    return sorted(set(int(month) for month in month_args))


def get_time_channel_index(args):
    index = args.time_channel_index
    if index < 0:
        index = args.latent_ch - 1
    if index < 0 or index >= args.latent_ch:
        raise ValueError(
            f"time_channel_index={index} is invalid for latent_ch={args.latent_ch}"
        )
    return index


def month_targets(months, max_month, target_mode):
    valid = months >= 0
    targets = months.float() / float(max_month)
    if target_mode == "minus_one_one":
        targets = targets * 2.0 - 1.0
    return targets, valid


def time_channel_losses(mu, months, args):
    """Align one latent channel's spatial mean with visit month."""
    if args.time_channel_weight <= 0 and args.time_channel_var_weight <= 0:
        zero = mu.new_tensor(0.0)
        return zero, zero

    targets, valid = month_targets(months, args.max_month, args.time_channel_target)
    if not bool(valid.any()):
        zero = mu.new_tensor(0.0)
        return zero, zero

    channel_index = get_time_channel_index(args)
    channel = mu[:, channel_index].flatten(1)
    pred = channel.mean(dim=1)
    align_loss = F.mse_loss(pred[valid], targets[valid])
    var_loss = channel[valid].var(dim=1, unbiased=False).mean()
    return align_loss, var_loss


# ============================================================
# Training
# ============================================================

def train(args):
    print("=" * 60)
    print("Training KL-VAE")
    print("=" * 60)

    # Use cuda:0 (PBS assigns specific GPU via CUDA_VISIBLE_DEVICES)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    exp_dir = args.exp_dir or "result/klvae"
    os.makedirs(exp_dir, exist_ok=True)

    # ---- Model ----
    model = KLVAE(
        latent_ch=args.latent_ch,
        base_ch=args.base_ch,
        ch_mult=tuple(args.ch_mult),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout
    ).to(device)
    print(f"KL-VAE  latent_ch={args.latent_ch}  base_ch={args.base_ch}  ch_mult={args.ch_mult}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ---- Gradient checkpointing ----
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing: ENABLED")

    # ---- Discriminator (optional) ----
    use_gan = args.gan_weight > 0
    disc = None
    opt_disc = None
    if use_gan:
        disc = PatchDiscriminator3D(in_ch=1, base_ch=32, n_layers=3).to(device)
        opt_disc = optim.Adam(disc.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.9))
        disc_params = sum(p.numel() for p in disc.parameters())
        print(f"Discriminator parameters: {disc_params:,}")
        # Disc starts training after warmup
        gan_start_epoch = args.gan_start_epoch
    else:
        gan_start_epoch = 99999

    # ---- Optimizer ----
    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    # ---- Data ----
    data_root = args.data_root
    train_months = get_allowed_months(args.train_months)
    val_months = get_allowed_months(args.val_months)
    train_ds = PETScanDataset(data_root, 'train', allowed_months=train_months)
    val_ds = PETScanDataset(data_root, 'val', allowed_months=val_months)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    if len(train_ds) == 0:
        print("ERROR: No training data found!")
        return

    if args.time_channel_weight > 0 or args.time_channel_var_weight > 0:
        channel_index = get_time_channel_index(args)
        print(
            f"Time-channel constraint: channel={channel_index} "
            f"target={args.time_channel_target} "
            f"weight={args.time_channel_weight} "
            f"var_weight={args.time_channel_var_weight}"
        )

    # ---- Early stopping ----
    best_val_loss = float('inf')
    patience_counter = 0

    # ---- CSV log ----
    log_path = os.path.join(exp_dir, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Recon', 'Train_KL', 'Train_GAN',
                         'Train_Time', 'Train_TimeVar',
                         'Val_Recon', 'Val_PSNR', 'Val_SSIM',
                         'Val_Time', 'Val_TimeVar', 'LR'])

    # ---- Training loop ----
    for epoch in range(args.epochs):
        model.train()
        if disc is not None:
            disc.train()

        train_recon_sum = 0.0
        train_kl_sum = 0.0
        train_gan_sum = 0.0
        train_time_sum = 0.0
        train_time_var_sum = 0.0
        n_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_data, _, months in loop:
            x = batch_data.to(device)
            months = months.to(device)

            # ---- Forward ----
            recon, mu, logvar = model(x, sample_posterior=True)
            recon_loss = F.l1_loss(recon, x)
            kl_loss = KLVAE.kl_loss(mu, logvar)
            loss = recon_loss + args.kl_weight * kl_loss
            time_loss, time_var_loss = time_channel_losses(mu, months, args)
            if args.time_channel_weight > 0:
                loss = loss + args.time_channel_weight * time_loss
            if args.time_channel_var_weight > 0:
                loss = loss + args.time_channel_var_weight * time_var_loss
            train_kl_sum += kl_loss.item()
            train_time_sum += time_loss.item()
            train_time_var_sum += time_var_loss.item()

            train_recon_sum += recon_loss.item()

            # ---- GAN loss (generator) ----
            g_loss_val = 0.0
            if use_gan and epoch >= gan_start_epoch:
                fake_logits = disc(recon)
                g_loss = hinge_g_loss(fake_logits)
                loss = loss + args.gan_weight * g_loss
                g_loss_val = g_loss.item()
            train_gan_sum += g_loss_val

            # ---- Update model ----
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            # ---- Update discriminator ----
            if use_gan and epoch >= gan_start_epoch:
                with torch.no_grad():
                    recon_det, _, _ = model(x, sample_posterior=False)

                real_logits = disc(x)
                fake_logits = disc(recon_det.detach())
                d_loss = hinge_d_loss(real_logits, fake_logits)

                opt_disc.zero_grad()
                d_loss.backward()
                opt_disc.step()

            n_batches += 1

            # Postfix
            info = {'recon': recon_loss.item(), 'kl': kl_loss.item()}
            if args.time_channel_weight > 0 or args.time_channel_var_weight > 0:
                info['time'] = time_loss.item()
            loop.set_postfix(info)

            # Memory cleanup
            del x, recon, loss
            torch.cuda.empty_cache()

        scheduler.step()

        # ---- Validation ----
        model.eval()
        val_recon_sum = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_time_sum = 0.0
        val_time_var_sum = 0.0
        val_count = 0
        val_batches = 0

        with torch.no_grad():
            for batch_data, _, months in val_loader:
                x = batch_data.to(device)
                months = months.to(device)
                recon, mu, _ = model(x, sample_posterior=False)

                val_recon_sum += F.l1_loss(recon, x).item() * x.size(0)
                psnr_v, ssim_v = compute_metrics(recon, x)
                val_psnr_sum += psnr_v
                val_ssim_sum += ssim_v
                time_loss, time_var_loss = time_channel_losses(mu, months, args)
                val_time_sum += time_loss.item()
                val_time_var_sum += time_var_loss.item()
                val_count += x.size(0)
                val_batches += 1

        avg_val_recon = val_recon_sum / val_count
        avg_val_psnr = val_psnr_sum / val_count
        avg_val_ssim = val_ssim_sum / val_count
        avg_train_time = train_time_sum / max(n_batches, 1)
        avg_train_time_var = train_time_var_sum / max(n_batches, 1)
        avg_val_time = val_time_sum / max(val_batches, 1)
        avg_val_time_var = val_time_var_sum / max(val_batches, 1)

        # ---- Logging ----
        avg_lr = scheduler.get_last_lr()[0]
        log_msg = (f"Epoch {epoch+1}: Recon={train_recon_sum/n_batches:.6f} "
                   f"KL={train_kl_sum/n_batches:.6f} GAN={train_gan_sum/n_batches:.6f} "
                   f"Time={avg_train_time:.6f} | "
                   f"Val: Recon={avg_val_recon:.6f} PSNR={avg_val_psnr:.2f} SSIM={avg_val_ssim:.4f}")
        log_row = [epoch+1, train_recon_sum/n_batches, train_kl_sum/n_batches,
                   train_gan_sum/n_batches, avg_train_time, avg_train_time_var,
                   avg_val_recon, avg_val_psnr, avg_val_ssim,
                   avg_val_time, avg_val_time_var, avg_lr]

        print(log_msg)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(log_row)

        # ---- Checkpointing ----
        if avg_val_recon < best_val_loss or (epoch + 1) % 100 == 0:
            if avg_val_recon < best_val_loss:
                best_val_loss = avg_val_recon
                patience_counter = 0
                ckpt_path = os.path.join(exp_dir, "klvae_best.pth.tar")
                save_checkpoint(model, opt, filename=ckpt_path)
                print(f"  → Saved best model (val_recon={best_val_loss:.6f})")
            else:
                patience_counter += 1

            # Also save periodic checkpoint
            if (epoch + 1) % 100 == 0:
                ckpt_path = os.path.join(exp_dir, f"klvae_epoch{epoch+1}.pth.tar")
                save_checkpoint(model, opt, filename=ckpt_path)
        else:
            patience_counter += 1

        # ---- Early stopping ----
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={args.patience})")
            break

    print(f"\nTraining complete! Best val recon loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to {exp_dir}/")


# ============================================================
# Test
# ============================================================

def write_test_metrics(exp_dir, metrics):
    """Persist split-level reconstruction metrics for downstream aggregation."""
    output_dir = Path(exp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return output_path


def test(args):
    print("=" * 60)
    print("Testing KL-VAE")
    print("=" * 60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = KLVAE(latent_ch=args.latent_ch, base_ch=args.base_ch,
                  ch_mult=tuple(args.ch_mult),
                  num_res_blocks=args.num_res_blocks).to(device)

    ckpt = args.checkpoint
    if ckpt is None:
        exp_dir = args.exp_dir or "result/klvae"
        ckpt = os.path.join(exp_dir, "klvae_best.pth.tar")

    if os.path.exists(ckpt):
        load_checkpoint(ckpt, model, None, args.lr)
        print(f"Loaded from {ckpt}")
    else:
        print(f"ERROR: checkpoint not found at {ckpt}")
        return

    model.eval()

    import matplotlib.pyplot as plt
    import nibabel as nib

    split_metrics = {}

    # Test on val and test splits
    for split in ['val', 'test']:
        ds = PETScanDataset(args.data_root, split, allowed_months=get_allowed_months(args.test_months))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

        psnr_sum = 0.0
        ssim_sum = 0.0
        recon_sum = 0.0
        count = 0

        # Create visualization directory
        vis_dir = os.path.join(args.exp_dir if args.exp_dir else "result/klvae", "visualizations", split)
        os.makedirs(vis_dir, exist_ok=True)

        with torch.no_grad():
            for i, (batch_data, filename, _) in enumerate(tqdm(loader, desc=f"Testing [{split}]")):
                x = batch_data.to(device)
                recon, _, _ = model(x, sample_posterior=False)

                recon_sum += F.l1_loss(recon, x).item()
                psnr_v, ssim_v = compute_metrics(recon, x)
                psnr_sum += psnr_v
                ssim_sum += ssim_v

                # Save visualizations for the first 5 cases in each split
                if count < 5:
                    x_np = x.squeeze().cpu().numpy()
                    recon_np = recon.squeeze().cpu().numpy()

                    # Get middle slices across three axes
                    mid_z, mid_y, mid_x = [dim // 2 for dim in x_np.shape]

                    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
                    fig.suptitle(f"Case: {filename[0]} | PSNR: {psnr_v:.2f} | SSIM: {ssim_v:.4f}")

                    # Row 1: Axial
                    axes[0, 0].imshow(np.rot90(x_np[:, :, mid_x]), cmap='gray')
                    axes[0, 0].set_title('Original (Axial)')
                    axes[0, 0].axis('off')
                    axes[0, 1].imshow(np.rot90(recon_np[:, :, mid_x]), cmap='gray')
                    axes[0, 1].set_title('Reconstructed (Axial)')
                    axes[0, 1].axis('off')

                    # Row 2: Coronal
                    axes[1, 0].imshow(np.rot90(x_np[:, mid_y, :]), cmap='gray')
                    axes[1, 0].set_title('Original (Coronal)')
                    axes[1, 0].axis('off')
                    axes[1, 1].imshow(np.rot90(recon_np[:, mid_y, :]), cmap='gray')
                    axes[1, 1].set_title('Reconstructed (Coronal)')
                    axes[1, 1].axis('off')

                    # Row 3: Sagittal
                    axes[2, 0].imshow(np.rot90(x_np[mid_z, :, :]), cmap='gray')
                    axes[2, 0].set_title('Original (Sagittal)')
                    axes[2, 0].axis('off')
                    axes[2, 1].imshow(np.rot90(recon_np[mid_z, :, :]), cmap='gray')
                    axes[2, 1].set_title('Reconstructed (Sagittal)')
                    axes[2, 1].axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f"{filename[0].split('.')[0]}_vis.png"), dpi=150)
                    plt.close()

                count += 1

        if count == 0:
            raise RuntimeError(f"No scans found for VAE {split} evaluation")

        split_metrics[split] = {
            "n": count,
            "recon_l1": recon_sum / count,
            "psnr": psnr_sum / count,
            "ssim": ssim_sum / count,
        }
        print(f"\n[{split}] Results ({count} scans):")
        print(f"  Recon L1: {split_metrics[split]['recon_l1']:.6f}")
        print(f"  PSNR:     {split_metrics[split]['psnr']:.2f}")
        print(f"  SSIM:     {split_metrics[split]['ssim']:.4f}")

    metrics_path = write_test_metrics(args.exp_dir or "result/klvae", split_metrics)
    print(f"Saved VAE test metrics to {metrics_path}")


# ============================================================
# Encode all data to latent space
# ============================================================

def encode_all(args):
    """Encode all PET data (train/val/test) to latent using trained KL-VAE."""
    print("=" * 60)
    print("Encoding all data with KL-VAE")
    print("=" * 60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = KLVAE(latent_ch=args.latent_ch, base_ch=args.base_ch,
                  ch_mult=tuple(args.ch_mult),
                  num_res_blocks=args.num_res_blocks).to(device)

    ckpt = args.checkpoint
    if ckpt is None:
        exp_dir = args.exp_dir or "result/klvae"
        ckpt = os.path.join(exp_dir, "klvae_best.pth.tar")

    if os.path.exists(ckpt):
        load_checkpoint(ckpt, model, None, args.lr)
        print(f"Loaded from {ckpt}")
    else:
        print(f"ERROR: checkpoint not found at {ckpt}")
        return

    model.eval()

    # Reference image for affine
    ref_path = config.path
    if os.path.exists(ref_path):
        ref_affine = nib.load(ref_path).affine
    else:
        ref_affine = np.eye(4)
        print("WARNING: Reference image not found, using identity affine")

    output_base = args.output_dir or "./data/latent_klvae"

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(args.data_root, split)
        if not os.path.exists(split_dir):
            print(f"[{split}] not found, skipping")
            continue

        out_split = os.path.join(output_base, split)

        subjects = sorted([d for d in os.listdir(split_dir)
                          if os.path.isdir(os.path.join(split_dir, d))])

        subject_count = 0
        scan_count = 0

        for subject in tqdm(subjects, desc=f"Encoding [{split}]"):
            subj_dir = os.path.join(split_dir, subject)
            out_subj = os.path.join(out_split, subject)
            os.makedirs(out_subj, exist_ok=True)

            scans = sorted([f for f in os.listdir(subj_dir)
                           if f.endswith('.nii.gz') or f.endswith('.nii')])

            for scan_file in scans:
                scan_path = os.path.join(subj_dir, scan_file)
                data = nib.load(scan_path).get_fdata().astype(np.float32)
                x = torch.from_numpy(data[np.newaxis, np.newaxis, ...]).to(device)

                with torch.no_grad():
                    z = model.get_latent(x)  # (1, C, D, H, W)

                # Save: keep subject folder structure
                # Latent shape: (C, D, H, W) — save as 4D NIfTI
                z_np = z.squeeze(0).cpu().numpy()  # (C, D, H, W)

                # For compatibility with existing pipeline,
                # if latent_ch > 1, save as 4D NIfTI (D, H, W, C)
                if z_np.shape[0] > 1:
                    z_np = z_np.transpose(1, 2, 3, 0)  # (D, H, W, C)

                # Map scan name: keep same name for compatibility
                out_name = scan_file.replace('.nii.gz', '.nii.gz').replace('.nii', '.nii.gz')
                # Use a simpler naming: subject_ses-Mxx.nii.gz
                # Extract month from filename
                month = parse_month_from_name(scan_file)
                if month is not None:
                    out_name = f"{subject}_ses-M{month:02d}.nii.gz"
                else:
                    out_name = scan_file

                out_path = os.path.join(out_subj, out_name)
                nifti = nib.Nifti1Image(z_np, ref_affine)
                nib.save(nifti, out_path)
                scan_count += 1

            subject_count += 1

        print(f"  [{split}] Encoded {scan_count} scans from {subject_count} subjects → {out_split}")

    print(f"\nAll data encoded to {output_base}/")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    seed_torch()

    parser = argparse.ArgumentParser(description="KL-VAE Training for PET")

    # Mode
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--test', action='store_true', help='Test model')
    parser.add_argument('--encode_all', action='store_true', help='Encode all data to latent')

    # Model
    parser.add_argument('--latent_ch', type=int, default=4, help='Latent channels')
    parser.add_argument('--base_ch', type=int, default=32, help='Base channel width')
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Channel multipliers per level (e.g. 1 2 4 8)')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='ResBlocks per level')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')

    # KL-VAE specific
    parser.add_argument('--kl_weight', type=float, default=1e-6, help='KL loss weight')
    parser.add_argument('--time_channel_weight', type=float, default=0.0,
                        help='Weight for latent time-channel mean supervision (0 disables)')
    parser.add_argument('--time_channel_var_weight', type=float, default=0.0,
                        help='Weight for spatial variance penalty on the time channel')
    parser.add_argument('--time_channel_index', type=int, default=-1,
                        help='Latent channel to reserve for month supervision (-1 = last channel)')
    parser.add_argument('--time_channel_target', type=str, default='zero_one',
                        choices=['zero_one', 'minus_one_one'],
                        help='Month target scaling for time-channel supervision')
    parser.add_argument('--max_month', type=float, default=24.0,
                        help='Maximum month used to normalize visit months')

    # GAN
    parser.add_argument('--gan_weight', type=float, default=0.0,
                        help='GAN loss weight (0 to disable)')
    parser.add_argument('--gan_start_epoch', type=int, default=50,
                        help='Epoch to start GAN training')

    # Training
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save GPU memory')

    # Data
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root dir with train/val/test subdirs')
    parser.add_argument('--train_months', type=int, nargs='*', default=[0, 24],
                        help='Optional month filter for training scans, e.g. --train_months 0 24')
    parser.add_argument('--val_months', type=int, nargs='*', default=[0, 24],
                        help='Optional month filter for validation scans')
    parser.add_argument('--test_months', type=int, nargs='*', default=[0, 24],
                        help='Optional month filter for test/visualization scans')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir for encoded latents')
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='Experiment directory (default: result/klvae)')

    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.test:
        test(args)
    elif args.encode_all:
        encode_all(args)
    else:
        parser.print_help()
