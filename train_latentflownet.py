"""Train and evaluate LatentFlowNet on frozen KL-VAE latent representations.

Training uses endpoint pairs and available intermediate visits. Evaluation
reports latent, decoded-image, and raw-image metrics when inputs are supplied.
Use predict.py for baseline-only deployment predictions.
"""

import os
import csv
import time
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import nibabel as nib

# Local imports
from latentflownet import (create_model, longitudinal_contrastive_loss,
                             adaptive_hinge_loss)
from dataset_longitudinal import (
    LongitudinalPETDataset,
    collate_longitudinal,
)
from model_vae import KLVAE
from utils import seed_torch

# Metrics
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr




# ============================================================
# Helpers
# ============================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path, model, optimizer=None):
    """Load model (and optionally optimizer) from checkpoint."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'state' in ckpt:
        model.load_state_dict(ckpt['state'])
    else:
        model.load_state_dict(ckpt)

    epoch = ckpt.get('epoch', 0)
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return epoch


def save_checkpoint(path, model, optimizer, epoch, metrics=None):
    """Save checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics or {},
    }, path)


def _extract_state_dict(ckpt):
    if 'state' in ckpt:
        return ckpt['state']
    if 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    if 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt








def load_obs_table(obs_file):
    if obs_file is None or not os.path.exists(obs_file):
        return {}, 0

    import pandas as pd

    df = pd.read_csv(obs_file, dtype={'subject_id': str})
    if 'subject_id' not in df.columns:
        raise ValueError(f"obs_file must contain subject_id column: {obs_file}")

    feature_cols = [col for col in df.columns if col != 'subject_id']
    obs_map = {}
    for row in df.to_dict(orient='records'):
        subject_id = str(row['subject_id']).strip()
        obs_map[subject_id] = np.asarray([row[col] for col in feature_cols], dtype=np.float32)
    return obs_map, len(feature_cols)


def get_obs_batch(subject_ids, obs_map, obs_dim, device):
    if obs_dim <= 0:
        return None, None

    obs_batch = []
    obs_mask = []
    default_obs = np.zeros((obs_dim,), dtype=np.float32)
    for subject_id in subject_ids:
        obs = obs_map.get(str(subject_id).strip())
        if obs is None:
            obs_batch.append(default_obs)
            obs_mask.append([0.0])
        else:
            obs_batch.append(obs)
            obs_mask.append([1.0])

    obs_tensor = torch.from_numpy(np.stack(obs_batch, axis=0)).to(device)
    obs_mask_tensor = torch.tensor(obs_mask, dtype=torch.float32, device=device)
    return obs_tensor, obs_mask_tensor


def load_vae(vae_checkpoint, latent_ch, device, obs_file=None, ch_mult=(1, 2, 4, 8)):
    """Load the frozen KL-VAE used to encode and decode PET images."""
    ckpt = torch.load(vae_checkpoint, map_location='cpu', weights_only=False)
    state_dict = _extract_state_dict(ckpt)
    vae = KLVAE(latent_ch=latent_ch, base_ch=32, ch_mult=ch_mult, num_res_blocks=2).to(device)
    vae.load_state_dict(state_dict)
    print(f"Loaded KL-VAE from {vae_checkpoint}")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    obs_map, _ = load_obs_table(obs_file)
    return vae, 'klvae', obs_map, 0


def decode_latent(vae, z, obs=None, obs_mask=None):
    """Decode latent z using the frozen KL-VAE."""
    with torch.no_grad():
        return vae.decode(z)


def compute_image_metrics(pred_img, gt_img):
    """Compute PSNR and SSIM in image space."""
    pred = pred_img.cpu().numpy().squeeze()
    gt = gt_img.cpu().numpy().squeeze()

    data_range = gt.max() - gt.min()
    if data_range < 1e-8:
        return 0.0, 0.0

    psnr = compare_psnr(gt, pred, data_range=data_range)
    ssim = compare_ssim(gt, pred, data_range=data_range)
    return psnr, ssim


def compute_raw_image_metrics(pred_img, gt_img, baseline_img=None):
    """Compute forecast and identity metrics against an original raw target."""
    metrics = {
        'psnr_raw': None,
        'ssim_raw': None,
        'mae_raw': None,
        'psnr_identity_raw': None,
        'ssim_identity_raw': None,
        'mae_identity_raw': None,
    }
    if gt_img is None:
        return metrics

    data_range = float(gt_img.max() - gt_img.min())
    if data_range <= 1e-8:
        return metrics

    metrics.update({
        'psnr_raw': compare_psnr(gt_img, pred_img, data_range=data_range),
        'ssim_raw': compare_ssim(gt_img, pred_img, data_range=data_range),
        'mae_raw': float(np.mean(np.abs(pred_img - gt_img))),
    })
    if baseline_img is not None:
        metrics.update({
            'psnr_identity_raw': compare_psnr(gt_img, baseline_img, data_range=data_range),
            'ssim_identity_raw': compare_ssim(gt_img, baseline_img, data_range=data_range),
            'mae_identity_raw': float(np.mean(np.abs(baseline_img - gt_img))),
        })
    return metrics


def load_raw_pet_image(raw_root, subject_id, month):
    subject_dir = os.path.join(raw_root, subject_id)
    if not os.path.isdir(subject_dir):
        return None

    session = f"M{month:02d}"
    filenames = sorted(os.listdir(subject_dir))
    preferred = [filename for filename in filenames if f"ses-{session}" in filename]
    for filename in preferred or filenames:
        if not (filename.endswith('.nii.gz') or filename.endswith('.nii')):
            continue
        if session not in filename:
            continue
        data = nib.load(os.path.join(subject_dir, filename)).get_fdata().astype(np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        return data
    return None


def align_volume_to_shape(volume, target_shape):
    if volume is None:
        return None
    if tuple(volume.shape) == tuple(target_shape):
        return volume

    slices = []
    for size, target in zip(volume.shape, target_shape):
        if size < target:
            return None
        start = (size - target) // 2
        slices.append(slice(start, start + target))
    return volume[tuple(slices)]


def _safe_mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def write_test_results(results, exp_dir, vae_type):
    """Report prediction and identity metrics against matching targets."""
    mean = lambda key: _safe_mean([row.get(key) for row in results])
    print(f"\n{'='*60}")
    print(f"Test Results: LatentFlowNet + {vae_type.upper()}")
    print(f"{'='*60}")
    print(f"  N subjects: {len(results)}")
    print(f"  Decoded target: PSNR={mean('psnr'):.2f} SSIM={mean('ssim'):.4f} MAE={mean('mae'):.5f}")
    print(f"  Decoded identity: PSNR={mean('psnr_identity_decoded'):.2f} SSIM={mean('ssim_identity_decoded'):.4f}")
    raw_pairs = [row for row in results if all(row.get(key) is not None for key in (
        'psnr_raw', 'ssim_raw', 'mae_raw', 'psnr_identity_raw',
        'ssim_identity_raw', 'mae_identity_raw',
    ))]
    if raw_pairs:
        raw_mean = lambda key: float(np.mean([row[key] for row in raw_pairs]))
        print(f"  Raw target (N={len(raw_pairs)}): PSNR={raw_mean('psnr_raw'):.2f} SSIM={raw_mean('ssim_raw'):.4f} MAE={raw_mean('mae_raw'):.5f}")
        print(f"  Raw identity: PSNR={raw_mean('psnr_identity_raw'):.2f} SSIM={raw_mean('ssim_identity_raw'):.4f} MAE={raw_mean('mae_identity_raw'):.5f}")
        print(f"  Raw improvement over identity: PSNR={raw_mean('psnr_raw') - raw_mean('psnr_identity_raw'):+.2f} dB SSIM={raw_mean('ssim_raw') - raw_mean('ssim_identity_raw'):+.4f}")
    else:
        print(f"  Decoded improvement over identity: PSNR={mean('psnr') - mean('psnr_identity_decoded'):+.2f} dB SSIM={mean('ssim') - mean('ssim_identity_decoded'):+.4f}")
    print(f"  Delta predicted: {mean('delta_pred'):.4f} | Delta target: {mean('delta_gt'):.4f}")
    out_csv = os.path.join(exp_dir, 'test_results.csv')
    with open(out_csv, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results: {out_csv}")


def _model_kwargs(args, batch, device):
    return {
        'obs': batch['obs'].float().to(device),
        'obs_mask': batch['obs_mask'].float().to(device),
    }


def _build_observations(args, batch, device):
    if args.no_intermediates:
        return None

    batch_intermediates = batch.get('intermediates')
    if not batch_intermediates:
        return None

    target_time = getattr(args, 'target_time', 24)

    months = sorted({
        month
        for sample_inters in batch_intermediates
        for month in sample_inters.keys()
        if month != target_time
    })
    obs_dict = {}
    for month in months:
        template = next(
            (sample_inters[month] for sample_inters in batch_intermediates if month in sample_inters),
            None,
        )
        if template is None:
            continue

        zeros = torch.zeros_like(template, dtype=torch.float32, device=device)
        z_batch = []
        mask = []
        for sample_inters in batch_intermediates:
            if month in sample_inters:
                z_batch.append(sample_inters[month].float().to(device))
                mask.append(1.0)
            else:
                z_batch.append(zeros.clone())
                mask.append(0.0)

        obs_dict[month] = {
            'z': torch.stack(z_batch, dim=0),
            'mask': torch.tensor(mask, dtype=torch.float32, device=device).view(-1, 1, 1, 1, 1),
        }
    return obs_dict or None


# ============================================================
# Training
# ============================================================

def train(args):
    print("=" * 70)
    print(f"Training: LatentFlowNet with KL-VAE latents")
    print(f"Residual prediction: ON")
    print(f"Intermediate supervision: {'OFF' if args.no_intermediates else 'ON'}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Data ----
    latent_dir = args.data_root or "./data/latent_klvae"
    print(f"Latent data: {latent_dir}")

    train_dataset = LongitudinalPETDataset(
        data_root=os.path.join(latent_dir, 'train'),
        label_file=args.label_file,
        obs_file=args.obs_file,
        baseline_time=args.baseline_time,
        target_time=args.target_time,
        unknown_label=args.unknown_label,
        stage='train',
    )
    val_dataset = LongitudinalPETDataset(
        data_root=os.path.join(latent_dir, 'val'),
        label_file=args.label_file,
        obs_file=args.obs_file,
        baseline_time=args.baseline_time,
        target_time=args.target_time,
        unknown_label=args.unknown_label,
        stage='val',
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_longitudinal,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_longitudinal,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} pairs | Val: {len(val_dataset)} pairs")

    # ---- Model ----
    model = create_model(
        method=args.method,
        latent_ch=args.latent_ch,
        hidden_ch=args.hidden_ch,
        time_dim=args.time_dim,
        num_blocks=args.num_blocks,
        num_classes=args.num_classes,
        dropout=args.dropout,
        n_steps=args.n_steps,
        obs_dim=getattr(train_dataset, 'obs_dim', 0),
        cond_dim=args.cond_dim,
        lambda_s=args.lambda_s,
        lambda_c=args.lambda_c,
        lambda_m=args.lambda_m,
        inter_weight=args.inter_weight,
        residual_weight=args.residual_weight,
        recon_weight=args.recon_weight,
        delta_mid_weight=args.delta_mid_weight,
        target_time=args.target_time,
    ).to(device)

    print(f"Model: LatentFlowNet | Params: {count_params(model):,}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- Load VAE for validation decoding ----
    vae = None
    if args.vae_checkpoint and os.path.exists(args.vae_checkpoint):
        vae, vae_type, obs_map, obs_dim = load_vae(
            args.vae_checkpoint, args.latent_ch, device, obs_file=args.obs_file
        )
        print(f"VAE loaded for image-space validation")

    # ---- Resume ----
    start_epoch = 0
    best_val_loss = float('inf')
    exp_dir = args.exp_dir or "result/latentflownet"
    os.makedirs(exp_dir, exist_ok=True)

    ckpt_path = os.path.join(exp_dir, 'latest.pth.tar')
    if args.resume and os.path.exists(ckpt_path):
        start_epoch = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resumed from epoch {start_epoch}")

    # ---- Logging ----
    log_path = os.path.join(exp_dir, 'training_log.csv')
    log_exists = os.path.exists(log_path) and args.resume
    log_file = open(log_path, 'a' if log_exists else 'w', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            'epoch', 'lr', 'train_loss', 'train_residual', 'train_recon',
            'val_loss', 'val_residual', 'val_recon',
            'val_psnr_latent', 'val_ssim_latent',
            'val_psnr_image', 'val_ssim_image',
            'delta_pred_norm', 'delta_gt_norm', 'time_min',
        ])

    # ---- Training loop ----
    patience_counter = 0
    epoch_start = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        train_metrics = {}
        n_batches = 0

        for batch in train_loader:
            z_0 = batch['z_0'].float().to(device)
            z_T = batch['z_T'].float().to(device)
            labels = batch['labels'].to(device) if args.num_classes > 0 else None

            obs_dict = _build_observations(args, batch, device)
            dyn_kwargs = _model_kwargs(args, batch, device)

            optimizer.zero_grad()
            loss, metrics = model.compute_loss(z_0, z_T, labels, obs_dict, **dyn_kwargs)

            # ---- Extra anti-identity losses (Delta/Contrastive/Hinge) ----
            extra_losses_active = (args.delta_weight > 0 or args.use_contrastive or args.use_hinge)
            if extra_losses_active:
                # Need z_T_pred for these losses
                z_T_pred = model.predict(z_0, labels, **dyn_kwargs)
                delta_pred = z_T_pred - z_0
                delta_gt = z_T - z_0

                # High-weight Delta MSE supervision
                if args.delta_weight > 0:
                    delta_mse = F.mse_loss(delta_pred, delta_gt)
                    loss = loss + args.delta_weight * delta_mse
                    metrics['delta_mse'] = delta_mse.item()

                # Longitudinal contrastive (InfoNCE)
                if args.use_contrastive:
                    contrast = longitudinal_contrastive_loss(z_T_pred, z_T, z_0)
                    loss = loss + 1.0 * contrast
                    metrics['contrast'] = contrast.item()

                # Adaptive hinge boundary
                if args.use_hinge:
                    hinge = adaptive_hinge_loss(z_T_pred, z_T, z_0, alpha=0.85)
                    loss = loss + 5.0 * hinge
                    metrics['hinge'] = hinge.item()

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in metrics.items():
                train_metrics[k] = train_metrics.get(k, 0.0) + v
            n_batches += 1

            # Free GPU memory
            if extra_losses_active:
                del z_0, z_T, loss, z_T_pred
            else:
                del z_0, z_T, loss
            torch.cuda.empty_cache()

        scheduler.step()

        # Average training metrics
        for k in train_metrics:
            train_metrics[k] /= max(n_batches, 1)

        # ---- Validation ----
        model.eval()
        val_metrics = {}
        val_psnr_latent = []
        val_ssim_latent = []
        val_psnr_image = []
        val_ssim_image = []
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                z_0 = batch['z_0'].float().to(device)
                z_T = batch['z_T'].float().to(device)
                labels = batch['labels'].to(device) if args.num_classes > 0 else None
                obs_dict = _build_observations(args, batch, device)
                dyn_kwargs = _model_kwargs(args, batch, device)

                loss, metrics = model.compute_loss(z_0, z_T, labels, obs_dict, **dyn_kwargs)
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v
                n_val += 1

                # Predict
                z_T_pred = model.predict(z_0, labels, **dyn_kwargs)

                # Latent-space metrics
                for b in range(z_0.shape[0]):
                    pred_np = z_T_pred[b].cpu().numpy()
                    gt_np = z_T[b].cpu().numpy()
                    dr = gt_np.max() - gt_np.min()
                    if dr > 1e-8:
                        val_psnr_latent.append(compare_psnr(gt_np, pred_np, data_range=dr))
                        val_ssim_latent.append(compare_ssim(
                            gt_np.squeeze(), pred_np.squeeze(),
                            data_range=dr,
                            channel_axis=0 if gt_np.shape[0] > 1 else None
                        ))

                # Image-space metrics (if VAE available, skip early epochs to save memory)
                if vae is not None and epoch % args.image_eval_every == 0:
                    try:
                        subject_ids = batch['subject_ids']
                        obs, obs_mask = get_obs_batch(subject_ids, obs_map, obs_dim, device)
                        x_pred = decode_latent(vae, z_T_pred, obs=obs, obs_mask=obs_mask)
                        x_gt = decode_latent(vae, z_T, obs=obs, obs_mask=obs_mask)
                        for b in range(x_pred.shape[0]):
                            psnr, ssim = compute_image_metrics(x_pred[b], x_gt[b])
                            val_psnr_image.append(psnr)
                            val_ssim_image.append(ssim)
                        del x_pred, x_gt
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            torch.cuda.empty_cache()
                            print(f"  [OOM during image decode, skipping image metrics]")
                        else:
                            raise

        torch.cuda.empty_cache()

        for k in val_metrics:
            val_metrics[k] /= max(n_val, 1)

        avg_val_psnr_l = np.mean(val_psnr_latent) if val_psnr_latent else 0.0
        avg_val_ssim_l = np.mean(val_ssim_latent) if val_ssim_latent else 0.0
        avg_val_psnr_i = np.mean(val_psnr_image) if val_psnr_image else 0.0
        avg_val_ssim_i = np.mean(val_ssim_image) if val_ssim_image else 0.0

        elapsed = (time.time() - epoch_start) / 60.0
        epoch_start = time.time()

        # ---- Logging ----
        lr = optimizer.param_groups[0]['lr']
        log_writer.writerow([
            epoch, f"{lr:.6f}",
            f"{train_metrics.get('loss', 0):.6f}",
            f"{train_metrics.get('residual', 0):.6f}",
            f"{train_metrics.get('recon', 0):.6f}",
            f"{val_metrics.get('loss', 0):.6f}",
            f"{val_metrics.get('residual', 0):.6f}",
            f"{val_metrics.get('recon', 0):.6f}",
            f"{avg_val_psnr_l:.2f}", f"{avg_val_ssim_l:.4f}",
            f"{avg_val_psnr_i:.2f}", f"{avg_val_ssim_i:.4f}",
            f"{train_metrics.get('delta_pred_norm', 0):.4f}",
            f"{train_metrics.get('delta_gt_norm', 0):.4f}",
            f"{elapsed:.1f}",
        ])
        log_file.flush()

        is_best = val_metrics.get('loss', float('inf')) < best_val_loss
        if is_best:
            best_val_loss = val_metrics.get('loss', float('inf'))
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        delta_info = ""
        dp = train_metrics.get('delta_pred_norm', 0)
        dg = train_metrics.get('delta_gt_norm', 0)
        if dg > 0:
            delta_info = f" | Δpred/Δgt={dp/dg:.3f}"

        extra = (
            f" | CL={train_metrics.get('contrastive', 0):.4f}"
            f" SM={train_metrics.get('smoothness', 0):.4f}"
            f" MN={train_metrics.get('monotonic', 0):.4f}"
            f" IN={train_metrics.get('inter', 0):.4f}"
        )

        # Append extra loss info if active
        if args.delta_weight > 0:
            extra += f" δW={train_metrics.get('delta_mse', 0):.4f}"
        if args.use_contrastive:
            extra += f" CL={train_metrics.get('contrast', 0):.4f}"
        if args.use_hinge:
            extra += f" HG={train_metrics.get('hinge', 0):.4f}"

        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"Train: loss={train_metrics.get('loss', 0):.5f} "
            f"res={train_metrics.get('residual', 0):.5f} "
            f"recon={train_metrics.get('recon', 0):.5f}{extra}{delta_info} | "
            f"Val: loss={val_metrics.get('loss', 0):.5f} "
            f"PSNR_l={avg_val_psnr_l:.2f} "
            f"SSIM_l={avg_val_ssim_l:.4f} "
            f"{'PSNR_i=' + f'{avg_val_psnr_i:.2f} ' if avg_val_psnr_i > 0 else ''}"
            f"{'★ ' if is_best else ''}"
            f"lr={lr:.6f} ({elapsed:.1f}min)"
        )

        # Save checkpoints
        save_checkpoint(
            os.path.join(exp_dir, 'latest.pth.tar'),
            model, optimizer, epoch, val_metrics
        )
        if is_best:
            save_checkpoint(
                os.path.join(exp_dir, 'best.pth.tar'),
                model, optimizer, epoch, val_metrics
            )

        if epoch % 50 == 0:
            save_checkpoint(
                os.path.join(exp_dir, f'epoch_{epoch}.pth.tar'),
                model, optimizer, epoch, val_metrics
            )

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    log_file.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to {exp_dir}/")


# ============================================================
# Testing (Full Evaluation)
# ============================================================

def test(args):
    print("=" * 70)
    print(f"Testing: LatentFlowNet with KL-VAE")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Load model ----
    model = create_model(
        method=args.method,
        latent_ch=args.latent_ch,
        hidden_ch=args.hidden_ch,
        time_dim=args.time_dim,
        num_blocks=args.num_blocks,
        num_classes=args.num_classes,
        n_steps=args.n_steps,
        obs_dim=load_obs_table(args.obs_file)[1],
        cond_dim=args.cond_dim,
        lambda_s=args.lambda_s,
        lambda_c=args.lambda_c,
        lambda_m=args.lambda_m,
        inter_weight=args.inter_weight,
        residual_weight=args.residual_weight,
        recon_weight=args.recon_weight,
        delta_mid_weight=args.delta_mid_weight,
        target_time=args.target_time,
    ).to(device)

    exp_dir = args.exp_dir or "result/latentflownet"
    ckpt = args.checkpoint or os.path.join(exp_dir, 'best.pth.tar')
    if os.path.exists(ckpt):
        load_checkpoint(ckpt, model)
        print(f"Loaded: {ckpt}")
    else:
        print(f"ERROR: checkpoint not found at {ckpt}")
        return

    model.eval()

    # ---- Load VAE for decoding ----
    vae, vae_type, obs_map, obs_dim = load_vae(
        args.vae_checkpoint, args.latent_ch, device, obs_file=args.obs_file
    )

    # ---- Data ----
    latent_dir = args.data_root or "./data/latent_klvae"
    test_dataset = LongitudinalPETDataset(
        data_root=os.path.join(latent_dir, 'test'),
        label_file=args.label_file,
        obs_file=args.obs_file,
        baseline_time=args.baseline_time,
        target_time=args.target_time,
        unknown_label=args.unknown_label,
        stage='test',
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_longitudinal,
    )

    # Also load raw PET for ground truth in image space
    raw_test_dir = args.raw_data_root or './data/test'

    print(f"Test: {len(test_dataset)} pairs")

    # ---- Evaluate ----
    results = []

    with torch.no_grad():
        for batch in test_loader:
            z_0 = batch['z_0'].float().to(device)
            z_T = batch['z_T'].float().to(device)
            labels = batch['labels'].to(device) if args.num_classes > 0 else None
            subject_id = batch['subject_ids'][0]
            dyn_kwargs = _model_kwargs(args, batch, device)

            # Predict
            z_T_pred = model.predict(z_0, labels, **dyn_kwargs)

            # Decode to image space
            obs, obs_mask = get_obs_batch([subject_id], obs_map, obs_dim, device)
            x_pred = decode_latent(vae, z_T_pred, obs=obs, obs_mask=obs_mask)
            x_gt = decode_latent(vae, z_T, obs=obs, obs_mask=obs_mask)
            x_0 = decode_latent(vae, z_0, obs=obs, obs_mask=obs_mask)

            # Metrics in decoded autoencoder space (legacy columns kept for continuity)
            pred_img = x_pred[0].cpu().numpy().squeeze()
            gt_img = x_gt[0].cpu().numpy().squeeze()
            bl_img = x_0[0].cpu().numpy().squeeze()

            dr = gt_img.max() - gt_img.min()
            if dr < 1e-8:
                raise RuntimeError(f"Decoded target has zero dynamic range for {subject_id}")

            psnr_pred = compare_psnr(gt_img, pred_img, data_range=dr)
            ssim_pred = compare_ssim(gt_img, pred_img, data_range=dr)

            # Decoded-space identity baseline (kept for reference only)
            psnr_id_decoded = compare_psnr(gt_img, bl_img, data_range=dr)
            ssim_id_decoded = compare_ssim(gt_img, bl_img, data_range=dr)

            # Raw-image metrics and true identity baseline
            raw_gt_img = align_volume_to_shape(
                load_raw_pet_image(raw_test_dir, subject_id, args.target_time), pred_img.shape
            )
            raw_bl_img = align_volume_to_shape(
                load_raw_pet_image(raw_test_dir, subject_id, args.baseline_time), pred_img.shape
            )

            raw_metrics = compute_raw_image_metrics(pred_img, raw_gt_img, raw_bl_img)
            if args.raw_data_root and any(
                raw_metrics[name] is None
                for name in (
                    'psnr_raw', 'ssim_raw', 'mae_raw',
                    'psnr_identity_raw', 'ssim_identity_raw', 'mae_identity_raw',
                )
            ):
                raise RuntimeError(
                    f"Complete raw target/baseline metrics are unavailable for {subject_id}"
                )
            psnr_raw = raw_metrics['psnr_raw']
            ssim_raw = raw_metrics['ssim_raw']
            mae_raw = raw_metrics['mae_raw']
            psnr_id_raw = raw_metrics['psnr_identity_raw']
            ssim_id_raw = raw_metrics['ssim_identity_raw']
            mae_id_raw = raw_metrics['mae_identity_raw']

            # Latent space metrics
            delta_pred = (z_T_pred - z_0).norm().item()
            delta_gt = (z_T - z_0).norm().item()

            mae = np.mean(np.abs(pred_img - gt_img))

            results.append({
                'subject': subject_id,
                'psnr': psnr_pred, 'ssim': ssim_pred,
                'psnr_raw': psnr_raw, 'ssim_raw': ssim_raw, 'mae_raw': mae_raw,
                'psnr_identity': psnr_id_raw if psnr_id_raw is not None else psnr_id_decoded,
                'ssim_identity': ssim_id_raw if ssim_id_raw is not None else ssim_id_decoded,
                'psnr_identity_raw': psnr_id_raw, 'ssim_identity_raw': ssim_id_raw,
                'mae_identity_raw': mae_id_raw,
                'psnr_identity_decoded': psnr_id_decoded, 'ssim_identity_decoded': ssim_id_decoded,
                'mae': mae,
                'delta_pred': delta_pred, 'delta_gt': delta_gt,
            })

            if len(results) <= 5:
                use_raw = psnr_raw is not None and psnr_id_raw is not None
                space = 'Raw' if use_raw else 'Decoded'
                shown_psnr = psnr_raw if use_raw else psnr_pred
                shown_ssim = ssim_raw if use_raw else ssim_pred
                shown_mae = mae_raw if use_raw else mae
                shown_id_psnr = psnr_id_raw if use_raw else psnr_id_decoded
                shown_id_ssim = ssim_id_raw if use_raw else ssim_id_decoded
                print(
                    f"  {subject_id} [{space}]: PSNR={shown_psnr:.2f} "
                    f"(identity={shown_id_psnr:.2f}) | SSIM={shown_ssim:.4f} "
                    f"(identity={shown_id_ssim:.4f}) | MAE={shown_mae:.5f} | "
                    f"Δ_pred={delta_pred:.4f} Δ_gt={delta_gt:.4f}"
                )

    # ---- Summary ----
    if results:
        write_test_results(results, exp_dir, vae_type)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train or evaluate LatentFlowNet'
    )
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--method', type=str, default='latentflownet',
                        choices=['latentflownet'], help='Published model name')

    # Model architecture
    parser.add_argument('--latent_ch', type=int, default=4,
                        help='Latent channels (must match VAE)')
    parser.add_argument('--hidden_ch', type=int, default=64,
                        help='Hidden channels in drift/velocity network')
    parser.add_argument('--time_dim', type=int, default=128,
                        help='Time embedding dimension')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='Number of residual blocks per path')
    parser.add_argument('--num_classes', type=int, default=0,
                        help='Disease label classes (0 = no conditioning)')
    parser.add_argument('--cond_dim', type=int, default=32,
                        help='Metadata condition embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=100,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for model initialization and data loading')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--delta_weight', type=float, default=0.0,
                        help='Extra weight on Delta MSE (e.g. 10.0 for anti-identity)')
    parser.add_argument('--use_contrastive', action='store_true',
                        help='Add longitudinal contrastive loss (InfoNCE)')
    parser.add_argument('--use_hinge', action='store_true',
                        help='Add adaptive hinge anti-shortcut loss')
    parser.add_argument('--lambda_s', type=float, default=0.01,
                        help='Curvature penalty weight')
    parser.add_argument('--lambda_c', type=float, default=0.05,
                        help='Endpoint contrastive penalty weight')
    parser.add_argument('--lambda_m', type=float, default=0.2,
                        help='Monotonic path penalty weight')
    parser.add_argument('--inter_weight', type=float, default=2.0,
                        help='Intermediate timepoint supervision weight')
    parser.add_argument('--residual_weight', type=float, default=1.0,
                        help='Endpoint residual loss weight')
    parser.add_argument('--recon_weight', type=float, default=0.5,
                        help='Endpoint state reconstruction loss weight')
    parser.add_argument('--delta_mid_weight', type=float, default=0.5,
                        help='Intermediate delta supervision weight')
    parser.add_argument('--image_eval_every', type=int, default=50,
                        help='Decode to image space every N epochs for validation')

    # Ablation
    parser.add_argument('--no_intermediates', action='store_true',
                        help='Disable intermediate timepoint supervision (ablation)')

    # Numerical integration
    parser.add_argument('--n_steps', type=int, default=20,
                        help='Fixed Euler integration steps')

    # Data
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root of latent data (default: ./data/latent_klvae)')
    parser.add_argument('--raw_data_root', type=str, default=None,
                        help='Raw split directory used for image-space evaluation')
    parser.add_argument('--baseline_time', type=int, default=0,
                        help='Relative baseline month')
    parser.add_argument('--target_time', type=int, default=24,
                        help='Relative endpoint month and trajectory horizon')
    parser.add_argument('--label_file', type=str, default=None)
    parser.add_argument('--unknown_label', type=int, default=0,
                        help='Label id used when a subject is missing from label_file')
    parser.add_argument('--num_workers', type=int, default=4)

    # Checkpoints
    parser.add_argument('--vae_checkpoint', type=str, default=None,
                        help='Path to pre-trained VAE checkpoint')
    parser.add_argument('--obs_file', type=str, default=None,
                        help='Optional subject metadata CSV for LatentFlowNet conditioning')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='LatentFlowNet checkpoint for evaluation')
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='Experiment directory (default: result/latentflownet)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    seed_torch(args.seed)

    # Auto-fill VAE checkpoint if not provided
    if args.vae_checkpoint is None:
        args.vae_checkpoint = "result/klvae/klvae_best.pth.tar"

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    inter_str = "no-inter" if args.no_intermediates else "with-inter"
    print(f"Method: LatentFlowNet | VAE: KL-VAE | Intermediates: {inter_str}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
