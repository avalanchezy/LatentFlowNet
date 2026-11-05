"""LatentFlowNet: metadata-conditioned latent trajectories for longitudinal PET.

The trajectory uses fixed-step Euler integration, sparse intermediate visit
supervision, and endpoint contrastive, curvature, and monotonic penalties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



class GroupNorm32(nn.GroupNorm):
    def __init__(self, channels):
        super().__init__(num_groups=min(32, channels), num_channels=channels)



class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)



class MetadataConditioner(nn.Module):
    """Encode subject metadata and its availability mask."""

    def __init__(self, obs_dim, cond_dim=32, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
            nn.SiLU(),
        )

    def forward(self, obs, obs_mask):
        return self.net(torch.cat([obs, obs_mask], dim=-1))



class ConditionalResidualBlock3D(nn.Module):
    """Residual block with joint time and metadata FiLM conditioning."""

    def __init__(self, channels, time_dim, cond_dim=0, dropout=0.1):
        super().__init__()
        self.norm1 = GroupNorm32(channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = GroupNorm32(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.act = Swish()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.time_proj = nn.Sequential(Swish(), nn.Linear(time_dim, channels))
        self.cond_proj = nn.Sequential(Swish(), nn.Linear(cond_dim, channels)) if cond_dim > 0 else None

    def forward(self, x, t_emb, cond_emb=None):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None, None]
        if self.cond_proj is not None and cond_emb is not None:
            h = h + self.cond_proj(cond_emb)[:, :, None, None, None]
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h



class LatentFlowDriftNet(nn.Module):
    """
    Time-conditioned drift network for LatentFlowNet.

    Uses time + metadata conditioning to modulate a shared latent flow field.
    """

    def __init__(self, latent_ch=4, hidden_ch=64, time_dim=128,
                 cond_dim=32, num_blocks=4, num_classes=0, dropout=0.1):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2), Swish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.label_embed = nn.Embedding(num_classes, cond_dim) if num_classes > 0 else None
        self.input_conv = nn.Conv3d(latent_ch, hidden_ch, 3, padding=1)
        self.cond_in = nn.Linear(cond_dim, hidden_ch) if cond_dim > 0 else None
        self.blocks = nn.ModuleList([
            ConditionalResidualBlock3D(hidden_ch, time_dim, cond_dim=cond_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.out_norm = GroupNorm32(hidden_ch)
        self.out_act = Swish()
        self.output_conv = nn.Conv3d(hidden_ch, latent_ch, 3, padding=1)

        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, t, z, cond_emb=None, label=None):
        B = z.shape[0]
        t_vec = t.expand(B) if t.dim() == 0 else t
        t_emb = self.time_mlp(self.time_embed(t_vec))
        if self.label_embed is not None and label is not None:
            label_emb = self.label_embed(label.long())
            cond_emb = label_emb if cond_emb is None else cond_emb + label_emb

        h = self.input_conv(z)
        if self.cond_in is not None and cond_emb is not None:
            h = h + self.cond_in(cond_emb)[:, :, None, None, None]

        for block in self.blocks:
            h = block(h, t_emb, cond_emb=cond_emb)

        h = self.out_norm(h)
        h = self.out_act(h)
        return self.output_conv(h)



def longitudinal_contrastive_loss(z_pred, z_T_gt, z_0, tau=0.1):
    """InfoNCE with the endpoint as positive and the baseline as negative."""
    B = z_pred.shape[0]
    # Flatten to vectors
    pred_flat = z_pred.reshape(B, -1)
    gt_flat = z_T_gt.reshape(B, -1)
    z0_flat = z_0.reshape(B, -1)

    # Cosine similarities
    sim_pos = F.cosine_similarity(pred_flat, gt_flat, dim=1) / tau
    sim_neg = F.cosine_similarity(pred_flat, z0_flat, dim=1) / tau

    # InfoNCE
    logits = torch.stack([sim_pos, sim_neg], dim=1)  # (B, 2)
    labels = torch.zeros(B, dtype=torch.long, device=z_pred.device)  # positive = index 0
    loss = F.cross_entropy(logits, labels)
    return loss



def adaptive_hinge_loss(z_pred, z_T_gt, z_0, alpha=0.85):
    """Optional hinge on predicted change relative to observed endpoint change."""
    delta_pred = z_pred - z_0
    delta_gt = z_T_gt - z_0

    # Per-sample L2 norms
    d_pred = delta_pred.reshape(z_pred.shape[0], -1).norm(dim=1)
    d_true = delta_gt.reshape(z_pred.shape[0], -1).norm(dim=1)

    # Hinge: penalize if prediction magnitude < α * true magnitude
    penalty = torch.clamp(alpha * d_true - d_pred, min=0.0)
    return penalty.mean()



class LatentFlowNet(nn.Module):
    """
    LatentFlowNet for longitudinal PET prediction.

    Evolves a frozen KL-VAE latent representation over normalized time:
    - conditions the drift with subject metadata
    - uses time-aware drift to exploit intermediate visits
    - regularizes curvature instead of directly shrinking velocity
    - adds endpoint contrastive and monotonic path penalties
    """

    def __init__(self, latent_ch=4, hidden_ch=64, time_dim=128,
                 num_blocks=4, num_classes=0, dropout=0.1,
                 n_steps=20, obs_dim=0, cond_dim=32,
                 lambda_c=0.05, lambda_s=0.01, lambda_m=0.2,
                 inter_weight=2.0, residual_weight=1.0,
                 recon_weight=0.5, delta_mid_weight=0.5,
                 target_time=24, **kwargs):
        super().__init__()
        self.latent_ch = latent_ch
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.cond_dim = cond_dim if obs_dim > 0 else 0
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.lambda_m = lambda_m
        self.inter_weight = inter_weight
        self.residual_weight = residual_weight
        self.recon_weight = recon_weight
        self.delta_mid_weight = delta_mid_weight
        self.target_time = int(target_time)
        if self.target_time <= 0:
            raise ValueError(f"target_time must be positive, got {target_time}")

        self.metadata_conditioner = None
        if self.cond_dim > 0:
            self.metadata_conditioner = MetadataConditioner(obs_dim=obs_dim, cond_dim=self.cond_dim)

        self.drift = LatentFlowDriftNet(
            latent_ch=latent_ch,
            hidden_ch=hidden_ch,
            time_dim=time_dim,
            cond_dim=self.cond_dim,
            num_blocks=num_blocks,
            num_classes=num_classes,
            dropout=dropout,
        )

    @staticmethod
    def _masked_mse(pred, target, mask=None):
        if mask is None:
            return F.mse_loss(pred, target)

        if mask.dim() < pred.dim():
            mask = mask.view(mask.shape[0], *([1] * (pred.dim() - 1)))
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        denom = mask.sum().clamp_min(1.0) * pred[0].numel()
        return ((pred - target).pow(2) * mask).sum() / denom

    @staticmethod
    def _coerce_observation(z_obs, ref_tensor):
        if isinstance(z_obs, dict):
            z_tensor = z_obs['z']
            mask = z_obs.get('mask')
        else:
            z_tensor = z_obs
            mask = None

        if z_tensor.dim() == 4:
            z_tensor = z_tensor.unsqueeze(0)
        z_tensor = z_tensor.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

        if mask is not None:
            mask = mask.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
            if mask.dim() == 1:
                mask = mask.view(-1, 1, 1, 1, 1)
            elif mask.dim() == 2:
                mask = mask.view(mask.shape[0], 1, 1, 1, 1)

        return z_tensor, mask

    def _prepare_condition(self, z_ref, obs=None, obs_mask=None):
        if self.metadata_conditioner is None:
            return None

        batch_size = z_ref.shape[0]
        device = z_ref.device
        dtype = z_ref.dtype
        if obs is None:
            obs = torch.zeros(batch_size, self.obs_dim, device=device, dtype=dtype)
            if obs_mask is None:
                obs_mask = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            else:
                obs_mask = obs_mask.to(device=device, dtype=dtype).view(batch_size, 1)
        else:
            obs = obs.to(device=device, dtype=dtype)
            if obs_mask is None:
                obs_mask = torch.ones(batch_size, 1, device=device, dtype=dtype)
            else:
                obs_mask = obs_mask.to(device=device, dtype=dtype).view(batch_size, 1)
        return self.metadata_conditioner(obs, obs_mask)

    def _rollout(self, z_0, label=None, cond_emb=None, capture_months=None):
        dt = 1.0 / self.n_steps
        z = z_0.clone()
        prev_dz = None
        prev_delta_norm = None
        curvature = torch.tensor(0.0, device=z_0.device)
        monotonic = torch.tensor(0.0, device=z_0.device)
        states = {}
        capture_months = sorted(set(capture_months or []))
        capture_taus = {
            month: month / self.target_time
            for month in capture_months
            if 0 < month < self.target_time
        }

        for step in range(self.n_steps):
            tau_start = step * dt
            tau_end = (step + 1) * dt
            z_step_start = z
            t = torch.full((z.shape[0],), tau_start, device=z.device, dtype=z.dtype)
            dz = self.drift(t, z_step_start, cond_emb=cond_emb, label=label)
            if prev_dz is not None:
                curvature = curvature + (dz - prev_dz).pow(2).mean()
            z = z_step_start + dt * dz

            delta_norm = (z - z_0).reshape(z.shape[0], -1).norm(dim=1)
            if prev_delta_norm is not None:
                monotonic = monotonic + F.relu(prev_delta_norm - delta_norm).mean()
            prev_delta_norm = delta_norm
            prev_dz = dz

            for month, tau_anchor in capture_taus.items():
                if month in states or not tau_start < tau_anchor <= tau_end + 1e-12:
                    continue
                step_fraction = min(1.0, max(0.0, (tau_anchor - tau_start) / dt))
                states[month] = z_step_start + (step_fraction * dt) * dz

        return z, states, curvature / max(self.n_steps - 1, 1), monotonic / max(self.n_steps - 1, 1)

    def predict(self, z_0, label=None, obs=None, obs_mask=None):
        cond_emb = self._prepare_condition(z_0, obs=obs, obs_mask=obs_mask)
        z_T_pred, _, _, _ = self._rollout(z_0, label=label, cond_emb=cond_emb)
        return z_T_pred

    def compute_loss(self, z_0, z_T_gt, label=None, observations=None, obs=None, obs_mask=None):
        cond_emb = self._prepare_condition(z_0, obs=obs, obs_mask=obs_mask)
        capture_months = [
            month for month in observations.keys() if month != self.target_time
        ] if observations else []
        z_T_pred, mid_states, curvature_loss, monotonic_loss = self._rollout(
            z_0, label=label, cond_emb=cond_emb, capture_months=capture_months
        )

        delta_pred = z_T_pred - z_0
        delta_gt = z_T_gt - z_0
        residual_loss = F.mse_loss(delta_pred, delta_gt)
        recon_loss = F.mse_loss(z_T_pred, z_T_gt)
        contrastive_loss = longitudinal_contrastive_loss(z_T_pred, z_T_gt, z_0)

        inter_loss = torch.tensor(0.0, device=z_0.device)
        weight_sum = 0.0
        if observations and len(observations) > 0:
            for month, z_obs in observations.items():
                if month == self.target_time or month not in mid_states:
                    continue
                z_mid = mid_states[month]
                z_obs_b, obs_mask_b = self._coerce_observation(z_obs, z_0)
                delta_mid_pred = z_mid - z_0
                delta_mid_gt = z_obs_b - z_0
                weight = month / self.target_time
                valid_fraction = 1.0
                if obs_mask_b is not None:
                    valid_fraction = obs_mask_b.view(obs_mask_b.shape[0], -1)[:, 0].mean().item()
                inter_loss = inter_loss + weight * valid_fraction * (
                    self._masked_mse(z_mid, z_obs_b, obs_mask_b) +
                    self.delta_mid_weight * self._masked_mse(delta_mid_pred, delta_mid_gt, obs_mask_b)
                )
                weight_sum += weight * valid_fraction
        if weight_sum > 0:
            inter_loss = inter_loss / weight_sum

        loss = (
            self.residual_weight * residual_loss
            + self.recon_weight * recon_loss
            + self.lambda_c * contrastive_loss
            + self.lambda_s * curvature_loss
            + self.lambda_m * monotonic_loss
        )
        if weight_sum > 0:
            loss = loss + self.inter_weight * inter_loss

        return loss, {
            'loss': loss.item(),
            'residual': residual_loss.item(),
            'recon': recon_loss.item(),
            'contrastive': contrastive_loss.item(),
            'smoothness': curvature_loss.item(),
            'monotonic': monotonic_loss.item(),
            'inter': inter_loss.item() if weight_sum > 0 else 0.0,
            'delta_pred_norm': delta_pred.norm().item(),
            'delta_gt_norm': delta_gt.norm().item(),
        }



def create_model(method='latentflownet', **kwargs):
    """Construct the published LatentFlowNet model."""
    if method != 'latentflownet':
        raise ValueError(f"Unknown method: {method}; expected 'latentflownet'")
    return LatentFlowNet(**kwargs)
