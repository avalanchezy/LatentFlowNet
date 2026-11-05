"""
VAE model architectures for longitudinal PET prediction.

Contains:
- KLVAE: KL-regularized Variational Autoencoder (Stable-Diffusion style)
- PatchDiscriminator3D: optional PatchGAN discriminator for perceptual training

LatentFlowNet uses four latent channels and fourfold spatial downsampling.
Input:  (B, 1, 112, 128, 112)
Latent: (B, 4, 28, 32, 28)
Output: (B, 1, 112, 128, 112)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import config


# ============================================================
# Basic Components
# ============================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.Module):
    """Group Normalization with max 32 groups."""
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(
            num_groups=min(num_groups, channels),
            num_channels=channels,
            eps=1e-6,
            affine=True
        )

    def forward(self, x):
        return self.gn(x)


class Upsample3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        return self.conv(x)


class Downsample3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ResBlock3D(nn.Module):
    """Residual block with pre-activation (GroupNorm → Swish → Conv)."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            GroupNorm32(in_ch),
            Swish(),
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            GroupNorm32(out_ch),
            Swish(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.use_checkpoint = False

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return self.skip(x) + checkpoint(self.block, x, use_reentrant=False)
        return self.skip(x) + self.block(x)


class AttnBlock3D(nn.Module):
    """Self-attention block for 3D feature maps (used at lowest resolution only).

    Uses F.scaled_dot_product_attention (PyTorch 2.0+) for memory-efficient
    attention. This avoids materializing the N×N attention matrix, reducing
    memory from O(N²) to O(N). Critical for large 3D spatial resolutions.
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = GroupNorm32(channels)
        self.q = nn.Conv3d(channels, channels, 1)
        self.k = nn.Conv3d(channels, channels, 1)
        self.v = nn.Conv3d(channels, channels, 1)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        B, C, D, H, W = h.shape
        N = D * H * W

        q = self.q(h).view(B, C, N).permute(0, 2, 1).unsqueeze(1)  # (B, 1, N, C)
        k = self.k(h).view(B, C, N).permute(0, 2, 1).unsqueeze(1)  # (B, 1, N, C)
        v = self.v(h).view(B, C, N).permute(0, 2, 1).unsqueeze(1)  # (B, 1, N, C)

        # Memory-efficient attention: does NOT materialize the N×N matrix
        h = F.scaled_dot_product_attention(q, k, v)  # (B, 1, N, C)

        h = h.squeeze(1).permute(0, 2, 1).view(B, C, D, H, W)  # (B, C, D, H, W)
        return x + self.proj(h)


# ============================================================
# KL-VAE Encoder / Decoder
# ============================================================

class VAEEncoder(nn.Module):
    """
    Encoder for KL-VAE.

    Input:  (B, 1, 112, 128, 112)
    Output: (B, 2*latent_ch, 28, 32, 28) — mu and logvar concatenated

    Architecture: 4 resolution levels, 2 ResBlocks each.
    Channels: 32 → 64 → 128 → 256
    Downsamples 2x at first 2 levels → 4x total.
    Self-attention at lowest resolution (28×32×28).
    """
    def __init__(self, in_ch=1, latent_ch=4, ch_mult=(1, 2, 4, 8),
                 base_ch=32, num_res_blocks=2, dropout=0.0,
                 double_z=True, use_attn_at_bottom=True):
        super().__init__()
        self.double_z = double_z
        channels = [base_ch * m for m in ch_mult]  # [32, 64, 128, 256]
        num_levels = len(channels)

        # Input conv
        self.input_conv = nn.Conv3d(in_ch, channels[0], 3, padding=1)

        # Downsampling levels
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        in_c = channels[0]
        for i in range(num_levels):
            out_c = channels[i]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3D(in_c, out_c, dropout))
                in_c = out_c
            self.down_blocks.append(blocks)

            # Downsample after first 2 levels (total 4x)
            if i < 2:
                self.downsamplers.append(Downsample3D(out_c))
            else:
                self.downsamplers.append(nn.Identity())

        # Bottom block (with optional attention)
        self.mid_block1 = ResBlock3D(channels[-1], channels[-1], dropout)
        self.mid_attn = AttnBlock3D(channels[-1]) if use_attn_at_bottom else nn.Identity()
        self.mid_block2 = ResBlock3D(channels[-1], channels[-1], dropout)

        # Output
        out_ch = latent_ch * 2 if double_z else latent_ch
        self.out_norm = GroupNorm32(channels[-1])
        self.out_act = Swish()
        self.out_conv = nn.Conv3d(channels[-1], out_ch, 3, padding=1)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on all ResBlocks to save memory."""
        for blocks in self.down_blocks:
            for block in blocks:
                block.use_checkpoint = True
        self.mid_block1.use_checkpoint = True
        self.mid_block2.use_checkpoint = True

    def forward(self, x):
        h = self.input_conv(x)

        for blocks, down in zip(self.down_blocks, self.downsamplers):
            for block in blocks:
                h = block(h)
            h = down(h)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h


class VAEDecoder(nn.Module):
    """
    Decoder for KL-VAE.

    Input:  (B, latent_ch, 28, 32, 28)
    Output: (B, 1, 112, 128, 112)

    Mirror of encoder. Upsamples at last 2 levels.
    """
    def __init__(self, out_ch=1, latent_ch=4, ch_mult=(1, 2, 4, 8),
                 base_ch=32, num_res_blocks=2, dropout=0.0,
                 use_attn_at_bottom=True):
        super().__init__()
        channels = [base_ch * m for m in ch_mult]  # [32, 64, 128, 256]
        num_levels = len(channels)

        # Input conv
        self.input_conv = nn.Conv3d(latent_ch, channels[-1], 3, padding=1)

        # Mid blocks
        self.mid_block1 = ResBlock3D(channels[-1], channels[-1], dropout)
        self.mid_attn = AttnBlock3D(channels[-1]) if use_attn_at_bottom else nn.Identity()
        self.mid_block2 = ResBlock3D(channels[-1], channels[-1], dropout)

        # Upsampling levels (reversed)
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        in_c = channels[-1]
        for i in reversed(range(num_levels)):
            out_c = channels[i]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):  # +1 for decoder
                blocks.append(ResBlock3D(in_c, out_c, dropout))
                in_c = out_c
            self.up_blocks.append(blocks)

            # Upsample at last 2 levels (matching encoder's 2 downsamples)
            if i < 2:
                self.upsamplers.append(Upsample3D(out_c))
            else:
                self.upsamplers.append(nn.Identity())

        # Output
        self.out_norm = GroupNorm32(channels[0])
        self.out_act = Swish()
        self.out_conv = nn.Conv3d(channels[0], out_ch, 3, padding=1)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on all ResBlocks to save memory."""
        for blocks in self.up_blocks:
            for block in blocks:
                block.use_checkpoint = True
        self.mid_block1.use_checkpoint = True
        self.mid_block2.use_checkpoint = True

    def forward(self, z):
        h = self.input_conv(z)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        for blocks, up in zip(self.up_blocks, self.upsamplers):
            for block in blocks:
                h = block(h)
            h = up(h)

        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h


# ============================================================
# KL-VAE
# ============================================================

class KLVAE(nn.Module):
    """
    KL-regularized Variational Autoencoder for 3D PET images.

    Follows the Stable Diffusion VAE design:
    - Continuous latent space with KL regularization
    - Deeper network with attention at bottom resolution
    - latent_channels=4

    Input:  (B, 1, 112, 128, 112)
    Latent: (B, 4, 28, 32, 28)
    Output: (B, 1, 112, 128, 112)
    """
    def __init__(self, latent_ch=4, base_ch=32, ch_mult=(1, 2, 4, 8),
                 num_res_blocks=2, dropout=0.0):
        super().__init__()
        self.latent_ch = latent_ch

        self.encoder = VAEEncoder(
            in_ch=1, latent_ch=latent_ch, ch_mult=ch_mult,
            base_ch=base_ch, num_res_blocks=num_res_blocks,
            dropout=dropout, double_z=True
        )
        self.decoder = VAEDecoder(
            out_ch=1, latent_ch=latent_ch, ch_mult=ch_mult,
            base_ch=base_ch, num_res_blocks=num_res_blocks,
            dropout=dropout
        )

        # Quant/post-quant conv (matches SD convention)
        self.quant_conv = nn.Conv3d(2 * latent_ch, 2 * latent_ch, 1)
        self.post_quant_conv = nn.Conv3d(latent_ch, latent_ch, 1)

    def encode(self, x):
        """Encode to distribution parameters (mu, logvar)."""
        h = self.encoder(x)
        h = self.quant_conv(h)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent to image."""
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        mu, logvar = self.encode(x)
        if sample_posterior:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        recon = self.decode(z)
        return recon, mu, logvar

    def get_latent(self, x):
        """Get deterministic latent (mu only) for downstream use."""
        mu, _ = self.encode(x)
        return mu

    @staticmethod
    def kl_loss(mu, logvar):
        """KL divergence: D_KL(q(z|x) || N(0,1))."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save GPU memory."""
        self.encoder.enable_gradient_checkpointing()
        self.decoder.enable_gradient_checkpointing()


# ============================================================
# Discriminator (shared, optional for both)
# ============================================================

class PatchDiscriminator3D(nn.Module):
    """
    PatchGAN discriminator for 3D volumes.
    Outputs per-patch real/fake scores.
    """
    def __init__(self, in_ch=1, base_ch=32, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        ch = base_ch
        for i in range(1, n_layers):
            prev_ch = ch
            ch = min(ch * 2, 256)
            layers += [
                nn.Conv3d(prev_ch, ch, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm3d(ch),
                nn.LeakyReLU(0.2, True)
            ]

        # Final conv → 1-channel patch score
        prev_ch = ch
        ch = min(ch * 2, 256)
        layers += [
            nn.Conv3d(prev_ch, ch, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(ch),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ch, 1, 4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 1, 112, 128, 112).to(device)

    print("=" * 60)
    print("Testing KL-VAE")
    print("=" * 60)
    klvae = KLVAE(latent_ch=4).to(device)
    params = sum(p.numel() for p in klvae.parameters())
    print(f"Parameters: {params:,}")

    recon, mu, logvar = klvae(x)
    print(f"Input:  {x.shape}")
    print(f"Latent: {mu.shape}")
    print(f"Recon:  {recon.shape}")
    kl = KLVAE.kl_loss(mu, logvar)
    print(f"KL loss: {kl.item():.4f}")

    z = klvae.get_latent(x)
    print(f"Deterministic latent: {z.shape}")

    print()
    print("=" * 60)
    print("Testing PatchDiscriminator3D")
    print("=" * 60)
    disc = PatchDiscriminator3D().to(device)
    params = sum(p.numel() for p in disc.parameters())
    print(f"Parameters: {params:,}")
    out = disc(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    print("\nAll tests passed!")
