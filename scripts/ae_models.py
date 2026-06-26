"""
ae_models.py
============
Unified autoencoder family for the architecture comparison study.

A single class `AneurysmAE` supports three modes via the `variational`
and `beta` arguments, so we can train and compare on identical footing:

  - Plain AE     : variational=False           (no KL, deterministic latent)
  - beta-VAE low : variational=True, beta=0.1   (weak latent regularisation)
  - VAE / beta=1 : variational=True, beta=1.0   (standard VAE)

This lets us probe the smoothness-vs-fidelity tradeoff your supervisor
flagged: stronger KL (higher beta) gives a smoother, more samplable latent
space but tends to bias generated shapes toward the mean (volume bias).
Weaker / no KL preserves size fidelity but yields a less structured space.

Encoder : PointNet-style (no BatchNorm in FC head; LayerNorm instead)
Decoder : implicit SDF MLP with positional encoding (raw linear output)

Shares the proven-stable design from vae_model.py:
  - fc_logvar initialised to small values (near-deterministic warmup)
  - no tanh on SDF output (avoids gradient death)
  - positional encoding with n_freqs=3
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, n_freqs: int = 3, include_input: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        freq_bands = 2.0 ** torch.arange(n_freqs) * np.pi
        self.register_buffer("freq_bands", freq_bands)
        self.out_dim = (3 if include_input else 0) + 3 * 2 * n_freqs

    def forward(self, x):
        out = [x] if self.include_input else []
        for f in self.freq_bands:
            out.append(torch.sin(x * f))
            out.append(torch.cos(x * f))
        return torch.cat(out, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    PointNet-style encoder.
    If variational, outputs (mu, logvar); else outputs (z, None).
    """
    def __init__(self, latent_dim=512, variational=True):
        super().__init__()
        self.latent_dim  = latent_dim
        self.variational = variational

        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),    nn.ReLU(inplace=True),
            nn.Linear(64, 128),  nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 512), nn.ReLU(inplace=True),
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        if variational:
            self.fc_logvar = nn.Linear(512, latent_dim)
            # Near-deterministic warmup: small initial sigma
            nn.init.zeros_(self.fc_logvar.weight)
            nn.init.constant_(self.fc_logvar.bias, -5.0)

    def forward(self, points):
        B, N, _ = points.shape
        x = points.reshape(B * N, 3)
        for layer in self.point_mlp:
            x = layer(x)
        x = x.reshape(B, N, -1).max(dim=1).values
        x = self.fc_shared(x)
        mu = self.fc_mu(x)
        if self.variational:
            logvar = torch.clamp(self.fc_logvar(x), -10.0, 10.0)
            return mu, logvar
        return mu, None


# ─────────────────────────────────────────────────────────────────────────────
# SDF Decoder
# ─────────────────────────────────────────────────────────────────────────────

class SDFDecoder(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=512, n_layers=8,
                 skip_layer=4, n_freqs=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.skip_layer = skip_layer
        self.pos_enc = PositionalEncoding(n_freqs=n_freqs)
        coord_dim = self.pos_enc.out_dim
        input_dim = coord_dim + latent_dim

        self.layers = nn.ModuleList()
        in_dim = input_dim
        for i in range(n_layers):
            if i == skip_layer:
                in_dim = hidden_dim + input_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.output_layer = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, points, z):
        B, N, _ = points.shape
        pe = self.pos_enc(points)
        z_exp = z.unsqueeze(1).expand(B, N, self.latent_dim)
        inp = torch.cat([pe, z_exp], dim=-1)
        x = inp.reshape(B * N, -1)
        inp_flat = x.clone()
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                x = torch.cat([x, inp_flat], dim=-1)
            x = F.relu(layer(x), inplace=True)
        sdf = self.output_layer(x)
        return sdf.reshape(B, N, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Unified model
# ─────────────────────────────────────────────────────────────────────────────

class AneurysmAE(nn.Module):
    """
    Unified autoencoder. Set variational=False for a plain AE, or
    variational=True with a chosen beta for a (beta-)VAE.

    mode_name is just a human-readable tag stored in checkpoints.
    """
    def __init__(self, latent_dim=512, hidden_dim=512, n_freqs=3,
                 variational=True, beta=1.0, mode_name="vae"):
        super().__init__()
        self.latent_dim  = latent_dim
        self.variational = variational
        self.beta        = beta
        self.mode_name   = mode_name

        self.encoder = Encoder(latent_dim, variational=variational)
        self.decoder = SDFDecoder(latent_dim, hidden_dim, n_freqs=n_freqs)

        n = sum(p.numel() for p in self.parameters())
        print(f"AneurysmAE [{mode_name}]: variational={variational}, "
              f"beta={beta}, {n:,} params")

    def reparameterise(self, mu, logvar):
        if not self.variational or logvar is None:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, surface_points):
        return self.encoder(surface_points)

    def decode(self, query_points, z):
        return self.decoder(query_points, z)

    def forward(self, surface_points, query_points):
        mu, logvar = self.encode(surface_points)
        z = self.reparameterise(mu, logvar)
        sdf = self.decode(query_points, z)
        return sdf, mu, logvar

    @torch.no_grad()
    def generate(self, query_points, n_samples=1, device=None, truncation=1.0):
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n_samples, self.latent_dim, device=device) * truncation
        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)
        qp = query_points.expand(n_samples, -1, -1)
        return self.decode(qp, z)

    @torch.no_grad()
    def interpolate(self, surface_a, surface_b, query_points, steps=7):
        mu_a, _ = self.encode(surface_a)
        mu_b, _ = self.encode(surface_b)
        results = []
        for t in torch.linspace(0, 1, steps):
            z_t = (1 - t) * mu_a + t * mu_b
            results.append(self.decode(query_points, z_t))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def ae_sdf_loss(pred_sdf, gt_sdf, mu, logvar, beta=1.0,
                kl_weight=1e-5, clamp_dist=0.2, free_bits=0.5,
                variational=True):
    """
    Clamped L1 SDF reconstruction + (optionally) beta-weighted KL with free bits.
    For a plain AE (variational=False), only the reconstruction term is used.
    """
    pred = pred_sdf.squeeze(-1) if pred_sdf.dim() == 3 else pred_sdf
    pred_c = torch.clamp(pred,   -clamp_dist, clamp_dist)
    gt_c   = torch.clamp(gt_sdf, -clamp_dist, clamp_dist)
    recon  = torch.abs(pred_c - gt_c).mean()

    if not variational or logvar is None:
        return recon, recon, torch.tensor(0.0, device=pred.device)

    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_clamped = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss    = kl_clamped.sum(dim=1).mean()
    kl_raw     = kl_per_dim.sum(dim=1).mean()

    total = recon + beta * kl_weight * kl_loss
    return total, recon, kl_raw


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, var, beta in [("plain_ae", False, 0.0),
                            ("beta_vae_low", True, 0.1),
                            ("vae", True, 1.0)]:
        m = AneurysmAE(variational=var, beta=beta, mode_name=name).to(device)
        surf = torch.randn(2, 2048, 3, device=device)
        qry  = torch.randn(2, 2048, 3, device=device)
        gt   = torch.randn(2, 2048, device=device) * 0.1
        sdf, mu, logvar = m(surf, qry)
        loss, recon, kl = ae_sdf_loss(sdf, gt, mu, logvar, beta=beta,
                                      variational=var)
        print(f"  [{name}] loss={loss.item():.4f} recon={recon.item():.4f} "
              f"kl={kl.item():.4f}\n")
    print("All model checks passed.")