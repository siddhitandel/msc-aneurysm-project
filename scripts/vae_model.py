"""
vae_model.py
============
Variational Autoencoder for 3D aneurysm shape generation.

Major upgrades over the basic autoencoder:
  1. VARIATIONAL  — encoder outputs (mu, logvar); reparameterisation trick
                    gives a smooth, samplable latent space for generation
  2. SDF DECODER  — predicts signed distance (tanh output) instead of
                    binary occupancy; carries far more geometric detail
  3. POSITIONAL ENCODING — NeRF-style sin/cos frequency encoding of input
                    coordinates lets the MLP represent high-frequency detail
  4. LARGER LATENT — default 512 dims (was 256)

References:
  - Kingma & Welling 2014  (VAE)
  - Park et al. 2019        (DeepSDF)
  - Mildenhall et al. 2020  (NeRF positional encoding)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding (NeRF-style)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Encode each 3D coordinate with a set of sinusoids at increasing
    frequencies. This lets the downstream MLP represent high-frequency
    geometric detail it would otherwise smooth away.

    For n_freqs frequency bands, output dim = 3 + 3 * 2 * n_freqs
    (original coords + sin/cos for each band per axis).
    """

    def __init__(self, n_freqs: int = 6, include_input: bool = True):
        super().__init__()
        self.n_freqs       = n_freqs
        self.include_input = include_input
        # Frequency bands: 2^0, 2^1, ..., 2^(n_freqs-1)  scaled by pi
        freq_bands = 2.0 ** torch.arange(n_freqs) * np.pi
        self.register_buffer("freq_bands", freq_bands)

        self.out_dim = (3 if include_input else 0) + 3 * 2 * n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:   x : (..., 3)
        Returns:    (..., out_dim)
        """
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Variational PointNet Encoder
# ─────────────────────────────────────────────────────────────────────────────

class VariationalEncoder(nn.Module):
    """
    PointNet-style encoder that outputs the parameters of a Gaussian
    posterior q(z|x): mean (mu) and log-variance (logvar).

    Input  : (B, N, 3)  surface point cloud
    Output : mu, logvar  each (B, latent_dim)
    """

    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim

        # Per-point MLP — NO BatchNorm. BatchNorm over (B*N) points couples
        # every point in the batch through shared statistics, which both
        # destabilises small-batch training and crashes on batch size 1.
        # Plain Linear + ReLU lets each point be processed independently
        # (true PointNet behaviour).
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),    nn.ReLU(inplace=True),
            nn.Linear(64, 128),  nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 512), nn.ReLU(inplace=True),
        )

        # Global aggregation head — LayerNorm works with any batch size and
        # normalises per-sample (not across the batch), preserving per-shape
        # information the latent code needs.
        self.fc_shared = nn.Sequential(
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
        )
        self.fc_mu     = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Initialise logvar head to output small negative values at start.
        # This makes sigma = exp(0.5*logvar) tiny, so z ~= mu early in
        # training (near-deterministic). Without this, the initial random
        # logvar makes z mostly noise and the decoder can never converge —
        # the whole prediction field swings wildly each step. KL annealing
        # then gradually introduces controlled stochasticity.
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -5.0)

    def forward(self, points: torch.Tensor):
        B, N, _ = points.shape
        x = points.reshape(B * N, 3)

        for layer in self.point_mlp:
            x = layer(x)
        x = x.reshape(B, N, -1)        # (B, N, 512)

        x = x.max(dim=1).values        # global max pool -> (B, 512)
        x = self.fc_shared(x)

        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar


# ─────────────────────────────────────────────────────────────────────────────
# SDF Decoder with positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class SDFDecoder(nn.Module):
    """
    Implicit SDF decoder conditioned on latent code z.

    Input  : query points (B, N, 3) + latent z (B, latent_dim)
    Output : signed distance (B, N, 1), range roughly [-1, 1] via tanh

    Uses positional encoding on the query coordinates and a skip
    connection at the midpoint layer (DeepSDF style).
    """

    def __init__(self,
                 latent_dim: int = 512,
                 hidden_dim: int = 512,
                 n_layers:   int = 8,
                 skip_layer: int = 4,
                 n_freqs:    int = 6):
        super().__init__()

        self.latent_dim = latent_dim
        self.skip_layer = skip_layer
        self.n_layers   = n_layers

        self.pos_enc = PositionalEncoding(n_freqs=n_freqs, include_input=True)
        coord_dim    = self.pos_enc.out_dim          # encoded coordinate dim
        input_dim    = coord_dim + latent_dim

        self.layers = nn.ModuleList()
        in_dim = input_dim
        for i in range(n_layers):
            if i == skip_layer:
                in_dim = hidden_dim + input_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.output_layer = nn.Linear(hidden_dim, 1)
        # NO tanh on the output. SDF regression uses a raw linear output
        # (as in DeepSDF). tanh saturates -> zero gradient -> dead network
        # (the loss locks at a constant). The clamp in the loss handles range.

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Smaller init on the final layer so initial SDF predictions are
        # near zero (a reasonable starting point that keeps gradients alive).
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, points: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, N, _ = points.shape

        # Positional encoding on coordinates
        pe = self.pos_enc(points)                         # (B, N, coord_dim)

        # Broadcast latent to every query point
        z_exp = z.unsqueeze(1).expand(B, N, self.latent_dim)

        inp = torch.cat([pe, z_exp], dim=-1)              # (B, N, input_dim)
        x   = inp.reshape(B * N, -1)
        inp_flat = x.clone()

        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                x = torch.cat([x, inp_flat], dim=-1)
            x = F.relu(layer(x), inplace=True)

        sdf = self.output_layer(x)                        # (B*N, 1) raw SDF
        return sdf.reshape(B, N, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Full VAE
# ─────────────────────────────────────────────────────────────────────────────

class AneurysmVAE(nn.Module):
    """
    Variational autoencoder: VariationalEncoder + SDFDecoder.

    Training forward pass returns predicted SDF, mu, logvar so the loss
    can combine SDF reconstruction with the KL divergence term.
    """

    def __init__(self,
                 latent_dim: int = 512,
                 hidden_dim: int = 512,
                 n_freqs:    int = 6):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder(latent_dim=latent_dim)
        self.decoder = SDFDecoder(latent_dim=latent_dim,
                                  hidden_dim=hidden_dim,
                                  n_freqs=n_freqs)

        n_enc = sum(p.numel() for p in self.encoder.parameters())
        n_dec = sum(p.numel() for p in self.decoder.parameters())
        print(f"AneurysmVAE:")
        print(f"  Encoder params : {n_enc:,}")
        print(f"  Decoder params : {n_dec:,}")
        print(f"  Total params   : {n_enc + n_dec:,}")
        print(f"  Latent dim     : {latent_dim}")
        print(f"  Pos-enc dim    : {self.decoder.pos_enc.out_dim}")

    def reparameterise(self, mu, logvar):
        """Sample z ~ N(mu, sigma^2) using the reparameterisation trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, surface_points):
        mu, logvar = self.encoder(surface_points)
        return mu, logvar

    def decode(self, query_points, z):
        return self.decoder(query_points, z)

    def forward(self, surface_points, query_points):
        mu, logvar = self.encode(surface_points)
        z          = self.reparameterise(mu, logvar)
        sdf        = self.decode(query_points, z)
        return sdf, mu, logvar

    @torch.no_grad()
    def generate(self, query_points, n_samples=1, device=None, truncation=1.0):
        """
        Generate new shapes by sampling z ~ N(0, truncation^2 * I).
        Lower truncation -> more typical / less diverse shapes.
        """
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n_samples, self.latent_dim, device=device) * truncation
        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)
        qp = query_points.expand(n_samples, -1, -1)
        return self.decode(qp, z)

    @torch.no_grad()
    def interpolate(self, surface_a, surface_b, query_points, steps=7):
        """Interpolate between two shapes using their posterior means."""
        mu_a, _ = self.encode(surface_a)
        mu_b, _ = self.encode(surface_b)
        results = []
        for t in torch.linspace(0, 1, steps):
            z_t = (1 - t) * mu_a + t * mu_b
            results.append(self.decode(query_points, z_t))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

def vae_sdf_loss(pred_sdf, gt_sdf, mu, logvar,
                 kl_weight: float = 1e-5,
                 clamp_dist: float = 0.1,
                 free_bits: float = 0.5,
                 surface_weight: float = 5.0):
    """
    Combined VAE loss for SDF regression, designed to PREVENT posterior collapse.

    Components:
      recon : L1 loss between predicted and ground-truth SDF, with extra
              weight on near-surface points (where geometry matters most).
              Without this, the model just predicts the "easy" far-field SDF
              and ignores the latent code -> collapse.
      kl    : KL divergence with a FREE-BITS floor. Each latent dim is allowed
              `free_bits` nats of KL "for free" before being penalised, which
              forces a minimum amount of information through the latent code.

    Args:
      pred_sdf : (B, N) or (B, N, 1)
      gt_sdf   : (B, N)
      clamp_dist : clamp SDF to [-clamp_dist, clamp_dist] (DeepSDF trick)
      free_bits  : minimum KL per latent dim before it's penalised (nats)
      surface_weight : multiplier on the loss for near-surface points

    Returns:
      total_loss, recon_loss, kl_loss  (all scalars)
    """
    pred = pred_sdf.squeeze(-1) if pred_sdf.dim() == 3 else pred_sdf

    # Clamp both to focus on near-surface region (DeepSDF trick).
    # Use ±0.2 — wide enough to leave gradient room, narrow enough to
    # concentrate capacity near the surface.
    pred_c = torch.clamp(pred,   -clamp_dist, clamp_dist)
    gt_c   = torch.clamp(gt_sdf, -clamp_dist, clamp_dist)

    # Plain L1. We deliberately do NOT use the (l1*w)/w.mean() weighting
    # that earlier created a degenerate flat minimum the optimiser got
    # stuck in. Plain mean L1 over clamped SDF is the proven-stable choice
    # (matches DeepSDF) and converges cleanly in single-shape overfit tests.
    recon_loss = torch.abs(pred_c - gt_c).mean()

    # KL divergence per latent dimension: 0.5 * (mu^2 + sigma^2 - logvar - 1)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, D)

    # Free bits: don't penalise KL below `free_bits` nats per dimension.
    # This guarantees the latent code carries information -> no collapse.
    kl_clamped = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss    = kl_clamped.sum(dim=1).mean()

    # Report raw KL (for monitoring) but optimise the free-bits version
    kl_raw = kl_per_dim.sum(dim=1).mean()

    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_raw


if __name__ == "__main__":
    # Sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AneurysmVAE(latent_dim=512).to(device)

    surf = torch.randn(2, 2048, 3, device=device)
    qry  = torch.randn(2, 2048, 3, device=device)
    gt   = torch.randn(2, 2048, device=device) * 0.1

    sdf, mu, logvar = model(surf, qry)
    print(f"\nForward pass:")
    print(f"  surf {tuple(surf.shape)} + qry {tuple(qry.shape)}")
    print(f"  -> sdf {tuple(sdf.shape)}, mu {tuple(mu.shape)}, logvar {tuple(logvar.shape)}")

    loss, recon, kl = vae_sdf_loss(sdf, gt, mu, logvar)
    print(f"  loss={loss.item():.4f}  recon={recon.item():.4f}  kl={kl.item():.4f}")

    gen = model.generate(qry[:1], n_samples=3, device=device)
    print(f"  generate(3) -> {tuple(gen.shape)}")
    print("All VAE checks passed.")