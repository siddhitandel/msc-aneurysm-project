"""
cvae_model.py
=============
Stage 2, Part 2: Conditional VAE (CVAE).

Extends the VAE to condition generation on rupture status, so we can
generate SPECIFICALLY ruptured or unruptured shapes. This is what lets us
synthesise minority-class (ruptured) examples to rebalance the dataset for
the augmentation experiment.

Conditioning mechanism:
  - The class label (0=unruptured, 1=ruptured) is embedded and concatenated
    to BOTH the encoder's global feature and the decoder's latent input.
  - At generation time we pick the class, sample z ~ N(0,I), and decode.

Reuses the proven-stable components from ae_models.py (LayerNorm encoder,
positional encoding, raw-linear SDF output, small-logvar init).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ae_models import PositionalEncoding


class ConditionalEncoder(nn.Module):
    def __init__(self, latent_dim=512, n_classes=2, class_embed=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.class_emb = nn.Embedding(n_classes, class_embed)

        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),    nn.ReLU(inplace=True),
            nn.Linear(64, 128),  nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 512), nn.ReLU(inplace=True),
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(512 + class_embed, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
        )
        self.fc_mu     = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -5.0)

    def forward(self, points, labels):
        B, N, _ = points.shape
        x = points.reshape(B * N, 3)
        for layer in self.point_mlp:
            x = layer(x)
        x = x.reshape(B, N, -1).max(dim=1).values         # (B, 512)
        c = self.class_emb(labels)                         # (B, class_embed)
        x = torch.cat([x, c], dim=-1)
        x = self.fc_shared(x)
        mu = self.fc_mu(x)
        logvar = torch.clamp(self.fc_logvar(x), -10.0, 10.0)
        return mu, logvar


class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=512, n_layers=8,
                 skip_layer=4, n_freqs=3, n_classes=2, class_embed=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.skip_layer = skip_layer
        self.class_emb = nn.Embedding(n_classes, class_embed)
        self.pos_enc = PositionalEncoding(n_freqs=n_freqs)
        coord_dim = self.pos_enc.out_dim
        input_dim = coord_dim + latent_dim + class_embed

        self.layers = nn.ModuleList()
        in_dim = input_dim
        for i in range(n_layers):
            if i == skip_layer:
                in_dim = hidden_dim + input_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.output_layer = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, points, z, labels):
        B, N, _ = points.shape
        pe = self.pos_enc(points)
        z_exp = z.unsqueeze(1).expand(B, N, self.latent_dim)
        c = self.class_emb(labels).unsqueeze(1).expand(B, N, -1)
        inp = torch.cat([pe, z_exp, c], dim=-1)
        x = inp.reshape(B * N, -1)
        inp_flat = x.clone()
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                x = torch.cat([x, inp_flat], dim=-1)
            x = F.relu(layer(x), inplace=True)
        sdf = self.output_layer(x)
        return sdf.reshape(B, N, 1)


class AneurysmCVAE(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=512, n_freqs=3,
                 n_classes=2, class_embed=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.encoder = ConditionalEncoder(latent_dim, n_classes, class_embed)
        self.decoder = ConditionalDecoder(latent_dim, hidden_dim,
                                          n_freqs=n_freqs, n_classes=n_classes,
                                          class_embed=class_embed)
        n = sum(p.numel() for p in self.parameters())
        print(f"AneurysmCVAE: {n:,} params, {n_classes} classes")

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, points, labels):
        return self.encoder(points, labels)

    def decode(self, query_points, z, labels):
        return self.decoder(query_points, z, labels)

    def forward(self, surface_points, query_points, labels):
        mu, logvar = self.encode(surface_points, labels)
        z = self.reparameterise(mu, logvar)
        sdf = self.decode(query_points, z, labels)
        return sdf, mu, logvar

    @torch.no_grad()
    def generate(self, query_points, labels, device=None, truncation=1.0):
        """Generate shapes of the given class labels. labels: (n,) long tensor."""
        if device is None:
            device = next(self.parameters()).device
        n = labels.shape[0]
        z = torch.randn(n, self.latent_dim, device=device) * truncation
        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)
        qp = query_points.expand(n, -1, -1)
        return self.decode(qp, z, labels)


def cvae_sdf_loss(pred_sdf, gt_sdf, mu, logvar, kl_weight=1e-5,
                  clamp_dist=0.2, free_bits=0.5):
    pred = pred_sdf.squeeze(-1) if pred_sdf.dim() == 3 else pred_sdf
    pred_c = torch.clamp(pred, -clamp_dist, clamp_dist)
    gt_c   = torch.clamp(gt_sdf, -clamp_dist, clamp_dist)
    recon  = torch.abs(pred_c - gt_c).mean()
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.clamp(kl_per_dim, min=free_bits).sum(dim=1).mean()
    kl_raw  = kl_per_dim.sum(dim=1).mean()
    return recon + kl_weight * kl_loss, recon, kl_raw


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = AneurysmCVAE().to(device)
    surf = torch.randn(4, 2048, 3, device=device)
    qry  = torch.randn(4, 2048, 3, device=device)
    gt   = torch.randn(4, 2048, device=device) * 0.1
    lab  = torch.randint(0, 2, (4,), device=device)
    sdf, mu, logvar = m(surf, qry, lab)
    loss, recon, kl = cvae_sdf_loss(sdf, gt, mu, logvar)
    print(f"loss={loss.item():.4f} recon={recon.item():.4f} kl={kl.item():.4f}")
    # Generate 3 ruptured shapes
    gen = m.generate(qry[:1], torch.ones(3, dtype=torch.long, device=device), device)
    print(f"generated ruptured: {tuple(gen.shape)}")
    print("CVAE checks passed.")