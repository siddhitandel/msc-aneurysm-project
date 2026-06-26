"""
autoencoder.py
==============
Generative Autoencoder for 3D aneurysm shape synthesis.

Architecture:
  Encoder : PointNet-style MLP on surface point cloud -> latent vector z
  Decoder : Implicit occupancy MLP conditioned on z   -> occupancy(x,y,z | z)

The encoder compresses each aneurysm surface point cloud (4096 x 3) into
a compact latent vector (latent_dim,). The decoder is the same implicit
MLP as the reconstruction model but now takes (xyz | z) as input, where
z is the latent code broadcast to every query point.

Once trained:
  - Reconstruction : encode a real shape -> z -> decode -> reconstructed shape
  - Generation     : sample z ~ N(0,I)  -> decode -> new synthetic shape
  - Interpolation  : lerp between z1 and z2 -> decode -> morphing sequence

Reference architecture inspired by:
  - IM-NET (Chen & Zhang, CVPR 2019)
  - Occupancy Networks (Mescheder et al., CVPR 2019)
  - DeepSDF (Park et al., CVPR 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Encoder  (PointNet-style)
# ─────────────────────────────────────────────────────────────────────────────

class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder: processes each point independently,
    then aggregates with global max pooling to get a fixed-size latent vector.

    Input  : (B, N, 3)  surface point cloud
    Output : (B, latent_dim)  latent code z

    Architecture:
      Per-point MLP: 3 -> 64 -> 128 -> 256 -> 512
      Global max pool over N points
      FC head: 512 -> 256 -> latent_dim
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # Per-point feature extraction
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # Global aggregation head
        self.fc_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
          points : (B, N, 3)

        Returns:
          z : (B, latent_dim)
        """
        B, N, _ = points.shape

        # Apply per-point MLP — BatchNorm1d expects (B*N, C)
        x = points.reshape(B * N, 3)
        x = self._apply_point_mlp(x, B, N)   # (B, N, 512)

        # Global max pooling over points dimension
        x = x.max(dim=1).values               # (B, 512)

        # FC head to latent space
        z = self.fc_head(x)                   # (B, latent_dim)
        return z

    def _apply_point_mlp(self, x: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """Apply the per-point MLP layer by layer, handling BatchNorm correctly."""
        for layer in self.point_mlp:
            if isinstance(layer, nn.BatchNorm1d):
                # BatchNorm1d works on (B*N, C) — already in that shape
                x = layer(x)
            else:
                x = layer(x)
        # Reshape to (B, N, 512)
        return x.reshape(B, N, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder  (Implicit MLP conditioned on latent code)
# ─────────────────────────────────────────────────────────────────────────────

class ImplicitDecoder(nn.Module):
    """
    Implicit occupancy decoder conditioned on a latent code z.

    Input  : query points (B, N, 3)  +  latent code (B, latent_dim)
    Output : occupancy probabilities (B, N, 1)

    The latent code is concatenated with each query point coordinate,
    so the network learns: f(x, y, z, z_latent) -> occupancy.

    Architecture: 8-layer MLP with skip connection at layer 4.
    Input dim = 3 + latent_dim at first layer.
    """

    def __init__(self,
                 latent_dim:  int   = 256,
                 hidden_dim:  int   = 256,
                 n_layers:    int   = 8,
                 skip_layer:  int   = 4):
        super().__init__()

        self.latent_dim = latent_dim
        self.skip_layer = skip_layer
        self.n_layers   = n_layers

        input_dim = 3 + latent_dim  # xyz concatenated with z

        self.layers = nn.ModuleList()
        in_dim = input_dim

        for i in range(n_layers):
            # Skip connection: re-inject input at skip_layer
            if i == skip_layer:
                in_dim = hidden_dim + input_dim

            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid      = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                points: torch.Tensor,
                z:      torch.Tensor) -> torch.Tensor:
        """
        Args:
          points : (B, N, 3)
          z      : (B, latent_dim)

        Returns:
          occ    : (B, N, 1)  occupancy in [0, 1]
        """
        B, N, _ = points.shape

        # Broadcast z to every query point: (B, latent_dim) -> (B, N, latent_dim)
        z_expanded = z.unsqueeze(1).expand(B, N, self.latent_dim)

        # Concatenate: (B, N, 3 + latent_dim)
        inp = torch.cat([points, z_expanded], dim=-1)

        # Flatten to (B*N, input_dim) for linear layers
        x    = inp.reshape(B * N, -1)
        inp_flat = x.clone()   # for skip connection

        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                x = torch.cat([x, inp_flat], dim=-1)
            x = F.relu(layer(x), inplace=True)

        x   = self.output_layer(x)           # (B*N, 1)
        occ = self.sigmoid(x)
        return occ.reshape(B, N, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Full Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class AneurysmAutoencoder(nn.Module):
    """
    Full generative autoencoder combining PointNetEncoder + ImplicitDecoder.

    Forward pass (training):
      1. Encode surface point cloud -> latent z
      2. Decode: for each space query point, predict occupancy given z
      3. Compute BCE loss against ground-truth occupancy labels

    Generation (inference):
      - Sample z ~ N(0, I) and decode to get a new shape
      - Or encode two real shapes z1, z2 and interpolate between them

    Args:
      latent_dim  : size of latent space (default 256)
      hidden_dim  : decoder hidden layer width (default 256)
    """

    def __init__(self,
                 latent_dim: int = 256,
                 hidden_dim: int = 256):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder    = PointNetEncoder(latent_dim=latent_dim)
        self.decoder    = ImplicitDecoder(latent_dim=latent_dim,
                                          hidden_dim=hidden_dim)

        n_enc = sum(p.numel() for p in self.encoder.parameters())
        n_dec = sum(p.numel() for p in self.decoder.parameters())
        print(f"AneurysmAutoencoder:")
        print(f"  Encoder params : {n_enc:,}")
        print(f"  Decoder params : {n_dec:,}")
        print(f"  Total params   : {n_enc + n_dec:,}")
        print(f"  Latent dim     : {latent_dim}")

    def encode(self, surface_points: torch.Tensor) -> torch.Tensor:
        """
        Encode a surface point cloud to a latent vector.
        Args:   surface_points : (B, N, 3)
        Returns: z             : (B, latent_dim)
        """
        return self.encoder(surface_points)

    def decode(self,
               query_points: torch.Tensor,
               z:            torch.Tensor) -> torch.Tensor:
        """
        Decode query points given latent code z.
        Args:   query_points : (B, N, 3)
                z            : (B, latent_dim)
        Returns: occ         : (B, N, 1)
        """
        return self.decoder(query_points, z)

    def forward(self,
                surface_points: torch.Tensor,
                query_points:   torch.Tensor) -> tuple:
        """
        Full forward pass for training.
        Args:
          surface_points : (B, N_surf, 3)  encoder input
          query_points   : (B, N_query, 3) decoder query points
        Returns:
          occ : (B, N_query, 1)  predicted occupancy
          z   : (B, latent_dim)  latent code (for regularisation)
        """
        z   = self.encode(surface_points)
        occ = self.decode(query_points, z)
        return occ, z

    def interpolate(self,
                    surface_a:   torch.Tensor,
                    surface_b:   torch.Tensor,
                    query_points: torch.Tensor,
                    steps:       int = 8) -> list:
        """
        Interpolate between two shapes in latent space.
        Returns a list of `steps` occupancy grids.

        Args:
          surface_a    : (1, N, 3)  first shape
          surface_b    : (1, N, 3)  second shape
          query_points : (1, M, 3)  3D query grid
          steps        : number of interpolation steps

        Returns:
          list of (1, M, 1) tensors, one per interpolation step
        """
        with torch.no_grad():
            z_a = self.encode(surface_a)   # (1, latent_dim)
            z_b = self.encode(surface_b)   # (1, latent_dim)

            results = []
            for t in torch.linspace(0, 1, steps):
                z_t   = (1 - t) * z_a + t * z_b
                occ_t = self.decode(query_points, z_t)
                results.append(occ_t)

        return results

    def generate(self,
                 query_points: torch.Tensor,
                 n_samples:    int = 1,
                 device:       torch.device = None) -> torch.Tensor:
        """
        Generate new shapes by sampling from the prior N(0, I).

        Args:
          query_points : (1, M, 3) or (M, 3)  query grid
          n_samples    : number of shapes to generate
          device       : torch device

        Returns:
          occ : (n_samples, M, 1)  occupancy for each generated shape
        """
        if device is None:
            device = next(self.parameters()).device

        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)

            if query_points.dim() == 2:
                query_points = query_points.unsqueeze(0)

            # Expand query points to match n_samples
            qp = query_points.expand(n_samples, -1, -1)
            occ = self.decode(qp, z)

        return occ