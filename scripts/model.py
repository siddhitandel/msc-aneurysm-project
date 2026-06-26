"""
model.py
========
Implicit Neural Reconstruction Model for 3D aneurysm shapes.

Architecture: MLP-based Occupancy Network
  Input  : (x, y, z) coordinate  →  3 floats
  Output : occupancy probability  →  1 float in [0, 1]

Design follows Occupancy Networks (Mescheder et al., CVPR 2019)
with a skip connection at the midpoint layer for better gradient flow.

              Input (3,)
                  │
            Linear(3→256)
            BatchNorm + ReLU
                  │
            Linear(256→256)
            BatchNorm + ReLU
                  │
            Linear(256→256)
            BatchNorm + ReLU
                  │
            Linear(256→256)
            BatchNorm + ReLU
                  │
         ┌────────┤  skip: concat input (3,) again
         │  Linear(256+3→256)
         │  BatchNorm + ReLU
         │        │
         │  Linear(256→256)
         │  BatchNorm + ReLU
         │        │
         │  Linear(256→256)
         │  BatchNorm + ReLU
         │        │
         │  Linear(256→1)
         └──────► Sigmoid → occupancy ∈ [0, 1]

The skip connection (concatenating the raw input halfway through)
helps the network retain fine geometric detail that can be lost in
deep MLPs — a technique used in NeRF and DeepSDF.
"""

import torch
import torch.nn as nn


class OccupancyMLP(nn.Module):
    """
    MLP-based implicit occupancy network.

    Args:
      hidden_dim   : width of each hidden layer (default 256)
      n_layers     : total number of hidden layers (default 8)
      skip_layer   : which layer index to inject the skip connection (default 4)
      dropout      : dropout probability, 0.0 = disabled (default 0.0)

    Forward:
      points : (B, N, 3)  or  (N, 3)  float32
      returns: (B, N, 1)  or  (N, 1)  float32  occupancy in [0, 1]
    """

    def __init__(self,
                 hidden_dim: int   = 256,
                 n_layers:   int   = 8,
                 skip_layer: int   = 4,
                 dropout:    float = 0.0):
        super().__init__()

        self.skip_layer = skip_layer
        self.n_layers   = n_layers
        input_dim       = 3

        layers = []
        in_dim = input_dim

        for i in range(n_layers):
            # At skip_layer, concatenate the original input again
            if i == skip_layer:
                in_dim = hidden_dim + input_dim

            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            in_dim = hidden_dim

        # Final output layer — no activation here, sigmoid applied separately
        layers.append(nn.Linear(hidden_dim, 1))

        self.layers = nn.ModuleList(layers)
        self.sigmoid = nn.Sigmoid()

        # Weight initialisation — Kaiming for ReLU networks
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
          points : (..., 3)  any leading batch dims are supported

        Returns:
          occ    : (..., 1)  occupancy probability
        """
        orig_shape = points.shape          # save for reshaping
        input_pts  = points                # keep original for skip

        # Flatten to (M, 3) for BatchNorm1d which needs 2D input
        pts = points.reshape(-1, 3)
        inp = pts.clone()                  # skip connection source

        layer_idx = 0
        hidden_layer_count = 0

        x = pts
        for module in self.layers:
            # Inject skip connection before the skip_layer-th Linear
            if (isinstance(module, nn.Linear) and
                    hidden_layer_count == self.skip_layer and
                    x.shape[-1] != 1):  # not the final output layer
                x = torch.cat([x, inp], dim=-1)

            if isinstance(module, nn.Linear):
                x = module(x)
                hidden_layer_count += 1
            else:
                x = module(x)

        # x is now (M, 1)
        occ = self.sigmoid(x)

        # Reshape back to match input leading dims
        out_shape = orig_shape[:-1] + (1,)
        return occ.reshape(out_shape)


# ─────────────────────────────────────────────────────────────────────────────
# Smaller / larger presets for ablation study (model capacity experiment)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(size: str = "medium") -> OccupancyMLP:
    """
    Convenience factory for the model capacity ablation study.

    Sizes:
      'small'  : 4 layers × 128 units  (~100K params)
      'medium' : 8 layers × 256 units  (~660K params)  ← default
      'large'  : 8 layers × 512 units  (~2.6M params)
    """
    configs = {
        "small"  : dict(hidden_dim=128, n_layers=4, skip_layer=2),
        "medium" : dict(hidden_dim=256, n_layers=8, skip_layer=4),
        "large"  : dict(hidden_dim=512, n_layers=8, skip_layer=4),
    }
    if size not in configs:
        raise ValueError(f"size must be one of {list(configs)}. Got: {size}")

    model = OccupancyMLP(**configs[size])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"OccupancyMLP [{size}]: {n_params:,} trainable parameters")
    return model


if __name__ == "__main__":
    # Quick sanity check
    for size in ["small", "medium", "large"]:
        m = build_model(size)
        x = torch.randn(2, 4096, 3)        # batch of 2 shapes, 4096 points each
        out = m(x)
        print(f"  [{size}] input {tuple(x.shape)} → output {tuple(out.shape)}")
        assert out.shape == (2, 4096, 1)
        assert out.min() >= 0.0 and out.max() <= 1.0
    print("All model checks passed.")