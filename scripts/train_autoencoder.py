"""
train_autoencoder.py
====================
Training script for the AneurysmAutoencoder.

Unlike the reconstruction model (which trains one MLP per shape),
the autoencoder trains a single shared model across all shapes.
The encoder learns a latent space that captures shape variation,
and the decoder learns to reconstruct any shape from its latent code.

Loss:
  BCE reconstruction loss between predicted and true occupancy labels.
  Optional KL regularisation (set --kl_weight > 0) to encourage a
  well-structured latent space better suited for random sampling.

Usage (quick test — 5 epochs):
  python scripts/train_autoencoder.py \
      --processed_dir aneurysm_project/data/processed \
      --out_dir       aneurysm_project/models/autoencoder \
      --n_epochs      5 \
      --latent_dim    256

Usage (full training via SLURM):
  sbatch jobs/train_autoencoder.job
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoencoder import AneurysmAutoencoder
from dataset import OccupancyDataset, SurfaceDataset


# ─────────────────────────────────────────────────────────────────────────────
# Collate function — pairs surface points with space query points
# ─────────────────────────────────────────────────────────────────────────────

class PairedDataset(torch.utils.data.Dataset):
    """
    Pairs surface point clouds with their corresponding space query points.
    Both datasets must have the same split and seed so indices align.
    """

    def __init__(self, surface_ds: SurfaceDataset, occupancy_ds: OccupancyDataset):
        assert len(surface_ds) == len(occupancy_ds), \
            f"Dataset size mismatch: {len(surface_ds)} vs {len(occupancy_ds)}"
        assert surface_ds.ids == occupancy_ds.ids, \
            "Dataset IDs don't match — check split/seed parameters"
        self.surface_ds   = surface_ds
        self.occupancy_ds = occupancy_ds

    def __len__(self):
        return len(self.surface_ds)

    def __getitem__(self, idx):
        surf = self.surface_ds[idx]
        occ  = self.occupancy_ds[idx]
        return {
            "surface_points" : surf["points"],        # (N_surf, 3)
            "query_points"   : occ["points"],         # (N_query, 3)
            "labels"         : occ["labels"],         # (N_query,)
            "rupture_label"  : surf["rupture_label"],
            "id"             : surf["id"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"Latent dim   : {args.latent_dim}")
    print(f"Epochs       : {args.n_epochs}")
    print(f"Batch size   : {args.batch_size}")
    print(f"LR           : {args.lr}")
    print(f"KL weight    : {args.kl_weight}\n")

    # ── Datasets ───────────────────────────────────────────────────────────
    def make_paired(split):
        surf = SurfaceDataset(args.processed_dir, split=split,
                              n_points=2048, augment=(split == "train"))
        occ  = OccupancyDataset(args.processed_dir, split=split,
                                n_points=2048)
        return PairedDataset(surf, occ)

    train_ds = make_paired("train")
    val_ds   = make_paired("val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}\n")

    # ── Model ──────────────────────────────────────────────────────────────
    model     = AneurysmAutoencoder(latent_dim=args.latent_dim,
                                    hidden_dim=args.hidden_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.n_epochs, eta_min=1e-6
    )
    criterion = nn.BCELoss()

    # ── Output dir ─────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    history        = {"train_loss": [], "val_loss": [], "lr": []}
    t_start        = time.time()

    for epoch in range(1, args.n_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            surf_pts = batch["surface_points"].to(device)  # (B, N, 3)
            qry_pts  = batch["query_points"].to(device)    # (B, N, 3)
            labels   = batch["labels"].to(device)          # (B, N)

            optimiser.zero_grad()
            occ, z = model(surf_pts, qry_pts)              # (B, N, 1), (B, D)
            occ     = occ.squeeze(-1)                      # (B, N)

            # Reconstruction loss
            recon_loss = criterion(occ, labels)

            # Optional: latent regularisation (encourage z ~ N(0,I))
            # This is a soft prior — not a true VAE but helps with generation
            kl_loss = 0.0
            if args.kl_weight > 0:
                kl_loss = args.kl_weight * (z ** 2).mean()

            loss = recon_loss + kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += recon_loss.item()

        train_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                surf_pts = batch["surface_points"].to(device)
                qry_pts  = batch["query_points"].to(device)
                labels   = batch["labels"].to(device)

                occ, _ = model(surf_pts, qry_pts)
                occ    = occ.squeeze(-1)
                val_loss += criterion(occ, labels).item()

        val_loss /= len(val_loader)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # ── Logging ────────────────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"Epoch {epoch:4d}/{args.n_epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {elapsed:.0f}s")

        # ── Save best checkpoint ───────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch"            : epoch,
                "model_state_dict" : model.state_dict(),
                "optimiser_state"  : optimiser.state_dict(),
                "val_loss"         : val_loss,
                "train_loss"       : train_loss,
                "latent_dim"       : args.latent_dim,
                "hidden_dim"       : args.hidden_dim,
            }, out_dir / "best_model.pt")

        # ── Periodic checkpoint every 50 epochs ───────────────────────────
        if epoch % 50 == 0:
            torch.save({
                "epoch"            : epoch,
                "model_state_dict" : model.state_dict(),
                "val_loss"         : val_loss,
                "latent_dim"       : args.latent_dim,
                "hidden_dim"       : args.hidden_dim,
            }, out_dir / f"checkpoint_epoch{epoch:04d}.pt")

    # ── Save training history ──────────────────────────────────────────────
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {out_dir}/best_model.pt")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train generative autoencoder")
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--n_epochs",      type=int,   default=300)
    p.add_argument("--batch_size",    type=int,   default=8,
                   help="Number of shapes per batch (default: 8)")
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--latent_dim",    type=int,   default=256,
                   help="Latent space dimensionality (default: 256)")
    p.add_argument("--hidden_dim",    type=int,   default=256,
                   help="Decoder hidden layer width (default: 256)")
    p.add_argument("--kl_weight",     type=float, default=1e-4,
                   help="Weight for latent regularisation (default: 1e-4)")
    args = p.parse_args()
    train(args)