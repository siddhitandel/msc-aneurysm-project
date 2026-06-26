"""
train_vae.py
============
Training script for the AneurysmVAE (SDF + variational + positional encoding).

Key features:
  - SDF L1 reconstruction loss (clamped near surface)
  - KL divergence with ANNEALING: kl_weight ramps up over the first
    `kl_anneal_epochs` epochs. This prevents posterior collapse early
    in training (the encoder would otherwise ignore the input).
  - Cosine LR schedule, gradient clipping, best-checkpoint saving

Requires preprocessing run with --representation sdf.

Usage (quick test):
  python scripts/train_vae.py \
      --processed_dir aneurysm_project/data/processed_sdf \
      --out_dir       aneurysm_project/models/vae \
      --n_epochs      5

Full run via SLURM:
  sbatch jobs/train_vae.job
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vae_model import AneurysmVAE, vae_sdf_loss
from dataset import SurfaceDataset


# ─────────────────────────────────────────────────────────────────────────────
# Paired dataset: surface (encoder input) + space SDF (decoder target)
# ─────────────────────────────────────────────────────────────────────────────

class SDFPairedDataset(Dataset):
    """
    Pairs each shape's surface point cloud with its space SDF samples.
    Reads from preprocessed npz files (representation='sdf').
    """

    def __init__(self, processed_dir, split="train",
                 n_surface=2048, n_query=2048, seed=42, augment=False):
        self.processed_dir = Path(processed_dir)
        self.n_surface = n_surface
        self.n_query   = n_query
        self.augment   = augment and (split == "train")

        # Reuse SurfaceDataset's split logic to get consistent IDs
        surf_ds = SurfaceDataset(processed_dir, split=split,
                                 n_points=n_surface, seed=seed)
        self.ids       = surf_ds.ids
        self.label_map = surf_ds.label_map

    def __len__(self):
        return len(self.ids)

    def _augment_rotation(self, pts):
        """Random 3D rotation for augmentation."""
        angles = np.random.uniform(0, 2*np.pi, size=3)
        cx, cy, cz = np.cos(angles); sx, sy, sz = np.sin(angles)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        R  = (Rz @ Ry @ Rx).astype(np.float32)
        return pts @ R.T, R

    def __getitem__(self, idx):
        aneurysm_id = self.ids[idx]

        surf_data = np.load(self.processed_dir / "surface" / f"{aneurysm_id}.npz")
        space_data = np.load(self.processed_dir / "space"  / f"{aneurysm_id}.npz")

        surf_pts = surf_data["points"].copy()           # (Ns, 3)
        qry_pts  = space_data["points"].copy()          # (Nq, 3)
        sdfs     = space_data["sdfs"].copy()            # (Nq,)

        # Subsample surface
        if len(surf_pts) > self.n_surface:
            idx_s = np.random.choice(len(surf_pts), self.n_surface, replace=False)
            surf_pts = surf_pts[idx_s]

        # Subsample query points + matching sdfs
        if len(qry_pts) > self.n_query:
            idx_q = np.random.choice(len(qry_pts), self.n_query, replace=False)
            qry_pts = qry_pts[idx_q]
            sdfs    = sdfs[idx_q]

        # Augmentation: rotate surface AND query points by the SAME rotation
        if self.augment:
            surf_pts, R = self._augment_rotation(surf_pts)
            qry_pts     = qry_pts @ R.T   # SDF values are rotation-invariant

        return {
            "surface_points" : torch.from_numpy(surf_pts).float(),
            "query_points"   : torch.from_numpy(qry_pts).float(),
            "sdfs"           : torch.from_numpy(sdfs).float(),
            "rupture_label"  : torch.tensor(self.label_map.get(aneurysm_id, -1)),
            "id"             : aneurysm_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# KL annealing schedule
# ─────────────────────────────────────────────────────────────────────────────

def kl_weight_schedule(epoch, max_weight, anneal_epochs):
    """
    Linear KL annealing: 0 -> max_weight over anneal_epochs, then constant.
    Prevents posterior collapse by letting the model first learn to
    reconstruct, then gradually regularise the latent space.
    """
    if epoch >= anneal_epochs:
        return max_weight
    return max_weight * (epoch / anneal_epochs)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    print(f"Latent dim      : {args.latent_dim}")
    print(f"Epochs          : {args.n_epochs}")
    print(f"KL max weight   : {args.kl_weight}")
    print(f"KL anneal epochs: {args.kl_anneal_epochs}")
    print(f"Augmentation    : {args.augment}\n")

    # Datasets
    train_ds = SDFPairedDataset(args.processed_dir, "train",
                                n_surface=2048, n_query=2048,
                                augment=args.augment)
    val_ds   = SDFPairedDataset(args.processed_dir, "val",
                                n_surface=2048, n_query=2048,
                                augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} shapes  Val: {len(val_ds)} shapes\n")

    # Model
    model = AneurysmVAE(latent_dim=args.latent_dim,
                        hidden_dim=args.hidden_dim,
                        n_freqs=args.n_freqs).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.n_epochs, eta_min=1e-6)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history  = {"train_recon": [], "train_kl": [], "val_recon": [],
                "kl_weight": [], "lr": []}
    t0 = time.time()

    for epoch in range(1, args.n_epochs + 1):
        kl_w = kl_weight_schedule(epoch, args.kl_weight, args.kl_anneal_epochs)

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_recon, tr_kl = 0.0, 0.0
        for batch in train_loader:
            surf = batch["surface_points"].to(device)
            qry  = batch["query_points"].to(device)
            gt   = batch["sdfs"].to(device)

            optimiser.zero_grad()
            pred_sdf, mu, logvar = model(surf, qry)
            loss, recon, kl = vae_sdf_loss(pred_sdf, gt, mu, logvar,
                                           kl_weight=kl_w,
                                           clamp_dist=args.clamp_dist,
                                           free_bits=args.free_bits,
                                           surface_weight=args.surface_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tr_recon += recon.item()
            tr_kl    += kl.item()

        tr_recon /= len(train_loader)
        tr_kl    /= len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_recon = 0.0
        with torch.no_grad():
            for batch in val_loader:
                surf = batch["surface_points"].to(device)
                qry  = batch["query_points"].to(device)
                gt   = batch["sdfs"].to(device)
                pred_sdf, mu, logvar = model(surf, qry)
                _, recon, _ = vae_sdf_loss(pred_sdf, gt, mu, logvar,
                                           kl_weight=kl_w,
                                           clamp_dist=args.clamp_dist,
                                           free_bits=args.free_bits,
                                           surface_weight=args.surface_weight)
                val_recon += recon.item()
        val_recon /= len(val_loader)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        history["train_recon"].append(tr_recon)
        history["train_kl"].append(tr_kl)
        history["val_recon"].append(val_recon)
        history["kl_weight"].append(kl_w)
        history["lr"].append(lr)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.n_epochs} | "
                  f"Recon: {tr_recon:.4f} | KL: {tr_kl:.2f} | "
                  f"Val: {val_recon:.4f} | kl_w: {kl_w:.1e} | "
                  f"LR: {lr:.1e} | {time.time()-t0:.0f}s")

        # Save best by validation reconstruction
        if val_recon < best_val:
            best_val = val_recon
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_recon": val_recon, "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim, "n_freqs": args.n_freqs,
                "representation": "sdf",
            }, out_dir / "best_vae.pt")

        if epoch % 100 == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_recon": val_recon, "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim, "n_freqs": args.n_freqs,
                "representation": "sdf",
            }, out_dir / f"vae_epoch{epoch:04d}.pt")

    with open(out_dir / "vae_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete in {(time.time()-t0)/60:.1f} min")
    print(f"Best val recon: {best_val:.4f}")
    print(f"Checkpoint    : {out_dir}/best_vae.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir",    type=Path, required=True)
    p.add_argument("--out_dir",          type=Path, required=True)
    p.add_argument("--n_epochs",         type=int,   default=800)
    p.add_argument("--batch_size",       type=int,   default=8)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--latent_dim",       type=int,   default=512)
    p.add_argument("--hidden_dim",       type=int,   default=512)
    p.add_argument("--n_freqs",          type=int,   default=6)
    p.add_argument("--kl_weight",        type=float, default=1e-5)
    p.add_argument("--kl_anneal_epochs", type=int,   default=100)
    p.add_argument("--clamp_dist",       type=float, default=0.2)
    p.add_argument("--free_bits",        type=float, default=0.5,
                   help="Min KL per latent dim before penalty (prevents collapse)")
    p.add_argument("--surface_weight",   type=float, default=5.0,
                   help="Extra loss weight on near-surface points")
    p.add_argument("--augment",          action="store_true")
    args = p.parse_args()
    train(args)