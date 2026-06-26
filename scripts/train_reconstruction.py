"""
train_reconstruction.py
=======================
Training script for the implicit occupancy reconstruction model.

Trains one OccupancyMLP per aneurysm shape (per-shape overfitting),
then evaluates reconstruction quality using Chamfer distance.

Usage:
  # Train on a single shape (quick test)
  python scripts/train_reconstruction.py \
      --processed_dir aneurysm_project/data/processed \
      --out_dir       aneurysm_project/models/reconstruction \
      --mode          single \
      --shape_id      p043_HAARCREcDAAQDQcbHgANDRQM

  # Train one model per shape across the full dataset
  python scripts/train_reconstruction.py \
      --processed_dir aneurysm_project/data/processed \
      --out_dir       aneurysm_project/models/reconstruction \
      --mode          all

  # Ablation: compare small / medium / large model capacity
  python scripts/train_reconstruction.py \
      --processed_dir aneurysm_project/data/processed \
      --out_dir       aneurysm_project/models/reconstruction \
      --mode          ablation \
      --n_shapes      20
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import build_model, OccupancyMLP


# ─────────────────────────────────────────────────────────────────────────────
# Chamfer Distance
# ─────────────────────────────────────────────────────────────────────────────

def chamfer_distance(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """
    Compute Chamfer distance between two point clouds.
    Both inputs: (N, 3) numpy arrays.

    CD = mean of nearest-neighbour distances in both directions.
    Lower is better.
    """
    try:
        import point_cloud_utils as pcu
        cd = pcu.chamfer_distance(pred_pts.astype(np.float32),
                                  gt_pts.astype(np.float32))
        return float(cd)
    except ImportError:
        # Fallback: pure numpy (slower but always works)
        # pred → gt
        diff1  = pred_pts[:, None, :] - gt_pts[None, :, :]   # (N, M, 3)
        dist1  = np.sqrt((diff1 ** 2).sum(-1))                 # (N, M)
        min1   = dist1.min(axis=1).mean()                      # scalar

        # gt → pred
        diff2  = gt_pts[:, None, :] - pred_pts[None, :, :]
        dist2  = np.sqrt((diff2 ** 2).sum(-1))
        min2   = dist2.min(axis=1).mean()

        return float((min1 + min2) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh extraction from trained occupancy network
# ─────────────────────────────────────────────────────────────────────────────

def extract_surface_points(model: OccupancyMLP,
                           device: torch.device,
                           resolution: int = 64,
                           threshold: float = 0.5,
                           n_surface_pts: int = 4096) -> np.ndarray:
    """
    Extract a point cloud from a trained occupancy network by:
      1. Evaluating occupancy on a dense 3D grid
      2. Keeping points near the decision boundary (occ ~ threshold)
      3. Returning as a point cloud for Chamfer distance evaluation

    Args:
      model       : trained OccupancyMLP
      device      : cuda or cpu
      resolution  : grid resolution per axis (64^3 = 262K points)
      threshold   : occupancy decision boundary (default 0.5)
      n_surface_pts : number of points to return

    Returns:
      points : (n_surface_pts, 3)  float32
    """
    model.eval()
    lin = np.linspace(-1.0, 1.0, resolution)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid_pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)

    # Evaluate in batches to avoid OOM
    batch_size = 32768
    all_occ = []
    with torch.no_grad():
        for i in range(0, len(grid_pts), batch_size):
            batch = torch.from_numpy(grid_pts[i:i+batch_size]).to(device)
            occ   = model(batch).squeeze(-1).cpu().numpy()
            all_occ.append(occ)

    all_occ = np.concatenate(all_occ)

    # Points near the surface: occupancy close to threshold
    near_surface = np.abs(all_occ - threshold) < 0.1
    surface_pts  = grid_pts[near_surface]

    if len(surface_pts) == 0:
        # Fallback: take points just inside the surface
        surface_pts = grid_pts[all_occ > threshold]

    if len(surface_pts) == 0:
        return np.zeros((n_surface_pts, 3), dtype=np.float32)

    # Subsample to fixed size
    if len(surface_pts) >= n_surface_pts:
        idx = np.random.choice(len(surface_pts), n_surface_pts, replace=False)
    else:
        idx = np.random.choice(len(surface_pts), n_surface_pts, replace=True)

    return surface_pts[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Single shape training
# ─────────────────────────────────────────────────────────────────────────────

def train_single_shape(shape_id: str,
                       processed_dir: Path,
                       out_dir: Path,
                       model_size: str   = "medium",
                       n_epochs: int     = 500,
                       lr: float         = 1e-4,
                       batch_size: int   = 4096,
                       device: torch.device = None) -> dict:
    """
    Train one OccupancyMLP on a single aneurysm shape.
    Returns a dict of metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load preprocessed data
    space_path   = processed_dir / "space"   / f"{shape_id}.npz"
    surface_path = processed_dir / "surface" / f"{shape_id}.npz"

    if not space_path.exists():
        raise FileNotFoundError(f"Space file not found: {space_path}")

    space_data  = np.load(space_path)
    points      = torch.from_numpy(space_data["points"]).to(device)  # (N, 3)
    labels      = torch.from_numpy(space_data["labels"]).to(device)  # (N,)
    gt_surface  = np.load(surface_path)["points"]                     # (N, 3) numpy

    # DataLoader
    dataset    = TensorDataset(points, labels)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model      = build_model(model_size).to(device)
    optimiser  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimiser, step_size=200, gamma=0.5)
    criterion  = nn.BCELoss()

    # Training loop
    model.train()
    losses = []
    t0     = time.time()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_pts, batch_labels in loader:
            optimiser.zero_grad()
            pred  = model(batch_pts).squeeze(-1)
            loss  = criterion(pred, batch_labels)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        scheduler.step()
        losses.append(epoch_loss / len(loader))

        if (epoch + 1) % 100 == 0:
            print(f"  [{shape_id[:20]}] Epoch {epoch+1:4d}/{n_epochs} "
                  f"| Loss: {losses[-1]:.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e} "
                  f"| Time: {time.time()-t0:.0f}s")

    # Evaluate: extract predicted surface and compute Chamfer distance
    pred_surface = extract_surface_points(model, device)
    cd           = chamfer_distance(pred_surface, gt_surface)

    print(f"  [{shape_id[:20]}] Chamfer distance: {cd:.6f}")

    # Save model checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{shape_id}.pt"
    torch.save({
        "model_state_dict" : model.state_dict(),
        "model_size"       : model_size,
        "shape_id"         : shape_id,
        "final_loss"       : losses[-1],
        "chamfer_distance" : cd,
        "n_epochs"         : n_epochs,
        "lr"               : lr,
    }, ckpt_path)

    return {
        "shape_id"         : shape_id,
        "model_size"       : model_size,
        "final_loss"       : losses[-1],
        "chamfer_distance" : cd,
        "train_time_s"     : time.time() - t0,
        "losses"           : losses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train implicit occupancy reconstruction model")
    p.add_argument("--processed_dir", type=Path, required=True,
                   help="Path to data/processed/")
    p.add_argument("--out_dir",       type=Path, required=True,
                   help="Output directory for model checkpoints")
    p.add_argument("--mode",          type=str, default="single",
                   choices=["single", "all", "ablation"],
                   help="single: one shape | all: full dataset | ablation: capacity study")
    p.add_argument("--shape_id",      type=str, default=None,
                   help="Shape ID for --mode single")
    p.add_argument("--model_size",    type=str, default="medium",
                   choices=["small", "medium", "large"],
                   help="Model size preset (default: medium)")
    p.add_argument("--n_epochs",      type=int, default=500,
                   help="Training epochs per shape (default: 500)")
    p.add_argument("--lr",            type=float, default=1e-4,
                   help="Learning rate (default: 1e-4)")
    p.add_argument("--batch_size",    type=int, default=4096,
                   help="Points per batch (default: 4096)")
    p.add_argument("--n_shapes",      type=int, default=None,
                   help="Limit number of shapes (default: all)")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processed_dir = Path(args.processed_dir)

    # Load manifest to get all shape IDs
    with open(processed_dir / "manifest.json") as f:
        manifest = json.load(f)
    all_ids = manifest["aneurysm_ids"]

    if args.mode == "single":
        # Train on one specified shape
        if args.shape_id is None:
            args.shape_id = all_ids[0]
            print(f"No --shape_id given, using first: {args.shape_id}")

        metrics = train_single_shape(
            shape_id      = args.shape_id,
            processed_dir = processed_dir,
            out_dir       = args.out_dir,
            model_size    = args.model_size,
            n_epochs      = args.n_epochs,
            lr            = args.lr,
            batch_size    = args.batch_size,
            device        = device,
        )
        print(f"\nFinal Chamfer Distance: {metrics['chamfer_distance']:.6f}")

    elif args.mode == "all":
        # Train on every shape in the dataset
        shape_ids = all_ids[:args.n_shapes] if args.n_shapes else all_ids
        all_metrics = []

        print(f"Training on {len(shape_ids)} shapes...")
        for shape_id in tqdm(shape_ids, desc="Shapes"):
            try:
                m = train_single_shape(
                    shape_id      = shape_id,
                    processed_dir = processed_dir,
                    out_dir       = args.out_dir,
                    model_size    = args.model_size,
                    n_epochs      = args.n_epochs,
                    lr            = args.lr,
                    batch_size    = args.batch_size,
                    device        = device,
                )
                all_metrics.append(m)
            except Exception as e:
                print(f"  ERROR {shape_id}: {e}")

        # Save summary
        summary = {
            "n_shapes"          : len(all_metrics),
            "mean_chamfer"      : float(np.mean([m["chamfer_distance"] for m in all_metrics])),
            "std_chamfer"       : float(np.std([m["chamfer_distance"]  for m in all_metrics])),
            "mean_final_loss"   : float(np.mean([m["final_loss"]       for m in all_metrics])),
            "per_shape"         : [{k: v for k, v in m.items() if k != "losses"}
                                   for m in all_metrics],
        }
        summary_path = args.out_dir / "reconstruction_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nMean Chamfer Distance : {summary['mean_chamfer']:.6f} "
              f"± {summary['std_chamfer']:.6f}")
        print(f"Summary saved to      : {summary_path}")

    elif args.mode == "ablation":
        # Model capacity ablation: compare small / medium / large
        shape_ids = all_ids[:args.n_shapes] if args.n_shapes else all_ids[:20]
        ablation_results = {}

        for size in ["small", "medium", "large"]:
            print(f"\n{'='*50}")
            print(f"  Ablation: model size = {size}")
            print(f"{'='*50}")
            size_metrics = []

            for shape_id in tqdm(shape_ids, desc=f"{size}"):
                try:
                    m = train_single_shape(
                        shape_id      = shape_id,
                        processed_dir = processed_dir,
                        out_dir       = args.out_dir / "ablation" / size,
                        model_size    = size,
                        n_epochs      = args.n_epochs,
                        lr            = args.lr,
                        batch_size    = args.batch_size,
                        device        = device,
                    )
                    size_metrics.append(m)
                except Exception as e:
                    print(f"  ERROR {shape_id}: {e}")

            ablation_results[size] = {
                "mean_chamfer" : float(np.mean([m["chamfer_distance"] for m in size_metrics])),
                "std_chamfer"  : float(np.std([m["chamfer_distance"]  for m in size_metrics])),
                "n_params"     : sum(p.numel() for p in build_model(size).parameters()),
            }

        # Print ablation summary table
        print(f"\n{'='*55}")
        print(f"  Model Capacity Ablation Results")
        print(f"{'='*55}")
        print(f"  {'Size':<10} {'Params':>10}  {'Mean CD':>12}  {'Std CD':>10}")
        print(f"  {'-'*46}")
        for size, res in ablation_results.items():
            print(f"  {size:<10} {res['n_params']:>10,}  "
                  f"{res['mean_chamfer']:>12.6f}  {res['std_chamfer']:>10.6f}")

        ablation_path = args.out_dir / "ablation_summary.json"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with open(ablation_path, "w") as f:
            json.dump(ablation_results, f, indent=2)
        print(f"\n  Saved to: {ablation_path}")