"""
level_set_check.py
==================
Verify the level-set offset correction ON YOUR OWN MODEL.

Takes the trained VAE, evaluates its SDF field on a 2D slice through a
generated shape, and draws the actual contours at:
  - SDF = 0.0   (raw zero level-set — the under-shoot)
  - SDF = +0.02 (the offset correction)
It also shows the full 3D volume ratio at a sweep of offsets, so you can
SEE that +0.02 is the value that best matches the real median volume.

This produces the real-data version of the conceptual diagram — a proper
methods/results figure for the dissertation.

Usage:
  python scripts/level_set_check.py \
      --processed_dir aneurysm_project/data/processed_sdf \
      --vae_ckpt      aneurysm_project/models/vae_compare/best.pt \
      --out_dir       aneurysm_project/results_levelset
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_models import AneurysmAE
from dataset import SurfaceDataset


def eval_sdf_grid(model, z, res, device, batch=65536):
    lin = np.linspace(-1, 1, res)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], -1).astype(np.float32)
    pts_t = torch.from_numpy(pts).to(device)
    vals = []
    with torch.no_grad():
        for i in range(0, len(pts_t), batch):
            chunk = pts_t[i:i+batch].unsqueeze(0)
            zb = z if z.dim() == 2 else z.unsqueeze(0)
            vals.append(model.decode(chunk, zb).squeeze().cpu().numpy())
    return np.concatenate(vals).reshape(res, res, res)


def mesh_volume(sdf_grid, level):
    try:
        verts, faces, _, _ = measure.marching_cubes(sdf_grid, level=level)
        res = sdf_grid.shape[0]
        verts = verts / (res - 1) * 2.0 - 1.0
        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        return abs(float(m.volume))
    except (ValueError, RuntimeError):
        return None


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res = args.resolution

    ck = torch.load(args.vae_ckpt, map_location=device)
    model = AneurysmAE(latent_dim=ck["latent_dim"], hidden_dim=ck["hidden_dim"],
                       n_freqs=ck["n_freqs"], variational=ck["variational"],
                       beta=ck.get("beta", 1.0), mode_name=ck["mode"]).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    print(f"Loaded {ck['mode']} (val recon {ck['val_recon']:.4f})")

    # ── Real median volume (reconstruct real test shapes) ─────────────────
    surf_ds = SurfaceDataset(args.processed_dir, split="test", n_points=2048)
    real_vols = []
    for sid in surf_ds.ids[:args.n_shapes]:
        pts = np.load(Path(args.processed_dir)/"surface"/f"{sid}.npz")["points"]
        if len(pts) > 2048:
            pts = pts[np.random.choice(len(pts), 2048, replace=False)]
        pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
        mu, _ = model.encode(pts_t)
        v = mesh_volume(eval_sdf_grid(model, mu, res, device), 0.0)
        if v: real_vols.append(v)
    real_median = float(np.median(real_vols))
    print(f"Real median volume: {real_median:.4f}")

    # ── Offset sweep on generated shapes ──────────────────────────────────
    offsets = np.linspace(-0.02, 0.06, 9)
    print("\nOffset sweep (generated volume ratio to real):")
    ratios = {o: [] for o in offsets}
    n_gen = args.n_shapes
    slice_grid = None
    for i in range(n_gen):
        z = torch.randn(1, model.latent_dim, device=device)
        grid = eval_sdf_grid(model, z, res, device)
        if i == 0:
            slice_grid = grid[res // 2]   # middle Z slice for the contour plot
        for o in offsets:
            v = mesh_volume(grid, level=o)
            if v:
                ratios[o].append(v / real_median)

    median_ratio = {o: (np.median(ratios[o]) if ratios[o] else np.nan) for o in offsets}
    for o in offsets:
        print(f"  offset {o:+.3f}: volume ×{median_ratio[o]:.2f} real  (n={len(ratios[o])})")

    # Best offset = closest ratio to 1.0
    valid = {o: r for o, r in median_ratio.items() if not np.isnan(r)}
    best_o = min(valid, key=lambda o: abs(valid[o] - 1.0))
    print(f"\nBest offset (ratio closest to 1.0): {best_o:+.3f}  (×{valid[best_o]:.2f})")

    # ── Plot 1: actual contours on a real slice ───────────────────────────
    lin = np.linspace(-1, 1, res)
    fig, ax = plt.subplots(figsize=(6, 6))
    # Filled SDF background
    im = ax.contourf(lin, lin, slice_grid.T, levels=30, cmap="RdBu", alpha=0.5)
    # The two contours that matter
    c0 = ax.contour(lin, lin, slice_grid.T, levels=[0.0],  colors="#993C1D", linewidths=2.5)
    c1 = ax.contour(lin, lin, slice_grid.T, levels=[0.02], colors="#0F6E56", linewidths=2.5)
    ax.clabel(c0, fmt={0.0: "SDF=0 (raw)"}, fontsize=9)
    ax.clabel(c1, fmt={0.02: "SDF=+0.02 (fix)"}, fontsize=9)
    ax.set_title("Actual SDF level-sets through a generated shape\n"
                 "(middle slice of your trained VAE)", fontsize=11)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, label="SDF value")
    plt.tight_layout()
    plt.savefig(out / "levelset_contours.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: offset vs volume ratio (the sweep) ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    os_ = list(valid.keys()); rs_ = [valid[o] for o in os_]
    ax.plot(os_, rs_, "o-", color="#185FA5", linewidth=2, markersize=7)
    ax.axhline(1.0, color="#5F5E5A", linestyle="--", linewidth=1, label="real volume")
    ax.axvline(best_o, color="#0F6E56", linestyle=":", linewidth=1.5,
               label=f"best offset {best_o:+.3f}")
    ax.axvline(0.02, color="#993C1D", linestyle=":", linewidth=1.5, label="offset +0.02")
    ax.set_xlabel("Level-set offset"); ax.set_ylabel("Generated volume ÷ real")
    ax.set_title("Volume ratio vs level-set offset (your VAE)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "offset_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: levelset_contours.png, offset_sweep.png")
    print("The contour plot is the real-data version of the conceptual diagram.")
    print("The sweep confirms which offset best matches real volume on YOUR model.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--vae_ckpt",      type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--resolution",    type=int, default=96)
    p.add_argument("--n_shapes",      type=int, default=20)
    args = p.parse_args()
    run(args)