"""
visualise_interpolation.py
==========================
Improved interpolation visualisation using higher grid resolution
and better surface extraction for cleaner dissertation figures.

Run after evaluate.py has completed.

Usage:
    python scripts/visualise_interpolation.py \
        --processed_dir aneurysm_project/data/processed \
        --ae_ckpt       aneurysm_project/models/autoencoder/best_model.pt \
        --out_dir       aneurysm_project/results \
        --n_steps       7
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from autoencoder import AneurysmAutoencoder
from dataset import SurfaceDataset
from evaluate import (load_autoencoder, chamfer_distance_np,
                      occ_to_surface_pts, make_query_grid)


def visualise_interpolation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_autoencoder(args.ae_ckpt, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Pick interpolation pair ───────────────────────────────────────────
    labels_df  = pd.read_csv(Path(args.processed_dir) / "labels.csv")
    surf_ds    = SurfaceDataset(args.processed_dir, split="test", n_points=2048)
    test_ids   = set(surf_ds.ids)

    ruptured   = labels_df[labels_df["status"] == "ruptured"]["dataset"].tolist()
    unruptured = labels_df[labels_df["status"] == "unruptured"]["dataset"].tolist()

    id_a = next(i for i in ruptured   if i in test_ids)
    id_b = next(i for i in unruptured if i in test_ids)

    def load_pts(shape_id, n=2048):
        data = np.load(Path(args.processed_dir) / "surface" / f"{shape_id}.npz")
        pts  = data["points"]
        if len(pts) > n:
            pts = pts[np.random.choice(len(pts), n, replace=False)]
        return torch.from_numpy(pts).unsqueeze(0).to(device)

    surf_a = load_pts(id_a)
    surf_b = load_pts(id_b)

    # ── Higher resolution grid for cleaner surfaces ───────────────────────
    resolution = 64    # 64^3 = 262K points — much sharper than 40^3
    grid       = make_query_grid(resolution=resolution, device=device)
    grid_pts   = grid.unsqueeze(0)
    grid_np    = grid.cpu().numpy()

    # ── Interpolate ───────────────────────────────────────────────────────
    n_steps       = args.n_steps
    t_vals        = np.linspace(0, 1, n_steps)
    interp_occs   = model.interpolate(surf_a, surf_b, grid_pts, steps=n_steps)

    # ── Extract surface point clouds at each step ─────────────────────────
    surfaces = []
    for occ_t in interp_occs:
        occ_np = occ_t.squeeze().cpu().numpy()
        pts    = occ_to_surface_pts(occ_np, grid_np, threshold=0.5, n_pts=1024)
        surfaces.append(pts)

    # ── Figure 1: 3D point cloud grid (better view angles) ────────────────
    fig = plt.figure(figsize=(n_steps * 2.8, 4.5))
    gs  = gridspec.GridSpec(1, n_steps, figure=fig,
                            wspace=0.02, hspace=0.0)

    cmap = plt.cm.coolwarm
    for i, (pts, t) in enumerate(zip(surfaces, t_vals)):
        ax = fig.add_subplot(gs[i], projection="3d")

        # Colour points by height (z) for better 3D depth perception
        z_norm = (pts[:,2] - pts[:,2].min()) / (pts[:,2].ptp() + 1e-8)
        colours = cmap(t) * np.ones((len(pts), 4))
        colours[:,3] = 0.7   # alpha

        ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                   s=2, c=[cmap(t)] * len(pts), alpha=0.75, depthshade=True)

        # Better viewing angle
        ax.view_init(elev=20, azim=45)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect([1,1,1])

        label = ""
        if i == 0:            label = "Ruptured"
        elif i == n_steps-1:  label = "Unruptured"
        ax.set_title(f"t={t:.2f}\n{label}", fontsize=10,
                     color="darkred" if t < 0.5 else "steelblue")

        # Remove pane backgrounds for cleaner look
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("lightgray")
        ax.yaxis.pane.set_edgecolor("lightgray")
        ax.zaxis.pane.set_edgecolor("lightgray")

    fig.suptitle("Latent Space Shape Interpolation\n"
                 "Ruptured (t=0)  ➜  Unruptured (t=1)",
                 fontsize=13, y=1.02)

    plt.savefig(out_dir / "03_interpolation_improved.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()

    # ── Figure 2: 2-row comparison (top = real endpoints, bottom = interp) ─
    fig2, axes = plt.subplots(2, n_steps, figsize=(n_steps * 2.5, 5),
                              subplot_kw={"projection": "3d"})

    # Top row: real surface point clouds along interpolation
    gt_a = np.load(Path(args.processed_dir) / "surface" / f"{id_a}.npz")["points"]
    gt_b = np.load(Path(args.processed_dir) / "surface" / f"{id_b}.npz")["points"]

    for i in range(n_steps):
        t   = t_vals[i]
        col = cmap(t)

        # Top: show real A for first half, real B for second half
        ax_top = axes[0][i]
        real_pts = gt_a if t <= 0.5 else gt_b
        idx = np.random.choice(len(real_pts), min(512, len(real_pts)), replace=False)
        ax_top.scatter(real_pts[idx,0], real_pts[idx,1], real_pts[idx,2],
                       s=1.5, c=[col]*len(idx), alpha=0.7, depthshade=True)
        ax_top.view_init(elev=20, azim=45)
        ax_top.set_xticks([]); ax_top.set_yticks([]); ax_top.set_zticks([])
        ax_top.set_box_aspect([1,1,1])
        ax_top.xaxis.pane.fill = False
        ax_top.yaxis.pane.fill = False
        ax_top.zaxis.pane.fill = False
        if i == 0:
            ax_top.set_ylabel("Reference", fontsize=9, labelpad=8)

        # Bottom: decoded interpolated shape
        ax_bot = axes[1][i]
        pts = surfaces[i]
        ax_bot.scatter(pts[:,0], pts[:,1], pts[:,2],
                       s=1.5, c=[col]*len(pts), alpha=0.7, depthshade=True)
        ax_bot.view_init(elev=20, azim=45)
        ax_bot.set_xticks([]); ax_bot.set_yticks([]); ax_bot.set_zticks([])
        ax_bot.set_box_aspect([1,1,1])
        ax_bot.xaxis.pane.fill = False
        ax_bot.yaxis.pane.fill = False
        ax_bot.zaxis.pane.fill = False
        ax_bot.set_title(f"t={t:.2f}", fontsize=9)
        if i == 0:
            ax_bot.set_ylabel("Interpolated", fontsize=9, labelpad=8)

    fig2.suptitle("Shape Morphing via Latent Space Interpolation  (Q2)\n"
                  "Top: reference endpoints   Bottom: decoded intermediate shapes",
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "03_interpolation_comparison.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: 03_interpolation_improved.png")
    print(f"Saved: 03_interpolation_comparison.png")

    # ── Compute and print Chamfer distances along path ─────────────────────
    print(f"\nChamfer distance along interpolation path:")
    print(f"  {'t':>6}  {'CD to A (ruptured)':>20}  {'CD to B (unruptured)':>22}")
    print(f"  {'-'*52}")
    for t, pts in zip(t_vals, surfaces):
        cd_a = chamfer_distance_np(pts, gt_a)
        cd_b = chamfer_distance_np(pts, gt_b)
        print(f"  {t:>6.2f}  {cd_a:>20.4f}  {cd_b:>22.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--ae_ckpt",       type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--n_steps",       type=int,  default=7)
    args = p.parse_args()
    visualise_interpolation(args)