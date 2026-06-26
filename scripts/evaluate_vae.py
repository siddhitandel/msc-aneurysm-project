"""
evaluate_vae.py
===============
Evaluation of the trained AneurysmVAE, mirroring the original evaluate.py
but adapted for the variational model, SDF representation, and marching-cubes
mesh extraction.

Produces:
  - 02_latent_space_vae.png   : PCA + t-SNE of the variational latent space
  - 03_interpolation_vae.png  : shape morphing via posterior-mean interpolation
  - 04_sensitivity_vae.png     : Chamfer distance vs input point density
  - generation visualisations are handled by mesh_extraction.py

Usage:
  python scripts/evaluate_vae.py \
      --processed_dir aneurysm_project/data/processed_sdf \
      --vae_ckpt      aneurysm_project/models/vae/best_vae.pt \
      --out_dir       aneurysm_project/results_vae
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from vae_model import AneurysmVAE
from dataset import SurfaceDataset


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def chamfer_np(pred, gt):
    try:
        import point_cloud_utils as pcu
        return float(pcu.chamfer_distance(pred.astype(np.float32),
                                          gt.astype(np.float32)))
    except Exception:
        d1 = np.sqrt(((pred[:, None] - gt[None]) ** 2).sum(-1)).min(1).mean()
        d2 = np.sqrt(((gt[:, None] - pred[None]) ** 2).sum(-1)).min(1).mean()
        return float((d1 + d2) / 2)


def make_grid(res, device):
    lin = np.linspace(-1, 1, res)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], -1).astype(np.float32)
    return torch.from_numpy(pts).to(device)


def sdf_grid_to_points(model, z, res, device, n_pts=2048, batch=65536):
    """Evaluate SDF on grid, extract near-surface points (|sdf| small)."""
    grid = make_grid(res, device)
    sdf_vals = []
    with torch.no_grad():
        for i in range(0, len(grid), batch):
            chunk = grid[i:i+batch].unsqueeze(0)
            z_b   = z if z.dim() == 2 else z.unsqueeze(0)
            sdf   = model.decode(chunk, z_b).squeeze().cpu().numpy()
            sdf_vals.append(sdf)
    sdf  = np.concatenate(sdf_vals)
    grid_np = grid.cpu().numpy()

    near = np.abs(sdf) < 0.03
    pts  = grid_np[near]
    if len(pts) == 0:
        pts = grid_np[np.abs(sdf) < 0.08]
    if len(pts) == 0:
        return np.zeros((n_pts, 3), dtype=np.float32)
    idx = np.random.choice(len(pts), n_pts, replace=(len(pts) < n_pts))
    return pts[idx]


def load_vae(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = AneurysmVAE(latent_dim=ckpt["latent_dim"],
                        hidden_dim=ckpt["hidden_dim"],
                        n_freqs=ckpt["n_freqs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded VAE from epoch {ckpt['epoch']} (val recon {ckpt['val_recon']:.4f})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1. Latent space (variational)
# ─────────────────────────────────────────────────────────────────────────────

def eval_latent(model, processed_dir, out_dir, device):
    print("\n[1/3] Latent space analysis (variational mu)")
    surf_ds = SurfaceDataset(processed_dir, split="test", n_points=2048)
    loader  = torch.utils.data.DataLoader(surf_ds, batch_size=16, shuffle=False)

    all_mu, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Encoding"):
            pts = batch["points"].to(device)
            mu, _ = model.encode(pts)
            all_mu.append(mu.cpu().numpy())
            labels.extend(batch["rupture_label"].numpy().tolist())

    Z = np.concatenate(all_mu)
    labels = np.array(labels)
    print(f"  Encoded {len(Z)} shapes, latent dim {Z.shape[1]}")

    pca   = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z)
    var   = pca.explained_variance_ratio_
    tsne  = TSNE(n_components=2, perplexity=min(30, len(Z)//4),
                 random_state=42)
    Z_tsne = tsne.fit_transform(Z)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, Z2, title in [(axes[0], Z_pca, f"PCA ({var.sum():.1%} var)"),
                          (axes[1], Z_tsne, "t-SNE")]:
        ax.scatter(Z2[labels==0,0], Z2[labels==0,1], c="#3498DB",
                   alpha=0.7, s=40, label="Unruptured")
        ax.scatter(Z2[labels==1,0], Z2[labels==1,1], c="#E74C3C",
                   alpha=0.7, s=40, label="Ruptured")
        ax.set_title(title, fontsize=13); ax.legend(); ax.grid(True, alpha=0.2)

    fig.suptitle("VAE Latent Space (posterior means)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "02_latent_space_vae.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 02_latent_space_vae.png  (PCA var: {var.sum():.1%})")

    np.save(out_dir / "vae_latent_mu.npy", Z)
    np.save(out_dir / "vae_latent_labels.npy", labels)
    return {"pca_var": float(var.sum()), "n": int(len(Z))}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Interpolation
# ─────────────────────────────────────────────────────────────────────────────

def eval_interp(model, processed_dir, out_dir, device, n_steps=7):
    print("\n[2/3] Shape interpolation")
    labels_df  = pd.read_csv(Path(processed_dir) / "labels.csv")
    surf_ds    = SurfaceDataset(processed_dir, split="test", n_points=2048)
    test_ids   = set(surf_ds.ids)
    rup  = [i for i in labels_df[labels_df.status=="ruptured"]["dataset"]   if i in test_ids]
    unr  = [i for i in labels_df[labels_df.status=="unruptured"]["dataset"] if i in test_ids]
    id_a, id_b = rup[0], unr[0]

    def load(sid):
        d = np.load(Path(processed_dir) / "surface" / f"{sid}.npz")
        p = d["points"]
        if len(p) > 2048: p = p[np.random.choice(len(p), 2048, replace=False)]
        return torch.from_numpy(p).float().unsqueeze(0).to(device)

    grid_pts = make_grid(64, device).unsqueeze(0)
    occs = model.interpolate(load(id_a), load(id_b), grid_pts, steps=n_steps)

    fig = plt.figure(figsize=(n_steps*2.6, 4))
    gs  = gridspec.GridSpec(1, n_steps, figure=fig, wspace=0.02)
    ts  = np.linspace(0, 1, n_steps)
    for i, (sdf_t, t) in enumerate(zip(occs, ts)):
        pts = sdf_grid_to_points_from_sdf(sdf_t, grid_pts, device, 800)
        ax  = fig.add_subplot(gs[i], projection="3d")
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=2,
                   c=[plt.cm.coolwarm(t)]*len(pts), alpha=0.75)
        ax.view_init(20, 45); ax.set_box_aspect([1,1,1])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for pane in [ax.xaxis, ax.yaxis, ax.zaxis]: pane.pane.fill = False
        lbl = "Ruptured" if i==0 else ("Unruptured" if i==n_steps-1 else "")
        ax.set_title(f"t={t:.2f}\n{lbl}", fontsize=9)

    fig.suptitle("VAE Latent Interpolation: Ruptured ➜ Unruptured", fontsize=12, y=1.02)
    plt.savefig(out_dir / "03_interpolation_vae.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Saved: 03_interpolation_vae.png  (shapes {id_a[:12]} ➜ {id_b[:12]})")
    return {"id_a": id_a, "id_b": id_b}


def sdf_grid_to_points_from_sdf(sdf_t, grid_pts, device, n_pts=800):
    """Extract near-surface points from an SDF tensor already on a grid."""
    sdf  = sdf_t.squeeze().cpu().numpy()
    grid = grid_pts.squeeze().cpu().numpy()
    near = np.abs(sdf) < 0.03
    pts  = grid[near]
    if len(pts) == 0:
        pts = grid[np.abs(sdf) < 0.08]
    if len(pts) == 0:
        return np.zeros((n_pts, 3), dtype=np.float32)
    idx = np.random.choice(len(pts), n_pts, replace=(len(pts) < n_pts))
    return pts[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Point cloud sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def eval_sensitivity(model, processed_dir, out_dir, device, n_shapes=20):
    print("\n[3/3] Point cloud sensitivity")
    densities = [128, 256, 512, 1024, 2048, 4096]
    surf_ds   = SurfaceDataset(processed_dir, split="test", n_points=4096)
    ids       = surf_ds.ids[:n_shapes]
    results   = {d: [] for d in densities}

    for sid in tqdm(ids, desc="  Sensitivity"):
        full = np.load(Path(processed_dir) / "surface" / f"{sid}.npz")["points"]
        for n in densities:
            pts = full[np.random.choice(len(full), min(n, len(full)), replace=False)]
            pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = model.encode(pts_t)
            pred = sdf_grid_to_points(model, mu, 48, device)
            results[n].append(chamfer_np(pred, full))

    means = [np.mean(results[d]) for d in densities]
    stds  = [np.std(results[d])  for d in densities]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.errorbar(densities, means, yerr=stds, fmt="o-", color="steelblue",
                capsize=5, linewidth=2, markersize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(densities); ax.set_xticklabels([str(d) for d in densities])
    ax.set_xlabel("Input Point Cloud Size"); ax.set_ylabel("Mean Chamfer Distance")
    ax.set_title("VAE: Reconstruction Quality vs Input Density (Q4)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "04_sensitivity_vae.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 04_sensitivity_vae.png")

    for d, m, s in zip(densities, means, stds):
        print(f"    {d:>5} pts: CD {m:.4f} ± {s:.4f}")
    return {"densities": densities, "mean_cd": means, "std_cd": stds}


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--vae_ckpt",      type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--n_sensitivity_shapes", type=int, default=20)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_vae(args.vae_ckpt, device)

    r1 = eval_latent(model, args.processed_dir, args.out_dir, device)
    r2 = eval_interp(model, args.processed_dir, args.out_dir, device)
    r3 = eval_sensitivity(model, args.processed_dir, args.out_dir, device,
                          n_shapes=args.n_sensitivity_shapes)

    with open(args.out_dir / "evaluation_vae_summary.json", "w") as f:
        json.dump({"latent": r1, "interpolation": r2, "sensitivity": r3}, f, indent=2)

    print("\nAll VAE evaluations complete.")
    print(f"Results: {args.out_dir}")