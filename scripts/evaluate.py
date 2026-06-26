"""
evaluate.py
===========
Comprehensive evaluation of the trained reconstruction model and
generative autoencoder. Produces all results needed for the dissertation.

Covers all four supervisor research questions:
  Q1. Topological flexibility  -> Chamfer distance per anatomical location
  Q2. Shape morphing           -> Latent space interpolation between shapes
  Q3. Model capacity           -> Already done in ablation (reconstruction job)
  Q4. Point cloud sensitivity  -> Chamfer distance vs. input point density

Also produces:
  - Latent space PCA / t-SNE coloured by rupture status
  - Generated (synthetic) shapes from sampled latent codes
  - Real vs. generated morphological descriptor distributions

Usage:
  python scripts/evaluate.py \
      --processed_dir aneurysm_project/data/processed \
      --recon_dir     aneurysm_project/models/reconstruction \
      --ae_ckpt       aneurysm_project/models/autoencoder/best_model.pt \
      --out_dir       aneurysm_project/results

All plots saved to out_dir as PNG files.
All numerical results saved as JSON.
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

from autoencoder import AneurysmAutoencoder
from dataset import SurfaceDataset, OccupancyDataset


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def chamfer_distance_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """Chamfer distance between two (N,3) point clouds. Pure numpy fallback."""
    try:
        import point_cloud_utils as pcu
        return float(pcu.chamfer_distance(pred.astype(np.float32),
                                          gt.astype(np.float32)))
    except Exception:
        d1 = np.sqrt(((pred[:, None] - gt[None]) ** 2).sum(-1)).min(1).mean()
        d2 = np.sqrt(((gt[:, None] - pred[None]) ** 2).sum(-1)).min(1).mean()
        return float((d1 + d2) / 2)


def load_autoencoder(ckpt_path: Path, device: torch.device) -> AneurysmAutoencoder:
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = AneurysmAutoencoder(
        latent_dim = ckpt.get("latent_dim", 256),
        hidden_dim = ckpt.get("hidden_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded autoencoder from epoch {ckpt.get('epoch','?')} "
          f"(val loss: {ckpt.get('val_loss', '?'):.4f})")
    return model


def make_query_grid(resolution: int = 48, device: torch.device = None) -> torch.Tensor:
    """Create a uniform 3D query grid inside the unit sphere."""
    lin = np.linspace(-1.0, 1.0, resolution)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)
    return torch.from_numpy(pts).to(device)


def occ_to_surface_pts(occ: np.ndarray,
                        grid_pts: np.ndarray,
                        threshold: float = 0.5,
                        n_pts: int = 2048) -> np.ndarray:
    """Extract near-surface points from occupancy predictions on a grid."""
    near = np.abs(occ - threshold) < 0.12
    pts  = grid_pts[near]
    if len(pts) == 0:
        pts = grid_pts[occ > threshold]
    if len(pts) == 0:
        return np.zeros((n_pts, 3), dtype=np.float32)
    idx = np.random.choice(len(pts), n_pts,
                           replace=(len(pts) < n_pts))
    return pts[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Reconstruction quality analysis (Q1 — topological flexibility)
# ─────────────────────────────────────────────────────────────────────────────

def eval_reconstruction(recon_dir: Path,
                        processed_dir: Path,
                        out_dir: Path):
    """
    Load reconstruction_summary.json and analyse Chamfer distances
    broken down by anatomical location and rupture status.
    Answers Q1: topological flexibility across morphology types.
    """
    print("\n" + "="*60)
    print("  [1/4] Reconstruction Quality Analysis")
    print("="*60)

    summary_path = recon_dir / "reconstruction_summary.json"
    if not summary_path.exists():
        print(f"  WARNING: {summary_path} not found. Skipping.")
        return {}

    with open(summary_path) as f:
        summary = json.load(f)

    labels_df = pd.read_csv(processed_dir / "labels.csv")
    label_map = dict(zip(labels_df["dataset"].astype(str),
                         labels_df[["status", "location"]].values.tolist()))

    per_shape = summary["per_shape"]

    # Attach metadata
    records = []
    for s in per_shape:
        meta = label_map.get(s["shape_id"], ["unknown", "unknown"])
        records.append({
            "shape_id" : s["shape_id"],
            "cd"       : s["chamfer_distance"],
            "loss"     : s["final_loss"],
            "status"   : meta[0],
            "location" : meta[1],
        })

    df = pd.DataFrame(records)

    # ── Overall stats ─────────────────────────────────────────────────────
    print(f"\n  Overall Chamfer Distance:")
    print(f"    Mean : {df['cd'].mean():.6f}")
    print(f"    Std  : {df['cd'].std():.6f}")
    print(f"    Min  : {df['cd'].min():.6f}")
    print(f"    Max  : {df['cd'].max():.6f}")

    # ── By rupture status ─────────────────────────────────────────────────
    print(f"\n  Chamfer Distance by Rupture Status:")
    for status, grp in df.groupby("status"):
        print(f"    {status:12s}: {grp['cd'].mean():.6f} ± {grp['cd'].std():.6f}  (n={len(grp)})")

    # ── By anatomical location (top 6) ────────────────────────────────────
    print(f"\n  Chamfer Distance by Anatomical Location (top 6 by count):")
    top_locs = df["location"].value_counts().head(6).index
    loc_stats = []
    for loc in top_locs:
        grp = df[df["location"] == loc]
        loc_stats.append({"location": loc, "mean_cd": grp["cd"].mean(),
                           "std_cd": grp["cd"].std(), "n": len(grp)})
        print(f"    {loc:15s}: {grp['cd'].mean():.6f} ± {grp['cd'].std():.6f}  (n={len(grp)})")

    # ── Plot: CD distribution by status ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram by rupture status
    for status, grp in df.groupby("status"):
        axes[0].hist(grp["cd"], bins=30, alpha=0.6, label=status)
    axes[0].set_xlabel("Chamfer Distance", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Reconstruction Quality by Rupture Status", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bar chart by location
    loc_df = pd.DataFrame(loc_stats).sort_values("mean_cd")
    axes[1].barh(loc_df["location"], loc_df["mean_cd"],
                 xerr=loc_df["std_cd"], capsize=4,
                 color="steelblue", alpha=0.8)
    axes[1].set_xlabel("Mean Chamfer Distance", fontsize=12)
    axes[1].set_title("Reconstruction Quality by Anatomical Location\n(Q1: Topological Flexibility)", fontsize=13)
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out_dir / "01_reconstruction_quality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: 01_reconstruction_quality.png")

    # Save numerical results
    results = {
        "overall_mean_cd" : float(df["cd"].mean()),
        "overall_std_cd"  : float(df["cd"].std()),
        "by_status"       : df.groupby("status")["cd"].agg(["mean","std","count"]).to_dict(),
        "by_location"     : loc_stats,
    }
    with open(out_dir / "01_reconstruction_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Latent space analysis
# ─────────────────────────────────────────────────────────────────────────────

def eval_latent_space(model: AneurysmAutoencoder,
                      processed_dir: Path,
                      out_dir: Path,
                      device: torch.device):
    """
    Encode all test shapes and visualise the latent space with PCA and t-SNE.
    Coloured by rupture status to see if the latent space separates them.
    """
    print("\n" + "="*60)
    print("  [2/4] Latent Space Analysis")
    print("="*60)

    surf_ds = SurfaceDataset(processed_dir, split="test", n_points=2048)
    loader  = torch.utils.data.DataLoader(surf_ds, batch_size=16,
                                           shuffle=False, num_workers=2)

    all_z, all_labels, all_ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Encoding shapes"):
            pts = batch["points"].to(device)          # (B, N, 3)
            z   = model.encode(pts).cpu().numpy()     # (B, latent_dim)
            all_z.append(z)
            all_labels.extend(batch["rupture_label"].numpy().tolist())
            all_ids.extend(batch["id"])

    Z      = np.concatenate(all_z, axis=0)            # (N_test, latent_dim)
    labels = np.array(all_labels)

    print(f"  Encoded {len(Z)} test shapes")
    print(f"  Latent space shape: {Z.shape}")
    print(f"  Ruptured: {(labels==1).sum()}  Unruptured: {(labels==0).sum()}")

    colours = ["#E74C3C" if l == 1 else "#3498DB" for l in labels]

    # ── PCA ──────────────────────────────────────────────────────────────
    pca      = PCA(n_components=2)
    Z_pca    = pca.fit_transform(Z)
    var_exp  = pca.explained_variance_ratio_

    # ── t-SNE ─────────────────────────────────────────────────────────────
    tsne   = TSNE(n_components=2, perplexity=min(30, len(Z)//4),
                  random_state=42, n_iter=1000)
    Z_tsne = tsne.fit_transform(Z)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, Z_2d, title in [
        (axes[0], Z_pca,  f"PCA  (PC1: {var_exp[0]:.1%}, PC2: {var_exp[1]:.1%})"),
        (axes[1], Z_tsne, "t-SNE"),
    ]:
        ax.scatter(Z_2d[labels==0, 0], Z_2d[labels==0, 1],
                   c="#3498DB", alpha=0.7, s=40, label="Unruptured", edgecolors="none")
        ax.scatter(Z_2d[labels==1, 0], Z_2d[labels==1, 1],
                   c="#E74C3C", alpha=0.7, s=40, label="Ruptured",   edgecolors="none")
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")

    fig.suptitle("Latent Space Visualisation — Aneurysm Autoencoder", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "02_latent_space.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 02_latent_space.png")

    # Save latent codes for further analysis
    np.save(out_dir / "latent_codes.npy",  Z)
    np.save(out_dir / "latent_labels.npy", labels)

    results = {
        "n_test_shapes"      : int(len(Z)),
        "latent_dim"         : int(Z.shape[1]),
        "pca_variance_pc1"   : float(var_exp[0]),
        "pca_variance_pc2"   : float(var_exp[1]),
        "pca_variance_total" : float(var_exp.sum()),
        "latent_mean_norm"   : float(np.linalg.norm(Z, axis=1).mean()),
        "latent_std"         : float(Z.std()),
    }
    with open(out_dir / "02_latent_space_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  PCA variance explained (PC1+PC2): {var_exp.sum():.1%}")
    return Z, labels, all_ids, results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Shape interpolation (Q2 — shape morphing)
# ─────────────────────────────────────────────────────────────────────────────

def eval_interpolation(model: AneurysmAutoencoder,
                       processed_dir: Path,
                       out_dir: Path,
                       device: torch.device,
                       n_steps: int = 7):
    """
    Interpolate between one ruptured and one unruptured aneurysm in latent space.
    Answers Q2: is continuous shape morphing possible?
    """
    print("\n" + "="*60)
    print("  [3/4] Shape Interpolation  (Q2: Shape Morphing)")
    print("="*60)

    labels_df = pd.read_csv(processed_dir / "labels.csv")
    ruptured   = labels_df[labels_df["status"] == "ruptured"]["dataset"].tolist()
    unruptured = labels_df[labels_df["status"] == "unruptured"]["dataset"].tolist()

    # Pick one of each from test set
    surf_ds  = SurfaceDataset(processed_dir, split="test", n_points=2048)
    test_ids = set(surf_ds.ids)

    id_a = next((i for i in ruptured   if i in test_ids), None)
    id_b = next((i for i in unruptured if i in test_ids), None)

    if id_a is None or id_b is None:
        print("  Could not find suitable pair for interpolation. Skipping.")
        return {}

    print(f"  Shape A (ruptured)  : {id_a[:30]}")
    print(f"  Shape B (unruptured): {id_b[:30]}")

    def load_surface(shape_id):
        data = np.load(processed_dir / "surface" / f"{shape_id}.npz")
        pts  = data["points"]
        if len(pts) > 2048:
            pts = pts[np.random.choice(len(pts), 2048, replace=False)]
        return torch.from_numpy(pts).unsqueeze(0).to(device)   # (1, N, 3)

    surf_a = load_surface(id_a)
    surf_b = load_surface(id_b)

    # Build query grid
    grid   = make_query_grid(resolution=40, device=device)  # (M, 3)
    grid_pts = grid.unsqueeze(0)                             # (1, M, 3)
    grid_np  = grid.cpu().numpy()

    # Interpolate
    interp_results = model.interpolate(surf_a, surf_b, grid_pts, steps=n_steps)

    # ── Compute Chamfer distance along interpolation path ─────────────────
    gt_a = np.load(processed_dir / "surface" / f"{id_a}.npz")["points"]
    gt_b = np.load(processed_dir / "surface" / f"{id_b}.npz")["points"]

    cd_to_a, cd_to_b = [], []
    for occ_t in interp_results:
        pred_pts = occ_to_surface_pts(occ_t.squeeze().cpu().numpy(), grid_np)
        cd_to_a.append(chamfer_distance_np(pred_pts, gt_a))
        cd_to_b.append(chamfer_distance_np(pred_pts, gt_b))

    # ── Plot: interpolation path visualised as 3D scatter ─────────────────
    fig = plt.figure(figsize=(n_steps * 2.5, 4))
    gs  = gridspec.GridSpec(1, n_steps, figure=fig)

    t_vals = np.linspace(0, 1, n_steps)
    for i, (occ_t, t) in enumerate(zip(interp_results, t_vals)):
        ax   = fig.add_subplot(gs[i], projection="3d")
        pts  = occ_to_surface_pts(occ_t.squeeze().cpu().numpy(), grid_np, n_pts=512)
        col  = plt.cm.RdBu_r(t)
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1, c=[col]*len(pts), alpha=0.6)
        ax.set_title(f"t={t:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect([1,1,1])

    fig.suptitle(f"Latent Space Interpolation\n"
                 f"Ruptured (t=0)  →  Unruptured (t=1)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "03_interpolation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot: Chamfer distances along path ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_vals, cd_to_a, "o-", color="#E74C3C", label="CD to ruptured shape A")
    ax.plot(t_vals, cd_to_b, "s-", color="#3498DB", label="CD to unruptured shape B")
    ax.set_xlabel("Interpolation parameter t", fontsize=12)
    ax.set_ylabel("Chamfer Distance", fontsize=12)
    ax.set_title("Shape Morphing: Chamfer Distance Along Interpolation Path\n(Q2: Shape Morphing)", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "03_interpolation_chamfer.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: 03_interpolation.png")
    print(f"  Saved: 03_interpolation_chamfer.png")
    print(f"  CD at t=0.0 (should be close to A): {cd_to_a[0]:.4f}")
    print(f"  CD at t=1.0 (should be close to B): {cd_to_b[-1]:.4f}")

    results = {
        "shape_a_id"    : id_a,
        "shape_b_id"    : id_b,
        "n_steps"       : n_steps,
        "cd_to_a"       : cd_to_a,
        "cd_to_b"       : cd_to_b,
        "t_values"      : t_vals.tolist(),
    }
    with open(out_dir / "03_interpolation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Point cloud sensitivity (Q4)
# ─────────────────────────────────────────────────────────────────────────────

def eval_point_cloud_sensitivity(model: AneurysmAutoencoder,
                                  processed_dir: Path,
                                  out_dir: Path,
                                  device: torch.device,
                                  n_shapes: int = 20):
    """
    Evaluate how reconstruction quality degrades with fewer input points.
    Answers Q4: how sensitive is the model to point cloud resolution?
    """
    print("\n" + "="*60)
    print("  [4/4] Point Cloud Sensitivity  (Q4)")
    print("="*60)

    densities  = [128, 256, 512, 1024, 2048, 4096]
    surf_ds    = SurfaceDataset(processed_dir, split="test", n_points=4096)
    test_ids   = surf_ds.ids[:n_shapes]

    grid       = make_query_grid(resolution=40, device=device)
    grid_pts   = grid.unsqueeze(0)
    grid_np    = grid.cpu().numpy()

    results_by_density = {d: [] for d in densities}

    for shape_id in tqdm(test_ids, desc="  Sensitivity eval"):
        data      = np.load(processed_dir / "surface" / f"{shape_id}.npz")
        full_pts  = data["points"]                         # (4096, 3)
        gt_pts    = full_pts                               # ground truth

        for n_pts in densities:
            # Subsample to n_pts
            if n_pts < len(full_pts):
                idx  = np.random.choice(len(full_pts), n_pts, replace=False)
                pts  = full_pts[idx]
            else:
                pts  = full_pts

            pts_t = torch.from_numpy(pts).unsqueeze(0).to(device)  # (1, n, 3)

            with torch.no_grad():
                z    = model.encode(pts_t)
                occ  = model.decode(grid_pts, z).squeeze().cpu().numpy()

            pred_pts = occ_to_surface_pts(occ, grid_np)
            cd       = chamfer_distance_np(pred_pts, gt_pts)
            results_by_density[n_pts].append(cd)

    # Summary
    means = [np.mean(results_by_density[d]) for d in densities]
    stds  = [np.std(results_by_density[d])  for d in densities]

    print(f"\n  {'Points':>8}  {'Mean CD':>10}  {'Std CD':>10}")
    print(f"  {'-'*32}")
    for d, m, s in zip(densities, means, stds):
        print(f"  {d:>8}  {m:>10.6f}  {s:>10.6f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(densities, means, yerr=stds, fmt="o-",
                color="steelblue", capsize=5, linewidth=2, markersize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(densities)
    ax.set_xticklabels([str(d) for d in densities])
    ax.set_xlabel("Input Point Cloud Size", fontsize=12)
    ax.set_ylabel("Mean Chamfer Distance", fontsize=12)
    ax.set_title("Reconstruction Quality vs. Input Point Cloud Density\n(Q4: Point Cloud Sensitivity)", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "04_point_cloud_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: 04_point_cloud_sensitivity.png")

    sensitivity_results = {
        "densities"       : densities,
        "mean_cd"         : means,
        "std_cd"          : stds,
        "n_shapes_tested" : n_shapes,
    }
    with open(out_dir / "04_sensitivity_results.json", "w") as f:
        json.dump(sensitivity_results, f, indent=2)
    return sensitivity_results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Generation — synthetic shape samples
# ─────────────────────────────────────────────────────────────────────────────

def eval_generation(model: AneurysmAutoencoder,
                    processed_dir: Path,
                    out_dir: Path,
                    device: torch.device,
                    n_generated: int = 6):
    """
    Generate new synthetic aneurysm shapes by sampling z ~ N(0, I).
    Compare generated vs. real shape statistics.
    """
    print("\n" + "="*60)
    print("  [5/5] Shape Generation")
    print("="*60)

    grid     = make_query_grid(resolution=40, device=device)
    grid_pts = grid.unsqueeze(0)
    grid_np  = grid.cpu().numpy()

    # Generate shapes
    occ_list = model.generate(grid_pts, n_samples=n_generated, device=device)
    gen_pts_list = []
    for i in range(n_generated):
        pts = occ_to_surface_pts(occ_list[i].squeeze().cpu().numpy(), grid_np)
        gen_pts_list.append(pts)

    # Load some real test shapes for comparison
    surf_ds  = SurfaceDataset(processed_dir, split="test", n_points=2048)
    real_ids = surf_ds.ids[:n_generated]
    real_pts_list = [
        np.load(processed_dir / "surface" / f"{sid}.npz")["points"]
        for sid in real_ids
    ]

    # ── Plot: real vs generated ────────────────────────────────────────────
    fig, axes = plt.subplots(2, n_generated, figsize=(n_generated * 2.5, 5),
                             subplot_kw={"projection": "3d"})

    row_labels = ["Real", "Generated"]
    for row, pts_list in enumerate([real_pts_list, gen_pts_list]):
        for col, pts in enumerate(pts_list):
            ax = axes[row][col]
            ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                       s=1, c=pts[:,2], cmap="viridis", alpha=0.6)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect([1,1,1])
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11)

    fig.suptitle("Real vs. Generated Aneurysm Shapes", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "05_real_vs_generated.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Compare point cloud statistics ────────────────────────────────────
    def shape_stats(pts_list):
        volumes  = [np.prod(p.max(0) - p.min(0)) for p in pts_list]
        spreads  = [np.linalg.norm(p.std(0))      for p in pts_list]
        return {"mean_volume": float(np.mean(volumes)),
                "std_volume":  float(np.std(volumes)),
                "mean_spread": float(np.mean(spreads)),
                "std_spread":  float(np.std(spreads))}

    real_stats = shape_stats(real_pts_list)
    gen_stats  = shape_stats(gen_pts_list)

    print(f"\n  Shape statistics comparison:")
    print(f"  {'Metric':<20}  {'Real':>12}  {'Generated':>12}")
    print(f"  {'-'*46}")
    for k in real_stats:
        print(f"  {k:<20}  {real_stats[k]:>12.4f}  {gen_stats[k]:>12.4f}")

    print(f"\n  Saved: 05_real_vs_generated.png")

    results = {"real_stats": real_stats, "generated_stats": gen_stats,
               "n_generated": n_generated}
    with open(out_dir / "05_generation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate reconstruction model and autoencoder")
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--recon_dir",     type=Path, required=True)
    p.add_argument("--ae_ckpt",       type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--n_sensitivity_shapes", type=int, default=20,
                   help="Shapes to use for sensitivity analysis (default: 20)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load autoencoder
    model = load_autoencoder(args.ae_ckpt, device)

    # Run all evaluations
    r1 = eval_reconstruction(args.recon_dir, args.processed_dir, args.out_dir)
    Z, labels, ids, r2 = eval_latent_space(model, args.processed_dir, args.out_dir, device)
    r3 = eval_interpolation(model, args.processed_dir, args.out_dir, device)
    r4 = eval_point_cloud_sensitivity(model, args.processed_dir, args.out_dir, device,
                                       n_shapes=args.n_sensitivity_shapes)
    r5 = eval_generation(model, args.processed_dir, args.out_dir, device)

    # Master summary
    master = {"reconstruction": r1, "latent_space": r2,
              "interpolation": r3,  "sensitivity": r4, "generation": r5}
    with open(args.out_dir / "evaluation_summary.json", "w") as f:
        json.dump(master, f, indent=2)

    print("\n" + "="*60)
    print("  All evaluations complete.")
    print(f"  Results saved to: {args.out_dir}")
    print("="*60)