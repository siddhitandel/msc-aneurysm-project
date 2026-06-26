"""
mesh_extraction.py
==================
Marching cubes mesh extraction from trained SDF/occupancy networks,
plus morphological descriptor computation for real vs generated comparison.

Marching cubes converts the implicit field (SDF or occupancy on a 3D grid)
into a proper watertight triangle mesh. This enables:
  - Clean mesh visualisation (vs sparse point clouds)
  - Real geometric measurements: volume, surface area, etc.
  - Morphological descriptors comparable to AneuX's own metrics

Usage:
  python scripts/mesh_extraction.py \
      --processed_dir aneurysm_project/data/processed_sdf \
      --vae_ckpt      aneurysm_project/models/vae/best_vae.pt \
      --out_dir       aneurysm_project/results_vae \
      --n_generated   20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from skimage import measure   # marching cubes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vae_model import AneurysmVAE
from dataset import SurfaceDataset


# ─────────────────────────────────────────────────────────────────────────────
# SDF grid evaluation + marching cubes
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sdf_grid(model, z, resolution, device, batch=65536):
    """Evaluate the decoder on a dense 3D grid, return SDF volume."""
    lin = np.linspace(-1, 1, resolution)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], -1).astype(np.float32)
    pts_t = torch.from_numpy(pts).to(device)

    sdf_vals = []
    with torch.no_grad():
        for i in range(0, len(pts_t), batch):
            chunk = pts_t[i:i+batch].unsqueeze(0)             # (1, b, 3)
            z_b   = z if z.dim() == 2 else z.unsqueeze(0)
            sdf   = model.decode(chunk, z_b).squeeze().cpu().numpy()
            sdf_vals.append(sdf)

    sdf_grid = np.concatenate(sdf_vals).reshape(resolution, resolution, resolution)
    return sdf_grid


def sdf_to_mesh(sdf_grid, level=0.0):
    """
    Run marching cubes on an SDF grid. Surface is the level set sdf=0.
    Returns a trimesh.Trimesh or None if extraction fails.
    """
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            sdf_grid, level=level)
        # Rescale vertices from grid indices to [-1, 1]
        res = sdf_grid.shape[0]
        verts = verts / (res - 1) * 2.0 - 1.0
        return trimesh.Trimesh(vertices=verts, faces=faces,
                               vertex_normals=normals, process=False)
    except (ValueError, RuntimeError) as e:
        print(f"    marching cubes failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Morphological descriptors
# ─────────────────────────────────────────────────────────────────────────────

def compute_morphology(mesh: trimesh.Trimesh) -> dict:
    """
    Compute morphological descriptors used in aneurysm shape analysis.
    These mirror metrics in the AneuX database (Raghavan et al.).

    Returns dict with:
      volume          : enclosed volume
      surface_area    : total surface area
      sphericity      : how sphere-like (1 = perfect sphere)
      aspect_ratio    : bounding-box elongation
      nsi             : non-sphericity index (1 - sphericity)
    """
    if mesh is None or len(mesh.faces) == 0:
        return None

    try:
        volume = abs(float(mesh.volume))
        area   = float(mesh.area)

        # Sphericity: ratio of sphere surface area (same volume) to actual area
        # sphericity = pi^(1/3) * (6V)^(2/3) / A
        if area > 1e-8 and volume > 1e-8:
            sphericity = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / area
        else:
            sphericity = 0.0

        # Aspect ratio from oriented bounding box
        extents = mesh.bounding_box_oriented.extents
        aspect_ratio = float(extents.max() / (extents.min() + 1e-8))

        return {
            "volume"       : volume,
            "surface_area" : area,
            "sphericity"   : float(np.clip(sphericity, 0, 1)),
            "nsi"          : float(1 - np.clip(sphericity, 0, 1)),
            "aspect_ratio" : aspect_ratio,
        }
    except Exception as e:
        print(f"    morphology computation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load VAE
    ckpt  = torch.load(args.vae_ckpt, map_location=device)
    model = AneurysmVAE(latent_dim=ckpt["latent_dim"],
                        hidden_dim=ckpt["hidden_dim"],
                        n_freqs=ckpt["n_freqs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded VAE from epoch {ckpt['epoch']} (val recon {ckpt['val_recon']:.4f})")

    out_dir = Path(args.out_dir)
    (out_dir / "meshes_real").mkdir(parents=True, exist_ok=True)
    (out_dir / "meshes_generated").mkdir(parents=True, exist_ok=True)

    res = args.resolution

    # ── Extract meshes for REAL test shapes (reconstruction) ───────────────
    print(f"\n[1/3] Reconstructing real test shapes (marching cubes)...")
    surf_ds  = SurfaceDataset(args.processed_dir, split="test", n_points=2048)
    real_ids = surf_ds.ids[:args.n_generated]

    real_morph = []
    for sid in real_ids:
        data    = np.load(Path(args.processed_dir) / "surface" / f"{sid}.npz")
        pts     = data["points"]
        if len(pts) > 2048:
            pts = pts[np.random.choice(len(pts), 2048, replace=False)]
        pts_t   = torch.from_numpy(pts).float().unsqueeze(0).to(device)

        mu, _   = model.encode(pts_t)
        sdf_grid = evaluate_sdf_grid(model, mu, res, device)
        mesh    = sdf_to_mesh(sdf_grid)

        if mesh is not None:
            mesh.export(out_dir / "meshes_real" / f"{sid}.obj")
            m = compute_morphology(mesh)
            if m:
                m["id"] = sid
                real_morph.append(m)

    print(f"  Extracted {len(real_morph)} real meshes")

    # ── Generate NEW shapes ────────────────────────────────────────────────
    print(f"\n[2/3] Generating synthetic shapes...")
    gen_morph = []
    for i in range(args.n_generated):
        z = torch.randn(1, model.latent_dim, device=device) * args.truncation
        sdf_grid = evaluate_sdf_grid(model, z, res, device)
        mesh = sdf_to_mesh(sdf_grid)

        if mesh is not None:
            mesh.export(out_dir / "meshes_generated" / f"gen_{i:03d}.obj")
            m = compute_morphology(mesh)
            if m:
                m["id"] = f"gen_{i:03d}"
                gen_morph.append(m)

    print(f"  Generated {len(gen_morph)} synthetic meshes")

    # ── Compare distributions ──────────────────────────────────────────────
    print(f"\n[3/3] Comparing morphological distributions...")

    if not real_morph or not gen_morph:
        print("  Not enough meshes for comparison.")
        return

    metrics = ["volume", "surface_area", "sphericity", "aspect_ratio"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 4, 4))

    comparison = {}
    for ax, metric in zip(axes, metrics):
        real_vals = [m[metric] for m in real_morph]
        gen_vals  = [m[metric] for m in gen_morph]

        ax.hist(real_vals, bins=12, alpha=0.6, label="Real",      color="#3498DB")
        ax.hist(gen_vals,  bins=12, alpha=0.6, label="Generated", color="#E74C3C")
        ax.set_title(metric.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        comparison[metric] = {
            "real_mean": float(np.mean(real_vals)),
            "real_std":  float(np.std(real_vals)),
            "gen_mean":  float(np.mean(gen_vals)),
            "gen_std":   float(np.std(gen_vals)),
        }

    fig.suptitle("Morphological Descriptor Distributions: Real vs Generated",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "morphology_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print comparison table
    print(f"\n  {'Metric':<16} {'Real (mean±std)':>22} {'Generated (mean±std)':>24}")
    print(f"  {'-'*62}")
    for metric in metrics:
        c = comparison[metric]
        print(f"  {metric:<16} "
              f"{c['real_mean']:>10.4f}±{c['real_std']:<10.4f} "
              f"{c['gen_mean']:>11.4f}±{c['gen_std']:<10.4f}")

    # Save results
    with open(out_dir / "morphology_results.json", "w") as f:
        json.dump({"comparison": comparison,
                   "real_morphology": real_morph,
                   "generated_morphology": gen_morph}, f, indent=2)

    print(f"\n  Saved: morphology_comparison.png")
    print(f"  Saved: morphology_results.json")
    print(f"  Meshes exported to: {out_dir}/meshes_real and meshes_generated")
    print(f"\nDownload meshes and open in MeshLab/Blender to inspect quality.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--vae_ckpt",      type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--resolution",    type=int, default=96,
                   help="Marching cubes grid resolution (default: 96)")
    p.add_argument("--n_generated",   type=int, default=20)
    p.add_argument("--truncation",    type=float, default=1.0,
                   help="Latent sampling truncation (lower=more typical)")
    args = p.parse_args()
    run(args)