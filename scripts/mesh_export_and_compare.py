"""
mesh_export_and_compare.py
==========================
Addresses assessor feedback:
  1. FIX mesh export so files open cleanly in MeshLab/Blender
     (clean topology: remove degenerate/duplicate faces, fix normals,
      remove unreferenced vertices, keep the largest connected component).
  2. Real-vs-generated mesh COMPARISON with per-vertex difference colouring
     (colour a generated mesh by distance to the nearest real surface).
  3. Publication-quality SURFACE renders (not sparse point clouds), with
     consistent camera + lighting, for the dissertation.

The VAE is the main model throughout (per assessor: build the story around
one model). Plain AE / beta-VAE only appear in the ablation script.

Requires: trimesh, scikit-image, matplotlib. All headless (Agg / offscreen).

Usage:
  python scripts/mesh_export_and_compare.py \
      --processed_dir aneurysm_project/data/processed_sdf \
      --vae_ckpt      aneurysm_project/models/vae_compare/best.pt \
      --out_dir       aneurysm_project/results_meshes \
      --n_show 6 --offset 0.01
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
from skimage import measure
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ae_models import AneurysmAE
from dataset import SurfaceDataset


# ─────────────────────────────────────────────────────────────────────────────
# SDF grid -> CLEAN mesh
# ─────────────────────────────────────────────────────────────────────────────

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


def grid_to_clean_mesh(sdf_grid, level=0.0):
    """
    Marching cubes -> a CLEAN, watertight-as-possible mesh that opens in MeshLab.

    The key fix vs the old exporter: build with process=True and explicitly
    clean topology. The old code used process=False (needed only for the
    distance-query bug earlier in the project) which left duplicate/degenerate
    faces and inconsistent winding that MeshLab rejects.
    """
    try:
        verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=level)
    except (ValueError, RuntimeError):
        return None
    res = sdf_grid.shape[0]
    verts = verts / (res - 1) * 2.0 - 1.0     # grid index -> [-1, 1]

    # Build WITH processing so trimesh merges duplicate vertices etc.
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Explicit clean-up for a well-formed export
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    # Keep only the largest connected component (drop stray floating bits)
    comps = mesh.split(only_watertight=False)
    if len(comps) > 1:
        mesh = max(comps, key=lambda m: len(m.faces))

    return mesh


def export_mesh(mesh, path):
    """Export as .obj AND .stl. STL is the most robust for MeshLab/Blender."""
    path = Path(path)
    mesh.export(path.with_suffix(".obj"))
    mesh.export(path.with_suffix(".stl"))


# ─────────────────────────────────────────────────────────────────────────────
# Surface render (proper shaded mesh, not a point cloud)
# ─────────────────────────────────────────────────────────────────────────────

def render_mesh(ax, mesh, color="#4C78A8", face_values=None, cmap="viridis",
                elev=18, azim=45):
    """Render a trimesh as a shaded surface on a 3D axis."""
    tris = mesh.vertices[mesh.faces]        # (F, 3, 3)
    coll = Poly3DCollection(tris, alpha=1.0, linewidths=0.0)

    if face_values is not None:
        norm = (face_values - face_values.min()) / (np.ptp(face_values) + 1e-9)
        coll.set_facecolor(plt.get_cmap(cmap)(norm))
    else:
        # Simple lambert-ish shading from face normals
        n = mesh.face_normals
        light = np.array([0.5, 0.4, 0.75]); light = light / np.linalg.norm(light)
        shade = np.clip(n @ light, 0.15, 1.0)
        base = np.array(_hex2rgb(color))
        coll.set_facecolor((base[None, :] * shade[:, None]).clip(0, 1))

    ax.add_collection3d(coll)
    v = mesh.vertices
    for setlim, lo, hi in [(ax.set_xlim, v[:,0].min(), v[:,0].max()),
                           (ax.set_ylim, v[:,1].min(), v[:,1].max()),
                           (ax.set_zlim, v[:,2].min(), v[:,2].max())]:
        setlim(lo, hi)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor((1, 1, 1, 0))
        pane.line.set_color((1, 1, 1, 0))


def _hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


# ─────────────────────────────────────────────────────────────────────────────
# Difference metric: per-vertex distance from generated -> nearest real surface
# ─────────────────────────────────────────────────────────────────────────────

def surface_difference(gen_mesh, real_surface_pts):
    """
    For each face of the generated mesh, distance from its centroid to the
    nearest point on a real surface. Returns per-face distances for colouring
    and summary stats (mean, max, Chamfer-like).
    """
    tree = cKDTree(real_surface_pts)
    centroids = gen_mesh.vertices[gen_mesh.faces].mean(axis=1)
    d, _ = tree.query(centroids)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "meshes_real").mkdir(exist_ok=True)
    (out / "meshes_generated").mkdir(exist_ok=True)
    res = args.resolution

    ck = torch.load(args.vae_ckpt, map_location=device)
    model = AneurysmAE(latent_dim=ck["latent_dim"], hidden_dim=ck["hidden_dim"],
                       n_freqs=ck["n_freqs"], variational=ck["variational"],
                       beta=ck.get("beta", 1.0), mode_name=ck["mode"]).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    print(f"Main model: VAE (val recon {ck['val_recon']:.4f})")

    # ── Real reconstructions (clean meshes + keep surface pts for diff) ────
    print("\n[1/4] Reconstructing & exporting REAL shapes (clean meshes)...")
    surf_ds = SurfaceDataset(args.processed_dir, split="test", n_points=2048)
    real_meshes, real_surfs, real_ids = [], [], []
    for sid in surf_ds.ids[:args.n_show]:
        pts = np.load(Path(args.processed_dir)/"surface"/f"{sid}.npz")["points"]
        real_surfs.append(pts.copy())
        p = pts[np.random.choice(len(pts), 2048, replace=False)] if len(pts) > 2048 else pts
        mu, _ = model.encode(torch.from_numpy(p).float().unsqueeze(0).to(device))
        m = grid_to_clean_mesh(eval_sdf_grid(model, mu, res, device), level=args.offset)
        if m is not None:
            export_mesh(m, out / "meshes_real" / sid)
            real_meshes.append(m); real_ids.append(sid)
    print(f"  Exported {len(real_meshes)} clean real meshes (.obj + .stl)")

    # ── Generated (clean meshes) ──────────────────────────────────────────
    print("[2/4] Generating & exporting SYNTHETIC shapes (clean meshes)...")
    gen_meshes = []
    tries = 0
    while len(gen_meshes) < args.n_show and tries < args.n_show * 3:
        tries += 1
        z = torch.randn(1, model.latent_dim, device=device)
        m = grid_to_clean_mesh(eval_sdf_grid(model, z, res, device), level=args.offset)
        if m is not None and len(m.faces) > 50:
            export_mesh(m, out / "meshes_generated" / f"gen_{len(gen_meshes):03d}")
            gen_meshes.append(m)
    print(f"  Exported {len(gen_meshes)} clean generated meshes (.obj + .stl)")

    # ── Figure 1: real vs generated SURFACE renders ───────────────────────
    print("[3/4] Rendering real vs generated surfaces...")
    ncol = min(args.n_show, len(real_meshes), len(gen_meshes))
    fig = plt.figure(figsize=(ncol * 2.4, 5))
    for i in range(ncol):
        ax = fig.add_subplot(2, ncol, i + 1, projection="3d")
        render_mesh(ax, real_meshes[i], color="#4C78A8")
        if i == 0: ax.set_title("Real\n", fontsize=11, loc="left")
        ax2 = fig.add_subplot(2, ncol, ncol + i + 1, projection="3d")
        render_mesh(ax2, gen_meshes[i], color="#E45756")
        if i == 0: ax2.set_title("Generated\n", fontsize=11, loc="left")
    fig.suptitle("Real vs Generated Aneurysm Surfaces (VAE)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "real_vs_generated_surfaces.png", dpi=170, bbox_inches="tight")
    plt.close()

    # ── Figure 2: difference colouring ────────────────────────────────────
    print("[4/4] Computing real-vs-generated surface difference...")
    all_real_pts = np.concatenate(real_surfs, axis=0)
    fig = plt.figure(figsize=(ncol * 2.4, 3.2))
    diff_stats = []
    for i in range(ncol):
        d = surface_difference(gen_meshes[i], all_real_pts)
        diff_stats.append({"mean": float(d.mean()), "max": float(d.max())})
        ax = fig.add_subplot(1, ncol, i + 1, projection="3d")
        render_mesh(ax, gen_meshes[i], face_values=d, cmap="magma")
        ax.set_title(f"mean Δ {d.mean():.3f}", fontsize=9)
    fig.suptitle("Generated Shapes Coloured by Distance to Nearest Real Surface\n"
                 "(darker = closer to the real data manifold)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out / "generated_difference_map.png", dpi=170, bbox_inches="tight")
    plt.close()

    mean_diff = float(np.mean([s["mean"] for s in diff_stats]))
    print(f"\n  Mean generated→real surface distance: {mean_diff:.4f}")
    print(f"\nSaved to {out}:")
    print("  real_vs_generated_surfaces.png   (shaded surface renders)")
    print("  generated_difference_map.png     (per-face error colouring)")
    print("  meshes_real/*.obj + *.stl        (open the .stl in MeshLab)")
    print("  meshes_generated/*.obj + *.stl")
    print("\nTip: open the .STL files in MeshLab — they are the most robust format.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--vae_ckpt",      type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--resolution",    type=int, default=96)
    p.add_argument("--n_show",        type=int, default=6)
    p.add_argument("--offset",        type=float, default=0.01,
                   help="level-set offset (use the calibrated +0.01)")
    args = p.parse_args()
    run(args)