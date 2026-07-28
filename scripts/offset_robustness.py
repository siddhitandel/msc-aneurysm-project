"""
offset_robustness.py
====================
Multi-seed robustness analysis for the level-set offset correction.

The single-run size analysis gives a volume ratio that depends on which
random shapes were sampled — generated volume is broadly distributed, so
one sample can say x0.73 and another x0.92 at the same offset. This script
runs the generation + measurement across several random seeds and reports
each size metric's ratio as MEAN +/- STD, plus how often it is statistically
indistinguishable from real (KS p > 0.05).

This turns a fragile single number into a defensible claim:
  "volume ratio 0.8 +/- 0.1 across N seeds; indistinguishable from real in
   K of N seeds."

Usage:
  python scripts/offset_robustness.py \
      --processed_dir aneurysm_project/data/processed_sdf \
      --vae_ckpt      aneurysm_project/models/vae_compare/best.pt \
      --out_dir       aneurysm_project/results_robustness \
      --offset 0.01 --n_seeds 5 --n_shapes 30
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from skimage import measure
from scipy.spatial.distance import pdist
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_models import AneurysmAE
from dataset import SurfaceDataset


METRICS = ["max_diameter", "equiv_diameter", "volume", "surface_area", "bbox_diagonal"]


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


def grid_to_mesh(sdf_grid, level=0.0):
    try:
        verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=level)
        res = sdf_grid.shape[0]
        verts = verts / (res - 1) * 2.0 - 1.0
        return trimesh.Trimesh(vertices=verts, faces=faces,
                               vertex_normals=normals, process=False)
    except (ValueError, RuntimeError):
        return None


def size_metrics(mesh):
    if mesh is None or len(mesh.faces) == 0:
        return None
    try:
        v = np.asarray(mesh.vertices)
        v_s = v[np.random.choice(len(v), 2000, replace=False)] if len(v) > 2000 else v
        max_diam = float(pdist(v_s).max())
        bbox_diag = float(np.linalg.norm(mesh.bounding_box.extents))
        vol = abs(float(mesh.volume)); area = float(mesh.area)
        equiv = float((6 * vol / np.pi) ** (1/3)) if vol > 1e-9 else 0.0
        return {"max_diameter": max_diam, "bbox_diagonal": bbox_diag,
                "equiv_diameter": equiv, "volume": vol, "surface_area": area}
    except Exception:
        return None


def real_metrics(model, processed_dir, device, res, n_shapes):
    surf_ds = SurfaceDataset(processed_dir, split="test", n_points=2048)
    out = []
    for sid in surf_ds.ids[:n_shapes]:
        pts = np.load(Path(processed_dir)/"surface"/f"{sid}.npz")["points"]
        if len(pts) > 2048:
            pts = pts[np.random.choice(len(pts), 2048, replace=False)]
        pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
        mu, _ = model.encode(pts_t)
        m = size_metrics(grid_to_mesh(eval_sdf_grid(model, mu, res, device), 0.0))
        if m: out.append(m)
    return out


def generate_metrics(model, device, res, n_shapes, offset):
    """Generate n_shapes and measure them at the given level offset."""
    out = []
    for _ in range(n_shapes):
        z = torch.randn(1, model.latent_dim, device=device)
        grid = eval_sdf_grid(model, z, res, device)
        m = size_metrics(grid_to_mesh(grid, level=offset))
        if m: out.append(m)
    return out


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
    print(f"Offset: {args.offset:+.3f} | Seeds: {args.n_seeds} | Shapes/seed: {args.n_shapes}\n")

    # Real reference (fixed — computed once)
    real = real_metrics(model, args.processed_dir, device, res, args.n_shapes)
    real_vals = {m: np.array([r[m] for r in real]) for m in METRICS}
    real_median_vol = float(np.median(real_vals["volume"]))
    print(f"Real median volume: {real_median_vol:.4f}\n")

    # Per-seed generation
    per_seed = {m: {"ratio": [], "ks_p": []} for m in METRICS}
    print("Running seeds...")
    for seed in range(args.n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        gen = generate_metrics(model, device, res, args.n_shapes, args.offset)
        if not gen:
            print(f"  seed {seed}: no valid shapes, skipping")
            continue
        for m in METRICS:
            gv = np.array([g[m] for g in gen])
            ratio = np.median(gv) / (np.median(real_vals[m]) + 1e-9)
            ks_p = float(ks_2samp(gv, real_vals[m]).pvalue)
            per_seed[m]["ratio"].append(ratio)
            per_seed[m]["ks_p"].append(ks_p)
        print(f"  seed {seed}: volume x{np.median([g['volume'] for g in gen])/real_median_vol:.2f} "
              f"(n={len(gen)})")

    # Aggregate
    print(f"\n{'='*72}")
    print(f"OFFSET ROBUSTNESS  (offset {args.offset:+.3f}, {args.n_seeds} seeds)")
    print(f"{'='*72}")
    print(f"  {'Metric':16s} {'ratio mean±std':>18} {'KS p mean':>12} {'indist. seeds':>14}")
    print(f"  {'-'*62}")
    summary = {"offset": args.offset, "n_seeds": args.n_seeds,
               "n_shapes": args.n_shapes, "real_median_volume": real_median_vol,
               "metrics": {}}
    for m in METRICS:
        ratios = np.array(per_seed[m]["ratio"])
        ksps   = np.array(per_seed[m]["ks_p"])
        n_indist = int((ksps > 0.05).sum())
        summary["metrics"][m] = {
            "ratio_mean": float(ratios.mean()), "ratio_std": float(ratios.std()),
            "ks_p_mean": float(ksps.mean()),
            "indistinguishable_seeds": n_indist, "total_seeds": len(ksps),
        }
        print(f"  {m:16s} {ratios.mean():>10.2f} ± {ratios.std():<5.2f} "
              f"{ksps.mean():>12.3f} {n_indist:>10}/{len(ksps)}")

    json.dump(summary, open(out / "offset_robustness.json", "w"), indent=2)

    # Plot: per-metric ratio with error bars
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(METRICS))
    means = [summary["metrics"][m]["ratio_mean"] for m in METRICS]
    stds  = [summary["metrics"][m]["ratio_std"]  for m in METRICS]
    ax.bar(xs, means, yerr=stds, capsize=6, color="#378ADD", alpha=0.85)
    ax.axhline(1.0, color="#5F5E5A", linestyle="--", linewidth=1, label="real")
    ax.set_xticks(xs)
    ax.set_xticklabels([m.replace("_", "\n") for m in METRICS], fontsize=9)
    ax.set_ylabel("Generated ÷ real (median ratio)")
    ax.set_title(f"Size-metric robustness across {args.n_seeds} seeds "
                 f"(offset {args.offset:+.3f})")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.2f}±{s:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "offset_robustness.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: offset_robustness.json, offset_robustness.png")
    vr = summary["metrics"]["volume"]
    print(f"\n>>> Volume ratio: {vr['ratio_mean']:.2f} ± {vr['ratio_std']:.2f} "
          f"across {args.n_seeds} seeds; indistinguishable from real in "
          f"{vr['indistinguishable_seeds']}/{vr['total_seeds']} seeds <<<")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--vae_ckpt",      type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--offset",        type=float, default=0.01)
    p.add_argument("--n_seeds",       type=int, default=5)
    p.add_argument("--n_shapes",      type=int, default=30)
    p.add_argument("--resolution",    type=int, default=96)
    args = p.parse_args()
    run(args)