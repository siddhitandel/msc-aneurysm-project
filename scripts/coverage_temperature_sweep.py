"""
coverage_temperature_sweep.py
=============================
Tests whether sampling the latent space at a higher "temperature"
(z ~ N(0, sigma^2 I) with sigma > 1) improves COVERAGE, following the
finding that COV was only ~39% at sigma=1 (shapes cluster near the mean).

For each temperature, generates a batch and recomputes the
fidelity / coverage / generalization triad, so you can see the
fidelity-vs-coverage tradeoff and pick the best operating point.

Higher sigma -> reach further into the latent tails -> more diverse shapes
(better coverage) but potentially less realistic (worse fidelity / more
non-samplable draws). The sweep quantifies that tradeoff.

Usage:
  python scripts/coverage_temperature_sweep.py \
      --processed_dir aneurysm_project/data/processed_sdf \
      --vae_ckpt      aneurysm_project/models/vae_compare/best.pt \
      --out_dir       aneurysm_project/results_coverage \
      --temps 1.0 1.25 1.5 2.0 --n_gen 80 --offset 0.01
"""

import argparse, json
from pathlib import Path

import numpy as np
import torch
import trimesh
from skimage import measure
from scipy.spatial import cKDTree
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


def grid_to_points(sdf_grid, level, n_pts=2048):
    try:
        verts, faces, _, _ = measure.marching_cubes(sdf_grid, level=level)
    except (ValueError, RuntimeError):
        return None
    res = sdf_grid.shape[0]
    verts = verts / (res - 1) * 2.0 - 1.0
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if len(m.faces) == 0:
        return None
    pts, _ = trimesh.sample.sample_surface(m, n_pts)
    return pts.astype(np.float32)


def chamfer(a, b):
    ta, tb = cKDTree(a), cKDTree(b)
    return float(tb.query(a)[0].mean() + ta.query(b)[0].mean())


def pairwise(setA, setB):
    M = np.zeros((len(setA), len(setB)))
    for i, a in enumerate(setA):
        for j, b in enumerate(setB):
            M[i, j] = chamfer(a, b)
    return M


def metrics(gen, real):
    D = pairwise(gen, real)
    mmd = float(D.min(axis=0).mean())
    cov = len(set(D.argmin(axis=1).tolist())) / len(real)
    # 1-NNA
    alls = list(gen) + list(real)
    lab = np.array([0]*len(gen) + [1]*len(real))
    n = len(alls); Dn = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i+1, n):
            d = chamfer(alls[i], alls[j]); Dn[i, j] = d; Dn[j, i] = d
    preds = np.array([lab[np.argmin(Dn[i])] for i in range(n)])
    nna = float((preds == lab).mean())
    return mmd, cov, nna


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res = args.resolution

    ck = torch.load(args.vae_ckpt, map_location=device)
    model = AneurysmAE(latent_dim=ck["latent_dim"], hidden_dim=ck["hidden_dim"],
                       n_freqs=ck["n_freqs"], variational=ck["variational"],
                       beta=ck.get("beta", 1.0), mode_name=ck["mode"]).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    ds = SurfaceDataset(args.processed_dir, split="test", n_points=2048)
    real = []
    for sid in ds.ids[:args.n_ref]:
        p = np.load(Path(args.processed_dir)/"surface"/f"{sid}.npz")["points"]
        if len(p) > 2048:
            p = p[np.random.choice(len(p), 2048, replace=False)]
        real.append(p.astype(np.float32))
    print(f"{len(real)} real reference shapes")

    results = {}
    for temp in args.temps:
        print(f"\n=== temperature sigma={temp} ===")
        gen = []; tries = 0
        while len(gen) < args.n_gen and tries < args.n_gen*4:
            tries += 1
            z = torch.randn(1, model.latent_dim, device=device) * temp
            pts = grid_to_points(eval_sdf_grid(model, z, res, device), args.offset)
            if pts is not None:
                gen.append(pts)
        samplable = len(gen) / tries
        mmd, cov, nna = metrics(gen, real)
        results[str(temp)] = {"MMD": mmd, "COV_pct": cov*100,
                              "1NNA_pct": nna*100, "samplable_rate": samplable,
                              "n_gen": len(gen)}
        print(f"  MMD {mmd:.4f} | COV {cov*100:.1f}% | 1-NNA {nna*100:.1f}% | "
              f"samplable {samplable*100:.0f}%")

    json.dump(results, open(out/"coverage_sweep.json","w"), indent=2)

    # Plot the tradeoff
    temps = [float(t) for t in results]
    covs = [results[str(t)]["COV_pct"] for t in temps]
    mmds = [results[str(t)]["MMD"] for t in temps]
    nnas = [results[str(t)]["1NNA_pct"] for t in temps]

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(temps, covs, "o-", color="#1A8754", label="Coverage (COV) %")
    ax1.plot(temps, nnas, "s-", color="#065A82", label="1-NNA % (ideal 50)")
    ax1.axhline(50, color="#888", linestyle=":", linewidth=1)
    ax1.set_xlabel("Latent sampling temperature (sigma)")
    ax1.set_ylabel("COV / 1-NNA (%)")
    ax2 = ax1.twinx()
    ax2.plot(temps, mmds, "^--", color="#C0392B", label="MMD (fidelity)")
    ax2.set_ylabel("MMD (lower=better)", color="#C0392B")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="center right", fontsize=9)
    ax1.set_title("Coverage vs Fidelity Tradeoff by Sampling Temperature")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out/"coverage_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Recommend best temp: highest COV with 1-NNA still reasonable (<75%)
    best = max(results, key=lambda t: results[t]["COV_pct"]
               if results[t]["1NNA_pct"] < 75 else -1)
    print(f"\nBest temperature (max coverage, 1-NNA<75%): sigma={best}")
    print(f"  COV {results[best]['COV_pct']:.1f}% | 1-NNA {results[best]['1NNA_pct']:.1f}%")
    print(f"\nSaved: coverage_sweep.json, coverage_sweep.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--vae_ckpt",      type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--temps",         type=float, nargs="+", default=[1.0,1.25,1.5,2.0])
    p.add_argument("--n_gen",         type=int, default=80)
    p.add_argument("--n_ref",         type=int, default=100)
    p.add_argument("--resolution",    type=int, default=96)
    p.add_argument("--offset",        type=float, default=0.01)
    args = p.parse_args()
    run(args)